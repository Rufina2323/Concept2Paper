import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch

from src.link_prediction.calculate_metrics import (
    compute_ranking_metrics,
    format_ranking_metrics,
    get_top_k_pairs,
)
from src.link_prediction.load_data import (
    DataSplit,
    GroupedBatchSampler,
    load_datasets,
)
from src.link_prediction.models.boosting_models import (
    LightGBMModel,
    XGBoostModel,
    groups_from_pair_ids,
)
from src.link_prediction.models.mlp import LinkPredictionMLP, MLPTrainer

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    name: str
    train_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    rank_metrics: Dict[str, float] = field(default_factory=dict)
    top_pairs: List[Dict] = field(default_factory=list)
    feature_importances: Optional[np.ndarray] = None
    best_epoch: Optional[int] = None
    # predict_fn(df, X) -> scores
    # df: ranking DataFrame
    # X: scaled feature matrix [n, d]
    predict_fn: Optional[Callable] = None


def run_mlp(
    name: str,
    data_split: DataSplit,
    cfg: Dict[str, Any],
    rank_k: List[int],
    model_dir: Optional[Path] = None,
) -> ModelResult:
    logger.info("\n%s\n%s\n%s", "═" * 60, name, "═" * 60)
    result = ModelResult(name=name)

    dim = data_split.train_ds.X.shape[1]
    if dim == 0:
        logger.warning("No features for %s, skipping", name)
        return result

    hidden = cfg.get("hidden_dims", [128, 64, 32])
    dropout = cfg.get("dropout", 0.3)
    lr = cfg.get("learning_rate", 0.001)
    wd = cfg.get("weight_decay", 1e-4)
    bs = cfg.get("batch_size", 1024)
    epochs = cfg.get("epochs", 100)
    patience = cfg.get("patience", 10)

    seed = cfg.get("seed")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    loss = cfg.get("loss", "contrastive")
    model = LinkPredictionMLP(dim, hidden, dropout)
    trainer = MLPTrainer(model, lr=lr, weight_decay=wd, loss=loss)

    if data_split.train_ds.pair_ids is not None:
        train_loader = DataLoader(
            data_split.train_ds,
            batch_sampler=GroupedBatchSampler(
                data_split.train_ds.pair_ids.numpy(), bs, shuffle=True
            ),
            num_workers=0,
        )
    else:
        train_loader = DataLoader(
            data_split.train_ds, batch_size=bs, shuffle=True, num_workers=0
        )

    if data_split.val_ds.pair_ids is not None:
        val_loader = DataLoader(
            data_split.val_ds,
            batch_sampler=GroupedBatchSampler(
                data_split.val_ds.pair_ids.numpy(), bs, shuffle=False
            ),
            num_workers=0,
        )
    else:
        val_loader = DataLoader(
            data_split.val_ds, batch_size=bs, shuffle=False, num_workers=0
        )

    result.best_epoch = trainer.fit(train_loader, val_loader, epochs, patience)

    for split_name, ds in (
        ("train", data_split.train_ds),
        ("val", data_split.val_ds),
        ("test", data_split.test_ds),
    ):
        X = ds.X.numpy()
        y = ds.y.numpy().astype(int)
        m = compute_ranking_metrics(y, trainer.predict(X), rank_k)
        setattr(result, f"{split_name}_metrics", m)
        logger.info(format_ranking_metrics(m, f"{name} — {split_name}"))

    result.predict_fn = lambda _df, X, _t=trainer: _t.predict(X)

    df_test = data_split.df_test
    test_pairs = list(zip(df_test["concept_a"], df_test["concept_b"]))
    test_scores = trainer.predict(data_split.test_ds.X.numpy())
    test_labels = data_split.test_ds.y.numpy().astype(int)
    result.top_pairs = get_top_k_pairs(
        test_pairs, test_labels, test_scores, k=10
    ).to_dict("records")

    if model_dir is not None:
        save_path = (
            model_dir
            / f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pt"
        )
        trainer.save(str(save_path), dim, hidden, dropout)

    return result


def run_boosting(
    name: str,
    model_cls,
    data_split: DataSplit,
    cfg: Dict[str, Any],
    rank_k: List[int],
    model_dir: Optional[Path] = None,
) -> ModelResult:
    logger.info("\n%s\n%s\n%s", "═" * 60, name, "═" * 60)
    result = ModelResult(name=name)

    X_train = data_split.train_ds.X.numpy()
    y_train = data_split.train_ds.y.numpy().astype(int)
    X_val = data_split.val_ds.X.numpy()
    y_val = data_split.val_ds.y.numpy().astype(int)
    X_test = data_split.test_ds.X.numpy()
    y_test = data_split.test_ds.y.numpy().astype(int)

    group_train = (
        groups_from_pair_ids(data_split.train_ds.pair_ids.numpy())
        if data_split.train_ds.pair_ids is not None
        else None
    )
    group_val = (
        groups_from_pair_ids(data_split.val_ds.pair_ids.numpy())
        if data_split.val_ds.pair_ids is not None
        else None
    )

    model = model_cls(cfg)
    model.fit(
        X_train, y_train, X_val, y_val, group_train=group_train, group_val=group_val
    )
    result.feature_importances = model.feature_importances_

    for split_name, X, y in (
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ):
        m = compute_ranking_metrics(y, model.predict(X), rank_k)
        setattr(result, f"{split_name}_metrics", m)
        logger.info(format_ranking_metrics(m, f"{name} — {split_name}"))

    result.predict_fn = lambda _df, X, _m=model: _m.predict(X)

    df_test = data_split.df_test
    test_pairs = list(zip(df_test["concept_a"], df_test["concept_b"]))
    result.top_pairs = get_top_k_pairs(
        test_pairs, y_test, model.predict(X_test), k=10
    ).to_dict("records")

    if model_dir is not None:
        ext = ".pkl"
        stem = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model.save(str(model_dir / f"{stem}{ext}"))

    return result


def run_ranking(
    results: Dict[str, ModelResult],
    ranking_csv: str,
    feature_groups_map: Dict[str, Any],
    scaler_map: Dict[str, Any],
    k_values: List[int],
) -> None:
    """
    Evaluate all models on the ranking CSV.

    ranking_csv: path to ranking_test.csv
    feature_groups_map: {model_name: feature_groups} — features each model uses;
                        models with no entry are called with predict_fn(df, None)
                        (used by self-contained models like Ensemble).
    scaler_map: {model_name: scaler} — fitted scaler for each model
    """
    from src.link_prediction.load_data import classify_features, _select_cols

    df = pd.read_csv(ranking_csv)
    logger.info("Ranking CSV loaded: %d candidates", len(df))

    y_rank = df["label"].values.astype(np.int32)
    pairs = list(zip(df["concept_a"], df["concept_b"]))

    for name, result in results.items():
        if result.predict_fn is None:
            continue

        fg = feature_groups_map.get(name)
        scaler = scaler_map.get(name)

        if fg is None or scaler is None:
            # Self-contained predict_fn (e.g. Ensemble) — pass df with X=None
            try:
                scores = result.predict_fn(df, None)
            except Exception as exc:
                logger.warning(
                    "No feature config for %s and predict_fn failed (%s) — skipping",
                    name,
                    exc,
                )
                continue
        else:
            groups = classify_features(df.columns.tolist())
            cols = _select_cols(groups, fg)
            if not cols:
                logger.warning("No columns for %s in ranking CSV — skipping", name)
                continue
            X = np.nan_to_num(df[cols].values.astype(np.float32))
            X = scaler.transform(X).astype(np.float32)
            scores = result.predict_fn(df, X)

        metrics = compute_ranking_metrics(y_rank, scores, k_values)
        result.rank_metrics = metrics
        logger.info(format_ranking_metrics(metrics, f"Ranking — {name}"))

        top_df = get_top_k_pairs(pairs, y_rank, scores, k=10)
        result.top_pairs = top_df.to_dict("records")
        logger.info("\n  Top 10 — %s:\n%s", name, top_df.to_string(index=False))


def build_ranking_summary(
    results: Dict[str, ModelResult], k_values: List[int]
) -> pd.DataFrame:
    metric_cols = (
        ["roc_auc", "mrr"]
        + [f"p_at_{k}" for k in k_values]
        + [f"ndcg_at_{k}" for k in k_values]
        + ["n_candidates", "n_pos", "pos_rate"]
    )
    rows = []
    for name, res in results.items():
        if not res.rank_metrics:
            continue
        row = {"model": name}
        row.update({mc: res.rank_metrics.get(mc) for mc in metric_cols})
        rows.append(row)
    return pd.DataFrame(rows)


class TrainingPipeline:
    """
    Trains and evaluates all models defined in the config.

    Expected config structure (see configs/training.yaml).
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.output_dir = Path(cfg.get("output_dir"))
        self.model_dir = self.output_dir / "models"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        eval_cfg = cfg.get("evaluation", {})
        self.rank_k = eval_cfg.get("ranking", {}).get("k_values", [10, 50, 100])
        self.do_ranking = eval_cfg.get("ranking", {}).get("enabled", False)

        self.data_cfg = cfg.get("data", {})
        self.models_cfg = cfg.get("models", {})

    def _paths(self):
        d = Path(self.data_cfg["dir"])
        return (
            str(d / self.data_cfg.get("train_file", "train.csv")),
            str(d / self.data_cfg.get("val_file", "val.csv")),
            str(d / self.data_cfg.get("test_file", "ranking_test.csv")),
        )

    def _ranking_csv(self) -> Optional[str]:
        p = self.data_cfg.get("ranking_test_file")
        if p:
            return str(Path(self.data_cfg["dir"]) / p)
        return None

    def train(self) -> Dict[str, ModelResult]:
        train_path, val_path, test_path = self._paths()
        results: Dict[str, ModelResult] = {}

        # Lazy DataSplit cache keyed by fg_key (same format as _build_cache)
        _cache: Dict[str, DataSplit] = {}

        def get_split(fg):
            key = json.dumps(fg, sort_keys=True) if isinstance(fg, list) else fg
            if key not in _cache:
                _cache[key] = load_datasets(train_path, val_path, test_path, fg)
            return _cache[key]

        mcfg = self.models_cfg

        # MLP (structure)
        if mcfg.get("mlp_structure", {}).get("enabled", True):
            split = get_split("structure")
            results["MLP (structure)"] = run_mlp(
                "MLP (structure)",
                split,
                mcfg["mlp_structure"],
                self.rank_k,
                self.model_dir,
            )

        # MLP (emb)
        if mcfg.get("mlp_emb", {}).get("enabled", True):
            split = get_split("emb")
            results["MLP (emb)"] = run_mlp(
                "MLP (emb)", split, mcfg["mlp_emb"], self.rank_k, self.model_dir
            )

        # MLP (all)
        if mcfg.get("mlp_all", {}).get("enabled", True):
            split = get_split(["structure", "emb"])
            results["MLP (all)"] = run_mlp(
                "MLP (all)",
                split,
                mcfg["mlp_all"],
                self.rank_k,
                self.model_dir,
            )

        # LightGBM (all)
        if mcfg.get("lightgbm_all", {}).get("enabled", True):
            split = get_split("all")
            results["LightGBM (all)"] = run_boosting(
                "LightGBM (all)",
                LightGBMModel,
                split,
                mcfg["lightgbm_all"],
                self.rank_k,
                self.model_dir,
            )

        # XGBoost (all)
        if mcfg.get("xgboost_all", {}).get("enabled", True):
            split = get_split("all")
            results["XGBoost (all)"] = run_boosting(
                "XGBoost (all)",
                XGBoostModel,
                split,
                mcfg["xgboost_all"],
                self.rank_k,
                self.model_dir,
            )

        # Training summary (test split)
        metric_cols = ["roc_auc", "mrr"] + [f"ndcg_at_{k}" for k in self.rank_k]
        rows = []
        for n, res in results.items():
            row = {"model": n}
            row.update({mc: res.test_metrics.get(mc) for mc in metric_cols})
            rows.append(row)
        test_summary = pd.DataFrame(rows)
        test_summary.to_csv(self.output_dir / "summary_test.csv", index=False)

        top_pairs_rows = []
        for n, res in results.items():
            for rec in res.top_pairs:
                top_pairs_rows.append({"model": n, **rec})
        if top_pairs_rows:
            pd.DataFrame(top_pairs_rows).to_csv(
                self.output_dir / "top10_pairs_test.csv", index=False
            )

        print("\n" + "═" * 80)
        print("TEST RESULTS (ranking metrics)")
        print("═" * 80)
        print(
            test_summary.to_string(
                index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "—"
            )
        )
        print("═" * 80)

        # Persist cache so ranking can reuse scalers
        self._last_cache = _cache
        self._last_results = results
        return results

    # Maps model display name → (saved file stem, kind, feature_groups)
    _MODEL_REGISTRY = [
        ("MLP (structure)", "mlp_structure", "mlp", "structure"),
        ("MLP (emb)", "mlp_emb", "mlp", "emb"),
        ("MLP (all)", "mlp_all", "mlp", "all"),
        ("LightGBM (all)", "lightgbm_all", "lgbm", "all"),
        ("XGBoost (all)", "xgboost_all", "xgb", "all"),
    ]

    def _build_cache(self) -> Dict[str, DataSplit]:
        """Re-fit scalers for every unique feature group that has an enabled model."""
        train_path, val_path, test_path = self._paths()
        needed_groups: set = set()
        for _, stem, _, fg in self._MODEL_REGISTRY:
            if self.models_cfg.get(stem, {}).get("enabled", True):
                needed_groups.add(
                    json.dumps(fg, sort_keys=True) if isinstance(fg, list) else fg
                )

        cache: Dict[str, DataSplit] = {}
        for fg_key in needed_groups:
            fg = json.loads(fg_key) if fg_key.startswith("[") else fg_key
            logger.info("Loading data split for feature group: %s", fg)
            cache[fg_key] = load_datasets(train_path, val_path, test_path, fg)
        return cache

    def _load_saved_models(
        self, cache: Optional[Dict[str, DataSplit]] = None
    ) -> Dict[str, ModelResult]:
        """Load saved model checkpoints from model_dir and attach predict_fn."""
        results: Dict[str, ModelResult] = {}

        for name, stem, kind, _ in self._MODEL_REGISTRY:
            if not self.models_cfg.get(stem, {}).get("enabled", True):
                continue

            if kind == "mlp":
                path = self.model_dir / f"{stem}.pt"
                if not path.exists():
                    logger.warning("No saved model for %s at %s", name, path)
                    continue
                trainer = MLPTrainer.load(str(path))
                res = ModelResult(name=name)
                res.predict_fn = lambda _df, X, _t=trainer: _t.predict(X)
                results[name] = res
                logger.info("Loaded %s <- %s", name, path)

            elif kind == "lgbm":
                path = self.model_dir / f"{stem}.pkl"
                if not path.exists():
                    logger.warning("No saved model for %s at %s", name, path)
                    continue
                model = LightGBMModel.load(str(path))
                res = ModelResult(name=name)
                res.predict_fn = lambda _df, X, _m=model: _m.predict(X)
                results[name] = res
                logger.info("Loaded %s <- %s", name, path)

            elif kind == "xgb":
                path = self.model_dir / f"{stem}.pkl"
                if not path.exists():
                    logger.warning("No saved model for %s at %s", name, path)
                    continue
                model = XGBoostModel.load(str(path))
                res = ModelResult(name=name)
                res.predict_fn = lambda _df, X, _m=model: _m.predict(X)
                results[name] = res
                logger.info("Loaded %s <- %s", name, path)

        return results

    def rank(self, results: Optional[Dict[str, ModelResult]] = None) -> None:
        if not self.do_ranking:
            logger.info("Ranking evaluation disabled")
            return

        ranking_csv = self._ranking_csv()
        if ranking_csv is None or not Path(ranking_csv).exists():
            logger.warning("Ranking CSV not found (%s) — skipping ranking", ranking_csv)
            return

        # In rank-only mode (no prior train() call), reload data and load models from disk
        cache = getattr(self, "_last_cache", {})
        if not cache:
            logger.info("No in-memory cache — reloading data splits from CSVs")
            cache = self._build_cache()

        if results is None or not results:
            logger.info("No in-memory results — loading saved models from disk")
            results = _results = self._load_saved_models(cache)
        else:
            _results = results

        if not _results:
            logger.warning("No models available for ranking")
            return

        def _cache_key(fg):
            return json.dumps(fg, sort_keys=True) if isinstance(fg, list) else fg

        feature_groups_map: Dict[str, Any] = {
            name: fg for name, _, _, fg in self._MODEL_REGISTRY
        }

        scaler_map: Dict[str, Any] = {}
        for name, _, _, fg in self._MODEL_REGISTRY:
            key = _cache_key(fg)
            if key in cache:
                scaler_map[name] = cache[key].scaler

        run_ranking(_results, ranking_csv, feature_groups_map, scaler_map, self.rank_k)

        rank_summary = build_ranking_summary(_results, self.rank_k)
        rank_summary.to_csv(self.output_dir / "summary_ranking.csv", index=False)

        if not rank_summary.empty:
            print("\n" + "═" * 80)
            print("RANKING RESULTS")
            print("═" * 80)
            print(
                rank_summary.to_string(
                    index=False,
                    float_format=lambda x: (
                        f"{x:.4f}" if isinstance(x, float) and pd.notna(x) else str(x)
                    ),
                )
            )
            print("═" * 80)

        self._save_json(_results)

    def run(self) -> Dict[str, ModelResult]:
        """Train all models then run ranking evaluation."""
        results = self.train()
        self.rank(results)
        return results

    def _save_json(self, results: Dict[str, ModelResult]) -> None:
        out = {
            name: {
                "train": res.train_metrics,
                "val": res.val_metrics,
                "test": res.test_metrics,
                "rank": res.rank_metrics,
                "top_pairs": res.top_pairs,
                "best_epoch": res.best_epoch,
            }
            for name, res in results.items()
        }
        path = self.output_dir / "results.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        logger.info("Results saved -> %s", path)
