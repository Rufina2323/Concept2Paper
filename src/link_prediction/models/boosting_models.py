import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from src.link_prediction.calculate_metrics import compute_ranking_metrics

logger = logging.getLogger(__name__)


def _clean64(X: np.ndarray) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float64)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=1e10, neginf=-1e10)
    return X


def _clean32(X: np.ndarray) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float32)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=1e10, neginf=-1e10)
    return X


def groups_from_pair_ids(pair_ids: np.ndarray) -> np.ndarray:
    """
    Convert a pair_ids array ([0,0,1,1,2,2,...]) to LightGBM/XGBoost group sizes ([2,2,2,...]).
    Assumes pair_ids are sorted (consecutive groups), as produced by sample_contrastive_negatives.
    """
    _, counts = np.unique(pair_ids, return_counts=True)
    return counts.astype(np.int32)


class LightGBMModel:
    """
    LightGBM LambdaRank wrapper.

    Requires group_train (and optionally group_val) derived from pair_ids.
    Raises ValueError if group_train is None.
    """

    def __init__(self, params: Dict[str, Any]):
        params = {
            k: v for k, v in params.items() if k not in ("enabled", "feature_groups")
        }
        self.early_stopping_rounds = params.pop("early_stopping_rounds", 50)
        params.setdefault("n_jobs", 1)
        self.params = params
        self.model = None
        self.best_iteration: Optional[int] = None
        self._feature_names: Optional[List[str]] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        group_train: Optional[np.ndarray] = None,
        group_val: Optional[np.ndarray] = None,
    ) -> None:

        if group_train is None:
            raise ValueError(
                "LightGBMModel requires group_train (pair_ids-derived group sizes). "
                "Ensure train.csv has a pair_id column produced by sample_contrastive_negatives."
            )

        X_train = _clean64(X_train)
        y_train = np.asarray(y_train, dtype=np.int32).ravel()
        self._feature_names = [f"f{i}" for i in range(X_train.shape[1])]
        df_train = pd.DataFrame(X_train, columns=self._feature_names)

        ranker_params = dict(self.params)
        ranker_params.pop("objective", None)
        self.model = lgb.LGBMRanker(
            **ranker_params,
            random_state=42,
            verbose=-1,
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[1, 5, 10],
            label_gain=[0, 1],
        )
        fit_kwargs: Dict[str, Any] = {"group": group_train}
        if X_val is not None and group_val is not None:
            df_val = pd.DataFrame(_clean64(X_val), columns=self._feature_names)
            y_val_arr = np.asarray(y_val, dtype=np.int32).ravel()
            fit_kwargs["eval_set"] = [(df_val, y_val_arr)]
            fit_kwargs["eval_group"] = [group_val]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(self.early_stopping_rounds),
                lgb.log_evaluation(100),
            ]
        self.model.fit(df_train, y_train, **fit_kwargs)
        self.best_iteration = getattr(self.model, "best_iteration_", None)
        logger.info("LightGBM best iter: %s", self.best_iteration)

    def predict(self, X: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(_clean64(X), columns=self._feature_names)
        return self.model.predict(df)

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        return self.model.feature_importances_ if self.model else None

    def evaluate_ranking(self, X, y, k_values=None):
        return compute_ranking_metrics(y, self.predict(X), k_values)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved LightGBM -> %s", path)

    @classmethod
    def load(cls, path: str) -> "LightGBMModel":
        with open(path, "rb") as f:
            return pickle.load(f)


class XGBoostModel:
    """
    XGBoost rank:ndcg wrapper using the low-level xgb.train() + DMatrix API.

    Requires group_train (and optionally group_val) derived from pair_ids.
    Raises ValueError if group_train is None.
    """

    def __init__(self, params: Dict[str, Any]):
        params = {
            k: v for k, v in params.items() if k not in ("enabled", "feature_groups")
        }

        use_gpu = torch.cuda.is_available()

        self.n_estimators = params.pop("n_estimators", 100)
        self.early_stopping = params.pop("early_stopping_rounds", 50)

        self._base_params = {
            "booster": "gbtree",
            "eta": params.pop("learning_rate", 0.05),
            "max_depth": params.pop("max_depth", 4),
            "subsample": params.pop("subsample", 0.8),
            "colsample_bytree": params.pop("colsample_bytree", 0.8),
            "colsample_bylevel": params.pop("colsample_bylevel", 0.9),
            "tree_method": "hist",
            "device": "cuda" if use_gpu else "cpu",
            "seed": 42,
            "nthread": 1,
            **params,
        }

        self.model = None
        self.best_iteration: Optional[int] = None
        self._feature_names: Optional[List[str]] = None
        self._n_features: Optional[int] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        group_train: Optional[np.ndarray] = None,
        group_val: Optional[np.ndarray] = None,
    ) -> None:

        if group_train is None:
            raise ValueError(
                "XGBoostModel requires group_train (pair_ids-derived group sizes). "
                "Ensure train.csv has a pair_id column produced by sample_contrastive_negatives."
            )

        X_train = _clean32(X_train)
        y_train = np.asarray(y_train, dtype=np.float32).ravel()
        self._n_features = X_train.shape[1]
        self._feature_names = [f"f{i}" for i in range(self._n_features)]

        xgb_params = {
            **self._base_params,
            "objective": "rank:ndcg",
            "eval_metric": "ndcg@10",
        }
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self._feature_names)
        dtrain.set_group(group_train)
        evals = [(dtrain, "train")]

        if X_val is not None and group_val is not None:
            X_val = _clean32(X_val)
            y_val = np.asarray(y_val, dtype=np.float32).ravel()
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self._feature_names)
            dval.set_group(group_val)
            evals.append((dval, "val"))
            self.model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=self.n_estimators,
                evals=evals,
                early_stopping_rounds=self.early_stopping,
                verbose_eval=100,
            )
            self.best_iteration = self.model.best_iteration
        else:
            self.model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=self.n_estimators,
                evals=evals,
                verbose_eval=100,
            )
        logger.info("XGBoost best iter: %s", self.best_iteration)

    def predict(self, X: np.ndarray) -> np.ndarray:
        dmat = xgb.DMatrix(_clean32(X), feature_names=self._feature_names)
        if self.best_iteration is not None:
            return self.model.predict(dmat, iteration_range=(0, self.best_iteration))
        return self.model.predict(dmat)

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        score_dict = self.model.get_score(importance_type="gain")
        imp = np.zeros(self._n_features, dtype=np.float64)
        for fname, score in score_dict.items():
            if fname in self._feature_names:
                imp[self._feature_names.index(fname)] = score
        return imp

    def evaluate_ranking(self, X, y, k_values=None):
        return compute_ranking_metrics(y, self.predict(X), k_values)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved XGBoost -> %s", path)

    @classmethod
    def load(cls, path: str) -> "XGBoostModel":
        with open(path, "rb") as f:
            return pickle.load(f)
