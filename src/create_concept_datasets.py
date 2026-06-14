import argparse
import logging
from pathlib import Path
from typing import Any, Dict

from src.utils import load_config, parse_concept_list

import pandas as pd

from src.dataset_construction.create_graph import (
    build_graph,
    get_test_edges,
    save_graph,
    load_graph,
    save_test_edges,
    load_test_edges,
)

from src.dataset_construction.features.compute_article_embeddings import (
    load_or_compute_article_embeddings,
)
from src.dataset_construction.sample_edges import (
    sample_positive_edges,
    sample_contrastive_negatives,
    generate_ranking_candidates,
    compute_and_store_split_features,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_concept_datasets(
    cfg: Dict[str, Any],
    df: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Execute the full dataset creation pipeline.

    Args:
        cfg: config dict (from dataset_creation.yaml)
        df: DataFrame with `_concepts` column (list of strings, pre-parsed)
        output_dir: directory for all outputs (graphs, embeddings, CSVs)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Year config
    years_cfg = cfg["years"]
    min_year = years_cfg["min_year"]  # 2017
    val_year = years_cfg["val_year"]  # 2024
    test_year = years_cfg["test_year"]  # 2025

    # Each graph ends one year before its target
    train_end = val_year - 2  # 2022  →  predict 2023
    val_end = val_year - 1  # 2023  →  predict 2024
    test_end = val_year  # 2024  →  predict 2025

    train_target = train_end + 1  # 2023
    val_target = val_end + 1  # 2024  == val_year
    test_target = test_year  # 2025

    cols = cfg.get("columns", {})
    concepts_col = "_concepts"
    year_col = cols.get("year", "year")
    title_col = cols.get("title", "title")
    abstract_col = cols.get("abstract", "abstract")

    feature_cfg = cfg.get("features", {})
    neg_cfg = cfg.get("negative_sampling", {})
    seed = neg_cfg.get("seed", 42)
    contrastive_cfg = neg_cfg.get("contrastive", {})
    contrastive_strategy = contrastive_cfg.get("strategy", "smart")
    n_neg_per_pos = contrastive_cfg.get("n_neg_per_pos", 1)
    augmentation_factor = contrastive_cfg.get("augmentation_factor", 1)
    contrastive_strategy_weights = contrastive_cfg.get("strategy_weights", None)
    test_cfg = neg_cfg.get("test", {})
    test_strategy_weights = test_cfg.get("strategy_weights", None)
    window_size = cfg.get("window_size", 5)

    embedding_combination = feature_cfg.get("embedding_combination", ["cosine", "l2"])

    max_ranking_candidates = cfg.get("max_ranking_candidates", 300_000)

    # Sparse years: last window_size years of each graph's range
    sparse_train = list(
        range(max(min_year, train_end - window_size + 1), train_end + 1)
    )
    sparse_val = list(range(max(min_year, val_end - window_size + 1), val_end + 1))
    sparse_test = list(range(max(min_year, test_end - window_size + 1), test_end + 1))

    logger.info(
        f"Split configuration:\n"
        f"  train : graph {min_year}-{train_end}  ->  predict {train_target}"
        f"  (sparse years: {sparse_train})\n"
        f"  val   : graph {min_year}-{val_end}    ->  predict {val_target}"
        f"  (sparse years: {sparse_val})\n"
        f"  test  : graph {min_year}-{test_end}   ->  predict {test_target}"
        f"  (sparse years: {sparse_test}) [ranking]"
    )

    # Build graphs
    logger.info("Step 1 | Building graphs")

    def _get_or_build(start, end):
        path = out / f"graph_{start}_{end}.pkl"
        if path.exists():
            return load_graph(str(path))
        g = build_graph(df, start, end, concepts_col, year_col)
        save_graph(g, str(path))
        return g

    graph_train = _get_or_build(min_year, train_end)
    graph_val = _get_or_build(min_year, val_end)
    graph_test = _get_or_build(min_year, test_end)

    # Next-year positive edges
    logger.info("Step 2 | Extracting next-year positive edges")

    def _get_or_extract(graph, target, label):
        path = out / f"test_edges_{label}_{target}.pkl"
        if path.exists():
            return load_test_edges(str(path))
        edges = get_test_edges(df, graph, target, concepts_col, year_col)
        save_test_edges(edges, str(path))
        return edges

    train_pos_edges = _get_or_extract(graph_train, train_target, "train")
    val_pos_edges = _get_or_extract(graph_val, val_target, "val")
    test_pos_edges = _get_or_extract(graph_test, test_target, "test")

    # Article embeddings
    logger.info("Step 3 | Article embeddings")

    article_embeddings = None
    if feature_cfg.get("use_embedding_features", True):
        # Use the largest graph so all articles in any split are covered
        article_embeddings = load_or_compute_article_embeddings(
            df,
            graph_test,
            embeddings_path=str(out / "article_embeddings.pkl"),
            model_name=feature_cfg.get("embedding_model", "all-MiniLM-L6-v2"),
            batch_size=feature_cfg.get("embedding_batch_size", 256),
            title_col=title_col,
            abstract_col=abstract_col,
        )

    # Sample edges + compute features
    logger.info("Step 4 & 5 | Sampling edges and computing features")

    common_feat_kwargs = dict(
        graph_feature_cfg=feature_cfg,
        embedding_combination=embedding_combination,
        article_embeddings=article_embeddings,
    )

    future_val_test = set(val_pos_edges) | set(test_pos_edges)
    future_test = set(test_pos_edges)

    # Train split — contrastive negatives
    # Exclude val and test positives so future links are never treated as negatives.
    train_csv = out / "train.csv"
    if not train_csv.exists():
        logger.info(
            f"Building train split  (graph {min_year}-{train_end}, positives from {train_target})"
        )
        train_pos = sample_positive_edges(graph_train, train_pos_edges)
        train_pairs, train_labels, train_pair_ids = sample_contrastive_negatives(
            train_pos,
            graph_train,
            n_neg_per_pos=n_neg_per_pos,
            strategy=contrastive_strategy,
            excluded=future_val_test,
            seed=seed,
            augmentation_factor=augmentation_factor,
            strategy_weights=contrastive_strategy_weights,
        )
        compute_and_store_split_features(
            train_pairs,
            train_labels,
            graph_train,
            output_path=str(train_csv),
            sparse_years=sparse_train,
            window_info={
                "window_start": min_year,
                "window_end": train_end,
                "target_year": train_target,
            },
            pair_ids=train_pair_ids,
            **common_feat_kwargs,
        )
    else:
        logger.info(f"Train CSV already exists, skipping: {train_csv}")

    # Val split — contrastive negatives (same strategy as train)
    # Exclude test positives from negatives.
    val_csv = out / "val.csv"
    if not val_csv.exists():
        logger.info(
            f"Building val split    (graph {min_year}-{val_end}, positives from {val_target})"
        )
        val_pos = sample_positive_edges(graph_val, val_pos_edges)
        val_pairs, val_labels, val_pair_ids = sample_contrastive_negatives(
            val_pos,
            graph_val,
            n_neg_per_pos=n_neg_per_pos,
            strategy=contrastive_strategy,
            excluded=future_test,
            seed=seed + 1,
            augmentation_factor=augmentation_factor,
            strategy_weights=contrastive_strategy_weights,
        )
        compute_and_store_split_features(
            val_pairs,
            val_labels,
            graph_val,
            output_path=str(val_csv),
            sparse_years=sparse_val,
            window_info={
                "window_start": min_year,
                "window_end": val_end,
                "target_year": val_target,
            },
            pair_ids=val_pair_ids,
            **common_feat_kwargs,
        )
    else:
        logger.info(f"Val CSV already exists, skipping: {val_csv}")

    # Test split — ranking candidates
    rank_test_csv = out / "ranking_test.csv"
    if not rank_test_csv.exists():
        logger.info(
            f"Building test split   (graph {min_year}-{test_end}, "
            f"ranking candidates, positives from {test_target})"
        )
        test_pairs, test_labels = generate_ranking_candidates(
            graph_test,
            test_pos_edges,
            max_candidates=max_ranking_candidates,
            seed=seed + 3,
            strategy_weights=test_strategy_weights,
        )
        compute_and_store_split_features(
            test_pairs,
            test_labels,
            graph_test,
            output_path=str(rank_test_csv),
            sparse_years=sparse_test,
            window_info={
                "window_start": min_year,
                "window_end": test_end,
                "target_year": test_target,
            },
            **common_feat_kwargs,
        )
    else:
        logger.info(f"Rank Test CSV already exists, skipping: {rank_test_csv}")

    logger.info("=" * 60)
    logger.info(f"Pipeline complete. Outputs: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build link prediction datasets")
    parser.add_argument("--config", default="configs/dataset_creation.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cols = cfg.get("columns", {})

    input_file = cfg["input_file"]
    logger.info(f"Loading {input_file}")
    df = pd.read_csv(input_file)
    df["_concepts"] = df[cols.get("concepts", "normalized_concepts")].map(
        parse_concept_list
    )
    year_col = cols.get("year", "year")
    logger.info(f"Loaded {len(df):,} rows, years: {sorted(df[year_col].unique())}")

    create_concept_datasets(cfg, df, cfg["output_dir"])


if __name__ == "__main__":
    main()
