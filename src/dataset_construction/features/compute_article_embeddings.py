import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Set

import networkx as nx
import numpy as np
import pandas as pd

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def load_or_compute_article_embeddings(
    df: pd.DataFrame,
    graph: nx.Graph,
    embeddings_path: str,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    title_col: str = "title",
    abstract_col: str = "abstract",
    id_col: Optional[str] = None,
) -> Dict[Any, np.ndarray]:
    """
    Return SentenceTransformer embeddings for every article referenced in graph edges.

    Only articles that appear in at least one edge's `articles` list are processed.
    Previously computed embeddings are loaded and reused; only missing ones are computed.
    The cache file is updated after each run.

    Args:
        df: DataFrame with article text (title, abstract) and row data
        graph: NetworkX graph; edges must have attr `articles` (list of article IDs)
        embeddings_path: .pkl cache file path for {article_id: embedding}
        id_col: column to use as article ID; uses df row index if None

    Returns:
        dict {article_id: np.ndarray of shape (embedding_dim,)}
    """
    path = Path(embeddings_path)

    existing: Dict[Any, np.ndarray] = {}
    if path.exists():
        with open(path, "rb") as f:
            existing = pickle.load(f)
        logger.info(f"Loaded {len(existing):,} cached article embeddings from {path}")

    needed_ids: Set = set()
    for _, _, data in graph.edges(data=True):
        needed_ids.update(data.get("articles", []))

    missing_ids = needed_ids - set(existing.keys())
    logger.info(
        f"Articles in graph edges: {len(needed_ids):,}, "
        f"already cached: {len(needed_ids) - len(missing_ids):,}, "
        f"to compute: {len(missing_ids):,}"
    )

    if not missing_ids:
        return {k: existing[k] for k in needed_ids if k in existing}

    # Build id -> row mapping
    if id_col and id_col in df.columns:
        id_to_row: Dict[Any, Any] = {row[id_col]: row for _, row in df.iterrows()}
    else:
        id_to_row = dict(df.iterrows())

    texts = []
    valid_ids = []
    for aid in sorted(missing_ids):
        row = id_to_row.get(aid)
        if row is None:
            continue
        title = str(row[title_col]) if title_col in row.index else ""
        abstract = str(row[abstract_col]) if abstract_col in row.index else ""
        if abstract and abstract.lower() != "nan":
            text = f"{title}. {abstract}"
        else:
            text = title
        texts.append(text)
        valid_ids.append(aid)

    if not texts:
        logger.warning("No valid texts found for missing articles")
        return existing

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        f"Computing embeddings for {len(texts):,} articles "
        f"with {model_name} on {device}"
    )
    model = SentenceTransformer(model_name, device=device)
    new_embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    for aid, emb in zip(valid_ids, new_embeddings):
        existing[aid] = emb

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(existing, f)
    logger.info(f"Saved {len(existing):,} article embeddings to {path}")

    return {k: existing[k] for k in needed_ids if k in existing}


def compute_concept_embeddings_from_articles(
    graph: nx.Graph,
    article_embeddings: Dict[Any, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Compute a mean article embedding per concept node.

    For each concept, collects the embeddings of all articles on its edges,
    then takes the L2-normalized mean.

    Returns:
        dict {concept_name: np.ndarray of shape (embedding_dim,)}
    """
    concept_article_ids: Dict[str, Set] = {}
    for u, v, data in graph.edges(data=True):
        for c in (u, v):
            if c not in concept_article_ids:
                concept_article_ids[c] = set()
            concept_article_ids[c].update(data.get("articles", []))

    concept_embeddings: Dict[str, np.ndarray] = {}
    for concept, article_ids in concept_article_ids.items():
        embs = [
            article_embeddings[aid] for aid in article_ids if aid in article_embeddings
        ]
        if not embs:
            continue
        mean_emb = np.mean(embs, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        concept_embeddings[concept] = mean_emb

    logger.info(f"Built concept embeddings for {len(concept_embeddings):,} concepts")
    return concept_embeddings
