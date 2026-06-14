import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.dataset_construction.features.compute_graph_features import (
    compute_graph_features,
    compute_sparse_matrix_features,
    compute_embedding_pair_features,
)
from src.dataset_construction.features.compute_article_embeddings import (
    compute_concept_embeddings_from_articles,
)

logger = logging.getLogger(__name__)


def make_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


class SmartCandidateSampler:
    """
    Biased non-edge sampler for a NetworkX graph.
    Builds internal neighbor/degree/strength structures once, then supports sample() calls for multiple negative sets.
    """

    _MAX_POOL = 2_000_000

    def __init__(
        self,
        graph: nx.Graph,
        nodes: Optional[List[str]] = None,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.rng = rng if rng is not None else np.random.RandomState(42)
        self.nodes: List[str] = nodes if nodes is not None else list(graph.nodes())
        self._node_set: Set[str] = set(self.nodes)

        self.adj: Set[Tuple[str, str]] = set()
        neighbors: Dict[str, Set[str]] = {n: set() for n in self.nodes}
        strength: Dict[str, float] = {n: 0.0 for n in self.nodes}

        for u, v, data in graph.edges(data=True):
            if u not in self._node_set or v not in self._node_set:
                continue
            self.adj.add(make_pair(u, v))
            neighbors[u].add(v)
            neighbors[v].add(u)
            w = float(data.get("weight", 1))
            strength[u] += w
            strength[v] += w

        self.neighbors = neighbors
        self.degree: Dict[str, int] = {n: len(nb) for n, nb in neighbors.items()}

        total = sum(strength.values()) or 1.0
        self.node_probs = np.array(
            [strength[n] / total for n in self.nodes], dtype=np.float64
        )
        if self.node_probs.sum() == 0:
            self.node_probs = np.ones(len(self.nodes)) / len(self.nodes)

    # strategies fo sampling

    def _2hop(self, n_samples: int) -> List[Tuple[str, str]]:
        degrees = np.array([self.degree[n] for n in self.nodes], dtype=float)
        probs = degrees / (degrees.sum() or 1.0)
        sample_size = min(5000, len(self.nodes))
        idx = self.rng.choice(len(self.nodes), size=sample_size, replace=False, p=probs)
        sample_nodes = [self.nodes[i] for i in idx]

        seen: Set[Tuple[str, str]] = set()
        candidates: List[Tuple[str, str]] = []
        weights: List[int] = []

        for node in sample_nodes:
            for nb in self.neighbors[node]:
                for target in self.neighbors[nb]:
                    if target == node:
                        continue
                    pair = make_pair(node, target)
                    if pair in seen or pair in self.adj:
                        continue
                    seen.add(pair)
                    cn = len(self.neighbors[node] & self.neighbors[target])
                    candidates.append(pair)
                    weights.append(cn)
                    if len(candidates) >= self._MAX_POOL:
                        break
                if len(candidates) >= self._MAX_POOL:
                    break
            if len(candidates) >= self._MAX_POOL:
                break

        if not candidates:
            return []
        w = np.array(weights, dtype=float)
        w /= w.sum()
        n_samples = min(n_samples, len(candidates))
        chosen = self.rng.choice(len(candidates), size=n_samples, replace=False, p=w)
        return [candidates[i] for i in chosen]

    def _resource_alloc(self, n_samples: int) -> List[Tuple[str, str]]:
        degrees = np.array([self.degree[n] for n in self.nodes], dtype=float)
        probs = degrees / (degrees.sum() or 1.0)
        sample_size = min(3000, len(self.nodes))
        idx = self.rng.choice(len(self.nodes), size=sample_size, replace=False, p=probs)
        sample_nodes = [self.nodes[i] for i in idx]

        seen: Set[Tuple[str, str]] = set()
        scored: List[Tuple[Tuple[str, str], float]] = []

        for node in sample_nodes:
            for nb in self.neighbors[node]:
                for target in self.neighbors[nb]:
                    if target == node:
                        continue
                    pair = make_pair(node, target)
                    if pair in seen or pair in self.adj:
                        continue
                    seen.add(pair)
                    common = self.neighbors[node] & self.neighbors[target]
                    ra = sum(1.0 / self.degree[z] for z in common if self.degree[z] > 0)
                    if ra > 0:
                        scored.append((pair, ra))
                    if len(scored) >= self._MAX_POOL:
                        break
                if len(scored) >= self._MAX_POOL:
                    break
            if len(scored) >= self._MAX_POOL:
                break

        scored.sort(key=lambda x: x[1], reverse=True)
        return [pair for pair, _ in scored[:n_samples]]

    def _pref_attach(self, n_samples: int) -> List[Tuple[str, str]]:
        nodes_arr = np.array(self.nodes)
        candidates: Set[Tuple[str, str]] = set()
        max_attempts = n_samples * 50
        attempts = 0
        while len(candidates) < n_samples and attempts < max_attempts:
            batch = min(n_samples * 5, 100_000)
            idx = self.rng.choice(len(nodes_arr), size=(batch, 2), p=self.node_probs)
            for i, j in idx:
                if i == j:
                    continue
                pair = make_pair(nodes_arr[i], nodes_arr[j])
                if pair not in self.adj:
                    candidates.add(pair)
                    if len(candidates) >= n_samples:
                        break
            attempts += batch
        return list(candidates)

    def _random(self, n_samples: int) -> List[Tuple[str, str]]:
        nodes_arr = np.array(self.nodes)
        candidates: Set[Tuple[str, str]] = set()
        max_attempts = n_samples * 50
        attempts = 0
        while len(candidates) < n_samples and attempts < max_attempts:
            batch = min(n_samples * 3, 50_000)
            idx = self.rng.randint(0, len(nodes_arr), size=(batch, 2))
            for i, j in idx:
                if i == j:
                    continue
                pair = make_pair(nodes_arr[i], nodes_arr[j])
                if pair not in self.adj:
                    candidates.add(pair)
                    if len(candidates) >= n_samples:
                        break
            attempts += batch
        return list(candidates)

    def sample(
        self,
        n_total: int,
        strategy_weights: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, str]]:
        if strategy_weights is None:
            strategy_weights = {
                "2hop": 0.40,
                "resource_alloc": 0.30,
                "pref_attach": 0.15,
                "random": 0.15,
            }

        dispatch = {
            "2hop": self._2hop,
            "resource_alloc": self._resource_alloc,
            "pref_attach": self._pref_attach,
            "random": self._random,
        }

        seen: Dict[Tuple[str, str], str] = {}
        for strategy, fraction in strategy_weights.items():
            n = int(n_total * fraction)
            pairs = dispatch[strategy](n)
            for pair in pairs:
                if pair not in seen:
                    seen[pair] = strategy
            logger.info(
                f"  [smart/{strategy}] "
                f"{sum(1 for s in seen.values() if s == strategy):,} unique pairs"
            )
        return list(seen.keys())


def sample_positive_edges(
    graph: nx.Graph,
    source_edges: Optional[List[Tuple[str, str]]] = None,
) -> List[Tuple[str, str]]:
    """
    Return positive (existing or future) edges as canonical sorted pairs.

    If source_edges is None, returns all graph edges (for training).
    Otherwise, returns source_edges filtered to canonical form (for val/test).
    """
    if source_edges is None:
        return [make_pair(u, v) for u, v in graph.edges()]
    return list({make_pair(a, b) for a, b in source_edges})


def _anchor_neg_random(
    anchor: str,
    nodes: np.ndarray,
    exclude: Set[Tuple[str, str]],
    n: int,
    rng: np.random.RandomState,
    weights: Optional[np.ndarray] = None,
) -> List[str]:
    """Sample n replacement nodes for anchor (random or degree-biased)."""
    result: List[str] = []
    max_attempts = n * 50
    attempts = 0
    while len(result) < n and attempts < max_attempts:
        attempts += 1
        idx = (
            rng.choice(len(nodes), p=weights)
            if weights is not None
            else rng.randint(0, len(nodes))
        )
        node = nodes[idx]
        if node != anchor and make_pair(anchor, node) not in exclude:
            result.append(node)
            exclude = exclude | {make_pair(anchor, node)}  # don't re-use
    return result


def _anchor_neg_smart(
    anchor: str,
    neighbors: Dict[str, Set[str]],
    degree: Dict[str, int],
    nodes: np.ndarray,
    degree_weights: np.ndarray,
    exclude: Set[Tuple[str, str]],
    n: int,
    rng: np.random.RandomState,
    strategy_weights: Optional[Dict[str, float]] = None,
) -> List[str]:
    """
    Sample n replacement nodes for anchor using the same four sub-strategies
    as SmartCandidateSampler, each applied per-anchor:

      2hop: 2-hop neighbours weighted by common-neighbour count
      resource_alloc: 2-hop neighbours weighted by Σ 1/deg(common_nb)
      pref_attach: all nodes weighted by degree (preferential attachment)
      random: uniform random

    Strategy ratios mirror SmartCandidateSampler defaults unless overridden.
    Any shortfall across strategies is filled with random sampling.
    """
    if strategy_weights is None:
        strategy_weights = {
            "2hop": 0.40,
            "resource_alloc": 0.30,
            "pref_attach": 0.15,
            "random": 0.15,
        }

    anchor_nb = neighbors.get(anchor, set())
    seen: Set[str] = set()
    result: List[str] = []

    def _add(node: str) -> bool:
        if node == anchor or node in seen or make_pair(anchor, node) in exclude:
            return False
        seen.add(node)
        result.append(node)
        return True

    # 2hop: weighted by common-neighbour count
    n_2hop = int(n * strategy_weights.get("2hop", 0.40))
    pool_2hop: Dict[str, int] = {}
    for nb in anchor_nb:
        for two_hop in neighbors.get(nb, set()):
            if two_hop != anchor and make_pair(anchor, two_hop) not in exclude:
                pool_2hop[two_hop] = pool_2hop.get(two_hop, 0) + 1
    if pool_2hop and n_2hop > 0:
        pl = list(pool_2hop.keys())
        pw = np.array([pool_2hop[x] + 1 for x in pl], dtype=np.float64)
        pw /= pw.sum()
        chosen = rng.choice(len(pl), size=min(n_2hop, len(pl)), replace=False, p=pw)
        for i in chosen:
            _add(pl[i])

    # resource_alloc: weighted by Σ 1/deg(common_nb)
    n_ra = int(n * strategy_weights.get("resource_alloc", 0.30))
    pool_ra: Dict[str, float] = {}
    for nb in anchor_nb:
        d_nb = max(degree.get(nb, 1), 1)
        for two_hop in neighbors.get(nb, set()):
            if two_hop != anchor and make_pair(anchor, two_hop) not in exclude:
                pool_ra[two_hop] = pool_ra.get(two_hop, 0.0) + 1.0 / d_nb
    if pool_ra and n_ra > 0:
        pl = list(pool_ra.keys())
        pw = np.array([pool_ra[x] for x in pl], dtype=np.float64)
        pw /= pw.sum()
        chosen = rng.choice(len(pl), size=min(n_ra, len(pl)), replace=False, p=pw)
        for i in chosen:
            _add(pl[i])

    # pref_attach: all nodes weighted by degree
    n_pa = int(n * strategy_weights.get("pref_attach", 0.15))
    if n_pa > 0:
        attempts = 0
        pa_found = 0
        while pa_found < n_pa and attempts < n_pa * 50:
            attempts += 1
            idx = rng.choice(len(nodes), p=degree_weights)
            if _add(nodes[idx]):
                pa_found += 1

    # random: fill remainder
    n_rand = n - len(result)
    if n_rand > 0:
        extra_exclude = exclude | {make_pair(anchor, r) for r in result}
        extra = _anchor_neg_random(anchor, nodes, extra_exclude, n_rand, rng)
        for node in extra:
            _add(node)

    return result[:n]


def _sample_contrastive_once(
    positive_pairs: List[Tuple[str, str]],
    graph: nx.Graph,
    n_neg_per_pos: int,
    strategy: str,
    base_excluded: Set[Tuple[str, str]],
    seed: int,
    pair_id_offset: int,
    strategy_weights: Optional[Dict[str, float]],
) -> Tuple[List[Tuple[str, str]], List[int], List[int]]:
    """Single-pass contrastive negative sampling (inner helper)."""
    rng = np.random.RandomState(seed)
    all_excluded = base_excluded | {make_pair(a, b) for a, b in positive_pairs}

    nodes = np.array(sorted(graph.nodes()))
    raw_deg = np.array(
        [float(graph.nodes[n].get("freq", 1)) + float(graph.degree(n)) for n in nodes],
        dtype=np.float64,
    )
    degree_weights = raw_deg / raw_deg.sum()

    neighbors: Optional[Dict[str, Set[str]]] = None
    degree_dict: Optional[Dict[str, int]] = None
    if strategy == "smart":
        neighbors = {n: set(graph.neighbors(n)) for n in graph.nodes()}
        degree_dict = dict(graph.degree())

    all_pairs: List[Tuple[str, str]] = []
    labels_out: List[int] = []
    pair_ids_out: List[int] = []

    for pid, (a, b) in tqdm(enumerate(positive_pairs), total=len(positive_pairs)):
        global_pid = pair_id_offset + pid
        all_pairs.append((a, b))
        labels_out.append(1)
        pair_ids_out.append(global_pid)

        anchor = a if rng.rand() < 0.5 else b

        if strategy == "smart" and neighbors is not None and degree_dict is not None:
            replacements = _anchor_neg_smart(
                anchor,
                neighbors,
                degree_dict,
                nodes,
                degree_weights,
                all_excluded,
                n_neg_per_pos,
                rng,
                strategy_weights,
            )
        elif strategy == "biased":
            replacements = _anchor_neg_random(
                anchor, nodes, all_excluded, n_neg_per_pos, rng, degree_weights
            )
        else:
            replacements = _anchor_neg_random(
                anchor, nodes, all_excluded, n_neg_per_pos, rng
            )

        for repl in replacements:
            neg = make_pair(anchor, repl)
            all_pairs.append(neg)
            labels_out.append(0)
            pair_ids_out.append(global_pid)
            all_excluded.add(neg)

    return all_pairs, labels_out, pair_ids_out


def sample_contrastive_negatives(
    positive_pairs: List[Tuple[str, str]],
    graph: nx.Graph,
    n_neg_per_pos: int = 1,
    strategy: str = "random",
    excluded: Optional[Set[Tuple[str, str]]] = None,
    seed: int = 42,
    augmentation_factor: int = 1,
    strategy_weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[Tuple[str, str]], np.ndarray, np.ndarray]:
    """
    For each positive pair (A, B), generate n_neg_per_pos negatives by
    anchoring one endpoint and replacing the other.

    One of (A, B) is chosen as the anchor; the replacement node is sampled
    according to `strategy`:
      random : uniform random from all graph nodes
      biased : degree-proportional (popular nodes more likely)
      smart  : four sub-strategies — 2hop, resource_alloc, pref_attach, random
               (weights controlled by strategy_weights dict)

    augmentation_factor > 1 repeats sampling with different seeds and offsets
    pair_ids, creating multiple independent groups per positive pair.
    Use together with smaller n_neg_per_pos to keep total dataset size constant
    while improving batch diversity (more distinct queries per batch).

    Returns:
        all_pairs: positives + negatives, grouped as (pos, neg1, neg2, …) per pair_id
        labels: int32 array, 1 = positive, 0 = negative
        pair_ids: int64 array, same id for a positive and all its negatives
    """
    graph_edges = {make_pair(u, v) for u, v in graph.edges()}
    base_excluded: Set[Tuple[str, str]] = (excluded or set()) | graph_edges

    n_pos = len(positive_pairs)
    all_pairs_out: List[Tuple[str, str]] = []
    all_labels: List[np.ndarray] = []
    all_ids: List[np.ndarray] = []

    for aug_i in range(max(augmentation_factor, 1)):
        aug_seed = seed + aug_i * 7919
        pairs, labels, pair_ids = _sample_contrastive_once(
            positive_pairs,
            graph,
            n_neg_per_pos,
            strategy,
            base_excluded,
            aug_seed,
            pair_id_offset=aug_i * n_pos,
            strategy_weights=strategy_weights,
        )
        all_pairs_out.extend(pairs)
        all_labels.append(np.array(labels, dtype=np.int32))
        all_ids.append(np.array(pair_ids, dtype=np.int64))

    labels_arr = np.concatenate(all_labels)
    ids_arr = np.concatenate(all_ids)
    n_pos_total = int((labels_arr == 1).sum())
    n_neg_total = int((labels_arr == 0).sum())
    logger.info(
        "Contrastive sampling done: %d pos, %d neg "
        "(strategy=%s, n_neg_per_pos=%d, augmentation_factor=%d)",
        n_pos_total,
        n_neg_total,
        strategy,
        n_neg_per_pos,
        augmentation_factor,
    )
    return all_pairs_out, labels_arr, ids_arr


def generate_ranking_candidates(
    graph: nx.Graph,
    test_edges: List[Tuple[str, str]],
    max_candidates: int = 300_000,
    seed: int = 42,
    strategy_weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    """
    Generate candidates for ranking evaluation.

    Falls back to exhaustive enumeration when total pairs ≤ max_candidates.
    Otherwise uses SmartCandidateSampler.

    Returns (candidates, labels) where label=1 if pair is in test_edges.
    """
    rng = np.random.RandomState(seed)
    graph_nodes = sorted(graph.nodes())
    n = len(graph_nodes)
    n_total = n * (n - 1) // 2
    positive_set = {make_pair(a, b) for a, b in test_edges}

    logger.info(
        f"Ranking candidates: {n:,} nodes, {len(positive_set):,} positives, "
        f"{n_total:,} total pairs (pos_rate={len(positive_set) / max(n_total, 1):.5%})"
    )

    if n_total <= max_candidates:
        candidates = [
            make_pair(graph_nodes[i], graph_nodes[j])
            for i in range(n)
            for j in range(i + 1, n)
        ]
    else:
        logger.info("Using SmartCandidateSampler for ranking candidate generation...")
        sampler = SmartCandidateSampler(graph, nodes=graph_nodes, rng=rng)
        candidates = sampler.sample(max_candidates, strategy_weights)

    labels = np.array(
        [1 if p in positive_set else 0 for p in candidates], dtype=np.int32
    )
    n_pos = int(labels.sum())
    logger.info(
        f"Final candidates: {len(candidates):,} "
        f"(pos={n_pos:,}, neg={len(candidates) - n_pos:,}, "
        f"pos_rate={n_pos / max(len(candidates), 1):.5%})"
    )
    return candidates, labels


def compute_and_store_split_features(
    pairs: List[Tuple[str, str]],
    labels: np.ndarray,
    graph: nx.Graph,
    output_path: str,
    article_embeddings: Optional[Dict] = None,
    sparse_years: Optional[List[int]] = None,
    graph_feature_cfg: Optional[Dict[str, Any]] = None,
    embedding_combination: Optional[List[str]] = None,
    window_info: Optional[Dict[str, Any]] = None,
    pair_ids: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compute all feature families for a list of pairs and save as CSV.

    Feature families computed:
      1. Graph topology (compute_graph_features)
      2. Per-year sparse matrix (compute_sparse_matrix_features), if sparse_years given
      3. Article embedding pair features (prefix="emb"), if article_embeddings given

    Embeddings use compute_embedding_pair_features.
    Default methods: article → ["cosine", "l2"].

    Args:
        pairs: list of (concept_a, concept_b) canonical sorted pairs
        labels: integer array (1 = positive, 0 = negative)
        graph: NetworkX graph used for feature computation
        output_path: CSV output path
        article_embeddings: {article_id: vector} for concept embedding derivation
        sparse_years: years for per-year sparse matrix features
        graph_feature_cfg: flags passed to compute_graph_features
        embedding_combination: methods for article embedding features (default: ["cosine", "l2"])
        window_info : dict of extra columns added to output (e.g. window_start, target_year)
    """
    logger.info(f"Computing features for {len(pairs):,} pairs...")

    all_parts: List[np.ndarray] = []
    all_names: List[str] = []

    # Graph topology features
    gf, gn = compute_graph_features(pairs, graph, graph_feature_cfg)
    if gf.shape[1] > 0:
        all_parts.append(gf)
        all_names += gn
        logger.info(f"  Graph features: {gf.shape[1]}")

    # Sparse per-year features
    if sparse_years:
        node_to_idx = {n: i for i, n in enumerate(sorted(graph.nodes()))}
        sf, sn = compute_sparse_matrix_features(pairs, graph, sparse_years, node_to_idx)
        if sf.shape[1] > 0:
            all_parts.append(sf)
            all_names += sn
            logger.info(f"  Sparse matrix features: {sf.shape[1]}")

    # Article (sentence) embedding pair features — cosine + l2 by default
    if article_embeddings is not None:
        concept_embs = compute_concept_embeddings_from_articles(
            graph, article_embeddings
        )
        if concept_embs:
            dim = len(next(iter(concept_embs.values())))
            methods = embedding_combination or ["cosine", "l2"]
            ef, en = compute_embedding_pair_features(
                pairs, concept_embs, methods, dim, prefix="emb"
            )
            if ef.shape[1] > 0:
                all_parts.append(ef)
                all_names += en
                logger.info(f"  Embedding pair features: {ef.shape[1]}")

    X = (
        np.hstack(all_parts).astype(np.float32)
        if all_parts
        else np.zeros((len(pairs), 0), np.float32)
    )
    logger.info(f"  Total features: {X.shape[1]}")

    meta_dict: Dict[str, Any] = {
        "concept_a": [a for a, b in pairs],
        "concept_b": [b for a, b in pairs],
        "label": labels,
    }
    if pair_ids is not None:
        meta_dict["pair_id"] = pair_ids
    meta = pd.DataFrame(meta_dict)
    feat_df = pd.DataFrame(X, columns=all_names)
    df_out = pd.concat([meta, feat_df], axis=1)

    if window_info:
        for k, v in window_info.items():
            df_out[k] = v

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(path, index=False)
    logger.info(f"Saved {len(df_out):,} samples to {path}")
    return df_out
