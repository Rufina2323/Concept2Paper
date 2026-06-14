import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy import sparse


logger = logging.getLogger(__name__)


def compute_graph_features(
    pairs: List[Tuple[str, str]],
    graph: nx.Graph,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute graph-topology features for pairs from a NetworkX graph.

    Args:
        pairs: node pairs for feature computing
        graph: NetworkX graph
        cfg: config with feature list to compute

    Returns:
        Feature names with values
    """
    if cfg is None:
        cfg = {}

    n = len(pairs)
    features_list: List[np.ndarray] = []
    names: List[str] = []

    neighbor_cache: Dict[str, set] = {}

    def nbrs(c: str) -> set:
        if c not in neighbor_cache:
            neighbor_cache[c] = set(graph.neighbors(c)) if graph.has_node(c) else set()
        return neighbor_cache[c]

    if cfg.get("use_cooccurrence_count", True):
        vals = np.array(
            [
                float(graph[a][b]["weight"]) if graph.has_edge(a, b) else 0.0
                for a, b in pairs
            ],
            dtype=np.float32,
        )
        features_list.append(vals.reshape(-1, 1))
        names.append("cooc_count")

    if cfg.get("use_common_neighbors", True):
        vals = np.array([len(nbrs(a) & nbrs(b)) for a, b in pairs], dtype=np.float32)
        features_list.append(vals.reshape(-1, 1))
        names.append("common_neighbors")

    if cfg.get("use_jaccard", True):
        vals = []
        for a, b in pairs:
            na, nb = nbrs(a), nbrs(b)
            union = na | nb
            vals.append(len(na & nb) / max(len(union), 1))
        features_list.append(np.array(vals, dtype=np.float32).reshape(-1, 1))
        names.append("jaccard")

    if cfg.get("use_adamic_adar", True):
        vals = []
        for a, b in pairs:
            common = nbrs(a) & nbrs(b)
            aa = sum(1.0 / np.log(max(len(nbrs(z)), 2)) for z in common)
            vals.append(aa)
        features_list.append(np.array(vals, dtype=np.float32).reshape(-1, 1))
        names.append("adamic_adar")

    if cfg.get("use_preferential_attachment", True):
        vals = np.array(
            [len(nbrs(a)) * len(nbrs(b)) for a, b in pairs], dtype=np.float32
        )
        features_list.append(vals.reshape(-1, 1))
        names.append("pref_attachment")

    if cfg.get("use_resource_allocation", True):
        vals = []
        for a, b in pairs:
            common = nbrs(a) & nbrs(b)
            ra = sum(1.0 / max(len(nbrs(z)), 1) for z in common)
            vals.append(ra)
        features_list.append(np.array(vals, dtype=np.float32).reshape(-1, 1))
        names.append("resource_allocation")

    if cfg.get("use_salton", True):
        vals = []
        for a, b in pairs:
            da, db = len(nbrs(a)), len(nbrs(b))
            denom = np.sqrt(da * db)
            vals.append(len(nbrs(a) & nbrs(b)) / max(denom, 1e-10))
        features_list.append(np.array(vals, dtype=np.float32).reshape(-1, 1))
        names.append("salton")

    if cfg.get("use_hub_promoted", True):
        vals = []
        for a, b in pairs:
            vals.append(
                len(nbrs(a) & nbrs(b)) / max(min(len(nbrs(a)), len(nbrs(b))), 1)
            )
        features_list.append(np.array(vals, dtype=np.float32).reshape(-1, 1))
        names.append("hub_promoted")

    if cfg.get("use_hub_suppressed", True):
        vals = []
        for a, b in pairs:
            vals.append(
                len(nbrs(a) & nbrs(b)) / max(max(len(nbrs(a)), len(nbrs(b))), 1)
            )
        features_list.append(np.array(vals, dtype=np.float32).reshape(-1, 1))
        names.append("hub_suppressed")

    if cfg.get("use_katz_approx", False):
        # Approximation using paths of length 2 and 3
        beta = float(cfg.get("katz_beta", 0.5))
        beta2 = beta**2
        beta3 = beta**3
        vals = []
        for a, b in pairs:
            na, nb = nbrs(a), nbrs(b)
            cn_count = len(na & nb)
            l3 = sum(len(nbrs(c) & nb) for c in na)
            vals.append(beta2 * cn_count + beta3 * l3)
        features_list.append(np.array(vals, dtype=np.float32).reshape(-1, 1))
        names.append("katz_approx")

    if cfg.get("use_concept_frequency", True):
        freq_a = np.array(
            [
                float(graph.nodes[a]["freq"]) if graph.has_node(a) else 0.0
                for a, _ in pairs
            ],
            dtype=np.float32,
        )
        freq_b = np.array(
            [
                float(graph.nodes[b]["freq"]) if graph.has_node(b) else 0.0
                for _, b in pairs
            ],
            dtype=np.float32,
        )
        features_list += [
            freq_a.reshape(-1, 1),
            freq_b.reshape(-1, 1),
            np.minimum(freq_a, freq_b).reshape(-1, 1),
            np.maximum(freq_a, freq_b).reshape(-1, 1),
            (freq_a + freq_b).reshape(-1, 1),
        ]
        names += ["freq_a", "freq_b", "freq_min", "freq_max", "freq_sum"]

    if cfg.get("use_year_trend", True):
        for label_idx, pos_label in [(0, "a"), (1, "b")]:
            vals = []
            for pair in pairs:
                c = pair[label_idx]
                if graph.has_node(c):
                    yf = graph.nodes[c].get("year_freq", {})
                    if len(yf) >= 2:
                        ys = sorted(yf.keys())
                        freqs = [yf[y] for y in ys]
                        slope = float(
                            np.polyfit(np.arange(len(freqs), dtype=float), freqs, 1)[0]
                        )
                    else:
                        slope = 0.0
                else:
                    slope = 0.0
                vals.append(slope)
            features_list.append(np.array(vals, dtype=np.float32).reshape(-1, 1))
            names.append(f"trend_{pos_label}")

    if not features_list:
        return np.zeros((n, 0), dtype=np.float32), []
    return np.hstack(features_list), names


def _build_sparse_adjacency(
    graph: nx.Graph,
    node_to_idx: Dict[str, int],
    n_nodes: int,
    year: int,
) -> sparse.csr_matrix:
    rows, cols, vals = [], [], []
    for u, v, data in graph.edges(data=True):
        if u not in node_to_idx or v not in node_to_idx:
            continue
        w = float(data.get("year_weights", {}).get(year, 0))
        if w == 0:
            continue
        i, j = node_to_idx[u], node_to_idx[v]
        rows += [i, j]
        cols += [j, i]
        vals += [w, w]
    return sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32
    )


def compute_sparse_matrix_features(
    pairs: List[Tuple[str, str]],
    graph: nx.Graph,
    sparse_years: List[int],
    node_to_idx: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Per-year degree and A^2-based features (5 features × len(sparse_years)).

    For year i:  deg_y{i}_a, deg_y{i}_b, deg2_y{i}_a, deg2_y{i}_b, aa_score_y{i}
    """
    if node_to_idx is None:
        node_to_idx = {n: i for i, n in enumerate(sorted(graph.nodes()))}

    n_nodes = len(node_to_idx)
    n_pairs = len(pairs)
    n_years = len(sparse_years)

    v1_idx = np.array([node_to_idx.get(a, 0) for a, _ in pairs])
    v2_idx = np.array([node_to_idx.get(b, 0) for _, b in pairs])

    features = np.zeros((n_pairs, 5 * n_years), dtype=np.float32)
    names: List[str] = []

    for i, year in enumerate(sorted(sparse_years)):
        A = _build_sparse_adjacency(graph, node_to_idx, n_nodes, year)

        A2 = A.dot(A).toarray()
        A2_norm = A2 / max(A2.max(), 1)

        deg = np.array(A.sum(axis=0)).flatten()
        deg = deg / max(deg.max(), 1)

        deg2 = A2_norm.sum(axis=0).flatten()
        deg2 = deg2 / max(deg2.max(), 1)

        base = i * 5
        features[:, base + 0] = deg[v1_idx]
        features[:, base + 1] = deg[v2_idx]
        features[:, base + 2] = deg2[v1_idx]
        features[:, base + 3] = deg2[v2_idx]
        features[:, base + 4] = A2_norm[v1_idx, v2_idx]

        names += [
            f"deg_y{i}_a",
            f"deg_y{i}_b",
            f"deg2_y{i}_a",
            f"deg2_y{i}_b",
            f"aa_score_y{i}",
        ]

    return features, names


def compute_embedding_pair_features(
    pairs: List[Tuple[str, str]],
    embeddings: Dict[str, np.ndarray],
    methods: List[str],
    dim: int,
    prefix: str = "emb",
) -> Tuple[np.ndarray, List[str]]:
    """
    Vectorised pair features from any concept/node embeddings.
    Missing-concept rows are zeroed.
    """
    zero = np.zeros(dim, dtype=np.float32)
    emb_a = np.stack([embeddings.get(a, zero) for a, _ in pairs]).astype(np.float32)
    emb_b = np.stack([embeddings.get(b, zero) for _, b in pairs]).astype(np.float32)
    valid = np.array([a in embeddings and b in embeddings for a, b in pairs])

    parts: List[np.ndarray] = []
    names: List[str] = []

    for method in methods:
        if method == "cosine":
            na = np.linalg.norm(emb_a, axis=1, keepdims=True).clip(1e-10)
            nb = np.linalg.norm(emb_b, axis=1, keepdims=True).clip(1e-10)
            cos = np.sum((emb_a / na) * (emb_b / nb), axis=1, keepdims=True)
            cos[~valid] = 0.0
            parts.append(cos)
            names.append(f"{prefix}_cosine")
        elif method == "l2":
            l2 = np.linalg.norm(emb_a - emb_b, axis=1, keepdims=True)
            l2[~valid] = 0.0
            parts.append(l2)
            names.append(f"{prefix}_l2")
        elif method == "l1":
            l1 = np.sum(np.abs(emb_a - emb_b), axis=1, keepdims=True)
            l1[~valid] = 0.0
            parts.append(l1)
            names.append(f"{prefix}_l1")
        elif method == "hadamard":
            had = emb_a * emb_b
            had[~valid] = 0.0
            parts.append(had)
            names += [f"{prefix}_had_{d}" for d in range(dim)]
        elif method == "avg":
            avg = (emb_a + emb_b) / 2.0
            avg[~valid] = 0.0
            parts.append(avg)
            names += [f"{prefix}_avg_{d}" for d in range(dim)]

    if not parts:
        return np.zeros((len(pairs), 0), dtype=np.float32), []
    return np.hstack(parts).astype(np.float32), names
