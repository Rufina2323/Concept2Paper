import logging
import pickle
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def make_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def build_graph(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    concepts_col: str = "_concepts",
    year_col: str = "year",
    id_col: Optional[str] = None,
) -> nx.Graph:
    """
    Build an undirected co-occurrence graph from articles in [start_year, end_year].

    Args:
        df : DataFrame with a parsed concepts column (list of strings per row)
        start_year, end_year : inclusive year range
        concepts_col : column with list-of-strings concepts (already parsed)
        id_col : column to use as article ID; uses df row index if None

    Returns:
        nx.Graph with graph-level attrs: start_year, end_year
    """
    G = nx.Graph()
    df_window = df[(df[year_col] >= start_year) & (df[year_col] <= end_year)]
    logger.info(
        f"Building graph {start_year}-{end_year} from {len(df_window):,} articles"
    )

    for row_idx, row in df_window.iterrows():
        year = int(row[year_col])
        article_id = row[id_col] if id_col and id_col in df_window.columns else row_idx
        concepts = row[concepts_col]
        if not isinstance(concepts, list):
            continue
        unique = sorted(set(c for c in concepts if c))

        for c in unique:
            if not G.has_node(c):
                G.add_node(c, freq=0, year_freq={})
            G.nodes[c]["freq"] += 1
            G.nodes[c]["year_freq"][year] = G.nodes[c]["year_freq"].get(year, 0) + 1

        for a, b in combinations(unique, 2):
            if not G.has_edge(a, b):
                G.add_edge(a, b, weight=0, year_weights={}, articles=[])
            G[a][b]["weight"] += 1
            G[a][b]["year_weights"][year] = G[a][b]["year_weights"].get(year, 0) + 1
            G[a][b]["articles"].append(article_id)

    G.graph["start_year"] = start_year
    G.graph["end_year"] = end_year
    logger.info(
        f"Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges"
    )
    return G


def get_test_edges(
    df: pd.DataFrame,
    graph: nx.Graph,
    target_year: int,
    concepts_col: str = "_concepts",
    year_col: str = "year",
) -> List[Tuple[str, str]]:
    """
    Extract edges from target_year that are new with respect to graph:
      - both nodes exist in graph
      - edge does NOT already exist in graph

    These serve as positive labels for evaluation on future data.
    """
    df_target = df[df[year_col] == target_year]
    graph_nodes: Set[str] = set(graph.nodes())
    graph_edges: Set[Tuple[str, str]] = set(make_pair(u, v) for u, v in graph.edges())

    test_edges: Set[Tuple[str, str]] = set()
    for _, row in df_target.iterrows():
        concepts = row[concepts_col]
        if not isinstance(concepts, list):
            continue
        unique = sorted(set(c for c in concepts if c))
        for a, b in combinations(unique, 2):
            if a in graph_nodes and b in graph_nodes:
                pair = make_pair(a, b)
                if pair not in graph_edges:
                    test_edges.add(pair)

    logger.info(
        f"Test edges for year {target_year}: {len(test_edges):,} "
        "(both nodes in graph, not yet connected)"
    )
    return list(test_edges)


def save_graph(graph: nx.Graph, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graph, f)
    logger.info(f"Saved graph to {path}")


def load_graph(path: str) -> nx.Graph:
    with open(Path(path), "rb") as f:
        graph = pickle.load(f)
    logger.info(
        f"Loaded graph from {path}: "
        f"{graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges"
    )
    return graph


def save_test_edges(test_edges: List[Tuple[str, str]], path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(test_edges, f)
    logger.info(f"Saved {len(test_edges):,} test edges to {path}")


def load_test_edges(path: str) -> List[Tuple[str, str]]:
    with open(Path(path), "rb") as f:
        edges = pickle.load(f)
    logger.info(f"Loaded {len(edges):,} test edges from {path}")
    return edges
