import logging
import pickle
from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)

_META = frozenset(
    {
        "concept_a",
        "concept_b",
        "label",
        "pair_id",
        "window_start",
        "window_end",
        "target_year",
    }
)


def classify_features(columns: List[str]) -> Dict[str, List[str]]:
    """Return {group_name: [col, ...]} for structure / emb / all."""
    all_f = [c for c in columns if c not in _META]
    structure = [c for c in all_f if not c.startswith(("emb_"))]
    emb = [c for c in all_f if c.startswith("emb_")]
    return {"structure": structure, "emb": emb, "all": all_f}


def _select_cols(
    groups: Dict[str, List[str]], feature_groups: Union[str, List[str]]
) -> List[str]:
    if isinstance(feature_groups, str):
        return list(groups[feature_groups])
    seen, cols = set(), []
    for g in feature_groups:
        for c in groups.get(g, []):
            if c not in seen:
                cols.append(c)
                seen.add(c)
    return cols


def _fit_scaler(X: np.ndarray, cols: List[str]) -> StandardScaler:
    """Fit StandardScaler but skip cosine columns (already in [-1, 1])."""
    scaler = StandardScaler()
    scaler.fit(X)
    for i, col in enumerate(cols):
        if "cosine" in col:
            scaler.mean_[i] = 0.0
            scaler.scale_[i] = 1.0
    return scaler


class PairDataset(Dataset):
    def __init__(
        self, X: np.ndarray, y: np.ndarray, pair_ids: Optional[np.ndarray] = None
    ):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.pair_ids: Optional[torch.Tensor] = (
            torch.from_numpy(pair_ids).long() if pair_ids is not None else None
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GroupedBatchSampler(Sampler):
    """
    Yields batches that contain only complete (pos + neg) groups.

    Each group shares a pair_id produced by sample_contrastive_negatives().
    Groups are shuffled each epoch; indices within a group keep their order
    so the positive always comes first (index 0 in the group).

    Args:
        pair_ids  : int array, one entry per dataset sample
        batch_size : target number of samples per batch (groups are never split)
        shuffle   : shuffle group order each epoch
    """

    def __init__(self, pair_ids: np.ndarray, batch_size: int, shuffle: bool = True):
        groups: Dict[int, List[int]] = defaultdict(list)
        for i, gid in enumerate(pair_ids):
            groups[int(gid)].append(i)
        self._groups = list(groups.values())
        self._batch_size = batch_size
        self._shuffle = shuffle

    def __iter__(self):
        order = list(range(len(self._groups)))
        if self._shuffle:
            np.random.shuffle(order)
        batch: List[int] = []
        for g in order:
            batch.extend(self._groups[g])
            if len(batch) >= self._batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        total = sum(len(g) for g in self._groups)
        return max(1, (total + self._batch_size - 1) // self._batch_size)


class DataSplit(NamedTuple):
    train_ds: PairDataset
    val_ds: PairDataset
    test_ds: PairDataset
    scaler: StandardScaler
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_test: pd.DataFrame
    feature_cols: List[str]


def load_datasets(
    train_path: str,
    val_path: str,
    test_path: str,
    feature_groups: Union[str, List[str]],
) -> DataSplit:
    """
    Load train/val/test CSVs, select features by group, fit scaler, and return PairDatasets ready for model training.

    Args:
        train_path, val_path, test_path : paths to CSV files
        feature_groups : "structure" | "emb" | "all"

    Returns:
        DataSplit with three PairDatasets, the fitted scaler, original DataFrames, and the list of selected feature column names.
    """
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    logger.info(
        "Loaded CSVs — train: %d, val: %d, test: %d",
        len(df_train),
        len(df_val),
        len(df_test),
    )

    groups = classify_features(df_train.columns.tolist())
    cols = _select_cols(groups, feature_groups)
    logger.info("Selected %d features from groups %s", len(cols), feature_groups)

    def _array(df: pd.DataFrame) -> np.ndarray:
        return np.nan_to_num(df[cols].values.astype(np.float32))

    X_train, X_val, X_test = _array(df_train), _array(df_val), _array(df_test)

    scaler = _fit_scaler(X_train, cols)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    y_train = df_train["label"].values.astype(np.int32)
    y_val = df_val["label"].values.astype(np.int32)
    y_test = df_test["label"].values.astype(np.int32)

    def _pair_ids(df: pd.DataFrame) -> Optional[np.ndarray]:
        if "pair_id" in df.columns:
            return df["pair_id"].values.astype(np.int64)
        return None

    return DataSplit(
        train_ds=PairDataset(X_train, y_train, _pair_ids(df_train)),
        val_ds=PairDataset(X_val, y_val, _pair_ids(df_val)),
        test_ds=PairDataset(X_test, y_test),
        scaler=scaler,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        feature_cols=cols,
    )


class GNNData:
    """
    Loads a networkx Graph pkl (created by create_concept_datasets.py) and prepares everything needed for GNN link prediction:
      - PyG Data object (edge_index + self-loops)
      - log-normalised degree node features  [N, 1]
      - vertex-pair integer indices per split
      - pairwise feature arrays (taken from the DataSplit already provided)

    Args:
        data_split  : DataSplit returned by load_datasets() for the pairwise features
        graph_path  : path to the graph_{start}_{end}.pkl networkx Graph file
    """

    def __init__(self, data_split: DataSplit, graph_path: str):
        from torch_geometric.data import Data
        from torch_geometric.utils import add_self_loops

        # Load nx.Graph
        with open(graph_path, "rb") as f:
            G: nx.Graph = pickle.load(f)
        logger.info(
            "Loaded graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges()
        )

        # Build concept -> integer index from graph nodes
        # (restrict to concepts that actually appear in the datasets)
        all_concepts: set = set(G.nodes())
        for split_df in (data_split.df_train, data_split.df_val, data_split.df_test):
            all_concepts.update(split_df["concept_a"].tolist())
            all_concepts.update(split_df["concept_b"].tolist())
        concepts = sorted(all_concepts)
        self.concept_to_idx: Dict[str, int] = {c: i for i, c in enumerate(concepts)}
        n_nodes = len(concepts)
        self.num_vertices = n_nodes

        # Build PyG edge_index + edge_attr
        # edge_attr columns: [log_weight, log_common_neighbors, resource_allocation]
        node_degree = np.zeros(n_nodes, dtype=np.float32)
        rows, cols_arr = [], []
        raw_weights: list = []

        # Build neighbor sets for structural edge features
        g_neighbors: dict = {c: set() for c in concepts}
        for u, v in G.edges():
            iu = self.concept_to_idx.get(u, -1)
            iv = self.concept_to_idx.get(v, -1)
            if iu >= 0 and iv >= 0 and iu != iv:
                rows += [iu, iv]
                cols_arr += [iv, iu]
                w = float(G[u][v].get("weight", 1.0))
                raw_weights += [w, w]
                node_degree[iu] += 1
                node_degree[iv] += 1
                g_neighbors[u].add(v)
                g_neighbors[v].add(u)

        if rows:
            src = torch.tensor(rows, dtype=torch.long)
            dst = torch.tensor(cols_arr, dtype=torch.long)
        else:
            src = dst = torch.zeros(0, dtype=torch.long)

        # Compute edge structural features (only for actual graph edges)
        ea_rows: list = []
        edge_list = list(G.edges())
        for u, v in edge_list:
            iu = self.concept_to_idx.get(u, -1)
            iv = self.concept_to_idx.get(v, -1)
            if iu < 0 or iv < 0 or iu == iv:
                continue
            nb_u = g_neighbors[u]
            nb_v = g_neighbors[v]
            common = nb_u & nb_v
            cn = len(common)
            ra = sum(1.0 / max(len(g_neighbors[z]), 1) for z in common)
            w = float(G[u][v].get("weight", 1.0))
            ea_rows += [[w, float(cn), ra], [w, float(cn), ra]]  # bidirectional

        if ea_rows:
            ea_arr = np.array(ea_rows, dtype=np.float32)
            for col in range(ea_arr.shape[1]):
                col_max = ea_arr[:, col].max()
                if col_max > 0:
                    ea_arr[:, col] = np.log1p(ea_arr[:, col]) / np.log1p(col_max)
            edge_attr = torch.from_numpy(ea_arr)
        else:
            edge_attr = torch.zeros((0, 3), dtype=torch.float32)

        edge_index = (
            torch.stack([src, dst], dim=0)
            if rows
            else torch.zeros((2, 0), dtype=torch.long)
        )

        # Self-loops: fill edge_attr with 1.0 (maximum weight, no common neighbors applies)
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr=edge_attr, fill_value=1.0, num_nodes=n_nodes
        )
        self.graph = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=n_nodes)
        self.num_edge_features: int = edge_attr.shape[1]  # 3

        # Node features: [log-degree, log-freq]  [N, 2]
        max_deg = node_degree.max()
        deg_feat = np.log1p(node_degree) / np.log1p(max(max_deg, 1.0))

        node_freq = np.array(
            [
                float(G.nodes[c].get("freq", 0)) if G.has_node(c) else 0.0
                for c in concepts
            ],
            dtype=np.float32,
        )
        max_freq = node_freq.max()
        freq_feat = np.log1p(node_freq) / np.log1p(max(max_freq, 1.0))

        nf = np.stack([deg_feat, freq_feat], axis=1).astype(np.float32)
        self.node_features = torch.from_numpy(nf)
        self.num_node_features = 2

        # Per-split vertex pairs + pairwise features
        self.vertex_pairs: Dict[str, np.ndarray] = {}
        self.pairwise_features: Dict[str, np.ndarray] = {}

        for split, df, ds in (
            ("train", data_split.df_train, data_split.train_ds),
            ("val", data_split.df_val, data_split.val_ds),
            ("test", data_split.df_test, data_split.test_ds),
        ):
            a_idx = (
                df["concept_a"]
                .map(self.concept_to_idx)
                .fillna(0)
                .astype(np.int64)
                .values
            )
            b_idx = (
                df["concept_b"]
                .map(self.concept_to_idx)
                .fillna(0)
                .astype(np.int64)
                .values
            )
            self.vertex_pairs[split] = np.stack([a_idx, b_idx], axis=1)
            self.pairwise_features[split] = ds.X.numpy()  # already scaled

        self.df_test = data_split.df_test

        self.num_pairwise_features = self.pairwise_features["train"].shape[1]
        logger.info(
            "GNNData ready — nodes: %d, node_feat: %d, pairwise_feat: %d",
            n_nodes,
            self.num_node_features,
            self.num_pairwise_features,
        )
