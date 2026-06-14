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
