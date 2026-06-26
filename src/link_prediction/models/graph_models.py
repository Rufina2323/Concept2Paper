import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.link_prediction.models.mlp import BPRLoss, ContrastiveLoss
from src.link_prediction.calculate_metrics import compute_ranking_metrics

logger = logging.getLogger(__name__)

try:
    from torch_geometric.nn import GATv2Conv

    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    GATv2Conv = None


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _DenseBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GNNLinkPredictor(nn.Module):
    """
    GATConv encoder (3 layers with residuals) + 5-block dense skip-connection head.

    Required args fields
    --------------------
    num_vertices, num_node_features, embedding_dim,
    num_pairwise_features, dnn_hidden_dim,
    gnn_dropout_rate, dnn_dropout_rate
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric required: pip install torch_geometric")

        self.args = args
        gnn_dim = args.num_node_features + args.embedding_dim

        edge_dim = getattr(args, "edge_dim", None)
        self.embedding = nn.Embedding(args.num_vertices, args.embedding_dim)
        self.conv1 = GATv2Conv(
            gnn_dim, gnn_dim, heads=1, concat=False, edge_dim=edge_dim
        )
        self.conv2 = GATv2Conv(
            gnn_dim, gnn_dim, heads=1, concat=False, edge_dim=edge_dim
        )
        self.conv3 = GATv2Conv(
            gnn_dim, gnn_dim, heads=1, concat=False, edge_dim=edge_dim
        )
        self.bn1 = nn.BatchNorm1d(gnn_dim)
        self.bn2 = nn.BatchNorm1d(gnn_dim)
        self.bn3 = nn.BatchNorm1d(gnn_dim)

        pair_dim = gnn_dim * 2 + args.num_pairwise_features
        h = args.dnn_hidden_dim
        d = args.dnn_dropout_rate
        self.dense1 = _DenseBlock(pair_dim, h, d)
        self.dense2 = _DenseBlock(pair_dim + h, h, d)
        self.dense3 = _DenseBlock(pair_dim + h * 2, h, d)
        self.dense4 = _DenseBlock(pair_dim + h * 3, h, d)
        self.dense5 = _DenseBlock(pair_dim + h * 4, 1, d)

    def encode(self, graph, node_features: torch.Tensor) -> torch.Tensor:
        edge_index = graph.edge_index
        edge_attr = getattr(graph, "edge_attr", None)
        x = torch.cat([node_features, self.embedding.weight], dim=-1)
        x = x + F.dropout(
            F.relu(self.bn1(self.conv1(x, edge_index, edge_attr))),
            p=self.args.gnn_dropout_rate,
            training=self.training,
        )
        x = x + F.dropout(
            F.relu(self.bn2(self.conv2(x, edge_index, edge_attr))),
            p=self.args.gnn_dropout_rate,
            training=self.training,
        )
        x = x + F.dropout(
            F.relu(self.bn3(self.conv3(x, edge_index, edge_attr))),
            p=self.args.gnn_dropout_rate,
            training=self.training,
        )
        return x

    def decode(
        self,
        node_embs: torch.Tensor,
        vertex_pairs: torch.Tensor,
        pairwise: torch.Tensor,
    ) -> torch.Tensor:
        h = torch.cat(
            [node_embs[vertex_pairs[:, 0]], node_embs[vertex_pairs[:, 1]], pairwise],
            dim=1,
        )
        h = torch.cat([h, self.dense1(h)], dim=1)
        h = torch.cat([h, self.dense2(h)], dim=1)
        h = torch.cat([h, self.dense3(h)], dim=1)
        h = torch.cat([h, self.dense4(h)], dim=1)
        return self.dense5(h).squeeze(1)

    def forward(self, graph, node_features, vertex_pairs, pairwise):
        return self.decode(self.encode(graph, node_features), vertex_pairs, pairwise)


class GNNTrainer:
    """
    Full-batch GNN training with early stopping on validation loss.
    Prediction is done in mini-batches to manage memory.
    """

    def __init__(
        self,
        model: GNNLinkPredictor,
        graph,
        node_features: torch.Tensor,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        loss: str = "contrastive",
    ):
        self.device = _device()
        self.model = model.to(self.device)
        self.graph = graph.to(self.device) if hasattr(graph, "to") else graph
        self.node_features = node_features.to(self.device)
        self.optim = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = BPRLoss() if loss == "bpr" else ContrastiveLoss()
        self.best_state: Optional[dict] = None
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def _epoch(
        self,
        vertex_pairs: np.ndarray,
        pairwise: np.ndarray,
        labels: np.ndarray,
        train: bool,
        batch_size: int,
    ) -> float:
        """
        One pass over all pairs in mini-batches.
        Training: re-encodes the full graph per batch so encoder gradients flow.
        Inference: encodes once, decodes in batches (no grad).
        """
        n = len(vertex_pairs)
        idx = np.random.permutation(n) if train else np.arange(n)
        total, count = 0.0, 0

        if not train:
            self.model.eval()
            with torch.no_grad():
                node_embs = self.model.encode(self.graph, self.node_features)
            for start in range(0, n, batch_size):
                batch = idx[start : start + batch_size]
                vp = torch.from_numpy(vertex_pairs[batch]).long().to(self.device)
                pw = torch.from_numpy(pairwise[batch]).float().to(self.device)
                lb = torch.from_numpy(labels[batch].astype(np.float32)).to(self.device)
                logits = self.model.decode(node_embs, vp, pw)
                total += self.loss_fn(logits, lb).item()
                count += 1
        else:
            self.model.train()
            for start in range(0, n, batch_size):
                batch = idx[start : start + batch_size]
                vp = torch.from_numpy(vertex_pairs[batch]).long().to(self.device)
                pw = torch.from_numpy(pairwise[batch]).float().to(self.device)
                lb = torch.from_numpy(labels[batch].astype(np.float32)).to(self.device)
                # Re-encode per batch so the full graph's encoder parameters receive gradients
                node_embs = self.model.encode(self.graph, self.node_features)
                logits = self.model.decode(node_embs, vp, pw)
                loss = self.loss_fn(logits, lb)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total += loss.item()
                count += 1

        return total / max(count, 1)

    def fit(
        self,
        train_pairs: np.ndarray,
        train_pairwise: np.ndarray,
        train_labels: np.ndarray,
        val_pairs: np.ndarray,
        val_pairwise: np.ndarray,
        val_labels: np.ndarray,
        epochs: int = 100,
        patience: int = 10,
        batch_size: int = 4096,
    ) -> int:
        no_improve = 0
        for epoch in range(1, epochs + 1):
            tl = self._epoch(
                train_pairs,
                train_pairwise,
                train_labels,
                train=True,
                batch_size=batch_size,
            )
            vl = self._epoch(
                val_pairs, val_pairwise, val_labels, train=False, batch_size=batch_size
            )

            if vl < self.best_val_loss:
                self.best_val_loss = vl
                self.best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                self.best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 10 == 0 or no_improve == 0:
                logger.info("  Epoch %3d | train=%.4f val=%.4f", epoch, tl, vl)

            if no_improve >= patience:
                logger.info("  Early stop at epoch %d", epoch)
                break

        if self.best_state:
            self.model.load_state_dict(self.best_state)
        self.model.eval()
        return self.best_epoch

    def predict(
        self,
        vertex_pairs: np.ndarray,
        pairwise: np.ndarray,
        batch_size: int = 4096,
    ) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            node_embs = self.model.encode(self.graph, self.node_features)

        preds: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(vertex_pairs), batch_size):
                vp = (
                    torch.from_numpy(vertex_pairs[i : i + batch_size])
                    .long()
                    .to(self.device)
                )
                pw = (
                    torch.from_numpy(pairwise[i : i + batch_size])
                    .float()
                    .to(self.device)
                )
                logits = self.model.decode(node_embs, vp, pw)
                preds.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(preds)

    def evaluate_ranking_with_labels(
        self,
        vertex_pairs: np.ndarray,
        pairwise: np.ndarray,
        y: np.ndarray,
        k_values=None,
    ):
        return compute_ranking_metrics(
            y, self.predict(vertex_pairs, pairwise), k_values
        )

    def save(self, path: str, args_dict: dict) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        loss_name = "bpr" if isinstance(self.loss_fn, BPRLoss) else "contrastive"
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "gnn_args": args_dict,
                "loss": loss_name,
            },
            path,
        )
        logger.info("Saved GNN -> %s", path)

    @classmethod
    def load(cls, path: str, graph, node_features: torch.Tensor) -> "GNNTrainer":
        device = _device()
        ckpt = torch.load(path, map_location=device, weights_only=False)
        args = argparse.Namespace(**ckpt["gnn_args"])
        model = GNNLinkPredictor(args)
        model.load_state_dict(ckpt["state_dict"])
        loss_name = ckpt.get("loss", "contrastive")
        trainer = cls.__new__(cls)
        trainer.device = device
        trainer.model = model.to(device)
        trainer.graph = graph.to(device) if hasattr(graph, "to") else graph
        trainer.node_features = node_features.to(device)
        trainer.model.eval()
        trainer.optim = None
        trainer.loss_fn = BPRLoss() if loss_name == "bpr" else ContrastiveLoss()
        trainer.best_state = None
        trainer.best_val_loss = float("inf")
        trainer.best_epoch = 0
        return trainer
