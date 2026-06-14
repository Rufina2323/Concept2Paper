import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.link_prediction.calculate_metrics import compute_ranking_metrics

logger = logging.getLogger(__name__)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinkPredictionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class BPRLoss(nn.Module):
    """
    Bayesian Personalised Ranking loss.
    For each positive in the batch, randomly pairs it with a negative.
    loss = -mean(log sigmoid(score_pos - score_neg))
    Falls back to BCE when the batch has no positive-negative pairs.
    """

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
        neg_idx = (labels == 0).nonzero(as_tuple=True)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            return F.binary_cross_entropy_with_logits(logits, labels)
        n = min(len(pos_idx), len(neg_idx))
        s_pos = logits[pos_idx[torch.randperm(len(pos_idx), device=logits.device)[:n]]]
        s_neg = logits[neg_idx[torch.randperm(len(neg_idx), device=logits.device)[:n]]]
        return -torch.log(torch.sigmoid(s_pos - s_neg) + 1e-8).mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss on sigmoid-similarity scores.

    For positive pairs (label=1): push similarity toward 1.
        loss = max(0, margin - sigmoid(logit))²

    For negative pairs (label=0): push similarity toward 0.
        loss = sigmoid(logit)²

    With margin=1.0 (default) the positive term becomes (1 - sigmoid(logit))²,
    so positives are trained to give logit → +∞ and negatives logit → -∞.
    Falls back to BCE when the batch has no mixed labels.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels.unique().numel() < 2:
            return F.binary_cross_entropy_with_logits(logits, labels)
        sim = torch.sigmoid(logits)
        pos_loss = torch.clamp(self.margin - sim, min=0).pow(2)
        neg_loss = sim.pow(2)
        return (labels * pos_loss + (1 - labels) * neg_loss).mean()


class MLPTrainer:
    """
    Handles training with early stopping, prediction, evaluation, and
    checkpoint save/load.  No retraining on val — the best-val-epoch model
    is used directly for testing.
    """

    def __init__(
        self,
        model: LinkPredictionMLP,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        loss: str = "contrastive",
    ):
        self.device = _device()
        self.model = model.to(self.device)
        self.optim = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = BPRLoss() if loss == "bpr" else ContrastiveLoss()
        self.best_state: Optional[dict] = None
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def _epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train() if train else self.model.eval()
        total, count = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
                if train:
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                total += loss.item()
                count += 1
        return total / max(count, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 10,
    ) -> int:
        """Train with BPR loss; early stop on val loss. Returns best epoch."""
        no_improve = 0
        for epoch in range(1, epochs + 1):
            tl = self._epoch(train_loader, train=True)
            vl = self._epoch(val_loader, train=False)

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

    def predict(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[i : i + batch_size]).float().to(self.device)
                preds.append(torch.sigmoid(self.model(xb)).cpu().numpy())
        return np.concatenate(preds)

    def evaluate_ranking(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        return compute_ranking_metrics(y, self.predict(X), k_values)

    def save(
        self, path: str, input_dim: int, hidden_dims: List[int], dropout: float
    ) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        loss_name = "bpr" if isinstance(self.loss_fn, BPRLoss) else "contrastive"
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_dim": input_dim,
                "hidden_dims": hidden_dims,
                "dropout": dropout,
                "loss": loss_name,
            },
            path,
        )
        logger.info("Saved MLP -> %s", path)

    @classmethod
    def load(cls, path: str) -> "MLPTrainer":
        device = _device()
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = LinkPredictionMLP(
            ckpt["input_dim"], ckpt["hidden_dims"], ckpt["dropout"]
        )
        model.load_state_dict(ckpt["state_dict"])
        loss_name = ckpt.get("loss", "contrastive")
        trainer = cls.__new__(cls)
        trainer.device = device
        trainer.model = model.to(device)
        trainer.model.eval()
        trainer.optim = None
        trainer.loss_fn = BPRLoss() if loss_name == "bpr" else ContrastiveLoss()
        trainer.best_state = None
        trainer.best_val_loss = float("inf")
        trainer.best_epoch = 0
        return trainer
