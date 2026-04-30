"""Shared neural-network utilities: MLP module + early-stopping training loop."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch (CPU + CUDA) for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(prefer: str | None = None) -> torch.device:
    """Return ``cuda`` or ``mps`` if available, else CPU. ``prefer`` overrides."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MLP(nn.Module):
    """Simple feedforward classifier: Linear -> ReLU -> Dropout, repeated."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden: Sequence[int] = (64, 32),
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_metric: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_metric: float = float("-inf")


def _argmax_predict(logits: torch.Tensor) -> np.ndarray:
    return logits.argmax(dim=1).cpu().numpy()


def train_classifier(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    loss_fn: nn.Module,
    val_metric_fn: Callable[[np.ndarray, np.ndarray], float],
    val_predict_fn: Callable[[torch.Tensor], np.ndarray] = _argmax_predict,
    target_dtype: torch.dtype = torch.long,
    epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    device: torch.device | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[nn.Module, TrainHistory]:
    """Train ``model`` with early stopping on a val metric (higher is better).

    ``val_predict_fn`` converts model outputs (logits) into the array shape
    expected by ``val_metric_fn``. ``target_dtype`` controls how ``y`` is
    cast — ``torch.long`` for cross-entropy, ``torch.float32`` for BCE.
    """
    set_seed(seed)
    device = device or pick_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    gen = torch.Generator()
    gen.manual_seed(seed)
    y_train_t = torch.tensor(np.asarray(y_train), dtype=target_dtype)
    train_loader = _make_loader_typed(
        X_train, y_train_t, batch_size=batch_size, shuffle=True, generator=gen
    )

    history = TrainHistory()
    best_state: dict[str, torch.Tensor] | None = None
    epochs_since_improve = 0

    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(np.asarray(y_val), dtype=target_dtype, device=device)
    y_val_arr = np.asarray(y_val)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * xb.size(0)
            n_samples += xb.size(0)
        train_loss = total_loss / max(n_samples, 1)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss_value = float(loss_fn(val_logits, y_val_t))
            preds = val_predict_fn(val_logits)
            metric = float(val_metric_fn(y_val_arr, preds))

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss_value)
        history.val_metric.append(metric)

        if metric > history.best_metric:
            history.best_metric = metric
            history.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if verbose:
            print(
                f"epoch {epoch:03d} train_loss={train_loss:.4f} "
                f"val_loss={val_loss_value:.4f} val_metric={metric:.4f}"
            )

        if epochs_since_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _make_loader_typed(
    X: np.ndarray,
    y: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator,
) -> DataLoader:
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), y)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        drop_last=False,
    )


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    X: np.ndarray,
    *,
    device: torch.device | None = None,
    binary: bool = False,
) -> np.ndarray:
    """Run the model and return softmax (multiclass) or sigmoid (binary) probs."""
    device = device or next(model.parameters()).device
    model.eval()
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    logits = model(X_t)
    if binary:
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        return torch.sigmoid(logits).cpu().numpy()
    return torch.softmax(logits, dim=1).cpu().numpy()
