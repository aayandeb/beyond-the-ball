"""Shared persistence helpers for trained models and the metrics CSV log."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_ARTIFACTS_DIR = Path("data/processed/models")
DEFAULT_METRICS_CSV = Path("reports/metrics_log.csv")

METRICS_LOG_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "model",
    "task",
    "feature_set",
    "split_seed",
    "n_train",
    "n_val",
    "n_test",
    "val_metric_name",
    "val_metric",
    "test_accuracy",
    "test_macro_f1",
    "test_f1",
    "test_roc_auc",
    "test_pr_auc",
    "extra",
)


def write_json(path: str | Path, payload: Any) -> Path:
    """Pretty-print JSON to ``path`` (creates parents)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return target


def read_json(path: str | Path) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    # numpy scalars/arrays + anything with a ``tolist`` method.
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"Cannot serialize {type(obj).__name__}: {obj!r}")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_metrics_log(row: dict[str, Any], path: str | Path = DEFAULT_METRICS_CSV) -> Path:
    """Append a single row to the shared metrics CSV (creates the file with header)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    full_row = {col: row.get(col) for col in METRICS_LOG_COLUMNS}
    if "extra" in row and not isinstance(row["extra"], str):
        full_row["extra"] = json.dumps(row["extra"], default=_json_default, sort_keys=True)
    df = pd.DataFrame([full_row], columns=list(METRICS_LOG_COLUMNS))
    if target.exists():
        df.to_csv(target, mode="a", index=False, header=False)
    else:
        df.to_csv(target, mode="w", index=False, header=True)
    return target


def model_dir(name: str, base: str | Path = DEFAULT_ARTIFACTS_DIR) -> Path:
    """Return (and create) the artifact directory for ``name``."""
    target = Path(base) / name
    target.mkdir(parents=True, exist_ok=True)
    return target
