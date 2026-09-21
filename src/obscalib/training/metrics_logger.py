"""Append-only JSONL logging for training and validation metrics."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


class MetricsLogger:
    """Write structured training records to one JSONL file."""

    def __init__(self, path: str | Path, *, append: bool = False, flush_every_record: bool = True) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.flush_every_record = flush_every_record
        self._file = self.path.open("a" if append else "w", encoding="utf-8")

    def log_metrics(self, metrics: dict[str, Any], *, split: str, epoch: int | None = None, step: int | None = None) -> None:
        """Write one training/validation/test metric record."""

        record = {
            "record_type": "metrics",
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "split": split,
            "epoch": epoch,
            "step": step,
            "metrics": _jsonable(metrics),
        }

        self._write(record)

    def log_config(self, config: Any, *, name: str = "config") -> None:
        """Store one configuration snapshot in the same log."""

        record = {
            "record_type": "config",
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "name": name,
            "config": _jsonable(config),
        }

        self._write(record)

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")

        if self.flush_every_record:
            self._file.flush()


def _jsonable(value: Any) -> Any:
    """Convert common scientific Python objects into JSON-compatible values."""

    if is_dataclass(value):
        return _jsonable(asdict(value))

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()

        return value.detach().cpu().tolist()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)