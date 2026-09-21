"""Save and restore restartable training checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import torch


def save_training_checkpoint(path: str | Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, scheduler: Any | None = None, scaler: Any | None = None, epoch: int = 0, global_step: int = 0, metrics: dict[str, Any] | None = None, generators: Mapping[str, torch.Generator] | None = None, extra_state: dict[str, Any] | None = None) -> None:
    """Atomically save model and training state."""

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "metrics": dict(metrics or {}),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "generator_states": {} if generators is None else {name: generator.get_state() for name, generator in generators.items()},
        "extra_state": dict(extra_state or {}),
    }

    temporary_path = path.with_name(path.name + ".tmp")

    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def load_training_checkpoint(path: str | Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, scheduler: Any | None = None, scaler: Any | None = None, generators: Mapping[str, torch.Generator] | None = None, map_location: torch.device | str | None = None, restore_rng_state: bool = True) -> dict[str, Any]:
    """Load a training checkpoint and restore the supplied training objects."""

    path = Path(path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"Training checkpoint does not exist: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    if generators is not None:
        stored_generator_states = checkpoint.get("generator_states", {})

        for name, generator in generators.items():
            if name in stored_generator_states:
                generator_state = stored_generator_states[name].cpu()
                generator.set_state(generator_state)

    if restore_rng_state:
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())

        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            cuda_rng_state_all = [
                state.cpu()
                for state in checkpoint["cuda_rng_state_all"]
            ]

            torch.cuda.set_rng_state_all(cuda_rng_state_all)

    return checkpoint

def read_training_checkpoint(
    path: str | Path,
    *,
    map_location: torch.device | str | None = "cpu",
) -> dict[str, Any]:
    """Read checkpoint contents without requiring the model to be constructed first."""

    path = Path(path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"Training checkpoint does not exist: {path}")

    return torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )