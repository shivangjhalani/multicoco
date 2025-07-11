"""Lightweight wrappers around Weights & Biases (wandb) to keep the rest of
MultiCoCo code clean and dependency-agnostic.

Usage pattern::

    from multicoco import wandb_utils as wdb
    run = wdb.init(project="multicoco", name="experiment")
    wdb.log({"train/loss": 0.123})
    wdb.finish()

All helper functions silently no-op when *wandb* is unavailable or the run has
not been initialised, so client code can call them unconditionally.
"""

from __future__ import annotations

import types
from typing import Any, Dict, Optional

try:
    import wandb  # type: ignore

    _wandb_available = True
except ImportError:  # pragma: no cover – optional dependency
    wandb = types.ModuleType("wandb")  # type: ignore
    _wandb_available = False

__all__ = [
    "init",
    "is_active",
    "log",
    "define_default_metrics",
    "finish",
]

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_run: Optional["wandb.sdk.wandb_run.Run"] = None  # type: ignore[name-defined]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def init(project: str, name: str, *, config: Optional[Dict[str, Any]] = None):  # noqa: D401,E501
    """Initialise (or retrieve) a global wandb run.

    Calling ``init`` multiple times returns the same *Run* object.  If the
    library is missing, the function returns ``None`` so callers can maintain
    a local reference if needed.
    """

    global _run  # pylint: disable=global-statement

    if not _wandb_available:
        return None

    if _run is None:
        _run = wandb.init(project=project, name=name, reinit=True)

        # Optionally attach experiment config for reproducibility.
        if config is not None:
            _run.config.update(config, allow_val_change=True)  # type: ignore[arg-type]

        define_default_metrics()

    return _run


def is_active() -> bool:
    """Return ``True`` when wandb is installed *and* a run is active."""

    return _wandb_available and _run is not None  # type: ignore[truthy-iterable]


def log(data: Dict[str, Any]) -> None:  # noqa: D401
    """Write a metrics dictionary to the current wandb run (if active)."""

    if is_active():
        wandb.log(data)  # type: ignore[attr-defined]


def define_default_metrics() -> None:  # noqa: D401
    """Declare a set of common metrics to tidy up dashboards."""

    if not is_active():  # avoid accidental import hits when missing
        return

    # Avoid re-defining metrics on repeated calls.
    if getattr(define_default_metrics, "_done", False):  # type: ignore[attr-defined]
        return

    wandb.define_metric("train/step")  # type: ignore[attr-defined]
    wandb.define_metric("train/batch_loss", step_metric="train/step", summary="min")  # type: ignore[attr-defined]
    wandb.define_metric("train/epoch_loss", summary="min")  # type: ignore[attr-defined]
    wandb.define_metric("eval/accuracy", step_metric="epoch", summary="max")  # type: ignore[attr-defined]
    wandb.define_metric("epoch")  # type: ignore[attr-defined]
    wandb.define_metric("stage")  # type: ignore[attr-defined]

    define_default_metrics._done = True  # type: ignore[attr-defined]


def finish() -> None:  # noqa: D401
    """Finalize the wandb run if active."""

    global _run  # pylint: disable=global-statement

    if is_active():
        wandb.finish()  # type: ignore[attr-defined]
        _run = None 