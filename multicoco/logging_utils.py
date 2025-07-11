"""Utility functions to configure uniform logging across MultiCoCo.

This module exposes a single public function, ``setup_logging``, which applies
consistent logging settings for both console and file outputs and is safe to
call from any process (main or worker).  All modules should import and invoke
this helper **once** at program start-up rather than re-implementing their own
logging logic.

The implementation is a light refactor of the previous inline logic found in
``run.py`` – extracting it here lets every entry-point share the same clean
behaviour.
"""

from __future__ import annotations

import logging
import os
import time
from logging.handlers import RotatingFileHandler
from typing import Optional

from multicoco.utils import TqdmLoggingHandler

# Optional third-party helpers – we degrade gracefully if unavailable.
try:
    from pythonjsonlogger import jsonlogger  # type: ignore
except ImportError:  # pragma: no cover – runtime fallback
    jsonlogger = None  # type: ignore

try:
    import colorlog  # type: ignore
except ImportError:  # pragma: no cover – runtime fallback
    colorlog = None  # type: ignore

__all__ = ["setup_logging"]


class _LoggingConfigLike:  # pylint: disable=too-few-public-methods
    """Structural helper so we can accept any object with the needed attrs."""

    log_dir: str
    log_level: str
    console_output: bool
    verbose: bool
    run_name: Optional[str]


def setup_logging(cfg: _LoggingConfigLike, *, local_rank: Optional[int] = None) -> None:  # noqa: D401,E501
    """Initialise root logger using the supplied configuration.

    The function is **idempotent** – repeated calls will not attach duplicate
    handlers.  Worker processes (i.e. when ``LOCAL_RANK`` is set and not 0)
    receive a minimal CRITICAL-only logger to keep output noise low.

    Args:
        cfg: Any object exposing the fields defined in ``_LoggingConfigLike``.
        local_rank: Override the process rank (defaults to ``$LOCAL_RANK`` env).
    """

    # ---------------------------------------------------------------------
    # Resolve execution context
    # ---------------------------------------------------------------------
    if local_rank is None:
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        except ValueError:
            local_rank = -1

    is_main_process = local_rank in (-1, 0)

    # ---------------------------------------------------------------------
    # Fast-exit for non-main ranks – keep logging silent to avoid clutter.
    # ---------------------------------------------------------------------
    if not is_main_process:
        logging.getLogger().setLevel(logging.CRITICAL)
        return

    # Ensure log directory exists early.
    os.makedirs(cfg.log_dir, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))

    # Remove any pre-existing handlers (e.g. Jupyter re-execution).
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # ------------------------------------------------------------------
    # Console handler – tqdm compatible, colour if *colorlog* present.
    # ------------------------------------------------------------------
    if cfg.console_output:
        console_handler: logging.Handler = TqdmLoggingHandler()
        if colorlog is not None:
            fmt_str = (
                "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - "
                "%(message)s"
            )
            console_formatter = colorlog.ColoredFormatter(
                fmt_str,
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        else:
            console_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # ------------------------------------------------------------------
    # Rotating JSON log file – falls back to plain text if dependency missing
    # ------------------------------------------------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_name = f"multicoco_{cfg.run_name or 'run'}_{timestamp}.log"
    file_path = os.path.join(cfg.log_dir, file_name)

    rotating_handler = RotatingFileHandler(
        file_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )

    if jsonlogger is not None:
        formatter: logging.Formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s %(module)s "
            "%(funcName)s %(lineno)d"
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    rotating_handler.setFormatter(formatter)
    root_logger.addHandler(rotating_handler)

    # ------------------------------------------------------------------
    # Summary INFO-level log – plain text for quick inspection
    # ------------------------------------------------------------------
    summary_path = os.path.join(cfg.log_dir, "summary.log")
    summary_handler = logging.FileHandler(summary_path, mode="a")
    summary_handler.setLevel(logging.INFO)
    summary_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(summary_handler)

    # ------------------------------------------------------------------
    # Quiet down excessively chatty libraries unless verbose requested.
    # ------------------------------------------------------------------
    if not cfg.verbose:
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING) 