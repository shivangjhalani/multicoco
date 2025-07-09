# Begin new minimalist utils module
"""
Lightweight utilities used by MultiCoCo.
Currently only provides TqdmLoggingHandler so that log messages do not
break tqdm progress-bars.
"""

import logging
from tqdm import tqdm

__all__ = ["TqdmLoggingHandler"]

class TqdmLoggingHandler(logging.Handler):
    """Route logging records through ``tqdm.write`` so the progress bar stays intact."""

    def __init__(self, level: int = logging.NOTSET) -> None:  # noqa: D401
        super().__init__(level)

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)
