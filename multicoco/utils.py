"""
Lightweight utilities for the MultiCoCo package.

This module provides utility classes and functions that support the core
MultiCoCo functionality. Currently contains logging utilities to maintain
clean output formatting during training and evaluation.

Classes:
    TqdmLoggingHandler: Custom logging handler that preserves tqdm progress bars
"""

import logging
import json
from typing import Dict, Any

from tqdm import tqdm

__all__ = ["TqdmLoggingHandler", "log_structured_eval"]


class TqdmLoggingHandler(logging.Handler):
    """
    Custom logging handler that routes log messages through tqdm.write.
    
    This handler ensures that log messages do not interfere with tqdm
    progress bars, maintaining clean console output during training
    and evaluation processes.
    
    Args:
        level: The logging level threshold for this handler
    """

    def __init__(self, level: int = logging.NOTSET) -> None:
        """Initialize the TqdmLoggingHandler with specified logging level."""
        super().__init__(level)

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record using tqdm.write to preserve progress bars.
        
        Args:
            record: The LogRecord to be logged
        """
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def log_structured_eval(details: Dict[str, Any], format='console', file_path=None):
    logger = logging.getLogger(__name__)
    if format == 'console':
        logger.info(
            f"Question: {details['question'][:50]}...\n"
            f"Ground Truth: {details['ground_truth']}\n"
            f"Generated: {details['generated_answer'][:50]}...\n"
            f"Extracted: {details['extracted_answer']}\n"
            f"Tokens: {str(details['generated_tokens'][:10])}... (len={len(details['generated_tokens'])})\n"
            f"Correct: {details['correct']}"
        )
    elif format == 'file':
        with open(file_path, 'a') as f:
            f.write(f"{json.dumps(details)}\n")
    elif format == 'json':
        with open(file_path, 'a') as f:
            json.dump(details, f)
            f.write('\n')
