"""
Lightweight utilities for the MultiCoCo package.

This module provides utility classes and functions that support the core
MultiCoCo functionality. Currently contains logging utilities to maintain
clean output formatting during training and evaluation, and WandB logging helpers.

Classes:
    TqdmLoggingHandler: Custom logging handler that preserves tqdm progress bars

Functions:
    log_wandb_samples: Log sample tables to WandB for research insights
"""

import logging
from typing import List, Optional, Union
from tqdm import tqdm
import wandb

__all__ = ["TqdmLoggingHandler", "log_wandb_samples"]


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


def log_wandb_samples(
    questions: List[str], 
    labels: List[str], 
    predictions: List[str], 
    images: Optional[List] = None, 
    max_samples: int = 20,
    table_name: str = "research/samples"
) -> None:
    """
    Log sample predictions to WandB as a table for research insights.
    
    Args:
        questions: List of input questions
        labels: List of ground truth labels
        predictions: List of model predictions
        images: Optional list of images (will be converted to wandb.Image)
        max_samples: Maximum number of samples to log
        table_name: Name for the WandB table
    """
    if wandb.run is None:
        return
    
    # Create table columns
    columns = ["Question", "Ground Truth", "Prediction", "Correct"]
    if images:
        columns.append("Image")
    
    table = wandb.Table(columns=columns)
    
    # Add samples to table
    num_samples = min(max_samples, len(questions))
    for i in range(num_samples):
        correct = predictions[i].strip() == labels[i].strip()
        row_data = [questions[i], labels[i], predictions[i], correct]
        
        if images and i < len(images) and images[i] is not None:
            img_data = wandb.Image(images[i]) if not isinstance(images[i], wandb.Image) else images[i]
            row_data.append(img_data)
        elif images:
            row_data.append(None)
        
        table.add_data(*row_data)
    
    wandb.log({table_name: table})
