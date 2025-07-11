"""
Lightweight utilities for the MultiCoCo package.

This module provides utility classes and functions that support the core
MultiCoCo functionality. Currently contains logging utilities to maintain
clean output formatting during training and evaluation.

Classes:
    TqdmLoggingHandler: Custom logging handler that preserves tqdm progress bars
    
Functions:
    log_wandb_samples: Log sample predictions with images to WandB
    log_wandb_compression_ratio: Log latent compression metrics
    log_wandb_multimodal_insights: Log custom multimodal research metrics
"""

import logging
from typing import List, Optional, Dict, Any, Union
from tqdm import tqdm

# WandB import (optional to avoid hard dependency)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

__all__ = [
    "TqdmLoggingHandler", 
    "log_wandb_samples", 
    "log_wandb_compression_ratio",
    "log_wandb_multimodal_insights"
]


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
    max_samples: int = 20
) -> None:
    """
    Log sample predictions with optional images to WandB for qualitative analysis.
    
    Args:
        questions: List of input questions
        labels: List of ground truth labels
        predictions: List of model predictions
        images: Optional list of images (PIL Images or numpy arrays)
        max_samples: Maximum number of samples to log
    """
    if not (WANDB_AVAILABLE and wandb is not None and wandb.run is not None):
        return
    
    try:
        num_samples = min(max_samples, len(questions))
        columns = ["Question", "Ground Truth", "Prediction", "Correct"]
        if images:
            columns.append("Image")
        
        table = wandb.Table(columns=columns)
        
        for i in range(num_samples):
            correct = predictions[i].strip() == labels[i].strip()
            row_data = [questions[i], labels[i], predictions[i], correct]
            
            if images and i < len(images) and images[i] is not None:
                row_data.append(wandb.Image(images[i]))
            elif images:
                row_data.append(None)
            
            table.add_data(*row_data)
        
        wandb.log({"research/sample_predictions": table})
        
    except Exception as e:
        print(f"Warning: Failed to log samples to WandB: {e}")


def log_wandb_compression_ratio(
    processed_samples: List[Dict[str, Any]], 
    scheduled_stage: int,
    stage_name: str = "training"
) -> None:
    """
    Log latent compression ratio metrics to WandB.
    
    Args:
        processed_samples: List of processed samples with reasoning and steps
        scheduled_stage: Current stage number
        stage_name: Name of the current stage (e.g., "training", "evaluation")
    """
    if not (WANDB_AVAILABLE and wandb is not None and wandb.run is not None):
        return
    
    try:
        if not processed_samples:
            return
        
        # Calculate compression ratio: reasoning tokens / latent tokens
        compression_ratios = []
        for sample in processed_samples:
            reasoning_tokens = len(sample.get('reasoning', '').split())
            latent_tokens = len(sample.get('steps', [])) + 1  # +1 for base reasoning
            
            if latent_tokens > 0:
                compression_ratios.append(reasoning_tokens / latent_tokens)
        
        if compression_ratios:
            avg_compression = sum(compression_ratios) / len(compression_ratios)
            
            wandb.log({
                f"data/{stage_name}_compression_ratio": avg_compression,
                f"data/{stage_name}_stage": scheduled_stage,
                f"data/{stage_name}_samples": len(processed_samples),
                f"data/{stage_name}_avg_reasoning_tokens": sum(
                    len(s.get('reasoning', '').split()) for s in processed_samples
                ) / len(processed_samples)
            })
            
    except Exception as e:
        print(f"Warning: Failed to log compression ratio to WandB: {e}")


def log_wandb_multimodal_insights(
    model_info: Dict[str, Any],
    performance_metrics: Dict[str, float],
    stage: Optional[int] = None
) -> None:
    """
    Log custom multimodal and latent-specific insights to WandB.
    
    Args:
        model_info: Dictionary containing model information and statistics
        performance_metrics: Dictionary of performance metrics
        stage: Optional stage number for progressive training
    """
    if not (WANDB_AVAILABLE and wandb is not None and wandb.run is not None):
        return
    
    try:
        # Prepare metrics for logging
        insights = {}
        
        # Model insights
        if model_info:
            for key, value in model_info.items():
                if isinstance(value, (int, float, str)):
                    insights[f"model/{key}"] = value
        
        # Performance insights
        if performance_metrics:
            for key, value in performance_metrics.items():
                if isinstance(value, (int, float)):
                    insights[f"performance/{key}"] = value
        
        # Stage-specific insights
        if stage is not None:
            insights["insights/stage"] = stage
            
        # Log system metrics if available
        try:
            import torch
            if torch.cuda.is_available():
                insights["system/gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
                insights["system/gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024**3   # GB
        except:
            pass
        
        if insights:
            wandb.log(insights)
            
    except Exception as e:
        print(f"Warning: Failed to log multimodal insights to WandB: {e}")


def log_wandb_gradient_histograms(
    model, 
    step: int, 
    log_frequency: int = 100
) -> None:
    """
    Log gradient histograms for latent-related parameters to WandB.
    
    Args:
        model: The model to log gradients for
        step: Current training step
        log_frequency: How often to log (every N steps)
    """
    if not (WANDB_AVAILABLE and wandb is not None and wandb.run is not None):
        return
    
    if step % log_frequency != 0:
        return
    
    try:
        gradient_logs = {}
        for name, param in model.named_parameters():
            if param.grad is not None and ("latent" in name.lower() or "embedding" in name.lower()):
                gradient_logs[f"gradients/{name}"] = wandb.Histogram(param.grad.cpu().numpy())
        
        if gradient_logs:
            wandb.log(gradient_logs, step=step)
            
    except Exception as e:
        print(f"Warning: Failed to log gradient histograms to WandB: {e}")
