"""
WandB utilities for MultiCoCo research logging.

Provides utility functions for comprehensive experiment tracking and visualization
following the coconut pattern for manual wandb integration.
"""

import logging
from copy import copy
from typing import Any, Dict, List, Optional, Union

import torch
import wandb
from PIL import Image

logger = logging.getLogger(__name__)


def log_wandb_samples(
    questions: List[str], 
    labels: List[str], 
    predictions: List[str], 
    images: Optional[List[Image.Image]] = None,
    max_samples: int = 20,
    table_name: str = "research/samples"
) -> None:
    """
    Log Q&A samples to wandb following coconut pattern.
    
    Args:
        questions: List of questions
        labels: List of ground truth answers
        predictions: List of model predictions
        images: Optional list of PIL images
        max_samples: Maximum number of samples to log
        table_name: Name for the wandb table
    """
    if wandb.run is None:
        return
        
    try:
        # Create table with appropriate columns
        columns = ["Question", "Ground Truth", "Prediction", "Correct"]
        if images:
            columns.append("Image")
            
        table = wandb.Table(columns=columns)
        
        # Add samples up to max_samples limit
        num_samples = min(max_samples, len(questions), len(labels), len(predictions))
        
        for i in range(num_samples):
            correct = predictions[i].strip() == labels[i].strip()
            
            # Truncate long questions for readability
            question_text = (questions[i][:200] + "..." 
                           if len(questions[i]) > 200 else questions[i])
            
            row_data = [question_text, labels[i], predictions[i], correct]
            
            # Add image if available
            if images and i < len(images) and images[i] is not None:
                row_data.append(wandb.Image(images[i]))
            elif images:
                row_data.append(None)
                
            table.add_data(*row_data)
        
        # Use copy to avoid wandb bug (like coconut does)
        wandb.log({table_name: copy(table)})
        
        logger.info(f"Logged {num_samples} samples to wandb table: {table_name}")
        
    except Exception as e:
        logger.warning(f"Failed to log wandb samples: {e}")


def log_latent_compression_ratio(
    samples: List[Dict[str, Any]], 
    stage: int,
    prefix: str = "data"
) -> None:
    """
    Log latent compression ratio for CoCoNut training.
    
    Args:
        samples: List of processed samples with reasoning and steps
        stage: Current training stage
        prefix: Prefix for wandb logging keys
    """
    if wandb.run is None or not samples:
        return
        
    try:
        # Calculate compression ratio: reasoning tokens / (steps + 1)
        compression_ratios = []
        
        for sample in samples:
            reasoning_length = len(sample.get('reasoning', '').split())
            steps_length = len(sample.get('steps', []))
            
            if steps_length > 0:
                ratio = reasoning_length / (steps_length + 1)
                compression_ratios.append(ratio)
        
        if compression_ratios:
            avg_compression = sum(compression_ratios) / len(compression_ratios)
            wandb.log({
                f"{prefix}/compression_ratio": avg_compression,
                f"{prefix}/stage": stage,
                f"{prefix}/num_samples": len(samples)
            })
            
            logger.info(f"Logged compression ratio: {avg_compression:.3f} for stage {stage}")
            
    except Exception as e:
        logger.warning(f"Failed to log compression ratio: {e}")


def log_gradient_histograms(
    model: torch.nn.Module, 
    step: int,
    sample_interval: int = 100
) -> None:
    """
    Log gradient histograms for latent-related parameters.
    
    Args:
        model: PyTorch model
        step: Current training step
        sample_interval: Log every N steps
    """
    if wandb.run is None or step % sample_interval != 0:
        return
        
    try:
        for name, param in model.named_parameters():
            # Focus on latent-related and embedding parameters
            if any(keyword in name.lower() for keyword in 
                  ['latent', 'embedding', 'embed', 'special']):
                if param.grad is not None:
                    wandb.log({
                        f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())
                    }, step=step)
        
        logger.debug(f"Logged gradient histograms at step {step}")
        
    except Exception as e:
        logger.warning(f"Failed to log gradient histograms: {e}")


def log_model_parameters(
    model: torch.nn.Module,
    prefix: str = "model"
) -> None:
    """
    Log model parameter statistics.
    
    Args:
        model: PyTorch model
        prefix: Prefix for wandb logging keys
    """
    if wandb.run is None:
        return
        
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        wandb.log({
            f"{prefix}/total_parameters": total_params,
            f"{prefix}/trainable_parameters": trainable_params,
            f"{prefix}/trainable_ratio": trainable_params / total_params if total_params > 0 else 0
        })
        
        logger.info(f"Logged model parameters: {trainable_params:,} / {total_params:,} trainable")
        
    except Exception as e:
        logger.warning(f"Failed to log model parameters: {e}")


def log_coconut_stage_metrics(
    stage: int,
    latent_tokens: int,
    dataset_size: int,
    accuracy: Optional[float] = None,
    loss: Optional[float] = None
) -> None:
    """
    Log CoCoNut stage-specific metrics.
    
    Args:
        stage: Current CoCoNut stage
        latent_tokens: Number of latent tokens for this stage
        dataset_size: Size of dataset for this stage
        accuracy: Optional accuracy metric
        loss: Optional loss metric
    """
    if wandb.run is None:
        return
        
    try:
        log_dict = {
            "coconut/stage": stage,
            "coconut/latent_tokens": latent_tokens,
            "coconut/dataset_size": dataset_size,
            "coconut/latent_density": latent_tokens / dataset_size if dataset_size > 0 else 0
        }
        
        if accuracy is not None:
            log_dict["coconut/stage_accuracy"] = accuracy
            
        if loss is not None:
            log_dict["coconut/stage_loss"] = loss
            
        wandb.log(log_dict)
        
        logger.info(f"Logged CoCoNut stage {stage} metrics")
        
    except Exception as e:
        logger.warning(f"Failed to log CoCoNut stage metrics: {e}")


def log_generation_examples(
    questions: List[str],
    responses: List[str],
    stage: Optional[int] = None,
    max_examples: int = 10
) -> None:
    """
    Log generation examples as text artifacts.
    
    Args:
        questions: List of input questions
        responses: List of generated responses
        stage: Optional stage number for CoCoNut training
        max_examples: Maximum number of examples to log
    """
    if wandb.run is None:
        return
        
    try:
        num_examples = min(max_examples, len(questions), len(responses))
        
        examples_text = []
        for i in range(num_examples):
            example = f"Example {i+1}:\n"
            example += f"Question: {questions[i]}\n"
            example += f"Response: {responses[i]}\n"
            example += "-" * 50 + "\n"
            examples_text.append(example)
        
        full_text = "\n".join(examples_text)
        
        # Create text artifact
        artifact_name = f"generation_examples_stage_{stage}" if stage is not None else "generation_examples"
        artifact = wandb.Artifact(artifact_name, type="text")
        
        # Save to temporary file and add to artifact
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(full_text)
            temp_path = f.name
        
        try:
            artifact.add_file(temp_path)
            wandb.log_artifact(artifact)
            logger.info(f"Logged {num_examples} generation examples as artifact")
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
            
    except Exception as e:
        logger.warning(f"Failed to log generation examples: {e}")


def log_evaluation_breakdown(
    predictions: List[str],
    labels: List[str],
    categories: Optional[List[str]] = None,
    prefix: str = "eval"
) -> None:
    """
    Log detailed evaluation breakdown with category analysis.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        categories: Optional categories for each sample
        prefix: Prefix for wandb logging keys
    """
    if wandb.run is None:
        return
        
    try:
        # Overall accuracy
        correct = sum(1 for p, l in zip(predictions, labels) if p.strip() == l.strip())
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0.0
        
        log_dict = {
            f"{prefix}/accuracy": accuracy,
            f"{prefix}/correct": correct,
            f"{prefix}/total": total
        }
        
        # Category breakdown if available
        if categories and len(categories) == len(predictions):
            category_stats = {}
            for pred, label, cat in zip(predictions, labels, categories):
                if cat not in category_stats:
                    category_stats[cat] = {"correct": 0, "total": 0}
                
                category_stats[cat]["total"] += 1
                if pred.strip() == label.strip():
                    category_stats[cat]["correct"] += 1
            
            # Log per-category accuracy
            for cat, stats in category_stats.items():
                cat_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
                log_dict[f"{prefix}/accuracy_{cat}"] = cat_accuracy
                log_dict[f"{prefix}/samples_{cat}"] = stats["total"]
        
        wandb.log(log_dict)
        logger.info(f"Logged evaluation breakdown: {accuracy:.4f} accuracy")
        
    except Exception as e:
        logger.warning(f"Failed to log evaluation breakdown: {e}")


def finish_wandb_run() -> None:
    """Finish the wandb run safely."""
    if wandb.run is not None:
        try:
            wandb.finish()
            logger.info("WandB run finished successfully")
        except Exception as e:
            logger.warning(f"Error finishing wandb run: {e}") 