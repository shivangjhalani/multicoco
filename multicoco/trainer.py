"""
Custom trainer for MultiCoCo with CoCoNut support.

Provides a custom trainer class that extends the HuggingFace Trainer to support
CoCoNut (Chain of Continuous Thought) training and evaluation with multimodal models.
"""

import gc
import logging
import os
import random
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Trainer
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import (
    LabelSmoother,
    find_batch_size,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
)
from transformers.trainer_utils import EvalPrediction, TrainOutput, get_last_checkpoint
from transformers.training_args import TrainingArguments

from .answer_extraction import extract_answer_choice
from .constants import (
    DEFAULT_INPUT_MAX_LENGTH,
    DEFAULT_MAX_NEW_TOKENS,
    EVAL_LOG_SEPARATOR,
    IMAGE_TOKEN,
    LOSS_IGNORE_INDEX,
    SAMPLE_LOG_SEPARATOR,
    VALID_CHOICE_NUMBERS,
)
from .exceptions import AnswerExtractionError, EvaluationError, GenerationError

logger = logging.getLogger(__name__)


class EvaluationResult:
    """Container for evaluation results."""
    
    def __init__(self, metrics: Dict[str, float], num_samples: int):
        self.metrics = metrics
        self.num_samples = num_samples


class CoCoTrainer(Trainer):
    """
    Custom trainer for MultiCoCo models.
    
    Extends the HuggingFace Trainer to support sophisticated answer extraction
    for multiple choice questions, detailed evaluation logging, proper dtype
    handling for multimodal inputs, and epoch-based training with progress bars.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the CoCoTrainer."""
        # Remove processor argument as it's handled by parent class
        kwargs.pop('processor', None)
        super().__init__(*args, **kwargs)
        
        # Initialize trainer state
        self.best_val_acc = 0.0
        self.total_train_steps = 0
        
        logger.info("CoCoTrainer initialized.")

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial=None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ) -> TrainOutput:
        """
        Custom training loop with epoch-based progress bars and evaluation.
        
        Implements training with individual progress bars for each epoch,
        evaluation and checkpoint saving after each epoch, and support
        for resuming from the last checkpoint.
        """
        # Setup training
        self._setup_epoch_training()
        
        # Handle checkpoint resumption
        start_epoch = self._handle_checkpoint_resumption(resume_from_checkpoint)
        
        # Get training dataloader and calculate steps
        train_dataloader = self.get_train_dataloader()
        steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        total_steps = steps_per_epoch * int(self.args.num_train_epochs)
        
        self._log_training_setup(steps_per_epoch, total_steps)
        
        # Initialize model and optimizer
        model = self._wrap_model(self.model_wrapped)
        self.create_optimizer_and_scheduler(num_training_steps=total_steps)
        
        # Training loop - epoch by epoch
        for epoch in range(start_epoch, int(self.args.num_train_epochs)):
            epoch_metrics = self._train_single_epoch(
                model, train_dataloader, epoch, steps_per_epoch
            )
            
            # Cleanup after epoch
            gc.collect()
            torch.cuda.empty_cache()
        
        logger.info("Training completed!")
        
        return TrainOutput(
            global_step=self.total_train_steps,
            training_loss=0.0,
            metrics={}
        )

    def _log_training_setup(self, steps_per_epoch: int, total_steps: int) -> None:
        """Log training setup information."""
        logger.info(f"Starting epoch-based training:")
        logger.info(f"  Steps per epoch: {steps_per_epoch}")
        logger.info(f"  Total epochs: {int(self.args.num_train_epochs)}")
        logger.info(f"  Total steps: {total_steps}")

    def _train_single_epoch(
        self, 
        model: nn.Module, 
        train_dataloader: DataLoader, 
        epoch: int, 
        steps_per_epoch: int
    ) -> Dict[str, float]:
        """Train a single epoch and return metrics."""
        epoch_start_time = time.time()
        logger.info(f"\nStarting Epoch {epoch + 1}/{int(self.args.num_train_epochs)}")
        
        # Run training for this epoch
        self._train_one_epoch(model, train_dataloader, epoch, steps_per_epoch)
        
        # Save checkpoint and evaluate after epoch
        checkpoint_dir = self._save_epoch_checkpoint(epoch)
        eval_metrics = self._evaluate_after_epoch(epoch)
        
        # Log epoch summary
        epoch_time = time.time() - epoch_start_time
        self._log_epoch_summary(epoch, eval_metrics, checkpoint_dir, epoch_time)
        
        return eval_metrics

    def _handle_checkpoint_resumption(
        self, resume_from_checkpoint: Optional[Union[str, bool]]
    ) -> int:
        """Handle checkpoint resumption and return start epoch."""
        start_epoch = 0
        if resume_from_checkpoint:
            checkpoint_path = None
            if resume_from_checkpoint is True:
                checkpoint_path = self._get_last_epoch_checkpoint(self.args.output_dir)
            else:
                checkpoint_path = resume_from_checkpoint
            
            if checkpoint_path:
                logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
                start_epoch = self._load_epoch_checkpoint(checkpoint_path)
            else:
                logger.warning(
                    "`resume_from_checkpoint` is True but no checkpoint found. "
                    "Starting from scratch."
                )
        return start_epoch

    def train_coconut_progressive(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial=None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ) -> TrainOutput:
        """
        Progressive CoCoNut training with curriculum learning stages.
        
        Implements progressive curriculum learning where the model trains
        on different numbers of latent tokens across multiple stages.
        """
        # Setup training
        self._setup_epoch_training()
        
        # Get CoCoNut parameters
        coconut_params = self._get_coconut_parameters()
        
        self._log_coconut_setup(coconut_params)
        
        # Training loop across stages
        for stage in range(coconut_params['max_latent_stage'] + 1):
            self._train_coconut_stage_with_logging(stage, coconut_params)
        
        logger.info("CoCoNut progressive training completed!")
        
        return TrainOutput(
            global_step=self.total_train_steps,
            training_loss=0.0,
            metrics={}
        )

    def _get_coconut_parameters(self) -> Dict[str, Any]:
        """Extract CoCoNut parameters from training arguments."""
        return {
            'c_thought': getattr(self.args, 'c_thought', 1),
            'max_latent_stage': getattr(self.args, 'max_latent_stage', 3),
            'epochs_per_stage': getattr(self.args, 'epochs_per_stage', 5),
            'reset_optimizer': getattr(self.args, 'reset_optimizer', True),
            'uniform_prob': getattr(self.args, 'uniform_prob', 0.0),
            'pad_latent_to_max': getattr(self.args, 'pad_latent_to_max', False)
        }

    def _log_coconut_setup(self, params: Dict[str, Any]) -> None:
        """Log CoCoNut training setup."""
        logger.info(f"Starting CoCoNut progressive training:")
        logger.info(f"  Max latent stage: {params['max_latent_stage']}")
        logger.info(f"  Epochs per stage: {params['epochs_per_stage']}")
        logger.info(f"  C-thought: {params['c_thought']}")

    def _train_coconut_stage_with_logging(self, stage: int, params: Dict[str, Any]) -> None:
        """Train a single CoCoNut stage with proper logging."""
        logger.info(f"\n{'='*60}")
        logger.info(f"STAGE {stage}: Training with {stage} latent tokens")
        logger.info(f"{'='*60}")
        
        # Apply curriculum to dataset
        if hasattr(self.train_dataset, 'apply_progressive_curriculum'):
            self.train_dataset.apply_progressive_curriculum(
                scheduled_stage=stage,
                c_thought=params['c_thought'],
                max_latent_stage=params['max_latent_stage'],
                uniform_prob=params['uniform_prob'],
                pad_latent_to_max=params['pad_latent_to_max']
            )
        
        # Reset optimizer if requested
        if params['reset_optimizer'] and stage > 0:
            self.optimizer = None
            self.lr_scheduler = None
            logger.info("Reset optimizer for new stage")
        
        # Train for this stage
        self._train_coconut_stage(stage, params['epochs_per_stage'])

    def _train_coconut_stage(self, stage: int, epochs_per_stage: int) -> None:
        """Train a single CoCoNut stage."""
        train_dataloader = self.get_train_dataloader()
        steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        total_steps = steps_per_epoch * epochs_per_stage
        
        # Initialize model and optimizer
        model = self._wrap_model(self.model_wrapped)
        if self.optimizer is None:
            self.create_optimizer_and_scheduler(num_training_steps=total_steps)
        
        # Train epochs for this stage
        for stage_epoch in range(epochs_per_stage):
            epoch_metrics = self._train_coconut_single_epoch(
                model, train_dataloader, stage, stage_epoch, steps_per_epoch
            )

    def _train_coconut_single_epoch(
        self, 
        model: nn.Module, 
        train_dataloader: DataLoader, 
        stage: int, 
        stage_epoch: int, 
        steps_per_epoch: int
    ) -> Dict[str, float]:
        """Train a single epoch in CoCoNut mode."""
        epoch_start_time = time.time()
        logger.info(f"Stage {stage}, Epoch {stage_epoch + 1}")
        
        # Run training for this epoch
        self._train_one_epoch(model, train_dataloader, stage_epoch, steps_per_epoch)
        
        # Save checkpoint and evaluate
        checkpoint_dir = self._save_epoch_checkpoint(stage_epoch)
        eval_metrics = self._evaluate_after_epoch(stage_epoch)
        
        # Log coconut-specific epoch summary
        epoch_time = time.time() - epoch_start_time
        self._log_coconut_epoch_summary(
            stage_epoch, stage, stage_epoch, eval_metrics, checkpoint_dir, epoch_time
        )
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return eval_metrics

    def _get_last_epoch_checkpoint(self, output_dir: str) -> Optional[str]:
        """Get the last epoch checkpoint directory."""
        if not os.path.exists(output_dir):
            return None
        
        # Look for epoch-X directories
        epoch_dirs = [d for d in os.listdir(output_dir) if d.startswith('epoch-')]
        if not epoch_dirs:
            return None
        
        # Sort by epoch number and return the latest
        epoch_nums = [int(d.split('-')[1]) for d in epoch_dirs if d.split('-')[1].isdigit()]
        if not epoch_nums:
            return None
        
        latest_epoch = max(epoch_nums)
        return os.path.join(output_dir, f'epoch-{latest_epoch}')

    def _load_epoch_checkpoint(self, checkpoint_path: str) -> int:
        """Load epoch checkpoint and return epoch number."""
        try:
            # Extract epoch number from path
            epoch_num = int(os.path.basename(checkpoint_path).split('-')[1])
            
            # Load the checkpoint using HuggingFace's method
            self._load_from_checkpoint(checkpoint_path)
            
            return epoch_num + 1  # Start from next epoch
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return 0

    def _setup_epoch_training(self) -> None:
        """Setup training state for epoch-based training."""
        # Reset training state
        self.state.global_step = 0
        self.state.epoch = 0
        self.state.total_flos = 0
        
        # Log training setup
        logger.info("Training state initialized for epoch-based training")

    def _train_one_epoch(
        self, 
        model: nn.Module, 
        train_dataloader: DataLoader, 
        epoch: int, 
        steps_per_epoch: int
    ) -> None:
        """Train for one epoch with progress tracking."""
        model.train()
        
        # Create progress bar for this epoch
        pbar = self._create_progress_bar(epoch, train_dataloader)
        
        # Training loop
        epoch_loss = 0.0
        step_count = 0
        
        for step, inputs in enumerate(pbar):
            # Perform training step
            loss = self.training_step(model, inputs)
            
            if loss is not None:
                epoch_loss += loss.item()
                step_count += 1
                
                # Update progress bar
                avg_loss = epoch_loss / step_count
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                # Log to wandb if configured
                self._log_training_step(loss, step)
            
            # Update global step counter
            if step % self.args.gradient_accumulation_steps == 0:
                self.total_train_steps += 1
        
        pbar.close()
        
        # Log epoch summary
        self._log_epoch_training_summary(epoch, epoch_loss, step_count)

    def _create_progress_bar(self, epoch: int, train_dataloader: DataLoader) -> tqdm:
        """Create progress bar for epoch training."""
        return tqdm(
            train_dataloader, 
            desc=f"Epoch {epoch + 1}",
            total=len(train_dataloader),
            disable=not self.is_world_process_zero()
        )

    def _log_training_step(self, loss: torch.Tensor, step: int) -> None:
        """Log training step to wandb if configured."""
        if (step % self.args.gradient_accumulation_steps == 0 and
            getattr(self.args, "report_to", None) and 
            "wandb" in self.args.report_to):
            try:
                import wandb  # type: ignore
                if wandb.run is not None:
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "train/step": self.total_train_steps,
                    })
            except ImportError:
                pass

    def _log_epoch_training_summary(self, epoch: int, epoch_loss: float, step_count: int) -> None:
        """Log epoch training summary."""
        if step_count > 0:
            avg_loss = epoch_loss / step_count
            logger.info(f"Epoch {epoch + 1} training complete. Average loss: {avg_loss:.4f}")

            # Epoch-level WandB logging
            self._log_epoch_to_wandb(avg_loss, epoch)

    def _log_epoch_to_wandb(self, avg_loss: float, epoch: int) -> None:
        """Log epoch metrics to wandb."""
        if getattr(self.args, "report_to", None) and "wandb" in self.args.report_to:
            try:
                import wandb  # type: ignore
                if wandb.run is not None:
                    wandb.log({
                        "train/epoch_loss": avg_loss,
                        "epoch": epoch + 1,
                    })
            except ImportError:
                pass

    def _save_epoch_checkpoint(self, epoch: int) -> str:
        """Save checkpoint after epoch completion."""
        checkpoint_dir = os.path.join(self.args.output_dir, f'epoch-{epoch}')
        
        # Save the checkpoint
        self.save_model(checkpoint_dir)
        
        # Save trainer state
        if self.is_world_process_zero():
            state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
            self.state.save_to_json(state_path)
        
        logger.info(f"Checkpoint saved to: {checkpoint_dir}")

        # Upload checkpoint as a WandB artifact
        self._save_checkpoint_to_wandb(checkpoint_dir, epoch)

        return checkpoint_dir

    def _save_checkpoint_to_wandb(self, checkpoint_dir: str, epoch: int) -> None:
        """Save checkpoint to wandb as artifact."""
        if (self.is_world_process_zero() and 
            getattr(self.args, "report_to", None) and 
            "wandb" in self.args.report_to):
            try:
                import wandb  # type: ignore
                if wandb.run is not None:
                    artifact = wandb.Artifact(
                        name=f"model_epoch_{epoch}",
                        type="model",
                        metadata={"epoch": epoch},
                    )
                    artifact.add_dir(checkpoint_dir)
                    wandb.log_artifact(artifact)
            except ImportError:
                pass

    def _evaluate_after_epoch(self, epoch: int) -> Dict[str, float]:
        """Evaluate model after epoch completion."""
        if self.eval_dataset is not None:
            logger.info(f"Running evaluation after epoch {epoch + 1}")
            eval_result = self.evaluate()
            
            # Track best validation accuracy
            if 'eval_accuracy' in eval_result:
                current_acc = eval_result['eval_accuracy']
                if current_acc > self.best_val_acc:
                    self.best_val_acc = current_acc
                    logger.info(f"New best validation accuracy: {self.best_val_acc:.4f}")
            
            return eval_result
        else:
            logger.info("No evaluation dataset provided, skipping evaluation")
            return {}

    def _log_epoch_summary(
        self, 
        epoch: int, 
        eval_metrics: Dict[str, float], 
        checkpoint_dir: str, 
        epoch_time: float
    ) -> None:
        """Log comprehensive epoch summary."""
        summary_lines = [
            f"\n{EVAL_LOG_SEPARATOR}",
            f"EPOCH {epoch + 1} SUMMARY",
            f"{EVAL_LOG_SEPARATOR}",
            f"Checkpoint: {checkpoint_dir}",
            f"Epoch time: {epoch_time:.2f}s",
        ]
        
        if eval_metrics:
            summary_lines.extend([
                f"Evaluation metrics:",
                *[f"  {k}: {v:.4f}" for k, v in eval_metrics.items()],
            ])
        
        summary_lines.append(f"{EVAL_LOG_SEPARATOR}\n")
        
        for line in summary_lines:
            logger.info(line)

    def _log_coconut_epoch_summary(
        self, 
        epoch: int, 
        current_stage: int,
        stage_epoch: int,
        eval_metrics: Dict[str, float], 
        checkpoint_dir: str, 
        epoch_time: float
    ) -> None:
        """Log CoCoNut-specific epoch summary."""
        summary_lines = [
            f"\n{EVAL_LOG_SEPARATOR}",
            f"COCONUT STAGE {current_stage} - EPOCH {stage_epoch + 1} SUMMARY",
            f"{EVAL_LOG_SEPARATOR}",
            f"Global epoch: {epoch + 1}",
            f"Stage: {current_stage}",
            f"Stage epoch: {stage_epoch + 1}",
            f"Checkpoint: {checkpoint_dir}",
            f"Epoch time: {epoch_time:.2f}s",
        ]
        
        if eval_metrics:
            summary_lines.extend([
                f"Evaluation metrics:",
                *[f"  {k}: {v:.4f}" for k, v in eval_metrics.items()],
            ])
        
        summary_lines.append(f"{EVAL_LOG_SEPARATOR}\n")
        
        for line in summary_lines:
            logger.info(line)
