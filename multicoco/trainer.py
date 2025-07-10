"""
Custom trainer for MultiCoCo with CoCoNut support.

This module provides a custom trainer class that extends the HuggingFace Trainer
to support CoCoNut (Chain of Continuous Thought) training and evaluation with
multimodal models.
"""

import os
import re
import logging
from types import SimpleNamespace
from typing import Optional, List, Tuple, Dict, Any, Union
import random
import gc
import time

# ** Core libraries
import torch
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

# ** Transformers components
from transformers import Trainer
from transformers.trainer_pt_utils import (
    find_batch_size,
    nested_concat,
    nested_numpify,
    nested_truncate,
    nested_detach
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import LabelSmoother
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_utils import TrainOutput
from transformers.training_args import TrainingArguments

# ** Local imports
from .constants import (
    VALID_CHOICE_NUMBERS,
    CHOICE_MAPPINGS,
    LOSS_IGNORE_INDEX,
    DEFAULT_MAX_NEW_TOKENS,
    IMAGE_TOKEN,
    EVAL_LOG_SEPARATOR,
    SAMPLE_LOG_SEPARATOR,
    DEFAULT_INPUT_MAX_LENGTH
)
from .exceptions import (
    EvaluationError,
    AnswerExtractionError,
    GenerationError
)

logger = logging.getLogger(__name__)


class EvaluationResult:
    """Container for evaluation results."""
    
    def __init__(self, metrics: Dict[str, float], num_samples: int):
        self.metrics = metrics
        self.num_samples = num_samples


class CoCoTrainer(Trainer):
    """
    Custom trainer for MultiCoCo models.
    
    This trainer extends the HuggingFace Trainer to support:
    - Sophisticated answer extraction for multiple choice questions
    - Detailed evaluation logging
    - Proper dtype handling for multimodal inputs
    - Epoch-based training with individual progress bars and evaluation
    
    Attributes:
        best_val_acc: Best validation accuracy achieved
        total_train_steps: Total training steps across all epochs
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the CoCoTrainer.
        
        Args:
            *args: Arguments passed to parent Trainer
            **kwargs: Keyword arguments passed to parent Trainer
        """
        # Remove processor argument as it's handled by parent class
        if 'processor' in kwargs:
            kwargs.pop('processor')
            
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
        
        This method implements training similar to the coconut approach:
        - Individual progress bars for each epoch
        - Evaluation and checkpoint saving after each epoch
        - Detailed logging of training progress
        - Support for resuming from the last checkpoint
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
            trial: Hyperparameter tuning trial object
            ignore_keys_for_eval: Keys to ignore during evaluation
            **kwargs: Additional keyword arguments
            
        Returns:
            TrainOutput containing training results
        """
        
        # Setup training
        self._setup_epoch_training()
        
        # Handle checkpoint resumption
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
                logger.warning("`resume_from_checkpoint` is True but no checkpoint was found. Starting from scratch.")

        # Get training dataloader
        train_dataloader = self.get_train_dataloader()
        
        # Calculate steps per epoch and total steps
        steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        total_steps = steps_per_epoch * int(self.args.num_train_epochs)
        
        logger.info(f"Starting epoch-based training:")
        logger.info(f"  Steps per epoch: {steps_per_epoch}")
        logger.info(f"  Total epochs: {int(self.args.num_train_epochs)}")
        logger.info(f"  Total steps: {total_steps}")
        
        # Initialize model and optimizer
        model = self._wrap_model(self.model_wrapped)
        
        # Create optimizer and scheduler
        self.create_optimizer_and_scheduler(num_training_steps=total_steps)
        
        # Training loop - epoch by epoch
        for epoch in range(start_epoch, int(self.args.num_train_epochs)):
            epoch_start_time = time.time()
            
            logger.info(f"\nStarting Epoch {epoch + 1}/{int(self.args.num_train_epochs)}")
            
            # Run training for this epoch
            self._train_one_epoch(model, train_dataloader, epoch, steps_per_epoch)
            
            # Save checkpoint after epoch
            checkpoint_dir = self._save_epoch_checkpoint(epoch)
            
            # Run evaluation after epoch
            eval_metrics = self._evaluate_after_epoch(epoch)
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            self._log_epoch_summary(epoch, eval_metrics, checkpoint_dir, epoch_time)
            
            # Clean up memory
            gc.collect()
            torch.cuda.empty_cache()
        
        # Final logging
        logger.info("Training completed!")
        
        return TrainOutput(
            global_step=self.total_train_steps,
            training_loss=0.0,  # Will be updated by actual loss tracking
            metrics={}
        )

    def train_coconut_progressive(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial=None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ) -> TrainOutput:
        """
        Progressive curriculum learning for CoCoNut training following original methodology.
        
        This method implements the core CoCoNut multi-stage training:
        - Stage 0: Already completed (CoT training)
        - Stage 1-N: Progressive replacement of reasoning steps with latent tokens
        - Each stage trains for epochs_per_stage epochs
        - Optimizer can be reset between stages
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
            trial: Hyperparameter tuning trial object
            ignore_keys_for_eval: Keys to ignore during evaluation
            **kwargs: Additional keyword arguments
            
        Returns:
            TrainOutput containing training results
        """
        logger.info("Starting CoCoNut progressive curriculum learning training")
        
        # Get CoCoNut configuration from args (set by runner)
        c_thought = getattr(self.args, 'c_thought', 1)
        epochs_per_stage = getattr(self.args, 'epochs_per_stage', 5)
        max_latent_stage = getattr(self.args, 'max_latent_stage', 6)
        reset_optimizer = getattr(self.args, 'reset_optimizer', True)
        uniform_prob = getattr(self.args, 'uniform_prob', 0.0)
        pad_latent_to_max = getattr(self.args, 'pad_latent_to_max', False)
        
        logger.info(f"CoCoNut Configuration:")
        logger.info(f"  c_thought: {c_thought}")
        logger.info(f"  epochs_per_stage: {epochs_per_stage}")
        logger.info(f"  max_latent_stage: {max_latent_stage}")
        logger.info(f"  reset_optimizer: {reset_optimizer}")
        
        # Setup training
        self._setup_epoch_training()
        
        # Calculate total training parameters
        total_stages = max_latent_stage + 1  # +1 for stage 0 (but we skip stage 0 in CoCoNut training)
        total_epochs = total_stages * epochs_per_stage
        
        logger.info(f"Progressive training plan:")
        logger.info(f"  Total stages: {total_stages} (stages 1-{max_latent_stage})")
        logger.info(f"  Epochs per stage: {epochs_per_stage}")
        logger.info(f"  Total epochs: {total_epochs}")
        
        # Handle checkpoint resumption
        start_epoch = 0
        if resume_from_checkpoint:
            checkpoint_path = None
            if resume_from_checkpoint is True:
                checkpoint_path = get_last_checkpoint(self.args.output_dir)
            else:
                checkpoint_path = resume_from_checkpoint
            
            if checkpoint_path:
                logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
                start_epoch = self._load_epoch_checkpoint(checkpoint_path)
        
        # Initialize model wrapper
        model = self._wrap_model(self.model_wrapped)
        
        # Stage-based training loop
        for epoch in range(start_epoch, total_epochs):
            # Calculate current stage so that the first CoCoNut epoch is stage 0 (full CoT)
            current_stage = (epoch // epochs_per_stage)
            stage_epoch = epoch % epochs_per_stage
            
            logger.info(f"\n{'='*80}")
            logger.info(f"EPOCH {epoch + 1}/{total_epochs} - STAGE {current_stage} - STAGE EPOCH {stage_epoch + 1}/{epochs_per_stage}")
            logger.info(f"{'='*80}")
            
            # Apply progressive curriculum to the training dataset
            if hasattr(self.train_dataset, 'apply_progressive_curriculum'):
                self.train_dataset.apply_progressive_curriculum(
                    scheduled_stage=current_stage,
                    c_thought=c_thought,
                    max_latent_stage=max_latent_stage,
                    uniform_prob=uniform_prob,
                    pad_latent_to_max=pad_latent_to_max,
                    shuffle=True
                )
            else:
                logger.warning("Training dataset does not support progressive curriculum")
            
            # Reset optimizer at the beginning of each stage (except first epoch)
            if reset_optimizer and stage_epoch == 0 and epoch > 0:
                logger.info(f"Resetting optimizer for stage {current_stage}")
                # Calculate remaining steps for the rest of training
                train_dataloader = self.get_train_dataloader()
                steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
                remaining_epochs = total_epochs - epoch
                remaining_steps = steps_per_epoch * remaining_epochs
                
                # Recreate optimizer and scheduler
                self.create_optimizer_and_scheduler(num_training_steps=remaining_steps)
            
            # Get updated dataloader with new progressive curriculum
            train_dataloader = self.get_train_dataloader()
            steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            
            epoch_start_time = time.time()
            
            # Run training for this epoch
            self._train_one_epoch(model, train_dataloader, epoch, steps_per_epoch)
            
            # Save checkpoint after epoch
            checkpoint_dir = self._save_epoch_checkpoint(epoch)
            
            # Run evaluation after epoch
            eval_metrics = self._evaluate_after_epoch(epoch)
            eval_metrics['eval_coconut_stage'] = current_stage
            eval_metrics['eval_max_latent_stage'] = max_latent_stage
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            self._log_coconut_epoch_summary(epoch, current_stage, stage_epoch, eval_metrics, checkpoint_dir, epoch_time)
            
            # Clean up memory
            gc.collect()
            torch.cuda.empty_cache()
        
        # Final logging
        logger.info("CoCoNut progressive curriculum learning completed!")
        
        return TrainOutput(
            global_step=self.total_train_steps,
            training_loss=0.0,
            metrics={}
        )

    def _get_last_epoch_checkpoint(self, output_dir: str) -> Optional[str]:
        """Find the last epoch-based checkpoint in the output directory."""
        if not os.path.isdir(output_dir):
            return None
        
        checkpoints = []
        for d in os.listdir(output_dir):
            if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("checkpoint-epoch-"):
                checkpoints.append(d)

        if not checkpoints:
            return None
            
        # Sort checkpoints by epoch number (the integer after the last '-')
        try:
            checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
        except (ValueError, IndexError):
            logger.warning(f"Could not parse epoch number from checkpoint directories in {output_dir}")
            return None
        
        last_checkpoint_name = checkpoints[-1]
        return os.path.join(output_dir, last_checkpoint_name)

    def _load_epoch_checkpoint(self, checkpoint_path: str) -> int:
        """Load state from an epoch-based checkpoint."""
        # Load model, optimizer, and scheduler states using the parent method
        # This is a protected method, but it's the intended way to do this
        self._load_from_checkpoint(checkpoint_path)
        
        # Load custom training info
        training_info_path = os.path.join(checkpoint_path, "training_info.pt")
        if os.path.exists(training_info_path):
            training_info = torch.load(training_info_path)
            start_epoch = training_info.get("epoch", 0)
            self.total_train_steps = training_info.get("total_train_steps", 0)
            self.best_val_acc = training_info.get("best_val_acc", 0.0)
            logger.info(f"Loaded training info: resuming from epoch {start_epoch + 1}")
            return start_epoch
        else:
            logger.warning("Could not find training_info.pt in checkpoint. Resuming epoch from 0.")
            return 0

    def _setup_epoch_training(self) -> None:
        """Setup for epoch-based training."""
        # Put model in training mode
        self.model.train()
        
        # Initialize distributed training if needed
        if self.args.local_rank != -1:
            torch.distributed.barrier()
        
        # Setup wandb logging if enabled
        if hasattr(self.args, 'report_to') and 'wandb' in self.args.report_to:
            import wandb
            if not wandb.run:
                # Use run_name from args, or construct from project info
                project_name = getattr(self.args, 'wandb_project', 'multicoco')
                run_name = getattr(self.args, 'run_name', 'train_multicoco')
                
                wandb.init(
                    project=project_name,
                    name=run_name,
                    config=self.args.to_dict() if hasattr(self.args, 'to_dict') else {}
                )

    def _train_one_epoch(
        self, 
        model: nn.Module, 
        train_dataloader: DataLoader, 
        epoch: int, 
        steps_per_epoch: int
    ) -> None:
        """Train for one epoch with progress bar."""
        
        # Create epoch-specific progress bar
        pbar = tqdm(
            total=steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{int(self.args.num_train_epochs)}",
            colour="blue",
            dynamic_ncols=True
        )
        
        model.train()
        epoch_loss = 0.0
        step_count = 0
        
        for step, batch in enumerate(train_dataloader):
            # Forward pass
            batch = self._prepare_inputs(batch)
            
            # Compute loss
            loss = self.compute_loss(model, batch)
            loss = loss / self.args.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update metrics
            epoch_loss += loss.item()
            self.total_train_steps += 1
            
            # Optimizer step
            if (step + 1) % self.args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                step_count += 1
                pbar.update(1)
                
                # Update progress bar postfix instead of description to avoid flicker
                pbar.set_postfix(
                    step=f"{step_count}/{steps_per_epoch}",
                    loss=f"{loss.item() * self.args.gradient_accumulation_steps:.4f}"
                )
                
                # Log to wandb
                if hasattr(self.args, 'report_to') and 'wandb' in self.args.report_to:
                    import wandb
                    if wandb.run:
                        log_dict = {
                            "train/epoch": epoch + 1,
                            "train/step": epoch * len(train_dataloader) + step,
                            "train/loss": loss.item() * self.args.gradient_accumulation_steps,
                            "train/learning_rate": self.lr_scheduler.get_last_lr()[0]
                        }
                        wandb.log(log_dict)
        
        pbar.close()
        
        # Log epoch training summary
        avg_loss = epoch_loss / max(step_count, 1)
        logger.info(f"  Training completed - Average loss: {avg_loss:.4f}")

    def _save_epoch_checkpoint(self, epoch: int) -> str:
        """Save checkpoint after epoch."""
        checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch-{epoch + 1}")
        
        # Save model state
        self.save_model(checkpoint_dir)
        
        # Save training state
        if hasattr(self, 'optimizer'):
            torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        if hasattr(self, 'lr_scheduler'):
            torch.save(self.lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
        
        # Save training info
        training_info = {
            "epoch": epoch + 1,
            "total_train_steps": self.total_train_steps,
            "best_val_acc": self.best_val_acc
        }
        torch.save(training_info, os.path.join(checkpoint_dir, "training_info.pt"))
        
        logger.info(f"  Checkpoint saved: {checkpoint_dir}")
        return checkpoint_dir

    def _evaluate_after_epoch(self, epoch: int) -> Dict[str, float]:
        """Run evaluation after epoch."""
        logger.info(f"  Running evaluation after epoch {epoch + 1}...")
        
        # Run evaluation
        eval_results = self.evaluate()
        
        # Extract metrics
        if hasattr(eval_results, 'metrics'):
            metrics = eval_results.metrics
        else:
            metrics = eval_results
        
        # Update best accuracy
        current_acc = metrics.get('eval_accuracy', 0.0)
        if current_acc > self.best_val_acc:
            self.best_val_acc = current_acc
            logger.info(f"  New best accuracy: {self.best_val_acc:.4f}")
        
        # Log to wandb
        if hasattr(self.args, 'report_to') and 'wandb' in self.args.report_to:
            import wandb
            if wandb.run:
                wandb_metrics = {"eval/epoch": epoch + 1}
                for key, value in metrics.items():
                    if key.startswith('eval_'):
                        wandb_metrics[f"eval/{key[5:]}"] = value
                wandb.log(wandb_metrics)
        
        return metrics

    def _log_epoch_summary(
        self, 
        epoch: int, 
        eval_metrics: Dict[str, float], 
        checkpoint_dir: str, 
        epoch_time: float
    ) -> None:
        """Log summary of epoch results."""
        accuracy = eval_metrics.get('eval_accuracy', 0.0)
        loss = eval_metrics.get('eval_loss', 0.0)
        
        logger.info(f"\nEpoch {epoch + 1} Summary:")
        logger.info(f"  Training time: {epoch_time:.2f}s")
        logger.info(f"  Evaluation accuracy: {accuracy:.4f}")
        logger.info(f"  Evaluation loss: {loss:.4f}")
        logger.info(f"  Best accuracy so far: {self.best_val_acc:.4f}")
        logger.info(f"  Checkpoint saved to: {checkpoint_dir}")
        
        logger.info("="*80)

    def _log_coconut_epoch_summary(
        self, 
        epoch: int, 
        current_stage: int,
        stage_epoch: int,
        eval_metrics: Dict[str, float], 
        checkpoint_dir: str, 
        epoch_time: float
    ) -> None:
        """Log summary of CoCoNut progressive training epoch results."""
        accuracy = eval_metrics.get('eval_accuracy', 0.0)
        loss = eval_metrics.get('eval_loss', 0.0)
        
        logger.info(f"\nEpoch {epoch + 1} Summary (Stage {current_stage}, Stage Epoch {stage_epoch + 1}):")
        logger.info(f"  Training time: {epoch_time:.2f}s")
        logger.info(f"  Evaluation accuracy: {accuracy:.4f}")
        logger.info(f"  Evaluation loss: {loss:.4f}")
        logger.info(f"  Best accuracy so far: {self.best_val_acc:.4f}")
        logger.info(f"  Current stage: {current_stage}")
        logger.info(f"  Checkpoint saved to: {checkpoint_dir}")
        
        logger.info("="*80)

    def _create_generation_config(self) -> Dict[str, Any]:
        """
        Create generation configuration for evaluation.
        
        Returns:
            Dictionary of generation parameters
        """
        gen_kwargs = getattr(self.args, "generation_kwargs", {}) or {}
        
        # Set default generation parameters
        defaults = {
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "do_sample": False,
            "num_beams": 1,
        }
        
        for key, value in defaults.items():
            if key not in gen_kwargs:
                gen_kwargs[key] = value
        
        # Add pad token ID to suppress warnings - use processing_class instead of deprecated tokenizer
        if self.processing_class.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = self.processing_class.pad_token_id
        
        return gen_kwargs

    def log(self, logs: Dict[str, float], **kwargs) -> None:
        """
        Log metrics with custom handling.
        
        Args:
            logs: Dictionary of metrics to log
            **kwargs: Additional keyword arguments
        """
        super().log(logs, **kwargs)
        
        # Update progress bar with current metrics
        self._update_progress_bar_with_metrics(logs)

    def _update_progress_bar_with_metrics(self, logs: Dict[str, float]) -> None:
        """
        Update progress bar description with current metrics.
        
        Args:
            logs: Dictionary of metrics
        """
        try:
            # Extract relevant metrics
            loss = logs.get('train_loss', logs.get('loss', 0.0))
            lr = logs.get('learning_rate', 0.0)
            
            # Build description string
            desc_parts = []
            if loss > 0:
                desc_parts.append(f"Loss: {loss:.4f}")
            if lr > 0:
                desc_parts.append(f"LR: {lr:.2e}")
            
            # Update tqdm description if available
            if hasattr(self, '_current_progress_bar') and self._current_progress_bar:
                desc = " | ".join(desc_parts) if desc_parts else "Training"
                self._current_progress_bar.set_description(desc)
                
        except Exception as e:
            # Don't let logging errors break training
            logger.debug(f"Failed to update progress bar: {e}")

    def _apply_coconut_masking_to_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply CoCoNut masking to input tensors.
        
        This method applies progressive masking of reasoning tokens
        based on the current stage of CoCoNut training.
        
        Args:
            inputs: Input tensors
            
        Returns:
            Modified input tensors with appropriate masking
        """
        # This would be implemented based on CoCoNut methodology
        # For now, return inputs unchanged
        return inputs

    def extract_answer_choice(self, generated_text: str, is_cot: bool = False) -> str:
        """
        Extract answer choice from generated text using comprehensive strategies.
        
        This method implements multiple extraction strategies to handle various
        answer formats commonly found in VQA datasets.
        
        Args:
            generated_text: The generated response text
            is_cot: Whether this is Chain of Thought evaluation
            
        Returns:
            Extracted answer choice (0, 1, 2, 3) or empty string if not found
            
        Raises:
            AnswerExtractionError: If extraction fails unexpectedly
        """
        if not generated_text:
            return ""
        
        try:
            text = generated_text.strip().lower()
            
            # Strategy 1: Look for explicit answer patterns
            answer = self._extract_number_colon_format(text)
            if answer:
                return answer
            
            # Strategy 2: Look for leading number  
            answer = self._extract_leading_number(text)
            if answer:
                return answer
            
            # Strategy 3: Look for "answer is X" pattern
            answer = self._extract_answer_is_format(text)
            if answer:
                return answer
            
            # Strategy 4: Look for any digit in valid range
            answer = self._extract_any_digit(text)
            if answer:
                return answer
            
            # Strategy 5: Look for word mappings (first, second, etc.)
            answer = self._extract_word_mappings(text)
            if answer:
                return answer
            
            return ""
            
        except Exception as e:
            raise AnswerExtractionError(f"Failed to extract answer from '{generated_text[:50]}...': {e}")

    def _extract_number_colon_format(self, text: str) -> str:
        """Extract from 'answer: X' or 'choice: X' patterns."""
        match = re.search(r'(?:answer|choice|option):\s*([0-3])', text)
        return match.group(1) if match else ""

    def _extract_leading_number(self, text: str) -> str:
        """Extract leading number if it's a valid choice."""
        match = re.match(r'^([0-3])', text.strip())
        return match.group(1) if match else ""

    def _extract_answer_is_format(self, text: str) -> str:
        """Extract from 'the answer is X' patterns."""
        match = re.search(r'(?:the\s+)?answer\s+is\s+([0-3])', text)
        return match.group(1) if match else ""

    def _extract_any_digit(self, text: str) -> str:
        """Extract any valid digit in the text."""
        for char in text:
            if char in VALID_CHOICE_NUMBERS:
                return char
        return ""

    def _extract_word_mappings(self, text: str) -> str:
        """Extract using word-to-number mappings."""
        for word, number in CHOICE_MAPPINGS.items():
            if word in text:
                return number
        return ""

    def compute_metrics(self, p: EvalPrediction) -> Dict[str, float]:
        """
        Compute evaluation metrics from predictions.
        
        Args:
            p: EvalPrediction containing predictions and labels
            
        Returns:
            Dictionary of computed metrics
        """
        # This would normally compute metrics from predictions
        # For now, return placeholder metrics
        return {"accuracy": 0.0}

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> SimpleNamespace:
        """
        Custom evaluation loop with detailed logging and answer extraction.
        
        This method implements a custom evaluation loop that:
        1. Generates predictions for each batch
        2. Extracts answers using sophisticated heuristics
        3. Logs detailed results for analysis
        4. Computes final accuracy metrics
        
        Args:
            dataloader: DataLoader for evaluation data
            description: Description for progress tracking
            prediction_loss_only: Whether to only compute loss
            ignore_keys: Keys to ignore in outputs
            metric_key_prefix: Prefix for metric names
            
        Returns:
            SimpleNamespace containing evaluation results
        """
        model = self._wrap_model(self.model, training=False)
        
        # Initialize tracking variables
        batch_size = dataloader.batch_size
        num_samples = len(dataloader.dataset) if hasattr(dataloader.dataset, '__len__') else 0
        logger.info(f"Running {description} on {num_samples} samples")
        
        # Setup evaluation logging
        log_path = self._setup_evaluation_logging()
        
        try:
            with open(log_path, 'w', encoding='utf-8') as log_file:
                # Write evaluation header
                self._write_evaluation_header(log_file)
                
                # Process batches with progress tracking
                all_results = {'predictions': [], 'labels': [], 'questions': []}
                
                with tqdm(total=len(dataloader), desc=description) as pbar:
                    for step, inputs in enumerate(dataloader):
                        batch_results = self._process_evaluation_batch(inputs, model, log_file)
                        
                        # Accumulate results
                        for key in all_results:
                            all_results[key].extend(batch_results[key])
                        
                        pbar.update(1)
                
                # Compute final metrics
                final_metrics = self._compute_final_metrics(
                    all_results['predictions'], 
                    all_results['labels'], 
                    metric_key_prefix
                )
                
                # Write summary
                self._write_evaluation_summary(log_file, final_metrics, num_samples)
                
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            final_metrics = {f"{metric_key_prefix}_accuracy": 0.0, f"{metric_key_prefix}_loss": -1.0}
        
        # Log final results
        logger.info(f"\n{EVAL_LOG_SEPARATOR}")
        logger.info("FINAL RESULTS")
        logger.info(f"{EVAL_LOG_SEPARATOR}")
        for key, value in final_metrics.items():
            logger.info(f"{key}: {value}")
        logger.info(f"{EVAL_LOG_SEPARATOR}")
        
        # Return results in expected format
        return SimpleNamespace(
            metrics=final_metrics,
            num_samples=num_samples,
            predictions=None,
            label_ids=None
        )

    def _setup_evaluation_logging(self) -> str:
        """Setup evaluation logging and return log file path."""
        eval_config = getattr(self.args, "eval_config", {})
        eval_type = self._get_eval_type_name(eval_config)
        
        log_dir = os.path.join(self.args.output_dir, "eval_logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_filename = f"evaluation_{eval_type}.log"
        log_path = os.path.join(log_dir, log_filename)
        
        logger.info(f"Evaluation log will be saved to: {log_path}")
        return log_path

    def _write_evaluation_header(self, log_file) -> None:
        """Write evaluation header to log file."""
        eval_config = self.args.eval_config
        eval_type = self._get_eval_type_name(eval_config)
        
        log_file.write(f"Evaluation Type: {eval_type.upper()}\n")
        log_file.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"{EVAL_LOG_SEPARATOR}\n\n")

    def _get_eval_type_name(self, eval_config: Dict[str, bool]) -> str:
        """Get evaluation type name from config."""
        if eval_config.get('coconut', False):
            return 'coconut'
        elif eval_config.get('cot', False):
            return 'cot'
        else:
            return 'vanilla'

    def _process_evaluation_batch(
        self, 
        inputs: Dict[str, torch.Tensor], 
        model: nn.Module, 
        log_file
    ) -> Dict[str, List[str]]:
        """Process a single evaluation batch and return results."""
        # Move inputs to model device
        inputs = self._prepare_inputs(inputs)
        
        # Extract batch information
        questions = inputs.get('questions', [])
        answers = inputs.get('answers', [])
        pixel_values = inputs.get('pixel_values')
        
        if len(questions) == 0:
            logger.warning("Empty batch encountered")
            return {'predictions': [], 'labels': [], 'questions': []}
        
        # Generate predictions for each sample in the batch
        predictions = []
        for i in range(len(questions)):
            try:
                # Extract single sample data
                question = questions[i]
                sample_pixel_values = pixel_values[i:i+1] if pixel_values is not None else None
                
                # Generate prediction
                prediction = self._generate_single_prediction(
                    question, sample_pixel_values, model
                )
                predictions.append(prediction)
                
                # Log sample result
                eval_config = getattr(self.args, "eval_config", {})
                is_cot = eval_config.get('cot', False)
                self._log_sample_result(
                    log_file, question, answers[i], prediction, i, is_cot
                )
                
            except Exception as e:
                logger.warning(f"Failed to generate prediction for sample {i}: {e}")
                predictions.append("")
        
        return {
            'predictions': predictions,
            'labels': answers,
            'questions': questions
        }

    def _generate_single_prediction(
        self, 
        question: str, 
        pixel_values: torch.Tensor, 
        model: nn.Module
    ) -> str:
        """Generate prediction for a single question-image pair."""
        try:
            # Check if this is CoT evaluation to determine the prompt
            eval_config = getattr(self.args, "eval_config", {})
            is_cot_eval = eval_config.get('cot', False)

            # Format input text to match training format exactly
            formatted_input = self._format_input_for_generation(question, is_cot_eval)
            
            # Create generation config
            generation_config = self._create_generation_config()
            
            # Access underlying model
            underlying_model = model.model if hasattr(model, 'model') else model
            
            # Ensure correct dtype
            pixel_values = self._ensure_correct_dtype(pixel_values, underlying_model)
            
            # Use direct generation instead of chat to match training format
            response = self._generate_with_training_format(
                underlying_model,
                self.processing_class,  # Use processing_class instead of deprecated tokenizer
                pixel_values,
                formatted_input,
                generation_config
            )
            
            # Clean up response
            return self._clean_generated_response(response)
            
        except Exception as e:
            raise GenerationError(f"Failed to generate prediction: {e}")

    def _format_input_for_generation(self, question: str, is_cot_eval: bool) -> str:
        """
        Format input text to match the training format exactly.
        
        Training format is: "<image>\n{question} {answer_start}"
        where answer_start depends on evaluation type.
        """
        # Use the same image token as training
        from multicoco.constants import IMAGE_TOKEN
        
        # Start with image token and question (matching training format)
        formatted_input = f"{IMAGE_TOKEN}\n{question}"
        
        if not is_cot_eval:
            # For vanilla evaluation, we want the model to give a direct answer
            # Add a space to encourage generation but don't constrain format
            formatted_input += " "
        else:
            # For CoT evaluation, let the model reason freely
            # Add a space to encourage generation
            formatted_input += " "
        
        return formatted_input

    def _generate_with_training_format(
        self,
        model: nn.Module,
        tokenizer,
        pixel_values: torch.Tensor,
        formatted_input: str,
        generation_config: Dict[str, Any]
    ) -> str:
        """
        Generate response using the same format as training data.
        
        This method bypasses the .chat() method to avoid conversation templates
        and uses direct .generate() with manual tokenization to match training format.
        """
        try:
            # Manually tokenize the input (same as training collate_fn)
            model_inputs = tokenizer(
                formatted_input,
                return_tensors='pt',
                add_special_tokens=True,
                padding=False,
                truncation=True,
                max_length=DEFAULT_INPUT_MAX_LENGTH
            )
            
            # Move inputs to model device
            device = model.device
            input_ids = model_inputs['input_ids'].to(device)
            attention_mask = model_inputs['attention_mask'].to(device)
            pixel_values = pixel_values.to(device)
            
            # Set up generation config for the underlying model
            gen_kwargs = generation_config.copy()
            
            # For InternVL, we need to call the model's generate method with proper arguments
            # Check if this is an InternVL model and adjust accordingly
            if hasattr(model, 'generate'):
                # Try InternVL-style generation first
                try:
                    with torch.no_grad():
                        generated_ids = model.generate(
                            pixel_values=pixel_values,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            **gen_kwargs
                        )
                except Exception as e:
                    logger.debug(f"InternVL-style generation failed: {e}")
                    # Fallback to standard generation
                    with torch.no_grad():
                        generated_ids = model.generate(
                            inputs_embeds=self._prepare_inputs_embeds(model, input_ids, pixel_values),
                            attention_mask=attention_mask,
                            **gen_kwargs
                        )
            else:
                raise RuntimeError("Model does not have generate method")
            
            # Decode only the newly generated tokens (excluding input)
            input_length = input_ids.shape[1]
            generated_tokens = generated_ids[0, input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            # Include the actual error message for better debugging
            logger.debug(f"Direct generation failed with error: {str(e)}")
            # Fallback to original chat method if generation fails
            logger.warning(f"Direct generation failed, falling back to chat method: {str(e)}")
            return self._fallback_to_chat_method(
                model, tokenizer, pixel_values, formatted_input, generation_config
            )
    
    def _prepare_inputs_embeds(self, model: nn.Module, input_ids: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
        """Prepare input embeddings for models that need explicit embedding preparation."""
        # This is a simplified version - in practice, you'd need to handle image token embedding properly
        return model.get_input_embeddings()(input_ids)

    def _fallback_to_chat_method(
        self,
        model: nn.Module,
        tokenizer,
        pixel_values: torch.Tensor,
        formatted_input: str,
        generation_config: Dict[str, Any]
    ) -> str:
        """
        Fallback to original chat method if direct generation fails.
        This ensures backward compatibility while we transition to the new method.
        """
        try:
            # Extract just the question part for chat method
            question_part = formatted_input.replace('<image>\n', '').strip()
            
            # Use original chat method
            response = model.chat(
                tokenizer,
                pixel_values,
                question_part,
                generation_config
            )
            
            return response
            
        except Exception as e:
            raise GenerationError(f"Both direct generation and fallback chat failed: {e}")

    def _ensure_correct_dtype(self, pixel_values: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Ensure pixel values have correct dtype for model."""
        if hasattr(model, 'dtype'):
            target_dtype = model.dtype
        elif hasattr(model, 'vision_model') and hasattr(model.vision_model, 'dtype'):
            target_dtype = model.vision_model.dtype
        else:
            target_dtype = torch.bfloat16  # Default
        
        return pixel_values.to(target_dtype)

    def _clean_generated_response(self, response: str) -> str:
        """Clean up generated response by removing thought tokens."""
        eval_config = self.args.eval_config
        
        # Remove thought tokens that might have been generated
        if eval_config.get('coconut', False):
            # Remove latent special tokens that may appear in generation
            from multicoco.constants import LATENT_TOKEN, START_LATENT_TOKEN, END_LATENT_TOKEN
            thought_tokens = [START_LATENT_TOKEN, LATENT_TOKEN, END_LATENT_TOKEN]
            for token in thought_tokens:
                response = response.replace(token, '')
        
        return response.strip()

    def _log_sample_result(
        self, 
        log_file, 
        question: str, 
        ground_truth: str, 
        prediction: str, 
        sample_idx: int,
        is_cot: bool = False
    ) -> None:
        """Log a single sample's result to the detailed log file."""
        if log_file is None:
            return
            
        try:
            log_file.write(f"Sample {sample_idx}:\n")
            log_file.write(f"  Question: {question}\n")
            log_file.write(f"  Ground Truth Answer: {ground_truth}\n")
            log_file.write(f"  Generated Answer: {prediction}\n")
            log_file.write(f"  Extracted Answer: {self.extract_answer_choice(prediction, is_cot)}\n")
            log_file.write(f"  Tokens Generated: {len(self.processing_class.tokenize(prediction))}\n")  # Use processing_class
            log_file.write(f"  Correct: {'Yes' if self.extract_answer_choice(prediction, is_cot) == ground_truth.strip() else 'No'}\n")
            log_file.write(SAMPLE_LOG_SEPARATOR + "\n\n")

        except Exception as e:
            logger.warning(f"Failed to log sample result for sample {sample_idx}: {e}")

    def _compute_final_metrics(
        self, 
        predictions: List[str], 
        labels: List[str], 
        metric_key_prefix: str
    ) -> Dict[str, float]:
        """Compute final evaluation metrics."""
        eval_config = self.args.eval_config
        is_cot = eval_config.get('cot', False)
        
        correct = 0
        total = len(labels)
        
        for pred, label in zip(predictions, labels):
            extracted_answer = self.extract_answer_choice(pred, is_cot)
            if extracted_answer == label.strip():
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        # Build metrics dictionary
        metrics = {
            f"{metric_key_prefix}_accuracy": accuracy,
            f"{metric_key_prefix}_loss": -1.0,  # Placeholder
        }

        return metrics

    def _write_evaluation_summary(
        self, 
        log_file, 
        metrics: Dict[str, float], 
        num_samples: int
    ) -> None:
        """Write evaluation summary to log file."""
        accuracy = metrics.get('eval_accuracy', 0.0)
        correct = int(accuracy * num_samples)
        
        log_file.write("Final Results:\n")
        log_file.write(f"Total Samples: {num_samples}\n")
        log_file.write(f"Correct Predictions: {correct}\n")
        log_file.write(f"Accuracy: {accuracy:.4f}\n")

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Custom prediction step for generation-based evaluation.
        
        Args:
            model: Model to use for prediction
            inputs: Input batch
            prediction_loss_only: Whether to only compute loss
            ignore_keys: Keys to ignore in outputs
            
        Returns:
            Tuple of (loss, predictions, labels)
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        gen_kwargs = self._create_generation_config()

        try:
            generated_tokens = self.model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )

            # In generation mode, there's no loss
            return (None, generated_tokens, None)
            
        except Exception as e:
            raise GenerationError(f"Prediction step failed: {e}")
