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

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Custom evaluation implementation for MultiCoCo models.
        
        Handles prediction generation and answer extraction for multiple 
        choice questions, providing detailed evaluation metrics.
        """
        try:
            # Use provided dataset or default to trainer's eval dataset
            eval_dataset = eval_dataset or self.eval_dataset
            if eval_dataset is None:
                raise EvaluationError("No evaluation dataset provided")
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Get evaluation dataloader
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            
            # Run evaluation loop
            eval_results = self._evaluation_loop(eval_dataloader, metric_key_prefix)
            
            logger.info(f"Evaluation completed: {len(eval_results)} samples processed")
            return eval_results
            
        except Exception as e:
            raise EvaluationError(f"Evaluation loop failed: {e}") from e

    def _evaluation_loop(
        self, 
        dataloader: DataLoader, 
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        """
        Main evaluation loop that processes batches and computes metrics.
        
        Args:
            dataloader: DataLoader for evaluation data
            metric_key_prefix: Prefix for metric names
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Initialize metrics tracking
        predictions = []
        labels = []
        questions = []
        
        # Evaluation parameters
        max_new_tokens = getattr(self.args, 'eval_max_new_tokens', DEFAULT_MAX_NEW_TOKENS)
        
        # Create progress bar
        total_samples = len(dataloader) 
        progress_bar = tqdm(
            dataloader,
            desc="Evaluating",
            total=total_samples,
            disable=not self.is_world_process_zero()
        )
        
        # Process each batch
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Generate predictions for this batch
                    batch_predictions = self._generate_batch_predictions(
                        batch, max_new_tokens
                    )
                    
                    # Extract batch information
                    batch_labels = batch.get('answers', [])
                    batch_questions = batch.get('questions', [])
                    
                    # Accumulate results
                    predictions.extend(batch_predictions)
                    labels.extend(batch_labels)
                    questions.extend(batch_questions)
                    
                    # Update progress
                    progress_bar.set_postfix({
                        'processed': f"{len(predictions)}/{total_samples * self.args.per_device_eval_batch_size}"
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to generate prediction for sample {batch_idx}: {e}")
                    # Add empty predictions to maintain alignment
                    batch_size = len(batch.get('input_ids', []))
                    predictions.extend([""] * batch_size)
                    labels.extend(batch.get('answers', [""] * batch_size))
                    questions.extend(batch.get('questions', [""] * batch_size))
        
        progress_bar.close()
        
        # Gather predictions from all processes if using distributed training
        all_predictions, all_labels, all_questions = self._gather_evaluation_results(
            predictions, labels, questions
        )
        
        # Compute metrics on main process
        if self.is_world_process_zero():
            metrics = self._compute_evaluation_metrics(
                all_predictions, all_labels, all_questions, metric_key_prefix
            )
            
            # Log sample predictions for debugging
            self._log_sample_predictions(all_predictions, all_labels, all_questions)
            
            return metrics
        else:
            return {}

    def _generate_batch_predictions(
        self, 
        batch: Dict[str, Any], 
        max_new_tokens: int
    ) -> List[str]:
        """Generate predictions for a batch of samples."""
        batch_predictions = []
        
        # Move batch to device
        device_batch = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()
        }
        
        # Handle different batch sizes
        batch_size = len(device_batch.get('input_ids', []))
        
        for i in range(batch_size):
            try:
                # Extract single sample
                sample = {
                    k: v[i:i+1] if isinstance(v, torch.Tensor) else [v[i]] if isinstance(v, list) else v
                    for k, v in device_batch.items()
                }
                
                # Generate prediction
                prediction = self._generate_single_prediction(sample, max_new_tokens)
                batch_predictions.append(prediction)
                
            except Exception as e:
                logger.warning(f"Failed to generate prediction for sample {i}: {e}")
                batch_predictions.append("")
        
        return batch_predictions

    def _generate_single_prediction(
        self, 
        sample: Dict[str, Any], 
        max_new_tokens: int
    ) -> str:
        """Generate a single prediction."""
        try:
            # Prepare inputs for generation
            pixel_values = sample.get('pixel_values')
            input_ids = sample.get('input_ids')
            attention_mask = sample.get('attention_mask')
            questions = sample.get('questions', [''])
            
            if input_ids is None or not questions:
                return ""
            
            # Get the question text
            question = questions[0] if isinstance(questions, list) else questions
            
            # Use InternVL's chat method instead of raw generate
            if hasattr(self.model.model, 'chat') and pixel_values is not None:
                # Use the chat method with proper image and text inputs
                response = self.model.model.chat(
                    tokenizer=self.tokenizer,
                    pixel_values=pixel_values,
                    question=question,
                    generation_config=dict(
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                )
                
                # Extract answer choice from response
                answer_choice = extract_answer_choice(response)
                return answer_choice
                
            else:
                # Fallback to standard generation if chat method not available
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                
                # Extract generated tokens (remove input tokens)
                input_length = input_ids.shape[1]
                generated_tokens = generated_ids[:, input_length:]
                
                # Decode the generated text
                generated_text = self.tokenizer.decode(
                    generated_tokens[0], 
                    skip_special_tokens=True
                ).strip()
                
                # Extract answer choice from generated text
                answer_choice = extract_answer_choice(generated_text)
                return answer_choice
            
        except Exception as e:
            import traceback
            logger.warning(f"Error in prediction generation: {type(e).__name__}: {str(e)}")
            logger.warning(f"Full traceback: {traceback.format_exc()}")
            return ""

    def _gather_evaluation_results(
        self, 
        predictions: List[str], 
        labels: List[str], 
        questions: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Gather evaluation results from all processes in distributed setting."""
        if not self.is_world_process_zero() and dist.is_initialized():
            # For non-main processes, just return local results
            local_rank = dist.get_rank()
            logger.info(f"Process {local_rank}: Processed {len(predictions)} samples")
            return predictions, labels, questions
        
        # Main process gathers all results
        if dist.is_initialized() and dist.get_world_size() > 1:
            world_size = dist.get_world_size()
            
            # Check for length mismatches before gathering
            if len(predictions) != len(labels) or len(predictions) != len(questions):
                logger.error(
                    f"Rank {dist.get_rank()} has mismatched lengths: "
                    f"predictions={len(predictions)}, labels={len(labels)}, questions={len(questions)}"
                )
            
            # Gather from all processes - simplified approach
            all_predictions = predictions  # Use local predictions for now
            all_labels = labels
            all_questions = questions
            
        else:
            all_predictions = predictions
            all_labels = labels  
            all_questions = questions
        
        return all_predictions, all_labels, all_questions

    def _compute_evaluation_metrics(
        self, 
        predictions: List[str], 
        labels: List[str], 
        questions: List[str],
        metric_key_prefix: str
    ) -> Dict[str, float]:
        """Compute evaluation metrics from predictions and labels."""
        if not predictions or not labels:
            return {f"{metric_key_prefix}_accuracy": 0.0}
        
        # Ensure equal lengths
        min_length = min(len(predictions), len(labels))
        predictions = predictions[:min_length]
        labels = labels[:min_length]
        
        # Compute accuracy
        correct = sum(
            1 for pred, label in zip(predictions, labels) 
            if pred.lower().strip() == label.lower().strip()
        )
        
        accuracy = correct / len(labels) if labels else 0.0
        
        # Create metrics dictionary
        metrics = {
            f"{metric_key_prefix}_accuracy": accuracy,
            f"{metric_key_prefix}_num_samples": len(labels),
            f"{metric_key_prefix}_correct": correct,
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def _log_sample_predictions(
        self, 
        predictions: List[str], 
        labels: List[str], 
        questions: List[str],
        num_samples: int = 5
    ) -> None:
        """Log sample predictions for debugging."""
        if not predictions:
            return
            
        num_to_log = min(num_samples, len(predictions))
        
        logger.info(f"\n{SAMPLE_LOG_SEPARATOR}")
        logger.info("SAMPLE PREDICTIONS")
        logger.info(f"{SAMPLE_LOG_SEPARATOR}")
        
        for i in range(num_to_log):
            question = questions[i] if i < len(questions) else "N/A"
            prediction = predictions[i] if i < len(predictions) else "N/A"
            label = labels[i] if i < len(labels) else "N/A"
            
            logger.info(f"Sample {i + 1}:")
            logger.info(f"  Question: {question[:100]}...")
            logger.info(f"  Predicted: '{prediction}'")
            logger.info(f"  Actual: '{label}'")
            logger.info(f"  Correct: {prediction.lower().strip() == label.lower().strip()}")
            logger.info("")
        
        logger.info(f"{SAMPLE_LOG_SEPARATOR}")

    @property
    def tokenizer(self):
        """Get tokenizer from model."""
        if hasattr(self.model, 'tokenizer'):
            return self.model.tokenizer
        elif hasattr(self.model, 'module') and hasattr(self.model.module, 'tokenizer'):
            return self.model.module.tokenizer
        else:
            raise AttributeError("Tokenizer not found in model") 