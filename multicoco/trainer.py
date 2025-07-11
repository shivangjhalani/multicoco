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
from copy import copy
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import wandb
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Trainer

# Global wandb_run variable (set by MultiCoCoRunner)
wandb_run = None
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
        
        logger.info(f"Starting epoch-based training:")
        logger.info(f"  Steps per epoch: {steps_per_epoch}")
        logger.info(f"  Total epochs: {int(self.args.num_train_epochs)}")
        logger.info(f"  Total steps: {total_steps}")
        
        # Initialize model and optimizer
        model = self._wrap_model(self.model_wrapped)
        self.create_optimizer_and_scheduler(num_training_steps=total_steps)
        
        # Training loop - epoch by epoch
        for epoch in range(start_epoch, int(self.args.num_train_epochs)):
            epoch_start_time = time.time()
            logger.info(f"\nStarting Epoch {epoch + 1}/{int(self.args.num_train_epochs)}")
            
            # Run training for this epoch
            self._train_one_epoch(model, train_dataloader, epoch, steps_per_epoch)
            
            # Save checkpoint and evaluate after epoch
            checkpoint_dir = self._save_epoch_checkpoint(epoch)
            eval_metrics = self._evaluate_after_epoch(epoch)
            
            # Log epoch summary and cleanup
            epoch_time = time.time() - epoch_start_time
            self._log_epoch_summary(epoch, eval_metrics, checkpoint_dir, epoch_time)
            
            gc.collect()
            torch.cuda.empty_cache()
        
        logger.info("Training completed!")
        
        return TrainOutput(
            global_step=self.total_train_steps,
            training_loss=0.0,
            metrics={}
        )

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
        c_thought = getattr(self.args, 'c_thought', 1)
        max_latent_stage = getattr(self.args, 'max_latent_stage', 3)
        epochs_per_stage = getattr(self.args, 'epochs_per_stage', 5)
        reset_optimizer = getattr(self.args, 'reset_optimizer', True)
        
        logger.info(f"Starting CoCoNut progressive training:")
        logger.info(f"  Max latent stage: {max_latent_stage}")
        logger.info(f"  Epochs per stage: {epochs_per_stage}")
        logger.info(f"  C-thought: {c_thought}")
        
        # Training loop across stages
        for stage in range(max_latent_stage + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"STAGE {stage}: Training with {stage} latent tokens")
            logger.info(f"{'='*60}")
            
            # Log coconut stage info to wandb like coconut does
            if wandb_run is not None and self.is_world_process_zero():
                wandb_run.log({
                    "coconut/stage": stage,
                    "coconut/latent_tokens": stage * c_thought,
                    "coconut/max_latent_stage": max_latent_stage,
                    "coconut/c_thought": c_thought
                })
            
            # Apply curriculum to dataset
            if hasattr(self.train_dataset, 'apply_progressive_curriculum'):
                self.train_dataset.apply_progressive_curriculum(
                    scheduled_stage=stage,
                    c_thought=c_thought,
                    max_latent_stage=max_latent_stage,
                    uniform_prob=getattr(self.args, 'uniform_prob', 0.0),
                    pad_latent_to_max=getattr(self.args, 'pad_latent_to_max', False)
                )
                
                # Log dataset curriculum info to wandb like coconut does
                if wandb_run is not None and self.is_world_process_zero():
                    dataset_size = len(self.train_dataset) if hasattr(self.train_dataset, '__len__') else 0
                    wandb_run.log({
                        "coconut/dataset_stage": stage,
                        "coconut/dataset_size": dataset_size,
                        "coconut/uniform_prob": getattr(self.args, 'uniform_prob', 0.0)
                    })
            
            # Reset optimizer if requested
            if reset_optimizer and stage > 0:
                self.optimizer = None
                self.lr_scheduler = None
                logger.info("Reset optimizer for new stage")
            
            # Train for this stage
            self._train_coconut_stage(stage, epochs_per_stage)
        
        logger.info("CoCoNut progressive training completed!")
        
        return TrainOutput(
            global_step=self.total_train_steps,
            training_loss=0.0,
            metrics={}
        )

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
            epoch_start_time = time.time()
            logger.info(f"Stage {stage}, Epoch {stage_epoch + 1}/{epochs_per_stage}")
            
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
        pbar = tqdm(
            train_dataloader, 
            desc=f"Epoch {epoch + 1}",
            total=len(train_dataloader),
            disable=not self.is_world_process_zero()
        )
        
        epoch_loss = 0.0
        step_count = 0
        
        for step, inputs in enumerate(pbar):
            # Perform training step
            loss = self.training_step(model, inputs)
            
            if loss is not None:
                epoch_loss += loss.item()
                step_count += 1
            
            # Update progress bar
            if step_count > 0:
                avg_loss = epoch_loss / step_count
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Update global step counter
            if step % self.args.gradient_accumulation_steps == 0:
                self.total_train_steps += 1
        
        pbar.close()
        
        # Log epoch training summary
        if step_count > 0:
            avg_loss = epoch_loss / step_count
            logger.info(f"Epoch {epoch + 1} training complete. Average loss: {avg_loss:.4f}")
            
            # Log to wandb like coconut does
            if wandb_run is not None and self.is_world_process_zero():
                wandb_run.log({
                    "train/loss": avg_loss,
                    "train/epoch": epoch + 1,
                    "train/step": self.total_train_steps
                })

    def _save_epoch_checkpoint(self, epoch: int) -> str:
        """Save checkpoint after epoch completion."""
        checkpoint_dir = os.path.join(self.args.output_dir, f'epoch-{epoch}')
        
        # Save the checkpoint
        self.save_model(checkpoint_dir)
        
        # Save trainer state
        if self.is_world_process_zero():
            state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
            self.state.save_to_json(state_path)
            
            # Log checkpoint as wandb artifact like coconut does
            if wandb_run is not None:
                try:
                    artifact_name = f"checkpoint-epoch-{epoch}"
                    artifact = wandb.Artifact(artifact_name, type="model")
                    artifact.add_dir(checkpoint_dir)
                    wandb_run.log_artifact(artifact)
                    logger.info(f"Checkpoint logged to wandb as artifact: {artifact_name}")
                except Exception as e:
                    logger.warning(f"Failed to log checkpoint artifact: {e}")
        
        logger.info(f"Checkpoint saved to: {checkpoint_dir}")
        return checkpoint_dir

    def _evaluate_after_epoch(self, epoch: int) -> Dict[str, float]:
        """Run evaluation after epoch completion."""
        if self.eval_dataset is None:
            logger.warning("No evaluation dataset provided, skipping evaluation")
            return {}
        
        try:
            # Run evaluation
            eval_output = self.evaluate()
            
            # Extract metrics
            metrics = eval_output.metrics if hasattr(eval_output, 'metrics') else eval_output
            
            # Update best validation accuracy
            if 'eval_accuracy' in metrics:
                current_acc = metrics['eval_accuracy']
                if current_acc > self.best_val_acc:
                    self.best_val_acc = current_acc
                    logger.info(f"New best validation accuracy: {current_acc:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed after epoch {epoch}: {e}")
            return {}

    def _log_epoch_summary(
        self, 
        epoch: int, 
        eval_metrics: Dict[str, float], 
        checkpoint_dir: str, 
        epoch_time: float
    ) -> None:
        """Log summary after epoch completion."""
        logger.info(f"\nEpoch {epoch + 1} Summary:")
        logger.info(f"  Time: {epoch_time:.2f}s")
        logger.info(f"  Checkpoint: {checkpoint_dir}")
        
        if eval_metrics:
            logger.info("  Evaluation metrics:")
            for key, value in eval_metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"    {key}: {value:.4f}")

    def _log_coconut_epoch_summary(
        self, 
        epoch: int, 
        current_stage: int,
        stage_epoch: int,
        eval_metrics: Dict[str, float], 
        checkpoint_dir: str, 
        epoch_time: float
    ) -> None:
        """Log summary after CoCoNut epoch completion."""
        logger.info(f"\nStage {current_stage}, Epoch {stage_epoch + 1} Summary:")
        logger.info(f"  Time: {epoch_time:.2f}s")
        logger.info(f"  Checkpoint: {checkpoint_dir}")
        
        if eval_metrics:
            logger.info("  Evaluation metrics:")
            for key, value in eval_metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"    {key}: {value:.4f}")
        
        # Log coconut stage progress to wandb like coconut does
        if wandb_run is not None and self.is_world_process_zero():
            wandb_log_dict = {
                "coconut/stage_epoch": stage_epoch + 1,
                "coconut/current_stage": current_stage,
                "coconut/epoch_time": epoch_time
            }
            
            # Add evaluation metrics
            if eval_metrics:
                for key, value in eval_metrics.items():
                    if isinstance(value, (int, float)):
                        wandb_log_dict[f"coconut/{key}"] = value
            
            wandb_run.log(wandb_log_dict)

    def _create_generation_config(self) -> Dict[str, Any]:
        """Create generation configuration from training arguments."""
        # Get generation kwargs from training arguments
        generation_kwargs = getattr(self.args, 'generation_kwargs', {})
        
        # Get tokenizer/processing_class with deprecation handling
        tokenizer = getattr(self, 'processing_class', None) or getattr(self, 'tokenizer', None)
        
        # Set defaults if not provided
        config = {
            'max_new_tokens': generation_kwargs.get('max_new_tokens', DEFAULT_MAX_NEW_TOKENS),
            'do_sample': generation_kwargs.get('do_sample', True),
            'temperature': generation_kwargs.get('temperature', 0.7),
            'top_p': generation_kwargs.get('top_p', 0.9),
            'top_k': generation_kwargs.get('top_k', 50),
            'num_beams': generation_kwargs.get('num_beams', 1),
            'pad_token_id': tokenizer.pad_token_id if tokenizer else None,
            'eos_token_id': tokenizer.eos_token_id if tokenizer else None,
        }
        
        return config

    def log(self, logs: Dict[str, float], **kwargs) -> None:
        """Override log method with custom wandb logging like coconut."""
        super().log(logs, **kwargs)
        self._update_progress_bar_with_metrics(logs)
        
        # Manual wandb logging like coconut does
        if wandb_run is not None and self.is_world_process_zero():
            # Add step information for proper wandb tracking
            step = kwargs.get('step', self.state.global_step)
            log_dict = {**logs}
            
            # Add epoch information if available
            if hasattr(self.state, 'epoch') and self.state.epoch is not None:
                log_dict['train/epoch'] = self.state.epoch
                
            wandb_run.log(log_dict)

    def _update_progress_bar_with_metrics(self, logs: Dict[str, float]) -> None:
        """Update progress bar with training metrics."""
        # This would update any active progress bars with metrics
        # Implementation depends on specific progress bar framework used
        pass

    def _apply_coconut_masking_to_inputs(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply CoCoNut masking to training inputs if needed."""
        # Apply any CoCoNut-specific input masking
        # This is a placeholder for CoCoNut-specific preprocessing
        return inputs

    def compute_metrics(self, p: EvalPrediction) -> Dict[str, float]:
        """
        Compute metrics for evaluation.
        
        This is a placeholder as the evaluation_loop calculates metrics directly.
        """
        return {}

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
        
        Supports distributed evaluation for multi-GPU setups and evaluation
        accumulation for memory efficiency.
        """
        try:
            # Prepare model and evaluation state
            model = self._wrap_model(self.model, training=False, dataloader=dataloader)
            model.eval()
            self.callback_handler.eval_dataloader = dataloader

            # Initialize result containers
            all_predictions, all_labels, all_questions = [], [], []

            # Set up logging (only on main process for distributed)
            is_main_process = self.is_world_process_zero()
            log_file = self._setup_evaluation_file(is_main_process)

            try:
                # Run evaluation with accumulation
                eval_accumulation_steps = getattr(self.args, 'eval_accumulation_steps', 1)
                accumulated_batches = []
                
                for step, inputs in enumerate(tqdm(dataloader, desc=description, disable=not is_main_process)):
                    accumulated_batches.append(inputs)
                    
                    # Process when accumulation complete or at end
                    if len(accumulated_batches) == eval_accumulation_steps or step == len(dataloader) - 1:
                        for batch_inputs in accumulated_batches:
                            batch_results = self._process_evaluation_batch(batch_inputs, model, log_file)
                            
                            all_predictions.extend(batch_results['predictions'])
                            all_labels.extend(batch_results['labels'])
                            all_questions.extend(batch_results['questions'])
                        
                        accumulated_batches = []

                # Handle distributed evaluation
                if torch.distributed.is_initialized():
                    all_predictions, all_labels, all_questions = self._gather_distributed_results(
                        all_predictions, all_labels, all_questions, is_main_process
                    )

                # Compute final metrics (only on main process)
                metrics = {}
                if is_main_process:
                    metrics = self._compute_final_metrics(all_predictions, all_labels, metric_key_prefix)
                    
                    # Log to wandb like coconut does
                    self._log_wandb_evaluation_results(
                        metrics, all_predictions, all_labels, all_questions
                    )
                    
                    if log_file:
                        self._write_evaluation_summary(log_file, metrics, len(all_labels))

            finally:
                if log_file:
                    log_file.close()

            # Broadcast metrics to all processes in distributed setting
            if torch.distributed.is_initialized():
                metrics = self._broadcast_metrics(metrics)

            return SimpleNamespace(
                predictions=all_predictions,
                label_ids=all_labels,
                metrics=metrics,
                num_samples=len(all_labels) if is_main_process else 0
            )

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise EvaluationError(f"Evaluation loop failed: {e}")

    def _setup_evaluation_file(self, is_main_process: bool):
        """Setup evaluation log file."""
        if not is_main_process:
            return None
            
        log_file_path = self._setup_evaluation_logging()
        log_file = open(log_file_path, 'w', encoding='utf-8')
        self._write_evaluation_header(log_file)
        return log_file

    def _gather_distributed_results(
        self, 
        all_predictions: List[str], 
        all_labels: List[str], 
        all_questions: List[str],
        is_main_process: bool
    ) -> Tuple[List[str], List[str], List[str]]:
        """Gather results from all processes in distributed evaluation with robustness checks."""
        if is_main_process:
            logger.info(f"Process 0: Processed {len(all_labels)} samples")
        
        torch.distributed.barrier()
        
        # Gather results from all processes
        world_size = torch.distributed.get_world_size()
        gathered_predictions = [None for _ in range(world_size)]
        gathered_labels = [None for _ in range(world_size)]
        gathered_questions = [None for _ in range(world_size)]
        
        torch.distributed.all_gather_object(gathered_predictions, all_predictions)
        torch.distributed.all_gather_object(gathered_labels, all_labels)
        torch.distributed.all_gather_object(gathered_questions, all_questions)
        
        # Only main process combines results
        if is_main_process:
            final_predictions, final_labels, final_questions = [], [], []
            
            # Add robustness checks for length consistency
            rank_lengths = []
            for rank in range(world_size):
                pred_len = len(gathered_predictions[rank]) if gathered_predictions[rank] is not None else 0
                label_len = len(gathered_labels[rank]) if gathered_labels[rank] is not None else 0
                question_len = len(gathered_questions[rank]) if gathered_questions[rank] is not None else 0
                
                # Check length consistency within rank
                if not (pred_len == label_len == question_len):
                    logger.error(f"Rank {rank} has mismatched lengths: "
                               f"predictions={pred_len}, labels={label_len}, questions={question_len}")
                    raise EvaluationError(f"Mismatched data lengths on rank {rank}")
                
                rank_lengths.append(pred_len)
                logger.info(f"Rank {rank}: {pred_len} samples")
            
            # Check for potential OOM issues (extreme length differences)
            if len(set(rank_lengths)) > 1:
                min_len, max_len = min(rank_lengths), max(rank_lengths)
                if min_len == 0:
                    logger.warning(f"Rank with 0 samples detected (possible OOM): {rank_lengths}")
                elif max_len / min_len > 10:  # More than 10x difference suggests issues
                    logger.warning(f"Large length discrepancy across ranks: {rank_lengths}")
            
            # Combine results from all ranks
            for rank in range(world_size):
                if (gathered_predictions[rank] is not None and 
                    gathered_labels[rank] is not None and 
                    gathered_questions[rank] is not None):
                    final_predictions.extend(gathered_predictions[rank])
                    final_labels.extend(gathered_labels[rank])
                    final_questions.extend(gathered_questions[rank])
                else:
                    logger.warning(f"Rank {rank} returned None results (possible failure)")
            
            # Final sanity check
            total_samples = len(final_predictions)
            assert len(final_labels) == total_samples, f"Final label count mismatch: {len(final_labels)} vs {total_samples}"
            assert len(final_questions) == total_samples, f"Final question count mismatch: {len(final_questions)} vs {total_samples}"
            
            logger.info(f"Successfully gathered {total_samples} total samples from {world_size} ranks")
            
            return final_predictions, final_labels, final_questions
        
        return all_predictions, all_labels, all_questions

    def _broadcast_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Broadcast metrics from main process to all processes."""
        if torch.distributed.is_initialized():
            # Convert metrics to list for broadcasting
            if self.is_world_process_zero():
                metrics_list = [metrics]
            else:
                metrics_list = [None]
            
            torch.distributed.broadcast_object_list(metrics_list, src=0)
            return metrics_list[0]
        
        return metrics
    
    def _log_wandb_evaluation_results(
        self, 
        metrics: Dict[str, float], 
        predictions: List[str], 
        labels: List[str], 
        questions: List[str],
        max_samples: int = 50
    ) -> None:
        """Log evaluation results to wandb following coconut pattern."""
        if wandb_run is None:
            return
            
        # Log main metrics using wandb_run like coconut does
        wandb_run.log(metrics)
        
        # Create sample table for detailed analysis like coconut does
        sample_table = wandb.Table(columns=["Question", "Ground Truth", "Prediction", "Correct"])
        
        # Limit samples to avoid memory issues
        num_samples = min(max_samples, len(predictions))
        
        for i in range(num_samples):
            if i < len(questions) and i < len(labels) and i < len(predictions):
                correct = predictions[i].strip() == labels[i].strip()
                sample_table.add_data(
                    questions[i][:200] + "..." if len(questions[i]) > 200 else questions[i],  # Truncate long questions
                    labels[i], 
                    predictions[i], 
                    correct
                )
        
        # Copy table to avoid wandb bug (like coconut does)
        wandb_run.log({"eval/samples": copy(sample_table)})
        
        # Log accuracy breakdown like coconut's eval/acc format
        correct_count = sum(1 for p, l in zip(predictions, labels) if p.strip() == l.strip())
        wandb_run.log({
            "eval/acc": correct_count / len(predictions) if predictions else 0.0,
            "eval/total": len(predictions),
            "eval/correct": correct_count
        })

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """Get evaluation dataloader with proper distributed setup."""
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("No evaluation dataset provided")
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        # Use parent class method to create dataloader with proper distributed setup
        return super().get_eval_dataloader(eval_dataset)

    def _setup_evaluation_logging(self) -> str:
        """Setup evaluation logging and return log file path."""
        # Create logs directory
        log_dir = os.path.join(self.args.output_dir, 'eval_logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(log_dir, f'evaluation_{timestamp}.log')
        
        return log_file_path

    def _write_evaluation_header(self, log_file) -> None:
        """Write evaluation header to log file."""
        eval_config = getattr(self.args, 'eval_config', {})
        eval_type = self._get_eval_type_name(eval_config)
        
        log_file.write(f"Evaluation Type: {eval_type}\n")
        log_file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"{EVAL_LOG_SEPARATOR}\n")

    def _get_eval_type_name(self, eval_config: Dict[str, bool]) -> str:
        """Get evaluation type name from config."""
        if eval_config.get('coconut', False):
            return "coconut"
        elif eval_config.get('cot', False):
            return "cot"
        else:
            return "vanilla"

    def _process_evaluation_batch(
        self, 
        inputs: Dict[str, torch.Tensor], 
        model: nn.Module, 
        log_file
    ) -> Dict[str, List[str]]:
        """Process a single evaluation batch and return results."""
        # Move inputs to device
        inputs = self._prepare_inputs(inputs)
        
        # Extract batch information
        questions = inputs.get('questions', [])
        answers = inputs.get('answers', [])
        pixel_values = inputs.get('pixel_values')
        
        # Generate predictions for each sample in batch
        predictions = []
        for i in range(len(questions)):
            question = questions[i]
            sample_pixel_values = pixel_values[i:i+1] if pixel_values is not None else None
            
            try:
                prediction = self._generate_single_prediction(question, sample_pixel_values, model)
                predictions.append(prediction)
                
                # Log sample result if log file available
                if log_file and i < len(answers):
                    self._log_sample_result(
                        log_file, question, answers[i], prediction, len(predictions)
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
        """Generate prediction for a single question."""
        try:
            # Get evaluation configuration
            eval_config = getattr(self.args, 'eval_config', {})
            generation_config = self._create_generation_config()
            
            # Access the underlying InternVL model for chat interface
            if hasattr(model, 'model'):
                internvl_model = model.model
            else:
                internvl_model = model
            
            # Ensure pixel values are properly formatted
            if pixel_values is not None:
                pixel_values = self._ensure_correct_dtype(pixel_values, model)
                device = next(model.parameters()).device
                pixel_values = pixel_values.to(device)
                
                # Ensure proper shape
                if pixel_values.dim() == 3:
                    pixel_values = pixel_values.unsqueeze(0)
            
            # Get tokenizer with deprecation handling
            tokenizer = getattr(self, 'processing_class', None) or getattr(self, 'tokenizer', None)
            
            # Handle CoCoNut evaluation with latent tokens
            if eval_config.get('coconut', False):
                response = self._generate_coconut_prediction(
                    question, pixel_values, model, tokenizer, generation_config, device
                )
            else:
                # Use InternVL chat interface for standard evaluation
                with torch.no_grad():
                    response = internvl_model.chat(
                        tokenizer=tokenizer,
                        pixel_values=pixel_values,
                        question=question,
                        generation_config=generation_config,
                        history=None,
                        return_history=False
                    )
            
            # Clean and extract answer
            response = self._clean_generated_response(response)
            is_cot = eval_config.get('cot', False)
            extracted_answer = extract_answer_choice(response, is_cot)
            
            return extracted_answer
            
        except Exception as e:
            logger.error(f"Failed to generate prediction: {e}", exc_info=True)
            raise GenerationError(f"Prediction generation failed: {e}")

    def _generate_coconut_prediction(
        self,
        question: str,
        pixel_values: torch.Tensor,
        model: nn.Module,
        tokenizer,
        generation_config: Dict[str, Any],
        device: torch.device
    ) -> str:
        """Generate prediction using CoCoNut latent reasoning."""
        # Get number of latent tokens from eval config
        eval_config = getattr(self.args, 'eval_config', {})
        eval_latent_tokens = eval_config.get('eval_latent_tokens', 3)
        
        # Create latent reasoning prompt
        latent_tokens = " ".join(["<|latent|>"] * eval_latent_tokens)
        latent_prompt = f"<|start_latent|> {latent_tokens} <|end_latent|>"
        
        prompt = f"<|im_start|>user\n<image>\n{question}<|im_end|><|im_start|>assistant\n{latent_prompt}"
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=DEFAULT_INPUT_MAX_LENGTH
        )
        
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Generate with latent wrapper
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )
        
        # Decode only the generated part
        if generated_ids.shape[1] > input_ids.shape[1]:
            new_tokens = generated_ids[:, input_ids.shape[1]:]
            response = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
        else:
            response = ""
        
        return response

    def _ensure_correct_dtype(self, pixel_values: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Ensure pixel values have the correct dtype for the model."""
        model_dtype = next(model.parameters()).dtype
        if pixel_values.dtype != model_dtype:
            return pixel_values.to(dtype=model_dtype)
        return pixel_values

    def _clean_generated_response(self, response: str) -> str:
        """Clean the generated response text."""
        # Remove extra whitespace and newlines
        response = response.strip()
        
        # Remove common artifacts
        response = response.replace('<|im_end|>', '').replace('<|im_start|>', '')
        
        return response

    def _log_sample_result(
        self, 
        log_file, 
        question: str, 
        ground_truth: str, 
        prediction: str, 
        sample_idx: int,
        is_cot: bool = False
    ) -> None:
        """Log individual sample evaluation result."""
        log_file.write(f"\n{SAMPLE_LOG_SEPARATOR}\n")
        log_file.write(f"Sample {sample_idx}\n")
        log_file.write(f"Question: {question}\n")
        log_file.write(f"Ground Truth: {ground_truth}\n")
        log_file.write(f"Prediction: {prediction}\n")
        
        # Check if prediction is correct
        is_correct = prediction.strip() == ground_truth.strip()
        log_file.write(f"Correct: {is_correct}\n")

    def _compute_final_metrics(
        self, 
        predictions: List[str], 
        labels: List[str], 
        metric_key_prefix: str
    ) -> Dict[str, float]:
        """Compute final evaluation metrics."""
        if not predictions or not labels or len(predictions) != len(labels):
            logger.warning("Empty or mismatched predictions/labels")
            return {}
        
        # Calculate accuracy
        correct = sum(1 for pred, label in zip(predictions, labels) if pred.strip() == label.strip())
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0.0
        
        # Create metrics dictionary
        metrics = {
            f'{metric_key_prefix}_accuracy': accuracy,
            f'{metric_key_prefix}_total_samples': total,
            f'{metric_key_prefix}_correct_samples': correct,
        }
        
        logger.info(f"Evaluation completed: {correct}/{total} correct ({accuracy:.4f})")
        
        return metrics

    def _write_evaluation_summary(
        self, 
        log_file, 
        metrics: Dict[str, float], 
        num_samples: int
    ) -> None:
        """Write evaluation summary to log file."""
        log_file.write(f"\n{EVAL_LOG_SEPARATOR}\n")
        log_file.write("EVALUATION SUMMARY\n")
        log_file.write(f"{EVAL_LOG_SEPARATOR}\n")
        
        accuracy = metrics.get('eval_accuracy', 0.0)
        log_file.write(f"Total Samples: {num_samples}\n")
        log_file.write(f"Accuracy: {accuracy:.4f}\n")
        
        log_file.write(f"{EVAL_LOG_SEPARATOR}\n")

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Perform a prediction step (used by HuggingFace evaluation)."""
        # This method is required by the HuggingFace Trainer interface
        # but we use our custom evaluation_loop instead
        model.eval()
        
        with torch.no_grad():
            inputs = self._prepare_inputs(inputs)
            outputs = model(**inputs)
            
            loss = outputs.loss if hasattr(outputs, 'loss') else None
            logits = outputs.logits if hasattr(outputs, 'logits') else None
            labels = inputs.get('labels')
            
        return (loss, logits, labels)
