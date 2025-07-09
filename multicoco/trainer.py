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
                checkpoint_path = get_last_checkpoint(self.args.output_dir)
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
            desc=f"Training Epoch {epoch + 1}",
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
                
                # Update progress bar description
                current_loss = loss.item() * self.args.gradient_accumulation_steps
                pbar.set_description(
                    f"Training Epoch {epoch + 1}/{int(self.args.num_train_epochs)}, "
                    f"Step {step_count}/{steps_per_epoch} "
                    f"(loss: {current_loss:.4f})"
                )
                
                # Log to wandb
                if hasattr(self.args, 'report_to') and 'wandb' in self.args.report_to:
                    import wandb
                    if wandb.run:
                        log_dict = {
                            "train/epoch": epoch + 1,
                            "train/step": epoch * len(train_dataloader) + step,
                            "train/loss": current_loss,
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
        
        # Add pad token ID to suppress warnings
        if self.processing_class.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = self.processing_class.pad_token_id
        
        return gen_kwargs

    def log(self, logs: Dict[str, float], **kwargs) -> None:
        """
        Override log method to update tqdm progress bar with current loss.
        
        Args:
            logs: Dictionary of metrics to log
            **kwargs: Additional arguments passed to parent log method
        """
        # Call parent log method first with all arguments
        super().log(logs, **kwargs)
        
        # Try to update progress bar with current metrics
        self._update_progress_bar_with_metrics(logs)
    
    def _update_progress_bar_with_metrics(self, logs: Dict[str, float]) -> None:
        """
        Find and update the tqdm progress bar with current metrics.
        """
        try:
            import inspect
            import sys
            
            # Look for tqdm progress bar in all frames
            for frame_info in inspect.stack():
                frame = frame_info.frame
                frame_locals = frame.f_locals
                frame_globals = frame.f_globals
                
                # Check both locals and globals for tqdm objects
                all_vars = {**frame_globals, **frame_locals}
                
                for var_name, var_value in all_vars.items():
                    # Check if this looks like a tqdm progress bar
                    if (hasattr(var_value, 'set_description') and 
                        hasattr(var_value, 'update') and
                        hasattr(var_value, 'n') and
                        hasattr(var_value, 'total') and
                        'tqdm' in str(type(var_value))):
                        
                        # Build description with current information
                        description_parts = []
                        
                        # Add epoch information
                        if hasattr(self.state, 'epoch') and self.state.epoch is not None:
                            description_parts.append(f"Epoch {self.state.epoch:.1f}")
                        
                        # Add current loss from logs
                        if 'train_loss' in logs:
                            description_parts.append(f"Loss: {logs['train_loss']:.4f}")
                        elif 'loss' in logs:
                            description_parts.append(f"Loss: {logs['loss']:.4f}")
                        
                        # Add learning rate if available
                        if 'learning_rate' in logs:
                            description_parts.append(f"LR: {logs['learning_rate']:.2e}")
                        
                        # Update progress bar description
                        if description_parts:
                            description = " | ".join(description_parts)
                            var_value.set_description(description)
                            return
                        
        except Exception as e:
            # Silently fail to avoid disrupting training
            pass

    def _apply_coconut_masking_to_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply CoCoNut masking to input batch."""
        input_ids = inputs.get('input_ids')
        labels = inputs.get('labels')
        
        if input_ids is not None and labels is not None:
            # The progressive masking logic is removed, so this function is now a no-op
            # or will need to be re-implemented if masking is re-introduced.
            # For now, we'll just return the original inputs.
            pass # No masking applied
        
        return inputs

    def extract_answer_choice(self, generated_text: str) -> str:
        """
        Extract answer choice from generated text with sophisticated pattern matching.
        
        This method handles various answer formats commonly seen in multiple choice
        questions and extracts the choice number (0, 1, 2, 3).
        
        Args:
            generated_text: Text generated by the model
            
        Returns:
            Extracted answer choice as string
            
        Raises:
            AnswerExtractionError: If extraction fails
        """
        try:
            text = generated_text.strip()
            
            # Try different extraction patterns in order of specificity
            extractors = [
                self._extract_number_colon_format,
                self._extract_leading_number,
                self._extract_answer_is_format,
                self._extract_any_digit,
                self._extract_word_mappings
            ]
            
            for extractor in extractors:
                result = extractor(text)
                if result in VALID_CHOICE_NUMBERS:
                    return result
            
            # If no valid choice found, return original for debugging
            logger.warning(f"Could not extract valid choice from: {text[:100]}")
            return text.strip()
            
        except Exception as e:
            raise AnswerExtractionError(f"Failed to extract answer from '{generated_text}': {e}")

    def _extract_number_colon_format(self, text: str) -> str:
        """Extract from "X : description" format."""
        match = re.search(r'(\d+)\s*:\s*[a-zA-Z]', text)
        return match.group(1) if match else ""

    def _extract_leading_number(self, text: str) -> str:
        """Extract number at the start of text."""
        match = re.search(r'^(\d+)(?:\s|$)', text.strip())
        return match.group(1) if match else ""

    def _extract_answer_is_format(self, text: str) -> str:
        """Extract from "The answer is X" format."""
        match = re.search(r'(?:answer is|choice is|option is)\s*(\d+)', text.lower())
        return match.group(1) if match else ""

    def _extract_any_digit(self, text: str) -> str:
        """Extract any valid digit from text."""
        matches = re.findall(r'(\d+)', text)
        for match in matches:
            if match in VALID_CHOICE_NUMBERS:
                return match
        return ""

    def _extract_word_mappings(self, text: str) -> str:
        """Extract using word-to-number mappings."""
        text_lower = text.lower()
        for word, choice in CHOICE_MAPPINGS.items():
            if word in text_lower:
                return choice
        return ""

    def compute_metrics(self, p: EvalPrediction) -> Dict[str, float]:
        """
        Compute metrics for evaluation.
        
        This is a placeholder as the evaluation_loop calculates metrics directly.
        
        Args:
            p: Evaluation predictions
            
        Returns:
            Empty dictionary (metrics computed in evaluation_loop)
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
        Supports distributed evaluation for multi-GPU setups.
        
        Args:
            dataloader: DataLoader for evaluation
            description: Description for progress bar
            prediction_loss_only: Whether to only compute loss
            ignore_keys: Keys to ignore in outputs
            metric_key_prefix: Prefix for metric keys
            
        Returns:
            SimpleNamespace with metrics and evaluation results
        """
        try:
            # Prepare model and evaluation state
            model = self._wrap_model(self.model, training=False, dataloader=dataloader)
            model.eval()
            self.callback_handler.eval_dataloader = dataloader

            # Initialize result containers
            all_predictions = []
            all_labels = []
            all_questions = []

            # Set up logging (only on main process for distributed)
            is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            log_file_path = None
            log_file = None
            
            if is_main_process:
                log_file_path = self._setup_evaluation_logging()
                log_file = open(log_file_path, 'w', encoding='utf-8')
                self._write_evaluation_header(log_file)

            try:
                # Run evaluation loop
                for step, inputs in enumerate(tqdm(dataloader, desc=description, disable=not is_main_process)):
                    # Process batch
                    batch_results = self._process_evaluation_batch(inputs, model, log_file)
                    
                    # Accumulate results
                    all_predictions.extend(batch_results['predictions'])
                    all_labels.extend(batch_results['labels'])
                    all_questions.extend(batch_results['questions'])

                # Gather results from all processes if using distributed training
                if torch.distributed.is_initialized():
                    # Synchronize all processes
                    torch.distributed.barrier()
                    
                    # Gather results from all processes
                    world_size = torch.distributed.get_world_size()
                    gathered_predictions = [None for _ in range(world_size)]
                    gathered_labels = [None for _ in range(world_size)]
                    
                    torch.distributed.all_gather_object(gathered_predictions, all_predictions)
                    torch.distributed.all_gather_object(gathered_labels, all_labels)
                    
                    if is_main_process:
                        # Flatten gathered results
                        all_predictions = [pred for sublist in gathered_predictions for pred in sublist]
                        all_labels = [label for sublist in gathered_labels for label in sublist]

                # Compute final metrics (only on main process)
                metrics = {}
                if is_main_process:
                    metrics = self._compute_final_metrics(
                        all_predictions, all_labels, metric_key_prefix
                    )
                    
                    # Write summary
                    if log_file:
                        self._write_evaluation_summary(log_file, metrics, len(all_labels))

                # Broadcast metrics to all processes
                if torch.distributed.is_initialized():
                    metrics_list = [metrics] if is_main_process else [{}]
                    torch.distributed.broadcast_object_list(metrics_list, src=0)
                    metrics = metrics_list[0]

            finally:
                if log_file:
                    log_file.close()

            # Log metrics
            if metrics:
                self.log(metrics)

            return SimpleNamespace(
                metrics=metrics,
                num_samples=len(all_labels),
                eval_preds=None
            )
            
        except Exception as e:
            raise EvaluationError(f"Evaluation loop failed: {e}")

    def _setup_evaluation_logging(self) -> str:
        """Set up logging for evaluation."""
        log_dir = getattr(self.args, 'log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Determine evaluation type
        eval_config = self.args.eval_config
        is_cot = eval_config.get('cot', False)
        is_coconut = eval_config.get('coconut', False)
        eval_type = "coconut" if is_coconut else "cot" if is_cot else "vanilla"
        
        return os.path.join(log_dir, f'evaluation_{eval_type}.log')

    def _write_evaluation_header(self, log_file) -> None:
        """Write evaluation header to log file."""
        eval_config = self.args.eval_config
        eval_type = self._get_eval_type_name(eval_config)
        
        log_file.write(f"Evaluation Results - {eval_type.upper()}\n")
        log_file.write(EVAL_LOG_SEPARATOR + "\n\n")

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
        """Process a single evaluation batch."""
        # Extract batch components
        questions = inputs.pop("questions")
        answers = inputs.pop("answers")
        pixel_values = inputs["pixel_values"].to(self.args.device)
        
        predictions = []
        
        # Generate predictions for each sample
        for i, question in enumerate(questions):
            try:
                prediction = self._generate_single_prediction(
                    question, pixel_values[i:i+1], model
                )
                predictions.append(prediction)
                
                # Log sample details
                self._log_sample_result(
                    log_file, question, answers[i], prediction, i
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
            # Prepare the text prompt
            # For simplicity, we'll use a generic prompt structure.
            # This can be customized further if needed.
            user_content = f"{IMAGE_TOKEN}\n{question}"

            # Tokenize the input text
            text_inputs = self.processing_class(
                user_content,
                return_tensors='pt',
                truncation=True,
                max_length=DEFAULT_INPUT_MAX_LENGTH
            ).to(self.args.device)
            
            # Create generation config
            generation_config = self._create_generation_config()
            
            # Ensure correct dtype for pixel values
            pixel_values = self._ensure_correct_dtype(pixel_values, model)

            # Generate response using the custom generate method
            generated_ids = model.generate(
                input_ids=text_inputs.input_ids,
                pixel_values=pixel_values,
                attention_mask=text_inputs.attention_mask,
                **generation_config
            )
            
            # Decode the generated tokens
            response = self.processing_class.decode(generated_ids[0], skip_special_tokens=True)

            return self._clean_generated_response(response)
            
        except Exception as e:
            raise GenerationError(f"Failed to generate prediction: {e}")

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
        # The progressive masking logic is removed, so this function is now a no-op
        # or will need to be re-implemented if masking is re-introduced.
        # For now, we'll just return the original response.
        return response.strip()

    def _log_sample_result(
        self, 
        log_file, 
        question: str, 
        ground_truth: str, 
        prediction: str, 
        sample_idx: int
    ) -> None:
        """Log a single sample's result to the detailed log file."""
        if log_file is None:
            return
            
        try:
            log_file.write(f"Sample {sample_idx}:\n")
            log_file.write(f"  Question: {question}\n")
            log_file.write(f"  Ground Truth Answer: {ground_truth}\n")
            log_file.write(f"  Generated Answer: {prediction}\n")
            log_file.write(f"  Extracted Answer: {self.extract_answer_choice(prediction)}\n")
            log_file.write(f"  Tokens Generated: {len(self.processing_class.tokenize(prediction))}\n")
            log_file.write(f"  Correct: {'Yes' if self.extract_answer_choice(prediction) == ground_truth.strip() else 'No'}\n")
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
        correct = 0
        total = len(labels)
        
        for pred, label in zip(predictions, labels):
            extracted_answer = self.extract_answer_choice(pred)
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
