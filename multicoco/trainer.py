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

import wandb
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
        self.wandb_run = None
        
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
        if self.args.local_rank == 0 and not self.args.debug:
            self.wandb_run = wandb.init(
                project=getattr(self.args, 'project', 'multicoco'),
                name=f"{self.args.name}-{time.strftime('%Y%m%d-%H%M%S')}",
                config=self.args,
                group=self.args.name,
                tags=["cot", "train"],
            )

        # Setup training
        self._setup_epoch_training()
        
        # Handle checkpoint resumption
        start_epoch = self._handle_checkpoint_resumption(resume_from_checkpoint)
        
        # Get training dataloader and calculate steps
        train_dataloader = self.get_train_dataloader()
        steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        num_train_epochs = int(self.args.num_train_epochs)
        total_steps = steps_per_epoch * num_train_epochs
        
        logger.info(f"Starting epoch-based training:")
        logger.info(f"  Steps per epoch: {steps_per_epoch}")
        logger.info(f"  Total epochs: {num_train_epochs}")
        logger.info(f"  Total steps: {total_steps}")
        
        # Initialize model and optimizer
        model = self._wrap_model(self.model_wrapped)
        self.create_optimizer_and_scheduler(num_training_steps=total_steps)
        
        # Training loop - epoch by epoch
        for epoch in range(start_epoch, num_train_epochs):
            epoch_start_time = time.time()
            logger.info(f"\nStarting Epoch {epoch + 1}/{num_train_epochs}")
            
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
        
        if self.wandb_run:
            self.wandb_run.finish()

        return TrainOutput(
            global_step=self.state.global_step,
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
                checkpoint_path = get_last_checkpoint(self.args.output_dir)
            else:
                checkpoint_path = resume_from_checkpoint
            
            if checkpoint_path:
                logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
                # This is a simplification. A real implementation should handle state loading.
                try:
                    start_epoch = int(os.path.basename(checkpoint_path).split('-')[-1]) + 1
                except (ValueError, IndexError):
                    logger.warning("Could not determine start epoch from checkpoint path. Starting from epoch 0.")
                    start_epoch = 0
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
        if self.args.local_rank == 0 and not self.args.debug:
            self.wandb_run = wandb.init(
                project=getattr(self.args, 'project', 'multicoco'),
                name=f"{self.args.name}-coconut-{time.strftime('%Y%m%d-%H%M%S')}",
                config=self.args,
                group=self.args.name,
                tags=["coconut", "progressive-train"],
            )

        self._setup_epoch_training()
        
        c_thought = self.args.coconut.c_thought
        max_latent_stage = self.args.coconut.max_latent_stage
        epochs_per_stage = self.args.coconut.epochs_per_stage
        reset_optimizer = self.args.coconut.reset_optimizer
        
        logger.info(f"Starting CoCoNut progressive training:")
        logger.info(f"  Max latent stage: {max_latent_stage}")
        logger.info(f"  Epochs per stage: {epochs_per_stage}")
        logger.info(f"  C-thought: {c_thought}")
        
        for stage in range(max_latent_stage + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"STAGE {stage}: Training with {stage} latent tokens")
            logger.info(f"{'='*60}")
            
            if hasattr(self.train_dataset, 'apply_progressive_curriculum'):
                self.train_dataset.apply_progressive_curriculum(
                    scheduled_stage=stage,
                    c_thought=c_thought,
                )
            
            if reset_optimizer and stage > 0:
                self.optimizer = None
                self.lr_scheduler = None
                logger.info("Reset optimizer for new stage")
            
            self._train_coconut_stage(stage, epochs_per_stage)
        
        logger.info("CoCoNut progressive training completed!")

        if self.wandb_run:
            self.wandb_run.finish()
        
        return TrainOutput(
            global_step=self.state.global_step,
            training_loss=0.0,
            metrics={}
        )

    def _train_coconut_stage(self, stage: int, epochs_per_stage: int) -> None:
        """Train a single CoCoNut stage."""
        train_dataloader = self.get_train_dataloader()
        steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        
        total_stage_steps = steps_per_epoch * epochs_per_stage

        model = self._wrap_model(self.model_wrapped)
        if self.optimizer is None:
            self.create_optimizer_and_scheduler(num_training_steps=total_stage_steps)
        
        for stage_epoch in range(epochs_per_stage):
            global_epoch = stage * epochs_per_stage + stage_epoch
            epoch_start_time = time.time()
            logger.info(f"Stage {stage}, Epoch {stage_epoch + 1}/{epochs_per_stage}")
            
            self._train_one_epoch(model, train_dataloader, global_epoch, steps_per_epoch)
            
            checkpoint_dir = self._save_epoch_checkpoint(global_epoch)
            eval_metrics = self._evaluate_after_epoch(global_epoch)
            
            epoch_time = time.time() - epoch_start_time
            self._log_coconut_epoch_summary(
                global_epoch, stage, stage_epoch, eval_metrics, checkpoint_dir, epoch_time
            )
            
            gc.collect()
            torch.cuda.empty_cache()

    def _setup_epoch_training(self) -> None:
        """Setup training state for epoch-based training."""
        self.state.max_steps = -1
        
    def _train_one_epoch(
        self, 
        model: nn.Module, 
        train_dataloader: DataLoader, 
        epoch: int, 
        steps_per_epoch: int
    ) -> None:
        """Run a single training epoch."""
        model.train()
        
        progress_bar = tqdm(
            total=steps_per_epoch,
            desc=f"Epoch {epoch + 1} Training",
            disable=not self.is_local_process_zero(),
        )
        
        for step, inputs in enumerate(train_dataloader):
            self.training_step(model, inputs)
            
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                progress_bar.update(1)
                self.state.global_step += 1
            
            if progress_bar.n >= steps_per_epoch:
                break
        
        progress_bar.close()

    def _save_epoch_checkpoint(self, epoch: int) -> str:
        """Save a model checkpoint after an epoch."""
        checkpoint_dir = os.path.join(self.args.output_dir, f"epoch-{epoch}")
        if self.is_local_process_zero():
            self.save_model(checkpoint_dir)
            if self.tokenizer:
                self.tokenizer.save_pretrained(checkpoint_dir)
        return checkpoint_dir

    def _evaluate_after_epoch(self, epoch: int) -> Dict[str, float]:
        """Run evaluation after an epoch and return metrics."""
        logger.info(f"Running evaluation for epoch {epoch + 1}...")
        
        eval_output = self.evaluate()
        
        metric_for_best_model = self.args.metric_for_best_model or "eval_accuracy"
        val_acc = eval_output.metrics.get(metric_for_best_model, 0.0)

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            logger.info(f"New best validation accuracy: {self.best_val_acc:.4f}")
            if self.is_local_process_zero():
                best_model_dir = os.path.join(self.args.output_dir, "best_model")
                self.save_model(best_model_dir)
                if self.tokenizer:
                    self.tokenizer.save_pretrained(best_model_dir)

        if self.wandb_run and self.is_local_process_zero():
            self._log_eval_predictions_to_wandb(eval_output, epoch)

        return eval_output.metrics

    def _log_epoch_summary(
        self, 
        epoch: int, 
        eval_metrics: Dict[str, float], 
        checkpoint_dir: str, 
        epoch_time: float
    ) -> None:
        """Log a summary of the epoch's results."""
        val_acc = eval_metrics.get("eval_accuracy", 0.0)
        logger.info(f"Epoch {epoch + 1} summary:")
        logger.info(f"  Validation Accuracy: {val_acc:.4f}")
        logger.info(f"  Best Validation Accuracy: {self.best_val_acc:.4f}")
        logger.info(f"  Checkpoint saved to: {checkpoint_dir}")
        logger.info(f"  Epoch time: {epoch_time:.2f}s")
        if self.wandb_run and self.is_local_process_zero():
            log_data = {
                "eval/accuracy": val_acc,
                "eval/best_accuracy": self.best_val_acc,
                "epoch": epoch,
                "epoch_time_seconds": epoch_time,
            }
            log_data.update({f"eval/{k.replace('eval_', '')}": v for k, v in eval_metrics.items()})
            self.wandb_run.log(log_data, step=self.state.global_step)

    def _log_coconut_epoch_summary(
        self, 
        epoch: int, 
        current_stage: int,
        stage_epoch: int,
        eval_metrics: Dict[str, float], 
        checkpoint_dir: str, 
        epoch_time: float
    ) -> None:
        """Log a summary of a CoCoNut epoch's results."""
        val_acc = eval_metrics.get("eval_accuracy", 0.0)
        logger.info(f"Stage {current_stage}, Epoch {stage_epoch + 1} summary:")
        logger.info(f"  Validation Accuracy: {val_acc:.4f}")
        logger.info(f"  Best Validation Accuracy: {self.best_val_acc:.4f}")
        logger.info(f"  Checkpoint saved to: {checkpoint_dir}")
        logger.info(f"  Epoch time: {epoch_time:.2f}s")
        if self.wandb_run and self.is_local_process_zero():
            log_data = {
                "eval/accuracy": val_acc,
                "eval/best_accuracy": self.best_val_acc,
                "epoch": epoch,
                "coconut/stage": current_stage,
                "coconut/stage_epoch": stage_epoch,
                "epoch_time_seconds": epoch_time,
            }
            log_data.update({f"eval/{k.replace('eval_', '')}": v for k, v in eval_metrics.items()})
            self.wandb_run.log(log_data, step=self.state.global_step)

    def _create_generation_config(self) -> Dict[str, Any]:
        """Create the generation configuration dictionary."""
        gen_args = self.args.generation
        return {
            "max_new_tokens": gen_args.max_new_tokens,
            "do_sample": gen_args.do_sample,
            "temperature": gen_args.temperature,
            "top_p": gen_args.top_p,
            "num_beams": gen_args.num_beams,
        }

    def log(self, logs: Dict[str, float], **kwargs) -> None:
        """Log training metrics."""
        super().log(logs, **kwargs)
        if self.wandb_run and self.is_local_process_zero():
            self.wandb_run.log(logs, step=self.state.global_step)

    def _update_progress_bar_with_metrics(self, logs: Dict[str, float]) -> None:
        """Update the progress bar with the latest metrics."""
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.set_postfix({k: f"{v:.4f}" for k, v in logs.items()})

    def compute_metrics(self, p: EvalPrediction) -> Dict[str, float]:
        """Compute metrics from predictions."""
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
        """
        is_main_process = self.is_local_process_zero()
        
        log_file_path = self._setup_evaluation_logging()
        
        model = self._wrap_model(self.model_wrapped)
        model.eval()
        
        all_predictions, all_labels, all_questions, all_image_paths = [], [], [], []
        eval_dataset = self.eval_dataset
        
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            if is_main_process:
                self._write_evaluation_header(log_file)
            
            for step, inputs in enumerate(tqdm(dataloader, desc=description, disable=not is_main_process)):
                batch_results = self._process_evaluation_batch(inputs, model, log_file)
                all_predictions.extend(batch_results["predictions"])
                all_labels.extend(batch_results["labels"])
                all_questions.extend(batch_results["questions"])
                
                start_index = step * self.args.eval_batch_size
                end_index = start_index + len(batch_results["predictions"])
                image_paths_batch = []
                if hasattr(eval_dataset, 'get_image_path'):
                    for i in range(start_index, end_index):
                        try:
                            image_paths_batch.append(eval_dataset.get_image_path(i))
                        except IndexError:
                            logger.warning(f"Index {i} out of range for eval_dataset of length {len(eval_dataset)}")
                            image_paths_batch.append(None)
                else:
                    image_paths_batch.extend([None] * len(batch_results["predictions"]))
                all_image_paths.extend(image_paths_batch)

        all_predictions, all_labels, all_questions, all_image_paths = self._gather_distributed_results(
            all_predictions, all_labels, all_questions, all_image_paths, is_main_process
        )
        
        metrics = {}
        if is_main_process:
            metrics = self._compute_final_metrics(all_predictions, all_labels, metric_key_prefix)
            self._write_evaluation_summary(log_file, metrics, len(all_predictions))
        
        metrics = self._broadcast_metrics(metrics)
        
        return SimpleNamespace(
            predictions=all_predictions,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=len(all_predictions),
            questions=all_questions,
            image_paths=all_image_paths
        )

    def _log_eval_predictions_to_wandb(self, eval_output: SimpleNamespace, epoch: int):
        """Log evaluation predictions to a wandb.Table."""
        num_samples_to_log = min(eval_output.num_samples, 100)
        sample_indices = random.sample(range(eval_output.num_samples), k=num_samples_to_log)

        columns = ["epoch", "image", "question", "prediction", "ground_truth", "is_correct"]
        table_data = []

        for i in sample_indices:
            question = eval_output.questions[i]
            prediction = eval_output.predictions[i]
            label = eval_output.label_ids[i]
            image_path = eval_output.image_paths[i]
            
            try:
                extracted_pred = extract_answer_choice(prediction)
                extracted_label = extract_answer_choice(label)
                is_correct = extracted_pred == extracted_label
            except AnswerExtractionError:
                is_correct = False

            image = wandb.Image(Image.open(image_path)) if image_path and os.path.exists(image_path) else "no_image"
            
            table_data.append([epoch, image, question, prediction, label, is_correct])

        self.wandb_run.log({
            f"eval_epoch_{epoch}/predictions": wandb.Table(columns=columns, data=table_data)
        }, step=self.state.global_step)

    def _setup_evaluation_logging(self) -> str:
        """Setup the evaluation log file and return its path."""
        if not self.is_local_process_zero():
            return "/dev/null"
        
        eval_log_dir = os.path.join(self.args.output_dir, "eval_logs")
        os.makedirs(eval_log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        return os.path.join(eval_log_dir, f"eval_results_{timestamp}.log")

    def _gather_distributed_results(
        self, 
        all_predictions: List[str], 
        all_labels: List[str], 
        all_questions: List[str],
        all_image_paths: List[str],
        is_main_process: bool
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Gather evaluation results from all processes in DDP."""
        if not self.args.distributed or not dist.is_initialized():
            return all_predictions, all_labels, all_questions, all_image_paths
        
        world_size = dist.get_world_size()
        gathered_predictions = [None] * world_size
        gathered_labels = [None] * world_size
        gathered_questions = [None] * world_size
        gathered_image_paths = [None] * world_size
        
        dist.all_gather_object(gathered_predictions, all_predictions)
        dist.all_gather_object(gathered_labels, all_labels)
        dist.all_gather_object(gathered_questions, all_questions)
        dist.all_gather_object(gathered_image_paths, all_image_paths)
        
        if is_main_process:
            all_predictions = [item for sublist in gathered_predictions for item in sublist]
            all_labels = [item for sublist in gathered_labels for item in sublist]
            all_questions = [item for sublist in gathered_questions for item in sublist]
            all_image_paths = [item for sublist in gathered_image_paths for item in sublist]
        
        return all_predictions, all_labels, all_questions, all_image_paths

    def _broadcast_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Broadcast metrics from the main process to all other processes."""
        if not self.args.distributed or not dist.is_initialized():
            return metrics
        
        metrics_list = [metrics]
        dist.broadcast_object_list(metrics_list, src=0)
        
        return metrics_list[0]

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """Get the evaluation dataloader."""
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        return super().get_eval_dataloader(eval_dataset)

    def _write_evaluation_header(self, log_file) -> None:
        """Write the header for the evaluation log file."""
        eval_config = self.args.eval
        eval_type = self._get_eval_type_name(eval_config)
        
        log_file.write(f"Evaluation Run: {eval_type}\n")
        log_file.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Model: {self.args.model_name_or_path}\n")
        log_file.write(EVAL_LOG_SEPARATOR + "\n\n")

    def _get_eval_type_name(self, eval_config) -> str:
        """Get a descriptive name for the evaluation type."""
        if eval_config.coconut:
            return "CoCoNut (Latent Reasoning)"
        if eval_config.cot:
            return "Chain-of-Thought (CoT)"
        return "Vanilla (Direct Answering)"

    def _process_evaluation_batch(
        self, 
        inputs: Dict[str, torch.Tensor], 
        model: nn.Module, 
        log_file
    ) -> Dict[str, List[str]]:
        """Process a single batch during evaluation."""
        inputs = self._prepare_inputs(inputs)
        
        pixel_values = inputs.get("pixel_values")
        questions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in inputs['input_ids']]
        raw_labels = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in inputs['labels']]
        
        generation_config = self._create_generation_config()
        generated_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **generation_config,
        )
        
        predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        cleaned_predictions = [self._clean_generated_response(p) for p in predictions]
        
        if self.is_local_process_zero():
            for i in range(len(cleaned_predictions)):
                self._log_sample_result(
                    log_file, questions[i], raw_labels[i], cleaned_predictions[i], i
                )
        
        return {
            "predictions": cleaned_predictions,
            "labels": raw_labels,
            "questions": questions,
        }

    def _ensure_correct_dtype(self, pixel_values: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Ensure pixel values have the correct dtype for the model."""
        if pixel_values is None:
            return None
        dtype = next(model.parameters()).dtype
        return pixel_values.to(dtype)

    def _clean_generated_response(self, response: str) -> str:
        """Clean the generated response text."""
        return response.split("ASSISTANT:")[-1].strip()

    def _log_sample_result(
        self, 
        log_file, 
        question: str, 
        ground_truth: str, 
        prediction: str, 
        sample_idx: int,
        is_cot: bool = False
    ) -> None:
        """Log a single sample's result to the log file."""
        log_file.write(f"Sample {sample_idx + 1}:\n")
        log_file.write(f"  Question: {question}\n")
        log_file.write(f"  Ground Truth: {ground_truth}\n")
        log_file.write(f"  Prediction: {prediction}\n")
        log_file.write(SAMPLE_LOG_SEPARATOR + "\n")

    def _compute_final_metrics(
        self, 
        predictions: List[str], 
        labels: List[str], 
        metric_key_prefix: str
    ) -> Dict[str, float]:
        """Compute final evaluation metrics."""
        correct = 0
        total = len(predictions)
        
        for pred, label in zip(predictions, labels):
            try:
                extracted_pred = extract_answer_choice(pred)
                extracted_label = extract_answer_choice(label)
                if extracted_pred == extracted_label:
                    correct += 1
            except AnswerExtractionError as e:
                logger.warning(f"Could not extract answer for pred='{pred}' or label='{label}': {e}")
        
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        
        return {f"{metric_key_prefix}_accuracy": accuracy}

    def _write_evaluation_summary(
        self, 
        log_file, 
        metrics: Dict[str, float], 
        num_samples: int
    ) -> None:
        """Write the evaluation summary to the log file."""
        log_file.write("\n" + EVAL_LOG_SEPARATOR + "\n")
        log_file.write("Evaluation Summary:\n")
        log_file.write(f"  Total Samples: {num_samples}\n")
        for key, value in metrics.items():
            log_file.write(f"  {key.replace(f'{metric_key_prefix}_', '')}: {value:.4f}\n")
        log_file.write(EVAL_LOG_SEPARATOR + "\n")

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform a prediction step on the model.
        """
        if prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys
            )
        
        return (None, None, inputs.get("labels")) 