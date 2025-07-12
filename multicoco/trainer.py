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
import json
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Trainer
from transformers.trainer_pt_utils import find_batch_size
from transformers.trainer_utils import TrainOutput

from .answer_extraction import extract_answer_choice
from .constants import (
    DEFAULT_MAX_NEW_TOKENS,
)
from .exceptions import EvaluationError

logger = logging.getLogger(__name__)


class CoCoTrainer(Trainer):
    """
    Custom trainer for MultiCoCo models.
    
    Extends the HuggingFace Trainer to support sophisticated answer extraction
    for multiple choice questions, detailed evaluation logging, proper dtype
    handling for multimodal inputs, and epoch-based training with progress bars.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the CoCoTrainer."""
        kwargs.pop('processor', None)
        super().__init__(*args, **kwargs)
        
        self.best_val_acc = 0.0
        self.total_train_steps = 0
        
        logger.info("CoCoTrainer initialized.")

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        **kwargs,
    ) -> TrainOutput:
        """
        Custom training loop with epoch-based progress bars and evaluation.
        """
        self._setup_epoch_training()
        
        start_epoch = self._handle_checkpoint_resumption(resume_from_checkpoint)
        
        train_dataloader = self.get_train_dataloader()
        steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        total_steps = steps_per_epoch * int(self.args.num_train_epochs)
        
        self._log_training_setup(steps_per_epoch, total_steps)
        
        model = self._wrap_model(self.model_wrapped)
        self.create_optimizer_and_scheduler(num_training_steps=total_steps)
        
        for epoch in range(start_epoch, int(self.args.num_train_epochs)):
            self._train_single_epoch(model, train_dataloader, epoch, steps_per_epoch)
            gc.collect()
            torch.cuda.empty_cache()
        
        logger.info("Training completed!")
        
        return TrainOutput(
            global_step=self.total_train_steps,
            training_loss=0.0,
            metrics={}
        )

    def _log_training_setup(self, steps_per_epoch: int, total_steps: int) -> None:
        logger.info("Starting epoch-based training:")
        logger.info(f"  Steps per epoch: {steps_per_epoch}")
        logger.info(f"  Total epochs: {int(self.args.num_train_epochs)}")
        logger.info(f"  Total steps: {total_steps}")

    def _train_single_epoch(
        self, model: nn.Module, train_dataloader: DataLoader, epoch: int, steps_per_epoch: int
    ) -> None:
        epoch_start_time = time.time()
        logger.info(f"\nStarting Epoch {epoch + 1}/{int(self.args.num_train_epochs)}")
        
        self._train_one_epoch(model, train_dataloader, epoch, steps_per_epoch)
        
        eval_metrics = self.evaluate()
        checkpoint_dir = self._save_checkpoint_with_metrics(epoch, eval_metrics)
        
        epoch_time = time.time() - epoch_start_time
        self._log_epoch_summary(epoch, eval_metrics, checkpoint_dir, epoch_time)

    def _handle_checkpoint_resumption(self, resume_from_checkpoint: Optional[Union[str, bool]]) -> int:
        start_epoch = 0
        checkpoint_path = None
        if resume_from_checkpoint:
            if resume_from_checkpoint is True:
                checkpoint_path = self._get_last_epoch_checkpoint(self.args.output_dir)
            else:
                checkpoint_path = str(resume_from_checkpoint)
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
                start_epoch = self._load_epoch_checkpoint(checkpoint_path)
            else:
                logger.warning("`resume_from_checkpoint` is set but no checkpoint found. Starting from scratch.")
        return start_epoch

    def _get_last_epoch_checkpoint(self, output_dir: str) -> Optional[str]:
        if not os.path.exists(output_dir):
            return None
        epoch_dirs = [d for d in os.listdir(output_dir) if d.startswith('epoch-')]
        if not epoch_dirs:
            return None
        epoch_nums = [int(d.split('-')[1]) for d in epoch_dirs if d.split('-')[1].isdigit()]
        if not epoch_nums:
            return None
        latest_epoch = max(epoch_nums)
        return os.path.join(output_dir, f'epoch-{latest_epoch}')

    def _load_epoch_checkpoint(self, checkpoint_path: str) -> int:
        try:
            epoch_num = int(os.path.basename(checkpoint_path).split('-')[1])
            self._load_from_checkpoint(checkpoint_path)
            return epoch_num + 1
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return 0

    def _setup_epoch_training(self) -> None:
        self.state.global_step = 0
        self.state.epoch = 0
        self.state.total_flos = 0
        logger.info("Training state initialized for epoch-based training")

    def _train_one_epoch(self, model: nn.Module, train_dataloader: DataLoader, epoch: int, steps_per_epoch: int) -> None:
        model.train()
        pbar = self._create_progress_bar(epoch, train_dataloader)
        epoch_loss = 0.0
        step_count = 0
        
        for step, inputs in enumerate(pbar):
            loss = self.training_step(model, inputs)
            if loss is not None:
                epoch_loss += loss.item()
                step_count += 1
                avg_loss = epoch_loss / step_count
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                self._log_training_step(loss, step)
            
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.total_train_steps += 1
        
        pbar.close()
        self._log_epoch_training_summary(epoch, epoch_loss, step_count)

    def _create_progress_bar(self, epoch: int, train_dataloader: DataLoader) -> tqdm:
        return tqdm(
            train_dataloader, 
            desc=f"Epoch {epoch + 1}",
            total=len(train_dataloader),
            disable=not self.is_world_process_zero(),
        )

    def _log_training_step(self, loss: torch.Tensor, step: int) -> None:
        if (step % self.args.gradient_accumulation_steps == 0 and "wandb" in self.args.report_to):
            try:
                import wandb
                if wandb.run:
                    wandb.log({"train/batch_loss": loss.item(), "train/step": self.total_train_steps})
            except ImportError:
                pass

    def _log_epoch_training_summary(self, epoch: int, epoch_loss: float, step_count: int) -> None:
        if step_count > 0:
            avg_loss = epoch_loss / step_count
            logger.info(f"Epoch {epoch + 1} training complete. Average loss: {avg_loss:.4f}")
            if "wandb" in self.args.report_to:
                try:
                    import wandb
                    if wandb.run:
                        wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch + 1})
                except ImportError:
                    pass

    def _save_checkpoint_with_metrics(self, epoch: int, metrics: Dict[str, float]) -> str:
        checkpoint_dir = os.path.join(self.args.output_dir, f'epoch-{epoch}')
        self.save_model(checkpoint_dir)
        if self.is_world_process_zero():
            metrics_path = os.path.join(checkpoint_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f'Checkpoint saved with metrics: {checkpoint_dir}')
        return checkpoint_dir

    def _log_epoch_summary(self, epoch: int, eval_metrics: Dict[str, float], checkpoint_dir: str, epoch_time: float) -> None:
        summary = [
            f"\nEPOCH {epoch + 1} SUMMARY",
            f"Checkpoint: {checkpoint_dir}",
            f"Epoch time: {epoch_time:.2f}s",
        ]
        if eval_metrics:
            summary.append("Evaluation metrics:")
            summary.extend([f"  {k}: {v:.4f}" for k, v in eval_metrics.items()])
        for line in summary:
            logger.info(line)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval') -> Dict[str, float]:
        return self.perform_evaluation(eval_dataset, metric_key_prefix)

    def perform_evaluation(self, eval_dataset=None, metric_key_prefix='eval', log_per_sample=False) -> Dict[str, float]:
        eval_dataset = eval_dataset or self.eval_dataset
        if eval_dataset is None:
            raise EvaluationError('No evaluation dataset provided')
        
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        all_preds, all_labels, all_questions, all_gen_texts, all_gen_tokens, all_ext_ans = [], [], [], [], [], []
        max_new_tokens = getattr(self.args, 'eval_max_new_tokens', DEFAULT_MAX_NEW_TOKENS)
        
        progress_bar = tqdm(eval_dataloader, desc='Evaluating', total=len(eval_dataloader), disable=not self.is_world_process_zero())
        with torch.no_grad():
            for batch in progress_bar:
                preds, gen_texts, gen_tokens = self._generate_batch_predictions_with_details(batch, max_new_tokens)
                all_preds.extend(preds)
                all_labels.extend(batch.get('answers', []))
                all_questions.extend(batch.get('questions', []))
                all_gen_texts.extend(gen_texts)
                all_gen_tokens.extend(gen_tokens)
                all_ext_ans.extend(preds)
        
        progress_bar.close()

        # Gather results from all processes
        gathered = self._gather_evaluation_results(
            all_preds, all_labels, all_questions, all_gen_texts, all_gen_tokens, all_ext_ans
        )
        all_preds, all_labels, all_questions, all_gen_texts, all_gen_tokens, all_ext_ans = gathered
        
        if self.is_world_process_zero():
            metrics = self._compute_evaluation_metrics(all_preds, all_labels, metric_key_prefix)
            logger.info(f'{metric_key_prefix.upper()} METRICS: {metrics}')
            
            if log_per_sample:
                correctness = np.array(all_preds) == np.array(all_labels)
                self._log_per_sample_details(all_questions, all_labels, all_gen_texts, all_ext_ans, all_gen_tokens, correctness)
            
            if "wandb" in self.args.report_to:
                try:
                    import wandb
                    if wandb.run:
                        wandb.log(metrics)
                except ImportError:
                    pass
            return metrics
        return {}

    def _gather_evaluation_results(
        self,
        predictions: List[str], 
        labels: List[str], 
        questions: List[str],
        generated_texts: List[str],
        generated_tokens: List[List[int]],
        extracted_answers: List[str]
    ) -> Tuple[List[str], List[str], List[str], List[str], List[List[int]], List[str]]:
        """Gather evaluation results from all processes in distributed setting."""
        if dist.is_initialized() and dist.get_world_size() > 1:
            local_results = list(zip(predictions, labels, questions, generated_texts, generated_tokens, extracted_answers))
            
            gathered_results = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_results, local_results)
            
            all_results = [item for sublist in gathered_results for item in sublist]
            
            all_predictions, all_labels, all_questions, all_generated_texts, all_generated_tokens, all_extracted = zip(*all_results)
            
            return list(all_predictions), list(all_labels), list(all_questions), list(all_generated_texts), list(all_generated_tokens), list(all_extracted)

        return predictions, labels, questions, generated_texts, generated_tokens, extracted_answers

    def _log_per_sample_details(self, questions, labels, generated_texts, extracted, generated_tokens, correctness):
        eval_logger = logging.getLogger('evaluation_details')
        for i in range(len(questions)):
            details = {
                'question': questions[i],
                'ground_truth': labels[i],
                'generated_answer': generated_texts[i],
                'extracted_answer': extracted[i],
                'generated_tokens': generated_tokens[i],
                'correct': bool(correctness[i])
            }
            eval_logger.info(json.dumps(details))

    def _generate_batch_predictions_with_details(self, batch: Dict[str, Any], max_new_tokens: int) -> Tuple[List[str], List[str], List[List[int]]]:
        device_batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch_size = find_batch_size(batch)
        
        batch_predictions, batch_generated_texts, batch_generated_tokens = [], [], []

        for i in range(batch_size):
            sample = {k: v[i:i+1] if isinstance(v, torch.Tensor) else [v[i]] for k, v in device_batch.items()}
            pixel_values, input_ids = sample.get('pixel_values'), sample.get('input_ids')

            if input_ids is None:
                batch_predictions.append(""); batch_generated_texts.append(""); batch_generated_tokens.append([])
                continue

            if hasattr(self.model.model, 'chat') and pixel_values is not None:
                response = self.model.model.chat(
                    tokenizer=self.tokenizer,
                    pixel_values=pixel_values.to(dtype=next(self.model.parameters()).dtype),
                    question=sample['questions'][0],
                    generation_config={'max_new_tokens': max_new_tokens, 'do_sample': False}
                )
                batch_predictions.append(extract_answer_choice(response))
                batch_generated_texts.append(response)
                batch_generated_tokens.append([])
            else:
                generated_ids = self.model.generate(
                    pixel_values=pixel_values, input_ids=input_ids, attention_mask=sample.get('attention_mask'),
                    max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
                )
                input_length = input_ids.shape[1]
                gen_part = generated_ids[:, input_length:]
                full_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                gen_text = self.tokenizer.decode(gen_part[0], skip_special_tokens=True)
                
                batch_predictions.append(extract_answer_choice(gen_text))
                batch_generated_texts.append(full_text)
                batch_generated_tokens.append(gen_part.tolist()[0])

        return batch_predictions, batch_generated_texts, batch_generated_tokens

    def _compute_evaluation_metrics(self, predictions: List[str], labels: List[str], prefix: str) -> Dict[str, float]:
        if not predictions or not labels:
            return {f"{prefix}_accuracy": 0.0}
        
        correct = sum(1 for pred, label in zip(predictions, labels) if pred.lower().strip() == label.lower().strip())
        accuracy = correct / len(labels) if labels else 0.0
        
        return {
            f"{prefix}_accuracy": accuracy,
            f"{prefix}_num_samples": len(labels),
            f"{prefix}_correct": correct,
        }

    @property
    def tokenizer(self):
        if hasattr(self.model, 'tokenizer'):
            return self.model.tokenizer
        elif hasattr(self.model, 'module') and hasattr(self.model.module, 'tokenizer'):
            return self.model.module.tokenizer
        raise AttributeError("Tokenizer not found in model")
