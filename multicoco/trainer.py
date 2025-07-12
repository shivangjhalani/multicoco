import gc
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Trainer
from transformers.trainer_pt_utils import find_batch_size
from transformers.trainer_utils import TrainOutput

from .answer_extraction import extract_answer_choice
from .constants import DEFAULT_MAX_NEW_TOKENS
from .exceptions import EvaluationError

logger = logging.getLogger(__name__)


class CoCoTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_train_steps = 0
        logger.info('CoCoTrainer initialized.')

    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None, **kwargs) -> TrainOutput:
        # Check if we're in CoCoNut mode with multi-stage training
        is_coconut_mode = hasattr(self.args, 'epochs_per_stage') and hasattr(self.args, 'max_latent_stage')
        
        if is_coconut_mode:
            return self._train_with_coconut_stages(resume_from_checkpoint, **kwargs)
        else:
            return self._train_standard(resume_from_checkpoint, **kwargs)
    
    def _train_standard(self, resume_from_checkpoint: Optional[Union[str, bool]] = None, **kwargs) -> TrainOutput:
        """Standard training without CoCoNut stage transitions."""
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
        logger.info('Training completed!')
        return TrainOutput(global_step=self.total_train_steps, training_loss=0.0, metrics={})
    
    def _train_with_coconut_stages(self, resume_from_checkpoint: Optional[Union[str, bool]] = None, **kwargs) -> TrainOutput:
        """Training with CoCoNut multi-stage curriculum."""
        logger.info('Starting CoCoNut multi-stage training with stage transitions')
        
        self._setup_epoch_training()
        start_epoch = self._handle_checkpoint_resumption(resume_from_checkpoint)
        
        # Initialize stage tracking
        self._last_stage = -1  # Track last processed stage
        
        # Initial training setup
        train_dataloader = self.get_train_dataloader()
        steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        total_steps = steps_per_epoch * int(self.args.num_train_epochs)
        self._log_training_setup(steps_per_epoch, total_steps)
        model = self._wrap_model(self.model_wrapped)
        self.create_optimizer_and_scheduler(num_training_steps=total_steps)
        
        # Main training loop with stage transitions
        for epoch in range(start_epoch, int(self.args.num_train_epochs)):
            # Calculate current stage based on epoch and epochs_per_stage
            current_stage = min(epoch // self.args.epochs_per_stage, self.args.max_latent_stage)
            
            # Handle stage transitions
            if current_stage != self._last_stage:
                self._update_for_stage(current_stage)
                self._last_stage = current_stage
                # Refresh dataloader after dataset update
                train_dataloader = self.get_train_dataloader()
            
            # Log stage progress
            stage_epoch = epoch % self.args.epochs_per_stage
            stage_progress = (stage_epoch + 1) / self.args.epochs_per_stage
            logger.info(f'Epoch {epoch + 1}/{int(self.args.num_train_epochs)} - '
                       f'CoCoNut Stage {current_stage}/{self.args.max_latent_stage} '
                       f'(Stage Epoch {stage_epoch + 1}/{self.args.epochs_per_stage})')
            
            # Log stage metrics to wandb if available
            self._log_coconut_stage_metrics(current_stage, stage_epoch, stage_progress)
            
            # Train the epoch
            self._train_single_epoch(model, train_dataloader, epoch, steps_per_epoch)
            gc.collect()
            torch.cuda.empty_cache()
        
        logger.info('CoCoNut multi-stage training completed!')
        return TrainOutput(global_step=self.total_train_steps, training_loss=0.0, metrics={})

    def _log_training_setup(self, steps_per_epoch: int, total_steps: int) -> None:
        logger.info('Starting epoch-based training:')
        logger.info(f'  Steps per epoch: {steps_per_epoch}')
        logger.info(f'  Total epochs: {int(self.args.num_train_epochs)}')
        logger.info(f'  Total steps: {total_steps}')

    def _train_single_epoch(self, model: torch.nn.Module, train_dataloader: DataLoader, epoch: int, steps_per_epoch: int) -> None:
        epoch_start_time = time.time()
        logger.info(f'\nStarting Epoch {epoch + 1}/{int(self.args.num_train_epochs)}')
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
                logger.info(f'Resuming training from checkpoint: {checkpoint_path}')
                start_epoch = self._load_epoch_checkpoint(checkpoint_path)
            else:
                logger.warning('`resume_from_checkpoint` is set but no checkpoint found. Starting from scratch.')
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
            logger.error(f'Failed to load checkpoint {checkpoint_path}: {e}')
            return 0

    def _setup_epoch_training(self) -> None:
        self.state.global_step = 0
        self.state.epoch = 0
        self.state.total_flos = 0
        logger.info('Training state initialized for epoch-based training')

    def _train_one_epoch(self, model: torch.nn.Module, train_dataloader: DataLoader, epoch: int, steps_per_epoch: int) -> None:
        model.train()
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}', total=len(train_dataloader), disable=not self.is_world_process_zero())
        epoch_loss = 0.0
        step_count = 0
        
        for step, inputs in enumerate(pbar):
            loss = self.training_step(model, inputs)
            if loss is not None:
                epoch_loss += loss.item()
                step_count += 1
                avg_loss = epoch_loss / step_count
                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{self.get_lr():.6f}'})
                
                # Enhanced training step logging with more metrics
                self._log_training_step(loss, step, epoch)
                
                # Log gradient norm if gradient clipping is applied
                if hasattr(self.args, 'max_grad_norm') and self.args.max_grad_norm > 0:
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        # Calculate gradient norm before clipping
                        total_norm = 0.0
                        for p in model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** (1. / 2)
                        self._last_grad_norm = total_norm
                        
                        # Apply gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                        
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.total_train_steps += 1
        pbar.close()
        if step_count > 0:
            avg_loss = epoch_loss / step_count
            logger.info(f'Epoch {epoch + 1} training complete. Average loss: {avg_loss:.4f}')
            
            # Log epoch average loss
            if 'wandb' in self.args.report_to:
                try:
                    import wandb
                    if wandb.run:
                        wandb.log({
                            'train/epoch_avg_loss': avg_loss,
                            'train/epoch': epoch + 1,
                            'train/steps_per_epoch': step_count,
                        })
                except ImportError:
                    pass

    def _log_training_step(self, loss: torch.Tensor, step: int, epoch: int = None) -> None:
        if step % self.args.gradient_accumulation_steps == 0 and 'wandb' in self.args.report_to:
            try:
                import wandb
                if wandb.run:
                    # Log comprehensive training metrics similar to coconut
                    log_dict = {
                        'train/batch_loss': loss.item(), 
                        'train/step': self.total_train_steps,
                        'train/global_step': self.state.global_step,
                        'train/learning_rate': self.get_lr(),
                    }
                    
                    # Add epoch information
                    if epoch is not None:
                        log_dict['train/epoch'] = epoch + 1
                    elif hasattr(self.state, 'epoch') and self.state.epoch is not None:
                        log_dict['train/epoch'] = self.state.epoch
                        
                    # Add gradient norm if available
                    if hasattr(self, '_last_grad_norm'):
                        log_dict['train/grad_norm'] = self._last_grad_norm
                        
                    # Add gradient accumulation info
                    log_dict['train/gradient_accumulation_steps'] = self.args.gradient_accumulation_steps
                        
                    wandb.log(log_dict)
            except ImportError:
                pass

    def _log_epoch_summary(self, epoch: int, eval_metrics: Dict[str, float], checkpoint_dir: str, epoch_time: float) -> None:
        summary = [f'\nEPOCH {epoch + 1} SUMMARY', f'Checkpoint: {checkpoint_dir}', f'Epoch time: {epoch_time:.2f}s']
        if eval_metrics:
            summary.append('Evaluation metrics:')
            summary.extend([f'  {k}: {v:.4f}' for k, v in eval_metrics.items()])
        for line in summary:
            logger.info(line)
            
        # Log epoch summary to wandb
        if 'wandb' in self.args.report_to:
            try:
                import wandb
                if wandb.run:
                    epoch_summary = {
                        'epoch/number': epoch + 1,
                        'epoch/time_seconds': epoch_time,
                        'epoch/checkpoint_dir': checkpoint_dir,
                    }
                    # Add evaluation metrics with epoch prefix
                    if eval_metrics:
                        for key, value in eval_metrics.items():
                            epoch_summary[f'epoch/{key}'] = value
                    wandb.log(epoch_summary)
            except ImportError:
                pass
                
    def _log_validation_loss(self, val_loss: float, epoch: int) -> None:
        """Log validation loss similar to coconut's eval loss logging"""
        if 'wandb' in self.args.report_to:
            try:
                import wandb
                if wandb.run:
                    wandb.log({
                        'eval/loss': val_loss,
                        'eval/epoch': epoch + 1,
                    })
                    logger.info(f"Validation loss: {val_loss:.4f}")
            except ImportError:
                pass

    def get_lr(self) -> float:
        """Get current learning rate from optimizer"""
        if self.optimizer is None:
            return 0.0
        return self.optimizer.param_groups[0]['lr']

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
        summary = [f'\nEPOCH {epoch + 1} SUMMARY', f'Checkpoint: {checkpoint_dir}', f'Epoch time: {epoch_time:.2f}s']
        if eval_metrics:
            summary.append('Evaluation metrics:')
            summary.extend([f'  {k}: {v:.4f}' for k, v in eval_metrics.items()])
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
        gathered = self._gather_evaluation_results(all_preds, all_labels, all_questions, all_gen_texts, all_gen_tokens, all_ext_ans)
        all_preds, all_labels, all_questions, all_gen_texts, all_gen_tokens, all_ext_ans = gathered
        if self.is_world_process_zero():
            metrics = self._compute_evaluation_metrics(all_preds, all_labels, metric_key_prefix)
            logger.info(f'{metric_key_prefix.upper()} METRICS: {metrics}')
            
            # Log evaluation metrics to wandb with more comprehensive metrics similar to coconut
            if 'wandb' in self.args.report_to:
                try:
                    import wandb
                    if wandb.run:
                        wandb_metrics = {}
                        
                        # Standard accuracy metrics (similar to coconut's eval/acc)
                        for key, value in metrics.items():
                            wandb_metrics[f'{metric_key_prefix}/{key}'] = value
                            
                        # Calculate additional metrics similar to coconut
                        total_samples = len(all_preds)
                        correct_predictions = sum(1 for pred, label in zip(all_preds, all_labels) if pred == label)
                        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
                        
                        # Add coconut-style metrics
                        wandb_metrics[f'{metric_key_prefix}/acc'] = accuracy
                        wandb_metrics[f'{metric_key_prefix}/total_samples'] = total_samples
                        wandb_metrics[f'{metric_key_prefix}/correct_predictions'] = correct_predictions
                        
                        # If we have reasoning text, compute CoT exact match (like coconut's eval/cot_em)
                        if all_gen_texts:
                            # Calculate exact match for generated reasoning text
                            cot_exact_matches = 0
                            for gen_text, label in zip(all_gen_texts, all_labels):
                                # Simple heuristic: check if label appears in generated text
                                if label.lower().strip() in gen_text.lower():
                                    cot_exact_matches += 1
                            cot_em_rate = cot_exact_matches / total_samples if total_samples > 0 else 0.0
                            wandb_metrics[f'{metric_key_prefix}/cot_em'] = cot_em_rate
                            
                        wandb.log(wandb_metrics)
                        logger.info(f'Logged comprehensive evaluation metrics to wandb: {wandb_metrics}')
                        
                        # Log sample generations table (similar to coconut's data_table)
                        if log_per_sample and len(all_questions) > 0:
                            self._log_evaluation_samples_to_wandb(
                                all_questions[:10], all_gen_texts[:10], 
                                all_labels[:10], all_preds[:10], metric_key_prefix
                            )
                except ImportError:
                    logger.warning("wandb not available for logging evaluation metrics")
                    
            if log_per_sample:
                correctness = np.array(all_preds) == np.array(all_labels)
                self._log_per_sample_details(all_questions, all_labels, all_gen_texts, all_ext_ans, all_gen_tokens, correctness)
            return metrics
        return {}

    def _log_evaluation_samples_to_wandb(self, questions: List[str], generated_texts: List[str], 
                                       labels: List[str], predictions: List[str], metric_prefix: str) -> None:
        """Log evaluation samples to wandb similar to coconut's text table logging"""
        try:
            import wandb
            if wandb.run:
                columns = ["Question", "Generated Text", "Ground Truth", "Prediction", "Correct"]
                data = []
                
                for i in range(len(questions)):
                    is_correct = predictions[i] == labels[i] if i < len(predictions) and i < len(labels) else False
                    data.append([
                        questions[i][:300] + "..." if len(questions[i]) > 300 else questions[i],
                        generated_texts[i][:500] + "..." if len(generated_texts[i]) > 500 else generated_texts[i],
                        labels[i] if i < len(labels) else "N/A",
                        predictions[i] if i < len(predictions) else "N/A",
                        "✓" if is_correct else "✗"
                    ])
                
                eval_table = wandb.Table(columns=columns, data=data)
                wandb.log({f"{metric_prefix}/sample_generations": eval_table})
        except Exception as e:
            logger.warning(f"Failed to log evaluation samples to wandb: {e}")

    def _gather_evaluation_results(self, predictions: List[str], labels: List[str], questions: List[str], generated_texts: List[str], generated_tokens: List[int], extracted_answers: List[str]) -> Tuple[List[str], List[str], List[str], List[str], List[int], List[str]]:
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

    def _generate_batch_predictions_with_details(self, batch: Dict[str, Any], max_new_tokens: int) -> Tuple[List[str], List[str], List[int]]:
        device_batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch_size = find_batch_size(batch)
        batch_predictions, batch_generated_texts, batch_generated_tokens = [], [], []
        for i in range(batch_size):
            sample = {k: v[i:i + 1] if isinstance(v, torch.Tensor) else [v[i]] for k, v in device_batch.items()}
            pixel_values, input_ids = sample.get('pixel_values'), sample.get('input_ids')
            if input_ids is None:
                batch_predictions.append('')
                batch_generated_texts.append('')
                batch_generated_tokens.append(0)
                continue
            if hasattr(self.model.model, 'chat') and pixel_values is not None:
                response = self.model.model.chat(tokenizer=self.tokenizer, pixel_values=pixel_values.to(dtype=next(self.model.parameters()).dtype), question=sample['questions'][0], generation_config={'max_new_tokens': max_new_tokens, 'do_sample': False})
                batch_predictions.append(extract_answer_choice(response))
                batch_generated_texts.append(response)
                # Count tokens in the generated response
                response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
                batch_generated_tokens.append(len(response_tokens))
            else:
                generated_ids = self.model.generate(pixel_values=pixel_values, input_ids=input_ids, attention_mask=sample.get('attention_mask'), max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
                input_length = input_ids.shape[1]
                gen_part = generated_ids[:, input_length:]
                full_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                gen_text = self.tokenizer.decode(gen_part[0], skip_special_tokens=True)
                batch_predictions.append(extract_answer_choice(gen_text))
                batch_generated_texts.append(full_text)
                # Store the count of generated tokens instead of the token list
                batch_generated_tokens.append(len(gen_part.tolist()[0]))
        return batch_predictions, batch_generated_texts, batch_generated_tokens

    def _compute_evaluation_metrics(self, predictions: List[str], labels: List[str], prefix: str) -> Dict[str, float]:
        if not predictions or not labels:
            return {f'{prefix}_accuracy': 0.0}
        correct = sum(1 for pred, label in zip(predictions, labels) if pred.lower().strip() == label.lower().strip())
        accuracy = correct / len(labels) if labels else 0.0
        return {f'{prefix}_accuracy': accuracy, f'{prefix}_num_samples': len(labels), f'{prefix}_correct': correct}

    @property
    def tokenizer(self):
        if hasattr(self.model, 'tokenizer'):
            return self.model.tokenizer
        elif hasattr(self.model, 'module') and hasattr(self.model.module, 'tokenizer'):
            return self.model.module.tokenizer
        raise AttributeError('Tokenizer not found in model')

    def _log_training_data_sample(self, batch: Dict, epoch: int, step: int) -> None:
        """Log training data samples to wandb similar to coconut's data table logging"""
        if not (step == 0 and epoch == 0):  # Only log on first step of first epoch
            return
            
        if 'wandb' in self.args.report_to:
            try:
                import wandb
                if wandb.run and self.is_world_process_zero():
                    
                    # Extract sample data from batch
                    input_ids = batch.get('input_ids', [])
                    labels = batch.get('labels', [])
                    
                    if hasattr(self, 'tokenizer') and len(input_ids) > 0:
                        # Create data table similar to coconut
                        columns = ["step", "sample_id", "token_id", "label_id", "token_text"]
                        data = []
                        
                        # Log first few samples
                        max_samples = min(2, len(input_ids))
                        for sample_idx in range(max_samples):
                            sample_input_ids = input_ids[sample_idx]
                            sample_labels = labels[sample_idx] if sample_idx < len(labels) else None
                            
                            # Log first 50 tokens of each sample
                            max_tokens = min(50, len(sample_input_ids))
                            for token_idx in range(max_tokens):
                                token_id = sample_input_ids[token_idx].item() if hasattr(sample_input_ids[token_idx], 'item') else sample_input_ids[token_idx]
                                label_id = sample_labels[token_idx].item() if sample_labels is not None and hasattr(sample_labels[token_idx], 'item') else -100
                                token_text = self.tokenizer.decode([token_id]) if hasattr(self, 'tokenizer') else f"token_{token_id}"
                                
                                data.append([
                                    self.total_train_steps,
                                    sample_idx,
                                    token_id,
                                    label_id,
                                    token_text.replace('\n', '\\n')  # Escape newlines
                                ])
                        
                        if data:
                            training_data_table = wandb.Table(columns=columns, data=data)
                            wandb.log({"train/data_samples": training_data_table})
                            logger.info(f"Logged {len(data)} training data tokens to wandb")
                            
            except Exception as e:
                logger.warning(f"Failed to log training data samples: {e}")

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Override training step to add data logging"""
        # Log training data samples (only on first step)
        if hasattr(self, 'state') and self.state.global_step == 0:
            self._log_training_data_sample(inputs, self.state.epoch or 0, 0)
            
        # Call parent training step
        return super().training_step(model, inputs)

    def _track_best_performance(self, metrics: Dict[str, float], epoch: int, checkpoint_dir: str) -> bool:
        """Track best model performance and log to wandb similar to coconut's best_acc tracking"""
        # Extract accuracy metric (could be 'accuracy', 'acc', or 'eval_accuracy')
        acc_key = None
        for key in ['accuracy', 'acc', 'eval_accuracy']:
            if key in metrics:
                acc_key = key
                break
                
        if acc_key is None:
            logger.warning("No accuracy metric found for best model tracking")
            return False
            
        current_acc = metrics[acc_key]
        
        # Initialize best accuracy if not set
        if not hasattr(self, 'best_accuracy'):
            self.best_accuracy = 0.0
            self.best_epoch = -1
            self.best_checkpoint = None
            
        is_best = current_acc > self.best_accuracy
        
        if is_best:
            self.best_accuracy = current_acc
            self.best_epoch = epoch
            self.best_checkpoint = checkpoint_dir
            logger.info(f"🎉 New best accuracy: {current_acc:.4f} at epoch {epoch + 1}")
            
            # Log best model info to wandb
            if 'wandb' in self.args.report_to:
                try:
                    import wandb
                    if wandb.run:
                        best_metrics = {
                            'best/accuracy': self.best_accuracy,
                            'best/epoch': self.best_epoch + 1,
                            'best/checkpoint': checkpoint_dir,
                        }
                        # Add all current metrics with 'best/' prefix
                        for key, value in metrics.items():
                            best_metrics[f'best/{key}'] = value
                            
                        wandb.log(best_metrics)
                        logger.info(f"Updated best model metrics in wandb: {best_metrics}")
                except ImportError:
                    pass
        else:
            logger.info(f"Current accuracy: {current_acc:.4f}, Best: {self.best_accuracy:.4f} (epoch {self.best_epoch + 1})")
            
        return is_best

    def _log_performance_summary(self) -> None:
        """Log final performance summary similar to coconut's final logging"""
        if hasattr(self, 'best_accuracy') and 'wandb' in self.args.report_to:
            try:
                import wandb
                if wandb.run and self.is_world_process_zero():
                    summary_metrics = {
                        'summary/best_accuracy': self.best_accuracy,
                        'summary/best_epoch': self.best_epoch + 1,
                        'summary/total_train_steps': self.total_train_steps,
                    }
                    
                    if hasattr(self, 'best_checkpoint'):
                        summary_metrics['summary/best_checkpoint'] = self.best_checkpoint
                        
                    wandb.log(summary_metrics)
                    logger.info(f"Logged training summary to wandb: {summary_metrics}")
                    
                    # Mark run as finished
                    wandb.finish()
            except ImportError:
                pass
    
    def _update_for_stage(self, stage: int) -> None:
        """Update dataset and training configuration for a new CoCoNut stage."""
        logger.info(f"Transitioning to CoCoNut stage {stage}")
        
        # Apply progressive curriculum to the training dataset
        if hasattr(self.train_dataset, 'apply_progressive_curriculum'):
            # Log dataset state before update for verification
            if hasattr(self.train_dataset, 'data') and len(self.train_dataset.data) > 0:
                sample_before = self.train_dataset.data[0] if len(self.train_dataset.data) > 0 else None
                logger.info(f"Dataset sample before curriculum update (stage {stage}): "
                           f"steps={sample_before.get('steps', 'N/A') if sample_before else 'No data'}")
            
            self.train_dataset.apply_progressive_curriculum(
                scheduled_stage=stage,
                c_thought=self.args.c_thought,
                max_latent_stage=self.args.max_latent_stage,
                uniform_prob=self.args.uniform_prob,
                pad_latent_to_max=self.args.pad_latent_to_max,
                no_cot=False,
            )
            
            # Log dataset state after update for verification
            if hasattr(self.train_dataset, 'data') and len(self.train_dataset.data) > 0:
                sample_after = self.train_dataset.data[0] if len(self.train_dataset.data) > 0 else None
                logger.info(f"Dataset sample after curriculum update (stage {stage}): "
                           f"steps={sample_after.get('steps', 'N/A') if sample_after else 'No data'}")
                
            logger.info(f"Applied progressive curriculum for stage {stage} - Dataset size: {len(self.train_dataset)}")
        else:
            logger.warning("Training dataset does not support progressive curriculum")
        
        # Refresh the dataloader to use updated dataset
        if hasattr(self, '_last_train_dataloader'):
            del self._last_train_dataloader  # Clear any cached dataloader
        logger.info("Dataloader will be refreshed for updated curriculum")
        
        # Reset optimizer if configured
        if hasattr(self.args, 'reset_optimizer') and self.args.reset_optimizer:
            logger.info("Resetting optimizer for new stage")
            self.create_optimizer()
            
        # Log stage transition to wandb
        if 'wandb' in self.args.report_to:
            try:
                import wandb
                if wandb.run:
                    stage_transition = {
                        'coconut/stage_transition': stage,
                        'coconut/latent_tokens_count': stage * self.args.c_thought,
                        'coconut/stage_timestamp': time.time(),
                        'coconut/dataset_size_after_update': len(self.train_dataset) if hasattr(self, 'train_dataset') else 0,
                    }
                    wandb.log(stage_transition)
                    logger.info(f"Logged stage transition to wandb: stage {stage}")
            except ImportError:
                pass
    
    def _log_coconut_stage_metrics(self, current_stage: int, stage_epoch: int, stage_progress: float) -> None:
        """Log CoCoNut specific stage progression metrics."""
        if 'wandb' in self.args.report_to:
            try:
                import wandb
                if wandb.run:
                    stage_metrics = {
                        'coconut/current_stage': current_stage,
                        'coconut/stage_epoch': stage_epoch,
                        'coconut/stage_progress': stage_progress,
                        'coconut/latent_replacement_ratio': current_stage / max(1, self.args.max_latent_stage),
                    }
                    wandb.log(stage_metrics)
            except ImportError:
                pass