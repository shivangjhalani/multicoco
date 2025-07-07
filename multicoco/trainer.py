import os
import torch
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from transformers import Trainer
from transformers.trainer_pt_utils import (
    find_batch_size,
    nested_concat,
    nested_numpify,
    nested_truncate,
    nested_detach
)
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import LabelSmoother
from typing import Optional, List, Tuple, Dict
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.trainer_utils import EvalPrediction
import random


class EvalOutput:
    def __init__(self, metrics, num_samples):
        self.metrics = metrics
        self.num_samples = num_samples

class CoCoTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # The processor is no longer needed here, the base Trainer
        # correctly handles the tokenizer.
        if 'processor' in kwargs:
            kwargs.pop('processor')
        super().__init__(*args, **kwargs)
        self.best_val_acc = 0.0
        
        # CoCoNut training parameters
        self.coconut_enabled = getattr(self.args, 'eval_config', {}).get('coconut', False)
        self.c_thought = getattr(self.args, 'c_thought', 0)
        self.max_latent_stage = getattr(self.args, 'max_latent_stage', 0)
        self.current_stage = 0
        
        # Special token IDs
        if hasattr(self.args, 'thought_token_id'):
            self.thought_token_id = self.args.thought_token_id
            self.start_thought_id = self.args.start_thought_id
            self.end_thought_id = self.args.end_thought_id
        else:
            # Fallback if not provided
            self.thought_token_id = self.tokenizer.convert_tokens_to_ids('<thought>')
            self.start_thought_id = self.tokenizer.convert_tokens_to_ids('<start_thought>')
            self.end_thought_id = self.tokenizer.convert_tokens_to_ids('<end_thought>')

    def _gen_kwargs_for_evaluation(self):
        """
        Generate generation kwargs for evaluation.
        """
        gen_kwargs = getattr(self.args, "generation_kwargs", {})
        if gen_kwargs is None:
            gen_kwargs = {}
        
        # Set default generation parameters
        if "max_new_tokens" not in gen_kwargs and "max_length" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 256
        if "do_sample" not in gen_kwargs:
            gen_kwargs["do_sample"] = False
        if "pad_token_id" not in gen_kwargs and self.tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        
        return gen_kwargs

    def apply_coconut_masking(self, input_ids, labels, stage):
        """
        Apply CoCoNut masking strategy for latent reasoning.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            labels: Labels for training [batch_size, seq_len] 
            stage: Current training stage (0 = no masking, higher = more masking)
        
        Returns:
            Modified input_ids and labels with thought tokens masked according to stage
        """
        if not self.coconut_enabled or stage == 0:
            return input_ids, labels
        
        batch_size, seq_len = input_ids.shape
        modified_input_ids = input_ids.clone()
        modified_labels = labels.clone()
        
        for i in range(batch_size):
            # Find thought token positions
            thought_positions = (input_ids[i] == self.thought_token_id).nonzero(as_tuple=True)[0]
            start_thought_positions = (input_ids[i] == self.start_thought_id).nonzero(as_tuple=True)[0]
            end_thought_positions = (input_ids[i] == self.end_thought_id).nonzero(as_tuple=True)[0]
            
            # Apply progressive masking based on stage
            if len(thought_positions) > 0:
                # Determine how many thought tokens to mask based on stage and c_thought
                total_thoughts = len(thought_positions)
                mask_ratio = min(stage / self.max_latent_stage, 1.0)
                num_to_mask = int(total_thoughts * mask_ratio * self.c_thought / 10.0)  # c_thought is a scaling factor
                
                if num_to_mask > 0:
                    # Randomly select which thought tokens to mask
                    mask_indices = random.sample(thought_positions.tolist(), min(num_to_mask, total_thoughts))
                    
                    for pos in mask_indices:
                        # Replace thought token with a special mask token or skip in loss
                        modified_labels[i, pos] = -100  # Ignore in loss calculation
                        
            # Handle start/end thought token pairs
            if len(start_thought_positions) > 0 and len(end_thought_positions) > 0:
                for start_pos, end_pos in zip(start_thought_positions, end_thought_positions):
                    if start_pos < end_pos:
                        # Apply masking to thought content based on stage
                        thought_length = end_pos - start_pos - 1
                        if thought_length > 0 and stage > 0:
                            mask_ratio = min(stage / self.max_latent_stage, 1.0)
                            num_to_mask = int(thought_length * mask_ratio)
                            
                            if num_to_mask > 0:
                                # Mask random positions within the thought
                                thought_range = list(range(start_pos + 1, end_pos))
                                mask_positions = random.sample(thought_range, min(num_to_mask, len(thought_range)))
                                
                                for pos in mask_positions:
                                    modified_labels[i, pos] = -100
        
        return modified_input_ids, modified_labels

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss to implement CoCoNut training logic.
        """
        if self.coconut_enabled:
            # Apply CoCoNut masking strategy
            input_ids = inputs.get('input_ids')
            labels = inputs.get('labels')
            
            if input_ids is not None and labels is not None:
                modified_input_ids, modified_labels = self.apply_coconut_masking(
                    input_ids, labels, self.current_stage
                )
                inputs['input_ids'] = modified_input_ids
                inputs['labels'] = modified_labels
        
        return super().compute_loss(model, inputs, return_outputs)

    # def training_step(self, model, inputs, num_items_in_batch=None):
    #     """
    #     Override training_step to handle CoCoNut staging.
    #     """
    #     # Check if we should advance to next stage
    #     if self.coconut_enabled and hasattr(self.state, 'epoch'):
    #         expected_stage = min(int(self.state.epoch), self.max_latent_stage)
    #         if expected_stage != self.current_stage:
    #             self.current_stage = expected_stage
    #             if self.is_local_process_zero():
    #                 print(f"Advanced to CoCoNut stage {self.current_stage}/{self.max_latent_stage}")
    #     
    #     # Call parent method with all arguments
    #     if num_items_in_batch is not None:
    #         return super().training_step(model, inputs, num_items_in_batch)
    #     else:
    #         return super().training_step(model, inputs)

    def compute_metrics(self, p: EvalPrediction):
        # This is a placeholder. The evaluation_loop calculates and returns metrics directly.
        return {}

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()
        
        self.callback_handler.eval_dataloader = dataloader

        all_preds_text = []
        all_labels_text = []

        for step, inputs in enumerate(tqdm(dataloader, desc=description)):
            questions = inputs.pop("questions")
            answers = inputs.pop("answers")
            pixel_values = inputs["pixel_values"].to(self.args.device)
            
            all_labels_text.extend(answers)
            
            is_cot = self.args.eval_config.get('cot', False)
            is_coconut = self.args.eval_config.get('coconut', False)

            for i, q in enumerate(questions):
                
                # For InternVL, we need to include <image> token in the text
                # The model uses this token to know where to inject visual features
                user_content_str = f"<image>\n{q}"
                if is_cot:
                    user_content_str += " Let's think step by step."
                elif is_coconut:
                    # For CoCoNut evaluation, we use thought tokens to encourage latent reasoning
                    user_content_str += " <start_thought>Let me think about this step by step.<end_thought> The answer is"
                else:
                    user_content_str += " The answer is"
            
                # Use InternVL's chat method which handles the conversation format properly
                generation_config = {
                    'max_new_tokens': 256,
                    'do_sample': False,
                    'num_beams': 1,
                }
                
                # Access the underlying InternVL model from our wrapper
                underlying_model = model.model if hasattr(model, 'model') else model
                
                # Ensure pixel_values have the correct dtype matching the model
                current_pixel_values = pixel_values[i:i+1]
                if hasattr(underlying_model, 'dtype'):
                    current_pixel_values = current_pixel_values.to(underlying_model.dtype)
                elif hasattr(underlying_model, 'vision_model') and hasattr(underlying_model.vision_model, 'dtype'):
                    current_pixel_values = current_pixel_values.to(underlying_model.vision_model.dtype)
                else:
                    # Default to bfloat16 if we can't determine the model dtype
                    current_pixel_values = current_pixel_values.to(torch.bfloat16)
                
                decoded_pred = underlying_model.chat(
                    self.tokenizer,
                    current_pixel_values,
                    user_content_str,
                    generation_config
                )
                
                # Clean up thought tokens from prediction if they appear
                if is_coconut:
                    # Remove any thought tokens that might have been generated
                    for token in ['<thought>', '<start_thought>', '<end_thought>']:
                        decoded_pred = decoded_pred.replace(token, '')
                    decoded_pred = decoded_pred.strip()
                
                all_preds_text.append(decoded_pred)

        # Post-process and compute metrics
        correct = 0
        is_cot = self.args.eval_config.get('cot', False)
        
        # Log some examples for debugging
        if self.is_local_process_zero():
            print(f"\n=== EVALUATION RESULTS ===")
            print(f"Total samples: {len(all_labels_text)}")
            for i, (pred, label) in enumerate(zip(all_preds_text[:5], all_labels_text[:5])):  # Show first 5
                print(f"Sample {i+1}:")
                print(f"  Prediction: '{pred}'")
                print(f"  Ground Truth: '{label}'")
                print(f"  Correct: {pred.strip().lower() == label.strip().lower()}")
                print()
        
        for pred, label in zip(all_preds_text, all_labels_text):
            pred_processed = pred.strip().lower()
            if is_cot:
                if "the answer is" in pred_processed:
                    pred_processed = pred_processed.split("the answer is")[-1].strip()
            
            if pred_processed == label.strip().lower():
                correct += 1
        
        accuracy = correct / len(all_labels_text) if len(all_labels_text) > 0 else 0.0
        
        if self.is_local_process_zero():
            print(f"Final Accuracy: {accuracy:.4f} ({correct}/{len(all_labels_text)})")
            print("=" * 50)
            
            # Also write to log file
            log_dir = getattr(self.args, 'log_dir', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            # Determine log file name based on evaluation type
            eval_type = "coconut" if self.args.eval_config.get('coconut', False) else "cot" if self.args.eval_config.get('cot', False) else "vanilla"
            log_file = os.path.join(log_dir, f'evaluation_{eval_type}.log')
            
            with open(log_file, 'a') as f:
                f.write(f"\n=== EVALUATION RESULTS ({eval_type.upper()}) ===\n")
                f.write(f"Total samples: {len(all_labels_text)}\n")
                f.write(f"Correct predictions: {correct}\n")
                f.write(f"Final Accuracy: {accuracy:.4f}\n")
                f.write(f"Timestamp: {torch.cuda.Event().query() if torch.cuda.is_available() else 'N/A'}\n")
                f.write("=" * 50 + "\n")
        
        # Log CoCoNut stage information
        stage_info = {}
        if self.coconut_enabled:
            stage_info[f"{metric_key_prefix}_coconut_stage"] = self.current_stage
            stage_info[f"{metric_key_prefix}_max_latent_stage"] = self.max_latent_stage
        
        metrics = {
            f"{metric_key_prefix}_accuracy": accuracy, 
            f"{metric_key_prefix}_loss": -1.0,
            **stage_info
        }
        
        self.log(metrics)

        # Return in the format expected by Trainer.evaluate()
        from types import SimpleNamespace
        return SimpleNamespace(
            metrics=metrics,
            num_samples=len(all_labels_text),
            eval_preds=None
        )

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        gen_kwargs = self._gen_kwargs_for_evaluation()

        generated_tokens = self.model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )

        # In generation mode, there's no loss.
        # The "logits" are the generated sequences. The "labels" are not used.
        return (None, generated_tokens, None)
