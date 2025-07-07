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
from typing import Optional, List, Tuple
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.trainer_utils import EvalPrediction


class EvalOutput:
    def __init__(self, metrics, num_samples):
        self.metrics = metrics
        self.num_samples = num_samples

class CoCoTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.processor = kwargs.pop('processor')
        super().__init__(*args, **kwargs)
        self.best_val_acc = 0.0

    def compute_metrics(self, p: EvalPrediction):
        # The predictions are the generated token IDs.
        # The labels are the ground truth answer token IDs, which are not directly available here
        # in the same format. We need to decode the predictions and compare to the original answers.
        # This requires a custom post-processing step.
        
        # For now, let's return a placeholder accuracy.
        # The actual implementation will depend on how we align predictions and labels.
        return {"accuracy": 0.0}

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        
        # We need to manually reconstruct the prompt for generation because our
        # data collator is designed for training, where input_ids include the answer.
        # For generation, we need input_ids that only contain the prompt.
        
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        self.callback_handler.eval_dataloader = dataloader

        all_preds = []
        all_labels = [] # We'll need to handle labels differently for generation

        for step, inputs in enumerate(dataloader):
            original_questions = inputs.pop("original_questions")
            answers = inputs.pop("answers") # Ground truth answers

            # Re-create the prompt string for generation
            image_token_placeholder = ' '.join(['<img>'] * 256)
            is_cot = self.args.eval_config.get('cot', False)
            prompts = []
            for q in original_questions:
                if is_cot:
                    prompt = f"{image_token_placeholder}\n{q} Let's think step by step."
                else: # Vanilla
                    prompt = f"{image_token_placeholder}\n{q} The answer is"
                prompts.append(prompt)

            # Tokenize the prompts to create new input_ids and attention_mask
            tokenized_prompts = self.processor(
                prompts, 
                return_tensors="pt", 
                padding=True,
            )
            inputs['input_ids'] = tokenized_prompts.input_ids.to(self.args.device)
            inputs['attention_mask'] = tokenized_prompts.attention_mask.to(self.args.device)
            
            # Re-create image_flags for the new inputs
            image_token_id = self.processor.convert_tokens_to_ids('<img>')
            inputs['image_flags'] = (inputs['input_ids'] == image_token_id).long()
            
            # Move pixel_values to the correct device
            inputs['pixel_values'] = inputs['pixel_values'].to(self.args.device)

            # We don't have ground truth labels in the same way for generation
            if 'labels' in inputs:
                inputs.pop('labels')
            
            _, logits, _ = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            if logits is not None:
                all_preds.append(logits.detach().cpu())
            # We will need a way to associate the original answers with the predictions
            all_labels.extend(answers)

        if len(all_preds) > 0:
            # Decode predictions
            decoded_preds = self.processor.batch_decode(torch.cat(all_preds, dim=0), skip_special_tokens=True)
            
            # Simple accuracy calculation
            correct = 0
            for pred, label in zip(decoded_preds, all_labels):
                if is_cot:
                    # For CoT, extract the final answer
                    if "the answer is" in pred.lower():
                        pred = pred.lower().split("the answer is")[-1].strip()
                if pred.strip().lower() == label.strip().lower():
                    correct += 1
            accuracy = correct / len(all_labels)
            metrics = {"accuracy": accuracy}
        else:
            metrics = {}

        metrics[f"{metric_key_prefix}_loss"] = -1.0 # No loss computed

        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return metrics

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
            image_flags=inputs.get("image_flags"),
            **gen_kwargs,
        )

        # In generation mode, there's no loss.
        # The "logits" are the generated sequences. The "labels" are not used.
        return (None, generated_tokens, None)
