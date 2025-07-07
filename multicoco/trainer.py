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
            # image_flags are passed implicitly with the rest of `inputs` to the model
            
            all_labels_text.extend(answers)
            
            is_cot = self.args.eval_config.get('cot', False)

            for i, q in enumerate(questions):
                
                user_content_str = f"<img>\n{q}"
                if is_cot:
                    user_content_str += " Let's think step by step."
                else: # This case may not be used if coconut is separate
                    user_content_str += " The answer is"
            
                prompt_messages = [{"role": "user", "content": user_content_str}]
                prompt = self.tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
                
                eval_inputs = self.tokenizer(text=prompt, return_tensors='pt').to(self.args.device)

                gen_kwargs = self._gen_kwargs_for_evaluation()
                if "max_length" not in gen_kwargs and "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 256 # Default value

                # The `image_flags` are part of `inputs` but not used explicitly in the generate call here
                # since we're passing the full `pixel_values` for the batch. This is a simplification
                # for this specific evaluation loop. The model's forward pass during training uses it correctly.
                generated_ids = model.generate(
                    pixel_values=pixel_values[i:i+1],
                    input_ids=eval_inputs.input_ids,
                    attention_mask=eval_inputs.attention_mask,
                    **gen_kwargs,
                )
                
                input_len = eval_inputs.input_ids.shape[1]
                decoded_pred = self.tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
                all_preds_text.append(decoded_pred)

        # Post-process and compute metrics
        correct = 0
        is_cot = self.args.eval_config.get('cot', False)
        for pred, label in zip(all_preds_text, all_labels_text):
            pred_processed = pred.strip().lower()
            if is_cot:
                if "the answer is" in pred_processed:
                    pred_processed = pred_processed.split("the answer is")[-1].strip()
            
            if pred_processed == label.strip().lower():
                correct += 1
        
        accuracy = correct / len(all_labels_text) if len(all_labels_text) > 0 else 0.0
        
        # This is a simplified metric calculation. `self.log` would be better.
        metrics = {f"{metric_key_prefix}_accuracy": accuracy, f"{metric_key_prefix}_loss": -1.0}
        
        self.log(metrics)

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
