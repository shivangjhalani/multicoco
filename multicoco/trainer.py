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
import json
import os
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


class EvalOutput:
    def __init__(self, metrics, num_samples):
        self.metrics = metrics
        self.num_samples = num_samples

class CoCoTrainer(Trainer):
    def __init__(self, *args, processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        # The eval_config is now passed directly in training_args
        self.eval_config = self.args.eval_config

    # The compute_metrics method is no longer needed as the evaluation_loop
    # calculates and logs metrics directly.

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ):
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        all_preds = []
        all_labels = []

        for step, inputs in enumerate(tqdm(dataloader, desc=description)):
            # Pop metadata that the model doesn't expect
            prompt_lengths = inputs.pop('prompt_lengths').cpu().numpy()
            question_ids = inputs.pop('question_ids')
            original_answers = inputs.pop('original_answers')
            
            # Move inputs to device
            inputs = self._prepare_inputs(inputs)

            # Create generation inputs by masking out the answer part of the sequence
            gen_input_ids = inputs['input_ids'].clone()
            gen_attention_mask = inputs['attention_mask'].clone()

            for i in range(gen_input_ids.size(0)):
                prompt_len = prompt_lengths[i]
                gen_input_ids[i, prompt_len:] = self.processor.tokenizer.pad_token_id
                gen_attention_mask[i, prompt_len:] = 0
            
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values=inputs['pixel_values'],
                    input_ids=gen_input_ids,
                    attention_mask=gen_attention_mask,
                    image_flags=inputs.get('image_flags'), # Use .get for safety
                    max_new_tokens=128,
                    num_beams=3,
                    do_sample=False
                )

            # Decode only the newly generated tokens
            decoded_preds = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # The prompt is still in the decoded output, so we need to parse it out.
            # A robust way is to decode the prompt part and remove it.
            for i in range(len(decoded_preds)):
                prompt_len = prompt_lengths[i]
                # The actual prompt text needs to be reconstructed and removed from the generated text
                # This is complex. A simpler way is to parse based on the expected output format.
                generated_text = decoded_preds[i] # This contains prompt + answer
                
                # Let's find the answer part based on how the prompt was constructed.
                # A common pattern is that the assistant's response starts after "assistant\n".
                # The processor.decode will give us the full text.
                if 'assistant\n' in generated_text:
                    generated_text = generated_text.split('assistant\n')[-1]

                # Parse output based on CoT or Coconut
                if self.eval_config.get('cot'):
                    if "The answer is" in generated_text:
                        parsed_pred = generated_text.split("The answer is")[-1].strip()
                    else:
                        parsed_pred = generated_text.strip()
                else:
                    parsed_pred = generated_text.strip()

                all_preds.append({"image_id": question_ids[i], "caption": parsed_pred})
                all_labels.append({"image_id": question_ids[i], "caption": original_answers[i]})


        # Save predictions to a file for evaluation
        results_path = os.path.join(self.args.output_dir, 'coco_predictions.json')
        with open(results_path, 'w') as f:
            json.dump(all_preds, f, indent=4)

        # Run COCO evaluation
        coco = COCO()
        coco.dataset = {
            'annotations': [{'image_id': item['image_id'], 'caption': item['caption'], 'id': i} for i, item in enumerate(all_labels)],
            'images': [{'id': item['image_id']} for item in all_labels]
        }
        coco.createIndex()

        coco_result = coco.loadRes(results_path)
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.evaluate()

        metrics = coco_eval.eval
        metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}

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
