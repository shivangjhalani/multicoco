import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class MultiCoCo(nn.Module):
    def __init__(self, model_id, latent_tokens={}, special_tokens=[]):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        if special_tokens:
            num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            if num_added_tokens > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))

        self.latent_tokens = latent_tokens

    def forward(self, input_ids, attention_mask, pixel_values, labels=None, image_flags=None):
        """
        Forward pass for the MultiCoCo model.
        It dynamically handles calls to the underlying transformer model,
        passing `image_flags` only if the model is the patched version.
        """
        # Check if the underlying model is our patched version by looking for a specific attribute in its config.
        is_patched_model = hasattr(self.model.config, 'num_image_token')

        if is_patched_model:
            # The patched model expects the `image_flags` argument.
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                image_flags=image_flags,
            )
        else:
            # The vanilla model does not have or expect the `image_flags` argument.
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            ) 