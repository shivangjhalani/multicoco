import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor

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
        self.image_processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Manually set the image context token ID for the model.
        # This is required for the vanilla model's generate function.
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<img>')

        if special_tokens:
            num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            if num_added_tokens > 0:
                self.model.language_model.resize_token_embeddings(len(self.tokenizer))

        self.latent_tokens = latent_tokens

    def forward(self, pixel_values, input_ids, attention_mask, labels, image_flags):
        output = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            image_flags=image_flags,
            return_dict=True
        )
        return output
