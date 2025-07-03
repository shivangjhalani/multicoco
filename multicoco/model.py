import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MultiCoCo(nn.Module):
    def __init__(self, model_id, latent_tokens, special_tokens):
        super(MultiCoCo, self).__init__()
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

        # Add special tokens
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.model.language_model.resize_token_embeddings(len(self.tokenizer))

        self.latent_tokens = latent_tokens

    def forward(self, pixel_values, input_ids, attention_mask, labels):
        # The underlying model's forward method for training does not expect num_patches_list.
        # It should identify images based on the <img> token in the input_ids.
        output = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return output
