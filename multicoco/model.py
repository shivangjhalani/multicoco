import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class MultiCoCo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_id = args.get('model_id', 'OpenGVLab/InternVL3-1B')
        
        torch_dtype = torch.bfloat16 if args.get('bf16', True) else torch.float32

        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            use_fast=False
        )

        # Add special tokens for latent thoughts
        special_tokens_to_add = ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]
        self.tokenizer.add_tokens(special_tokens_to_add)
        self.model.language_model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, pixel_values, input_ids, attention_mask, labels, num_patches_list, image_flags):
        
        # num_patches_list is not used in the forward pass for training,
        # it's primarily for the chat/generation methods.
        output = self.model(
            pixel_values=pixel_values.to(self.model.dtype),
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            image_flags=image_flags,
        )
        return output
