import logging
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
logger = logging.getLogger(__name__)

class LatentWrapper(nn.Module):
    """
    Simplified LatentWrapper that follows CoCoNut's end-to-end learning approach.
    Removes hidden state injection to allow natural latent token learning.
    """

    def __init__(self, model: nn.Module, tokenizer):
        super().__init__()
        # Store the wrapped model and tokenizer
        self.model = model
        self.tokenizer = tokenizer
        self.latent_id = self.tokenizer.convert_tokens_to_ids('<|latent|>')
        self.start_id = self.tokenizer.convert_tokens_to_ids('<|start_latent|>')
        self.end_id = self.tokenizer.convert_tokens_to_ids('<|end_latent|>')
        if self.latent_id is None or self.start_id is None or self.end_id is None:
            logger.warning('Some latent tokens not found in tokenizer vocabulary')

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, pixel_values: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, **kwargs):
        # No injection: Just call base model for end-to-end learning
        return self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels, **kwargs)

    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, pixel_values: Optional[torch.Tensor]=None, **kwargs) -> torch.Tensor:
        # Use InternVL's generate directly (handles multimodal)
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, **kwargs)

    # Explicit delegation for commonly used attributes to maintain compatibility
    @property
    def device(self):
        return self.model.device
    
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def resize_token_embeddings(self, new_num_tokens):
        return self.model.resize_token_embeddings(new_num_tokens)
    
    def train(self, mode=True):
        self.model.train(mode)
        return super().train(mode)
    
    def eval(self):
        self.model.eval()
        return super().eval()
    
    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return super().to(*args, **kwargs)