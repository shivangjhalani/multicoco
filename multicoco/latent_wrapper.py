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

    def __getattr__(self, name):
        if name in ['model', 'tokenizer']:
            return super().__getattribute__(name)
        if hasattr(self.model, name):
            return getattr(self.model, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")