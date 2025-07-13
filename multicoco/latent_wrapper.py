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
        # Store the wrapped model and tokenizer using the same naming as original CoCoNut
        self.base_causallm = model  # Use same name as original CoCoNut for compatibility
        self.tokenizer = tokenizer
        self.latent_id = self.tokenizer.convert_tokens_to_ids('<|latent|>')
        self.start_id = self.tokenizer.convert_tokens_to_ids('<|start_latent|>')
        self.end_id = self.tokenizer.convert_tokens_to_ids('<|end_latent|>')
        if self.latent_id is None or self.start_id is None or self.end_id is None:
            logger.warning('Some latent tokens not found in tokenizer vocabulary')

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, pixel_values: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, **kwargs):
        # No injection: Just call base model for end-to-end learning
        return self.base_causallm(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels, **kwargs)

    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, pixel_values: Optional[torch.Tensor]=None, **kwargs) -> torch.Tensor:
        # Use InternVL's generate directly (handles multimodal)
        return self.base_causallm.generate(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, **kwargs)

    def __getattr__(self, name):
        # Delegate to the wrapped model for attributes not found on this class
        if name in ['base_causallm', 'tokenizer']:
            return super().__getattribute__(name)
        if hasattr(self.base_causallm, name):
            return getattr(self.base_causallm, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")