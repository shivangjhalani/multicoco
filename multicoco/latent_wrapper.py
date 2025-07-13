import logging
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
logger = logging.getLogger(__name__)

class LatentWrapper(nn.Module):
    """
    Simplified LatentWrapper that follows CoCoNut's end-to-end learning approach.
    No hidden state injection - latent tokens are learned naturally like regular vocabulary.
    
    This implementation removes the problematic two-pass forward with hidden state injection
    and instead relies on the model learning latent token representations naturally through
    standard autoregressive training, as in the original CoCoNut paper.
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
        
        logger.info("LatentWrapper initialized for end-to-end latent learning (CoCoNut-style)")

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, pixel_values: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, **kwargs):
        """
        Direct delegation to base model - no injection.
        Latent tokens are treated as regular vocabulary and learned through autoregressive loss.
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels, **kwargs)

    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, pixel_values: Optional[torch.Tensor]=None, **kwargs) -> torch.Tensor:
        """
        Direct delegation to InternVL's optimized generate - no custom injection.
        Uses InternVL's native multimodal generation capabilities.
        """
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, **kwargs)

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped model."""
        if name in ['model', 'tokenizer']:
            return super().__getattribute__(name)
        if hasattr(self.model, name):
            return getattr(self.model, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")