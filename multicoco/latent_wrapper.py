import logging
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
logger = logging.getLogger(__name__)

class LatentWrapper(nn.Module):
    """
    LatentWrapper implementing the CoCoNut algorithm with hidden state injection.
    This follows the original Facebook Research CoCoNut implementation adapted for multimodal models.
    """

    def __init__(self, model: nn.Module, tokenizer):
        super().__init__()
        # Store the wrapped model using add_module to ensure proper registration
        self.add_module('base_model', model)
        self.tokenizer = tokenizer  # Tokenizer is not a PyTorch module
        self.latent_id = self.tokenizer.convert_tokens_to_ids('<|latent|>')
        self.start_id = self.tokenizer.convert_tokens_to_ids('<|start_latent|>')
        self.end_id = self.tokenizer.convert_tokens_to_ids('<|end_latent|>')
        if self.latent_id is None or self.start_id is None or self.end_id is None:
            logger.warning('Some latent tokens not found in tokenizer vocabulary')
        
        # Get embedding layer for CoCoNut algorithm
        if hasattr(self.model, 'language_model'):
            self.embedding = self.model.language_model.model.embed_tokens
        else:
            self.embedding = self.model.get_input_embeddings()

    @property
    def model(self):
        """Access the wrapped model through PyTorch's module system"""
        return self._modules['base_model']

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, pixel_values: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, **kwargs):
        # Check if there are latent spans to process
        spans = self._extract_latent_spans(input_ids)
        if not any(spans):
            # No latent tokens, use standard forward
            return self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels, **kwargs)
        
        # CoCoNut algorithm: hidden state injection
        image_embeds = self._compute_vision_embeddings(pixel_values)
        last_hidden = self._first_pass_hidden_states(input_ids, attention_mask, image_embeds)
        inputs_embeds = self._build_modified_embeddings(input_ids, spans, last_hidden)
        return self._second_pass_forward(input_ids, attention_mask, inputs_embeds, image_embeds, labels)

    def _extract_latent_spans(self, input_ids: torch.Tensor) -> List[List[Tuple[int, int]]]:
        """Extract latent token spans between start_latent and end_latent tokens"""
        spans = []
        for batch_idx in range(input_ids.shape[0]):
            ids = input_ids[batch_idx].tolist()
            sample_spans = []
            current_pos = 0
            while True:
                try:
                    start = ids.index(self.start_id, current_pos)
                    end = ids.index(self.end_id, start + 1)
                    sample_spans.append((start, end))
                    current_pos = end + 1
                except ValueError:
                    break
            spans.append(sample_spans)
        return spans

    def _compute_vision_embeddings(self, pixel_values: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Compute vision embeddings using InternVL's vision tower and projector"""
        if pixel_values is None:
            return None
        
        # Ensure correct dtype
        if hasattr(self.model, 'model'):
            model_dtype = next(self.model.model.parameters()).dtype
            pixel_values = pixel_values.to(dtype=model_dtype)
        
        # Use InternVL's vision processing
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_tower'):
            vision_embeds = self.model.model.vision_tower(pixel_values)
            if hasattr(self.model.model, 'projector'):
                return self.model.model.projector(vision_embeds)
            return vision_embeds
        
        return None

    def _first_pass_hidden_states(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], image_embeds: Optional[torch.Tensor]) -> torch.Tensor:
        """First pass to get hidden states before injecting into latent tokens"""
        with torch.inference_mode():
            # Use InternVL's multimodal preparation
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'prepare_inputs_for_multimodal'):
                first_pass_embeds = self.model.model.prepare_inputs_for_multimodal(
                    input_ids=input_ids, 
                    pixel_values=None, 
                    image_embeds=image_embeds
                )
                first_out = self.model.model.language_model(
                    inputs_embeds=first_pass_embeds, 
                    attention_mask=attention_mask, 
                    output_hidden_states=True
                )
            else:
                # Fallback for models without prepare_inputs_for_multimodal
                first_out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
            
            return first_out.hidden_states[-1]

    def _build_modified_embeddings(self, input_ids: torch.Tensor, spans: List[List[Tuple[int, int]]], last_hidden: torch.Tensor) -> torch.Tensor:
        """Replace latent token embeddings with hidden states from previous token position"""
        inputs_embeds = self.embedding(input_ids).clone()
        
        for batch_idx, span_pairs in enumerate(spans):
            for start, end in span_pairs:
                if start == 0:
                    continue  # Skip if latent span starts at position 0
                
                span_length = end - start
                # Replace latent tokens with the hidden state from the previous token
                inputs_embeds[batch_idx, start:end] = last_hidden[batch_idx, start - 1].unsqueeze(0).repeat(span_length, 1)
        
        return inputs_embeds

    def _second_pass_forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], inputs_embeds: torch.Tensor, image_embeds: Optional[torch.Tensor], labels: Optional[torch.Tensor]) -> dict:
        """Second pass with modified embeddings containing injected hidden states"""
        # Use InternVL's multimodal preparation with modified embeddings
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'prepare_inputs_for_multimodal'):
            second_pass_embeds = self.model.model.prepare_inputs_for_multimodal(
                input_ids=input_ids,
                pixel_values=None,
                image_embeds=image_embeds,
                inputs_embeds=inputs_embeds
            )
            second_out = self.model.model.language_model(
                inputs_embeds=second_pass_embeds,
                attention_mask=attention_mask,
                use_cache=True
            )
        else:
            # Fallback for models without prepare_inputs_for_multimodal
            second_out = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True
            )
        
        logits = second_out.logits
        loss = None
        
        if labels is not None:
            # Compute cross-entropy loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {'loss': loss, 'logits': logits}

    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, pixel_values: Optional[torch.Tensor]=None, **kwargs) -> torch.Tensor:
        # Check if there are latent spans to process
        spans = self._extract_latent_spans(input_ids)
        if not any(spans):
            # No latent tokens, use standard generation
            return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, **kwargs)
        
        # For generation with latent tokens, we need to implement custom generation
        # This is more complex and would require implementing the full generation loop
        # For now, fall back to standard generation (this could be improved)
        logger.warning("Generation with latent tokens is not fully implemented. Using standard generation.")
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
        self._modules['base_model'] = self.model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

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
        self._modules['base_model'] = self.model.to(*args, **kwargs)
        return super().to(*args, **kwargs)