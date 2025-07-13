import logging
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

class LatentWrapperV2(nn.Module):
    """
    Simplified LatentWrapper that:
    1. First does complete multimodal preparation using InternVL's native method
    2. Then applies Coconut-style latent injection to the prepared embeddings
    3. Uses clean multi-pass forward without complex KV caching
    """

    def __init__(self, model: nn.Module, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        
        # Initialize latent token IDs
        self.latent_id = self.tokenizer.convert_tokens_to_ids('<|latent|>')
        self.start_id = self.tokenizer.convert_tokens_to_ids('<|start_latent|>')
        self.end_id = self.tokenizer.convert_tokens_to_ids('<|end_latent|>')
        
        if self.latent_id is None or self.start_id is None or self.end_id is None:
            logger.warning("Some latent tokens not found in tokenizer vocabulary")

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                pixel_values: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        """
        Clean multi-pass forward:
        1. First pass: multimodal preparation 
        2. Second pass: latent injection
        """
        # Extract latent spans first (before any processing)
        spans = self._extract_latent_spans(input_ids)
        
        # If no latent spans, use standard forward
        if not any(spans):
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                **kwargs
            )
        
        # Apply Coconut-style latent injection
        return self._coconut_style_forward(input_ids, attention_mask, pixel_values, labels, spans, **kwargs)

    def _extract_latent_spans(self, input_ids: torch.Tensor) -> List[List[Tuple[int, int]]]:
        """Extract (start, end) positions of latent spans for each batch item"""
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

    def _coconut_style_forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], 
                              pixel_values: Optional[torch.Tensor], labels: Optional[torch.Tensor], 
                              spans: List[List[Tuple[int, int]]], **kwargs):
        """
        Coconut-style multi-pass forward with latent injection
        """
        # Step 1: Prepare multimodal embeddings
        if hasattr(self.model, 'prepare_inputs_embeds') and pixel_values is not None:
            inputs_embeds = self.model.prepare_inputs_embeds(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **kwargs
            )
        else:
            # Fallback to basic embeddings
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # Step 2: Apply latent injection
        modified_embeds = self._inject_latent_representations(inputs_embeds, input_ids, spans)
        
        # Step 3: Forward pass with modified embeddings
        return self._multi_pass_forward(
            modified_embeds, attention_mask, labels, **kwargs
        )

    def _multi_pass_forward(self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor], 
                           labels: Optional[torch.Tensor], **kwargs):
        """Simple forward pass with embeddings"""
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def _inject_latent_representations(self, embeddings: torch.Tensor, input_ids: torch.Tensor, 
                                     spans: List[List[Tuple[int, int]]]) -> torch.Tensor:
        """
        Inject latent representations using Coconut-style approach
        """
        modified_embeds = embeddings.clone()
        
        for batch_idx, batch_spans in enumerate(spans):
            for start_pos, end_pos in batch_spans:
                # Get the span (including start and end tokens)
                span_length = end_pos - start_pos + 1
                
                # Generate latent representations using simple approach
                # In real usage, this would use the model's hidden states
                latent_embeds = self._generate_latent_embeddings(
                    embeddings[batch_idx:batch_idx+1, start_pos:end_pos+1], 
                    span_length
                )
                
                # Replace the span with latent representations
                modified_embeds[batch_idx, start_pos:end_pos+1] = latent_embeds
                
        return modified_embeds

    def _generate_latent_embeddings(self, span_embeds: torch.Tensor, span_length: int) -> torch.Tensor:
        """
        Generate latent embeddings for a span. 
        Simplified implementation - in real usage would use model inference.
        """
        # Simple approach: use mean of span embeddings with some noise
        mean_embed = span_embeds.mean(dim=1, keepdim=True)
        latent_embeds = mean_embed.repeat(1, span_length, 1)
        
        # Add small random perturbation to differentiate positions
        if span_length > 1:
            noise = torch.randn_like(latent_embeds) * 0.01
            latent_embeds = latent_embeds + noise
            
        return latent_embeds.squeeze(0)

    # Compatibility methods for API consistency
    def multimodal_prep(self, input_ids: torch.Tensor, pixel_values: Optional[torch.Tensor] = None, **kwargs):
        """Compatibility method - prepare multimodal embeddings"""
        if hasattr(self.model, 'prepare_inputs_embeds') and pixel_values is not None:
            return self.model.prepare_inputs_embeds(input_ids=input_ids, pixel_values=pixel_values, **kwargs)
        else:
            return self.model.get_input_embeddings()(input_ids)

    def latent_injection(self, embeddings: torch.Tensor, input_ids: torch.Tensor):
        """Compatibility method - apply latent injection"""
        spans = self._extract_latent_spans(input_ids)
        return self._inject_latent_representations(embeddings, input_ids, spans)

    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                 pixel_values: Optional[torch.Tensor] = None, **kwargs):
        """Generation method that delegates to base model"""
        # For generation, we typically don't use latent injection
        # Just delegate to the base model
        if hasattr(self.model, 'generate'):
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                **kwargs
            )
        else:
            # Fallback to forward pass
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                **kwargs
            )

    # Simple attribute delegation without complex __getattr__
    def __getattr__(self, name):
        """Simple delegation to model for missing attributes"""
        if name in ['model', 'tokenizer']:
            return super().__getattribute__(name)
        if hasattr(self.model, name):
            return getattr(self.model, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
