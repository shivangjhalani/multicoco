"""
Simplified LatentWrapper that properly separates multimodal preparation from latent injection.
This follows Coconut's clean multi-pass approach while respecting InternVL's multimodal architecture.
"""
import logging
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
from .constants import END_LATENT_TOKEN, ENABLE_KV_CACHING, LATENT_TOKEN, START_LATENT_TOKEN

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
        self.base_model = model  # Keep internal name as base_model for consistency
        self.tokenizer = tokenizer
        self.enable_norm_logging = False  # Default to False for compatibility
        
        # Get token IDs for latent spans
        self.latent_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
        self.start_id = tokenizer.convert_tokens_to_ids(START_LATENT_TOKEN)
        self.end_id = tokenizer.convert_tokens_to_ids(END_LATENT_TOKEN)
        
        # Get embedding layer for token replacement
        self.embedding = model.get_input_embeddings()
        
        if self.latent_id is None or self.start_id is None or self.end_id is None:
            logger.warning("Some latent tokens not found in tokenizer vocabulary")

    def __getattr__(self, name):
        """Delegate attribute access to base model"""
        try:
            return getattr(self.base_model, name)
        except AttributeError:
            pass
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def get_input_embeddings(self):
        """Get input embeddings layer from base model"""
        return self.base_model.get_input_embeddings()
    
    def multimodal_prep(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                       pixel_values: Optional[torch.Tensor] = None, **kwargs):
        """
        Prepare multimodal embeddings using InternVL's native method.
        This is a compatibility method that exposes the multimodal preparation step.
        """
        if pixel_values is None:
            # No images, return standard text embeddings
            return self.embedding(input_ids)
        
        # Use InternVL's multimodal preparation
        return self.base_model.prepare_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def latent_injection(self, embeddings: torch.Tensor, input_ids: torch.Tensor):
        """
        Apply latent injection to prepared embeddings.
        This is a compatibility method that exposes the latent injection step.
        """
        spans = self._extract_latent_spans(input_ids)
        if not any(spans):
            return embeddings
        
        # Apply Coconut-style latent injection
        return self._inject_latent_representations(embeddings, input_ids, spans)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                pixel_values: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, 
                image_embeds: Optional[torch.Tensor] = None, **kwargs):
        """
        Main forward pass with clean separation of concerns:
        1. Check for latent spans
        2. Prepare multimodal embeddings once
        3. Apply Coconut-style latent injection
        4. Forward through language model
        """
        # Extract latent spans
        spans = self._extract_latent_spans(input_ids)
        
        # If no latent spans, use standard forward
        if not any(spans):
            return self.base_model(
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
        Implement Coconut's clean multi-pass approach:
        1. Prepare multimodal embeddings once
        2. For each latent token position, do a forward pass to get hidden states
        3. Replace latent tokens with hidden states from previous positions
        4. Final forward pass with modified embeddings
        """
        # Step 1: Prepare multimodal embeddings using InternVL's native method
        inputs_embeds = self._prepare_multimodal_embeddings(input_ids, pixel_values)
        
        # Step 2: Apply Coconut's latent injection logic
        max_n_latents = max(len(span_list) for span_list in spans) if spans else 0
        
        if max_n_latents == 0:
            # No latent tokens, proceed normally
            return self.base_model.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
        
        # Get positions of all latent tokens across the batch
        latent_positions = self._get_all_latent_positions(input_ids, spans)
        
        # Multi-pass forward following Coconut's approach
        return self._multi_pass_forward(
            inputs_embeds, attention_mask, labels, latent_positions, max_n_latents, **kwargs
        )
    
    def _prepare_multimodal_embeddings(self, input_ids: torch.Tensor, pixel_values: Optional[torch.Tensor]) -> torch.Tensor:
        """Prepare multimodal embeddings using InternVL's native method"""
        if hasattr(self.base_model.model, 'prepare_inputs_for_multimodal'):
            # Use InternVL's multimodal preparation
            return self.base_model.model.prepare_inputs_for_multimodal(
                input_ids=input_ids, 
                pixel_values=pixel_values
            )
        else:
            # Fallback to text embeddings only
            return self.embedding(input_ids)
    
    def _get_all_latent_positions(self, input_ids: torch.Tensor, spans: List[List[Tuple[int, int]]]) -> List[List[int]]:
        """Get all latent token positions for multi-pass processing"""
        latent_positions = []
        for batch_idx in range(input_ids.shape[0]):
            batch_positions = []
            for start, end in spans[batch_idx]:
                # Include all positions between start and end (exclusive)
                for pos in range(start + 1, end):  # Skip start/end markers
                    if input_ids[batch_idx, pos].item() == self.latent_id:
                        batch_positions.append(pos)
            latent_positions.append(batch_positions)
        return latent_positions
    
    def _multi_pass_forward(self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor], 
                           labels: Optional[torch.Tensor], latent_positions: List[List[int]], 
                           max_n_latents: int, **kwargs):
        """
        Coconut-style multi-pass forward:
        - For each pass, replace the next latent token with hidden state from previous position
        - Build up reasoning incrementally
        """
        current_embeds = inputs_embeds.clone()
        
        # Multiple passes, each handling one "layer" of latent tokens
        for pass_idx in range(max_n_latents):
            # Determine which positions to process in this pass
            positions_to_process = []
            for batch_idx, positions in enumerate(latent_positions):
                if pass_idx < len(positions):
                    positions_to_process.append((batch_idx, positions[pass_idx]))
            
            if not positions_to_process:
                break
            
            # Forward pass to get hidden states
            with torch.no_grad():
                outputs = self.base_model.model.language_model(
                    inputs_embeds=current_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    **kwargs
                )
                hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
            
            # Replace latent tokens with hidden states from previous positions
            for batch_idx, pos in positions_to_process:
                if pos > 0:  # Ensure we have a previous position
                    # Use hidden state from the position immediately before
                    prev_hidden = hidden_states[batch_idx, pos - 1]
                    current_embeds[batch_idx, pos] = prev_hidden
                    
                    if self.enable_norm_logging:
                        norm = prev_hidden.norm().item()
                        logger.debug(f'Pass {pass_idx}, Batch {batch_idx}, Pos {pos}: Hidden norm {norm:.4f}')
        
        # Final forward pass with all latent tokens replaced
        return self.base_model.model.language_model(
            inputs_embeds=current_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def _inject_latent_representations(self, embeddings: torch.Tensor, input_ids: torch.Tensor, 
                                     spans: List[List[Tuple[int, int]]]) -> torch.Tensor:
        """
        Inject latent representations into embeddings at specified positions.
        This is a simplified version for the compatibility API.
        """
        max_n_latents = max(len(span_list) for span_list in spans) if spans else 0
        if max_n_latents == 0:
            return embeddings
            
        # Get positions of all latent tokens across the batch  
        latent_positions = self._get_all_latent_positions(input_ids, spans)
        
        # For each latent position, do a forward pass to get hidden states
        current_embeds = embeddings.clone()
        
        for pass_idx in range(max_n_latents):
            # Get hidden states from current embeddings
            with torch.no_grad():
                outputs = self.base_model.model.language_model(
                    inputs_embeds=current_embeds,
                    attention_mask=None,  # Simplified for compatibility
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1]  # Last layer
            
            # Replace latent tokens with hidden states from previous positions
            for batch_idx, positions in enumerate(latent_positions):
                if pass_idx < len(positions):
                    pos = positions[pass_idx]
                    if pos > 0:  # Use hidden state from previous position
                        prev_hidden = hidden_states[batch_idx, pos - 1]
                        current_embeds[batch_idx, pos] = prev_hidden
        
        return current_embeds

    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                pixel_values: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Generation method that handles latent injection during generation"""
        # For generation, we need to be more careful about when to apply latent injection
        spans = self._extract_latent_spans(input_ids)
        
        if not any(spans):
            # No latent spans, use standard generation
            return self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                **kwargs
            )
        else:
            # For now, fall back to standard generation
            # TODO: Implement proper latent-aware generation
            logger.warning("Latent-aware generation not yet implemented, falling back to standard generation")
            return self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                pixel_values=pixel_values,
                **kwargs
            )
