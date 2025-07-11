"""
LatentWrapper: inject hidden states for <|start_latent|> xxx <|end_latent|> spans.

Expects the underlying model to expose `.model` attribute identical to
`AutoModelForCausalLM` behavior (as `MultiCoCo` does).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from .constants import END_LATENT_TOKEN, LATENT_TOKEN, START_LATENT_TOKEN


class LatentWrapper(nn.Module):
    """
    Wrap a causal-LM-style model to perform CoCoNut hidden-state injection.
    
    NOTE: This wrapper directly accesses internal model components such as
    `.model.vision_tower` and `.model.projector`. This is based on the current
    InternVL structure and may break if the architecture changes.
    """

    def __init__(self, base_model: nn.Module, tokenizer):
        """Initialize the LatentWrapper with base model and tokenizer."""
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.latent_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
        self.start_id = tokenizer.convert_tokens_to_ids(START_LATENT_TOKEN)
        self.end_id = tokenizer.convert_tokens_to_ids(END_LATENT_TOKEN)
        self.embedding = base_model.get_input_embeddings()

    def __getattr__(self, name):
        """Delegate unknown attributes to base_model for compatibility."""
        if hasattr(self, '__dict__') and 'base_model' in self.__dict__:
            try:
                return getattr(self.base_model, name)
            except AttributeError:
                pass
        
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Custom generation method that handles latent token injection.
        
        For inputs containing latent spans, uses token-by-token generation.
        For inputs without latent spans, delegates to base model for efficiency.
        """
        # Early return for inputs without latent spans
        if not self._has_latent_spans(input_ids):
            return self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                **kwargs
            )
        
        # Use custom generation loop for latent injection
        return self._generate_with_latent_injection(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            **kwargs
        )

    def _has_latent_spans(self, input_ids: torch.Tensor) -> bool:
        """Check if input contains latent token spans."""
        for ids in input_ids.tolist():
            if self.start_id in ids and self.end_id in ids:
                return True
        return False

    def _generate_with_latent_injection(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Token-by-token generation with latent injection.
        
        NOTE: This does not use KV caching, so it re-evaluates the full
        sequence at each step. KV caching integration would improve performance.
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Set default token IDs
        pad_token_id = (pad_token_id or self.tokenizer.pad_token_id or 
                       self.tokenizer.eos_token_id)
        eos_token_id = eos_token_id or self.tokenizer.eos_token_id
        
        # Cache vision embeddings
        image_embeds = self._get_cached_vision_embeddings(pixel_values, device)
        
        # Initialize generation state
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        generated_ids = input_ids.clone()
        attention_mask = (attention_mask if attention_mask is not None 
                         else torch.ones_like(input_ids))
        
        # Generation loop
        for _ in range(max_new_tokens):
            # Forward pass with latent injection
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    pixel_values=None,
                    image_embeds=image_embeds,
                )
            
            # Get and process logits
            logits = outputs["logits"][:, -1, :]
            logits = self._apply_generation_filters(
                logits, temperature, top_k, top_p
            )
            
            # Sample next token
            next_token_id = self._sample_next_token(logits, do_sample)
            
            # Handle finished sequences
            next_token_id = self._handle_finished_sequences(
                next_token_id, unfinished_sequences, pad_token_id
            )
            
            # Update generation state
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            attention_mask = torch.cat(
                [attention_mask, unfinished_sequences.unsqueeze(-1)], dim=1
            )
            
            # Update unfinished sequences
            if eos_token_id is not None:
                newly_finished = ((next_token_id.squeeze(-1) == eos_token_id) & 
                                (unfinished_sequences == 1))
                unfinished_sequences.mul_((~newly_finished).long())
            
            # Early termination if all sequences finished
            if unfinished_sequences.max() == 0:
                break
        
        return generated_ids
    
    def _get_cached_vision_embeddings(
        self, pixel_values: Optional[torch.Tensor], device: torch.device
    ) -> Optional[torch.Tensor]:
        """Get cached vision embeddings if pixel_values provided."""
        if pixel_values is None:
            return None
            
        with torch.inference_mode():
            vision_embeds = self.base_model.model.vision_tower(
                pixel_values.to(device=device, dtype=self.base_model.model.dtype)
            )
            return self.base_model.model.projector(vision_embeds)
    
    def _apply_generation_filters(
        self, logits: torch.Tensor, temperature: float, top_k: int, top_p: float
    ) -> torch.Tensor:
        """Apply temperature, top_k, and top_p filtering to logits."""
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply top_k filtering
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(
                logits, min(top_k, logits.size(-1))
            )
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(1, top_k_indices, top_k_logits)
        
        # Apply top_p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        return logits
    
    def _sample_next_token(
        self, logits: torch.Tensor, do_sample: bool
    ) -> torch.Tensor:
        """Sample next token from logits."""
        if do_sample:
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        else:
            return torch.argmax(logits, dim=-1, keepdim=True)
    
    def _handle_finished_sequences(
        self, 
        next_token_id: torch.Tensor, 
        unfinished_sequences: torch.Tensor, 
        pad_token_id: int
    ) -> torch.Tensor:
        """Handle finished sequences by using pad_token_id."""
        return (next_token_id * unfinished_sequences.unsqueeze(-1) + 
                pad_token_id * (1 - unfinished_sequences.unsqueeze(-1)))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass with efficient two-pass latent injection.
        
        1. First pass (no grad) gets hidden states for the whole sequence
        2. Replace <|latent|> token embeddings with hidden state from previous token
        3. Second pass produces logits/loss reusing cached image embeddings
        """
        batch_size, seq_len = input_ids.shape
        
        # Extract latent spans
        spans = self._extract_latent_spans(input_ids)
        
        # Fast path for inputs without latent spans
        if not any(spans):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                **kwargs,
            )
        
        # Two-pass latent injection
        image_embeds = self._compute_vision_embeddings(pixel_values, image_embeds)
        last_hidden = self._first_pass_hidden_states(
            input_ids, attention_mask, image_embeds
        )
        inputs_embeds = self._build_modified_embeddings(
            input_ids, spans, last_hidden
        )
        
        return self._second_pass_forward(
            input_ids, attention_mask, inputs_embeds, image_embeds, labels
        )
    
    def _extract_latent_spans(self, input_ids: torch.Tensor) -> list[list[tuple[int, int]]]:
        """Extract latent spans for each batch item."""
        spans = []
        for b in range(input_ids.shape[0]):
            ids = input_ids[b].tolist()
            sample_spans = []
            cur = 0
            
            while True:
                try:
                    start = ids.index(self.start_id, cur)
                    end = ids.index(self.end_id, start + 1)
                    sample_spans.append((start, end))
                    cur = end + 1
                except ValueError:
                    break
            
            spans.append(sample_spans)
        
        return spans
    
    def _compute_vision_embeddings(
        self, 
        pixel_values: Optional[torch.Tensor], 
        image_embeds: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Compute vision embeddings if not provided."""
        if image_embeds is not None:
            return image_embeds
        
        if pixel_values is not None:
            vision_embeds = self.base_model.model.vision_tower(
                pixel_values.to(dtype=self.base_model.model.dtype)
            )
            return self.base_model.model.projector(vision_embeds)
        
        return None
    
    def _first_pass_hidden_states(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor], 
        image_embeds: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """First pass to obtain hidden states."""
        with torch.inference_mode():
            first_pass_embeds = self.base_model.model.prepare_inputs_for_multimodal(
                input_ids=input_ids,
                pixel_values=None,
                image_embeds=image_embeds
            )
            
            first_out = self.base_model.model.language_model(
                inputs_embeds=first_pass_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        
        return first_out.hidden_states[-1]
    
    def _build_modified_embeddings(
        self, 
        input_ids: torch.Tensor, 
        spans: list[list[tuple[int, int]]], 
        last_hidden: torch.Tensor
    ) -> torch.Tensor:
        """Build modified input embeddings with latent token replacement."""
        inputs_embeds = self.embedding(input_ids).clone()
        
        for b, pairs in enumerate(spans):
            for start, end in pairs:
                # Skip if no token before start (need previous token for injection)
                if start == 0:
                    continue
                
                # Replace latent tokens with hidden state from previous token
                span_length = end - start
                inputs_embeds[b, start:end] = (
                    last_hidden[b, start - 1]
                    .unsqueeze(0)
                    .repeat(span_length, 1)
                )
        
        return inputs_embeds
    
    def _second_pass_forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor], 
        inputs_embeds: torch.Tensor, 
        image_embeds: Optional[torch.Tensor], 
        labels: Optional[torch.Tensor]
    ) -> dict:
        """Second pass forward with modified embeddings."""
        second_pass_embeds = self.base_model.model.prepare_inputs_for_multimodal(
            input_ids=input_ids,
            pixel_values=None,
            image_embeds=image_embeds,
            inputs_embeds=inputs_embeds
        )
        
        second_out = self.base_model.model.language_model(
            inputs_embeds=second_pass_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        
        logits = second_out.logits
        loss = None
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
        
        return {"loss": loss, "logits": logits} 