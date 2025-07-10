"""LatentWrapper: inject hidden states for <|start_latent|> xxx <|end_latent|> spans.

It expects the underlying model to expose `.model` attribute identical to `AutoModelForCausalLM` behaviour (as `MultiCoCo` does).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from multicoco.constants import LATENT_TOKEN, START_LATENT_TOKEN, END_LATENT_TOKEN

class LatentWrapper(nn.Module):
    """Wrap a causal-LM-style model to perform CoCoNut hidden-state injection."""

    def __init__(self, base_model: nn.Module, tokenizer):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.latent_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
        self.start_id = tokenizer.convert_tokens_to_ids(START_LATENT_TOKEN)
        self.end_id = tokenizer.convert_tokens_to_ids(END_LATENT_TOKEN)

        self.embedding = self.base_model.get_input_embeddings()

    def __getattr__(self, name):
        """Delegate unknown attributes to base_model for compatibility."""
        # Prevent recursion by checking if base_model exists
        if name == 'base_model':
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        # Get base_model using object.__getattribute__ to avoid recursion
        try:
            base_model = object.__getattribute__(self, 'base_model')
            return getattr(base_model, name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Custom generation method that handles latent token injection.
        
        For inputs containing latent spans, we need to do token-by-token generation
        calling our forward method to ensure proper latent injection at each step.
        For inputs without latent spans, we delegate to the base model's generate.
        """
        # Check if input contains latent spans
        has_latent_spans = self._has_latent_spans(input_ids)
        
        if not has_latent_spans:
            # No latent spans, delegate to base model for efficiency
            return self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                **kwargs
            )
        
        # Contains latent spans, use custom generation loop
        return self._generate_with_latent_injection(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            **kwargs
        )

    def _has_latent_spans(self, input_ids: torch.Tensor) -> bool:
        """Check if input contains latent token spans."""
        batch_size = input_ids.shape[0]
        for b in range(batch_size):
            ids = input_ids[b].tolist()
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
        
        This implements a simplified generation loop that calls our forward method
        at each step to ensure latent token injection is properly applied.
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # Set default token IDs
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        
        # Initialize generation state
        generated_ids = input_ids.clone()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Generation loop
        for _ in range(max_new_tokens):
            # Forward pass with latent injection
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values
                )
            
            # Get next token logits
            logits = outputs["logits"][:, -1, :]  # (batch_size, vocab_size)
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top_k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Apply top_p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            
            # Update attention mask
            new_mask_col = torch.ones((batch_size, 1), device=device)
            if attention_mask is not None:
                new_mask_col = new_mask_col.to(dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, new_mask_col], dim=1)
            
            # Check for EOS token (simple check for all sequences)
            if eos_token_id is not None and (next_token_id == eos_token_id).all():
                break
        
        return generated_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass with efficient two-pass latent injection.

        1. First pass (no grad) gets hidden states for the whole sequence.
        2. Replace <|latent|> token embeddings for each span with the hidden
           state of the token *before* the span (same semantics as original).
        3. Second pass produces logits / loss in the usual way.
        """

        batch_size, seq_len = input_ids.shape

        # ------------------------------------------------------------------
        # Locate <|start_latent|> … <|end_latent|> spans per sample
        # ------------------------------------------------------------------
        spans: list[list[tuple[int, int]]] = []  # (start_idx, end_idx)
        for b in range(batch_size):
            ids = input_ids[b].tolist()
            cur = 0
            sample_spans = []
            while True:
                try:
                    s = ids.index(self.start_id, cur)
                    e = ids.index(self.end_id, s + 1)
                    sample_spans.append((s, e))
                    cur = e + 1
                except ValueError:
                    break
            spans.append(sample_spans)

        # Fast path – no latent spans → delegate directly
        if all(len(pairs) == 0 for pairs in spans):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                **kwargs,
            )

        # ------------------------------------------------------------------
        # Pass 1: obtain last hidden state for the whole sequence (no grad)
        # ------------------------------------------------------------------
        with torch.inference_mode():
            first_out = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                output_hidden_states=True,
            )
        last_hidden = first_out.hidden_states[-1]  # (bs, seq_len, hidden)

        # ------------------------------------------------------------------
        # Build modified input embeddings with latent-token replacement
        # ------------------------------------------------------------------
        inputs_embeds = self.embedding(input_ids).clone()
        for b, pairs in enumerate(spans):
            for s, e in pairs:
                # Need a token before <|start_latent|> to copy from
                if s == 0:
                    continue
                # Replace the latent tokens *inside* the span (s … e-1)
                inputs_embeds[b, s:e] = last_hidden[b, s - 1].unsqueeze(0)

        # ------------------------------------------------------------------
        # Pass 2: real forward with modified embeddings (autograd enabled)
        # ------------------------------------------------------------------
        second_out = self.base_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            use_cache=False,
        )
        logits = second_out.logits  # (bs, seq_len, vocab)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
        } 