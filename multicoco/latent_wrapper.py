"""LatentWrapper: inject hidden states for <|start_latent|> xxx <|end_latent|> spans.

It expects the underlying model to expose `.model` attribute identical to `AutoModelForCausalLM` behaviour (as `MultiCoCo` does).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from multicoco.constants import LATENT_TOKEN, START_LATENT_TOKEN, END_LATENT_TOKEN

class LatentWrapper(nn.Module):
    """Wrap a causal-LM-style model to perform CoCoNut hidden-state injection.
    
    NOTE on Model Structure Dependency: This wrapper directly accesses internal
    components of the base model, such as `.model.vision_tower` and 
    `.model.projector`. This is based on the current structure of InternVL and
    may break if the underlying model architecture changes in future versions.
    """

    def __init__(self, base_model: nn.Module, tokenizer):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.latent_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
        self.start_id = tokenizer.convert_tokens_to_ids(START_LATENT_TOKEN)
        self.end_id = tokenizer.convert_tokens_to_ids(END_LATENT_TOKEN)
        self.embedding = base_model.get_input_embeddings()

    def __getattr__(self, name):
        """Delegate unknown attributes to base_model for compatibility."""
        # Only delegate if base_model is properly initialized
        if hasattr(self, '__dict__') and 'base_model' in self.__dict__:
            try:
                return getattr(self.base_model, name)
            except AttributeError:
                pass
        
        # If we get here, the attribute doesn't exist
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
        It caches vision embeddings to avoid re-computation at each step.

        NOTE on KV Caching: This loop does not currently use past_key_values (KV cache),
        so it re-evaluates the full sequence at each step. Integrating KV caching is
        non-trivial due to the two-pass latent injection but would significantly
        improve performance.
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # Set default token IDs
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        
        # Cache vision embeddings once before the loop
        # NOTE on Dynamic Vision: This assumes vision embeddings are static. If the
        # model uses dynamic patching based on text, this caching could be invalid.
        image_embeds = None
        if pixel_values is not None:
            with torch.inference_mode():
                image_embeds = self.base_model.model.vision_tower(pixel_values.to(device=device, dtype=self.base_model.model.dtype))
                image_embeds = self.base_model.model.projector(image_embeds)

        # Initialize generation state
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        generated_ids = input_ids.clone()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Generation loop
        for _ in range(max_new_tokens):
            # Forward pass with latent injection using cached vision embeddings
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    pixel_values=None,  # Pass None to avoid re-computation
                    image_embeds=image_embeds, # Pass cached embeds
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
            
            # Use pad_token_id for finished sequences
            next_token_id = next_token_id * unfinished_sequences.unsqueeze(-1) + pad_token_id * (1 - unfinished_sequences.unsqueeze(-1))

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            
            # Update attention mask
            attention_mask = torch.cat([attention_mask, unfinished_sequences.unsqueeze(-1)], dim=1)
            
            # Update unfinished sequences
            if eos_token_id is not None:
                newly_finished = (next_token_id.squeeze(-1) == eos_token_id) & (unfinished_sequences == 1)
                unfinished_sequences.mul_((~newly_finished).long())

            # Check for EOS token (stop when all sequences are finished)
            if unfinished_sequences.max() == 0:
                break
        
        return generated_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass with efficient two-pass latent injection.

        It avoids re-computing vision embeddings by caching them.

        1. First pass (no grad) gets hidden states for the whole sequence.
           If pixel_values are provided, it also computes and caches image_embeds.
        2. Replace <|latent|> token embeddings for each span with the hidden
           state of the token *before* the span (same semantics as original).
        3. Second pass produces logits / loss in the usual way, reusing the
           cached image_embeds if available.
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
        # Pass 1: obtain last hidden state and vision embeddings (no grad)
        # ------------------------------------------------------------------
        with torch.inference_mode():
            # If image_embeds are not provided, compute them from pixel_values
            if image_embeds is None and pixel_values is not None:
                vision_embeds = self.base_model.model.vision_tower(pixel_values.to(dtype=self.base_model.model.dtype))
                image_embeds = self.base_model.model.projector(vision_embeds)
            
            # Prepare inputs for the language model part
            first_pass_embeds = self.base_model.model.prepare_inputs_for_multimodal(
                input_ids=input_ids,
                pixel_values=None, # We use pre-computed image_embeds
                image_embeds=image_embeds
            )
            
            first_out = self.base_model.model.language_model(
                inputs_embeds=first_pass_embeds,
                attention_mask=attention_mask,
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
                
                # NOTE: This assumes the token at `s-1` is a text token. If the
                # prompt format changes such that `s-1` could be an image token,
                # this injection logic may become unstable.
                # Sequentially replace the latent tokens *inside* the span (s … e-1)
                # with the hidden states from the previous tokens. This creates a
                # "latent reasoning chain" where each step builds on the last.
                for i in range(s, e):
                    inputs_embeds[b, i] = last_hidden[b, i - 1]

        # ------------------------------------------------------------------
        # Pass 2: real forward with modified embeddings and cached vision embeds
        # ------------------------------------------------------------------
        # Prepare inputs again, this time with the modified text embeddings
        second_pass_embeds = self.base_model.model.prepare_inputs_for_multimodal(
            input_ids=input_ids,
            pixel_values=None, # Reuse cached image_embeds
            image_embeds=image_embeds,
            inputs_embeds=inputs_embeds # Provide the modified embeddings
        )

        second_out = self.base_model.model.language_model(
            inputs_embeds=second_pass_embeds,
            attention_mask=attention_mask,
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