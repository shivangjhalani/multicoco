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