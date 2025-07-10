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
        # locate latent spans per sample
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        inputs_embeds = self.embedding(input_ids)

        # Pre-compute the positions of start/end tokens for each sample
        spans = []
        for b in range(batch_size):
            ids = input_ids[b].tolist()
            span_pairs = []
            cur = 0
            while True:
                try:
                    s = ids.index(self.start_id, cur)
                    e = ids.index(self.end_id, s + 1)
                    span_pairs.append((s, e))
                    cur = e + 1
                except ValueError:
                    break
            spans.append(span_pairs)

        if all(len(pairs) == 0 for pairs in spans):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                **kwargs,
            )

        past_kv = None
        logits_chunks = []
        for t in range(seq_len):
            # prepare one-token slice to feed
            tok_embed = inputs_embeds[:, t : t + 1, :]
            attn_slice = attention_mask[:, t : t + 1] if attention_mask is not None else None
            output = self.base_model.model(
                inputs_embeds=tok_embed,
                attention_mask=attention_mask[:, : t + 1] if attention_mask is not None else None,
                past_key_values=past_kv,
                pixel_values=pixel_values if t == 0 else None,
                use_cache=True,
            )
            logits_chunks.append(output.logits)
            past_kv = output.past_key_values

            # if this token is END_LATENT, inject hidden state into the FOLLOWING LATENT token (if any)
            last_hidden = output.hidden_states[-1][:, -1, :]
            for b, pairs in enumerate(spans):
                for s, e in pairs:
                    # replace all latent tokens inside span with the hidden state at (e-1)
                    if t == s - 1:
                        # slice e-s-1 latent tokens
                        inputs_embeds[b, s : e] = last_hidden[b]

        logits = torch.cat(logits_chunks, dim=1)
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