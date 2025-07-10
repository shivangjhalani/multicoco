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

        # ------------------------------------------------------------------
        # Optimised chunk-based forward (mirrors Meta CoCoNut reference)
        # ------------------------------------------------------------------

        # Collect positions of LATENT_TOKEN (one per token inside spans)
        latent_pos = (input_ids == self.latent_id).nonzero(as_tuple=False)
        latent_lists = [
            [idx[1].item() for idx in latent_pos if idx[0] == b]
            for b in range(batch_size)
        ]

        max_n_latents = max(len(l) for l in latent_lists)

        # Determine first compute range (prefix up to earliest latent across batch)
        if max_n_latents > 0:
            first_latent_idx = latent_pos[:, 1].min().item()
            next_compute_range = (0, first_latent_idx)  # [start, end)
        else:
            next_compute_range = (0, seq_len)

        past_kv = None
        logits_chunks = []

        # We'll perform `max_n_latents + 1` passes: prefix + one per latent + suffix
        for pass_idx in range(max_n_latents + 1):

            slice_start, slice_end = next_compute_range

            # Skip empty slices (can happen when multiple sequences share prefix length 0)
            if slice_start == slice_end:
                # prepare next_compute_range for upcoming iteration and continue
                if pass_idx < max_n_latents:
                    next_compute_range = (slice_end, slice_end + 1) if slice_end + 1 <= seq_len else (slice_end, slice_end)
                continue

            outputs = self.base_model.model(
                inputs_embeds=inputs_embeds[:, slice_start:slice_end, :],
                attention_mask=attention_mask[:, :slice_end] if attention_mask is not None else None,
                past_key_values=past_kv,
                pixel_values=pixel_values if past_kv is None else None,  # only feed images once
                use_cache=True,
                output_hidden_states=True,
            )

            logits_chunks.append(outputs.logits)
            past_kv = outputs.past_key_values

            # After computing up to (and including) token slice_end-1, we may need to
            # inject hidden state into latent tokens whose index == slice_end
            if pass_idx < max_n_latents:
                hidden_last = outputs.hidden_states[-1][:, -1, :]  # (bs, hidden)

                # For every sample that still has a latent token at position `slice_end`
                for b in range(batch_size):
                    if len(latent_lists[b]) > pass_idx:
                        token_idx = latent_lists[b][pass_idx]
                        # token_idx should equal slice_end in most cases; use guard
                        if slice_end <= token_idx < seq_len:
                            inputs_embeds[b, token_idx] = hidden_last[b]

                # Next slice is exactly the latent token we just processed
                next_compute_range = (slice_end, slice_end + 1)
            else:
                # Final suffix: process rest of sequence in last iteration
                next_compute_range = (slice_end, seq_len)

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