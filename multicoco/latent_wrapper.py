import logging
from typing import Optional
import torch
import torch.nn as nn
from .constants import END_LATENT_TOKEN, LATENT_TOKEN, START_LATENT_TOKEN
logger = logging.getLogger(__name__)

class LatentWrapper(nn.Module):

    def __init__(self, base_model: nn.Module, tokenizer, enable_norm_logging: bool=False):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.enable_norm_logging = enable_norm_logging
        self.latent_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
        self.start_id = tokenizer.convert_tokens_to_ids(START_LATENT_TOKEN)
        self.end_id = tokenizer.convert_tokens_to_ids(END_LATENT_TOKEN)
        self.embedding = base_model.get_input_embeddings()

    def __getattr__(self, name):
        if hasattr(self, '__dict__') and 'base_model' in self.__dict__:
            try:
                return getattr(self.base_model, name)
            except AttributeError:
                pass
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, pixel_values: Optional[torch.Tensor]=None, **kwargs) -> torch.Tensor:
        if not self._has_latent_spans(input_ids):
            return self.base_model.generate(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, **kwargs)
        return self._generate_with_latent_injection(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, **kwargs)

    def _has_latent_spans(self, input_ids: torch.Tensor) -> bool:
        return any((self.start_id in ids.tolist() and self.end_id in ids.tolist() for ids in input_ids))

    def _generate_with_latent_injection(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, pixel_values: Optional[torch.Tensor]=None, max_new_tokens: int=50, do_sample: bool=False, temperature: float=1.0, top_p: float=1.0, top_k: int=50, pad_token_id: Optional[int]=None, eos_token_id: Optional[int]=None, **kwargs) -> torch.Tensor:
        device = input_ids.device
        batch_size = input_ids.shape[0]
        generation_state = self._initialize_generation_state(batch_size, device, input_ids, attention_mask, pad_token_id, eos_token_id)
        image_embeds = self._get_cached_vision_embeddings(pixel_values, device)
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.forward(input_ids=generation_state['generated_ids'], attention_mask=generation_state['attention_mask'], pixel_values=None, image_embeds=image_embeds)
            next_token = self._sample_and_update_token(outputs['logits'], generation_state, temperature, top_k, top_p, do_sample)
            if generation_state['unfinished_sequences'].max() == 0:
                break
        return generation_state['generated_ids']

    def _initialize_generation_state(self, batch_size: int, device: torch.device, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], pad_token_id: Optional[int], eos_token_id: Optional[int]) -> dict:
        pad_token_id = pad_token_id or self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        eos_token_id = eos_token_id or self.tokenizer.eos_token_id
        return {'generated_ids': input_ids.clone(), 'attention_mask': attention_mask if attention_mask is not None else torch.ones_like(input_ids), 'unfinished_sequences': torch.ones(batch_size, dtype=torch.long, device=device), 'pad_token_id': pad_token_id, 'eos_token_id': eos_token_id}

    def _sample_and_update_token(self, logits: torch.Tensor, generation_state: dict, temperature: float, top_k: int, top_p: float, do_sample: bool) -> torch.Tensor:
        current_logits = logits[:, -1, :]
        current_logits = self._apply_generation_filters(current_logits, temperature, top_k, top_p)
        next_token_id = self._sample_next_token(current_logits, do_sample)
        next_token_id = self._handle_finished_sequences(next_token_id, generation_state['unfinished_sequences'], generation_state['pad_token_id'])
        generation_state['generated_ids'] = torch.cat([generation_state['generated_ids'], next_token_id], dim=1)
        generation_state['attention_mask'] = torch.cat([generation_state['attention_mask'], generation_state['unfinished_sequences'].unsqueeze(-1)], dim=1)
        if generation_state['eos_token_id'] is not None:
            newly_finished = (next_token_id.squeeze(-1) == generation_state['eos_token_id']) & (generation_state['unfinished_sequences'] == 1)
            generation_state['unfinished_sequences'].mul_((~newly_finished).long())
        return next_token_id

    def _get_cached_vision_embeddings(self, pixel_values: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
        if pixel_values is None:
            return None
        with torch.inference_mode():
            vision_embeds = self.base_model.model.vision_tower(pixel_values.to(device=device, dtype=self.base_model.model.dtype))
            return self.base_model.model.projector(vision_embeds)

    def _apply_generation_filters(self, logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
        if temperature != 1.0:
            logits = logits / temperature
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(1, top_k_indices, top_k_logits)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        return logits

    def _sample_next_token(self, logits: torch.Tensor, do_sample: bool) -> torch.Tensor:
        if do_sample:
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(logits, dim=-1, keepdim=True)

    def _handle_finished_sequences(self, next_token_id: torch.Tensor, unfinished_sequences: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        return next_token_id * unfinished_sequences.unsqueeze(-1) + pad_token_id * (1 - unfinished_sequences.unsqueeze(-1))

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, pixel_values: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, image_embeds: Optional[torch.Tensor]=None, **kwargs):
        spans = self._extract_latent_spans(input_ids)
        if not any(spans):
            return self.base_model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels, **kwargs)
        image_embeds = self._compute_vision_embeddings(pixel_values, image_embeds)
        last_hidden = self._first_pass_hidden_states(input_ids, attention_mask, image_embeds)
        inputs_embeds = self._build_modified_embeddings(input_ids, spans, last_hidden)
        return self._second_pass_forward(input_ids, attention_mask, inputs_embeds, image_embeds, labels)

    def _extract_latent_spans(self, input_ids: torch.Tensor) -> list[list[tuple[int, int]]]:
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

    def _compute_vision_embeddings(self, pixel_values: Optional[torch.Tensor], image_embeds: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if image_embeds is not None:
            return image_embeds
        if pixel_values is not None:
            vision_embeds = self.base_model.model.vision_tower(pixel_values.to(dtype=self.base_model.model.dtype))
            return self.base_model.model.projector(vision_embeds)
        return None

    def _first_pass_hidden_states(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], image_embeds: Optional[torch.Tensor]) -> torch.Tensor:
        with torch.inference_mode():
            img_token_positions = None
            if self.enable_norm_logging and hasattr(self.base_model.model, 'img_context_token_id'):
                img_token_positions = self._get_image_token_positions(input_ids)
            first_pass_embeds = self.base_model.model.prepare_inputs_for_multimodal(input_ids=input_ids, pixel_values=None, image_embeds=image_embeds)
            first_out = self.base_model.model.language_model(inputs_embeds=first_pass_embeds, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = first_out.hidden_states[-1]
            if self.enable_norm_logging and img_token_positions is not None and (image_embeds is not None):
                self._log_vision_text_norms(hidden_states, img_token_positions)
        return hidden_states

    def _build_modified_embeddings(self, input_ids: torch.Tensor, spans: list[list[tuple[int, int]]], last_hidden: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.embedding(input_ids).clone()
        for batch_idx, span_pairs in enumerate(spans):
            for start, end in span_pairs:
                if start == 0:
                    continue
                span_length = end - start
                inputs_embeds[batch_idx, start:end] = last_hidden[batch_idx, start - 1].unsqueeze(0).repeat(span_length, 1)
        return inputs_embeds

    def _get_image_token_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        img_context_token_id = getattr(self.base_model.model, 'img_context_token_id', None)
        if img_context_token_id is None:
            return torch.empty(0, dtype=torch.bool, device=input_ids.device)
        return input_ids == img_context_token_id

    def _log_vision_text_norms(self, hidden_states: torch.Tensor, img_token_positions: torch.Tensor) -> None:
        try:
            batch_size, seq_len, hidden_size = hidden_states.shape
            token_norms = torch.norm(hidden_states, p=2, dim=-1)
            for batch_idx in range(batch_size):
                batch_img_positions = img_token_positions[batch_idx]
                batch_norms = token_norms[batch_idx]
                if batch_img_positions.any():
                    self._log_vision_and_text_norms(batch_norms, batch_img_positions, batch_idx)
                else:
                    self._log_text_only_norms(batch_norms, batch_idx)
        except Exception as e:
            logger.warning(f'Failed to log vision-text norms: {e}')

    def _log_vision_and_text_norms(self, batch_norms: torch.Tensor, batch_img_positions: torch.Tensor, batch_idx: int) -> None:
        vision_norms = batch_norms[batch_img_positions]
        text_norms = batch_norms[~batch_img_positions]
        vision_mean = vision_norms.mean().item()
        vision_std = vision_norms.std().item() if len(vision_norms) > 1 else 0.0
        text_mean = text_norms.mean().item()
        text_std = text_norms.std().item() if len(text_norms) > 1 else 0.0
        ratio = vision_mean / text_mean if text_mean != 0 else 0.0
        logger.info(f'Hidden state norms - Batch {batch_idx}: Vision tokens: {len(vision_norms)} tokens, mean={vision_mean:.4f}, std={vision_std:.4f} | Text tokens: {len(text_norms)} tokens, mean={text_mean:.4f}, std={text_std:.4f} | Ratio (vision/text): {ratio:.4f}')
        self._log_to_wandb({'model/vision_norm_mean': vision_mean, 'model/text_norm_mean': text_mean, 'model/vision_text_ratio': ratio})

    def _log_text_only_norms(self, batch_norms: torch.Tensor, batch_idx: int) -> None:
        text_mean = batch_norms.mean().item()
        text_std = batch_norms.std().item() if len(batch_norms) > 1 else 0.0
        logger.info(f'Hidden state norms - Batch {batch_idx}: No vision tokens, Text only: {len(batch_norms)} tokens, mean={text_mean:.4f}, std={text_std:.4f}')
        self._log_to_wandb({'model/text_only_norm_mean': text_mean, 'model/text_only_norm_std': text_std})

    def _log_to_wandb(self, metrics: dict) -> None:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics)
        except ImportError:
            pass

    def _second_pass_forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], inputs_embeds: torch.Tensor, image_embeds: Optional[torch.Tensor], labels: Optional[torch.Tensor]) -> dict:
        second_pass_embeds = self.base_model.model.prepare_inputs_for_multimodal(input_ids=input_ids, pixel_values=None, image_embeds=image_embeds, inputs_embeds=inputs_embeds)
        second_out = self.base_model.model.language_model(inputs_embeds=second_pass_embeds, attention_mask=attention_mask, use_cache=True)
        logits = second_out.logits
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {'loss': loss, 'logits': logits}