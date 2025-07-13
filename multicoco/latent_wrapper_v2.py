import logging
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Any
logger = logging.getLogger(__name__)

class LatentWrapperV2(nn.Module):

    def __init__(self, model: nn.Module, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.latent_id = self.tokenizer.convert_tokens_to_ids('<|latent|>')
        self.start_id = self.tokenizer.convert_tokens_to_ids('<|start_latent|>')
        self.end_id = self.tokenizer.convert_tokens_to_ids('<|end_latent|>')
        if self.latent_id is None or self.start_id is None or self.end_id is None:
            logger.warning('Some latent tokens not found in tokenizer vocabulary')

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, pixel_values: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, **kwargs):
        spans = self._extract_latent_spans(input_ids)
        if not any(spans):
            return self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels, **kwargs)
        return self._coconut_style_forward(input_ids, attention_mask, pixel_values, labels, spans, **kwargs)

    def _extract_latent_spans(self, input_ids: torch.Tensor) -> List[List[Tuple[int, int]]]:
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

    def _coconut_style_forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], pixel_values: Optional[torch.Tensor], labels: Optional[torch.Tensor], spans: List[List[Tuple[int, int]]], **kwargs):
        image_embeds = self._compute_vision_embeddings(pixel_values)
        last_hidden = self._first_pass_hidden_states(input_ids, attention_mask, image_embeds)
        inputs_embeds = self._build_modified_embeddings(input_ids, spans, last_hidden)
        return self._second_pass_forward(input_ids, attention_mask, inputs_embeds, image_embeds, labels)

    def _second_pass_forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], inputs_embeds: torch.Tensor, image_embeds: Optional[torch.Tensor], labels: Optional[torch.Tensor]) -> dict:
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'prepare_inputs_for_multimodal'):
            second_pass_embeds = self.model.model.prepare_inputs_for_multimodal(input_ids=input_ids, pixel_values=None, image_embeds=image_embeds, inputs_embeds=inputs_embeds)
        else:
            second_pass_embeds = inputs_embeds
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model'):
            second_out = self.model.model.language_model(inputs_embeds=second_pass_embeds, attention_mask=attention_mask, use_cache=True)
            logits = second_out.logits
        else:
            output = self.model(inputs_embeds=second_pass_embeds, attention_mask=attention_mask, labels=labels)
            return output
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {'loss': loss, 'logits': logits}

    def _compute_vision_embeddings(self, pixel_values: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if pixel_values is None:
            return None
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_model'):
            with torch.inference_mode():
                model_dtype = next(self.model.parameters()).dtype
                vision_embeds = self.model.model.vision_model(pixel_values.to(dtype=model_dtype))
                return self.model.model.mlp1(vision_embeds.last_hidden_state)
        return None

    def _first_pass_hidden_states(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], image_embeds: Optional[torch.Tensor]) -> torch.Tensor:
        with torch.inference_mode():
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'prepare_inputs_for_multimodal'):
                first_pass_embeds = self.model.model.prepare_inputs_for_multimodal(input_ids=input_ids, pixel_values=None, image_embeds=image_embeds)
            else:
                first_pass_embeds = self.model.get_input_embeddings()(input_ids)
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model'):
                first_out = self.model.model.language_model(inputs_embeds=first_pass_embeds, attention_mask=attention_mask, output_hidden_states=True)
                return first_out.hidden_states[-1]
            else:
                logger.warning('Cannot access hidden states directly - using embeddings as fallback')
                return first_pass_embeds

    def _build_modified_embeddings(self, input_ids: torch.Tensor, spans: List[List[Tuple[int, int]]], last_hidden: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.model.get_input_embeddings()(input_ids).clone()
        for batch_idx, batch_spans in enumerate(spans):
            for start_pos, end_pos in batch_spans:
                if start_pos == 0:
                    logger.warning(f'Latent span starts at position 0 - skipping injection for batch {batch_idx}')
                    continue
                prev_hidden = last_hidden[batch_idx, start_pos - 1].unsqueeze(0)
                for pos in range(start_pos, end_pos + 1):
                    inputs_embeds[batch_idx, pos] = prev_hidden.squeeze(0)
        return inputs_embeds

    def multimodal_prep(self, input_ids: torch.Tensor, pixel_values: Optional[torch.Tensor]=None, **kwargs):
        image_embeds = self._compute_vision_embeddings(pixel_values)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'prepare_inputs_for_multimodal'):
            return self.model.model.prepare_inputs_for_multimodal(input_ids=input_ids, pixel_values=None, image_embeds=image_embeds, **kwargs)
        else:
            return self.model.get_input_embeddings()(input_ids)

    def latent_injection(self, embeddings: torch.Tensor, input_ids: torch.Tensor):
        spans = self._extract_latent_spans(input_ids)
        if not any(spans):
            return embeddings
        logger.warning('latent_injection called directly - using embeddings as hidden states proxy')
        return self._build_modified_embeddings(input_ids, spans, embeddings)

    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, pixel_values: Optional[torch.Tensor]=None, **kwargs):
        if hasattr(self.model, 'generate'):
            return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, **kwargs)
        else:
            return self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, **kwargs)

    def __getattr__(self, name):
        if name in ['model', 'tokenizer']:
            return super().__getattribute__(name)
        if hasattr(self.model, name):
            return getattr(self.model, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")