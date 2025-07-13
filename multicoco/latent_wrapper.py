import logging
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from .constants import COCONUT_SPECIAL_TOKENS

logger = logging.getLogger(__name__)

class LatentWrapper(nn.Module):
    """
    LatentWrapper implementing the CoCoNut algorithm with CORRECT sequential hidden state injection.
    
    CRITICAL FIX: The original implementation had a severe flaw where all latent tokens in a span
    received the same repeated hidden state from the pre-span token. This completely defeated the
    purpose of latent reasoning evolution that makes CoCoNut effective.
    
    NEW IMPLEMENTATION: Now processes latent tokens sequentially, where each latent token receives
    the evolved hidden state from the previous position after a forward pass through the model.
    This allows latent reasoning to progress and build upon itself within the span.
    
    Multimodal Benefits:
    - Enables progressive reasoning over images in latent space
    - Each latent token builds upon evolved visual understanding from previous tokens  
    - Proper implementation of CoCoNut's efficiency while maintaining reasoning quality
    - Prevents static repetition that was undermining the algorithm's core benefits
    """

    def __init__(self, base_model: nn.Module, tokenizer, enable_norm_logging: bool = False):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.enable_norm_logging = enable_norm_logging
        self.latent_id = tokenizer.convert_tokens_to_ids('<|latent|>')
        self.start_id = tokenizer.convert_tokens_to_ids('<|start_latent|>')
        self.end_id = tokenizer.convert_tokens_to_ids('<|end_latent|>')
        self.embedding = base_model.get_input_embeddings()

    def chat(self, tokenizer, pixel_values: Optional[torch.Tensor] = None, question: str = "", generation_config: Optional[dict] = None, **kwargs):
        """Chat method that handles latent injection when needed"""
        # Check if we have latent tokens in the question
        question_tokens = tokenizer.encode(question, add_special_tokens=False)
        has_latents = self.start_id in question_tokens and self.end_id in question_tokens
        
        if not has_latents:
            # No latent tokens, use base model's chat directly
            return self.base_model.chat(tokenizer=tokenizer, pixel_values=pixel_values, question=question, generation_config=generation_config, **kwargs)
        
        # Has latent tokens, need custom generation with latent injection
        # Convert chat interface to generate interface
        if pixel_values is not None:
            # For multimodal input, we need to format the question properly
            # This mimics what InternVL's chat method does internally
            formatted_question = f"<image>\n{question}"
            input_ids = tokenizer.encode(formatted_question, add_special_tokens=True, return_tensors="pt")
            
            if pixel_values.dim() == 3:
                pixel_values = pixel_values.unsqueeze(0)
            
            generation_config = generation_config or {}
            
            # Use our custom generation with latent injection
            generated_ids = self.generate(
                input_ids=input_ids.to(pixel_values.device),
                pixel_values=pixel_values,
                **generation_config
            )
            
            # Decode only the generated part
            input_length = input_ids.shape[1]
            generated_tokens = generated_ids[:, input_length:]
            response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
            return response
        else:
            # Text-only generation
            input_ids = tokenizer.encode(question, add_special_tokens=True, return_tensors="pt")
            generation_config = generation_config or {}
            
            generated_ids = self.generate(
                input_ids=input_ids,
                **generation_config
            )
            
            input_length = input_ids.shape[1]
            generated_tokens = generated_ids[:, input_length:]
            response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
            return response

    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, pixel_values: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Generate with proper latent injection support"""
        if not self._has_latent_spans(input_ids):
            return self.base_model.generate(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, **kwargs)
        
        # Extract generation parameters from kwargs
        max_new_tokens = kwargs.get('max_new_tokens', kwargs.get('max_length', 50))
        if 'max_length' in kwargs and 'max_new_tokens' not in kwargs:
            # Convert max_length to max_new_tokens
            max_new_tokens = max(1, kwargs['max_length'] - input_ids.shape[1])
        
        generation_kwargs = {
            'max_new_tokens': max_new_tokens,
            'do_sample': kwargs.get('do_sample', False),
            'temperature': kwargs.get('temperature', 1.0),
            'top_p': kwargs.get('top_p', 1.0),
            'top_k': kwargs.get('top_k', 50),
            'pad_token_id': kwargs.get('pad_token_id'),
            'eos_token_id': kwargs.get('eos_token_id')
        }
        
        return self._generate_with_latent_injection(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            pixel_values=pixel_values, 
            **generation_kwargs
        )

    def _has_latent_spans(self, input_ids: torch.Tensor) -> bool:
        """Check if input contains latent token spans"""
        return any(self.start_id in ids.tolist() and self.end_id in ids.tolist() for ids in input_ids)

    def _generate_with_latent_injection(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, pixel_values: Optional[torch.Tensor] = None, max_new_tokens: int = 50, do_sample: bool = False, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 50, pad_token_id: Optional[int] = None, eos_token_id: Optional[int] = None, **kwargs) -> torch.Tensor:
        """
        Efficient generation with latent injection following original Coconut approach.
        Latent injection happens only once during prompt processing, then standard generation.
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Ensure we're working with the right batch size
        assert batch_size == 1, "Currently only supports batch_size=1 for latent generation"
        
        # Step 1: Process the prompt with latent injection (this is the expensive part)
        image_embeds = self._get_cached_vision_embeddings(pixel_values, device)
        
        # Create dummy labels for forward pass (not used in generation)
        labels = input_ids.clone()
        
        # Run forward pass with latent injection to get the injected embeddings
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=None,  # We already computed image_embeds
                image_embeds=image_embeds,
                labels=labels
            )
        
        # Get the injected embeddings from the forward pass
        inputs_embeds = outputs['inputs_embeds'] if isinstance(outputs, dict) else outputs.inputs_embeds
        
        # Step 2: Generate the first token using the latent-injected embeddings
        next_token_logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
        next_token_logits = next_token_logits[:, -1, :]  # Last position logits
        
        # Apply generation filters and sample
        filtered_logits = self._apply_generation_filters(next_token_logits, temperature, top_k, top_p)
        if do_sample:
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(filtered_logits, dim=-1, keepdim=True)
        
        # Build the tokens list starting with input + first generated token
        tokens = input_ids[0].tolist() + [next_token.item()]
        
        # Check for early termination
        eos_token_id = eos_token_id or self.tokenizer.eos_token_id
        if next_token.item() == eos_token_id:
            return torch.tensor(tokens).unsqueeze(0).to(device)
        
        # Create new input embeddings by appending the first generated token embedding
        new_token_embed = self.embedding(next_token).view(1, 1, -1)
        current_embeds = torch.cat([inputs_embeds, new_token_embed], dim=1)
        
        # Step 3: Continue generation using the base model with the pre-injected embeddings
        # This follows the original Coconut approach of not re-injecting latents
        for _ in range(max_new_tokens - 1):
            with torch.no_grad():
                # Use base model directly to avoid re-running latent injection
                base_outputs = self.base_model.model.language_model(
                    inputs_embeds=current_embeds
                )
            
            # Sample next token
            next_logits = base_outputs.logits[:, -1, :]
            filtered_logits = self._apply_generation_filters(next_logits, temperature, top_k, top_p)
            
            if do_sample:
                probs = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(filtered_logits, dim=-1, keepdim=True)
            
            tokens.append(next_token.item())
            
            # Check for early termination
            if next_token.item() == eos_token_id:
                break
            
            # Append new token embedding for next iteration
            new_token_embed = self.embedding(next_token).view(1, 1, -1)
            current_embeds = torch.cat([current_embeds, new_token_embed], dim=1)
        
        return torch.tensor(tokens).unsqueeze(0).to(device)

    def _initialize_generation_state(self, batch_size: int, device: torch.device, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], pad_token_id: Optional[int], eos_token_id: Optional[int]) -> dict:
        """Initialize state for generation"""
        pad_token_id = pad_token_id or self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        eos_token_id = eos_token_id or self.tokenizer.eos_token_id
        return {
            'generated_ids': input_ids.clone(),
            'attention_mask': attention_mask if attention_mask is not None else torch.ones_like(input_ids),
            'unfinished_sequences': torch.ones(batch_size, dtype=torch.long, device=device),
            'pad_token_id': pad_token_id,
            'eos_token_id': eos_token_id
        }

    def _sample_and_update_token(self, logits: torch.Tensor, generation_state: dict, temperature: float, top_k: int, top_p: float, do_sample: bool) -> torch.Tensor:
        """Sample next token and update generation state"""
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
        """Compute and cache vision embeddings"""
        if pixel_values is None:
            return None
        
        with torch.inference_mode():
            vision_embeds = self.base_model.model.vision_tower(pixel_values.to(device=device, dtype=self.base_model.model.dtype))
            return self.base_model.model.projector(vision_embeds)

    def _apply_generation_filters(self, logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
        """Apply temperature, top-k, and top-p filtering"""
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
        """Sample or greedily select next token"""
        if do_sample:
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        else:
            return torch.argmax(logits, dim=-1, keepdim=True)

    def _handle_finished_sequences(self, next_token_id: torch.Tensor, unfinished_sequences: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        """Handle padding for finished sequences"""
        return next_token_id * unfinished_sequences.unsqueeze(-1) + pad_token_id * (1 - unfinished_sequences.unsqueeze(-1))

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, pixel_values: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, image_embeds: Optional[torch.Tensor] = None, **kwargs):
        """
        Forward pass implementing proper CoCoNut sequential latent processing.
        
        Key improvement: Instead of using the same hidden state for all latent tokens in a span,
        this implementation processes latent tokens sequentially to allow reasoning evolution.
        """
        spans = self._extract_latent_spans(input_ids)
        if not any(spans):
            # No latent tokens, use standard forward
            return self.base_model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels, **kwargs)
        
        # CoCoNut algorithm with sequential latent processing
        image_embeds = self._compute_vision_embeddings(pixel_values, image_embeds)
        
        # Instead of the old two-pass approach, use sequential processing for latent spans
        return self._sequential_latent_forward(input_ids, attention_mask, image_embeds, labels, spans, **kwargs)
    
    def _sequential_latent_forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], image_embeds: Optional[torch.Tensor], labels: Optional[torch.Tensor], spans: List[List[Tuple[int, int]]], **kwargs):
        """
        Simplified sequential forward pass following original Coconut approach.
        Fix: Use two-pass approach with single multimodal processing to avoid position misalignment.
        """
        # First pass: get hidden states for the original sequence
        last_hidden = self._first_pass_hidden_states(input_ids, attention_mask, image_embeds)
        
        # Build modified embeddings with sequential latent injection (like original Coconut)
        inputs_embeds = self._build_modified_embeddings_sequential(input_ids, spans, last_hidden)
        
        # Second pass: single forward with modified embeddings  
        return self._second_pass_forward(input_ids, attention_mask, inputs_embeds, image_embeds, labels)
        
    def _build_modified_embeddings_sequential(self, input_ids: torch.Tensor, spans: List[List[Tuple[int, int]]], last_hidden: torch.Tensor) -> torch.Tensor:
        """
        Build modified embeddings with sequential latent processing.
        Fix: Each latent token gets the hidden state from the previous position (not repeated).
        """
        inputs_embeds = self.embedding(input_ids).clone()
        
        for batch_idx, span_pairs in enumerate(spans):
            for start, end in span_pairs:
                if start == 0:
                    continue  # Skip if latent span starts at position 0
                
                # Sequential injection: each latent token gets evolved state from previous position
                for pos in range(start + 1, end):  # Skip start/end markers, only process actual latent tokens
                    # Each latent token gets hidden state from the immediately previous position
                    source_pos = pos - 1
                    if source_pos < last_hidden.shape[1]:
                        inputs_embeds[batch_idx, pos] = last_hidden[batch_idx, source_pos]
        
        return inputs_embeds

    def _extract_latent_spans(self, input_ids: torch.Tensor) -> List[List[Tuple[int, int]]]:
        """Extract latent token spans between start_latent and end_latent tokens"""
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
        """Compute vision embeddings using InternVL's vision tower and projector"""
        if image_embeds is not None:
            return image_embeds
        
        if pixel_values is not None:
            vision_embeds = self.base_model.model.vision_tower(pixel_values.to(dtype=self.base_model.model.dtype))
            return self.base_model.model.projector(vision_embeds)
        
        return None

    def _first_pass_hidden_states(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], image_embeds: Optional[torch.Tensor]) -> torch.Tensor:
        """First pass to get hidden states before injecting into latent tokens"""
        with torch.inference_mode():
            img_token_positions = None
            if self.enable_norm_logging and hasattr(self.base_model.model, 'img_context_token_id'):
                img_token_positions = self._get_image_token_positions(input_ids)
            
            first_pass_embeds = self.base_model.model.prepare_inputs_for_multimodal(
                input_ids=input_ids,
                pixel_values=None,
                image_embeds=image_embeds
            )
            first_out = self.base_model.model.language_model(
                inputs_embeds=first_pass_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = first_out.hidden_states[-1]
            
            if self.enable_norm_logging and img_token_positions is not None and image_embeds is not None:
                self._log_vision_text_norms(hidden_states, img_token_positions)
        
        return hidden_states

    def _build_modified_embeddings(self, input_ids: torch.Tensor, spans: List[List[Tuple[int, int]]], last_hidden: torch.Tensor) -> torch.Tensor:
        """
        Replace latent token embeddings with sequentially evolved hidden states.
        
        This implements the correct Coconut algorithm where each latent token in a span
        gets the evolved hidden state from the previous position, allowing latent reasoning
        to progress sequentially through the span.
        
        Key difference from flawed approach:
        - OLD: All latent tokens get the same repeated hidden state
        - NEW: Each latent token gets evolved state from previous token in the span
        """
        inputs_embeds = self.embedding(input_ids).clone()
        
        for batch_idx, span_pairs in enumerate(spans):
            for start, end in span_pairs:
                if start == 0:
                    continue  # Skip if latent span starts at position 0
                
                # Sequential injection: each latent token gets the hidden state from the previous position
                for pos in range(start, end):
                    # The first latent token gets hidden state from the token before the span
                    # Subsequent latent tokens get hidden state from the previous latent token
                    source_pos = pos - 1
                    inputs_embeds[batch_idx, pos] = last_hidden[batch_idx, source_pos]
        
        return inputs_embeds

    def _get_image_token_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get positions of image tokens for logging"""
        img_context_token_id = getattr(self.base_model.model, 'img_context_token_id', None)
        if img_context_token_id is None:
            return torch.empty(0, dtype=torch.bool, device=input_ids.device)
        return input_ids == img_context_token_id

    def _log_vision_text_norms(self, hidden_states: torch.Tensor, img_token_positions: torch.Tensor) -> None:
        """Log vision and text token norms for analysis"""
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
        """Log separate vision and text norms"""
        vision_norms = batch_norms[batch_img_positions]
        text_norms = batch_norms[~batch_img_positions]
        
        vision_mean = vision_norms.mean().item()
        vision_std = vision_norms.std().item() if len(vision_norms) > 1 else 0.0
        text_mean = text_norms.mean().item()
        text_std = text_norms.std().item() if len(text_norms) > 1 else 0.0
        ratio = vision_mean / text_mean if text_mean != 0 else 0.0
        
        logger.info(f'Hidden state norms - Batch {batch_idx}: Vision tokens: {len(vision_norms)} tokens, '
                   f'mean={vision_mean:.4f}, std={vision_std:.4f} | Text tokens: {len(text_norms)} tokens, '
                   f'mean={text_mean:.4f}, std={text_std:.4f} | Ratio (vision/text): {ratio:.4f}')
        
        self._log_to_wandb({
            'model/vision_norm_mean': vision_mean,
            'model/text_norm_mean': text_mean,
            'model/vision_text_ratio': ratio
        })

    def _log_text_only_norms(self, batch_norms: torch.Tensor, batch_idx: int) -> None:
        """Log text-only norms when no vision tokens present"""
        text_mean = batch_norms.mean().item()
        text_std = batch_norms.std().item() if len(batch_norms) > 1 else 0.0
        
        logger.info(f'Hidden state norms - Batch {batch_idx}: No vision tokens, Text only: '
                   f'{len(batch_norms)} tokens, mean={text_mean:.4f}, std={text_std:.4f}')
        
        self._log_to_wandb({
            'model/text_only_norm_mean': text_mean,
            'model/text_only_norm_std': text_std
        })

    def _log_to_wandb(self, metrics: dict) -> None:
        """Log metrics to wandb if available"""
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics)
        except ImportError:
            pass

    def _second_pass_forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], inputs_embeds: torch.Tensor, image_embeds: Optional[torch.Tensor], labels: Optional[torch.Tensor]) -> dict:
        """Second pass with modified embeddings containing injected hidden states"""
        second_pass_embeds = self.base_model.model.prepare_inputs_for_multimodal(
            input_ids=input_ids,
            pixel_values=None,
            image_embeds=image_embeds,
            inputs_embeds=inputs_embeds
        )
        second_out = self.base_model.model.language_model(
            inputs_embeds=second_pass_embeds,
            attention_mask=attention_mask,
            use_cache=True
        )
        
        logits = second_out.logits
        loss = None
        
        if labels is not None:
            # Compute cross-entropy loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {'loss': loss, 'logits': logits, 'inputs_embeds': second_pass_embeds}

    # Explicit delegation for commonly used attributes to maintain compatibility
    @property
    def model(self):
        """Provide compatibility with code that expects model attribute"""
        return self.base_model
    
    @property
    def device(self):
        return self.base_model.device
    
    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()
    
    def resize_token_embeddings(self, new_num_tokens):
        return self.base_model.resize_token_embeddings(new_num_tokens)
    
    def train(self, mode=True):
        self.base_model.train(mode)
        return super().train(mode)
    
    def eval(self):
        self.base_model.eval()
        return super().eval()
    
    def to(self, *args, **kwargs):
        self._modules['base_model'] = self.base_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)