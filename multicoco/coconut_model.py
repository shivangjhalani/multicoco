"""
CoCoNut Model for Multimodal Latent Space Reasoning.

This module defines the CoCoNutModel class, which wraps a pretrained
multimodal model (like InternVL) to enable true latent space reasoning
by manually orchestrating the forward pass and injecting hidden states.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from transformers import PreTrainedModel, PreTrainedTokenizer

from .constants import IMAGE_TOKEN, LOSS_IGNORE_INDEX

class CoCoNutModel(nn.Module):
    """
    A wrapper model that implements the CoCoNuT (Chain of Continuous Thought)
    methodology for a multimodal model.

    This model manually controls the forward pass to:
    1. Process the visual input once.
    2. Fuse visual and textual embeddings.
    3. Iteratively generate "thoughts" by injecting the model's own
       hidden states back into its input stream at designated <latent>
       token positions.
    """
    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        latent_token_id: int,
    ):
        """
        Initializes the CoCoNutModel.

        Args:
            base_model: The pretrained multimodal model to wrap.
            tokenizer: The tokenizer associated with the model.
            latent_token_id: The ID of the special <latent> token.
        """
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.latent_token_id = latent_token_id
        # A simple check to ensure the base model has the required components
        if not all(hasattr(self.base_model, attr) for attr in ['vision_model', 'language_model', 'mlp1']):
            raise ValueError("The provided base_model is not compatible. It must have 'vision_model', 'language_model', and 'mlp1' attributes.")

    def _prepare_multimodal_embeds(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Processes the image and fuses it with text embeddings.
        This is done once at the beginning of the forward pass.
        """
        # 1. Get text embeddings
        text_embeds = self.base_model.language_model.get_input_embeddings()(input_ids)

        # 2. Process image through vision tower and projector
        image_features = self.base_model.vision_model(pixel_values, output_hidden_states=True)
        image_embeds = self.base_model.mlp1(image_features.hidden_states[self.base_model.config.vision_select_layer])

        # 3. Find image token positions and replace with image embeds
        image_token_mask = (input_ids == self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN))
        
        # Ensure the mask is on the same device as text_embeds
        image_token_mask = image_token_mask.to(text_embeds.device)

        # Use the mask to create a tensor of indices
        image_token_indices = image_token_mask.nonzero(as_tuple=True)[0]
        
        # Check if the number of image tokens matches the number of image embeddings
        if len(image_token_indices) != image_embeds.shape[0] * image_embeds.shape[1]:
             # This can happen in a multi-image setup, which we simplify for now.
             # We assume one image per sample.
             batch_size = text_embeds.shape[0]
             # Reshape to (batch_size, num_patches, hidden_size)
             image_embeds = image_embeds.view(batch_size, -1, image_embeds.shape[-1])
             
             # Locate where to insert the image embeddings for each sample in the batch
             for i in range(batch_size):
                 sample_mask = (input_ids[i] == self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN))
                 # The text_embeds for this sample will be updated in place
                 text_embeds[i, sample_mask] = image_embeds[i]
        else:
             # Simpler case for single image, single sample in batch
             text_embeds[image_token_mask] = image_embeds.view(-1, image_embeds.shape[-1])

        return text_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Performs the CoCoNuT forward pass with state injection.

        The process is as follows:
        1.  The image and text inputs are processed once to create a single
            `inputs_embeds` tensor containing the fused multimodal features.
        2.  The code identifies all positions of the `<latent>` token.
        3.  It then iterates through the sequence in segments, divided by the
            latent tokens.
        4.  For each segment, it performs a forward pass through the language
            model, using the KV cache (`past_key_values`) for efficiency.
        5.  After processing a segment, it captures the hidden state of the
            very last token in that segment. This hidden state represents
            the model's "thought".
        6.  This "thought vector" is then physically injected into the
            `inputs_embeds` tensor, replacing the embedding of the upcoming
            `<latent>` token.
        7.  This process repeats until all latent steps are completed.
        8.  Finally, it processes the remainder of the sequence and computes
            a single loss value across all the generated logits.
        """
        # 1. Prepare combined multimodal embeddings once.
        inputs_embeds = self._prepare_multimodal_embeds(input_ids, pixel_values, attention_mask)
        
        # 2. Find all latent token positions
        latent_token_mask = (input_ids == self.latent_token_id)
        latent_indices = latent_token_mask.nonzero(as_tuple=True)[1]
        
        # If no latent tokens, perform a standard forward pass
        if len(latent_indices) == 0:
            return self.base_model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
            
        # 3. Iterative forward pass with state injection
        all_logits = []
        past_key_values = None
        current_pos = 0

        for i, latent_pos in enumerate(latent_indices):
            # a. Process the segment up to the latent token
            segment_embeds = inputs_embeds[:, current_pos:latent_pos, :]
            segment_attention_mask = attention_mask[:, current_pos:latent_pos] if attention_mask is not None else None
            
            outputs = self.base_model.language_model(
                inputs_embeds=segment_embeds,
                attention_mask=segment_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            # b. Capture the hidden state of the token *before* the latent token
            # The last hidden state is the "thought" we will inject.
            thought_vector = outputs.hidden_states[-1][:, -1, :]
            
            # c. Inject the thought vector as the embedding for the latent token
            # This is the core of the CoCoNuT method.
            inputs_embeds[:, latent_pos, :] = thought_vector
            
            # d. Store logits and update KV cache for the next iteration
            all_logits.append(outputs.logits)
            past_key_values = outputs.past_key_values
            current_pos = latent_pos

        # e. Process the final segment after the last latent token
        final_segment_embeds = inputs_embeds[:, current_pos:, :]
        final_segment_attention_mask = attention_mask[:, current_pos:] if attention_mask is not None else None
        
        final_outputs = self.base_model.language_model(
            inputs_embeds=final_segment_embeds,
            attention_mask=final_segment_attention_mask,
            past_key_values=past_key_values,
        )
        all_logits.append(final_outputs.logits)
        
        # 4. Concatenate logits and compute loss
        logits = torch.cat(all_logits, dim=1)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=LOSS_IGNORE_INDEX)
            loss = loss_fct(shift_logits.view(-1, self.base_model.config.text_config.vocab_size), shift_labels.view(-1))
            
        return {
            "loss": loss,
            "logits": logits,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        """
        Handles generation for the CoCoNutModel, mirroring the `forward` pass logic.

        This custom generation method is necessary because the default `generate`
        function cannot handle our manual state injection. It iteratively
        processes segments of the input, injects thought vectors, and uses the
        base model's generate function only for the final answer segment.
        """
        # 1. Prepare combined multimodal embeddings once.
        inputs_embeds = self._prepare_multimodal_embeds(input_ids, pixel_values, attention_mask)

        # 2. Find all latent token positions
        latent_token_mask = (input_ids == self.latent_token_id)
        latent_indices = latent_token_mask.nonzero(as_tuple=True)[1]

        # If no latent tokens, use the standard generation method
        if len(latent_indices) == 0:
            return self.base_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generation_kwargs
            )
        
        # 3. Iterative generation with state injection
        past_key_values = None
        generated_sequence = input_ids[:, :1].clone() # Start with BOS token

        current_pos = 0
        for i, latent_pos in enumerate(latent_indices):
            # a. Process segment up to the latent token
            segment_input_ids = input_ids[:, current_pos:latent_pos]
            segment_attention_mask = attention_mask[:, current_pos:latent_pos] if attention_mask is not None else None
            
            # Note: We pass input_ids here, not embeds, as we need the discrete tokens
            # for the base model's internal processing during generation.
            outputs = self.base_model.language_model(
                input_ids=segment_input_ids,
                attention_mask=segment_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            # b. Capture the hidden state and inject it
            thought_vector = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            
            # We need to run a single forward step with the thought vector
            # to get the next set of past_key_values. This simulates the model
            # "processing" its own thought.
            thought_outputs = self.base_model.language_model(
                inputs_embeds=thought_vector,
                past_key_values=outputs.past_key_values,
                use_cache=True
            )
            past_key_values = thought_outputs.past_key_values
            current_pos = latent_pos + 1 # Move past the latent token

        # c. Generate the final answer after the last latent token
        # We start generation from the last known position
        final_generation_ids = self.base_model.language_model.generate(
            input_ids=input_ids[:, current_pos:],
            attention_mask=attention_mask[:, current_pos:] if attention_mask is not None else None,
            past_key_values=past_key_values,
            **generation_kwargs
        )
        
        return final_generation_ids 