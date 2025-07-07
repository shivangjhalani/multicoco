import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor, AutoConfig
import inspect
from collections import namedtuple
# from multicoco.conversation import get_conv_template # No longer needed

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])

class MultiCoCo(nn.Module):
    def __init__(self, model_id, config_id=None, tokenizer_id=None, image_processor_id=None, special_tokens=[], **kwargs):
        super().__init__()
        
        conf_id = config_id if config_id else model_id
        config = AutoConfig.from_pretrained(conf_id, trust_remote_code=True)
        config.attn_implementation = "eager"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        tok_id = tokenizer_id if tokenizer_id else model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=True)
        
        proc_id = image_processor_id if image_processor_id else model_id
        self.image_processor = AutoImageProcessor.from_pretrained(proc_id, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Manually set the image context token ID for the model.
        # This is required for the vanilla model's generate function.
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<img>')
        
        if special_tokens:
            self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            # The attribute might be nested differently depending on the model architecture
            if hasattr(self.model, 'language_model'):
                self.model.language_model.resize_token_embeddings(len(self.tokenizer))
            else:
                self.model.resize_token_embeddings(len(self.tokenizer))

        self.thought_token_id = self.tokenizer.convert_tokens_to_ids('<thought>')
        self.eos_token_id = self.tokenizer.eos_token_id

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(self, input_ids, attention_mask, labels, pixel_values, **kwargs):
        
        latent_indices = (input_ids == self.thought_token_id).nonzero()
        
        if latent_indices.shape[0] == 0:  # No latent tokens, standard forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
                return_dict=True,
                **kwargs,
            )
            return outputs

        inputs_embeds = self.get_input_embeddings()(input_ids)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]
        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, latent_indices[:, 1].min().item())
        kv_cache = None
        logits_list = []

        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                outputs = self.model(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, :next_compute_range[1]],
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                )
                hidden_states_offset = 0
            else:
                past_key_values = [(k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :]) for k, v in kv_cache]
                outputs = self.model(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, :next_compute_range[1]],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )
                hidden_states_offset = next_compute_range[0]
            
            logits_list.append(outputs.logits)
            kv_cache = outputs.past_key_values
            hidden_states = outputs.hidden_states[-1]

            filling_indices = [(i, l[pass_idx]) for i, l in enumerate(latent_lists) if len(l) > pass_idx]
            
            for i, token_idx in filling_indices:
                inputs_embeds[i, token_idx, :] = hidden_states[i, token_idx - 1 - hidden_states_offset, :]

            next_compute_range = (next_compute_range[1], next_compute_range[1] + 1 if pass_idx + 1 < max_n_latents else input_ids.shape[1])

        # Final pass
        past_key_values = [(k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :]) for k, v in kv_cache]
        outputs = self.model(
            inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
            attention_mask=attention_mask[:, :next_compute_range[1]],
            past_key_values=past_key_values,
            output_hidden_states=True,
        )
        logits_list.append(outputs.logits)
        
        logits = torch.cat(logits_list, dim=1)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def generate(self, pixel_values, input_ids, attention_mask, image_flags=None, **kwargs):
        """
        Custom generate function to handle the CoCo methodology.
        This will be called by the Trainer's evaluate method.
        """
        # The base pretrained model's generate function is called directly.
        # It handles the combination of vision and language embeddings internally.
        outputs = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            **kwargs,
        )
        return outputs
