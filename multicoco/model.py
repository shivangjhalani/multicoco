import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor, AutoConfig
import inspect
from collections import namedtuple
from multicoco.conversation import get_conv_template

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
        
        inputs_embeds = self.get_input_embeddings()(input_ids)
        latent_indices = (input_ids == self.thought_token_id).nonzero()
        
        if latent_indices.shape[0] == 0: # No latent tokens, standard forward pass
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
                return_dict=True
            )
            return Outputs(loss=outputs.loss, inputs_embeds=inputs_embeds, logits=outputs.logits)

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

    def batch_chat(self, pixel_values, questions, generation_config):
        """
        Handles batch chatting with the model for vanilla/CoT modes.
        """
        num_image_tokens = self.model.config.num_image_token
        
        prompts = []
        for q in questions:
            question_with_img = '<img>' * num_image_tokens + '\n' + q
            conv = get_conv_template(self.model.conv_template)
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            conv.append_message(roles["human"], question_with_img)
            conv.append_message(roles["gpt"], None)
            prompts.append(conv.get_prompt())

        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        inputs['pixel_values'] = pixel_values.to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)

        responses = [self.tokenizer.decode(output, skip_special_tokens=True).split(conv.roles[1] + ':')[-1].strip() for output in outputs]
        return responses

    def generate_coconut(self, pixel_values, questions, generation_config):
        # This is a simplified single-instance generate for clarity. Batching needs more work.
        if len(questions) > 1:
            # Note: This is a placeholder. True batch generation for this model is complex.
            # For now, we process one by one.
            responses = []
            for i in range(len(questions)):
                responses.extend(self.generate_coconut(pixel_values[i:i+1], [questions[i]], generation_config))
            return responses

        num_image_tokens = self.model.config.num_image_token
        c_thought = generation_config.get('c_thought', 1)
        
        question_with_thoughts = (
            '<img>' * num_image_tokens + '\n' +
            questions[0] +
            ' ' +
            '<start_thought>' +
            '<thought>' * c_thought +
            '<end_thought>'
        )
        
        conv = get_conv_template(self.model.conv_template)
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conv.append_message(roles["human"], question_with_thoughts)
        conv.append_message(roles["gpt"], None)
        prompt = conv.get_prompt()

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        
        # Forward pass to get thought-out embeddings
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=torch.full_like(input_ids, -100), # Dummy labels
            pixel_values=pixel_values
        )
        
        # Generate from the final embeddings
        final_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(final_logits, dim=-1).unsqueeze(0)
        
        generated_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Simple autoregressive generation loop
        for _ in range(generation_config.get('max_new_tokens', 100) - 1):
            with torch.no_grad():
                outputs = self.model(input_ids=generated_ids, pixel_values=pixel_values)
            
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            if next_token.item() in [self.eos_token_id, self.tokenizer.convert_tokens_to_ids('<|im_end|>')]:
                break
                
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        response = response.split(conv.roles[1] + ':')[-1].strip()
        return [response]
