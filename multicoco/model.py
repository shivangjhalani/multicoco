import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor, AutoConfig
import inspect

class MultiCoCo(nn.Module):
    def __init__(self, model_id, config_id=None, tokenizer_id=None, image_processor_id=None, latent_tokens={}, special_tokens=[]):
        super().__init__()
        
        # Determine the ID to use for loading the configuration.
        # Default to model_id if no specific config_id is provided.
        conf_id = config_id if config_id else model_id
        
        # Load config, force eager attention, and then load the model with this config.
        # This is the most reliable way to prevent Flash Attention errors.
        config = AutoConfig.from_pretrained(conf_id, trust_remote_code=True)
        config.attn_implementation = "eager"

        # Load the model itself using model_id and the modified config
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # Load tokenizer and processor using their specific IDs, or default to model_id
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
            num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            if num_added_tokens > 0:
                # The attribute might be nested differently depending on the model architecture
                if hasattr(self.model, 'language_model'):
                    self.model.language_model.resize_token_embeddings(len(self.tokenizer))
                else:
                    self.model.resize_token_embeddings(len(self.tokenizer))


        self.latent_tokens = latent_tokens

    def forward(self, pixel_values, input_ids, attention_mask, labels, image_flags):
        # The vanilla model's forward pass doesn't accept image_flags.
        # We can inspect the model's forward signature to pass it conditionally.
        model_forward_params = inspect.signature(self.model.forward).parameters
        if 'image_flags' in model_forward_params:
            output = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                image_flags=image_flags,
                return_dict=True
            )
        else:
            output = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
        return output

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config):
        """
        Handles batch chatting with the model.
        """
        # Manually build the prompts for the batch
        prompts = [f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {q} ASSISTANT:" for q in questions]

        # Tokenize the batch of prompts
        inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Add pixel values to the inputs
        inputs['pixel_values'] = pixel_values.to(self.model.device)
        
        # Generate responses for the batch
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)

        # Decode the generated responses
        responses = [tokenizer.decode(output, skip_special_tokens=True).split("ASSISTANT:")[-1].strip() for output in outputs]
        
        return responses
