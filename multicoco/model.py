import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoProcessor
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
        # The processor now correctly handles both tokenizer and image processor roles
        self.processor = AutoProcessor.from_pretrained(tok_id, trust_remote_code=True)
        # For this model, the processor IS the tokenizer.
        self.tokenizer = self.processor

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

    def forward(self, **kwargs):
        # These are arguments from our custom data collator
        # that are not expected by the model's forward pass during training.
        kwargs.pop('question_ids', None)
        kwargs.pop('questions', None)
        kwargs.pop('original_questions', None)
        kwargs.pop('answers', None)
        kwargs.pop('num_items_in_batch', None)
        
        # We pass all other arguments to the underlying model.
        return self.model(**kwargs)

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
