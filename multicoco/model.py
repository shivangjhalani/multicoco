import contextlib
import logging
from typing import Any, Dict, List, Optional
import torch
from torch import nn
from transformers import AutoConfig, AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer
from .constants import DEFAULT_DTYPE, DEFAULT_MODEL_NAME, IMAGE_TOKEN, IMG_CONTEXT_TOKEN
from .exceptions import DtypeMismatchError, ModelInitializationError
logger = logging.getLogger(__name__)

@contextlib.contextmanager
def suppress_internvl_messages():
    import builtins
    original_print = builtins.print
    suppress_phrases = ['dynamic ViT batch size:', 'warning: The size of tensor a', 'input_embeds[selected].shape=', 'vit_embeds.shape=']

    def filtered_print(*args, **kwargs):
        message = ' '.join((str(arg) for arg in args))
        if not any((phrase in message for phrase in suppress_phrases)):
            original_print(*args, **kwargs)
    builtins.print = filtered_print
    try:
        yield
    finally:
        builtins.print = original_print

class MultiCoCo(nn.Module):

    def __init__(self, model_id: str=DEFAULT_MODEL_NAME, config_id: Optional[str]=None, tokenizer_id: Optional[str]=None, image_processor_id: Optional[str]=None, special_tokens: Optional[List[str]]=None, torch_dtype: str=DEFAULT_DTYPE, trust_remote_code: bool=True, low_cpu_mem_usage: bool=True, **kwargs) -> None:
        super().__init__()
        special_tokens = special_tokens or []
        try:
            self.model, self.tokenizer, self.image_processor = self._initialize_components(model_id, config_id, tokenizer_id, image_processor_id, special_tokens, torch_dtype, trust_remote_code, low_cpu_mem_usage)
            self._resize_special_token_embeddings()
            self._setup_special_tokens()
        except Exception as e:
            raise ModelInitializationError(f'Failed to initialize MultiCoCo model: {e}') from e
        param_count = sum((p.numel() for p in self.model.parameters()))
        logger.info(f'MultiCoCo model initialized with {param_count} parameters')

    def _initialize_components(self, model_id: str, config_id: Optional[str], tokenizer_id: Optional[str], image_processor_id: Optional[str], special_tokens: List[str], torch_dtype: str, trust_remote_code: bool, low_cpu_mem_usage: bool) -> tuple[nn.Module, AutoTokenizer, AutoImageProcessor]:
        model = self._create_model(model_id, config_id, torch_dtype, trust_remote_code, low_cpu_mem_usage)
        tokenizer = self._create_tokenizer(tokenizer_id or model_id, special_tokens)
        image_processor = AutoImageProcessor.from_pretrained(image_processor_id or model_id, trust_remote_code=True, use_fast=True)
        return (model, tokenizer, image_processor)

    def _create_model(self, model_id: str, config_id: Optional[str], torch_dtype: str, trust_remote_code: bool, low_cpu_mem_usage: bool) -> nn.Module:
        config = AutoConfig.from_pretrained(config_id or model_id, trust_remote_code=trust_remote_code)
        config.attn_implementation = 'sdpa'
        dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
        if torch_dtype not in dtype_map:
            raise ModelInitializationError(f'Unsupported dtype: {torch_dtype}')
        dtype = dtype_map[torch_dtype]
        return AutoModelForCausalLM.from_pretrained(model_id, config=config, torch_dtype=dtype, low_cpu_mem_usage=low_cpu_mem_usage, trust_remote_code=trust_remote_code)

    def _create_tokenizer(self, tokenizer_id: str, special_tokens: List[str]) -> AutoTokenizer:
        from .constants import PROMPT_TOKENS
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info('Set pad_token to eos_token')
        all_special_tokens = special_tokens.copy() if special_tokens else []
        existing_tokens = set(tokenizer.get_vocab().keys())
        
        # Only add PROMPT_TOKENS if we actually have special tokens to add
        # This prevents unnecessary embedding resizing in vanilla mode
        if special_tokens:  # Only if we're in a mode that needs special tokens
            for token in PROMPT_TOKENS:
                if token not in existing_tokens and token not in all_special_tokens:
                    all_special_tokens.append(token)
        
        if all_special_tokens:
            tokenizer.add_special_tokens({'additional_special_tokens': all_special_tokens})
            logger.info(f'Added {len(all_special_tokens)} special tokens: {all_special_tokens}')
        else:
            logger.info('No special tokens added - using base tokenizer vocabulary')
        return tokenizer

    def _resize_token_embeddings(self, tokenizer: AutoTokenizer) -> None:
        if hasattr(self.model, 'language_model'):
            self.model.language_model.resize_token_embeddings(len(tokenizer))
        else:
            self.model.resize_token_embeddings(len(tokenizer))

    def _resize_special_token_embeddings(self) -> None:
        current_size = len(self.tokenizer)
        
        # Get vocab size from the correct config attribute
        # InternVL models use different config structure
        if hasattr(self.model.config, 'vocab_size'):
            model_vocab_size = self.model.config.vocab_size
        elif hasattr(self.model.config, 'llm_config') and hasattr(self.model.config.llm_config, 'vocab_size'):
            model_vocab_size = self.model.config.llm_config.vocab_size
        elif hasattr(self.model, 'language_model') and hasattr(self.model.language_model.config, 'vocab_size'):
            model_vocab_size = self.model.language_model.config.vocab_size
        else:
            # Fallback: get vocab size from the actual embedding layer
            if hasattr(self.model, 'language_model'):
                embed_layer = self.model.language_model.get_input_embeddings()
            else:
                embed_layer = self.model.get_input_embeddings()
            model_vocab_size = embed_layer.num_embeddings
            logger.warning(f'Could not find vocab_size in config, using embedding layer size: {model_vocab_size}')
        
        # Only resize if we actually added tokens
        if current_size > model_vocab_size:
            logger.info(f'Resizing embeddings from {model_vocab_size} to {current_size} for {current_size - model_vocab_size} new tokens')
            if hasattr(self.model, 'language_model'):
                self.model.language_model.resize_token_embeddings(current_size)
            else:
                self.model.resize_token_embeddings(current_size)
        else:
            logger.info(f'No embedding resize needed - tokenizer size {current_size} matches model vocab size {model_vocab_size}')

    def _setup_special_tokens(self) -> None:
        img_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        if img_token_id is not None:
            self.model.img_context_token_id = img_token_id
        else:
            logger.warning(f"Image context token '{IMG_CONTEXT_TOKEN}' not found in tokenizer")
        self.eos_token_id = self.tokenizer.eos_token_id

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    @property
    def device(self):
        return next(self.parameters()).device

    def _ensure_dtype_consistency(self, **kwargs) -> Dict[str, Any]:
        model_dtype = next(self.model.parameters()).dtype
        if (pixel_values := kwargs.get('pixel_values')) is not None:
            if pixel_values.dtype != model_dtype:
                kwargs['pixel_values'] = pixel_values.to(dtype=model_dtype)
        return kwargs

    def _clean_forward_kwargs(self, **kwargs) -> Dict[str, Any]:
        custom_args = {'question_ids', 'questions', 'original_questions', 'answers', 'num_items_in_batch'}
        return {k: v for k, v in kwargs.items() if k not in custom_args}

    def _generate_image_flags(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        return torch.ones(batch_size, dtype=torch.bool, device=device).unsqueeze(-1)

    def forward(self, **kwargs) -> Any:
        kwargs = self._clean_forward_kwargs(**kwargs)
        kwargs = self._ensure_dtype_consistency(**kwargs)
        if 'image_flags' not in kwargs and (pixel_values := kwargs.get('pixel_values')) is not None:
            kwargs['image_flags'] = self._generate_image_flags(pixel_values)
        with suppress_internvl_messages():
            return self.model(pixel_values=kwargs.get('pixel_values'), input_ids=kwargs.get('input_ids'), attention_mask=kwargs.get('attention_mask'), labels=kwargs.get('labels'), image_flags=kwargs.get('image_flags'))

    def generate(self, pixel_values: Optional[torch.Tensor], input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        if pixel_values is not None:
            model_dtype = next(self.model.parameters()).dtype
            if pixel_values.dtype != model_dtype:
                pixel_values = pixel_values.to(dtype=model_dtype)
        generation_kwargs = {k: v for k, v in kwargs.items() if k != 'image_flags'}
        with suppress_internvl_messages():
            return self.model.generate(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)