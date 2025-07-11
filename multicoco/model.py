"""
Model wrapper for MultiCoCo multimodal AI.

Provides a wrapper around InternVL models to enable CoCoNut (Chain of
Continuous Thought) training and evaluation with proper dtype handling
and special token management.
"""

import contextlib
import logging
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from .constants import (
    COCONUT_SPECIAL_TOKENS,
    DEFAULT_DTYPE,
    DEFAULT_MODEL_NAME,
    IMAGE_TOKEN,
    IMG_CONTEXT_TOKEN,
)
from .exceptions import DtypeMismatchError, ModelInitializationError

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_internvl_messages():
    """
    Context manager to suppress specific InternVL verbose messages during training.
    
    Suppresses dynamic ViT batch size and tensor warning messages.
    """
    import builtins
    original_print = builtins.print
    
    # Filter out specific InternVL messages
    suppress_phrases = [
        'dynamic ViT batch size:',
        'warning: The size of tensor a',
        'input_embeds[selected].shape=',
        'vit_embeds.shape='
    ]
    
    def filtered_print(*args, **kwargs):
        message = ' '.join(str(arg) for arg in args)
        if not any(phrase in message for phrase in suppress_phrases):
            original_print(*args, **kwargs)
    
    builtins.print = filtered_print
    try:
        yield
    finally:
        builtins.print = original_print


class MultiCoCo(nn.Module):
    """
    MultiCoCo model wrapper for InternVL with CoCoNut support.
    
    Wraps an InternVL model and provides additional functionality for CoCoNut
    training, including special token handling and dtype consistency.
    
    Args:
        model_id: HuggingFace model identifier
        config_id: Optional separate config identifier
        tokenizer_id: Optional separate tokenizer identifier  
        image_processor_id: Optional separate image processor identifier
        special_tokens: List of special tokens to add to tokenizer
        torch_dtype: PyTorch dtype for model weights
        trust_remote_code: Whether to trust remote code
        low_cpu_mem_usage: Whether to use low CPU memory loading
    """

    def __init__(
        self, 
        model_id: str = DEFAULT_MODEL_NAME,
        config_id: Optional[str] = None,
        tokenizer_id: Optional[str] = None,
        image_processor_id: Optional[str] = None,
        special_tokens: Optional[List[str]] = None,
        torch_dtype: str = DEFAULT_DTYPE,
        trust_remote_code: bool = True,
        low_cpu_mem_usage: bool = True,
        **kwargs
    ) -> None:
        super().__init__()
        
        special_tokens = special_tokens or []
        
        try:
            # Initialize all components in one go
            self.model, self.tokenizer, self.image_processor = self._initialize_components(
                model_id, config_id, tokenizer_id, image_processor_id,
                special_tokens, torch_dtype, trust_remote_code, low_cpu_mem_usage
            )
            self._setup_special_tokens()
            
        except Exception as e:
            raise ModelInitializationError(
                f"Failed to initialize MultiCoCo model: {e}"
            ) from e
        
        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"MultiCoCo model initialized with {param_count} parameters")

    def _initialize_components(
        self, 
        model_id: str, 
        config_id: Optional[str], 
        tokenizer_id: Optional[str],
        image_processor_id: Optional[str],
        special_tokens: List[str],
        torch_dtype: str,
        trust_remote_code: bool, 
        low_cpu_mem_usage: bool
    ) -> tuple[nn.Module, AutoTokenizer, AutoImageProcessor]:
        """Initialize all model components."""
        # Create model
        model = self._create_model(
            model_id, config_id, torch_dtype, trust_remote_code, low_cpu_mem_usage
        )
        
        # Create tokenizer
        tokenizer = self._create_tokenizer(tokenizer_id or model_id, special_tokens)
        
        # Create image processor
        image_processor = AutoImageProcessor.from_pretrained(
            image_processor_id or model_id, trust_remote_code=True, use_fast=True
        )
        
        return model, tokenizer, image_processor

    def _create_model(
        self, 
        model_id: str, 
        config_id: Optional[str], 
        torch_dtype: str,
        trust_remote_code: bool, 
        low_cpu_mem_usage: bool
    ) -> nn.Module:
        """Create and configure the base model."""
        # Load and configure model config
        config = AutoConfig.from_pretrained(
            config_id or model_id, trust_remote_code=trust_remote_code
        )
        config.attn_implementation = "sdpa"  # Use optimized attention

        # Convert string dtype to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        
        if torch_dtype not in dtype_map:
            raise ModelInitializationError(f"Unsupported dtype: {torch_dtype}")
        
        dtype = dtype_map[torch_dtype]

        # Load the model
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            trust_remote_code=trust_remote_code,
        )

    def _create_tokenizer(
        self, tokenizer_id: str, special_tokens: List[str]
    ) -> AutoTokenizer:
        """Create and configure tokenizer with special tokens."""
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, trust_remote_code=True
        )
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        # Add special tokens if provided
        if special_tokens:
            tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            self._resize_token_embeddings(tokenizer)
            logger.info(f"Added {len(special_tokens)} special tokens: {special_tokens}")
        
        return tokenizer

    def _resize_token_embeddings(self, tokenizer: AutoTokenizer) -> None:
        """Resize token embeddings after adding special tokens."""
        # Handle different model architectures
        if hasattr(self.model, 'language_model'):
            self.model.language_model.resize_token_embeddings(len(tokenizer))
        else:
            self.model.resize_token_embeddings(len(tokenizer))

    def _setup_special_tokens(self) -> None:
        """Set up special token IDs for the model."""
        # Set image context token ID for InternVL
        img_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        if img_token_id is not None:
            self.model.img_context_token_id = img_token_id
        else:
            logger.warning(
                f"Image context token '{IMG_CONTEXT_TOKEN}' not found in tokenizer"
            )

        # Keep reference to eos id for convenience
        self.eos_token_id = self.tokenizer.eos_token_id

    def get_input_embeddings(self) -> nn.Module:
        """Get input embedding layer."""
        return self.model.get_input_embeddings()

    @property
    def device(self):
        """Return the device of the model's parameters."""
        return next(self.parameters()).device

    def _ensure_dtype_consistency(self, **kwargs) -> Dict[str, Any]:
        """Ensure all input tensors match the model's dtype."""
        try:
            model_dtype = next(self.model.parameters()).dtype
            
            # Convert pixel_values to model dtype if present
            if (pixel_values := kwargs.get('pixel_values')) is not None:
                if pixel_values.dtype != model_dtype:
                    kwargs['pixel_values'] = pixel_values.to(dtype=model_dtype)
                    logger.debug(
                        f"Converted pixel_values from {pixel_values.dtype} to {model_dtype}"
                    )
                    
            return kwargs
            
        except Exception as e:
            raise DtypeMismatchError("unknown", "unknown") from e

    def _clean_forward_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Remove custom arguments that shouldn't be passed to the base model."""
        # Custom arguments from data collator that should not be passed
        # Note: image_flags is needed by InternVL models, so we keep it
        custom_args = {
            'question_ids', 'questions', 'original_questions', 
            'answers', 'num_items_in_batch'
        }
        
        return {k: v for k, v in kwargs.items() if k not in custom_args}

    def _generate_image_flags(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Generate image flags for InternVL models."""
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        return torch.ones(batch_size, dtype=torch.bool, device=device).unsqueeze(-1)

    def forward(self, **kwargs) -> Any:
        """Forward pass through the model."""
        # Clean arguments and ensure dtype consistency
        kwargs = self._clean_forward_kwargs(**kwargs)
        kwargs = self._ensure_dtype_consistency(**kwargs)
        
        # Generate image_flags if not provided (InternVL models require this)
        if ('image_flags' not in kwargs and 
            (pixel_values := kwargs.get('pixel_values')) is not None):
            kwargs['image_flags'] = self._generate_image_flags(pixel_values)
        
        # Forward pass through the underlying model with message suppression
        with suppress_internvl_messages():
            return self.model(
                pixel_values=kwargs.get('pixel_values'),
                input_ids=kwargs.get('input_ids'),
                attention_mask=kwargs.get('attention_mask'),
                labels=kwargs.get('labels'),
                image_flags=kwargs.get('image_flags')
            )

    def generate(
        self, 
        pixel_values: Optional[torch.Tensor], 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            pixel_values: Image pixel values (optional)
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional generation arguments
            
        Returns:
            Generated token sequences
        """
        # Ensure dtype consistency for inputs
        if pixel_values is not None:
            model_dtype = next(self.model.parameters()).dtype
            if pixel_values.dtype != model_dtype:
                pixel_values = pixel_values.to(dtype=model_dtype)
        
        # Remove custom arguments that might interfere with generation
        generation_kwargs = {k: v for k, v in kwargs.items() if k != 'image_flags'}
        
        # Generate using the base model with message suppression
        with suppress_internvl_messages():
            return self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
