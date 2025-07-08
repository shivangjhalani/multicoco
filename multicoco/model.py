"""
Model wrapper for MultiCoCo multimodal AI.

This module provides a wrapper around InternVL models to enable CoCoNut
(Chain of Continuous Thought) training and evaluation with proper dtype
handling and special token management.
"""

import logging
from collections import namedtuple
from typing import Dict, List, Optional, Any, Union

# ** Core libraries  
import torch
from torch import nn

# ** Transformers components
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoImageProcessor, 
    AutoConfig
)

# ** Local imports
from .constants import (
    DEFAULT_MODEL_NAME,
    DEFAULT_DTYPE,
    IMAGE_TOKEN,
    IMG_CONTEXT_TOKEN,
    COCONUT_SPECIAL_TOKENS
)
from .exceptions import (
    ModelInitializationError,
    DtypeMismatchError,
    MissingSpecialTokenError
)

logger = logging.getLogger(__name__)

# Named tuple for model outputs
ModelOutputs = namedtuple("ModelOutputs", ["loss", "inputs_embeds", "logits"])


class MultiCoCo(nn.Module):
    """
    MultiCoCo model wrapper for InternVL with CoCoNut support.
    
    This class wraps an InternVL model and provides additional functionality
    for CoCoNut training, including special token handling and dtype consistency.
    
    Args:
        model_id: HuggingFace model identifier
        config_id: Optional separate config identifier
        tokenizer_id: Optional separate tokenizer identifier  
        image_processor_id: Optional separate image processor identifier
        special_tokens: List of special tokens to add to tokenizer
        torch_dtype: PyTorch dtype for model weights
        trust_remote_code: Whether to trust remote code
        low_cpu_mem_usage: Whether to use low CPU memory loading
        **kwargs: Additional arguments
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
            self._initialize_model(
                model_id, config_id, torch_dtype, 
                trust_remote_code, low_cpu_mem_usage
            )
            self._initialize_tokenizer(tokenizer_id or model_id, special_tokens)
            self._initialize_image_processor(image_processor_id or model_id)
            self._setup_special_tokens()
            
        except Exception as e:
            raise ModelInitializationError(f"Failed to initialize MultiCoCo model: {e}")
        
        logger.info(f"MultiCoCo model initialized with {self._count_parameters()} parameters")

    def _initialize_model(
        self, 
        model_id: str, 
        config_id: Optional[str], 
        torch_dtype: str,
        trust_remote_code: bool, 
        low_cpu_mem_usage: bool
    ) -> None:
        """Initialize the base model with configuration."""
        conf_id = config_id if config_id else model_id
        
        # Load and configure model config
        config = AutoConfig.from_pretrained(conf_id, trust_remote_code=trust_remote_code)
        config.attn_implementation = "eager"  # Use eager attention for compatibility

        # Convert string dtype to torch dtype
        if torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "float32":
            dtype = torch.float32
        else:
            raise ModelInitializationError(f"Unsupported dtype: {torch_dtype}")

        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            trust_remote_code=trust_remote_code,
        )

    def _initialize_tokenizer(self, tokenizer_id: str, special_tokens: List[str]) -> None:
        """Initialize tokenizer with special tokens."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, 
            trust_remote_code=True
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        # Add special tokens if provided
        if special_tokens:
            self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            self._resize_token_embeddings()
            logger.info(f"Added {len(special_tokens)} special tokens: {special_tokens}")

    def _initialize_image_processor(self, processor_id: str) -> None:
        """Initialize image processor."""
        self.image_processor = AutoImageProcessor.from_pretrained(
            processor_id, 
            trust_remote_code=True,
            use_fast=True
        )

    def _setup_special_tokens(self) -> None:
        """Set up special token IDs for the model."""
        # Set image context token ID for InternVL
        img_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        if img_token_id is not None:
            self.model.img_context_token_id = img_token_id
        else:
            logger.warning(f"Image context token '{IMG_CONTEXT_TOKEN}' not found in tokenizer")

        # Set up CoCoNut special token IDs
        self.thought_token_id = self.tokenizer.convert_tokens_to_ids('<thought>')
        self.eos_token_id = self.tokenizer.eos_token_id

    def _resize_token_embeddings(self) -> None:
        """Resize token embeddings after adding special tokens."""
        # Handle different model architectures
        if hasattr(self.model, 'language_model'):
            self.model.language_model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def _count_parameters(self) -> int:
        """Count the number of model parameters."""
        return sum(p.numel() for p in self.model.parameters())

    def get_input_embeddings(self) -> nn.Module:
        """Get input embedding layer."""
        return self.model.get_input_embeddings()

    def _ensure_dtype_consistency(self, **kwargs) -> Dict[str, Any]:
        """
        Ensure all input tensors match the model's dtype.
        
        Args:
            **kwargs: Input arguments that may contain tensors
            
        Returns:
            Updated kwargs with consistent dtypes
            
        Raises:
            DtypeMismatchError: If dtype conversion fails
        """
        try:
            # Get the model's dtype
            model_dtype = next(self.model.parameters()).dtype
            
            # Convert pixel_values to model dtype if present
            if 'pixel_values' in kwargs and kwargs['pixel_values'] is not None:
                pixel_values = kwargs['pixel_values']
                if pixel_values.dtype != model_dtype:
                    kwargs['pixel_values'] = pixel_values.to(dtype=model_dtype)
                    logger.debug(f"Converted pixel_values from {pixel_values.dtype} to {model_dtype}")
                    
            return kwargs
            
        except Exception as e:
            raise DtypeMismatchError("unknown", "unknown") from e

    def _clean_forward_kwargs(self, **kwargs) -> Dict[str, Any]:
        """
        Remove custom arguments that shouldn't be passed to the base model.
        
        Args:
            **kwargs: Input arguments
            
        Returns:
            Cleaned kwargs for model forward pass
        """
        # These are custom arguments from our data collator that should not
        # be passed to the underlying model's forward method
        custom_args = {
            'question_ids', 'questions', 'original_questions', 
            'answers', 'num_items_in_batch', 'image_flags'
        }
        
        cleaned_kwargs = {k: v for k, v in kwargs.items() if k not in custom_args}
        return cleaned_kwargs

    def forward(self, **kwargs) -> Any:
        """
        Forward pass through the model.
        
        Args:
            **kwargs: Input arguments including pixel_values, input_ids, etc.
            
        Returns:
            Model outputs
        """
        # Clean arguments and ensure dtype consistency
        kwargs = self._clean_forward_kwargs(**kwargs)
        kwargs = self._ensure_dtype_consistency(**kwargs)
        
        # Forward pass through the underlying model
        return self.model(
            pixel_values=kwargs.get('pixel_values'),
            input_ids=kwargs.get('input_ids'),
            attention_mask=kwargs.get('attention_mask'),
            labels=kwargs.get('labels')
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
            
        Raises:
            DtypeMismatchError: If dtype conversion fails
        """
        # Ensure dtype consistency for inputs
        if pixel_values is not None:
            model_dtype = next(self.model.parameters()).dtype
            if pixel_values.dtype != model_dtype:
                pixel_values = pixel_values.to(dtype=model_dtype)
        
        # Remove custom arguments that might interfere with generation
        generation_kwargs = {k: v for k, v in kwargs.items() if k != 'image_flags'}
        
        # Generate using the base model
        return self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
