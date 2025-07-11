"""
MultiCoCo: Chain of Continuous Thought for Multimodal AI

A package for implementing and evaluating CoCoNut (Chain of Continuous Thought)
methodology with multimodal language models, specifically designed for InternVL.
"""

__version__ = "0.1.0"
__author__ = "MultiCoCo Team"

# Core components
from .model import MultiCoCo
from .trainer import CoCoTrainer
from .data import SupervisedDataset, collate_fn
from .answer_extraction import extract_answer_choice

# Configuration and utilities
from .config import (
    MultiCoCoConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    EvaluationConfig,
    CoCoNutConfig,
    GenerationConfig,
    LoggingConfig
)


# Constants
from .constants import (
    DEFAULT_MODEL_NAME,
    COCONUT_SPECIAL_TOKENS,
    VALID_CHOICE_NUMBERS,
    EVAL_TYPES
)

# Exceptions
from .exceptions import (
    MultiCoCoError,
    ConfigurationError,
    ModelInitializationError,
    DatasetError,
    EvaluationError,
    GenerationError,
    AnswerExtractionError
)

# Main exports
__all__ = [
    # Core classes
    "MultiCoCo",
    "CoCoTrainer", 
    "SupervisedDataset",
    "collate_fn",
    "extract_answer_choice",
    
    # Configuration
    "MultiCoCoConfig",
    "ModelConfig",
    "TrainingConfig", 
    "DataConfig",
    "EvaluationConfig",
    "CoCoNutConfig",
    "GenerationConfig",
    "LoggingConfig",
    
    # Constants
    "DEFAULT_MODEL_NAME",
    "COCONUT_SPECIAL_TOKENS",
    "VALID_CHOICE_NUMBERS",
    "EVAL_TYPES",
    
    # Exceptions
    "MultiCoCoError",
    "ConfigurationError",
    "ModelInitializationError",
    "DatasetError", 
    "EvaluationError",
    "GenerationError",
    "AnswerExtractionError",
]
