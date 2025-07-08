"""
Custom exceptions for the MultiCoCo package.

This module defines specific exception types to provide better error handling
and more informative error messages throughout the codebase.
"""

from typing import List, Optional


class MultiCoCoError(Exception):
    """Base exception class for MultiCoCo package."""
    pass


class ConfigurationError(MultiCoCoError):
    """Raised when there are configuration-related errors."""
    pass


class ModelInitializationError(MultiCoCoError):
    """Raised when model initialization fails."""
    pass


class DatasetError(MultiCoCoError):
    """Raised when there are dataset-related errors."""
    pass


class DataLoadingError(DatasetError):
    """Raised when data loading fails."""
    pass


class ImageProcessingError(DatasetError):
    """Raised when image processing fails."""
    pass


class TokenizationError(MultiCoCoError):
    """Raised when tokenization fails."""
    pass


class GenerationError(MultiCoCoError):
    """Raised when text generation fails."""
    pass


class EvaluationError(MultiCoCoError):
    """Raised when evaluation fails."""
    pass


class AnswerExtractionError(EvaluationError):
    """Raised when answer extraction from generated text fails."""
    pass


class CoCoNutTrainingError(MultiCoCoError):
    """Raised when CoCoNut-specific training operations fail."""
    pass


class InvalidAnswerChoiceError(EvaluationError):
    """Raised when an invalid answer choice is encountered."""
    def __init__(self, choice: str, valid_choices: list):
        self.choice = choice
        self.valid_choices = valid_choices
        super().__init__(f"Invalid answer choice '{choice}'. Valid choices are: {valid_choices}")


class MissingSpecialTokenError(TokenizationError):
    """Raised when required special tokens are missing from tokenizer."""
    def __init__(self, token_name: str):
        self.token_name = token_name
        super().__init__(f"Required special token '{token_name}' not found in tokenizer vocabulary")


class IncompatibleConfigurationError(ConfigurationError):
    """Raised when configuration options are incompatible with each other."""
    def __init__(self, message: str, conflicting_options: Optional[List[str]] = None):
        self.conflicting_options = conflicting_options if conflicting_options is not None else []
        super().__init__(f"{message}. Conflicting options: {self.conflicting_options}")


class DtypeMismatchError(MultiCoCoError):
    """Raised when tensor dtypes don't match model expectations."""
    def __init__(self, expected_dtype: str, actual_dtype: str):
        self.expected_dtype = expected_dtype
        self.actual_dtype = actual_dtype
        super().__init__(f"Dtype mismatch: expected {expected_dtype}, got {actual_dtype}") 