"""
Custom exceptions for the MultiCoCo package.

This module defines specific exception types to provide better error handling
and more informative error messages throughout the codebase.
"""


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


class GenerationError(MultiCoCoError):
    """Raised when text generation fails."""
    pass


class EvaluationError(MultiCoCoError):
    """Raised when evaluation fails."""
    pass


class AnswerExtractionError(EvaluationError):
    """Raised when answer extraction from generated text fails."""
    pass


class DtypeMismatchError(MultiCoCoError):
    """Raised when tensor dtypes don't match model expectations."""
    def __init__(self, expected_dtype: str, actual_dtype: str):
        self.expected_dtype = expected_dtype
        self.actual_dtype = actual_dtype
        super().__init__(f"Dtype mismatch: expected {expected_dtype}, got {actual_dtype}") 