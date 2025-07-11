"""
Custom exceptions for the MultiCoCo package.

Defines specific exception types to provide better error handling and more
informative error messages throughout the codebase.
"""

from typing import Optional


class MultiCoCoError(Exception):
    """
    Base exception class for MultiCoCo package.
    
    All custom exceptions in the MultiCoCo package should inherit from this
    base class to provide consistent error handling and identification.
    """


class ConfigurationError(MultiCoCoError):
    """
    Raised when there are configuration-related errors.
    
    This exception is used for issues with YAML configuration files,
    invalid parameter combinations, or missing required configuration values.
    """


class ModelInitializationError(MultiCoCoError):
    """
    Raised when model initialization fails.
    
    This exception covers errors during model loading, tokenizer setup,
    checkpoint loading, or any other model-related initialization issues.
    """


class DatasetError(MultiCoCoError):
    """
    Base class for dataset-related errors.
    
    Provides a common base for all dataset and data processing related
    exceptions to enable targeted error handling.
    """


class DataLoadingError(DatasetError):
    """
    Raised when data loading fails.
    
    This exception is used when there are issues loading training or
    evaluation datasets, including file not found, parsing errors,
    or invalid data format issues.
    """


class ImageProcessingError(DatasetError):
    """
    Raised when image processing fails.
    
    This exception covers errors during image loading, preprocessing,
    resizing, or any other image processing operations.
    """


class EvaluationError(MultiCoCoError):
    """
    Raised when evaluation fails.
    
    This exception covers errors during model evaluation, metric
    computation, or evaluation result processing.
    """


class AnswerExtractionError(EvaluationError):
    """
    Raised when answer extraction from generated text fails.
    
    This exception is used when the answer extraction utilities cannot
    parse or extract a valid answer choice from model-generated text.
    """ 