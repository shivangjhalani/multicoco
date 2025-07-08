"""
Data loading and preprocessing modules for MultiCoCo.

This module provides dataset classes and data collation functions for handling
multimodal data (images and text) for training and evaluation with InternVL models.
"""

import json
import os
from typing import Dict, List, Any, Optional, Union
import logging

# ** Core libraries
import torch
from torch.utils.data import Dataset
from PIL import Image

# ** Local imports
from .constants import (
    DEFAULT_INPUT_MAX_LENGTH,
    DEFAULT_TARGET_MAX_LENGTH,
    DEFAULT_MAX_LENGTH,
    TEST_DATASET_LIMIT,
    LOSS_IGNORE_INDEX
)
from .exceptions import (
    DataLoadingError,
    ImageProcessingError,
    DatasetError
)

logger = logging.getLogger(__name__)


class SupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning with multimodal data.
    
    This dataset handles loading and preprocessing of image-text pairs
    for training and evaluation with InternVL models.
    
    Args:
        data_path: Path to the JSON data file
        data_dir: Directory containing the images
        test_limit: Optional limit for testing (loads only first N samples)
    """

    def __init__(
        self, 
        data_path: str, 
        data_dir: str, 
        test_limit: Optional[int] = None
    ) -> None:
        super().__init__()
        
        if not os.path.exists(data_path):
            raise DataLoadingError(f"Data file not found: {data_path}")
        
        if not os.path.exists(data_dir):
            raise DataLoadingError(f"Data directory not found: {data_dir}")
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise DataLoadingError(f"Failed to load data from {data_path}: {e}")
        
        # Apply test limit for development/testing
        if test_limit is not None:
            self.data = self.data[:test_limit]
            logger.info(f"Limited dataset to {test_limit} samples for testing")
        
        self.data_dir = data_dir
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Union[Image.Image, str]]:
        """
        Get a single sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Dictionary containing 'image', 'question', and 'answer' keys
            
        Raises:
            ImageProcessingError: If image loading fails
            DatasetError: If sample data is invalid
        """
        if index >= len(self.data):
            raise DatasetError(f"Index {index} out of range for dataset of size {len(self.data)}")
        
        item = self.data[index]
        
        # Validate required fields
        if 'image' not in item:
            raise DatasetError(f"Sample {index} missing 'image' field")
        if 'question' not in item:
            raise DatasetError(f"Sample {index} missing 'question' field")
        
        # Build image path
        image_file = item['image']
        image_path = os.path.join(self.data_dir, image_file)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            # Return a black placeholder image as fallback
            image = self._create_fallback_image()
        
        # Extract question and answer
        question = item['question']
        answer = item.get('answer', item.get('direct_answer', ''))
        
        return {
            'image': image,
            'question': question,
            'answer': answer
        }
    
    def _create_fallback_image(self) -> Image.Image:
        """Create a fallback black image when image loading fails."""
        return Image.new('RGB', (224, 224), color=(0, 0, 0))


def collate_fn(
    batch: List[Dict[str, Any]], 
    tokenizer: Any, 
    image_processor: Any
) -> Dict[str, torch.Tensor]:
    """
    Collate function for the SupervisedDataset.
    
    This function processes a batch of samples, handling image processing
    and text tokenization for training with InternVL models.
    
    Args:
        batch: List of sample dictionaries from the dataset
        tokenizer: Tokenizer for text processing
        image_processor: Image processor for image preprocessing
        
    Returns:
        Dictionary containing processed batch data ready for model training
        
    Raises:
        ImageProcessingError: If image processing fails
        DatasetError: If batch processing fails
    """
    if not batch:
        raise DatasetError("Empty batch provided to collate function")
    
    try:
        # Extract components from batch
        images = [item['image'] for item in batch]
        questions = [item['question'] for item in batch]
        answers = [item['answer'] for item in batch]
        
        # Process images
        pixel_values = _process_images(images, image_processor)
        
        # Create training data by concatenating questions and answers
        full_texts = [f"{question} {answer}" for question, answer in zip(questions, answers)]
        
        # Tokenize the full texts
        full_encodings = tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=DEFAULT_MAX_LENGTH,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        # Create labels for training - mask the question part
        labels = _create_training_labels(full_encodings['input_ids'], questions, tokenizer)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': full_encodings['input_ids'],
            'attention_mask': full_encodings['attention_mask'],
            'labels': labels,
            'questions': questions,  # Preserve for evaluation
            'answers': answers       # Preserve for evaluation
        }
        
    except Exception as e:
        raise DatasetError(f"Failed to process batch: {e}")


def _process_images(images: List[Image.Image], image_processor: Any) -> torch.Tensor:
    """
    Process a list of images using the provided image processor.
    
    Args:
        images: List of PIL Images
        image_processor: Image processor from transformers
        
    Returns:
        Processed pixel values as a tensor
        
    Raises:
        ImageProcessingError: If image processing fails
    """
    try:
        processed = image_processor(images, return_tensors='pt')
        return processed['pixel_values']
    except Exception as e:
        raise ImageProcessingError(f"Failed to process images: {e}")


def _create_training_labels(
    input_ids: torch.Tensor, 
    questions: List[str], 
    tokenizer: Any
) -> torch.Tensor:
    """
    Create training labels by masking question tokens.
    
    For training, we want the model to learn to generate answers given questions.
    This function masks question tokens in the labels so the model only learns
    to predict answer tokens.
    
    Args:
        input_ids: Tokenized input sequences
        questions: List of original questions
        tokenizer: Tokenizer used for encoding
        
    Returns:
        Labels tensor with question tokens masked (set to LOSS_IGNORE_INDEX)
    """
    labels = input_ids.clone()
    
    for i, question in enumerate(questions):
        try:
            # Tokenize just the question to determine mask length
            question_tokens = tokenizer(question, add_special_tokens=False)['input_ids']
            question_len = len(question_tokens)
            
            # Mask question tokens in labels (ignore in loss calculation)
            if question_len < labels.shape[1]:
                labels[i, :question_len] = LOSS_IGNORE_INDEX
        except Exception as e:
            logger.warning(f"Failed to mask question tokens for sample {i}: {e}")
            # If masking fails, keep original labels
            continue
    
    return labels