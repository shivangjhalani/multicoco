"""
Data loading and preprocessing modules for MultiCoCo.

This module provides dataset classes and data collation functions for handling
multimodal data (images and text) for training and evaluation with InternVL models.
"""

import json
import os
from typing import Dict, List, Any, Optional, Union
import logging

import torch
from torch.utils.data import Dataset
from PIL import Image
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
        self._original_data = self.data.copy()  # Keep original data for progressive training
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def apply_progressive_curriculum(
        self, 
        scheduled_stage: int, 
        c_thought: int, 
        max_latent_stage: int,
        uniform_prob: float = 0.0,
        pad_latent_to_max: bool = False,
        no_cot: bool = False
    ) -> None:
        """
        Apply progressive curriculum learning to the dataset.
        
        This method transforms the dataset according to the current training stage,
        progressively replacing reasoning steps with latent tokens.
        
        Args:
            scheduled_stage: Current training stage (0=CoT, 1+=progressive latent)
            c_thought: Number of continuous thoughts per reasoning step
            max_latent_stage: Maximum number of latent stages
            uniform_prob: Probability to randomly sample from other stages
            pad_latent_to_max: Whether to pad latent tokens to max stage
            no_cot: If True, skip all reasoning steps
        """
        logger.info(f"Applying progressive curriculum for stage {scheduled_stage}")
        
        # Use the progressive latent dataset creation function
        processed_data = create_progressive_latent_dataset(
            scheduled_stage=scheduled_stage,
            base_dataset=self._original_data,
            c_thought=c_thought,
            max_latent_stage=max_latent_stage,
            uniform_prob=uniform_prob,
            pad_latent_to_max=pad_latent_to_max,
            no_cot=no_cot
        )
        
        # Update the dataset with processed data
        self.data = processed_data
        logger.info(f"Dataset updated with {len(self.data)} progressive curriculum samples")

    def __getitem__(self, index: int) -> Dict[str, Union[Image.Image, str]]:
        """
        Get a single sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Dictionary containing 'image', 'question', 'answer', and optionally 'steps' keys
            
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
        
        # Create return dictionary
        result = {
            'image': image,
            'question': question,
            'answer': answer
        }
        
        # Add reasoning steps if available (for CoT training)
        if 'steps' in item and item['steps']:
            result['steps'] = item['steps']
        
        return result
    
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
    and text tokenization for training with InternVL models. It now
    formats the data into a chat-style format consistent with evaluation.
    
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
        
        # Create chat-formatted training data
        full_texts = []
        prompts = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            # Check if we have reasoning steps for CoT training
            reasoning_steps = batch[i].get('steps', [])
            reasoning_text = batch[i].get('reasoning', '')
            
            assistant_part = ''
            if reasoning_text:
                # Progressive curriculum format
                assistant_part = f"{reasoning_text} The answer is {answer}"
            elif reasoning_steps and len(reasoning_steps) > 0:
                # CoT format
                reasoning_combined = " ".join(reasoning_steps)
                assistant_part = f"{reasoning_combined} The answer is {answer}"
            else:
                # Vanilla format
                assistant_part = f"{answer}"
            
            # Construct the prompt using the chat format with the <image> token
            # This aligns training with the evaluation format used by .chat()
            prompt = f"<|im_start|>user\n<image>\n{question}<|im_end|><|im_start|>assistant\n"
            full_text = f"{prompt}{assistant_part}"
            
            full_texts.append(full_text)
            prompts.append(prompt)
        
        # Tokenize the full texts
        full_encodings = tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=DEFAULT_MAX_LENGTH,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        # Create labels for training - mask the prompt part
        labels = _create_training_labels(full_encodings['input_ids'], prompts, tokenizer)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': full_encodings['input_ids'],
            'attention_mask': full_encodings['attention_mask'],
            'labels': labels,
            'questions': questions,
            'answers': answers
        }
    
    except Exception as e:
        logger.error(f"Error in collate_fn: {e}", exc_info=True)
        # Depending on desired behavior, either raise or return an empty batch
        raise DatasetError(f"Failed to collate batch: {e}")


def _process_images(images: List[Image.Image], image_processor: Any) -> torch.Tensor:
    """
    Process a list of PIL images into a batch of tensors.
    
    Args:
        images: List of PIL images
        image_processor: Image processor (e.g., from Hugging Face)
        
    Returns:
        A batch of preprocessed image tensors
        
    Raises:
        ImageProcessingError: If image processing fails
    """
    try:
        # The image_processor from Hugging Face expects a list of PIL images
        # and returns a dictionary containing 'pixel_values'
        processed = image_processor(images, return_tensors='pt')
        return processed['pixel_values']
    except Exception as e:
        logger.error(f"Failed to process images: {e}", exc_info=True)
        raise ImageProcessingError(f"Error during image processing: {e}")


def _create_training_labels(
    input_ids: torch.Tensor, 
    prompts: List[str], 
    tokenizer: Any
) -> torch.Tensor:
    """
    Create labels for training, masking the prompt part of the input.
    
    This ensures that the loss is only calculated on the assistant's
    response, not on the user's prompt.
    
    Args:
        input_ids: The tokenized input sequences for the batch
        prompts: The list of prompt strings for each item in the batch
        tokenizer: The tokenizer used for encoding
        
    Returns:
        A tensor of labels with the prompt tokens masked
    """
    labels = input_ids.clone()
    
    for i, prompt in enumerate(prompts):
        # Tokenize the prompt to find its length (without adding special tokens)
        prompt_tokens = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors='pt'
        ).input_ids[0]
        
        prompt_length = len(prompt_tokens)
        
        # Mask the prompt tokens by setting them to the ignore index
        labels[i, :prompt_length] = LOSS_IGNORE_INDEX
        
    return labels


def create_progressive_latent_dataset(
    scheduled_stage: int,
    base_dataset: List[Dict],
    c_thought: int,
    max_latent_stage: int,
    uniform_prob: float = 0.0,
    pad_latent_to_max: bool = False,
    no_cot: bool = False
) -> List[Dict]:
    """
    Create a dataset for a specific stage of progressive latent training.
    
    This function implements the core progressive curriculum learning:
    - Stage 0: Full CoT (question + reasoning_steps + answer)
    - Stage 1: Replace 1st reasoning step with latent tokens  
    - Stage 2: Replace 2nd reasoning step with additional latent tokens
    - Stage N: Replace N reasoning steps with N×c_thought latent tokens
    
    Args:
        scheduled_stage: The target stage for curriculum application
        base_dataset: The original dataset with full reasoning steps
        c_thought: Number of continuous thoughts per reasoning step
        max_latent_stage: The maximum number of latent stages
        uniform_prob: Probability to sample from other stages for diversity
        pad_latent_to_max: Whether to pad latent tokens to the max stage
        no_cot: If True, skip all reasoning steps
        
    Returns:
        A new dataset (list of dicts) with curriculum applied
    """
    import random
    import itertools
    
    logger.info(f"Creating progressive latent dataset for stage {scheduled_stage}")
    logger.info(f"Parameters: c_thought={c_thought}, max_latent_stage={max_latent_stage}")
    
    processed_samples = []
    
    for sample in base_dataset:
        # Parse the sample for reasoning steps
        steps = sample.get('steps', [])
        if isinstance(steps, str):
            # If steps is a single string, split it into individual steps
            steps = [step.strip() for step in steps.split('\n') if step.strip()]
        
        # Determine the training stage for this sample
        if random.random() < uniform_prob:
            # With some probability, randomly sample stage for curriculum mixing
            scheduled_stage_to_train = random.choice(list(range(len(steps) + 1)))
        else:
            scheduled_stage_to_train = scheduled_stage
        
        # Calculate how many steps to skip and how many latent tokens to use
        if scheduled_stage_to_train > max_latent_stage:
            n_skip_steps = 10000  # Skip all steps
            if pad_latent_to_max:
                n_latent_tokens = max_latent_stage
            else:
                n_latent_tokens = min(len(steps), max_latent_stage)
        else:
            n_skip_steps = scheduled_stage_to_train
            n_latent_tokens = scheduled_stage_to_train
        
        if no_cot:
            n_skip_steps = 100  # Skip all steps
            n_latent_tokens = 0
        
        # Multiply by c_thought to get total latent tokens
        total_latent_tokens = n_latent_tokens * c_thought
        
        # Build the reasoning text with progressive replacement
        reasoning_text = ""
        
        from multicoco.constants import START_LATENT_TOKEN, END_LATENT_TOKEN, LATENT_TOKEN

        # Add latent tokens wrapped with boundary markers if any
        if total_latent_tokens > 0:
            latent_block = " ".join([LATENT_TOKEN] * total_latent_tokens)
            reasoning_text += f"{START_LATENT_TOKEN} {latent_block} {END_LATENT_TOKEN}".strip()
        
        
        # Add remaining reasoning steps (those not replaced by latent tokens)
        remaining_steps = steps[n_skip_steps:] if n_skip_steps < len(steps) else []
        if remaining_steps:
            if reasoning_text:
                reasoning_text += " "
            reasoning_text += " ".join(remaining_steps)
        
        # Create the processed sample
        processed_sample = {
            'question': sample['question'],
            'reasoning': reasoning_text.strip() if reasoning_text.strip() else "",
            'answer': sample['answer'],
            'stage': scheduled_stage_to_train,
            'n_latent_tokens': total_latent_tokens,
            'n_skip_steps': n_skip_steps
        }
        
        processed_samples.append(processed_sample)
    
    # Shuffling is now handled by the DataLoader, not here
    return processed_samples