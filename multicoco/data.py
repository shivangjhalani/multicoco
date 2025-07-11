"""
Data loading and preprocessing modules for MultiCoCo.

Provides dataset classes and data collation functions for handling multimodal
data (images and text) for training and evaluation with InternVL models.
"""

import json
import logging
import os
# Standard libs
import random

# Third-party nad central utils
from multicoco import wandb_utils as wdb
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from torch.utils.data import Dataset

from .constants import (
    DEFAULT_INPUT_MAX_LENGTH,
    DEFAULT_MAX_LENGTH,
    DEFAULT_TARGET_MAX_LENGTH,
    END_LATENT_TOKEN,
    LATENT_TOKEN,
    LOSS_IGNORE_INDEX,
    START_LATENT_TOKEN,
    TEST_DATASET_LIMIT,
)
from .exceptions import DataLoadingError, DatasetError, ImageProcessingError

logger = logging.getLogger(__name__)


class SupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning with multimodal data.
    
    Handles loading and preprocessing of image-text pairs for training and
    evaluation with InternVL models.
    
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
        
        self._validate_paths(data_path, data_dir)
        self.data = self._load_data(data_path, test_limit)
        self.data_dir = data_dir
        self._original_data = self.data.copy()
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")

    def _validate_paths(self, data_path: str, data_dir: str) -> None:
        """Validate that required paths exist."""
        if not os.path.exists(data_path):
            raise DataLoadingError(f"Data file not found: {data_path}")
        if not os.path.exists(data_dir):
            raise DataLoadingError(f"Data directory not found: {data_dir}")

    def _load_data(self, data_path: str, test_limit: Optional[int]) -> List[Dict]:
        """Load and optionally limit data from JSON file."""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise DataLoadingError(f"Failed to load data from {data_path}: {e}")
        
        if test_limit is not None:
            data = data[:test_limit]
            logger.info(f"Limited dataset to {test_limit} samples for testing")
        
        return data

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
        
        Transforms the dataset according to the current training stage,
        progressively replacing reasoning steps with latent tokens.
        """
        logger.info(f"Applying progressive curriculum for stage {scheduled_stage}")
        
        self.data = create_progressive_latent_dataset(
            scheduled_stage=scheduled_stage,
            base_dataset=self._original_data,
            c_thought=c_thought,
            max_latent_stage=max_latent_stage,
            uniform_prob=uniform_prob,
            pad_latent_to_max=pad_latent_to_max,
            no_cot=no_cot
        )
        
        logger.info(f"Dataset updated with {len(self.data)} curriculum samples")

    def __getitem__(self, index: int) -> Dict[str, Union[Image.Image, str]]:
        """Get a single sample from the dataset."""
        if index >= len(self.data):
            raise DatasetError(
                f"Index {index} out of range for dataset of size {len(self.data)}"
            )
        
        item = self.data[index]
        self._validate_item(item, index)
        
        # Load image with fallback
        image = self._load_image(item['image'])
        
        # Extract text components
        question = item['question']
        answer = item.get('answer', item.get('direct_answer', ''))
        
        # Build result dictionary
        result = {
            'image': image,
            'question': question,
            'answer': answer
        }
        
        # Add reasoning steps if available
        if steps := item.get('steps'):
            result['steps'] = steps
            
        return result
    
    def _validate_item(self, item: Dict, index: int) -> None:
        """Validate that item has required fields."""
        required_fields = ['image', 'question']
        for field in required_fields:
            if field not in item:
                raise DatasetError(f"Sample {index} missing '{field}' field")
    
    def _load_image(self, image_file: str) -> Image.Image:
        """Load image with error handling and fallback."""
        image_path = os.path.join(self.data_dir, image_file)
        
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return self._create_fallback_image()
    
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
    
    Processes a batch of samples, handling image processing and text
    tokenization for training with InternVL models using chat format.
    """
    if not batch:
        raise DatasetError("Empty batch provided to collate function")
    
    try:
        # Extract components using list comprehensions
        images = [item['image'] for item in batch]
        questions = [item['question'] for item in batch]
        answers = [item['answer'] for item in batch]
        
        # Process images
        pixel_values = _process_images(images, image_processor)
        
        # Create chat-formatted training data
        full_texts, prompts = _create_chat_formatted_texts(batch, questions, answers)
        
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
        raise DatasetError(f"Failed to collate batch: {e}")


def _create_chat_formatted_texts(
    batch: List[Dict[str, Any]], 
    questions: List[str], 
    answers: List[str]
) -> tuple[List[str], List[str]]:
    """Create chat-formatted texts and prompts for training."""
    full_texts = []
    prompts = []
    
    for i, (question, answer) in enumerate(zip(questions, answers)):
        # Determine assistant response format
        reasoning_text = batch[i].get('reasoning', '')
        reasoning_steps = batch[i].get('steps', [])
        
        if reasoning_text:
            assistant_part = f"{reasoning_text} The answer is {answer}"
        elif reasoning_steps:
            reasoning_combined = " ".join(reasoning_steps)
            assistant_part = f"{reasoning_combined} The answer is {answer}"
        else:
            assistant_part = answer
        
        # Construct chat format
        prompt = f"<|im_start|>user\n<image>\n{question}<|im_end|><|im_start|>assistant\n"
        full_text = f"{prompt}{assistant_part}"
        
        full_texts.append(full_text)
        prompts.append(prompt)
    
    return full_texts, prompts


def _process_images(images: List[Image.Image], image_processor: Any) -> torch.Tensor:
    """Process a list of PIL images into a batch of tensors."""
    try:
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
    
    Ensures that loss is only calculated on the assistant's response,
    not on the user's prompt.
    """
    labels = input_ids.clone()
    
    for i, prompt in enumerate(prompts):
        # Tokenize prompt to find its length
        prompt_tokens = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors='pt'
        ).input_ids[0]
        
        prompt_length = len(prompt_tokens)
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
    
    Implements the core progressive curriculum learning:
    - Stage 0: Full CoT (question + reasoning_steps + answer)
    - Stage 1: Replace 1st reasoning step with latent tokens  
    - Stage 2: Replace 2nd reasoning step with additional latent tokens
    - Stage N: Replace N reasoning steps with N×c_thought latent tokens
    """
    logger.info(f"Creating progressive latent dataset for stage {scheduled_stage}")
    logger.info(f"Parameters: c_thought={c_thought}, max_latent_stage={max_latent_stage}")
    
    processed_samples = []
    stage_counts = {s: 0 for s in range(max_latent_stage + 1)}
    
    for sample in base_dataset:
        # Parse reasoning steps
        steps = _parse_reasoning_steps(sample.get('steps', []))
        
        # Determine training stage with uniform probability mixing
        stage_to_train = (random.choice(range(len(steps) + 1)) 
                         if random.random() < uniform_prob 
                         else scheduled_stage)
        
        # Calculate latent tokens and steps to skip
        n_skip_steps, n_latent_tokens = _calculate_curriculum_params(
            stage_to_train, max_latent_stage, steps, pad_latent_to_max, no_cot
        )
        
        total_latent_tokens = n_latent_tokens * c_thought
        
        # Build reasoning text with progressive replacement
        reasoning_text = _build_reasoning_text(
            total_latent_tokens, steps, n_skip_steps
        )
        
        # Create processed sample
        processed_sample = {
            'question': sample['question'],
            'reasoning': reasoning_text,
            'answer': sample['answer'],
            'stage': stage_to_train,
            'n_latent_tokens': total_latent_tokens,
            'n_skip_steps': n_skip_steps
        }
        
        processed_samples.append(processed_sample)
        if stage_to_train in stage_counts:
            stage_counts[stage_to_train] += 1
    
    # Log overall stage distribution via central helper (if active)
    if wdb.is_active():
        import wandb  # type: ignore  # local import only if library present

        table = wandb.Table(
            data=[[int(k), int(v)] for k, v in stage_counts.items()],
            columns=["stage", "count"],
        )
        wdb.log({
            "data/stage_distribution": wandb.plot.bar(
                table, "stage", "count", title="Curriculum Stage Distribution"
            )
        })

    return processed_samples


def _parse_reasoning_steps(steps: Union[List[str], str]) -> List[str]:
    """Parse reasoning steps from various input formats."""
    if isinstance(steps, str):
        return [step.strip() for step in steps.split('\n') if step.strip()]
    return steps


def _calculate_curriculum_params(
    stage_to_train: int, 
    max_latent_stage: int, 
    steps: List[str], 
    pad_latent_to_max: bool, 
    no_cot: bool
) -> tuple[int, int]:
    """Calculate curriculum parameters for progressive training."""
    if no_cot:
        return 100, 0  # Skip all steps, no latent tokens
    
    if stage_to_train > max_latent_stage:
        n_skip_steps = 10000  # Skip all steps
        n_latent_tokens = (max_latent_stage if pad_latent_to_max 
                          else min(len(steps), max_latent_stage))
    else:
        n_skip_steps = stage_to_train
        n_latent_tokens = stage_to_train
    
    return n_skip_steps, n_latent_tokens


def _build_reasoning_text(
    total_latent_tokens: int, 
    steps: List[str], 
    n_skip_steps: int
) -> str:
    """Build reasoning text with latent tokens and remaining steps."""
    reasoning_parts = []
    
    # Add latent tokens if any
    if total_latent_tokens > 0:
        latent_block = " ".join([LATENT_TOKEN] * total_latent_tokens)
        reasoning_parts.append(f"{START_LATENT_TOKEN} {latent_block} {END_LATENT_TOKEN}")
    
    # Add remaining reasoning steps
    if remaining_steps := steps[n_skip_steps:]:
        reasoning_parts.append(" ".join(remaining_steps))
    
    return " ".join(reasoning_parts).strip()