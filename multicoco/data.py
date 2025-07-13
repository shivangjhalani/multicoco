import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from PIL import Image
from torch.utils.data import Dataset
from .constants import DEFAULT_MAX_LENGTH, END_LATENT_TOKEN, FALLBACK_IMAGE_SIZE, IMAGE_TOKEN, LATENT_TOKEN, LOSS_IGNORE_INDEX, START_LATENT_TOKEN
from .exceptions import DataLoadingError, DatasetError, ImageProcessingError
logger = logging.getLogger(__name__)

class SupervisedDataset(Dataset):

    def __init__(self, data_path: str, data_dir: str, test_limit: Optional[int]=None) -> None:
        super().__init__()
        self._validate_paths(data_path, data_dir)
        self.data = self._load_data(data_path, test_limit)
        self.data_dir = data_dir
        self._original_data = self.data.copy()
        logger.info(f'Loaded {len(self.data)} samples from {data_path}')

    def _validate_paths(self, data_path: str, data_dir: str) -> None:
        for path, name in [(data_path, 'Data file'), (data_dir, 'Data directory')]:
            if not os.path.exists(path):
                raise DataLoadingError(f'{name} not found: {path}')

    def _load_data(self, data_path: str, test_limit: Optional[int]) -> List[Dict]:
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise DataLoadingError(f'Failed to load data from {data_path}: {e}') from e
        if test_limit is not None:
            data = data[:test_limit]
            logger.info(f'Limited dataset to {test_limit} samples for testing')
        return data

    def __len__(self) -> int:
        return len(self.data)

    def apply_progressive_curriculum(self, scheduled_stage: int, c_thought: int, max_latent_stage: int, uniform_prob: float=0.0, pad_latent_to_max: bool=False, no_cot: bool=False) -> None:
        logger.info(f'Applying progressive curriculum for stage {scheduled_stage}')
        self.data = create_progressive_latent_dataset(scheduled_stage=scheduled_stage, base_dataset=self._original_data, c_thought=c_thought, max_latent_stage=max_latent_stage, uniform_prob=uniform_prob, pad_latent_to_max=pad_latent_to_max, no_cot=no_cot)
        logger.info(f'Dataset updated with {len(self.data)} curriculum samples')

    def __getitem__(self, index: int) -> Dict[str, Union[Image.Image, str]]:
        if index >= len(self.data):
            raise DatasetError(f'Index {index} out of range for dataset of size {len(self.data)}')
        item = self.data[index]
        self._validate_item(item, index)
        image = self._load_image(item['image'])
        question = item['question']
        answer = item.get('answer', item.get('direct_answer', ''))
        result = {'image': image, 'question': question, 'answer': answer}
        if (steps := item.get('steps')):
            result['steps'] = steps
        if (reasoning := item.get('reasoning')):
            result['reasoning'] = reasoning
        return result

    def _validate_item(self, item: Dict, index: int) -> None:
        required_fields = ['image', 'question']
        missing_fields = [field for field in required_fields if field not in item]
        if missing_fields:
            raise DatasetError(f'Sample {index} missing fields: {missing_fields}')

    def _load_image(self, image_file: str) -> Image.Image:
        if os.path.isabs(image_file):
            image_path = image_file
        else:
            image_path = os.path.join(self.data_dir, image_file)
        try:
            if not os.path.exists(image_path):
                logger.warning(f'Image file not found: {image_path}')
                return Image.new('RGB', (FALLBACK_IMAGE_SIZE, FALLBACK_IMAGE_SIZE), color=(0, 0, 0))
            image = Image.open(image_path)
            return image.convert('RGB')
        except (OSError, IOError) as e:
            logger.warning(f'Failed to load image {image_path}: {e}')
            return Image.new('RGB', (FALLBACK_IMAGE_SIZE, FALLBACK_IMAGE_SIZE), color=(0, 0, 0))
        except Exception as e:
            logger.warning(f'Unexpected error loading image {image_path}: {e}')
            return Image.new('RGB', (FALLBACK_IMAGE_SIZE, FALLBACK_IMAGE_SIZE), color=(0, 0, 0))

def collate_fn(batch: List[Dict[str, Any]], tokenizer: Any, image_processor: Any) -> Dict[str, torch.Tensor]:
    if not batch:
        raise DatasetError('Empty batch provided to collate function')
    try:
        images = [item['image'] for item in batch]
        questions = [item['question'] for item in batch]
        answers = [item['answer'] for item in batch]
        pixel_values = _process_images(images, image_processor)
        full_texts, prompts = _create_chat_formatted_texts(batch, questions, answers)
        full_encodings = tokenizer(full_texts, padding=True, truncation=True, max_length=DEFAULT_MAX_LENGTH, return_tensors='pt', add_special_tokens=True)
        labels = _create_training_labels(full_encodings['input_ids'], prompts, tokenizer)
        return {'pixel_values': pixel_values, 'input_ids': full_encodings['input_ids'], 'attention_mask': full_encodings['attention_mask'], 'labels': labels, 'questions': questions, 'answers': answers}
    except Exception as e:
        raise DatasetError(f'Failed to collate batch: {e}') from e

def _create_chat_formatted_texts(batch: List[Dict[str, Any]], questions: List[str], answers: List[str]) -> Tuple[List[str], List[str]]:
    full_texts = []
    prompts = []
    for i, (question, answer) in enumerate(zip(questions, answers)):
        assistant_part = _build_assistant_response(batch[i], answer)
        prompt = f'<|im_start|>user\n{IMAGE_TOKEN}\n{question}<|im_end|><|im_start|>assistant\n'
        full_text = f'{prompt}{assistant_part}'
        full_texts.append(full_text)
        prompts.append(prompt)
    return (full_texts, prompts)

def _build_assistant_response(item: Dict[str, Any], answer: str) -> str:
    if (reasoning_text := item.get('reasoning', '')):
        return f'{reasoning_text} The answer is {answer}'
    elif (reasoning_steps := item.get('steps', [])):
        reasoning_combined = ' '.join(reasoning_steps)
        return f'{reasoning_combined} The answer is {answer}'
    return answer

def _process_images(images: List[Image.Image], image_processor: Any) -> torch.Tensor:
    try:
        processed = image_processor(images, return_tensors='pt')
        return processed['pixel_values']
    except Exception as e:
        raise ImageProcessingError(f'Error during image processing: {e}') from e

def _create_training_labels(input_ids: torch.Tensor, prompts: List[str], tokenizer: Any) -> torch.Tensor:
    labels = input_ids.clone()
    for i, prompt in enumerate(prompts):
        prompt_tokens = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids[0]
        prompt_length = len(prompt_tokens)
        labels[i, :prompt_length] = LOSS_IGNORE_INDEX
    return labels

def create_progressive_latent_dataset(scheduled_stage: int, base_dataset: List[Dict], c_thought: int, max_latent_stage: int, uniform_prob: float=0.0, pad_latent_to_max: bool=False, no_cot: bool=False) -> List[Dict]:
    logger.info(f'Creating progressive latent dataset for stage {scheduled_stage}')
    processed_samples = []
    for sample in base_dataset:
        steps = _parse_reasoning_steps(sample.get('steps', []))
        stage_to_train = random.choice(range(len(steps) + 1)) if random.random() < uniform_prob else scheduled_stage
        n_skip_steps, n_latent_tokens = _calculate_curriculum_params(stage_to_train, max_latent_stage, steps, pad_latent_to_max, no_cot)
        total_latent_tokens = n_latent_tokens * c_thought
        reasoning_text = _build_reasoning_text(total_latent_tokens, steps, n_skip_steps)
        processed_sample = {**sample, 'reasoning': reasoning_text, 'stage': stage_to_train, 'n_latent_tokens': total_latent_tokens, 'n_skip_steps': n_skip_steps}
        processed_samples.append(processed_sample)
    return processed_samples

def _parse_reasoning_steps(steps: Union[List[str], str]) -> List[str]:
    if isinstance(steps, str):
        return [step.strip() for step in steps.split('\n') if step.strip()]
    return steps

def _calculate_curriculum_params(stage_to_train: int, max_latent_stage: int, steps: List[str], pad_latent_to_max: bool, no_cot: bool) -> Tuple[int, int]:
    if no_cot:
        return (100, 0)
    if stage_to_train > max_latent_stage:
        n_skip_steps = 10000
        n_latent_tokens = max_latent_stage if pad_latent_to_max else min(len(steps), max_latent_stage)
    else:
        n_skip_steps = stage_to_train
        n_latent_tokens = stage_to_train
    return (n_skip_steps, n_latent_tokens)

def _build_reasoning_text(total_latent_tokens: int, steps: List[str], n_skip_steps: int) -> str:
    reasoning_parts = []
    if total_latent_tokens > 0:
        latent_block = ' '.join([LATENT_TOKEN] * total_latent_tokens)
        reasoning_parts.append(f'{START_LATENT_TOKEN} {latent_block} {END_LATENT_TOKEN}')
    if (remaining_steps := steps[n_skip_steps:]):
        reasoning_parts.append(' '.join(remaining_steps))
    return ' '.join(reasoning_parts).strip()