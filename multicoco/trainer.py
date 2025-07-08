"""
Custom trainer for MultiCoCo with CoCoNut support.

This module provides a custom trainer class that extends the HuggingFace Trainer
to support CoCoNut (Chain of Continuous Thought) training and evaluation with
multimodal models.
"""

import os
import re
import logging
from types import SimpleNamespace
from typing import Optional, List, Tuple, Dict, Any, Union
import random

# ** Core libraries
import torch
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

# ** Transformers components
from transformers import Trainer
from transformers.trainer_pt_utils import (
    find_batch_size,
    nested_concat,
    nested_numpify,
    nested_truncate,
    nested_detach
)
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import LabelSmoother
from transformers.trainer_utils import EvalPrediction

# ** Local imports
from .constants import (
    VALID_CHOICE_NUMBERS,
    CHOICE_MAPPINGS,
    LOSS_IGNORE_INDEX,
    DEFAULT_MAX_NEW_TOKENS,
    IMAGE_TOKEN,
    EVAL_LOG_SEPARATOR,
    SAMPLE_LOG_SEPARATOR
)
from .exceptions import (
    EvaluationError,
    AnswerExtractionError,
    CoCoNutTrainingError,
    GenerationError
)

logger = logging.getLogger(__name__)


class EvaluationResult:
    """Container for evaluation results."""
    
    def __init__(self, metrics: Dict[str, float], num_samples: int):
        self.metrics = metrics
        self.num_samples = num_samples


class CoCoTrainer(Trainer):
    """
    Custom trainer for CoCoNut training and evaluation.
    
    This trainer extends the HuggingFace Trainer to support:
    - CoCoNut (Chain of Continuous Thought) training with progressive masking
    - Sophisticated answer extraction for multiple choice questions
    - Detailed evaluation logging
    - Proper dtype handling for multimodal inputs
    
    Attributes:
        best_val_acc: Best validation accuracy achieved
        coconut_enabled: Whether CoCoNut training is enabled
        c_thought: CoCoNut thought scaling factor
        max_latent_stage: Maximum CoCoNut training stage
        current_stage: Current CoCoNut training stage
        thought_token_id: ID for thought tokens
        start_thought_id: ID for start thought tokens
        end_thought_id: ID for end thought tokens
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the CoCoTrainer.
        
        Args:
            *args: Arguments passed to parent Trainer
            **kwargs: Keyword arguments passed to parent Trainer
        """
        # Remove processor argument as it's handled by parent class
        if 'processor' in kwargs:
            kwargs.pop('processor')
            
        super().__init__(*args, **kwargs)
        
        # Initialize trainer state
        self.best_val_acc = 0.0

        # Initialize CoCoNut parameters
        self._initialize_coconut_config()
        
        # Initialize special token IDs
        self._initialize_special_tokens()
        
        logger.info(f"CoCoTrainer initialized with CoCoNut={'enabled' if self.coconut_enabled else 'disabled'}")

    def _initialize_coconut_config(self) -> None:
        """Initialize CoCoNut configuration from training arguments."""
        eval_config = getattr(self.args, 'eval_config', {})
        self.coconut_enabled = eval_config.get('coconut', False)
        self.c_thought = getattr(self.args, 'c_thought', 0)
        self.max_latent_stage = getattr(self.args, 'max_latent_stage', 0)
        self.current_stage = 0

    def _initialize_special_tokens(self) -> None:
        """Initialize special token IDs for CoCoNut training."""
        if hasattr(self.args, 'thought_token_id'):
            self.thought_token_id = self.args.thought_token_id
            self.start_thought_id = self.args.start_thought_id
            self.end_thought_id = self.args.end_thought_id
        else:
            # Fallback to processing_class if not provided
            self.thought_token_id = self._safe_convert_token('<thought>')
            self.start_thought_id = self._safe_convert_token('<start_thought>')
            self.end_thought_id = self._safe_convert_token('<end_thought>')

    def _safe_convert_token(self, token: str) -> Optional[int]:
        """Safely convert token to ID, returning None if not found."""
        try:
            return self.processing_class.convert_tokens_to_ids(token)
        except (AttributeError, KeyError):
            logger.warning(f"Token '{token}' not found in tokenizer vocabulary")
            return None

    def _create_generation_config(self) -> Dict[str, Any]:
        """
        Create generation configuration for evaluation.
        
        Returns:
            Dictionary of generation parameters
        """
        gen_kwargs = getattr(self.args, "generation_kwargs", {}) or {}
        
        # Set default generation parameters
        defaults = {
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "do_sample": False,
            "num_beams": 1,
        }
        
        for key, value in defaults.items():
            if key not in gen_kwargs:
                gen_kwargs[key] = value
        
        # Add pad token ID to suppress warnings
        if self.processing_class.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = self.processing_class.pad_token_id
        
        return gen_kwargs

    def apply_coconut_masking(
        self, 
        input_ids: torch.Tensor, 
        labels: torch.Tensor, 
        stage: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply CoCoNut masking strategy for latent reasoning.
        
        This method implements progressive masking of thought tokens based on
        the current training stage, encouraging the model to internalize reasoning.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            labels: Labels for training [batch_size, seq_len]
            stage: Current training stage (0 = no masking, higher = more masking)
            
        Returns:
            Tuple of (modified_input_ids, modified_labels)
        """
        if not self.coconut_enabled or stage == 0:
            return input_ids, labels
        
        try:
            batch_size, seq_len = input_ids.shape
            modified_input_ids = input_ids.clone()
            modified_labels = labels.clone()
            
            for i in range(batch_size):
                modified_labels[i] = self._apply_sample_masking(
                    input_ids[i], modified_labels[i], stage
                )
            
            return modified_input_ids, modified_labels
            
        except Exception as e:
            raise CoCoNutTrainingError(f"Failed to apply CoCoNut masking: {e}")

    def _apply_sample_masking(
        self, 
        sample_input_ids: torch.Tensor, 
        sample_labels: torch.Tensor, 
        stage: int
    ) -> torch.Tensor:
        """Apply masking to a single sample."""
        # Find thought token positions
        thought_positions = self._find_token_positions(sample_input_ids, self.thought_token_id)
        start_positions = self._find_token_positions(sample_input_ids, self.start_thought_id)
        end_positions = self._find_token_positions(sample_input_ids, self.end_thought_id)
        
        # Apply progressive masking to thought tokens
        if len(thought_positions) > 0:
            sample_labels = self._mask_thought_tokens(
                sample_labels, thought_positions, stage
            )
        
        # Apply masking to thought content between start/end tokens
        if len(start_positions) > 0 and len(end_positions) > 0:
            sample_labels = self._mask_thought_content(
                sample_labels, start_positions, end_positions, stage
            )
        
        return sample_labels

    def _find_token_positions(self, input_ids: torch.Tensor, token_id: Optional[int]) -> List[int]:
        """Find positions of a specific token in input_ids."""
        if token_id is None:
            return []
        return (input_ids == token_id).nonzero(as_tuple=True)[0].tolist()

    def _mask_thought_tokens(
        self, 
        labels: torch.Tensor, 
        positions: List[int], 
        stage: int
    ) -> torch.Tensor:
        """Mask thought tokens based on stage and c_thought parameter."""
        if not positions:
            return labels
        
        # Calculate how many tokens to mask
        mask_ratio = min(stage / self.max_latent_stage, 1.0)
        num_to_mask = int(len(positions) * mask_ratio * self.c_thought / 10.0)
        
        if num_to_mask > 0:
            mask_indices = random.sample(positions, min(num_to_mask, len(positions)))
            for pos in mask_indices:
                labels[pos] = LOSS_IGNORE_INDEX
        
        return labels

    def _mask_thought_content(
        self, 
        labels: torch.Tensor, 
        start_positions: List[int], 
        end_positions: List[int], 
        stage: int
    ) -> torch.Tensor:
        """Mask content between start and end thought tokens."""
        for start_pos, end_pos in zip(start_positions, end_positions):
            if start_pos < end_pos:
                thought_length = end_pos - start_pos - 1
                if thought_length > 0 and stage > 0:
                    mask_ratio = min(stage / self.max_latent_stage, 1.0)
                    num_to_mask = int(thought_length * mask_ratio)
                    
                    if num_to_mask > 0:
                        thought_range = list(range(start_pos + 1, end_pos))
                        mask_positions = random.sample(
                            thought_range, min(num_to_mask, len(thought_range))
                        )
                        
                        for pos in mask_positions:
                            labels[pos] = LOSS_IGNORE_INDEX
        
        return labels

    def compute_loss(
        self, 
        model: nn.Module, 
        inputs: Dict[str, torch.Tensor], 
        return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Override compute_loss to implement CoCoNut training logic.
        
        Args:
            model: The model to compute loss for
            inputs: Input batch
            return_outputs: Whether to return model outputs
            
        Returns:
            Loss tensor or tuple of (loss, outputs)
        """
        if self.coconut_enabled:
            inputs = self._apply_coconut_masking_to_inputs(inputs)
        
        return super().compute_loss(model, inputs, return_outputs)

    def _apply_coconut_masking_to_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply CoCoNut masking to input batch."""
        input_ids = inputs.get('input_ids')
        labels = inputs.get('labels')
        
        if input_ids is not None and labels is not None:
            modified_input_ids, modified_labels = self.apply_coconut_masking(
                input_ids, labels, self.current_stage
            )
            inputs['input_ids'] = modified_input_ids
            inputs['labels'] = modified_labels
        
        return inputs

    def extract_answer_choice(self, generated_text: str, is_cot: bool = False) -> str:
        """
        Extract answer choice from generated text with sophisticated pattern matching.
        
        This method handles various answer formats commonly seen in multiple choice
        questions and extracts the choice number (0, 1, 2, 3).
        
        Args:
            generated_text: Text generated by the model
            is_cot: Whether this is Chain of Thought generation
            
        Returns:
            Extracted answer choice as string
            
        Raises:
            AnswerExtractionError: If extraction fails
        """
        try:
            text = generated_text.strip()
            
            # Handle CoT format - look for "the answer is" pattern first
            if is_cot and "the answer is" in text.lower():
                text = text.lower().split("the answer is")[-1].strip()
            
            # Try different extraction patterns in order of specificity
            extractors = [
                self._extract_number_colon_format,
                self._extract_leading_number,
                self._extract_answer_is_format,
                self._extract_any_digit,
                self._extract_word_mappings
            ]
            
            for extractor in extractors:
                result = extractor(text)
                if result in VALID_CHOICE_NUMBERS:
                    return result
            
            # If no valid choice found, return original for debugging
            logger.warning(f"Could not extract valid choice from: {text[:100]}")
            return text.strip()
            
        except Exception as e:
            raise AnswerExtractionError(f"Failed to extract answer from '{generated_text}': {e}")

    def _extract_number_colon_format(self, text: str) -> str:
        """Extract from "X : description" format."""
        match = re.search(r'(\d+)\s*:\s*[a-zA-Z]', text)
        return match.group(1) if match else ""

    def _extract_leading_number(self, text: str) -> str:
        """Extract number at the start of text."""
        match = re.search(r'^(\d+)(?:\s|$)', text.strip())
        return match.group(1) if match else ""

    def _extract_answer_is_format(self, text: str) -> str:
        """Extract from "The answer is X" format."""
        match = re.search(r'(?:answer is|choice is|option is)\s*(\d+)', text.lower())
        return match.group(1) if match else ""

    def _extract_any_digit(self, text: str) -> str:
        """Extract any valid digit from text."""
        matches = re.findall(r'(\d+)', text)
        for match in matches:
            if match in VALID_CHOICE_NUMBERS:
                return match
        return ""

    def _extract_word_mappings(self, text: str) -> str:
        """Extract using word-to-number mappings."""
        text_lower = text.lower()
        for word, choice in CHOICE_MAPPINGS.items():
            if word in text_lower:
                return choice
        return ""

    def compute_metrics(self, p: EvalPrediction) -> Dict[str, float]:
        """
        Compute metrics for evaluation.
        
        This is a placeholder as the evaluation_loop calculates metrics directly.
        
        Args:
            p: Evaluation predictions
            
        Returns:
            Empty dictionary (metrics computed in evaluation_loop)
        """
        return {}

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> SimpleNamespace:
        """
        Custom evaluation loop with detailed logging and answer extraction.
        
        Args:
            dataloader: DataLoader for evaluation
            description: Description for progress bar
            prediction_loss_only: Whether to only compute loss
            ignore_keys: Keys to ignore in outputs
            metric_key_prefix: Prefix for metric keys
            
        Returns:
            SimpleNamespace with metrics and evaluation results
        """
        try:
            # Prepare model and evaluation state
            model = self._wrap_model(self.model, training=False, dataloader=dataloader)
            model.eval()
            self.callback_handler.eval_dataloader = dataloader

            # Initialize result containers
            all_predictions = []
            all_labels = []
            all_questions = []

            # Set up logging
            log_file_path = self._setup_evaluation_logging()
            
            # Run evaluation loop
            with open(log_file_path, 'w', encoding='utf-8') as log_file:
                self._write_evaluation_header(log_file)
                
                for step, inputs in enumerate(tqdm(dataloader, desc=description)):
                    # Process batch
                    batch_results = self._process_evaluation_batch(inputs, model, log_file)
                    
                    # Accumulate results
                    all_predictions.extend(batch_results['predictions'])
                    all_labels.extend(batch_results['labels'])
                    all_questions.extend(batch_results['questions'])

                # Compute final metrics
                metrics = self._compute_final_metrics(
                    all_predictions, all_labels, metric_key_prefix
                )
                
                # Write summary
                self._write_evaluation_summary(log_file, metrics, len(all_labels))

            # Log metrics
            self.log(metrics)

            return SimpleNamespace(
                metrics=metrics,
                num_samples=len(all_labels),
                eval_preds=None
            )
            
        except Exception as e:
            raise EvaluationError(f"Evaluation loop failed: {e}")

    def _setup_evaluation_logging(self) -> str:
        """Set up logging for evaluation."""
        log_dir = getattr(self.args, 'log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Determine evaluation type
        eval_config = self.args.eval_config
        is_cot = eval_config.get('cot', False)
        is_coconut = eval_config.get('coconut', False)
        eval_type = "coconut" if is_coconut else "cot" if is_cot else "vanilla"
        
        return os.path.join(log_dir, f'evaluation_{eval_type}.log')

    def _write_evaluation_header(self, log_file) -> None:
        """Write evaluation header to log file."""
        eval_config = self.args.eval_config
        eval_type = self._get_eval_type_name(eval_config)
        
        log_file.write(f"Evaluation Results - {eval_type.upper()}\n")
        log_file.write(EVAL_LOG_SEPARATOR + "\n\n")

    def _get_eval_type_name(self, eval_config: Dict[str, bool]) -> str:
        """Get evaluation type name from config."""
        if eval_config.get('coconut', False):
            return "coconut"
        elif eval_config.get('cot', False):
            return "cot"
        else:
            return "vanilla"

    def _process_evaluation_batch(
        self, 
        inputs: Dict[str, torch.Tensor], 
        model: nn.Module, 
        log_file
    ) -> Dict[str, List[str]]:
        """Process a single evaluation batch."""
        # Extract batch components
        questions = inputs.pop("questions")
        answers = inputs.pop("answers")
        pixel_values = inputs["pixel_values"].to(self.args.device)
        
        predictions = []
        
        # Generate predictions for each sample
        for i, question in enumerate(questions):
            try:
                prediction = self._generate_single_prediction(
                    question, pixel_values[i:i+1], model
                )
                predictions.append(prediction)
                
                # Log sample details
                self._log_sample_result(
                    log_file, question, answers[i], prediction, i
                )
                
            except Exception as e:
                logger.warning(f"Failed to generate prediction for sample {i}: {e}")
                predictions.append("")
        
        return {
            'predictions': predictions,
            'labels': answers,
            'questions': questions
        }

    def _generate_single_prediction(
        self, 
        question: str, 
        pixel_values: torch.Tensor, 
        model: nn.Module
    ) -> str:
        """Generate prediction for a single question-image pair."""
        try:
            # Format input text with image token
            user_content = f"{IMAGE_TOKEN}\n{question}"
            
            # Create generation config
            generation_config = self._create_generation_config()
            
            # Access underlying model
            underlying_model = model.model if hasattr(model, 'model') else model
            
            # Ensure correct dtype
            pixel_values = self._ensure_correct_dtype(pixel_values, underlying_model)
            
            # Generate response
            response = underlying_model.chat(
                self.processing_class,
                pixel_values,
                user_content,
                generation_config
            )
            
            # Clean up response
            return self._clean_generated_response(response)
            
        except Exception as e:
            raise GenerationError(f"Failed to generate prediction: {e}")

    def _ensure_correct_dtype(self, pixel_values: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Ensure pixel values have correct dtype for model."""
        if hasattr(model, 'dtype'):
            target_dtype = model.dtype
        elif hasattr(model, 'vision_model') and hasattr(model.vision_model, 'dtype'):
            target_dtype = model.vision_model.dtype
        else:
            target_dtype = torch.bfloat16  # Default
        
        return pixel_values.to(target_dtype)

    def _clean_generated_response(self, response: str) -> str:
        """Clean up generated response by removing thought tokens."""
        if self.coconut_enabled:
            # Remove thought tokens that might have been generated
            for token in ['<thought>', '<start_thought>', '<end_thought>']:
                response = response.replace(token, '')
        
        return response.strip()

    def _log_sample_result(
        self, 
        log_file, 
        question: str, 
        ground_truth: str, 
        prediction: str, 
        sample_idx: int
    ) -> None:
        """Log detailed result for a single sample."""
        eval_config = self.args.eval_config
        is_cot = eval_config.get('cot', False)
        
        # Extract answer for correctness check
        extracted_answer = self.extract_answer_choice(prediction, is_cot)
        is_correct = extracted_answer == ground_truth.strip()
        tokens_generated = len(self.processing_class.tokenize(prediction))
        
        # Write to log file
        log_file.write(SAMPLE_LOG_SEPARATOR + "\n")
        log_file.write(f"Question: {question}\n")
        log_file.write(f"Ground Truth Answer: {ground_truth}\n")
        log_file.write(f"Generated Answer: {prediction}\n")
        log_file.write(f"Extracted Answer: {extracted_answer}\n")
        log_file.write(f"Tokens Generated: {tokens_generated}\n")
        log_file.write(f"Correct: {'Yes' if is_correct else 'No'}\n")
        log_file.write(SAMPLE_LOG_SEPARATOR + "\n\n")

    def _compute_final_metrics(
        self, 
        predictions: List[str], 
        labels: List[str], 
        metric_key_prefix: str
    ) -> Dict[str, float]:
        """Compute final evaluation metrics."""
        eval_config = self.args.eval_config
        is_cot = eval_config.get('cot', False)
        
        correct = 0
        total = len(labels)
        
        for pred, label in zip(predictions, labels):
            extracted_answer = self.extract_answer_choice(pred, is_cot)
            if extracted_answer == label.strip():
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        # Build metrics dictionary
        metrics = {
            f"{metric_key_prefix}_accuracy": accuracy,
            f"{metric_key_prefix}_loss": -1.0,  # Placeholder
        }
        
        # Add CoCoNut stage information if applicable
        if self.coconut_enabled:
            metrics.update({
                f"{metric_key_prefix}_coconut_stage": self.current_stage,
                f"{metric_key_prefix}_max_latent_stage": self.max_latent_stage
            })

        return metrics

    def _write_evaluation_summary(
        self, 
        log_file, 
        metrics: Dict[str, float], 
        num_samples: int
    ) -> None:
        """Write evaluation summary to log file."""
        accuracy = metrics.get('eval_accuracy', 0.0)
        correct = int(accuracy * num_samples)
        
        log_file.write("Final Results:\n")
        log_file.write(f"Total Samples: {num_samples}\n")
        log_file.write(f"Correct Predictions: {correct}\n")
        log_file.write(f"Accuracy: {accuracy:.4f}\n")

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Custom prediction step for generation-based evaluation.
        
        Args:
            model: Model to use for prediction
            inputs: Input batch
            prediction_loss_only: Whether to only compute loss
            ignore_keys: Keys to ignore in outputs
            
        Returns:
            Tuple of (loss, predictions, labels)
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        gen_kwargs = self._create_generation_config()

        try:
            generated_tokens = self.model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )

            # In generation mode, there's no loss
        return (None, generated_tokens, None)
            
        except Exception as e:
            raise GenerationError(f"Prediction step failed: {e}")
