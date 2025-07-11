"""
Answer extraction utilities for multiple choice question evaluation.

This module provides utility functions for extracting answer choices from
generated text using various patterns and heuristics.
"""

import re
import logging
from typing import List

from .constants import VALID_CHOICE_NUMBERS, CHOICE_MAPPINGS
from .exceptions import AnswerExtractionError

logger = logging.getLogger(__name__)


def extract_answer_choice(generated_text: str, is_cot: bool = False) -> str:
    """
    Extract answer choice from generated text using multiple strategies.
    
    This function employs a hierarchical approach to answer extraction:
    1. Number-colon format (e.g., "1 : explanation")
    2. Leading number format
    3. "Answer is X" format
    4. Any valid digit in text
    5. Word mappings (first, second, etc.)
    
    Args:
        generated_text: Raw generated text from model
        is_cot: Whether this is chain-of-thought evaluation
        
    Returns:
        Extracted answer choice as string (0-3)
        
    Raises:
        AnswerExtractionError: If extraction fails
    """
    try:
        text = generated_text.strip()
        if not text:
            return ""
        
        # Try different extraction patterns in order of specificity
        extractors = [
            _extract_number_colon_format,
            _extract_leading_number,
            _extract_answer_is_format,
            _extract_any_digit,
            _extract_word_mappings
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


def _extract_number_colon_format(text: str) -> str:
    """Extract from "X : description" format."""
    match = re.search(r'(\d+)\s*:\s*[a-zA-Z]', text)
    return match.group(1) if match else ""


def _extract_leading_number(text: str) -> str:
    """Extract number at the start of text."""
    match = re.search(r'^(\d+)(?:\s|$)', text.strip())
    return match.group(1) if match else ""


def _extract_answer_is_format(text: str) -> str:
    """Extract from "The answer is X" format."""
    match = re.search(r'(?:answer is|choice is|option is)\s*(\d+)', text.lower())
    return match.group(1) if match else ""


def _extract_any_digit(text: str) -> str:
    """Extract any valid digit from text."""
    matches = re.findall(r'(\d+)', text)
    for match in matches:
        if match in VALID_CHOICE_NUMBERS:
            return match
    return ""


def _extract_word_mappings(text: str) -> str:
    """Extract using word-to-number mappings."""
    text_lower = text.lower()
    for word, choice in CHOICE_MAPPINGS.items():
        if word in text_lower:
            return choice
    return "" 