"""
Answer extraction utilities for multiple choice question evaluation.

Provides utility functions for extracting answer choices from generated text
using various patterns and heuristics.
"""

import logging
import re
from typing import List, Pattern, Tuple

from .constants import CHOICE_MAPPINGS, VALID_CHOICE_NUMBERS
from .exceptions import AnswerExtractionError

logger = logging.getLogger(__name__)

# Compiled regex patterns for performance (in order of specificity)
EXTRACTION_PATTERNS: List[Tuple[Pattern[str], str]] = [
    (re.compile(r'(\d+)\s*:\s*[a-zA-Z]'), "number_colon"),
    (re.compile(r'^(\d+)(?:\s|$)'), "leading_number"),
    (re.compile(r'(?:answer is|choice is|option is)\s*(\d+)', re.IGNORECASE), "answer_format"),
    (re.compile(r'(\d+)'), "any_digit"),
]


def extract_answer_choice(generated_text: str, is_cot: bool = False) -> str:
    """
    Extract answer choice from generated text using multiple strategies.
    
    Employs a hierarchical approach to answer extraction:
    1. Number-colon format (e.g., "1 : explanation")
    2. Leading number format
    3. "Answer is X" format
    4. Any valid digit in text
    5. Word mappings (first, second, etc.)
    
    Args:
        generated_text: Raw generated text from model
        is_cot: Whether this is chain-of-thought evaluation (unused but kept for compatibility)
        
    Returns:
        Extracted answer choice as string (0-3)
        
    Raises:
        AnswerExtractionError: If extraction fails
    """
    try:
        text = generated_text.strip()
        if not text:
            return ""
        
        # Try regex patterns in order of specificity
        for pattern, pattern_name in EXTRACTION_PATTERNS:
            result = _extract_with_pattern(pattern, text, pattern_name)
            if result in VALID_CHOICE_NUMBERS:
                return result
        
        # Try word mappings as last resort
        result = _extract_word_mappings(text)
        if result in VALID_CHOICE_NUMBERS:
            return result
        
        # If no valid choice found, return original for debugging
        logger.warning(f"Could not extract valid choice from: {text[:100]}")
        return text.strip()
        
    except Exception as e:
        raise AnswerExtractionError(
            f"Failed to extract answer from '{generated_text}': {e}"
        ) from e


def _extract_with_pattern(pattern: Pattern[str], text: str, pattern_name: str) -> str:
    """Extract answer using a compiled regex pattern."""
    if pattern_name == "any_digit":
        # For any_digit, find all matches and return first valid one
        return next(
            (match for match in pattern.findall(text) if match in VALID_CHOICE_NUMBERS),
            ""
        )
    
    # For other patterns, get first match
    match = pattern.search(text)
    return match.group(1) if match else ""


def _extract_word_mappings(text: str) -> str:
    """Extract using word-to-number mappings."""
    text_lower = text.lower()
    for word, choice in CHOICE_MAPPINGS.items():
        if word in text_lower:
            return choice
    return "" 