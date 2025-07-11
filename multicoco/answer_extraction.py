"""
Answer extraction utilities for multiple choice question evaluation.

Provides utility functions for extracting answer choices from generated text
using various patterns and heuristics.
"""

import re
import logging
from typing import List, Tuple

from .constants import CHOICE_MAPPINGS, VALID_CHOICE_NUMBERS
from .exceptions import AnswerExtractionError

logger = logging.getLogger(__name__)

# Compiled regex patterns for performance
EXTRACTION_PATTERNS = [
    (re.compile(r'(\d+)\s*:\s*[a-zA-Z]'), "number_colon"),
    (re.compile(r'^(\d+)(?:\s|$)'), "leading_number"),
    (re.compile(r'(?:answer is|choice is|option is)\s*(\d+)', re.IGNORECASE), "answer_format"),
    (re.compile(r'(\d+)'), "any_digit"),
]


def extract_answer_choice(generated_text: str, is_cot: bool = False) -> str:
    """
    Extract answer choice from generated text using multiple strategies with truncation handling.
    
    Employs a hierarchical approach to answer extraction:
    1. Number-colon format (e.g., "1 : explanation")
    2. Leading number format
    3. "Answer is X" format
    4. Any valid digit in text
    5. Word mappings (first, second, etc.)
    6. Truncation-aware partial matching
    
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
        
        # Check for truncation indicators
        is_truncated = _detect_truncation(text)
        if is_truncated:
            logger.info(f"Detected truncated response: {text[:50]}...")
        
        # Try regex patterns in order of specificity
        for pattern, pattern_name in EXTRACTION_PATTERNS:
            result = _extract_with_pattern(pattern, text, pattern_name)
            if result in VALID_CHOICE_NUMBERS:
                if is_truncated:
                    logger.info(f"Extracted '{result}' from truncated response using {pattern_name}")
                return result
        
        # Try word mappings as last resort
        result = _extract_word_mappings(text)
        if result in VALID_CHOICE_NUMBERS:
            if is_truncated:
                logger.info(f"Extracted '{result}' from truncated response using word mappings")
            return result
        
        # Handle truncation cases with partial matching
        if is_truncated:
            result = _extract_truncated_answer(text)
            if result in VALID_CHOICE_NUMBERS:
                logger.info(f"Extracted '{result}' from truncated response using partial matching")
                return result
        
        # If no valid choice found, return original for debugging
        warning_msg = f"Could not extract valid choice from: {text[:100]}"
        if is_truncated:
            warning_msg += " [TRUNCATED]"
        logger.warning(warning_msg)
        return text.strip()
        
    except Exception as e:
        raise AnswerExtractionError(
            f"Failed to extract answer from '{generated_text}': {e}"
        )


def _extract_with_pattern(pattern: re.Pattern, text: str, 
                         pattern_name: str) -> str:
    """Extract answer using a compiled regex pattern."""
    if pattern_name == "any_digit":
        # For any_digit, find all matches and return first valid one
        matches = pattern.findall(text)
        for match in matches:
            if match in VALID_CHOICE_NUMBERS:
                return match
        return ""
    else:
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


def _detect_truncation(text: str) -> bool:
    """Detect if response appears to be truncated."""
    # Common truncation indicators
    truncation_indicators = [
        "I need to",
        "Let me",
        "To solve this",
        "First,",
        "The answer is",  # cut off mid-sentence
    ]
    
    # Check if text ends abruptly with common truncation patterns
    text_stripped = text.strip()
    if not text_stripped:
        return True
        
    # Very short responses might be truncated
    if len(text_stripped.split()) < 3:
        return True
        
    # Check if ends mid-sentence (no proper punctuation)
    last_char = text_stripped[-1]
    if last_char not in '.!?':
        # But allow if it's a valid single digit answer
        if text_stripped in VALID_CHOICE_NUMBERS:
            return False
        return True
    
    # Check for incomplete reasoning patterns
    for indicator in truncation_indicators:
        if text_stripped.endswith(indicator):
            return True
    
    return False


def _extract_truncated_answer(text: str) -> str:
    """Extract answer from truncated text using partial matching."""
    # Try to find partial answer patterns even in incomplete text
    text_stripped = text.strip()
    
    # Check if text starts with a valid choice (common in truncated responses)
    if text_stripped and text_stripped[0] in VALID_CHOICE_NUMBERS:
        return text_stripped[0]
    
    # Look for partial answer patterns like "The answer" followed by space and digit
    partial_patterns = [
        r'(?:answer|choice|option)\s*(?:is)?\s*([0-3])',
        r'([0-3])\s*(?:is|\.)',
        r'option\s*([0-3])',
        r'choice\s*([0-3])',
    ]
    
    for pattern in partial_patterns:
        match = re.search(pattern, text_stripped, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Check for answers at the very end (even if cut off)
    words = text_stripped.split()
    if words:
        last_word = words[-1].strip('.,!?:;')
        if last_word in VALID_CHOICE_NUMBERS:
            return last_word
    
    return "" 