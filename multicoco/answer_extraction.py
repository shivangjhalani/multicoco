import logging
import re
from typing import List, Pattern, Tuple

from .constants import CHOICE_MAPPINGS, VALID_CHOICE_NUMBERS
from .exceptions import AnswerExtractionError

logger = logging.getLogger(__name__)

EXTRACTION_PATTERNS: List[Tuple[Pattern[str], str]] = [
    (re.compile(r'(\d+)\s*:\s*[a-zA-Z]'), 'number_colon'),
    (re.compile(r'^(\d+)(?:\s|$)'), 'leading_number'),
    (re.compile(r'(?:answer is|choice is|option is)\s*(\d+)', re.IGNORECASE), 'answer_format'),
    (re.compile(r'(\d+)'), 'any_digit')
]


def extract_answer_choice(generated_text: str, is_cot: bool = False) -> str:
    text = generated_text.strip()
    if not text:
        return ''
    for pattern, pattern_name in EXTRACTION_PATTERNS:
        result = _extract_with_pattern(pattern, text, pattern_name)
        if result in VALID_CHOICE_NUMBERS:
            return result
    result = _extract_word_mappings(text)
    if result in VALID_CHOICE_NUMBERS:
        return result
    logger.warning(f'Could not extract valid choice from: {text[:100]}')
    return text.strip()


def _extract_with_pattern(pattern: Pattern[str], text: str, pattern_name: str) -> str:
    if pattern_name == 'any_digit':
        return next((match for match in pattern.findall(text) if match in VALID_CHOICE_NUMBERS), '')
    match = pattern.search(text)
    return match.group(1) if match else ''


def _extract_word_mappings(text: str) -> str:
    text_lower = text.lower()
    for word, choice in CHOICE_MAPPINGS.items():
        if word in text_lower:
            return choice
    return ''