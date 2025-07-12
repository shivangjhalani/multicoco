"""
Constants used throughout the MultiCoCo package.

Centralizes all magic numbers, default values, and configuration constants
to improve maintainability and avoid scattered magic numbers.
"""

from enum import Enum
from typing import Dict, List

# ============================================================================
# SPECIAL TOKENS
# ============================================================================

# Individual token definitions
END_LATENT_TOKEN = '<|end_latent|>'
IMAGE_TOKEN = '<image>'
IMG_CONTEXT_TOKEN = '<img>'
LATENT_TOKEN = '<|latent|>'
START_LATENT_TOKEN = '<|start_latent|>'

# Collection for easy tokenizer addition
COCONUT_SPECIAL_TOKENS = [START_LATENT_TOKEN, LATENT_TOKEN, END_LATENT_TOKEN]

# ============================================================================
# ANSWER CHOICE VALIDATION
# ============================================================================

class AnswerChoice(Enum):
    """Valid answer choice mappings for multiple choice questions."""
    A = '0'
    B = '1'
    C = '2'
    D = '3'

CHOICE_MAPPINGS: Dict[str, str] = {
    'a': AnswerChoice.A.value, 'first': AnswerChoice.A.value, 'zero': AnswerChoice.A.value,
    'b': AnswerChoice.B.value, 'one': AnswerChoice.B.value, 'second': AnswerChoice.B.value,
    'c': AnswerChoice.C.value, 'third': AnswerChoice.C.value, 'two': AnswerChoice.C.value,
    'd': AnswerChoice.D.value, 'fourth': AnswerChoice.D.value, 'three': AnswerChoice.D.value
}

VALID_CHOICE_NUMBERS: List[str] = [choice.value for choice in AnswerChoice]

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

DEFAULT_DTYPE = "bfloat16"
# Removed unused constants DEFAULT_IMAGE_SIZE and DEFAULT_NUM_BEAMS
DEFAULT_MODEL_NAME = "OpenGVLab/InternVL3-1B-Pretrained"

# ============================================================================
# GENERATION PARAMETERS
# ============================================================================

DEFAULT_MAX_LENGTH = 768
DEFAULT_MAX_NEW_TOKENS = 256

# ============================================================================
# TRAINING DEFAULTS
# ============================================================================

DEFAULT_BATCH_SIZE = 2
DEFAULT_EVAL_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_NUM_EPOCHS = 3

# ============================================================================
# COCONUT TRAINING PARAMETERS
# ============================================================================

DEFAULT_C_THOUGHT = 0
DEFAULT_MAX_LATENT_STAGE = 0
LOSS_IGNORE_INDEX = -100

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

TEST_DATASET_LIMIT = 20

# ============================================================================
# LOGGING AND OUTPUT
# ============================================================================

DEFAULT_LOG_DIR = 'logs'
DEFAULT_OUTPUT_DIR = 'checkpoints'
DEFAULT_EVAL_LOG_FORMAT = 'console'
