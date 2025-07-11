"""
Constants used throughout the MultiCoCo package.

Centralizes all magic numbers, default values, and configuration constants
to improve maintainability and avoid scattered magic numbers.
"""

# ============================================================================
# ANSWER CHOICE VALIDATION
# ============================================================================

CHOICE_MAPPINGS = {
    'a': '0', 'first': '0', 'zero': '0',
    'b': '1', 'one': '1', 'second': '1', 
    'c': '2', 'third': '2', 'two': '2',
    'd': '3', 'fourth': '3', 'three': '3'
}

VALID_CHOICE_NUMBERS = ['0', '1', '2', '3']

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
# EVALUATION TYPES
# ============================================================================

EVAL_TYPES = {
    'COCONUT': 'coconut',
    'COT': 'cot', 
    'VANILLA': 'vanilla'
}

# ============================================================================
# GENERATION PARAMETERS
# ============================================================================

DEFAULT_INPUT_MAX_LENGTH = 512
DEFAULT_MAX_LENGTH = 768
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_NUM_BEAMS = 1
DEFAULT_TARGET_MAX_LENGTH = 256

# ============================================================================
# LOGGING AND OUTPUT
# ============================================================================

DEFAULT_LOG_DIR = 'logs'
DEFAULT_OUTPUT_DIR = './output'
EVAL_LOG_SEPARATOR = "=" * 50
SAMPLE_LOG_SEPARATOR = "-" * 40

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

DEFAULT_DTYPE = "bfloat16"
DEFAULT_IMAGE_SIZE = 448
DEFAULT_MODEL_NAME = "OpenGVLab/InternVL3-1B-Pretrained"

# ============================================================================
# SPECIAL TOKENS
# ============================================================================

# Collection for easy tokenizer addition
COCONUT_SPECIAL_TOKENS = [
    '<|start_latent|>',
    '<|latent|>',
    '<|end_latent|>'
]

# Individual token definitions
END_LATENT_TOKEN = '<|end_latent|>'
IMAGE_TOKEN = '<image>'
IMG_CONTEXT_TOKEN = '<img>'
LATENT_TOKEN = '<|latent|>'
START_LATENT_TOKEN = '<|start_latent|>'

# ============================================================================
# TRAINING DEFAULTS
# ============================================================================

DEFAULT_BATCH_SIZE = 2
DEFAULT_EVAL_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_NUM_EPOCHS = 3 