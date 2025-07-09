"""
Constants used throughout the MultiCoCo package.

This module centralizes all magic numbers, default values, and configuration
constants to improve maintainability and avoid magic numbers scattered
throughout the codebase.
"""

from typing import Dict, List, Tuple

# Model Configuration
DEFAULT_MODEL_NAME = "OpenGVLab/InternVL3-1B-Pretrained"
DEFAULT_DTYPE = "bfloat16"

# Generation Parameters
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_MAX_LENGTH = 768
DEFAULT_INPUT_MAX_LENGTH = 512
DEFAULT_TARGET_MAX_LENGTH = 256
DEFAULT_NUM_BEAMS = 1

# Image Processing
DEFAULT_IMAGE_SIZE = 448
DEFAULT_MIN_PATCHES = 1
DEFAULT_MAX_PATCHES = 12
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Special Tokens
# Latent reasoning tokens used in CoCoNut curriculum
LATENT_TOKEN = '<|latent|>'
START_LATENT_TOKEN = '<|start_latent|>'
END_LATENT_TOKEN = '<|end_latent|>'

# Collection for easy tokenizer addition
COCONUT_SPECIAL_TOKENS = [START_LATENT_TOKEN, LATENT_TOKEN, END_LATENT_TOKEN]

IMAGE_TOKEN = '<image>'
IMG_CONTEXT_TOKEN = '<img>'

# CoCoNut Training Parameters
DEFAULT_C_THOUGHT = 0
DEFAULT_MAX_LATENT_STAGE = 0
LOSS_IGNORE_INDEX = -100

# Answer Choice Validation
VALID_CHOICE_NUMBERS = ['0', '1', '2', '3']
CHOICE_MAPPINGS = {
    'first': '0', 'zero': '0', 'a': '0',
    'second': '1', 'one': '1', 'b': '1', 
    'third': '2', 'two': '2', 'c': '2',
    'fourth': '3', 'three': '3', 'd': '3'
}

# Dataset Configuration
TEST_DATASET_LIMIT = 20  # For development/testing

# Training Configuration
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_WARMUP_STEPS = 500
DEFAULT_LOGGING_STEPS = 10
DEFAULT_SAVE_STEPS = 500
DEFAULT_EVAL_STEPS = 500
DEFAULT_BATCH_SIZE = 2
DEFAULT_EVAL_BATCH_SIZE = 2
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_NUM_EPOCHS = 3

# Logging and Output
DEFAULT_LOG_DIR = 'logs'
DEFAULT_OUTPUT_DIR = './output'
EVAL_LOG_SEPARATOR = "=" * 50
SAMPLE_LOG_SEPARATOR = "-" * 40

# Evaluation Types
EVAL_TYPES = {
    'VANILLA': 'vanilla',
    'COT': 'cot', 
    'COCONUT': 'coconut'
}

# File Extensions and Patterns
CONFIG_FILE_EXT = '.yaml'
LOG_FILE_EXT = '.log' 