"""
Configuration classes for the MultiCoCo package.

This module provides configuration classes with validation and type safety
to replace direct dictionary access throughout the codebase.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import os
from pathlib import Path

from .constants import (
    DEFAULT_MODEL_NAME, DEFAULT_LEARNING_RATE, DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_BATCH_SIZE, DEFAULT_NUM_EPOCHS, DEFAULT_LOG_DIR,
    DEFAULT_OUTPUT_DIR, DEFAULT_MAX_NEW_TOKENS, DEFAULT_C_THOUGHT,
    DEFAULT_MAX_LATENT_STAGE, COCONUT_SPECIAL_TOKENS
)


@dataclass
class GenerationConfig:
    """Configuration for text generation parameters."""
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    do_sample: bool = False
    num_beams: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    pad_token_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for use with transformers."""
        config = {
            'max_new_tokens': self.max_new_tokens,
            'do_sample': self.do_sample,
            'num_beams': self.num_beams,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
        }
        if self.pad_token_id is not None:
            config['pad_token_id'] = self.pad_token_id
        return config


@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings."""
    vanilla: bool = True
    cot: bool = False
    coconut: bool = False
    detailed_logging: bool = True
    
    def get_eval_type(self) -> str:
        """Get the evaluation type as a string."""
        if self.coconut:
            return "coconut"
        elif self.cot:
            return "cot"
        else:
            return "vanilla"


@dataclass
class CoCoNutConfig:
    """Configuration for CoCoNut training parameters."""
    enabled: bool = False
    c_thought: int = DEFAULT_C_THOUGHT
    max_latent_stage: int = DEFAULT_MAX_LATENT_STAGE
    epochs_per_stage: int = 1
    special_tokens: List[str] = field(default_factory=lambda: COCONUT_SPECIAL_TOKENS.copy())
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.c_thought < 0:
            raise ValueError("c_thought must be non-negative")
        if self.max_latent_stage < 0:
            raise ValueError("max_latent_stage must be non-negative")
        if self.epochs_per_stage < 0:
            raise ValueError("epochs_per_stage must be non-negative")


@dataclass
class DataConfig:
    """Configuration for dataset and data loading."""
    data_dir: str = ""
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    limit_for_testing: bool = False
    
    def __post_init__(self):
        """Validate data configuration."""
        # Convert relative paths to absolute paths
        if self.data_dir:
            self.data_dir = os.path.abspath(self.data_dir)
        
        if self.train_data_path:
            self.train_data_path = os.path.abspath(self.train_data_path)
            if not os.path.exists(self.train_data_path):
                raise FileNotFoundError(f"Training data file not found: {self.train_data_path}")
        
        if self.eval_data_path:
            self.eval_data_path = os.path.abspath(self.eval_data_path)
            if not os.path.exists(self.eval_data_path):
                raise FileNotFoundError(f"Evaluation data file not found: {self.eval_data_path}")


@dataclass
class ModelConfig:
    """Configuration for model initialization."""
    model_name: str = DEFAULT_MODEL_NAME
    config_id: Optional[str] = None
    tokenizer_id: Optional[str] = None
    image_processor_id: Optional[str] = None
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    low_cpu_mem_usage: bool = True
    
    def get_special_tokens(self, coconut_config: CoCoNutConfig) -> List[str]:
        """Get special tokens based on configuration."""
        return coconut_config.special_tokens if coconut_config.enabled else []


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    eval_only: bool = False
    output_dir: str = DEFAULT_OUTPUT_DIR
    num_epochs: int = DEFAULT_NUM_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE
    gradient_accumulation_steps: int = 1
    learning_rate: float = DEFAULT_LEARNING_RATE
    warmup_steps: int = 500
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    bf16: bool = True
    fp16: bool = False
    remove_unused_columns: bool = False
    dataloader_pin_memory: bool = False
    dataloader_num_workers: int = 4
    weight_decay: float = 0.01
    seed: int = 42
    data_seed: int = 42
    
    def __post_init__(self):
        """Validate training configuration."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""
    log_dir: str = DEFAULT_LOG_DIR
    log_level: str = "INFO"
    use_wandb: bool = True
    console_output: bool = True
    verbose: bool = False
    
    def __post_init__(self):
        """Initialize logging configuration."""
        os.makedirs(self.log_dir, exist_ok=True)


@dataclass
class MultiCoCoConfig:
    """Main configuration class that combines all sub-configurations."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    coconut: CoCoNutConfig = field(default_factory=CoCoNutConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MultiCoCoConfig':
        """Create configuration from dictionary (typically loaded from YAML)."""
        # Extract sub-configurations
        model_config = ModelConfig(
            model_name=config_dict.get('model_name', DEFAULT_MODEL_NAME),
        )
        
        training_config = TrainingConfig(
            eval_only=config_dict.get('eval_only', False),
            output_dir=config_dict.get('output_dir', DEFAULT_OUTPUT_DIR),
            num_epochs=config_dict.get('num_epochs', DEFAULT_NUM_EPOCHS),
            batch_size=config_dict.get('batch_size', DEFAULT_BATCH_SIZE),
            eval_batch_size=config_dict.get('eval_batch_size', DEFAULT_EVAL_BATCH_SIZE),
            learning_rate=float(config_dict.get('learning_rate', DEFAULT_LEARNING_RATE)),
            gradient_accumulation_steps=config_dict.get('gradient_accumulation_steps', 1),
        )
        
        # Handle both 'eval_data_path' and 'val_data_path' for backward compatibility
        eval_data_path = config_dict.get('eval_data_path') or config_dict.get('val_data_path')
        
        data_config = DataConfig(
            data_dir=config_dict.get('data_dir', ''),
            train_data_path=config_dict.get('train_data_path'),
            eval_data_path=eval_data_path,
            limit_for_testing=config_dict.get('limit_for_testing', False)
        )
        
        evaluation_config = EvaluationConfig(
            vanilla=config_dict.get('vanilla', True),
            coconut=config_dict.get('coconut', False),
            cot=config_dict.get('cot', False),
            detailed_logging=config_dict.get('detailed_logging', True),
        )
        
        coconut_config = CoCoNutConfig(
            enabled=config_dict.get('coconut', False),
            c_thought=config_dict.get('c_thought', DEFAULT_C_THOUGHT),
            max_latent_stage=config_dict.get('max_latent_stage', DEFAULT_MAX_LATENT_STAGE),
            epochs_per_stage=config_dict.get('epochs_per_stage', 1),
        )
        
        logging_config = LoggingConfig(
            log_dir=config_dict.get('log_dir', DEFAULT_LOG_DIR),
            log_level=config_dict.get('log_level', 'INFO'),
            use_wandb=config_dict.get('use_wandb', True),
            console_output=config_dict.get('console_output', True),
            verbose=config_dict.get('verbose', False),
        )
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            evaluation=evaluation_config,
            coconut=coconut_config,
            logging=logging_config,
        )
    
    def validate(self) -> None:
        """Validate the entire configuration."""
        # Check that we have data for the intended operation
        if not self.training.eval_only and not self.data.train_data_path:
            raise ValueError("Training data path is required when not in eval_only mode")
        
        if self.training.eval_only and not self.data.eval_data_path:
            raise ValueError("Evaluation data path is required in eval_only mode")
        
        # Validate CoCoNut configuration
        if self.coconut.enabled and not any([self.evaluation.coconut, self.evaluation.cot]):
            raise ValueError("CoCoNut is enabled but neither coconut nor cot evaluation is configured")
    
    def get_wandb_report_to(self) -> List[str]:
        """Get the report_to list for training arguments."""
        return ["wandb"] if self.logging.use_wandb else []


def load_config_from_yaml(yaml_path: str) -> MultiCoCoConfig:
    """
    Load configuration from YAML file.
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        Complete MultiCoCo configuration
        
    Raises:
        ConfigurationError: If configuration loading fails
    """
    try:
        import yaml
        
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Handle nested configuration format
        model_config = ModelConfig(**yaml_config.get('model', {}))
        training_config = TrainingConfig(**yaml_config.get('training', {}))
        data_config = DataConfig(**yaml_config.get('data', {}))
        evaluation_config = EvaluationConfig(**yaml_config.get('evaluation', {}))
        generation_config = GenerationConfig(**yaml_config.get('generation', {}))
        logging_config = LoggingConfig(**yaml_config.get('logging', {}))
        
        # Handle CoCoNut config carefully - could be boolean or dict
        coconut_section = yaml_config.get('coconut', {})
        if isinstance(coconut_section, dict):
            coconut_config = CoCoNutConfig(**coconut_section)
        else:
            # Fallback for boolean value (old format)
            coconut_config = CoCoNutConfig(enabled=bool(coconut_section))
        
        return MultiCoCoConfig(
            model=model_config,
            training=training_config,
            data=data_config,
            evaluation=evaluation_config,
            coconut=coconut_config,
            generation=generation_config,
            logging=logging_config
        )
        
    except Exception as e:
        from .exceptions import ConfigurationError
        raise ConfigurationError(f"Failed to load configuration from {yaml_path}: {e}") 