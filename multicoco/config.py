"""
Configuration classes for the MultiCoCo package.

Provides configuration classes with validation and type safety to replace
direct dictionary access throughout the codebase.
"""

import os
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .constants import (
    COCONUT_SPECIAL_TOKENS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_C_THOUGHT,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOG_DIR,
    DEFAULT_MAX_LATENT_STAGE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_OUTPUT_DIR,
)


class TrainingMode(str, Enum):
    """Training mode enumeration."""
    EVAL_ONLY = "eval_only"
    COT_TRAIN = "cot_train"
    COCONUT_TRAIN = "coconut_train"


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
    """Configuration for evaluation modes."""
    vanilla: bool = True
    cot: bool = False
    coconut: bool = False
    eval_latent_tokens: Optional[int] = None
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
    special_tokens: List[str] = field(
        default_factory=lambda: COCONUT_SPECIAL_TOKENS.copy()
    )
    uniform_prob: float = 0.0
    pad_latent_to_max: bool = False
    reset_optimizer: bool = True


@dataclass
class DataConfig:
    """Configuration for dataset and data loading."""
    data_dir: str = ""
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    limit_for_testing: bool = False
    
    def __post_init__(self):
        """Convert relative paths to absolute paths."""
        if self.data_dir:
            self.data_dir = os.path.abspath(self.data_dir)
        
        if self.train_data_path:
            self.train_data_path = os.path.abspath(self.train_data_path)
        
        if self.eval_data_path:
            self.eval_data_path = os.path.abspath(self.eval_data_path)


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
    load_model_path: Optional[str] = None
    
    def get_special_tokens(self, coconut_config: CoCoNutConfig) -> List[str]:
        """Get special tokens based on configuration."""
        # Return empty list - latent tokens are handled during model init
        return []


@dataclass
class TrainingConfig:
    """Configuration for training-related settings."""
    output_dir: str = DEFAULT_OUTPUT_DIR
    num_epochs: int = DEFAULT_NUM_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE
    gradient_accumulation_steps: int = 1
    eval_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {'use_reentrant': False}
    )
    learning_rate: float = DEFAULT_LEARNING_RATE
    warmup_steps: int = 500
    logging_steps: int = 10
    save_steps: int = 1000
    eval_steps: int = 1000
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    greater_is_better: bool = False
    bf16: bool = True
    fp16: bool = False
    remove_unused_columns: bool = False
    resume_from_checkpoint: bool = False
    dataloader_pin_memory: bool = False
    dataloader_num_workers: int = 4
    weight_decay: float = 0.01
    seed: Optional[int] = None
    data_seed: Optional[int] = None
    mode: TrainingMode = TrainingMode.COT_TRAIN
    name: Optional[str] = None
    max_checkpoints_to_keep: int = 3
    keep_best_checkpoints: bool = True
    use_run_name_in_output_dir: bool = True
    
    def __post_init__(self):
        """Initialize and validate training configuration."""
        # Set random seeds
        if self.seed is None:
            self.seed = random.randint(0, 2**32 - 1)
        if self.data_seed is None:
            self.data_seed = self.seed
        
        # Modify output directory to include run name if requested
        if self.use_run_name_in_output_dir and self.name:
            base_dir = (os.path.dirname(self.output_dir) or "checkpoints")
            dir_name = os.path.basename(self.output_dir)
            self.output_dir = os.path.join(base_dir, f"{dir_name}_{self.name}")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""
    log_dir: str = DEFAULT_LOG_DIR
    log_level: str = "INFO"
    use_wandb: bool = True
    console_output: bool = True
    verbose: bool = False
    run_name: Optional[str] = None
    wandb_project: str = "multicoco-research"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_group: Optional[str] = None

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
    
    def to_dict(self) -> Dict[str, Any]:
      """Serializes the config object to a dictionary."""
      # Use dataclasses.asdict for deep conversion
      from dataclasses import asdict
      return asdict(self)

    def __post_init__(self):
        """Validate the entire configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate the entire configuration."""
        # Validate training configuration
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.training.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.training.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.training.bf16 and self.training.fp16:
            raise ValueError("Cannot enable both bf16 and fp16 simultaneously")
        
        # Validate CoCoNut configuration
        if self.coconut.c_thought < 0:
            raise ValueError("c_thought must be non-negative")
        if self.coconut.max_latent_stage < 0:
            raise ValueError("max_latent_stage must be non-negative")
        if self.coconut.epochs_per_stage < 0:
            raise ValueError("epochs_per_stage must be non-negative")
        if not 0.0 <= self.coconut.uniform_prob <= 1.0:
            raise ValueError("uniform_prob must be between 0.0 and 1.0")
        
        # Validate data requirements
        is_training = self.training.mode != TrainingMode.EVAL_ONLY
        if is_training and not self.data.train_data_path:
            raise ValueError("Training data path required for training modes")
        if self.training.mode == TrainingMode.EVAL_ONLY and not self.data.eval_data_path:
            raise ValueError("Evaluation data path required for eval_only mode")
        
        # Validate CoCoNut evaluation configuration
        if self.coconut.enabled and not any([
            self.evaluation.coconut, self.evaluation.cot
        ]):
            raise ValueError(
                "CoCoNut enabled but no compatible evaluation configured"
            )
        
        # Validate file existence (only if paths exist)
        if self.data.train_data_path and not os.path.exists(self.data.train_data_path):
            raise FileNotFoundError(f"Training data not found: {self.data.train_data_path}")
        if self.data.eval_data_path and not os.path.exists(self.data.eval_data_path):
            raise FileNotFoundError(f"Evaluation data not found: {self.data.eval_data_path}")

    @classmethod
    def load_with_base(cls, config_path: str, 
                      base_config_path: str = "args/base.yaml") -> 'MultiCoCoConfig':
        """Load configuration with inheritance from a base config file."""
        import yaml
        
        # Load base configuration
        base_dict = {}
        if os.path.exists(base_config_path):
            with open(base_config_path, 'r') as f:
                base_dict = yaml.safe_load(f) or {}
        
        # Load specific configuration
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
        
        # Merge configurations (config_dict overrides base_dict)
        merged_dict = {**base_dict, **config_dict}
        
        # Handle nested dictionaries
        for key in ['eval_config', 'coconut', 'generation', 'logging']:
            if (key in base_dict and key in config_dict and 
                isinstance(base_dict[key], dict) and 
                isinstance(config_dict[key], dict)):
                merged_dict[key] = {**base_dict[key], **config_dict[key]}
        
        return cls.from_dict(merged_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MultiCoCoConfig':
        """Create configuration from dictionary."""
        # Determine torch_dtype from precision settings
        torch_dtype = cls._get_torch_dtype(config_dict)
        
        # Create sub-configurations
        model_config = cls._create_model_config(config_dict, torch_dtype)
        training_config = cls._create_training_config(config_dict)
        data_config = cls._create_data_config(config_dict)
        evaluation_config = cls._create_evaluation_config(config_dict)
        coconut_config = cls._create_coconut_config(config_dict)
        generation_config = cls._create_generation_config(config_dict)
        logging_config = cls._create_logging_config(config_dict, training_config)
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            evaluation=evaluation_config,
            coconut=coconut_config,
            generation=generation_config,
            logging=logging_config,
        )
    
    @staticmethod
    def _get_torch_dtype(config_dict: Dict[str, Any]) -> str:
        """Determine torch_dtype from precision settings."""
        bf16 = config_dict.get('bf16', True)
        fp16 = config_dict.get('fp16', False)
        
        if bf16:
            return "bfloat16"
        elif fp16:
            return "float16"
        else:
            return "float32"
    
    @staticmethod
    def _create_model_config(config_dict: Dict[str, Any], torch_dtype: str) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(
            model_name=config_dict.get('model_name', DEFAULT_MODEL_NAME),
            torch_dtype=torch_dtype,
            config_id=config_dict.get('config_id'),
            tokenizer_id=config_dict.get('tokenizer_id'),
            image_processor_id=config_dict.get('image_processor_id'),
            trust_remote_code=config_dict.get('trust_remote_code', True),
            low_cpu_mem_usage=config_dict.get('low_cpu_mem_usage', True),
            load_model_path=config_dict.get('load_model_path')
        )
    
    @staticmethod
    def _create_training_config(config_dict: Dict[str, Any]) -> TrainingConfig:
        """Create training configuration."""
        return TrainingConfig(
            output_dir=config_dict.get('output_dir', DEFAULT_OUTPUT_DIR),
            num_epochs=config_dict.get('num_epochs', DEFAULT_NUM_EPOCHS),
            batch_size=config_dict.get('batch_size', DEFAULT_BATCH_SIZE),
            eval_batch_size=config_dict.get('eval_batch_size', DEFAULT_EVAL_BATCH_SIZE),
            learning_rate=float(config_dict.get('learning_rate', DEFAULT_LEARNING_RATE)),
            gradient_accumulation_steps=config_dict.get('gradient_accumulation_steps', 1),
            eval_accumulation_steps=config_dict.get('eval_accumulation_steps', 1),
            resume_from_checkpoint=config_dict.get('resume_from_checkpoint', False),
            mode=TrainingMode(config_dict.get('mode', 'cot_train')),
            bf16=config_dict.get('bf16', True),
            fp16=config_dict.get('fp16', False),
            gradient_checkpointing=config_dict.get('gradient_checkpointing', True),
            gradient_checkpointing_kwargs=config_dict.get('gradient_checkpointing_kwargs', {'use_reentrant': False}),
            warmup_steps=config_dict.get('warmup_steps', 500),
            logging_steps=config_dict.get('logging_steps', 10),
            save_steps=config_dict.get('save_steps', 1000),
            eval_steps=config_dict.get('eval_steps', 1000),
            save_total_limit=config_dict.get('save_total_limit', 2),
            max_checkpoints_to_keep=config_dict.get('max_checkpoints_to_keep', 3),
            keep_best_checkpoints=config_dict.get('keep_best_checkpoints', True),
            use_run_name_in_output_dir=config_dict.get('use_run_name_in_output_dir', True),
            load_best_model_at_end=config_dict.get('load_best_model_at_end', True),
            metric_for_best_model=config_dict.get('metric_for_best_model', 'accuracy'),
            greater_is_better=config_dict.get('greater_is_better', False),
            weight_decay=config_dict.get('weight_decay', 0.01),
            seed=config_dict.get('seed'),
            data_seed=config_dict.get('data_seed'),
            name=config_dict.get('name') or config_dict.get('run_name')
        )
    
    @staticmethod
    def _create_data_config(config_dict: Dict[str, Any]) -> DataConfig:
        """Create data configuration."""
        eval_data_path = (config_dict.get('eval_data_path') or 
                         config_dict.get('val_data_path'))
        
        return DataConfig(
            data_dir=config_dict.get('data_dir', ''),
            train_data_path=config_dict.get('train_data_path'),
            eval_data_path=eval_data_path,
            limit_for_testing=config_dict.get('limit_for_testing', False)
        )
    
    @staticmethod
    def _create_evaluation_config(config_dict: Dict[str, Any]) -> EvaluationConfig:
        """Create evaluation configuration."""
        return EvaluationConfig(
            vanilla=config_dict.get('vanilla', True),
            coconut=config_dict.get('coconut', False),
            cot=config_dict.get('cot', False),
            eval_latent_tokens=config_dict.get('eval_latent_tokens'),
            detailed_logging=config_dict.get('detailed_logging', True),
        )
    
    @staticmethod
    def _create_coconut_config(config_dict: Dict[str, Any]) -> CoCoNutConfig:
        """Create CoCoNut configuration."""
        coconut_dict = config_dict.get('coconut', {})
        
        if isinstance(coconut_dict, bool):
            # Handle boolean coconut flag for backward compatibility
            coconut_enabled = coconut_dict
            coconut_dict = {}
        else:
            coconut_enabled = coconut_dict.get('enabled', 
                                             config_dict.get('coconut', False))
        
        return CoCoNutConfig(
            enabled=coconut_enabled,
            c_thought=coconut_dict.get('c_thought', 
                                     config_dict.get('c_thought', DEFAULT_C_THOUGHT)),
            max_latent_stage=coconut_dict.get('max_latent_stage', 
                                            config_dict.get('max_latent_stage', DEFAULT_MAX_LATENT_STAGE)),
            epochs_per_stage=coconut_dict.get('epochs_per_stage', 
                                            config_dict.get('epochs_per_stage', 1)),
            uniform_prob=coconut_dict.get('uniform_prob', 
                                        config_dict.get('uniform_prob', 0.0)),
            pad_latent_to_max=coconut_dict.get('pad_latent_to_max', 
                                             config_dict.get('pad_latent_to_max', False)),
            reset_optimizer=coconut_dict.get('reset_optimizer', 
                                           config_dict.get('reset_optimizer', True))
        )
    
    @staticmethod
    def _create_generation_config(config_dict: Dict[str, Any]) -> GenerationConfig:
        """Create generation configuration."""
        generation_dict = config_dict.get('generation', {})
        return GenerationConfig(
            max_new_tokens=generation_dict.get('max_new_tokens', DEFAULT_MAX_NEW_TOKENS),
            do_sample=generation_dict.get('do_sample', False),
            num_beams=generation_dict.get('num_beams', 1),
            temperature=generation_dict.get('temperature', 1.0),
            top_p=generation_dict.get('top_p', 1.0),
            top_k=generation_dict.get('top_k', 50),
        )
    
    @staticmethod
    def _create_logging_config(config_dict: Dict[str, Any], 
                              training_config: TrainingConfig) -> LoggingConfig:
        """Create logging configuration."""
        logging_dict = config_dict.get('logging', config_dict)
        return LoggingConfig(
            log_dir=logging_dict.get('log_dir', DEFAULT_LOG_DIR),
            log_level=logging_dict.get('log_level', 'INFO'),
            use_wandb=logging_dict.get('use_wandb', True),
            console_output=logging_dict.get('console_output', True),
            verbose=logging_dict.get('verbose', False),
            run_name=training_config.name,
            wandb_project=logging_dict.get('wandb_project', "multicoco-research"),
            wandb_entity=logging_dict.get('wandb_entity'),
            wandb_tags=logging_dict.get('wandb_tags', []),
            wandb_group=logging_dict.get('wandb_group')
        )
    
    def get_wandb_report_to(self) -> List[str]:
        """Get the report_to list for training arguments."""
        return ["wandb"] if self.logging.use_wandb else [] 