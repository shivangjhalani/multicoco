#!/usr/bin/env python3
"""
Main entry point for MultiCoCo training and evaluation.

Provides a unified interface for running training and evaluation with the
MultiCoCo framework, supporting vanilla, CoT, and CoCoNut methodologies.
"""

import argparse
import logging
import os
import random
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
# --- Patch: ensure torch.utils.checkpoint is called with explicit use_reentrant ---
import torch.utils.checkpoint as _checkpoint_module  # type: ignore  # noqa: E402

if not getattr(_checkpoint_module.checkpoint, "_patched_use_reentrant", False):
    _orig_checkpoint_fn = _checkpoint_module.checkpoint

    def _checkpoint_with_explicit_use_reentrant(function, *args, **kwargs):  # type: ignore
        """Wrapper that sets use_reentrant=False if caller did not specify it."""
        if "use_reentrant" not in kwargs:
            kwargs["use_reentrant"] = False  # recommended by PyTorch >=2.1
        return _orig_checkpoint_fn(function, *args, **kwargs)

    _checkpoint_with_explicit_use_reentrant._patched_use_reentrant = True  # type: ignore
    _checkpoint_module.checkpoint = _checkpoint_with_explicit_use_reentrant
# -------------------------------------------------------------------------------
from transformers import AutoModelForCausalLM, TrainingArguments

# WandB import (optional to avoid hard dependency)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from multicoco.config import MultiCoCoConfig, TrainingMode
from multicoco.constants import (
    COCONUT_SPECIAL_TOKENS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOG_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_OUTPUT_DIR,
    IMAGE_TOKEN,
    TEST_DATASET_LIMIT,
)
from multicoco.data import SupervisedDataset, collate_fn
from multicoco.exceptions import (
    ConfigurationError,
    DataLoadingError,
    EvaluationError,
    ModelInitializationError,
)
from multicoco.latent_wrapper import LatentWrapper
from multicoco.model import MultiCoCo
from multicoco.trainer import CoCoTrainer
from multicoco.utils import TqdmLoggingHandler

logger = logging.getLogger(__name__)


class MultiCoCoRunner:
    """
    Main runner class for MultiCoCo training and evaluation.
    
    Orchestrates the complete training and evaluation pipeline, handling
    configuration, model initialization, data loading, and execution.
    """

    def __init__(self, config: MultiCoCoConfig):
        """Initialize the runner with configuration."""
        self.config = config
        self.model: Optional[MultiCoCo] = None
        self.trainer: Optional[CoCoTrainer] = None
        self.train_dataset: Optional[SupervisedDataset] = None
        self.eval_dataset: Optional[SupervisedDataset] = None
        
        self._initialize()
        
        mode_type = ('training' if config.training.mode != TrainingMode.EVAL_ONLY 
                    else 'evaluation')
        logger.info(f"MultiCoCoRunner initialized for {mode_type}")

    def _initialize(self) -> None:
        """Set up environment, logging, and random seeds."""
        # Set random seeds for reproducibility
        if self.config.training.seed is not None:
            self._set_random_seeds(self.config.training.seed)
        
        # Configure logging
        self._setup_logging()
        
        # Set up CUDA environment
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            device_count = torch.cuda.device_count()
            logger.info(f"CUDA available with {device_count} devices")
        else:
            logger.warning("CUDA not available, using CPU")

        # WandB will be initialized by HuggingFace Trainer via report_to parameter
        # We'll add custom metrics after trainer initialization

    def _set_random_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info(f"Set random seed to {seed}")

    def _setup_logging(self) -> None:
        """Configure logging based on configuration."""
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank not in [-1, 0]:
            logging.getLogger().setLevel(logging.CRITICAL)
            return
            
        log_config = self.config.logging
        os.makedirs(log_config.log_dir, exist_ok=True)
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_config.log_level))
        
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        # File handler
        file_path = os.path.join(log_config.log_dir, 'multicoco.log')
        file_handler = logging.FileHandler(file_path, mode='w')
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)

        # Console handler
        if log_config.console_output:
            console_handler = TqdmLoggingHandler()
            console_handler.setFormatter(logging.Formatter(log_format))
            root_logger.addHandler(console_handler)
        
        # Suppress verbose logging if needed
        if not log_config.verbose:
            logging.getLogger("transformers").setLevel(logging.WARNING)
            logging.getLogger("torch").setLevel(logging.WARNING)

    def _setup_wandb_config(self) -> None:
        """Setup WandB configuration and custom metrics after HF Trainer initialization."""
        if not WANDB_AVAILABLE or not self.config.logging.use_wandb:
            return
        
        # Only setup on main process
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank not in [-1, 0]:
            return
            
        if wandb is None or wandb.run is None:
            return
            
        try:
            # Update WandB config with our custom configuration
            wandb.config.update({
                "project_name": self.config.logging.wandb_project,
                "run_group": self.config.logging.wandb_group,
                "tags": self.config.logging.wandb_tags,
                **self.config.to_dict()
            }, allow_val_change=True)
            
            # Define custom metrics for better tracking
            wandb.define_metric("train/loss", summary="min")
            wandb.define_metric("eval/accuracy", summary="max") 
            wandb.define_metric("coconut/stage", summary="max")
            
            logger.info("WandB configuration updated with custom metrics")
            
        except Exception as e:
            logger.warning(f"Failed to setup WandB configuration: {e}")

    def initialize_model(self) -> None:
        """Initialize the model from configuration with proper phase separation."""
        try:
            model_config = self.config.model
            coconut_config = self.config.coconut
            training_mode = self.config.training.mode
            
            # Determine special tokens based on training phase
            special_tokens = self._get_special_tokens(
                coconut_config, training_mode
            )
            
            # Initialize base model
            base_model_source, checkpoint_path = self._get_model_source()
            
            self.model = MultiCoCo(
                model_id=base_model_source,
                config_id=model_config.config_id,
                tokenizer_id=model_config.tokenizer_id,
                image_processor_id=model_config.image_processor_id,
                special_tokens=special_tokens,
                torch_dtype=model_config.torch_dtype,
                trust_remote_code=model_config.trust_remote_code,
                low_cpu_mem_usage=model_config.low_cpu_mem_usage
            )
            
            # Load checkpoint weights if provided
            if checkpoint_path:
                self._load_checkpoint_weights(checkpoint_path)
            
            # Initialize latent token embeddings if needed
            if self._has_latent_tokens(special_tokens):
                self._initialize_latent_token_embeddings()
            
            # Wrap with LatentWrapper for CoCoNut training/evaluation
            if self._needs_latent_wrapper(coconut_config, training_mode):
                self.model = LatentWrapper(self.model, self.model.tokenizer)
            
            self._log_model_info(checkpoint_path, training_mode, coconut_config)
            
        except Exception as e:
            raise ModelInitializationError(f"Model initialization failed: {e}")

    def _get_special_tokens(self, coconut_config, training_mode):
        """Get special tokens based on configuration and training phase."""
        special_tokens = []
        if (coconut_config.enabled or 
            training_mode == TrainingMode.COCONUT_TRAIN):
            base_tokens = set(self.config.model.get_special_tokens(coconut_config))
            coconut_tokens = set(COCONUT_SPECIAL_TOKENS)
            special_tokens = list(base_tokens | coconut_tokens)
            logger.info(f"Adding latent special tokens: {special_tokens}")
        else:
            special_tokens = self.config.model.get_special_tokens(coconut_config)
            logger.info("CoT training phase - no latent tokens added")
        return special_tokens

    def _get_model_source(self):
        """Get model source and checkpoint path."""
        model_config = self.config.model
        if model_config.load_model_path:
            logger.info(f"Loading from checkpoint: {model_config.load_model_path}")
            return model_config.model_name, model_config.load_model_path
        else:
            logger.info(f"Loading base model: {model_config.model_name}")
            return model_config.model_name, None

    def _has_latent_tokens(self, special_tokens):
        """Check if latent tokens were added."""
        latent_tokens = ['<|latent|>', '<|start_latent|>', '<|end_latent|>']
        return any(tok in special_tokens for tok in latent_tokens)

    def _needs_latent_wrapper(self, coconut_config, training_mode):
        """Check if LatentWrapper is needed."""
        return (coconut_config.enabled or 
                training_mode == TrainingMode.COCONUT_TRAIN)

    def _log_model_info(self, checkpoint_path, training_mode, coconut_config):
        """Log model initialization information."""
        source_info = (f"checkpoint: {checkpoint_path}" if checkpoint_path 
                      else f"base model: {self.config.model.model_name}")
        logger.info(f"Model initialized from {source_info}")
        logger.info(f"Dtype: {self.config.model.torch_dtype}, "
                   f"BF16: {self.config.training.bf16}, "
                   f"FP16: {self.config.training.fp16}")
        logger.info(f"Mode: {training_mode}, CoCoNut: {coconut_config.enabled}")

    def _load_checkpoint_weights(self, checkpoint_path: str) -> None:
        """Load checkpoint weights into the base model."""
        if self.model is None:
            raise ModelInitializationError("Model must be initialized first")
            
        if not os.path.exists(checkpoint_path):
            raise ModelInitializationError(
                f"Checkpoint path does not exist: {checkpoint_path}"
            )
        
        try:
            checkpoint_model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=self.model.model.dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            missing_keys, unexpected_keys = self.model.model.load_state_dict(
                checkpoint_model.state_dict(), strict=False
            )
            
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
                
            del checkpoint_model
            
        except Exception as e:
            raise ModelInitializationError(
                f"Failed to load checkpoint weights: {e}"
            )

    def _initialize_latent_token_embeddings(self) -> None:
        """Initialize latent token embeddings with multimodal-aware approach."""
        if self.model is None:
            raise ModelInitializationError("Model must be initialized first")
            
        try:
            embed_layer = self.model.get_input_embeddings()
            with torch.no_grad():
                eos_token_id = self.model.tokenizer.eos_token_id
                eos_embedding = embed_layer.weight[eos_token_id].clone()
                
                image_token_id = self.model.tokenizer.convert_tokens_to_ids(
                    IMAGE_TOKEN
                )
                
                if (image_token_id is None or 
                    image_token_id >= embed_layer.weight.size(0)):
                    logger.warning(
                        f"'{IMAGE_TOKEN}' not found. Using EOS-only init."
                    )
                    multimodal_embedding = eos_embedding
                else:
                    image_embedding = embed_layer.weight[image_token_id].clone()
                    multimodal_embedding = (eos_embedding + image_embedding) / 2.0
                    logger.info("Created multimodal-aware embedding from "
                              f"EOS and '{IMAGE_TOKEN}'.")

                # Apply to all latent tokens
                for token in COCONUT_SPECIAL_TOKENS:
                    token_id = self.model.tokenizer.convert_tokens_to_ids(token)
                    if (token_id is not None and 
                        token_id < embed_layer.weight.size(0)):
                        embed_layer.weight[token_id] = multimodal_embedding
                        logger.info(f"Initialized '{token}' with multimodal embedding.")
                    else:
                        logger.warning(f"Could not initialize token: {token}")

        except Exception as e:
            raise ModelInitializationError(
                f"Failed to initialize latent token embeddings: {e}"
            )

    def setup_datasets(self) -> None:
        """Initialize and prepare datasets for training and evaluation."""
        try:
            data_config = self.config.data
            test_limit = (TEST_DATASET_LIMIT if data_config.limit_for_testing 
                         else None)
            
            # Load training dataset if not eval-only
            if (self.config.training.mode != TrainingMode.EVAL_ONLY and 
                data_config.train_data_path):
                self.train_dataset = SupervisedDataset(
                    data_path=data_config.train_data_path,
                    data_dir=data_config.data_dir,
                    test_limit=test_limit
                )
                logger.info(f"Training dataset: {len(self.train_dataset)} samples")
            
            # Load evaluation dataset
            if data_config.eval_data_path:
                self.eval_dataset = SupervisedDataset(
                    data_path=data_config.eval_data_path,
                    data_dir=data_config.data_dir,
                    test_limit=test_limit
                )
                logger.info(f"Evaluation dataset: {len(self.eval_dataset)} samples")
            else:
                raise DataLoadingError("Evaluation data path is required")
                
        except Exception as e:
            raise DataLoadingError(f"Failed to setup datasets: {e}")

    def create_trainer(self) -> None:
        """Create and configure the trainer."""
        if self.model is None:
            raise ModelInitializationError(
                "Model must be initialized before creating trainer"
            )
        
        try:
            training_args = self._create_training_arguments()
            data_collator = lambda batch: collate_fn(
                batch, self.model.tokenizer, self.model.image_processor
            )
            
            self.trainer = CoCoTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                processing_class=self.model.tokenizer,
                data_collator=data_collator
            )
            
            # Set configurations
            self.trainer.args.eval_config = self._create_eval_config()
            self.trainer.args.generation_kwargs = self._create_generation_kwargs()
            
            # Set CoCoNut parameters
            if self.config.coconut.enabled:
                self._set_coconut_trainer_params()
            
            # Debug WandB state after trainer initialization
            if WANDB_AVAILABLE and wandb is not None:
                logger.info(f"WandB run after trainer init: {wandb.run}")
                if wandb.run is not None:
                    logger.info(f"WandB run name: {wandb.run.name}")
                    logger.info(f"WandB project: {wandb.run.project}")
                else:
                    logger.warning("WandB run is None after trainer initialization")
            
            # Setup WandB configuration and custom metrics
            self._setup_wandb_config()
            
            logger.info("Trainer created successfully")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create trainer: {e}")

    def _set_coconut_trainer_params(self):
        """Set CoCoNut parameters on the trainer."""
        coconut_config = self.config.coconut
        self.trainer.args.c_thought = coconut_config.c_thought
        self.trainer.args.max_latent_stage = coconut_config.max_latent_stage
        self.trainer.args.epochs_per_stage = coconut_config.epochs_per_stage
        self.trainer.args.uniform_prob = coconut_config.uniform_prob
        self.trainer.args.pad_latent_to_max = coconut_config.pad_latent_to_max
        self.trainer.args.reset_optimizer = coconut_config.reset_optimizer

    def _create_training_arguments(self) -> TrainingArguments:
        """Create HuggingFace TrainingArguments from configuration."""
        training_config = self.config.training
        is_training = training_config.mode != TrainingMode.EVAL_ONLY

        return (self._create_training_args(training_config) if is_training 
                else self._create_evaluation_args(training_config))

    def _create_training_args(self, training_config) -> TrainingArguments:
        """Create training arguments for training modes."""
        # Set WandB environment variables for HuggingFace integration
        if self.config.logging.use_wandb and WANDB_AVAILABLE:
            # Ensure WandB is logged in before starting training
            try:
                wandb.login()
                logger.info("WandB login successful")
            except Exception as e:
                logger.warning(f"WandB login failed: {e}. You may need to run 'wandb login' manually or set WANDB_API_KEY environment variable.")
            
            os.environ["WANDB_PROJECT"] = self.config.logging.wandb_project
            if self.config.logging.wandb_entity:
                os.environ["WANDB_ENTITY"] = self.config.logging.wandb_entity
            if self.config.training.name:
                os.environ["WANDB_NAME"] = self.config.training.name
            if self.config.logging.wandb_tags:
                os.environ["WANDB_TAGS"] = ",".join(self.config.logging.wandb_tags)
        
        report_to = self.config.get_wandb_report_to()
        logger.info(f"TrainingArguments report_to: {report_to}")
        logger.info(f"WandB use_wandb: {self.config.logging.use_wandb}")
        logger.info(f"WANDB_AVAILABLE: {WANDB_AVAILABLE}")
        
        return TrainingArguments(
            output_dir=training_config.output_dir,
            num_train_epochs=training_config.num_epochs,
            per_device_train_batch_size=training_config.batch_size,
            per_device_eval_batch_size=training_config.eval_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            eval_accumulation_steps=training_config.eval_accumulation_steps,
            gradient_checkpointing=training_config.gradient_checkpointing,
            gradient_checkpointing_kwargs=training_config.gradient_checkpointing_kwargs,
            learning_rate=training_config.learning_rate,
            warmup_steps=training_config.warmup_steps,
            logging_steps=training_config.logging_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=training_config.save_total_limit,
            load_best_model_at_end=training_config.load_best_model_at_end,
            metric_for_best_model=training_config.metric_for_best_model,
            greater_is_better=training_config.greater_is_better,
            weight_decay=training_config.weight_decay,
            bf16=training_config.bf16,
            fp16=training_config.fp16,
            remove_unused_columns=training_config.remove_unused_columns,
            dataloader_pin_memory=training_config.dataloader_pin_memory,
            dataloader_num_workers=training_config.dataloader_num_workers,
            do_train=True,
            do_eval=True,
            report_to=report_to,
            run_name=getattr(training_config, 'name', None)
        )

    def _create_evaluation_args(self, training_config) -> TrainingArguments:
        """Create training arguments for evaluation-only mode."""
        return TrainingArguments(
            output_dir=training_config.output_dir,
            per_device_eval_batch_size=training_config.eval_batch_size,
            eval_accumulation_steps=training_config.eval_accumulation_steps,
            bf16=training_config.bf16,
            fp16=training_config.fp16,
            remove_unused_columns=training_config.remove_unused_columns,
            dataloader_pin_memory=training_config.dataloader_pin_memory,
            dataloader_num_workers=training_config.dataloader_num_workers,
            do_train=False,
            do_eval=True,
            report_to=[],
        )

    def _create_generation_kwargs(self) -> Dict[str, Any]:
        """Create generation keyword arguments."""
        gen_config = self.config.generation
        
        kwargs = {
            "max_new_tokens": gen_config.max_new_tokens,
            "do_sample": gen_config.do_sample,
            "num_beams": gen_config.num_beams,
            "temperature": gen_config.temperature,
            "top_p": gen_config.top_p,
            "top_k": gen_config.top_k,
        }
        
        if (hasattr(self.model, 'tokenizer') and 
            self.model.tokenizer.pad_token_id is not None):
            kwargs["pad_token_id"] = self.model.tokenizer.pad_token_id
        
        return kwargs

    def _create_eval_config(self) -> Dict[str, Any]:
        """Create evaluation configuration dictionary."""
        eval_config = self.config.evaluation
        
        return {
            "vanilla": eval_config.vanilla,
            "cot": eval_config.cot,
            "coconut": eval_config.coconut,
            "eval_latent_tokens": eval_config.eval_latent_tokens,
            "detailed_logging": eval_config.detailed_logging
        }

    def run_training(self) -> None:
        """Run the training loop."""
        if self.trainer is None:
            raise ConfigurationError("Trainer not initialized")
        
        logger.info("Starting training...")
        self.trainer.train(
            resume_from_checkpoint=self.config.training.resume_from_checkpoint
        )

    def run_coconut_training(self) -> None:
        """Run CoCoNut progressive curriculum learning training."""
        if self.trainer is None:
            raise ConfigurationError("Trainer not initialized")
        
        logger.info("Starting CoCoNut progressive curriculum learning...")
        self.trainer.train_coconut_progressive(
            resume_from_checkpoint=self.config.training.resume_from_checkpoint
        )

    def run_evaluation(self) -> Dict[str, float]:
        """Run standalone evaluation."""
        if self.trainer is None:
            raise ConfigurationError("Trainer must be created before evaluation")
        
        if self.eval_dataset is None:
            raise DataLoadingError("Evaluation dataset must be loaded")
        
        try:
            logger.info("Starting evaluation...")
            eval_results = self.trainer.evaluate()
            
            # Extract metrics
            metrics = (eval_results.metrics if hasattr(eval_results, 'metrics') 
                      else eval_results)
            
            self._log_evaluation_results(metrics)
            
            if self.trainer.is_world_process_zero():
                logger.info("Evaluation completed successfully")
            return metrics
            
        except Exception as e:
            raise EvaluationError(f"Evaluation failed: {e}")

    def _log_evaluation_results(self, metrics: Dict[str, float]) -> None:
        """Log evaluation results in structured format."""
        if self.trainer and self.trainer.is_world_process_zero():
            logger.info("\n" + "="*50)
            logger.info("FINAL RESULTS")
            logger.info("="*50)
            
            accuracy = metrics.get('eval_accuracy', 0.0)
            loss = metrics.get('eval_loss', 0.0)
            
            logger.info(f"Evaluation Results:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Loss: {loss:.4f}")
            
            # Log CoCoNut specific metrics if available
            if 'eval_coconut_stage' in metrics:
                stage = metrics['eval_coconut_stage']
                max_stage = metrics['eval_max_latent_stage']
                logger.info(f"  CoCoNut Stage: {stage}/{max_stage}")
                
            logger.info("="*50)

    def run(self) -> Dict[str, float]:
        """Orchestrate the full pipeline based on training mode."""
        try:
            # Initialize model and datasets
            self.initialize_model()
            self.setup_datasets()
            
            # Execute based on mode using dispatch dict
            mode_handlers = {
                TrainingMode.EVAL_ONLY: self._run_eval_only,
                TrainingMode.COT_TRAIN: self._run_cot_training,
                TrainingMode.COCONUT_TRAIN: self._run_coconut_training,
            }
            
            mode = self.config.training.mode
            if mode not in mode_handlers:
                raise ConfigurationError(f"Invalid training mode: {mode}")
            
            return mode_handlers[mode]()

        except (ConfigurationError, ModelInitializationError, 
                DataLoadingError, EvaluationError) as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def _run_eval_only(self) -> Dict[str, float]:
        """Handle evaluation-only mode."""
        logger.info("Starting evaluation only...")
        self.create_trainer()
        results = self.run_evaluation()
        self._log_evaluation_results(results)
        return results

    def _run_cot_training(self) -> Dict[str, float]:
        """Handle CoT training mode."""
        logger.info("Starting CoT training...")
        self.create_trainer()
        self.run_training()
        return self._run_final_evaluation()

    def _run_coconut_training(self) -> Dict[str, float]:
        """Handle CoCoNut training mode."""
        logger.info("Starting CoCoNut training...")
        self.create_trainer()
        self.run_coconut_training()
        return self._run_final_evaluation()

    def _run_final_evaluation(self) -> Dict[str, float]:
        """Run final evaluation and return results."""
        logger.info("Running final evaluation...")
        results = self.run_evaluation()
        self._log_evaluation_results(results)
        return results


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="MultiCoCo: Two-phase training for multimodal models.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only (skip training)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        help="Override model name"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser


def apply_cli_overrides(config: MultiCoCoConfig, 
                       args: argparse.Namespace) -> MultiCoCoConfig:
    """Apply command line overrides to configuration."""
    if args.eval_only:
        config.training.eval_only = True
    
    if args.output_dir:
        config.training.output_dir = args.output_dir
    
    if args.model_name:
        config.model.model_name = args.model_name
    
    if args.verbose:
        config.logging.verbose = True
        config.logging.console_output = True
    
    return config


def _load_config(config_path: str) -> MultiCoCoConfig:
    """Load configuration with base inheritance."""
    base_cfg_path = os.path.join(os.path.dirname(config_path), "base.yaml")
    return MultiCoCoConfig.load_with_base(
        config_path=config_path,
        base_config_path=base_cfg_path
    )


def main() -> None:
    """Main entry point."""
    try:
        parser = create_parser()
        args = parser.parse_args()
        
        config = _load_config(args.config_path)
        config = apply_cli_overrides(config, args)
        
        # Auto-configure evaluation mode for CoT training
        if config.training.mode == TrainingMode.COT_TRAIN:
            logger.info("CoT training mode: setting CoT evaluation")
            config.evaluation.cot = True
            config.evaluation.vanilla = False
        
        runner = MultiCoCoRunner(config)
        metrics = runner.run()
        
        # Print final results
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        for key, value in metrics.items():
            print(f"{key}: {value}")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
