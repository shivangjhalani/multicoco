#!/usr/bin/env python3
"""
Main entry point for MultiCoCo training and evaluation.

This script provides a unified interface for running training and evaluation
with the MultiCoCo framework, supporting vanilla, CoT, and CoCoNut methodologies.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import random

# ** Core libraries
import torch
import numpy as np
from transformers import TrainingArguments

# ** Local imports
from multicoco.config import (
    MultiCoCoConfig,
    ModelConfig,
    TrainingConfig, 
    DataConfig,
    EvaluationConfig,
    CoCoNutConfig,
    GenerationConfig,
    LoggingConfig,
    load_config_from_yaml
)
from multicoco.model import MultiCoCo
from multicoco.trainer import CoCoTrainer
from multicoco.data import SupervisedDataset, collate_fn
from multicoco.constants import (
    DEFAULT_MODEL_NAME,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_LOG_DIR,
    TEST_DATASET_LIMIT,
    COCONUT_SPECIAL_TOKENS
)
from multicoco.exceptions import (
    ConfigurationError,
    ModelInitializationError,
    DataLoadingError,
    EvaluationError
)
from multicoco.utils import load_image

logger = logging.getLogger(__name__)


class MultiCoCoRunner:
    """
    Main runner class for MultiCoCo training and evaluation.
    
    This class orchestrates the entire training and evaluation pipeline,
    handling configuration, model initialization, data loading, and execution.
    """

    def __init__(self, config: MultiCoCoConfig):
        """
        Initialize the runner with configuration.
        
        Args:
            config: Complete configuration for the run
        """
        self.config = config
        self.model: Optional[MultiCoCo] = None
        self.trainer: Optional[CoCoTrainer] = None
        self.train_dataset: Optional[SupervisedDataset] = None
        self.eval_dataset: Optional[SupervisedDataset] = None
        
        # Setup environment
        self._setup_environment()
        self._setup_logging()
        
        logger.info(f"MultiCoCoRunner initialized for {'training' if not config.training.eval_only else 'evaluation'}")

    def _setup_environment(self) -> None:
        """Set up the execution environment."""
        # Set random seeds for reproducibility
        if self.config.training.seed is not None:
            self._set_random_seeds(self.config.training.seed)
        
        # Set up device and distributed training
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            logger.info(f"CUDA available with {torch.cuda.device_count()} devices")
        else:
            logger.warning("CUDA not available, using CPU")

    def _set_random_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info(f"Set random seed to {seed}")

    def _setup_logging(self) -> None:
        """Configure logging based on configuration."""
        log_config = self.config.logging
        
        # Create log directory
        os.makedirs(log_config.log_dir, exist_ok=True)
        
        # Configure logging format and level
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        if log_config.console_output:
            logging.basicConfig(
                level=getattr(logging, log_config.log_level),
                format=log_format,
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler(os.path.join(log_config.log_dir, 'multicoco.log'))
                ]
            )
        else:
            # File logging only
            logging.basicConfig(
                level=getattr(logging, log_config.log_level),
                format=log_format,
                handlers=[
                    logging.FileHandler(os.path.join(log_config.log_dir, 'multicoco.log'))
                ]
            )
        
        # Suppress transformers warnings if needed
        if not log_config.verbose:
            logging.getLogger("transformers").setLevel(logging.WARNING)
            logging.getLogger("torch").setLevel(logging.WARNING)

    def initialize_model(self) -> None:
        """Initialize the model with configuration."""
        try:
            model_config = self.config.model
            special_tokens = COCONUT_SPECIAL_TOKENS if self.config.coconut.enabled else []
            
            self.model = MultiCoCo(
                model_id=model_config.model_name,
                special_tokens=special_tokens,
                torch_dtype=model_config.torch_dtype,
                trust_remote_code=model_config.trust_remote_code,
                low_cpu_mem_usage=model_config.low_cpu_mem_usage
            )
            
            logger.info(f"Model '{model_config.model_name}' initialized successfully")
            
        except Exception as e:
            raise ModelInitializationError(f"Failed to initialize model: {e}")

    def setup_datasets(self) -> None:
        """Set up training and evaluation datasets."""
        try:
            data_config = self.config.data
            
            # Determine test limit for development
            test_limit = TEST_DATASET_LIMIT if data_config.limit_for_testing else None
            
            # Load training dataset if not eval-only
            if not self.config.training.eval_only and data_config.train_data_path:
                self.train_dataset = SupervisedDataset(
                    data_path=data_config.train_data_path,
                    data_dir=data_config.data_dir,
                    test_limit=test_limit
                )
                logger.info(f"Training dataset loaded with {len(self.train_dataset)} samples")
            
            # Load evaluation dataset
            if data_config.eval_data_path:
                self.eval_dataset = SupervisedDataset(
                    data_path=data_config.eval_data_path,
                    data_dir=data_config.data_dir,
                    test_limit=test_limit
                )
                logger.info(f"Evaluation dataset loaded with {len(self.eval_dataset)} samples")
            else:
                raise DataLoadingError("Evaluation data path is required")
                
        except Exception as e:
            raise DataLoadingError(f"Failed to setup datasets: {e}")

    def create_trainer(self) -> None:
        """Create and configure the trainer."""
        if self.model is None:
            raise ModelInitializationError("Model must be initialized before creating trainer")
        
        try:
            # Create training arguments
            training_args = self._create_training_arguments()
            
            # Create data collator
            data_collator = lambda batch: collate_fn(
                batch, self.model.tokenizer, self.model.image_processor
            )
            
            # Initialize trainer
            self.trainer = CoCoTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                processing_class=self.model.tokenizer,  # For compatibility with new transformers
                data_collator=data_collator
            )
            
            # Set evaluation configuration
            self.trainer.args.eval_config = self._create_eval_config()
            
            # Set generation kwargs
            self.trainer.args.generation_kwargs = self._create_generation_kwargs()
            
            # Set CoCoNut parameters
            if self.config.coconut.enabled:
                self.trainer.args.c_thought = self.config.coconut.c_thought
                self.trainer.args.max_latent_stage = self.config.coconut.max_latent_stage
                self.trainer.args.epochs_per_stage = self.config.coconut.epochs_per_stage
            
            logger.info("Trainer created successfully")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create trainer: {e}")

    def _create_training_arguments(self) -> TrainingArguments:
        """Create training arguments from configuration."""
        training_config = self.config.training
        logging_config = self.config.logging
        
        # Configure wandb based on eval_only flag and explicit settings
        if training_config.eval_only:
            # Disable wandb for evaluation
            report_to = []
            run_name = f"eval_{self._get_eval_type_name()}"
        else:
            # Enable wandb for training if configured
            report_to = ["wandb"] if logging_config.use_wandb else []
            run_name = f"train_{self._get_eval_type_name()}"
        
        return TrainingArguments(
            output_dir=training_config.output_dir,
            num_train_epochs=training_config.num_epochs,
            per_device_train_batch_size=training_config.batch_size,
            per_device_eval_batch_size=training_config.eval_batch_size,
            learning_rate=float(training_config.learning_rate),  # Ensure float
            warmup_steps=training_config.warmup_steps,
            logging_steps=training_config.logging_steps,
            save_steps=training_config.save_steps,
            eval_steps=training_config.eval_steps,
            eval_strategy=training_config.evaluation_strategy,
            save_strategy=training_config.save_strategy,
            load_best_model_at_end=training_config.load_best_model_at_end,
            metric_for_best_model=training_config.metric_for_best_model,
            greater_is_better=training_config.greater_is_better,
            seed=training_config.seed,
            data_seed=training_config.data_seed,
            bf16=training_config.bf16,
            fp16=training_config.fp16,
            dataloader_num_workers=training_config.dataloader_num_workers,
            remove_unused_columns=training_config.remove_unused_columns,
            report_to=report_to,
            run_name=run_name,
            logging_dir=logging_config.log_dir,
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
        
        # Add pad token ID if available
        if hasattr(self.model, 'tokenizer') and self.model.tokenizer.pad_token_id is not None:
            kwargs["pad_token_id"] = self.model.tokenizer.pad_token_id
        
        return kwargs

    def _create_eval_config(self) -> Dict[str, Any]:
        """Create evaluation configuration dictionary."""
        eval_config = self.config.evaluation
        
        return {
            "vanilla": eval_config.vanilla,
            "cot": eval_config.cot,
            "coconut": eval_config.coconut,
            "detailed_logging": eval_config.detailed_logging
        }

    def _get_eval_type_name(self) -> str:
        """Get the evaluation type name for logging."""
        eval_config = self.config.evaluation
        
        if eval_config.coconut:
            return "coconut"
        elif eval_config.cot:
            return "cot"
        else:
            return "vanilla"

    def run_training(self) -> None:
        """Run the training process."""
        if self.trainer is None:
            raise ConfigurationError("Trainer must be created before running training")
        
        if self.config.training.eval_only:
            logger.info("Skipping training (eval_only=True)")
            return
        
        try:
            logger.info("Starting training...")
            
            # Run training
            self.trainer.train()
            
            # Save final model
            final_model_path = os.path.join(self.config.training.output_dir, "final_model")
            self.trainer.save_model(final_model_path)
            
            logger.info(f"Training completed. Model saved to {final_model_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Training failed: {e}")

    def run_evaluation(self) -> Dict[str, float]:
        """
        Run the evaluation process.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.trainer is None:
            raise ConfigurationError("Trainer must be created before running evaluation")
        
        if self.eval_dataset is None:
            raise DataLoadingError("Evaluation dataset must be loaded before evaluation")
        
        try:
            logger.info("Starting evaluation...")
            
            # Run evaluation
            eval_results = self.trainer.evaluate()
            
            # Extract metrics - handle both dict and object with .metrics attribute
            if hasattr(eval_results, 'metrics'):
                metrics = eval_results.metrics
            else:
                # eval_results is already a dict of metrics
                metrics = eval_results
            
            # Log results
            self._log_evaluation_results(metrics)
            
            logger.info("Evaluation completed successfully")
            return metrics
            
        except Exception as e:
            raise EvaluationError(f"Evaluation failed: {e}")

    def _log_evaluation_results(self, metrics: Dict[str, float]) -> None:
        """Log evaluation results."""
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

    def run(self) -> Dict[str, float]:
        """
        Run the complete pipeline (training and/or evaluation).
        
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Initialize components
            logger.info("Initializing MultiCoCo pipeline...")
            self.initialize_model()
            self.setup_datasets()
            self.create_trainer()
            
            # Run training if not eval-only
            if not self.config.training.eval_only:
                self.run_training()
            
            # Run evaluation
            metrics = self.run_evaluation()
            
            logger.info("Pipeline completed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise



def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="MultiCoCo: Chain of Continuous Thought for Multimodal AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "config",
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


def apply_cli_overrides(config: MultiCoCoConfig, args: argparse.Namespace) -> MultiCoCoConfig:
    """
    Apply command line overrides to configuration.
    
    Args:
        config: Base configuration
        args: Command line arguments
        
    Returns:
        Updated configuration
    """
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


def main() -> None:
    """Main entry point."""
    try:
        # Parse command line arguments
        parser = create_parser()
        args = parser.parse_args()
        
        # Load configuration
        # Load config using from_dict to support flat YAML format
        import yaml
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        config = MultiCoCoConfig.from_dict(yaml_config)
        
        # Apply command line overrides
        config = apply_cli_overrides(config, args)
        
        # Create and run pipeline
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
