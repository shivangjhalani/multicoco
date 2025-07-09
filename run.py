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
from typing import Dict, Any, Optional
import random

# ** Core libraries
import torch
import torch.utils.checkpoint as cp  # type: ignore
from functools import partial
import numpy as np
from transformers import TrainingArguments, AutoModelForCausalLM

if not getattr(cp.checkpoint, "__patched_use_reentrant", False):
    cp.checkpoint = partial(cp.checkpoint, use_reentrant=False)  # type: ignore[arg-type]
    cp.checkpoint.__patched_use_reentrant = True  # type: ignore[attr-defined]
# ** Local imports
from multicoco.config import (
    MultiCoCoConfig,
    MultiCoCoConfig as _MC,
    TrainingMode
)
from multicoco.model import MultiCoCo
from multicoco.trainer import CoCoTrainer
from multicoco.data import SupervisedDataset, collate_fn
from multicoco.utils import TqdmLoggingHandler
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
from multicoco.latent_wrapper import LatentWrapper

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
        
        logger.info(f"MultiCoCoRunner initialized for {'training' if self.config.training.mode != TrainingMode.EVAL_ONLY else 'evaluation'}")

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
        # Only configure logging on the main process (rank 0)
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank not in [-1, 0]:
            logging.getLogger().setLevel(logging.CRITICAL)
            return
            
        log_config = self.config.logging
        os.makedirs(log_config.log_dir, exist_ok=True)
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Get the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_config.log_level))
        
        # Clear any existing handlers to prevent duplicates
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        # Create file handler
        file_handler = logging.FileHandler(os.path.join(log_config.log_dir, 'multicoco.log'), mode='w')
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)

        # Create console handler if enabled
        if log_config.console_output:
            # Use TqdmLoggingHandler for clean console output with progress bars
            console_handler = TqdmLoggingHandler()
            console_handler.setFormatter(logging.Formatter(log_format))
            root_logger.addHandler(console_handler)
        
        # Suppress transformers warnings if needed
        if not log_config.verbose:
            logging.getLogger("transformers").setLevel(logging.WARNING)
            logging.getLogger("torch").setLevel(logging.WARNING)

    def initialize_model(self) -> None:
        """Initialize the model from configuration with proper phase separation."""
        try:
            model_config = self.config.model
            coconut_config = self.config.coconut
            training_mode = self.config.training.mode
            
            # Phase-aware token handling: only add latent tokens when actually needed
            special_tokens = []
            if coconut_config.enabled or training_mode == TrainingMode.COCONUT_TRAIN:
                # Only add latent tokens for CoCoNut training or evaluation
                from multicoco.constants import COCONUT_SPECIAL_TOKENS
                special_tokens = list(set(model_config.get_special_tokens(coconut_config)) | set(COCONUT_SPECIAL_TOKENS))
                logger.info(f"Adding latent special tokens for CoCoNut phase: {special_tokens}")
            else:
                # CoT training - use only base special tokens
                special_tokens = model_config.get_special_tokens(coconut_config)
                logger.info("CoT training phase - no latent tokens added")
            
            # Initialize base model first
            if model_config.load_model_path:
                logger.info(f"Loading model from checkpoint: {model_config.load_model_path}")
                base_model_source = model_config.model_name  # Use base model for architecture
                checkpoint_path = model_config.load_model_path
            else:
                logger.info(f"Loading base model: {model_config.model_name}")
                base_model_source = model_config.model_name
                checkpoint_path = None
            
            # Initialize model with consistent architecture
            self.model = MultiCoCo(
                model_id=base_model_source,  # Always use base model for architecture
                config_id=model_config.config_id,
                tokenizer_id=model_config.tokenizer_id,
                image_processor_id=model_config.image_processor_id,
                special_tokens=special_tokens,
                torch_dtype=model_config.torch_dtype,
                trust_remote_code=model_config.trust_remote_code,
                low_cpu_mem_usage=model_config.low_cpu_mem_usage
            )
            
            # Load checkpoint state if provided (after base model initialization)
            if checkpoint_path:
                self._load_checkpoint_weights(checkpoint_path)
                logger.info(f"Loaded checkpoint weights from: {checkpoint_path}")
            
            # Initialize embeddings for latent tokens if they were added
            if special_tokens and any(tok in special_tokens for tok in ['<|latent|>', '<|start_latent|>', '<|end_latent|>']):
                self._initialize_latent_token_embeddings()
            
            # Wrap with LatentWrapper only for CoCoNut training/evaluation
            if coconut_config.enabled or training_mode == TrainingMode.COCONUT_TRAIN:
                logger.info("Wrapping model with LatentWrapper for CoCoNut training")
                self.model = LatentWrapper(self.model, self.model.tokenizer)
            
            # Log final model state
            if checkpoint_path:
                logger.info(f"Model initialized from checkpoint: {checkpoint_path}")
            else:
                logger.info(f"Base model '{model_config.model_name}' initialized successfully")
            logger.info(f"Model dtype: {model_config.torch_dtype}")
            logger.info(f"Model precision - BF16: {self.config.training.bf16}, FP16: {self.config.training.fp16}")
            logger.info(f"Training mode: {training_mode}")
            logger.info(f"CoCoNut enabled: {coconut_config.enabled}")
            
        except Exception as e:
            raise ModelInitializationError(f"Model initialization failed: {e}")

    def _load_checkpoint_weights(self, checkpoint_path: str) -> None:
        """Load checkpoint weights into the base model."""
        if self.model is None:
            raise ModelInitializationError("Model must be initialized before loading checkpoint weights")
            
        if not os.path.exists(checkpoint_path):
            raise ModelInitializationError(f"Checkpoint path does not exist: {checkpoint_path}")
        
        try:
            # Load the checkpoint model to get its state dict
            checkpoint_model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=self.model.model.dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Transfer weights to our model
            missing_keys, unexpected_keys = self.model.model.load_state_dict(
                checkpoint_model.state_dict(), strict=False
            )
            
            if missing_keys:
                logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
                
            # Clean up checkpoint model
            del checkpoint_model
            
        except Exception as e:
            raise ModelInitializationError(f"Failed to load checkpoint weights: {e}")

    def _initialize_latent_token_embeddings(self) -> None:
        """Initialize embeddings for latent tokens by copying EOS token embedding."""
        if self.model is None:
            raise ModelInitializationError("Model must be initialized before initializing token embeddings")
            
        try:
            embed_layer = self.model.get_input_embeddings()
            eos_vec = embed_layer.weight.data[self.model.tokenizer.eos_token_id].clone()
            
            from multicoco.constants import LATENT_TOKEN, START_LATENT_TOKEN, END_LATENT_TOKEN
            latent_tokens = [START_LATENT_TOKEN, LATENT_TOKEN, END_LATENT_TOKEN]
            
            initialized_tokens = []
            for tok in latent_tokens:
                tid = self.model.tokenizer.convert_tokens_to_ids(tok)
                if tid != self.model.tokenizer.unk_token_id:  # Token exists
                    embed_layer.weight.data[tid] = eos_vec.clone()
                    initialized_tokens.append(tok)
            
            if initialized_tokens:
                logger.info(f"Initialized embeddings for latent tokens: {initialized_tokens}")
                
        except Exception as e:
            logger.warning(f"Failed to initialize latent token embeddings: {e}")

    def setup_datasets(self) -> None:
        """Set up training and evaluation datasets."""
        try:
            data_config = self.config.data
            
            # Determine test limit for development
            test_limit = TEST_DATASET_LIMIT if data_config.limit_for_testing else None
            
            # Load training dataset if not eval-only
            if self.config.training.mode != TrainingMode.EVAL_ONLY and data_config.train_data_path:
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
                self.trainer.args.uniform_prob = self.config.coconut.uniform_prob
                self.trainer.args.pad_latent_to_max = self.config.coconut.pad_latent_to_max
                self.trainer.args.reset_optimizer = self.config.coconut.reset_optimizer
            
            logger.info("Trainer created successfully")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create trainer: {e}")

    def _create_training_arguments(self) -> TrainingArguments:
        """Create HuggingFace TrainingArguments from the configuration."""
        training_config = self.config.training
        is_training = training_config.mode != TrainingMode.EVAL_ONLY

        if is_training:
            # Training mode configuration
            return self._create_training_args(training_config)
        else:
            # Evaluation-only mode configuration
            return self._create_evaluation_args(training_config)

    def _create_training_args(self, training_config) -> TrainingArguments:
        """Create training arguments for training modes."""
        return TrainingArguments(
            output_dir=training_config.output_dir,
            num_train_epochs=training_config.num_epochs,
            per_device_train_batch_size=training_config.batch_size,
            per_device_eval_batch_size=training_config.eval_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            gradient_checkpointing=training_config.gradient_checkpointing,
            gradient_checkpointing_kwargs=training_config.gradient_checkpointing_kwargs,
            learning_rate=training_config.learning_rate,
            warmup_steps=training_config.warmup_steps,
            logging_steps=training_config.logging_steps,
            eval_strategy="epoch",  # Evaluate after each epoch
            save_strategy="epoch",       # Save after each epoch  
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
            report_to=["wandb"] if self.config.logging.use_wandb else [],
            run_name=training_config.name if hasattr(training_config, 'name') else None
        )

    def _create_evaluation_args(self, training_config) -> TrainingArguments:
        """Create training arguments for evaluation-only mode."""
        return TrainingArguments(
            output_dir=training_config.output_dir,
            per_device_eval_batch_size=training_config.eval_batch_size,
            bf16=training_config.bf16,
            fp16=training_config.fp16,
            remove_unused_columns=training_config.remove_unused_columns,
            dataloader_pin_memory=training_config.dataloader_pin_memory,
            dataloader_num_workers=training_config.dataloader_num_workers,
            do_train=False,
            do_eval=True,
            report_to=[],  # Disable wandb for eval-only
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
        """Run the training loop."""
        if self.trainer is None:
            raise ConfigurationError("Trainer not initialized")
        
        logger.info("Starting training...")
        self.trainer.train(resume_from_checkpoint=self.config.training.resume_from_checkpoint)

    def run_coconut_training(self) -> None:
        """
        Runs the second phase of training using progressive curriculum learning.
        
        This method implements the original CoCoNut methodology:
        - Progressive replacement of reasoning steps with latent tokens
        - Stage-by-stage training with epochs_per_stage epochs per stage
        - Optional optimizer reset between stages
        - Requires a CoT-trained model as initialization
        """
        if self.trainer is None:
            raise ConfigurationError("Trainer not initialized")
        
        logger.info("Starting CoCoNut progressive curriculum learning training...")
        
        # Latent tokens should already be properly initialized during model initialization
        # No need for redundant validation here
        
        # Use the new progressive curriculum learning training method
        self.trainer.train_coconut_progressive(
            resume_from_checkpoint=self.config.training.resume_from_checkpoint
        )

    def run_evaluation(self) -> Dict[str, float]:
        """Run a standalone evaluation."""
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
            
            if self.trainer.is_world_process_zero():
                logger.info("Evaluation completed successfully")
            return metrics
            
        except Exception as e:
            raise EvaluationError(f"Evaluation failed: {e}")

    def _log_evaluation_results(self, metrics: Dict[str, float]) -> None:
        """Log evaluation results in a structured format."""
        # Only log from the main process
        if self.trainer and self.trainer.is_world_process_zero():
            logger.info("\n==================================================")
            logger.info("FINAL RESULTS")
            logger.info("==================================================")
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
                
            logger.info("==================================================")

    def run(self) -> Dict[str, float]:
        """
        Orchestrates the full pipeline based on the training mode in the config.

        This method acts as a dispatcher:
        - `eval_only`: Runs only the evaluation loop.
        - `cot_train`: Runs a standard supervised fine-tuning pass. This is
                       the first phase of the full CoCoNuT process.
        - `coconut_train`: Runs the second phase, loading a CoT-tuned model
                           and training it with the CoCoNutModel for latent
                           space reasoning.
        """
        try:
            # Initialize model and datasets
            self.initialize_model()
            self.setup_datasets()
            
            # Execute based on training mode
            mode = self.config.training.mode

            if mode == TrainingMode.EVAL_ONLY:
                logger.info("Starting evaluation only...")
                self.create_trainer()
                results = self.run_evaluation()
                self._log_evaluation_results(results)
                return results

            elif mode == TrainingMode.COT_TRAIN:
                logger.info("Starting CoT training...")
                self.create_trainer()
                self.run_training()

            elif mode == TrainingMode.COCONUT_TRAIN:
                logger.info("Starting CoCoNuT training...")
                self.create_trainer()
                self.run_coconut_training()
                
            else:
                raise ConfigurationError(f"Invalid training mode: {mode}")

            # Run final evaluation
            logger.info("Running final evaluation...")
            results = self.run_evaluation()
            self._log_evaluation_results(results)
            return results

        except (ConfigurationError, ModelInitializationError, DataLoadingError, EvaluationError) as e:
            logger.error(f"Pipeline failed: {e}")
            raise



def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="MultiCoCo Runner: A two-phase training pipeline for multimodal models.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "config_path",
        type=str,
        help="""Path to the YAML configuration file.
            
To run the full pipeline, execute two runs:
1. First, run with a 'cot_train' config to fine-tune the model.
   Example: torchrun --nproc_per_node 1 run.py args/aokvqa_cot_train.yaml
   
2. Second, run with a 'coconut_train' config to perform latent reasoning training.
   Make sure the `model_name` in this config points to the checkpoint from step 1.
   Example: torchrun --nproc_per_node 1 run.py args/aokvqa_coconut_train.yaml
"""
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
        with open(args.config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        config = MultiCoCoConfig.from_dict(yaml_config)
        
        # Apply command line overrides
        config = apply_cli_overrides(config, args)
        
        # Automatically set evaluation mode based on training mode
        if config.training.mode == TrainingMode.COT_TRAIN:
            logger.info("CoT training mode detected, setting evaluation to CoT mode automatically.")
            config.evaluation.cot = True
            config.evaluation.vanilla = False
        
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
