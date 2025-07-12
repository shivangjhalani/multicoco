import argparse
import logging
import os
import random
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.checkpoint as checkpoint_module
from transformers import AutoModelForCausalLM, TrainingArguments
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()

from multicoco.config import MultiCoCoConfig, TrainingMode
from multicoco.constants import COCONUT_SPECIAL_TOKENS, DEFAULT_BATCH_SIZE, DEFAULT_EVAL_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_LOG_DIR, DEFAULT_MODEL_NAME, DEFAULT_NUM_EPOCHS, DEFAULT_OUTPUT_DIR, IMAGE_TOKEN
from multicoco.data import SupervisedDataset, collate_fn
from multicoco.exceptions import ConfigurationError, DataLoadingError, EvaluationError, ModelInitializationError
from multicoco.latent_wrapper import LatentWrapper
from multicoco.model import MultiCoCo
from multicoco.trainer import CoCoTrainer

logger = logging.getLogger(__name__)


class MultiCoCoRunner:
    def __init__(self, config: MultiCoCoConfig):
        self.config = config
        self.model: Optional[MultiCoCo] = None
        self.trainer: Optional[CoCoTrainer] = None
        self.train_dataset: Optional[SupervisedDataset] = None
        self.eval_dataset: Optional[SupervisedDataset] = None
        self.wandb_run: Optional[Any] = None
        self.run_log_dir: Optional[str] = None
        self._initialize()
        mode_type = 'training' if config.training.mode != TrainingMode.EVAL_ONLY else 'evaluation'
        logger.info(f'MultiCoCoRunner initialized for {mode_type}')

    def _initialize(self) -> None:
        if self.config.training.seed is not None:
            self._set_random_seeds(self.config.training.seed)
        self._setup_logging()
        self._setup_cuda()
        self._setup_wandb()

    def _set_random_seeds(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info(f'Set random seed to {seed}')

    def _setup_cuda(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            logger.info(f'CUDA available with {torch.cuda.device_count()} devices')
        else:
            logger.warning('CUDA not available, using CPU')

    def _setup_logging(self) -> None:
        local_rank = int(os.environ.get('LOCAL_RANK', -1))
        if local_rank > 0:
            logging.getLogger().setLevel(logging.CRITICAL)
            return
        log_cfg = self.config.logging
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        run_name = log_cfg.run_name or 'run'
        self.run_log_dir = os.path.join(log_cfg.log_dir, f'{run_name}_{timestamp}')
        os.makedirs(self.run_log_dir, exist_ok=True)
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_cfg.log_level))
        if root_logger.hasHandlers():
            root_logger.handlers.clear()
        if log_cfg.log_to_file:
            run_log_path = os.path.join(self.run_log_dir, 'run.log')
            handler = logging.FileHandler(run_log_path)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)
        logger.info(f'Logging initialized. Output saved to: {self.run_log_dir}')

    def _setup_wandb(self) -> None:
        if not self.config.logging.use_wandb:
            return
        local_rank = int(os.environ.get('LOCAL_RANK', -1))
        if local_rank not in [-1, 0]:
            return
        try:
            import wandb
            from dataclasses import asdict
            run_name = self.config.logging.run_name or self.config.training.name or f'run_{random.randint(0, 1000000)}'
            project = self.config.logging.project or 'multicoco'
            self.wandb_run = wandb.init(project=project, name=run_name, reinit=True)
            cfg_dict = asdict(self.config)
            self.wandb_run.config.update(cfg_dict, allow_val_change=True)
            logger.info(f'Initialized wandb run: project={project}, name={run_name}')
        except ImportError:
            logger.warning('wandb not found; skipping integration')
            self.config.logging.use_wandb = False

    def initialize_model(self) -> None:
        try:
            model_config = self.config.model
            coconut_config = self.config.coconut
            training_mode = self.config.training.mode
            special_tokens = self._get_special_tokens(coconut_config, training_mode)
            base_model_source, checkpoint_path = self._get_model_source()
            self.model = MultiCoCo(model_id=base_model_source, config_id=model_config.config_id, tokenizer_id=model_config.tokenizer_id, image_processor_id=model_config.image_processor_id, special_tokens=special_tokens, torch_dtype=model_config.torch_dtype, trust_remote_code=model_config.trust_remote_code, low_cpu_mem_usage=model_config.low_cpu_mem_usage)
            self._finalize_model_setup(checkpoint_path, special_tokens, coconut_config, training_mode)
        except Exception as e:
            raise ModelInitializationError(f'Model initialization failed: {e}') from e

    def _finalize_model_setup(self, checkpoint_path: Optional[str], special_tokens: list, coconut_config, training_mode) -> None:
        if checkpoint_path:
            self._load_checkpoint_weights(checkpoint_path)
        if self._has_latent_tokens(special_tokens):
            self._initialize_latent_token_embeddings()
        if self._needs_latent_wrapper(coconut_config, training_mode):
            self.model = LatentWrapper(self.model, self.model.tokenizer)
        self._log_model_info(checkpoint_path, training_mode, coconut_config)

    def _get_special_tokens(self, coconut_config, training_mode) -> list:
        if coconut_config.enabled or training_mode == TrainingMode.COCONUT_TRAIN:
            special_tokens = list(set(self.config.model.get_special_tokens(coconut_config)) | set(COCONUT_SPECIAL_TOKENS))
            logger.info(f'Adding latent special tokens: {special_tokens}')
        else:
            special_tokens = self.config.model.get_special_tokens(coconut_config)
            logger.info('CoT training phase - no latent tokens added')
        return special_tokens

    def _get_model_source(self) -> tuple[str, Optional[str]]:
        model_config = self.config.model
        if model_config.load_model_path:
            logger.info(f'Loading from checkpoint: {model_config.load_model_path}')
            return (model_config.model_name, model_config.load_model_path)
        else:
            logger.info(f'Loading base model: {model_config.model_name}')
            return (model_config.model_name, None)

    def _has_latent_tokens(self, special_tokens: list) -> bool:
        latent_tokens = ['<|latent|>', '<|start_latent|>', '<|end_latent|>']
        return any(tok in special_tokens for tok in latent_tokens)

    def _needs_latent_wrapper(self, coconut_config, training_mode) -> bool:
        return coconut_config.enabled or training_mode == TrainingMode.COCONUT_TRAIN

    def _log_model_info(self, checkpoint_path: Optional[str], training_mode, coconut_config) -> None:
        source_info = f'checkpoint: {checkpoint_path}' if checkpoint_path else f'base model: {self.config.model.model_name}'
        logger.info(f'Model initialized from {source_info}')
        logger.info(f'Dtype: {self.config.model.torch_dtype}, BF16: {self.config.training.bf16}, FP16: {self.config.training.fp16}')
        logger.info(f'Mode: {training_mode}, CoCoNut: {coconut_config.enabled}')

    def _load_checkpoint_weights(self, checkpoint_path: str) -> None:
        if self.model is None:
            raise ModelInitializationError('Model must be initialized first')
        if not os.path.exists(checkpoint_path):
            raise ModelInitializationError(f'Checkpoint path does not exist: {checkpoint_path}')
        try:
            checkpoint_model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=self.model.model.dtype, trust_remote_code=True, low_cpu_mem_usage=True)
            missing_keys, unexpected_keys = self.model.model.load_state_dict(checkpoint_model.state_dict(), strict=False)
            if missing_keys:
                logger.warning(f'Missing keys: {missing_keys}')
            if unexpected_keys:
                logger.warning(f'Unexpected keys: {unexpected_keys}')
            del checkpoint_model
        except Exception as e:
            raise ModelInitializationError(f'Failed to load checkpoint weights: {e}') from e

    def _initialize_latent_token_embeddings(self) -> None:
        if self.model is None:
            raise ModelInitializationError('Model must be initialized first')
        try:
            embed_layer = self.model.get_input_embeddings()
            with torch.no_grad():
                eos_token_id = self.model.tokenizer.eos_token_id
                eos_embedding = embed_layer.weight[eos_token_id].clone()
                image_token_id = self.model.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
                if image_token_id is None or image_token_id >= embed_layer.weight.size(0):
                    multimodal_embedding = eos_embedding
                else:
                    image_embedding = embed_layer.weight[image_token_id].clone()
                    multimodal_embedding = (eos_embedding + image_embedding) / 2.0
                for token in COCONUT_SPECIAL_TOKENS:
                    token_id = self.model.tokenizer.convert_tokens_to_ids(token)
                    if token_id is not None and token_id < embed_layer.weight.size(0):
                        embed_layer.weight[token_id] = multimodal_embedding.clone()
        except Exception as e:
            raise ModelInitializationError(f'Failed to initialize latent token embeddings: {e}') from e

    def setup_datasets(self) -> None:
        try:
            data_config = self.config.data
            test_limit = data_config.limit_for_testing
            
            # Convert boolean True to a reasonable test limit, False to None
            if isinstance(test_limit, bool):
                test_limit = 20 if test_limit else None
            
            if self.config.training.mode != TrainingMode.EVAL_ONLY and data_config.train_data_path:
                self.train_dataset = SupervisedDataset(data_path=data_config.train_data_path, data_dir=data_config.data_dir, test_limit=test_limit)
                logger.info(f'Training dataset: {len(self.train_dataset)} samples')
            if data_config.eval_data_path:
                self.eval_dataset = SupervisedDataset(data_path=data_config.eval_data_path, data_dir=data_config.data_dir, test_limit=test_limit)
                logger.info(f'Evaluation dataset: {len(self.eval_dataset)} samples')
        except Exception as e:
            raise DataLoadingError(f'Dataset loading failed: {e}') from e

    def create_trainer(self) -> None:
        if self.model is None:
            raise ModelInitializationError('Model must be initialized first')
        try:
            training_args = self._create_training_arguments()
            self.trainer = CoCoTrainer(model=self.model, args=training_args, train_dataset=self.train_dataset, eval_dataset=self.eval_dataset, data_collator=lambda batch: collate_fn(batch, self.model.tokenizer, self.model.image_processor))
            if self.config.coconut.enabled:
                self._set_coconut_trainer_params()
            logger.info('Trainer created successfully')
        except Exception as e:
            raise ModelInitializationError(f'Trainer creation failed: {e}') from e

    def _set_coconut_trainer_params(self) -> None:
        if self.trainer is None:
            return
        coconut_cfg = self.config.coconut
        attrs = ['c_thought', 'max_latent_stage', 'epochs_per_stage', 'uniform_prob', 'pad_latent_to_max', 'reset_optimizer']
        for attr in attrs:
            setattr(self.trainer.args, attr, getattr(coconut_cfg, attr))

    def _create_training_arguments(self) -> TrainingArguments:
        training_config = self.config.training
        common_args = {
            'output_dir': training_config.output_dir,
            'num_train_epochs': training_config.num_epochs,
            'per_device_train_batch_size': training_config.batch_size,
            'per_device_eval_batch_size': training_config.eval_batch_size,
            'gradient_accumulation_steps': training_config.gradient_accumulation_steps,
            'eval_accumulation_steps': training_config.eval_accumulation_steps,
            'learning_rate': training_config.learning_rate,
            'warmup_steps': training_config.warmup_steps,
            'logging_steps': training_config.logging_steps,
            'save_steps': training_config.save_steps,
            'eval_steps': training_config.eval_steps,
            'save_strategy': 'epoch',
            'eval_strategy': 'epoch',
            'save_total_limit': training_config.save_total_limit,
            'load_best_model_at_end': training_config.load_best_model_at_end,
            'metric_for_best_model': training_config.metric_for_best_model,
            'greater_is_better': training_config.greater_is_better,
            'bf16': training_config.bf16,
            'fp16': training_config.fp16,
            'remove_unused_columns': training_config.remove_unused_columns,
            'dataloader_pin_memory': training_config.dataloader_pin_memory,
            'dataloader_num_workers': training_config.dataloader_num_workers,
            'gradient_checkpointing': training_config.gradient_checkpointing,
            'gradient_checkpointing_kwargs': training_config.gradient_checkpointing_kwargs,
            'weight_decay': training_config.weight_decay,
            'seed': training_config.seed,
            'data_seed': training_config.data_seed,
            'report_to': self.config.get_wandb_report_to(),
        }
        return TrainingArguments(**common_args)

    def run_training(self) -> None:
        if self.trainer is None:
            raise ModelInitializationError('Trainer must be initialized first')
        if self.train_dataset is None or len(self.train_dataset) == 0:
            raise DataLoadingError('Training dataset is empty or not loaded')
        logger.info('Starting training...')
        self.trainer.train()

    def run_evaluation(self) -> Dict[str, float]:
        if self.trainer is None or self.eval_dataset is None or len(self.eval_dataset) == 0:
            raise EvaluationError('Evaluation dataset is empty or not initialized')
        logger.info('Starting evaluation...')
        metrics = self.trainer.perform_evaluation(log_per_sample=self.config.evaluation.log_per_sample)
        self._log_evaluation_results(metrics)
        return metrics

    def run(self) -> Dict[str, float]:
        try:
            self.initialize_model()
            self.setup_datasets()
            mode = self.config.training.mode
            if mode == TrainingMode.EVAL_ONLY:
                return self._run_eval_only()
            elif mode == TrainingMode.COT_TRAIN:
                return self._run_training_mode()
            elif mode == TrainingMode.COCONUT_TRAIN:
                return self._run_coconut_mode()
            else:
                raise ConfigurationError(f'Invalid training mode: {mode}')
        except (ConfigurationError, ModelInitializationError, DataLoadingError, EvaluationError) as e:
            logger.error(f'Pipeline failed: {e}')
            raise
        finally:
            if self.wandb_run is not None:
                import wandb
                wandb.finish()

    def _run_eval_only(self) -> Dict[str, float]:
        logger.info('Starting evaluation only...')
        self.create_trainer()
        return self.run_evaluation()

    def _run_training_mode(self) -> Dict[str, float]:
        logger.info('Starting CoT training...')
        self.create_trainer()
        self.run_training()
        return self._run_final_evaluation()

    def _run_coconut_mode(self) -> Dict[str, float]:
        logger.info('Starting CoCoNut training...')
        self.create_trainer()
        self.run_training()
        return self._run_final_evaluation()

    def _run_final_evaluation(self) -> Dict[str, float]:
        logger.info('Running final evaluation...')
        return self.run_evaluation()

    def _log_evaluation_results(self, metrics: Dict[str, float]) -> None:
        if self.trainer and self.trainer.is_world_process_zero():
            logger.info('\n' + '=' * 50)
            logger.info('EVALUATION SUMMARY')
            logger.info('=' * 50)
            for key, value in metrics.items():
                logger.info(f'  {key}: {value:.4f}')
            logger.info('=' * 50)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='MultiCoCo: Two-phase training for multimodal models.')
    parser.add_argument('config_path', type=str, help='Path to YAML configuration file')
    parser.add_argument('--eval-only', action='store_true', help='Run evaluation only (skip training)')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    parser.add_argument('--model-name', type=str, help='Override model name')
    return parser


def apply_cli_overrides(config: MultiCoCoConfig, args: argparse.Namespace) -> MultiCoCoConfig:
    if args.eval_only:
        config.training.mode = TrainingMode.EVAL_ONLY
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.model_name:
        config.model.model_name = args.model_name
    return config


def _load_config(config_path: str) -> MultiCoCoConfig:
    base_cfg_path = os.path.join(os.path.dirname(config_path), 'base.yaml')
    return MultiCoCoConfig.load_with_base(config_path=config_path, base_config_path=base_cfg_path)


def main() -> None:
    try:
        parser = create_parser()
        args = parser.parse_args()
        config = _load_config(args.config_path)
        config = apply_cli_overrides(config, args)
        if config.training.mode == TrainingMode.COT_TRAIN:
            config.evaluation.cot = True
            config.evaluation.vanilla = False
        runner = MultiCoCoRunner(config)
        metrics = runner.run()
        print('\n' + '=' * 50)
        print('FINAL RESULTS')
        print('=' * 50)
        for key, value in metrics.items():
            print(f'{key}: {value}')
        print('=' * 50)
    except KeyboardInterrupt:
        print('\nInterrupted by user')
        sys.exit(1)
    except Exception as e:
        print(f'Error: {e}')
        sys.exit(1)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()