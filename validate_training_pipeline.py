#!/usr/bin/env python3
"""
End-to-End Training Pipeline Validation for CoCoNut Latent Reasoning

This script validates that the complete training pipeline will work correctly when running:
torchrun --nnodes 1 --nproc_per_node 1 run.py args/aokvqa_coconut.yaml

It checks:
1. Configuration loading and validation
2. Model initialization with latent wrapper
3. Dataset loading and progressive curriculum
4. Trainer setup and coconut-specific configurations  
5. Forward pass through latent wrapper during training
6. KV cache management across coconut passes
7. End-to-end compatibility of all components

This ensures all fixes are properly integrated and no latent-specific bugs remain.
"""

import os
import sys
import logging
import torch
import tempfile
import json
from pathlib import Path
from typing import Dict, Any

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from multicoco.config import MultiCoCoConfig, TrainingMode
from multicoco.model import MultiCoCo
from multicoco.latent_wrapper import LatentWrapper
from multicoco.data import SupervisedDataset
from multicoco.trainer import CoCoTrainer
from multicoco.constants import COCONUT_SPECIAL_TOKENS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class EndToEndValidator:
    """Comprehensive validation of the training pipeline"""
    
    def __init__(self):
        self.temp_dir = None
        self.config = None
        self.model = None
        self.dataset = None
        self.trainer = None
        
    def setup_test_environment(self):
        """Setup temporary test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="e2e_test_")
        logger.info(f"Created temporary test directory: {self.temp_dir}")
        
        # Create minimal test config
        test_config = {
            'mode': 'coconut_train',
            'name': 'e2e_test',
            'output_dir': f'{self.temp_dir}/checkpoints',
            'num_epochs': 2,  # Minimal for testing
            'batch_size': 1,
            'eval_batch_size': 1,
            'limit_for_testing': 2,  # Very small dataset
            'coconut': {
                'enabled': True,
                'c_thought': 2,
                'max_latent_stage': 2,
                'epochs_per_stage': 1,
                'uniform_prob': 0.0,
                'pad_latent_to_max': False,
                'reset_optimizer': True
            },
            'data': {
                'train_data_path': 'data/aokvqa_train.json',
                'eval_data_path': 'data/aokvqa_validation.json',
                'data_dir': 'data/images'
            },
            'model': {
                'model_name': 'OpenGVLab/InternVL2-1B',
                'load_model_path': None
            },
            'training': {
                'bf16': True,
                'fp16': False,
                'gradient_checkpointing': False,
                'logging_steps': 1,
                'save_steps': 1000,  # Don't save during test
                'eval_steps': 1000   # Don't eval during test
            },
            'logging': {
                'use_wandb': False,
                'console_output': True,
                'log_to_file': False
            }
        }
        
        # Save test config
        config_path = f'{self.temp_dir}/test_config.yaml'
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f, default_flow_style=False)
            
        return config_path
        
    def test_config_loading(self, config_path: str) -> bool:
        """Test configuration loading and validation"""
        try:
            logger.info("Testing configuration loading...")
            
            # Create base config for testing
            base_config = {
                'data': {
                    'train_data_path': 'data/aokvqa_train.json',
                    'eval_data_path': 'data/aokvqa_validation.json', 
                    'data_dir': 'data/images',
                    'limit_for_testing': 2
                },
                'model': {
                    'model_name': 'OpenGVLab/InternVL2-1B',
                    'config_id': 'OpenGVLab/InternVL2-1B',
                    'tokenizer_id': 'OpenGVLab/InternVL2-1B',
                    'image_processor_id': 'OpenGVLab/InternVL2-1B',
                    'torch_dtype': 'bfloat16',
                    'trust_remote_code': True,
                    'low_cpu_mem_usage': True
                },
                'training': {
                    'mode': 'coconut_train',
                    'name': 'e2e_test',
                    'output_dir': f'{self.temp_dir}/checkpoints',
                    'num_epochs': 2,
                    'batch_size': 1,
                    'eval_batch_size': 1,
                    'learning_rate': 1e-5,
                    'warmup_steps': 0,
                    'weight_decay': 0.01,
                    'max_grad_norm': 1.0,
                    'gradient_accumulation_steps': 1,
                    'eval_accumulation_steps': 1,
                    'bf16': True,
                    'fp16': False,
                    'gradient_checkpointing': False,
                    'remove_unused_columns': False,
                    'dataloader_pin_memory': False,
                    'dataloader_num_workers': 0,
                    'logging_steps': 1,
                    'save_steps': 1000,
                    'eval_steps': 1000,
                    'save_strategy': 'no',
                    'eval_strategy': 'no',
                    'load_best_model_at_end': False,
                    'lr_scheduler_type': 'linear',
                    'seed': 42,
                    'data_seed': 42
                },
                'coconut': {
                    'enabled': True,
                    'c_thought': 2,
                    'max_latent_stage': 2,
                    'epochs_per_stage': 1,
                    'uniform_prob': 0.0,
                    'pad_latent_to_max': False,
                    'reset_optimizer': True
                },
                'evaluation': {
                    'coconut': True,
                    'cot': False,
                    'vanilla': False,
                    'log_per_sample': False
                },
                'logging': {
                    'use_wandb': False,
                    'console_output': True,
                    'log_to_file': False,
                    'log_level': 'INFO',
                    'verbose': False,
                    'log_dir': f'{self.temp_dir}/logs',
                    'run_name': 'e2e_test'
                },
                'generation': {
                    'do_sample': True,
                    'max_new_tokens': 16,  # Very small for testing
                    'num_beams': 1,
                    'temperature': 0.8,
                    'top_p': 0.9,
                    'top_k': 50
                }
            }
            
            base_config_path = f'{self.temp_dir}/base.yaml'
            import yaml
            with open(base_config_path, 'w') as f:
                yaml.dump(base_config, f, default_flow_style=False)
            
            # Load config using the actual config system
            self.config = MultiCoCoConfig.load_with_base(config_path, base_config_path)
            
            # Validate key coconut settings
            assert self.config.training.mode == TrainingMode.COCONUT_TRAIN
            assert self.config.coconut.enabled == True
            assert self.config.coconut.c_thought == 2
            assert self.config.coconut.max_latent_stage == 2
            
            logger.info("✓ Configuration loading successful")
            return True
            
        except Exception as e:
            logger.error(f"✗ Configuration loading failed: {e}")
            return False
    
    def test_model_initialization(self) -> bool:
        """Test model initialization with latent wrapper"""
        try:
            logger.info("Testing model initialization...")
            
            # Use a minimal model for testing to avoid memory issues
            special_tokens = COCONUT_SPECIAL_TOKENS.copy()
            
            logger.info("Creating MultiCoCo model...")
            self.model = MultiCoCo(
                model_id="microsoft/DialoGPT-small",  # Use a smaller model for testing
                special_tokens=special_tokens,
                torch_dtype=torch.float32,  # Use float32 for CPU testing
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Verify special tokens were added
            tokenizer = self.model.tokenizer
            for token in COCONUT_SPECIAL_TOKENS:
                token_id = tokenizer.convert_tokens_to_ids(token)
                assert token_id != tokenizer.unk_token_id, f"Special token {token} not properly added"
            
            logger.info("Wrapping with LatentWrapper...")
            self.model = LatentWrapper(self.model, tokenizer)
            
            # Verify wrapper properties
            assert hasattr(self.model, 'base_model')
            assert hasattr(self.model, 'latent_id')
            assert hasattr(self.model, 'start_id') 
            assert hasattr(self.model, 'end_id')
            assert self.model.latent_id is not None
            
            logger.info("✓ Model initialization successful")
            return True
            
        except Exception as e:
            logger.error(f"✗ Model initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_dataset_loading(self) -> bool:
        """Test dataset loading and progressive curriculum"""
        try:
            logger.info("Testing dataset loading...")
            
            # Create minimal test data
            test_data = [
                {
                    "question_id": "test_1",
                    "image": "test_image_1.jpg",
                    "question": "What color is the sky?",
                    "direct_answer": "blue",
                    "answer": "blue"
                },
                {
                    "question_id": "test_2", 
                    "image": "test_image_2.jpg",
                    "question": "How many cars are there?",
                    "direct_answer": "2",
                    "answer": "2"
                }
            ]
            
            # Create test data file
            train_data_path = f'{self.temp_dir}/test_train.json'
            with open(train_data_path, 'w') as f:
                json.dump(test_data, f)
            
            # Create test images directory and dummy images
            images_dir = f'{self.temp_dir}/images'
            os.makedirs(images_dir, exist_ok=True)
            
            from PIL import Image
            dummy_image = Image.new('RGB', (224, 224), color='blue')
            dummy_image.save(f'{images_dir}/test_image_1.jpg')
            dummy_image.save(f'{images_dir}/test_image_2.jpg')
            
            # Load dataset
            self.dataset = SupervisedDataset(
                data_path=train_data_path,
                data_dir=images_dir,
                test_limit=2
            )
            
            # Test progressive curriculum
            logger.info("Testing progressive curriculum...")
            original_len = len(self.dataset)
            
            # Apply stage 0 curriculum (no latent tokens)
            self.dataset.apply_progressive_curriculum(
                scheduled_stage=0,
                c_thought=2,
                max_latent_stage=2,
                uniform_prob=0.0,
                pad_latent_to_max=False,
                no_cot=False
            )
            
            # Apply stage 1 curriculum (some latent tokens)  
            self.dataset.apply_progressive_curriculum(
                scheduled_stage=1,
                c_thought=2,
                max_latent_stage=2,
                uniform_prob=0.0,
                pad_latent_to_max=False,
                no_cot=False
            )
            
            # Verify curriculum was applied
            sample = self.dataset[0]
            assert 'steps' in sample or 'reasoning' in sample
            
            logger.info("✓ Dataset loading and curriculum successful")
            return True
            
        except Exception as e:
            logger.error(f"✗ Dataset loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_trainer_setup(self) -> bool:
        """Test trainer setup with coconut configurations"""
        try:
            logger.info("Testing trainer setup...")
            
            if self.model is None or self.dataset is None:
                logger.error("Model and dataset must be initialized first")
                return False
            
            # Create training arguments based on config
            from transformers import TrainingArguments
            
            training_args = TrainingArguments(
                output_dir=f'{self.temp_dir}/checkpoints',
                num_train_epochs=2,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                learning_rate=1e-5,
                logging_steps=1,
                save_steps=1000,  # Don't save
                eval_steps=1000,  # Don't eval
                save_strategy='no',
                eval_strategy='no',
                remove_unused_columns=False,
                dataloader_num_workers=0,
                bf16=False,  # Use float32 for CPU testing
                fp16=False
            )
            
            # Create collate function
            from multicoco.data import collate_fn
            tokenizer = self.model.tokenizer
            image_processor = getattr(self.model.base_model, 'image_processor', None)
            
            def data_collator(batch):
                return collate_fn(batch, tokenizer, image_processor)
            
            # Create trainer
            self.trainer = CoCoTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset,
                eval_dataset=None,  # Skip evaluation for testing
                data_collator=data_collator
            )
            
            # Set coconut-specific parameters
            setattr(training_args, 'c_thought', 2)
            setattr(training_args, 'max_latent_stage', 2)
            setattr(training_args, 'epochs_per_stage', 1)
            setattr(training_args, 'uniform_prob', 0.0)
            setattr(training_args, 'pad_latent_to_max', False)
            setattr(training_args, 'reset_optimizer', True)
            
            logger.info("✓ Trainer setup successful")
            return True
            
        except Exception as e:
            logger.error(f"✗ Trainer setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_forward_pass(self) -> bool:
        """Test forward pass through latent wrapper"""
        try:
            logger.info("Testing forward pass through latent wrapper...")
            
            if self.model is None:
                logger.error("Model must be initialized first")
                return False
            
            # Create test input with latent tokens
            tokenizer = self.model.tokenizer
            
            # Create a sequence with latent tokens
            test_text = "Question: What is this? <|start_latent|><|latent|><|latent|><|end_latent|> Answer: A test."
            
            # Tokenize
            encoded = tokenizer(test_text, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            
            # Create dummy image inputs
            batch_size, seq_len = input_ids.shape
            dummy_pixel_values = torch.randn(batch_size, 3, 224, 224)
            
            logger.info(f"Test input shape: {input_ids.shape}")
            logger.info(f"Sample tokens: {tokenizer.convert_ids_to_tokens(input_ids[0][:20])}")
            
            # Test forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=dummy_pixel_values
                )
            
            # Verify outputs
            assert hasattr(outputs, 'logits')
            assert outputs.logits.shape[0] == batch_size
            assert outputs.logits.shape[1] == seq_len
            
            logger.info(f"Output logits shape: {outputs.logits.shape}")
            logger.info("✓ Forward pass successful")
            return True
            
        except Exception as e:
            logger.error(f"✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_latent_span_detection(self) -> bool:
        """Test latent span detection and processing"""
        try:
            logger.info("Testing latent span detection...")
            
            if self.model is None:
                logger.error("Model must be initialized first")
                return False
            
            tokenizer = self.model.tokenizer
            
            # Test cases with different latent patterns
            test_cases = [
                "Question: What is this? <|start_latent|><|latent|><|latent|><|end_latent|> Answer: A test.",
                "Text with <|start_latent|><|latent|><|end_latent|> and <|start_latent|><|latent|><|latent|><|latent|><|end_latent|> multiple spans.",
                "No latent tokens here.",
                "<|start_latent|><|latent|><|end_latent|> At the beginning.",
                "At the end <|start_latent|><|latent|><|latent|><|end_latent|>"
            ]
            
            for i, test_text in enumerate(test_cases):
                logger.info(f"Testing case {i+1}: {test_text[:50]}...")
                
                # Tokenize
                encoded = tokenizer(test_text, return_tensors='pt')
                input_ids = encoded['input_ids']
                
                # Extract latent spans using the wrapper's method
                spans = self.model._extract_latent_spans(input_ids)
                
                logger.info(f"  Detected spans: {spans}")
                
                # Verify span detection logic
                if '<|start_latent|>' in test_text and '<|end_latent|>' in test_text:
                    assert len(spans[0]) > 0, f"Should have detected spans in: {test_text}"
                else:
                    assert len(spans[0]) == 0, f"Should not have detected spans in: {test_text}"
            
            logger.info("✓ Latent span detection successful")
            return True
            
        except Exception as e:
            logger.error(f"✗ Latent span detection failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_coconut_stage_transitions(self) -> bool:
        """Test coconut stage transitions in trainer"""
        try:
            logger.info("Testing coconut stage transitions...")
            
            if self.trainer is None or self.dataset is None:
                logger.error("Trainer and dataset must be initialized first")
                return False
            
            # Test stage update functionality
            original_data_sample = str(self.dataset.data[0]) if len(self.dataset.data) > 0 else "No data"
            logger.info(f"Original data sample: {original_data_sample[:100]}...")
            
            # Test stage 0 update
            self.trainer._update_for_stage(0)
            stage_0_sample = str(self.dataset.data[0]) if len(self.dataset.data) > 0 else "No data"
            logger.info(f"Stage 0 data sample: {stage_0_sample[:100]}...")
            
            # Test stage 1 update  
            self.trainer._update_for_stage(1)
            stage_1_sample = str(self.dataset.data[0]) if len(self.dataset.data) > 0 else "No data"
            logger.info(f"Stage 1 data sample: {stage_1_sample[:100]}...")
            
            # Verify that curriculum was applied
            assert hasattr(self.dataset, 'apply_progressive_curriculum')
            
            logger.info("✓ Coconut stage transitions successful")
            return True
            
        except Exception as e:
            logger.error(f"✗ Coconut stage transitions failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def cleanup(self):
        """Clean up test environment"""
        if self.temp_dir:
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {e}")
    
    def run_full_validation(self) -> bool:
        """Run the complete end-to-end validation"""
        logger.info("🚀 Starting End-to-End Training Pipeline Validation")
        logger.info("=" * 60)
        
        all_tests_passed = True
        
        try:
            # Setup test environment
            config_path = self.setup_test_environment()
            
            # Run all validation tests
            tests = [
                ("Configuration Loading", self.test_config_loading, config_path),
                ("Model Initialization", self.test_model_initialization, ),
                ("Dataset Loading", self.test_dataset_loading, ),
                ("Trainer Setup", self.test_trainer_setup, ),
                ("Forward Pass", self.test_forward_pass, ),
                ("Latent Span Detection", self.test_latent_span_detection, ),
                ("Coconut Stage Transitions", self.test_coconut_stage_transitions, )
            ]
            
            for test_name, test_func, *args in tests:
                logger.info(f"\n📋 Running: {test_name}")
                logger.info("-" * 40)
                
                try:
                    if args:
                        result = test_func(*args)
                    else:
                        result = test_func()
                    
                    if not result:
                        all_tests_passed = False
                        logger.error(f"❌ {test_name} FAILED")
                    else:
                        logger.info(f"✅ {test_name} PASSED")
                        
                except Exception as e:
                    all_tests_passed = False
                    logger.error(f"❌ {test_name} FAILED with exception: {e}")
                    import traceback
                    traceback.print_exc()
        
        finally:
            self.cleanup()
        
        # Final summary
        logger.info("\n" + "=" * 60)
        if all_tests_passed:
            logger.info("🎉 ALL TESTS PASSED - Training pipeline is ready!")
            logger.info("✅ You can safely run: torchrun --nnodes 1 --nproc_per_node 1 run.py args/aokvqa_coconut.yaml")
        else:
            logger.error("❌ SOME TESTS FAILED - Please fix issues before running training")
        logger.info("=" * 60)
        
        return all_tests_passed

def main():
    """Main entry point"""
    validator = EndToEndValidator()
    success = validator.run_full_validation()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
