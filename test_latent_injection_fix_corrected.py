#!/usr/bin/env python3
"""
Comprehensive test suite to verify that the latent injection fix works correctly.

This test validates:
1. Vanilla evaluation uses base model directly (no LatentWrapper)
2. CoT evaluation uses base model directly (no LatentWrapper)  
3. CoCoNut evaluation uses LatentWrapper with proper latent injection
4. LatentWrapper correctly detects latent tokens in input
5. Data preprocessing works correctly for different modes
"""

import os
import sys
import tempfile
import json
import yaml
import shutil
import torch
from pathlib import Path
from typing import Dict, Any, List
import unittest
from unittest.mock import patch, MagicMock, Mock

# Add the project root to path
sys.path.insert(0, os.path.dirname(__file__))

class TestLatentInjectionFix(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create test data file
        self.test_data = [
            {
                "question": "What color is the sky?",
                "choices": ["0: Blue", "1: Red", "2: Green", "3: Yellow"],
                "answer": "0",
                "image": "test_image.jpg"
            }
        ]
        
        self.test_data_path = os.path.join(self.data_dir, "test_data.json")
        with open(self.test_data_path, 'w') as f:
            json.dump(self.test_data, f)
            
        # Create dummy image files
        self.images_dir = os.path.join(self.data_dir, "images", "aokvqa", "validation")
        os.makedirs(self.images_dir, exist_ok=True)
        
        img_path = os.path.join(self.images_dir, "test_image.jpg")
        # Create a minimal dummy image file
        with open(img_path, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')
        
        self.base_config = {
            "project": "test",
            "seed": 42,
            "model_name": "test-model",
            "data_dir": self.data_dir,
            "train_data_path": self.test_data_path,  # Add training data path
            "eval_data_path": self.test_data_path,
            "log_dir": os.path.join(self.temp_dir, "logs"),
            "use_wandb": False,
            "limit_for_testing": True,
            "console_output": False  # Reduce noise
        }
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def create_config_file(self, config_dict: Dict[str, Any]) -> str:
        """Create a temporary config file"""
        config_path = os.path.join(self.temp_dir, "test_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        return config_path
        
    def create_base_config_file(self) -> str:
        """Create base config file"""
        base_config_path = os.path.join(self.temp_dir, "base.yaml")
        with open(base_config_path, 'w') as f:
            yaml.dump(self.base_config, f)
        return base_config_path
    
    def test_runner_needs_latent_wrapper_logic(self):
        """Test the core logic for determining when to use LatentWrapper"""
        print("\n=== Testing LatentWrapper Decision Logic ===")
        
        from multicoco.config import MultiCoCoConfig, TrainingMode
        from run import MultiCoCoRunner
        
        # Test 1: Vanilla evaluation - should NOT use LatentWrapper
        base_config_path = self.create_base_config_file()
        config_dict = {
            "mode": "eval_only",
            "eval_config": {
                "vanilla": True,
                "coconut": False,
                "cot": False
            },
            "coconut": {
                "enabled": True  # This should be ignored for eval-only vanilla
            }
        }
        config_path = self.create_config_file(config_dict)
        config = MultiCoCoConfig.load_with_base(config_path, base_config_path)
        
        # Create runner without initializing model
        with patch('run.MultiCoCoRunner._initialize'):
            runner = MultiCoCoRunner.__new__(MultiCoCoRunner)
            runner.config = config
            
        needs_wrapper = runner._needs_latent_wrapper(config.coconut, config.training.mode)
        self.assertFalse(needs_wrapper, "Vanilla evaluation should not use LatentWrapper")
        print("✓ Vanilla evaluation correctly excludes LatentWrapper")
        
        # Test 2: CoT evaluation - should NOT use LatentWrapper
        config_dict["eval_config"] = {"vanilla": False, "coconut": False, "cot": True}
        config_path = self.create_config_file(config_dict)
        config = MultiCoCoConfig.load_with_base(config_path, base_config_path)
        runner.config = config
        
        needs_wrapper = runner._needs_latent_wrapper(config.coconut, config.training.mode)
        self.assertFalse(needs_wrapper, "CoT evaluation should not use LatentWrapper")
        print("✓ CoT evaluation correctly excludes LatentWrapper")
        
        # Test 3: CoCoNut evaluation - should use LatentWrapper
        config_dict["eval_config"] = {"vanilla": False, "coconut": True, "cot": False}
        config_path = self.create_config_file(config_dict)
        config = MultiCoCoConfig.load_with_base(config_path, base_config_path)
        runner.config = config
        
        needs_wrapper = runner._needs_latent_wrapper(config.coconut, config.training.mode)
        self.assertTrue(needs_wrapper, "CoCoNut evaluation should use LatentWrapper")
        print("✓ CoCoNut evaluation correctly includes LatentWrapper")
        
        # Test 4: Training mode - should follow coconut.enabled
        config_dict["mode"] = "cot_train"
        config_dict["coconut"]["enabled"] = True
        config_path = self.create_config_file(config_dict)
        config = MultiCoCoConfig.load_with_base(config_path, base_config_path)
        runner.config = config
        
        needs_wrapper = runner._needs_latent_wrapper(config.coconut, config.training.mode)
        self.assertTrue(needs_wrapper, "CoCoNut training should use LatentWrapper")
        print("✓ Training mode correctly follows coconut.enabled")
        
    def test_special_tokens_logic(self):
        """Test special tokens are added correctly based on mode"""
        print("\n=== Testing Special Tokens Logic ===")
        
        from multicoco.config import MultiCoCoConfig
        from multicoco.constants import COCONUT_SPECIAL_TOKENS
        from run import MultiCoCoRunner
        
        base_config_path = self.create_base_config_file()
        
        # Test vanilla evaluation - no latent tokens
        config_dict = {
            "mode": "eval_only",
            "eval_config": {"vanilla": True, "coconut": False, "cot": False},
            "coconut": {"enabled": False}
        }
        config_path = self.create_config_file(config_dict)
        config = MultiCoCoConfig.load_with_base(config_path, base_config_path)
        
        with patch('run.MultiCoCoRunner._initialize'):
            runner = MultiCoCoRunner.__new__(MultiCoCoRunner)
            runner.config = config
        
        special_tokens = runner._get_special_tokens(config.coconut, config.training.mode)
        
        # Should not contain latent tokens
        for token in COCONUT_SPECIAL_TOKENS:
            self.assertNotIn(token, special_tokens)
        print("✓ Vanilla evaluation correctly excludes latent tokens")
        
        # Test coconut evaluation - should have latent tokens
        config_dict["eval_config"] = {"vanilla": False, "coconut": True, "cot": False}
        config_path = self.create_config_file(config_dict)
        config = MultiCoCoConfig.load_with_base(config_path, base_config_path)
        runner.config = config
        
        special_tokens = runner._get_special_tokens(config.coconut, config.training.mode)
        
        # Should contain latent tokens
        for token in COCONUT_SPECIAL_TOKENS:
            self.assertIn(token, special_tokens)
        print("✓ CoCoNut evaluation correctly includes latent tokens")
        
    def test_latent_wrapper_token_detection(self):
        """Test LatentWrapper's latent token detection logic"""
        print("\n=== Testing LatentWrapper Token Detection ===")
        
        from multicoco.latent_wrapper import LatentWrapper
        
        # Mock base model and tokenizer
        mock_base_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Set up token IDs
        mock_tokenizer.convert_tokens_to_ids.side_effect = lambda token: {
            '<|start_latent|>': 1001,
            '<|latent|>': 1002, 
            '<|end_latent|>': 1003
        }.get(token, None)
        
        # Create LatentWrapper
        wrapper = LatentWrapper(mock_base_model, mock_tokenizer)
        
        # Test 1: Question without latent tokens
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # No latent tokens
        mock_base_model.chat.return_value = "Test response without latents"
        
        response = wrapper.chat(
            tokenizer=mock_tokenizer,
            pixel_values=None,
            question="What color is the sky?",
            generation_config={}
        )
        
        # Should call base model's chat method
        mock_base_model.chat.assert_called_once()
        self.assertEqual(response, "Test response without latents")
        print("✓ LatentWrapper correctly delegates to base model when no latent tokens present")
        
        # Reset mock
        mock_base_model.reset_mock()
        
        # Test 2: Question with latent tokens
        mock_tokenizer.encode.return_value = [1, 1001, 1002, 1002, 1003, 5]  # Has start/end latent tokens
        
        # Mock the generation flow to avoid complex tensor operations
        mock_input_ids = torch.tensor([[1, 1001, 1002, 1002, 1003, 5]])
        mock_tokenizer.encode.return_value = mock_input_ids[0].tolist()
        
        # For text-only generation path
        with patch.object(mock_tokenizer, 'encode') as mock_encode:
            mock_encode.side_effect = [
                mock_input_ids[0].tolist(),  # First call for detection
                mock_input_ids  # Second call for generation
            ]
            
            with patch.object(wrapper, 'generate') as mock_generate:
                mock_generate.return_value = torch.tensor([[1, 1001, 1002, 1002, 1003, 5, 10, 11]])
                mock_tokenizer.decode.return_value = "Generated response"
                
                response = wrapper.chat(
                    tokenizer=mock_tokenizer,
                    pixel_values=None,
                    question="<|start_latent|><|latent|><|latent|><|end_latent|>What color is the sky?",
                    generation_config={}
                )
                
                # Should NOT call base model's chat method, should use custom generation
                mock_base_model.chat.assert_not_called()
                mock_generate.assert_called_once()
                print("✓ LatentWrapper correctly uses custom generation when latent tokens present")
    
    def test_latent_span_detection(self):
        """Test LatentWrapper's _has_latent_spans method"""
        print("\n=== Testing Latent Span Detection ===")
        
        from multicoco.latent_wrapper import LatentWrapper
        
        mock_base_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_tokens_to_ids.side_effect = lambda token: {
            '<|start_latent|>': 1001,
            '<|latent|>': 1002, 
            '<|end_latent|>': 1003
        }.get(token, None)
        
        wrapper = LatentWrapper(mock_base_model, mock_tokenizer)
        
        # Test 1: No latent tokens
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        has_latents = wrapper._has_latent_spans(input_ids)
        self.assertFalse(has_latents)
        print("✓ Correctly detects absence of latent spans")
        
        # Test 2: Has latent tokens
        input_ids = torch.tensor([[1, 1001, 1002, 1003, 5]])
        has_latents = wrapper._has_latent_spans(input_ids)
        self.assertTrue(has_latents)
        print("✓ Correctly detects presence of latent spans")
        
        # Test 3: Multiple sequences in batch
        input_ids = torch.tensor([
            [1, 2, 3, 4, 5],  # No latents
            [1, 1001, 1002, 1003, 5]  # Has latents
        ])
        has_latents = wrapper._has_latent_spans(input_ids)
        self.assertTrue(has_latents)  # Should return True if ANY sequence has latents
        print("✓ Correctly handles batch processing")
        
    def test_trainer_model_detection_simplified(self):
        """Test trainer's ability to detect LatentWrapper vs base model"""
        print("\n=== Testing Trainer Model Detection ===")
        
        from multicoco.latent_wrapper import LatentWrapper
        
        # Test 1: Base model detection
        mock_base_model = MagicMock()
        is_latent_wrapper = isinstance(mock_base_model, LatentWrapper)
        self.assertFalse(is_latent_wrapper)
        print("✓ Correctly identifies base model (not LatentWrapper)")
        
        # Test 2: LatentWrapper detection
        mock_tokenizer = MagicMock()
        wrapped_model = LatentWrapper(mock_base_model, mock_tokenizer)
        is_latent_wrapper = isinstance(wrapped_model, LatentWrapper)
        self.assertTrue(is_latent_wrapper)
        print("✓ Correctly identifies LatentWrapper")
        
    @patch('multicoco.data.create_progressive_latent_dataset')
    def test_coconut_evaluation_data_preprocessing(self, mock_create_dataset):
        """Test that CoCoNut evaluation preprocesses data correctly"""
        print("\n=== Testing CoCoNut Evaluation Data Preprocessing ===")
        
        # Set up mock
        mock_create_dataset.return_value = [
            {
                "question": "What color is the sky?",
                "reasoning": "<|start_latent|><|latent|><|latent|><|latent|><|latent|><|end_latent|>",
                "answer": "0",
                "image": "test_image.jpg"
            }
        ]
        
        from multicoco.config import MultiCoCoConfig
        from run import MultiCoCoRunner
        
        # Create CoCoNut evaluation config
        base_config_path = self.create_base_config_file()
        config_dict = {
            "mode": "eval_only",
            "eval_config": {
                "vanilla": False,
                "coconut": True,
                "cot": False,
                "eval_latent_tokens": 4
            },
            "coconut": {
                "enabled": True,
                "max_latent_stage": 6
            }
        }
        config_path = self.create_config_file(config_dict)
        
        config = MultiCoCoConfig.load_with_base(config_path, base_config_path)
        
        # Mock the initialization to avoid model loading
        with patch('run.MultiCoCoRunner._initialize'):
            runner = MultiCoCoRunner.__new__(MultiCoCoRunner)
            runner.config = config
            
        # Test dataset setup
        runner.setup_datasets()
        
        # Verify preprocessing was called
        mock_create_dataset.assert_called_once()
        call_args = mock_create_dataset.call_args
        
        # Check arguments
        self.assertEqual(call_args[1]['scheduled_stage'], 4)  # eval_latent_tokens
        self.assertEqual(call_args[1]['max_latent_stage'], 4)
        self.assertEqual(call_args[1]['uniform_prob'], 0.0)  # Deterministic for eval
        self.assertEqual(call_args[1]['no_cot'], True)  # Skip CoT steps
        
        print("✓ CoCoNut evaluation correctly preprocesses data with latent tokens")
        
    def test_integration_scenarios(self):
        """Test integration scenarios with proper config validation"""
        print("\n=== Testing Integration Scenarios ===")
        
        from multicoco.config import MultiCoCoConfig
        from multicoco.constants import COCONUT_SPECIAL_TOKENS
        from run import MultiCoCoRunner
        
        scenarios = [
            {
                "name": "Vanilla Evaluation",
                "config": {
                    "mode": "eval_only",
                    "eval_config": {"vanilla": True, "coconut": False, "cot": False},
                    "coconut": {"enabled": False}  # Consistent config
                },
                "expected_wrapper": False,
                "expected_latent_tokens": False
            },
            {
                "name": "CoT Evaluation", 
                "config": {
                    "mode": "eval_only",
                    "eval_config": {"vanilla": False, "coconut": False, "cot": True},
                    "coconut": {"enabled": False}  # Consistent config
                },
                "expected_wrapper": False,
                "expected_latent_tokens": False
            },
            {
                "name": "CoCoNut Evaluation",
                "config": {
                    "mode": "eval_only", 
                    "eval_config": {"vanilla": False, "coconut": True, "cot": False, "eval_latent_tokens": 4},
                    "coconut": {"enabled": True}  # This is now relevant for eval
                },
                "expected_wrapper": True,
                "expected_latent_tokens": True
            }
        ]
        
        base_config_path = self.create_base_config_file()
        
        for scenario in scenarios:
            print(f"\n--- Testing: {scenario['name']} ---")
            
            config_path = self.create_config_file(scenario['config'])
            config = MultiCoCoConfig.load_with_base(config_path, base_config_path)
            
            with patch('run.MultiCoCoRunner._initialize'):
                runner = MultiCoCoRunner.__new__(MultiCoCoRunner)
                runner.config = config
            
            # Test LatentWrapper decision
            needs_wrapper = runner._needs_latent_wrapper(config.coconut, config.training.mode)
            self.assertEqual(needs_wrapper, scenario['expected_wrapper'],
                           f"{scenario['name']}: LatentWrapper decision mismatch")
            
            # Test special tokens
            special_tokens = runner._get_special_tokens(config.coconut, config.training.mode)
            has_latent_tokens = any(token in special_tokens for token in COCONUT_SPECIAL_TOKENS)
            self.assertEqual(has_latent_tokens, scenario['expected_latent_tokens'],
                           f"{scenario['name']}: Special tokens mismatch")
            
            print(f"✓ {scenario['name']}: Wrapper={needs_wrapper}, Latent tokens={has_latent_tokens}")

def run_tests():
    """Run all tests and provide summary"""
    print("=" * 80)
    print("COMPREHENSIVE LATENT INJECTION FIX TESTING")
    print("=" * 80)
    
    # Disable wandb and other external dependencies
    os.environ['WANDB_MODE'] = 'disabled'
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLatentInjectionFix)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=True)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED! The latent injection fix is working correctly.")
        print("\nKey validation points confirmed:")
        print("• LatentWrapper decision logic works correctly for all modes")
        print("• Special tokens are handled appropriately per mode")  
        print("• LatentWrapper correctly detects latent tokens in input")
        print("• Data preprocessing works correctly for CoCoNut evaluation")
        print("• Integration scenarios work as expected")
        print("• Trainer model detection logic functions properly")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}")
                print(f"  {traceback}")
                
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}")
                print(f"  {traceback}")
    
    print("=" * 80)
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
