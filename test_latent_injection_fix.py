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
from pathlib import Path
from typing import Dict, Any, List
import unittest
from unittest.mock import patch, MagicMock, Mock

# Add the project root to path
sys.path.insert(0, os.path.dirname(__file__))

from multicoco.config import MultiCoCoConfig, TrainingMode
from multicoco.constants import COCONUT_SPECIAL_TOKENS
from multicoco.data import SupervisedDataset
from multicoco.latent_wrapper import LatentWrapper
from run import MultiCoCoRunner

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
            },
            {
                "question": "How many legs does a cat have?", 
                "choices": ["0: Two", "1: Four", "2: Six", "3: Eight"],
                "answer": "1",
                "image": "test_image2.jpg"
            }
        ]
        
        self.test_data_path = os.path.join(self.data_dir, "test_data.json")
        with open(self.test_data_path, 'w') as f:
            json.dump(self.test_data, f)
            
        # Create dummy image files
        self.images_dir = os.path.join(self.data_dir, "images", "aokvqa", "validation")
        os.makedirs(self.images_dir, exist_ok=True)
        
        for img_name in ["test_image.jpg", "test_image2.jpg"]:
            img_path = os.path.join(self.images_dir, img_name)
            # Create a minimal dummy image file
            with open(img_path, 'wb') as f:
                f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82')
        
        self.base_config = {
            "project": "test",
            "seed": 42,
            "model_name": "OpenGVLab/InternVL3-1B-Pretrained",
            "data_dir": self.data_dir,
            "eval_data_path": self.test_data_path,
            "log_dir": os.path.join(self.temp_dir, "logs"),
            "use_wandb": False,
            "limit_for_testing": True
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
    
    @patch('multicoco.model.MultiCoCo')
    @patch('multicoco.trainer.CoCoTrainer')
    def test_vanilla_evaluation_no_latent_wrapper(self, mock_trainer, mock_model):
        """Test that vanilla evaluation doesn't use LatentWrapper"""
        print("\n=== Testing Vanilla Evaluation (No LatentWrapper) ===")
        
        # Create vanilla evaluation config
        base_config_path = self.create_base_config_file()
        config_dict = {
            "mode": "eval_only",
            "eval_config": {
                "vanilla": True,
                "coconut": False,
                "cot": False
            },
            "coconut": {
                "enabled": False
            }
        }
        config_path = self.create_config_file(config_dict)
        
        # Mock model and tokenizer
        mock_model_instance = MagicMock()
        mock_model_instance.tokenizer = MagicMock()
        mock_model_instance.image_processor = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Load config and create runner
        config = MultiCoCoConfig.load_with_base(config_path, base_config_path)
        runner = MultiCoCoRunner(config)
        
        # Test model initialization
        runner.initialize_model()
        
        # Verify LatentWrapper was NOT used
        self.assertNotIsInstance(runner.model, LatentWrapper)
        self.assertEqual(runner.model, mock_model_instance)
        print("✓ Vanilla evaluation correctly uses base model without LatentWrapper")
        
    @patch('multicoco.model.MultiCoCo')
    @patch('multicoco.trainer.CoCoTrainer')
    def test_cot_evaluation_no_latent_wrapper(self, mock_trainer, mock_model):
        """Test that CoT evaluation doesn't use LatentWrapper"""
        print("\n=== Testing CoT Evaluation (No LatentWrapper) ===")
        
        # Create CoT evaluation config
        base_config_path = self.create_base_config_file()
        config_dict = {
            "mode": "eval_only",
            "eval_config": {
                "vanilla": False,
                "coconut": False,
                "cot": True
            },
            "coconut": {
                "enabled": False
            }
        }
        config_path = self.create_config_file(config_dict)
        
        # Mock model and tokenizer
        mock_model_instance = MagicMock()
        mock_model_instance.tokenizer = MagicMock()
        mock_model_instance.image_processor = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Load config and create runner
        config = MultiCoCoConfig.load_with_base(config_path, base_config_path)
        runner = MultiCoCoRunner(config)
        
        # Test model initialization
        runner.initialize_model()
        
        # Verify LatentWrapper was NOT used
        self.assertNotIsInstance(runner.model, LatentWrapper)
        self.assertEqual(runner.model, mock_model_instance)
        print("✓ CoT evaluation correctly uses base model without LatentWrapper")
        
    @patch('multicoco.model.MultiCoCo')
    @patch('multicoco.trainer.CoCoTrainer')
    def test_coconut_evaluation_uses_latent_wrapper(self, mock_trainer, mock_model):
        """Test that CoCoNut evaluation uses LatentWrapper"""
        print("\n=== Testing CoCoNut Evaluation (Uses LatentWrapper) ===")
        
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
                "enabled": True,  # This should be ignored for eval-only mode
                "max_latent_stage": 6
            }
        }
        config_path = self.create_config_file(config_dict)
        
        # Mock model and tokenizer
        mock_model_instance = MagicMock()
        mock_model_instance.tokenizer = MagicMock()
        mock_model_instance.image_processor = MagicMock()
        mock_model_instance.get_input_embeddings.return_value = MagicMock()
        
        # Mock tokenizer methods
        mock_tokenizer = mock_model_instance.tokenizer
        mock_tokenizer.convert_tokens_to_ids.side_effect = lambda token: {
            '<|start_latent|>': 1001,
            '<|latent|>': 1002,
            '<|end_latent|>': 1003,
            '<image>': 1004
        }.get(token, None)
        mock_tokenizer.eos_token_id = 2
        
        mock_model.return_value = mock_model_instance
        
        # Load config and create runner
        config = MultiCoCoConfig.load_with_base(config_path, base_config_path)
        runner = MultiCoCoRunner(config)
        
        # Test model initialization
        runner.initialize_model()
        
        # Verify LatentWrapper WAS used
        self.assertIsInstance(runner.model, LatentWrapper)
        self.assertEqual(runner.model.base_model, mock_model_instance)
        print("✓ CoCoNut evaluation correctly uses LatentWrapper")
        
    def test_latent_wrapper_token_detection(self):
        """Test LatentWrapper's latent token detection logic"""
        print("\n=== Testing LatentWrapper Token Detection ===")
        
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
        
        # Mock generate method to avoid complex generation logic in test
        with patch.object(wrapper, 'generate') as mock_generate:
            mock_generate.return_value = Mock()
            mock_tokenizer.decode.return_value = "Response with latent injection"
            
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
        runner = MultiCoCoRunner(config)
        
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
        
    def test_trainer_model_detection_logic(self):
        """Test that trainer correctly detects LatentWrapper vs base model"""
        print("\n=== Testing Trainer Model Detection Logic ===")
        
        # Import here to avoid circular imports in actual code
        from multicoco.trainer import CoCoTrainer
        
        # Test 1: Base model (no LatentWrapper)
        mock_base_model = MagicMock()
        mock_args = MagicMock()
        
        trainer = CoCoTrainer(model=mock_base_model, args=mock_args)
        
        # Test the detection logic from trainer
        from multicoco.latent_wrapper import LatentWrapper
        is_latent_wrapper = isinstance(trainer.model, LatentWrapper)
        self.assertFalse(is_latent_wrapper)
        print("✓ Trainer correctly identifies base model (not LatentWrapper)")
        
        # Test 2: LatentWrapper
        mock_tokenizer = MagicMock()
        wrapped_model = LatentWrapper(mock_base_model, mock_tokenizer)
        trainer_with_wrapper = CoCoTrainer(model=wrapped_model, args=mock_args)
        
        is_latent_wrapper = isinstance(trainer_with_wrapper.model, LatentWrapper)
        self.assertTrue(is_latent_wrapper)
        print("✓ Trainer correctly identifies LatentWrapper")
        
    def test_special_tokens_logic(self):
        """Test special tokens are added correctly based on mode"""
        print("\n=== Testing Special Tokens Logic ===")
        
        # Test vanilla evaluation - no latent tokens
        base_config_path = self.create_base_config_file()
        config_dict = {
            "mode": "eval_only",
            "eval_config": {
                "vanilla": True,
                "coconut": False,
                "cot": False
            },
            "coconut": {
                "enabled": False
            }
        }
        config_path = self.create_config_file(config_dict)
        config = MultiCoCoConfig.load_with_base(config_path, base_config_path)
        
        runner = MultiCoCoRunner(config)
        special_tokens = runner._get_special_tokens(config.coconut, config.training.mode)
        
        # Should not contain latent tokens
        for token in COCONUT_SPECIAL_TOKENS:
            self.assertNotIn(token, special_tokens)
        print("✓ Vanilla evaluation correctly excludes latent tokens")
        
        # Test coconut evaluation - should have latent tokens
        config_dict["eval_config"]["vanilla"] = False
        config_dict["eval_config"]["coconut"] = True
        config_path = self.create_config_file(config_dict)
        config = MultiCoCoConfig.load_with_base(config_path, base_config_path)
        
        runner = MultiCoCoRunner(config)
        special_tokens = runner._get_special_tokens(config.coconut, config.training.mode)
        
        # Should contain latent tokens
        for token in COCONUT_SPECIAL_TOKENS:
            self.assertIn(token, special_tokens)
        print("✓ CoCoNut evaluation correctly includes latent tokens")
        
    def test_integration_scenario(self):
        """Integration test simulating real evaluation scenarios"""
        print("\n=== Testing Integration Scenarios ===")
        
        scenarios = [
            {
                "name": "Vanilla Evaluation of CoCoNut Model",
                "config": {
                    "mode": "eval_only",
                    "eval_config": {"vanilla": True, "coconut": False, "cot": False},
                    "coconut": {"enabled": True}  # Model was trained with coconut
                },
                "expected_wrapper": False,
                "expected_latent_tokens": False
            },
            {
                "name": "CoT Evaluation of CoCoNut Model", 
                "config": {
                    "mode": "eval_only",
                    "eval_config": {"vanilla": False, "coconut": False, "cot": True},
                    "coconut": {"enabled": True}  # Model was trained with coconut
                },
                "expected_wrapper": False,
                "expected_latent_tokens": False
            },
            {
                "name": "CoCoNut Evaluation of CoCoNut Model",
                "config": {
                    "mode": "eval_only", 
                    "eval_config": {"vanilla": False, "coconut": True, "cot": False, "eval_latent_tokens": 4},
                    "coconut": {"enabled": True}
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
            
            runner = MultiCoCoRunner(config)
            
            # Test LatentWrapper decision
            needs_wrapper = runner._needs_latent_wrapper(config.coconut, config.training.mode)
            self.assertEqual(needs_wrapper, scenario['expected_wrapper'])
            
            # Test special tokens
            special_tokens = runner._get_special_tokens(config.coconut, config.training.mode)
            has_latent_tokens = any(token in special_tokens for token in COCONUT_SPECIAL_TOKENS)
            self.assertEqual(has_latent_tokens, scenario['expected_latent_tokens'])
            
            print(f"✓ {scenario['name']}: Wrapper={needs_wrapper}, Latent tokens={has_latent_tokens}")

def run_tests():
    """Run all tests and provide summary"""
    print("=" * 80)
    print("COMPREHENSIVE LATENT INJECTION FIX TESTING")
    print("=" * 80)
    
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
        print("• Vanilla evaluation uses base model directly (no LatentWrapper)")
        print("• CoT evaluation uses base model directly (no LatentWrapper)")
        print("• CoCoNut evaluation uses LatentWrapper with proper latent injection")
        print("• LatentWrapper correctly detects latent tokens in input")
        print("• Data preprocessing works correctly for different modes")
        print("• Special tokens are handled appropriately per mode")
        print("• Integration scenarios work as expected")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
                
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
    
    print("=" * 80)
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
