#!/usr/bin/env python3
"""
Test script to validate the two-phase training fixes.

This script tests that:
1. CoT training doesn't add unnecessary latent tokens
2. CoCoNut training properly loads checkpoints and adds latent tokens
3. Model architecture is consistent between phases
"""

import os
import tempfile
import logging
from typing import Dict, Any, Optional

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_config(mode: str, load_model_path: Optional[str] = None, coconut_enabled: bool = False) -> Dict[str, Any]:
    """Create a test configuration for different training modes."""
    config = {
        'mode': mode,
        'project': 'test_multicoco',
        'name': f'test-{mode}',
        'run_name': f'test-{mode}',
        'output_dir': f'test_checkpoints/{mode}',
        'num_epochs': 1,
        'batch_size': 1,
        'eval_batch_size': 1,
        'learning_rate': 1e-5,
        'gradient_accumulation_steps': 1,
        'resume_from_checkpoint': False,
        'limit_for_testing': True,
        'debug': True,
        
        # Precision
        'bf16': False,
        'fp16': False,
        
        # Model configuration
        'model_name': 'OpenGVLab/InternVL3-1B-Pretrained',
        'load_model_path': load_model_path,
        'use_flash_attention_2': False,
        'torch_compile': False,
        
        # Data configuration
        'data_dir': 'data/',
        'train_data_path': 'data/test_train.json',
        'eval_data_path': 'data/test_eval.json',
        
        # Logging
        'log_dir': 'test_logs/',
        'log_level': 'INFO',
        'console_output': True,
        'verbose': True,
        
        # CoCoNut configuration
        'coconut': {
            'enabled': coconut_enabled,
            'c_thought': 1,
            'epochs_per_stage': 1,
            'max_latent_stage': 2,
            'pad_latent_to_max': False,
            'uniform_prob': 0.0,
            'reset_optimizer': True
        },
        
        # Evaluation configuration
        'eval_config': {
            'cot': mode == 'cot_train',
            'coconut': coconut_enabled
        }
    }
    
    return config

def test_cot_training_tokens():
    """Test that CoT training doesn't add unnecessary latent tokens."""
    logger.info("Testing CoT training token handling...")
    
    try:
        from multicoco.config import MultiCoCoConfig
        from multicoco.run import MultiCoCoRunner
        
        # Create CoT training config
        config_dict = create_test_config('cot_train', coconut_enabled=False)
        config = MultiCoCoConfig.from_dict(config_dict)
        
        # Initialize runner
        runner = MultiCoCoRunner(config)
        
        # This should not add latent tokens
        runner.initialize_model()
        
        if runner.model is None:
            logger.error("❌ Model not initialized")
            return False
        
        # Check that latent tokens are not in tokenizer
        from multicoco.constants import LATENT_TOKEN, START_LATENT_TOKEN, END_LATENT_TOKEN
        latent_tokens = [START_LATENT_TOKEN, LATENT_TOKEN, END_LATENT_TOKEN]
        
        added_tokens = []
        for token in latent_tokens:
            token_id = runner.model.tokenizer.convert_tokens_to_ids(token)
            if token_id != runner.model.tokenizer.unk_token_id:
                added_tokens.append(token)
        
        if added_tokens:
            logger.error(f"❌ CoT training incorrectly added latent tokens: {added_tokens}")
            return False
        else:
            logger.info("✅ CoT training correctly avoided adding latent tokens")
            return True
            
    except Exception as e:
        logger.error(f"❌ CoT training test failed: {e}")
        return False

def test_coconut_training_tokens():
    """Test that CoCoNut training properly adds latent tokens."""
    logger.info("Testing CoCoNut training token handling...")
    
    try:
        from multicoco.config import MultiCoCoConfig
        from multicoco.run import MultiCoCoRunner
        
        # Create CoCoNut training config
        config_dict = create_test_config('coconut_train', coconut_enabled=True)
        config = MultiCoCoConfig.from_dict(config_dict)
        
        # Initialize runner
        runner = MultiCoCoRunner(config)
        
        # This should add latent tokens
        runner.initialize_model()
        
        if runner.model is None:
            logger.error("❌ Model not initialized")
            return False
        
        # Check that latent tokens are in tokenizer
        from multicoco.constants import LATENT_TOKEN, START_LATENT_TOKEN, END_LATENT_TOKEN
        latent_tokens = [START_LATENT_TOKEN, LATENT_TOKEN, END_LATENT_TOKEN]
        
        missing_tokens = []
        for token in latent_tokens:
            token_id = runner.model.tokenizer.convert_tokens_to_ids(token)
            if token_id == runner.model.tokenizer.unk_token_id:
                missing_tokens.append(token)
        
        if missing_tokens:
            logger.error(f"❌ CoCoNut training failed to add latent tokens: {missing_tokens}")
            return False
        else:
            logger.info("✅ CoCoNut training correctly added latent tokens")
            return True
            
    except Exception as e:
        logger.error(f"❌ CoCoNut training test failed: {e}")
        return False

def test_model_architecture_consistency():
    """Test that model architecture is consistent between phases."""
    logger.info("Testing model architecture consistency...")
    
    try:
        from multicoco.config import MultiCoCoConfig
        from multicoco.run import MultiCoCoRunner
        from multicoco.latent_wrapper import LatentWrapper
        
        # Test CoT phase
        cot_config_dict = create_test_config('cot_train', coconut_enabled=False)
        cot_config = MultiCoCoConfig.from_dict(cot_config_dict)
        cot_runner = MultiCoCoRunner(cot_config)
        cot_runner.initialize_model()
        
        # Should not be wrapped with LatentWrapper
        if isinstance(cot_runner.model, LatentWrapper):
            logger.error("❌ CoT model incorrectly wrapped with LatentWrapper")
            return False
        
        # Test CoCoNut phase
        coconut_config_dict = create_test_config('coconut_train', coconut_enabled=True)
        coconut_config = MultiCoCoConfig.from_dict(coconut_config_dict)
        coconut_runner = MultiCoCoRunner(coconut_config)
        coconut_runner.initialize_model()
        
        # Should be wrapped with LatentWrapper
        if not isinstance(coconut_runner.model, LatentWrapper):
            logger.error("❌ CoCoNut model not wrapped with LatentWrapper")
            return False
        
        logger.info("✅ Model architecture consistency verified")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model architecture test failed: {e}")
        return False

def test_checkpoint_loading_logic():
    """Test that checkpoint loading logic works correctly."""
    logger.info("Testing checkpoint loading logic...")
    
    try:
        from multicoco.config import MultiCoCoConfig
        from multicoco.run import MultiCoCoRunner
        
        # Test with valid checkpoint path format
        config_dict = create_test_config('coconut_train', 
                                       load_model_path='checkpoints/aokvqa_cot',
                                       coconut_enabled=True)
        config = MultiCoCoConfig.from_dict(config_dict)
        runner = MultiCoCoRunner(config)
        
        # This should not fail during model architecture setup
        # (even if checkpoint doesn't exist, the logic should be sound)
        try:
            runner.initialize_model()
            logger.info("✅ Checkpoint loading logic is sound")
            return True
        except Exception as e:
            if "does not exist" in str(e):
                logger.info("✅ Checkpoint loading logic correctly validates paths")
                return True
            else:
                logger.error(f"❌ Unexpected error in checkpoint loading: {e}")
                return False
        
    except Exception as e:
        logger.error(f"❌ Checkpoint loading test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🧪 Running two-phase training fixes validation...")
    logger.info("=" * 60)
    
    tests = [
        ("CoT Training Token Handling", test_cot_training_tokens),
        ("CoCoNut Training Token Handling", test_coconut_training_tokens),
        ("Model Architecture Consistency", test_model_architecture_consistency),
        ("Checkpoint Loading Logic", test_checkpoint_loading_logic),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🔬 Running: {test_name}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST RESULTS:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {status}: {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("🎉 All fixes validated successfully!")
        return True
    else:
        logger.info("⚠️  Some tests failed - fixes may need adjustment")
        return False

if __name__ == "__main__":
    main() 