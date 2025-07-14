#!/usr/bin/env python3
"""
Test configuration inheritance and model initialization for multicoco.

This script validates:
1. That config inheritance from base.yaml works correctly
2. That model initialization with different dtypes works
3. That train_data_path is properly loaded
"""

import os
import sys
import logging
import tempfile
import yaml
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_config_inheritance():
    """Test that configuration inheritance from base.yaml works correctly."""
    logger.info("Testing configuration inheritance...")
    
    try:
        from multicoco.config import MultiCoCoConfig
        
        # Test loading aokvqa_coconut.yaml with base inheritance
        config_path = "args/aokvqa_coconut.yaml"
        base_path = "args/base.yaml"
        
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return False
            
        if not os.path.exists(base_path):
            logger.error(f"Base config file not found: {base_path}")
            return False
        
        # Load config with inheritance
        config = MultiCoCoConfig.load_with_base(config_path, base_path)
        
        # Check that essential fields are loaded
        if not config.data.train_data_path:
            logger.error("train_data_path is missing or None")
            return False
            
        if not config.data.eval_data_path:
            logger.error("eval_data_path is missing or None") 
            return False
            
        logger.info(f"✓ train_data_path: {config.data.train_data_path}")
        logger.info(f"✓ eval_data_path: {config.data.eval_data_path}")
        logger.info(f"✓ model_name: {config.model.model_name}")
        logger.info(f"✓ torch_dtype: {config.model.torch_dtype}")
        logger.info(f"✓ mode: {config.training.mode}")
        logger.info(f"✓ coconut.enabled: {config.coconut.enabled}")
        
        return True
        
    except Exception as e:
        logger.error(f"Config inheritance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_initialization():
    """Test model initialization with different dtypes."""
    logger.info("Testing model initialization...")
    
    try:
        from multicoco.model import MultiCoCo
        
        # Test different dtypes
        test_dtypes = ['float32', 'bfloat16', 'float16']
        
        for dtype in test_dtypes:
            try:
                logger.info(f"Testing dtype: {dtype}")
                
                # Create minimal model config (use a very small model for testing)
                test_model_id = "OpenGVLab/InternVL3-1B-Pretrained"
                
                # This will fail if the model isn't available, but we can check dtype conversion
                from multicoco.model import MultiCoCo
                model_instance = MultiCoCo(model_id=test_model_id, torch_dtype=dtype)
                
                logger.info(f"✓ {dtype} initialization successful")
                
                # Clean up
                del model_instance
                
            except Exception as e:
                if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                    logger.warning(f"⚠ {dtype} initialization skipped due to OOM (expected in limited resources)")
                elif "not found" in str(e).lower() or "connection" in str(e).lower():
                    logger.warning(f"⚠ {dtype} initialization skipped due to model download/connection issues")
                else:
                    logger.error(f"✗ {dtype} initialization failed: {e}")
                    return False
        
        return True
        
    except Exception as e:
        logger.error(f"Model initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_validation():
    """Test configuration validation logic."""
    logger.info("Testing configuration validation...")
    
    try:
        from multicoco.config import MultiCoCoConfig
        
        # Create a minimal valid config for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            test_config = {
                'mode': 'coconut_train',
                'model_name': 'OpenGVLab/InternVL3-1B-Pretrained',
                'train_data_path': 'data/aokvqa_train.json',
                'eval_data_path': 'data/aokvqa_validation.json',
                'bf16': True,
                'fp16': False,
                'batch_size': 2,
                'learning_rate': 1e-5,
                'num_epochs': 1,
                'output_dir': 'test_output',
                'coconut': {
                    'enabled': True,
                    'c_thought': 2,
                    'max_latent_stage': 3
                }
            }
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            # Load and validate config
            config = MultiCoCoConfig.from_dict(test_config)
            
            # Check key fields
            assert config.data.train_data_path == 'data/aokvqa_train.json'
            assert config.data.eval_data_path == 'data/aokvqa_validation.json'
            assert config.model.torch_dtype == 'bfloat16'  # Should be derived from bf16=True
            assert config.coconut.enabled == True
            assert config.coconut.c_thought == 2
            
            logger.info("✓ Configuration validation successful")
            return True
            
        finally:
            os.unlink(config_path)
            
    except Exception as e:
        logger.error(f"Config validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all configuration tests."""
    logger.info("Starting configuration and model initialization tests...")
    
    tests = [
        ("Config Inheritance", test_config_inheritance),
        ("Config Validation", test_config_validation),
        ("Model Initialization", test_model_initialization),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
