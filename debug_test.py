#!/usr/bin/env python3
"""
Simple diagnostic test to identify where the main test is hanging.
"""

import sys
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test basic Python imports."""
    try:
        logger.info("Testing basic imports...")
        import os
        import tempfile
        from typing import Dict, Any, Optional
        logger.info("✅ Basic imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Basic imports failed: {e}")
        return False

def test_torch_import():
    """Test PyTorch import."""
    try:
        logger.info("Testing PyTorch import...")
        import torch
        logger.info(f"✅ PyTorch import successful - version: {torch.__version__}")
        logger.info(f"✅ CUDA available: {torch.cuda.is_available()}")
        return True
    except Exception as e:
        logger.error(f"❌ PyTorch import failed: {e}")
        return False

def test_transformers_import():
    """Test transformers import."""
    try:
        logger.info("Testing transformers import...")
        import transformers
        logger.info(f"✅ Transformers import successful - version: {transformers.__version__}")
        return True
    except Exception as e:
        logger.error(f"❌ Transformers import failed: {e}")
        return False

def test_multicoco_imports():
    """Test MultiCoCo package imports."""
    try:
        logger.info("Testing MultiCoCo imports...")
        
        logger.info("  Importing config...")
        from multicoco.config import MultiCoCoConfig, TrainingMode
        logger.info("  ✅ Config imported")
        
        logger.info("  Importing constants...")
        from multicoco.constants import LATENT_TOKEN, START_LATENT_TOKEN, END_LATENT_TOKEN
        logger.info("  ✅ Constants imported")
        
        logger.info("  Importing exceptions...")
        from multicoco.exceptions import ModelInitializationError
        logger.info("  ✅ Exceptions imported")
        
        # This might be the problematic one
        logger.info("  Importing model (this might take time or fail)...")
        from multicoco.model import MultiCoCo
        logger.info("  ✅ Model imported")
        
        logger.info("  Importing runner...")
        from multicoco.run import MultiCoCoRunner
        logger.info("  ✅ Runner imported")
        
        logger.info("✅ All MultiCoCo imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ MultiCoCo imports failed: {e}")
        return False

def test_config_creation():
    """Test configuration creation."""
    try:
        logger.info("Testing configuration creation...")
        from multicoco.config import MultiCoCoConfig
        
        config_dict = {
            'mode': 'cot_train',
            'project': 'test',
            'name': 'test',
            'run_name': 'test',
            'output_dir': 'test_output',
            'num_epochs': 1,
            'batch_size': 1,
            'eval_batch_size': 1,
            'learning_rate': 1e-5,
            'model_name': 'OpenGVLab/InternVL3-1B-Pretrained',
            'data_dir': 'data/',
            'log_dir': 'logs/',
            'log_level': 'INFO',
            'console_output': True,
            'verbose': False,
            'coconut': {'enabled': False},
            'eval_config': {'cot': True, 'coconut': False}
        }
        
        config = MultiCoCoConfig.from_dict(config_dict)
        logger.info("✅ Configuration creation successful")
        return True
    except Exception as e:
        logger.error(f"❌ Configuration creation failed: {e}")
        return False

def test_runner_creation():
    """Test runner creation without model initialization."""
    try:
        logger.info("Testing runner creation...")
        from multicoco.config import MultiCoCoConfig
        from multicoco.run import MultiCoCoRunner
        
        config_dict = {
            'mode': 'cot_train',
            'project': 'test',
            'name': 'test',
            'run_name': 'test',
            'output_dir': 'test_output',
            'num_epochs': 1,
            'batch_size': 1,
            'eval_batch_size': 1,
            'learning_rate': 1e-5,
            'model_name': 'OpenGVLab/InternVL3-1B-Pretrained',
            'data_dir': 'data/',
            'log_dir': 'logs/',
            'log_level': 'INFO',
            'console_output': True,
            'verbose': False,
            'coconut': {'enabled': False},
            'eval_config': {'cot': True, 'coconut': False}
        }
        
        config = MultiCoCoConfig.from_dict(config_dict)
        runner = MultiCoCoRunner(config)
        logger.info("✅ Runner creation successful")
        return True
    except Exception as e:
        logger.error(f"❌ Runner creation failed: {e}")
        return False

def main():
    """Run diagnostic tests."""
    logger.info("🔍 Running diagnostic tests...")
    logger.info("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("PyTorch Import", test_torch_import),
        ("Transformers Import", test_transformers_import),
        ("MultiCoCo Imports", test_multicoco_imports),
        ("Config Creation", test_config_creation),
        ("Runner Creation", test_runner_creation),
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            result = test_func()
            if not result:
                logger.error(f"❌ {test_name} failed - stopping here")
                break
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            break
        logger.info(f"✅ {test_name} completed")
    
    logger.info("\n" + "=" * 50)
    logger.info("🏁 Diagnostic complete")

if __name__ == "__main__":
    main() 