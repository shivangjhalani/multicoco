#!/usr/bin/env python3

import logging
import sys
from multicoco.config import MultiCoCoConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_config_inheritance():
    """Test that train_data_path is properly inherited from base.yaml"""
    try:
        logger.info("Testing configuration inheritance...")
        
        # Load config using inheritance
        config = MultiCoCoConfig.load_with_base('args/aokvqa_coconut.yaml')
        
        logger.info(f"✓ train_data_path: {config.data.train_data_path}")
        logger.info(f"✓ eval_data_path: {config.data.eval_data_path}")
        logger.info(f"✓ model_name: {config.model.model_name}")
        logger.info(f"✓ torch_dtype: {config.model.torch_dtype}")
        logger.info(f"✓ mode: {config.training.mode}")
        logger.info(f"✓ coconut.enabled: {config.coconut.enabled}")
        logger.info(f"✓ evaluation.coconut: {config.evaluation.coconut}")
        
        # Verify inheritance worked
        assert config.data.train_data_path is not None, "train_data_path should be inherited"
        assert "aokvqa_train.json" in config.data.train_data_path, "Should inherit train_data_path from base.yaml"
        
        logger.info("✅ Configuration inheritance working correctly!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration inheritance failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config_inheritance()
    sys.exit(0 if success else 1)
