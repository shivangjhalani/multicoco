#!/usr/bin/env python3

import logging
import sys
from multicoco.config import MultiCoCoConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_config_loading():
    """Test configuration loading and validation"""
    try:
        logger.info("Loading aokvqa_coconut.yaml configuration...")
        config = MultiCoCoConfig.load_with_base('args/aokvqa_coconut.yaml')
        
        logger.info("Configuration loaded successfully!")
        logger.info(f"Training mode: {config.training.mode}")
        logger.info(f"CoCoNut enabled: {config.coconut.enabled}")
        logger.info(f"Evaluation coconut: {config.evaluation.coconut}")
        logger.info(f"Evaluation cot: {config.evaluation.cot}")
        logger.info(f"Evaluation vanilla: {config.evaluation.vanilla}")
        logger.info(f"Train data path: {config.data.train_data_path}")
        logger.info(f"Eval data path: {config.data.eval_data_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config_loading()
    sys.exit(0 if success else 1)
