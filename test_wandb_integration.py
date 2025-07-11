#!/usr/bin/env python3
"""
Test script for WandB integration in MultiCoCo.

This script performs basic validation of the WandB integration without 
running a full training loop.
"""

import os
import sys
import tempfile
from typing import Dict, Any

# Add the multicoco module to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import wandb
    from multicoco.config import MultiCoCoConfig
    from multicoco.utils import log_wandb_samples
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_config_loading():
    """Test that WandB configuration loads properly."""
    print("\n--- Testing Configuration Loading ---")
    
    try:
        # Test loading base config
        config = MultiCoCoConfig.load_with_base("args/aokvqa_coconut.yaml", "args/base.yaml")
        
        # Check WandB config
        assert hasattr(config.logging, 'use_wandb'), "Missing use_wandb field"
        assert hasattr(config.logging, 'wandb_project'), "Missing wandb_project field"
        assert hasattr(config.logging, 'wandb_tags'), "Missing wandb_tags field"
        assert hasattr(config.logging, 'wandb_group'), "Missing wandb_group field"
        
        print(f"✓ WandB enabled: {config.logging.use_wandb}")
        print(f"✓ WandB project: {config.logging.wandb_project}")
        print(f"✓ WandB tags: {config.logging.wandb_tags}")
        print(f"✓ WandB group: {config.logging.wandb_group}")
        
        # Test to_dict method
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict), "to_dict() should return a dictionary"
        assert 'logging' in config_dict, "to_dict() should contain logging config"
        
        print("✓ Configuration loading successful")
        return config
        
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return None


def test_wandb_utils():
    """Test the WandB utility functions."""
    print("\n--- Testing WandB Utilities ---")
    
    try:
        # Test log_wandb_samples function with dummy data
        questions = ["What is the color of the sky?", "How many legs does a cat have?"]
        labels = ["blue", "four"]
        predictions = ["blue", "4"]
        
        # This would normally log to WandB, but since we're not initializing WandB,
        # it should return gracefully
        log_wandb_samples(questions, labels, predictions, max_samples=2)
        print("✓ log_wandb_samples function works correctly (no WandB run)")
        
        return True
        
    except Exception as e:
        print(f"✗ WandB utilities test failed: {e}")
        return False


def test_wandb_report_to():
    """Test the get_wandb_report_to method."""
    print("\n--- Testing WandB Report Configuration ---")
    
    try:
        # Create a config with WandB enabled
        config_dict = {"use_wandb": True}
        config = MultiCoCoConfig.from_dict(config_dict)
        
        report_to = config.get_wandb_report_to()
        assert report_to == ["wandb"], f"Expected ['wandb'], got {report_to}"
        print("✓ WandB report_to works when enabled")
        
        # Test with WandB disabled
        config_dict = {"use_wandb": False}
        config = MultiCoCoConfig.from_dict(config_dict)
        
        report_to = config.get_wandb_report_to()
        assert report_to == [], f"Expected [], got {report_to}"
        print("✓ WandB report_to works when disabled")
        
        return True
        
    except Exception as e:
        print(f"✗ WandB report_to test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=== MultiCoCo WandB Integration Test ===")
    
    # Change to multicoco directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    success = True
    
    # Test configuration loading
    config = test_config_loading()
    if config is None:
        success = False
    
    # Test WandB utilities
    if not test_wandb_utils():
        success = False
    
    # Test WandB report configuration
    if not test_wandb_report_to():
        success = False
    
    print("\n=== Test Summary ===")
    if success:
        print("✓ All tests passed! WandB integration is working correctly.")
        print("\nNext steps:")
        print("1. Run 'wandb login' to authenticate")
        print("2. Start training with: python run.py args/aokvqa_coconut.yaml")
        print("3. Check your WandB dashboard for logged metrics and artifacts")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main() 