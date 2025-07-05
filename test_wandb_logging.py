#!/usr/bin/env python3
"""
Test script to validate wandb logging implementation in MultiCoCo.
This script tests the wandb initialization and logging without running full training.
"""

import sys
import yaml
import wandb
from copy import copy

def test_wandb_init():
    """Test wandb initialization with sample config."""
    print("Testing wandb initialization...")
    
    # Sample config similar to multicoco configs
    test_config = {
        'project': 'multicoco-test',
        'name': 'test-run',
        'debug': False,
        'only_eval': False,
        'batch_size_training': 8,
        'lr': 5e-5,
        'weight_decay': 0.01
    }
    
    try:
        # Initialize wandb (similar to run.py)
        wandb_run = wandb.init(
            project=test_config.get('project', 'multicoco'),
            name=test_config.get('name', 'default-run'),
            config=test_config,
            mode='offline'  # Use offline mode for testing
        )
        
        # Create text table for logging training data
        text_table = wandb.Table(columns=["step", "text"])
        
        print("✓ Wandb initialization successful")
        
        # Test logging training metrics
        log_dict = {
            "train/stage": 0,
            "train/epoch": 1,
            "train/step": 1,
            "train/loss": 0.5,
        }
        wandb_run.log(log_dict)
        print("✓ Training metrics logging successful")
        
        # Test logging validation metrics
        eval_dict = {
            "eval/stage": 0,
            "eval/epoch": 1,
            "eval/acc": 0.75,
            "eval/loss": 0.3,
        }
        wandb_run.log(eval_dict)
        print("✓ Validation metrics logging successful")
        
        # Test logging training data
        text_table.add_data(1, "Sample training data log")
        wandb_run.log({"data_table": copy(text_table)})
        print("✓ Training data logging successful")
        
        wandb_run.finish()
        print("✓ Wandb run finished successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Wandb test failed: {e}")
        return False

def test_config_loading():
    """Test loading config files with wandb parameters."""
    print("\nTesting config file loading...")
    
    config_files = [
        'args/aokvqa_cot.yaml',
        'args/aokvqa_coconut.yaml'
    ]
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check for required wandb parameters
            required_params = ['project', 'name', 'debug']
            missing_params = [param for param in required_params if param not in config]
            
            if missing_params:
                print(f"✗ {config_file}: Missing parameters: {missing_params}")
                return False
            else:
                print(f"✓ {config_file}: All wandb parameters present")
                
        except FileNotFoundError:
            print(f"✗ {config_file}: File not found")
            return False
        except Exception as e:
            print(f"✗ {config_file}: Error loading: {e}")
            return False
    
    return True

def main():
    """Run all tests."""
    print("MultiCoCo Wandb Logging Test Suite")
    print("=" * 40)
    
    # Test 1: Config file loading
    config_test_passed = test_config_loading()
    
    # Test 2: Wandb initialization and logging
    wandb_test_passed = test_wandb_init()
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Summary:")
    print(f"Config loading: {'PASS' if config_test_passed else 'FAIL'}")
    print(f"Wandb logging: {'PASS' if wandb_test_passed else 'FAIL'}")
    
    if config_test_passed and wandb_test_passed:
        print("\n✓ All tests passed! Wandb logging is ready to use.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 