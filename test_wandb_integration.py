#!/usr/bin/env python3
"""
Test script for WandB integration with MultiCoCo.

This script verifies that all WandB logging components work correctly
including configuration loading, initialization, metric logging, and utilities.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add multicoco package to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from multicoco.config import MultiCoCoConfig
    from multicoco.utils import (
        log_wandb_samples, 
        log_wandb_compression_ratio,
        log_wandb_multimodal_insights
    )
    print("✓ Successfully imported MultiCoCo components")
except ImportError as e:
    print(f"✗ Failed to import MultiCoCo components: {e}")
    sys.exit(1)

try:
    import wandb
    print("✓ WandB is available")
    WANDB_AVAILABLE = True
except ImportError:
    print("✗ WandB is not available. Install with: pip install wandb")
    WANDB_AVAILABLE = False


def test_config_loading():
    """Test that WandB configuration fields are properly loaded."""
    print("\n--- Testing WandB Configuration Loading ---")
    
    try:
        # Test with base config
        config = MultiCoCoConfig.load_with_base("args/base.yaml")
        
        # Check WandB config fields
        assert hasattr(config.logging, 'wandb_project'), "Missing wandb_project field"
        assert hasattr(config.logging, 'wandb_entity'), "Missing wandb_entity field" 
        assert hasattr(config.logging, 'wandb_tags'), "Missing wandb_tags field"
        assert hasattr(config.logging, 'wandb_group'), "Missing wandb_group field"
        
        print(f"✓ WandB project: {config.logging.wandb_project}")
        print(f"✓ WandB entity: {config.logging.wandb_entity}")
        print(f"✓ WandB tags: {config.logging.wandb_tags}")
        print(f"✓ WandB group: {config.logging.wandb_group}")
        print(f"✓ Use WandB: {config.logging.use_wandb}")
        
        # Test to_dict serialization
        config_dict = config.to_dict()
        assert 'logging' in config_dict, "Missing logging section in config dict"
        assert 'wandb_project' in config_dict['logging'], "Missing wandb_project in serialized config"
        
        print("✓ Configuration serialization works")
        print("✓ WandB configuration loading: PASSED")
        
    except Exception as e:
        print(f"✗ WandB configuration loading: FAILED - {e}")
        return False
    
    return True


def test_wandb_utilities():
    """Test WandB utility functions."""
    print("\n--- Testing WandB Utility Functions ---")
    
    try:
        # Test sample logging utility
        questions = ["What is this?", "How does it work?"]
        labels = ["A", "B"] 
        predictions = ["A", "C"]
        
        # This should not crash even if WandB is not initialized
        log_wandb_samples(questions, labels, predictions, max_samples=2)
        print("✓ log_wandb_samples utility works")
        
        # Test compression ratio logging
        sample_data = [
            {"reasoning": "This is a test reasoning", "steps": ["step1", "step2"]},
            {"reasoning": "Another reasoning example", "steps": ["step1"]}
        ]
        log_wandb_compression_ratio(sample_data, scheduled_stage=1)
        print("✓ log_wandb_compression_ratio utility works")
        
        # Test multimodal insights logging
        model_info = {"parameter_count": 1000000, "model_type": "test"}
        performance_metrics = {"accuracy": 0.85, "loss": 0.15}
        log_wandb_multimodal_insights(model_info, performance_metrics, stage=1)
        print("✓ log_wandb_multimodal_insights utility works")
        
        print("✓ WandB utilities: PASSED")
        
    except Exception as e:
        print(f"✗ WandB utilities: FAILED - {e}")
        return False
    
    return True


def test_wandb_initialization():
    """Test WandB initialization (dry run)."""
    print("\n--- Testing WandB Initialization ---")
    
    if not WANDB_AVAILABLE:
        print("⚠ Skipping WandB initialization test (WandB not installed)")
        return True
    
    try:
        # Test offline mode to avoid requiring WandB login
        os.environ["WANDB_MODE"] = "offline"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["WANDB_DIR"] = temp_dir
            
            # Initialize WandB run
            run = wandb.init(
                project="test-multicoco",
                name="integration-test",
                config={
                    "test": True,
                    "framework": "multicoco"
                },
                mode="offline"
            )
            
            # Test basic logging
            wandb.log({"test_metric": 0.5, "step": 1})
            
            # Test table logging
            table = wandb.Table(columns=["question", "answer", "correct"])
            table.add_data("test question", "test answer", True)
            wandb.log({"test_table": table})
            
            # Finish run
            wandb.finish()
            
        print("✓ WandB initialization and logging: PASSED")
        
    except Exception as e:
        print(f"✗ WandB initialization: FAILED - {e}")
        return False
    finally:
        # Clean up environment
        os.environ.pop("WANDB_MODE", None)
        os.environ.pop("WANDB_DIR", None)
    
    return True


def test_config_compatibility():
    """Test that CoCoNut config works with WandB settings."""
    print("\n--- Testing CoCoNut + WandB Config Compatibility ---")
    
    try:
        # Load CoCoNut config
        config = MultiCoCoConfig.load_with_base("args/aokvqa_coconut.yaml")
        
        # Verify WandB settings are present
        assert config.logging.use_wandb == True, "WandB should be enabled in CoCoNut config"
        assert len(config.logging.wandb_tags) > 0, "CoCoNut config should have WandB tags"
        assert config.logging.wandb_group is not None, "CoCoNut config should have WandB group"
        
        print(f"✓ CoCoNut WandB tags: {config.logging.wandb_tags}")
        print(f"✓ CoCoNut WandB group: {config.logging.wandb_group}")
        
        # Verify CoCoNut parameters are present
        assert config.coconut.enabled == True, "CoCoNut should be enabled"
        assert config.coconut.c_thought >= 1, "c_thought should be >= 1"
        assert config.coconut.max_latent_stage >= 1, "max_latent_stage should be >= 1"
        
        print("✓ CoCoNut + WandB compatibility: PASSED")
        
    except Exception as e:
        print(f"✗ CoCoNut + WandB compatibility: FAILED - {e}")
        return False
    
    return True


def main():
    """Run all WandB integration tests."""
    print("=" * 60)
    print("MultiCoCo WandB Integration Test Suite")
    print("=" * 60)
    
    tests = [
        test_config_loading,
        test_wandb_utilities, 
        test_config_compatibility,
        test_wandb_initialization,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "PASSED" if result else "FAILED"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All WandB integration tests PASSED!")
        print("\nNext steps:")
        print("1. Run a full training with: python run.py --config args/aokvqa_coconut.yaml")
        print("2. Check WandB dashboard for logged metrics and artifacts")
        print("3. Try hyperparameter sweeps with: wandb sweep sweep_simple.yaml")
        return 0
    else:
        print("❌ Some tests FAILED. Please fix issues before using WandB integration.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 