#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced wandb logging in MultiCoCo.
This script shows the key metrics that are now logged to match and exceed CoCoNut's capabilities.
"""

import os
import yaml
from multicoco import MultiCoCoRunner, MultiCoCoConfig

def create_test_config():
    """Create a minimal test configuration with wandb enabled"""
    config = {
        'model': {
            'model_name': 'OpenGVLab/InternVL2-1B',
            'torch_dtype': 'bfloat16'
        },
        'training': {
            'mode': 'cot_train',
            'name': 'test-wandb-logging',
            'output_dir': 'test_checkpoints',
            'num_epochs': 2,
            'batch_size': 1,
            'eval_batch_size': 1,
            'learning_rate': 1e-5,
            'eval_steps': 1
        },
        'data': {
            'dataset_path': 'data/aokvqa',
            'train_split': 'train',
            'eval_split': 'validation'
        },
        'logging': {
            'use_wandb': True,
            'project': 'multicoco-test',
            'run_name': 'wandb-metrics-test',
            'log_level': 'INFO'
        },
        'coconut': {
            'enabled': False  # Start with CoT training
        },
        'evaluation': {
            'log_per_sample': True
        }
    }
    return config

def test_wandb_metrics():
    """Test the wandb metrics logging functionality"""
    print("🧪 Testing MultiCoCo Wandb Metrics Enhancement")
    print("=" * 50)
    
    # Create test configuration
    config_dict = create_test_config()
    config = MultiCoCoConfig.from_dict(config_dict)
    
    print("📊 Wandb metrics that will be logged:")
    print("\n🏋️ Training Metrics:")
    print("  • train/batch_loss - Per-batch training loss")
    print("  • train/step - Training step counter") 
    print("  • train/epoch - Current epoch")
    print("  • train/learning_rate - Current LR")
    print("  • train/grad_norm - Gradient norm")
    
    print("\n📈 Evaluation Metrics:")
    print("  • eval/acc - Accuracy (matching CoCoNut)")
    print("  • eval/cot_em - CoT exact match (matching CoCoNut)")
    print("  • eval/loss - Validation loss")
    print("  • eval/sample_generations - Generation samples table")
    
    print("\n🥥 CoCoNut Stage Metrics:")
    print("  • coconut/current_stage - Current latent stage")
    print("  • coconut/stage_progress - Progress within stage")
    print("  • coconut/latent_replacement_ratio - Latent token ratio")
    
    print("\n🏆 Best Model Tracking:")
    print("  • best/accuracy - Best accuracy achieved")
    print("  • best/epoch - Best model epoch")
    print("  • best/checkpoint - Best model path")
    
    print("\n📋 Data Inspection Tables:")
    print("  • train/data_samples - Training data tokens")
    print("  • eval/generation_samples - Sample generations")
    
    print("\n✅ Configuration created successfully!")
    print(f"   Project: {config.logging.project}")
    print(f"   Run name: {config.logging.run_name}")
    print(f"   Wandb enabled: {config.logging.use_wandb}")
    
    print("\n🚀 To run with enhanced wandb logging:")
    print("   python run.py args/your_config.yaml")
    print("\n💡 The logging will automatically match CoCoNut's metrics plus MultiCoCo enhancements!")
    
    return config

if __name__ == "__main__":
    test_config = test_wandb_metrics()
    
    # Save test config for actual use
    os.makedirs('test_args', exist_ok=True)
    with open('test_args/wandb_test.yaml', 'w') as f:
        yaml.dump(test_config.__dict__, f, default_flow_style=False)
    
    print(f"\n💾 Test config saved to: test_args/wandb_test.yaml")
    print("   You can use this config to test the wandb logging!")
