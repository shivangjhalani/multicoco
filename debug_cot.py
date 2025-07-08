#!/usr/bin/env python3
"""
Debug script to isolate the CoT training issue.
"""

import sys
sys.path.append('.')

import torch
from multicoco.config import load_config_from_yaml
from multicoco.data import SupervisedDataset, collate_fn
from multicoco.model import MultiCoCo
from multicoco.constants import COCONUT_SPECIAL_TOKENS
from torch.utils.data import DataLoader

def test_data_loading():
    """Test if data loading works correctly."""
    print("🔍 Testing data loading...")
    
    try:
        # Load configuration
        config = load_config_from_yaml('args/aokvqa_cot.yaml')
        print(f"✓ Config loaded: CoT={config.evaluation.cot}")
        
        # Load model and processor
        model = MultiCoCo(
            model_id=config.model.model_name,
            special_tokens=COCONUT_SPECIAL_TOKENS if config.coconut.enabled else [],
            torch_dtype=config.model.torch_dtype,
            trust_remote_code=config.model.trust_remote_code,
            low_cpu_mem_usage=config.model.low_cpu_mem_usage
        )
        processor = model.tokenizer  # Use the tokenizer as processor
        print(f"✓ Model loaded: {config.model.model_name}")
        
        # Create dataset
        dataset = SupervisedDataset(
            data_path=config.data.train_data_path,
            data_dir=config.data.data_dir,
            test_limit=2  # Only test with 2 samples
        )
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Test single sample
        sample = dataset[0]
        print(f"✓ Single sample keys: {list(sample.keys())}")
        print(f"  - Image type: {type(sample['image'])}")
        print(f"  - Question: {sample['question'][:50]}...")
        print(f"  - Answer: {sample['answer'][:50]}...")
        print(f"  - Has steps: {'steps' in sample}")
        
        # Test collate function
        print("\n🔍 Testing collate function...")
        batch = [dataset[0], dataset[1]]
        
        # Create collate wrapper
        def collate_wrapper(batch):
            return collate_fn(batch, model.tokenizer, model.image_processor)
        
        collated = collate_wrapper(batch)
        print(f"✓ Collation successful")
        print(f"  - Batch keys: {list(collated.keys())}")
        print(f"  - Pixel values shape: {collated['pixel_values'].shape}")
        print(f"  - Input IDs shape: {collated['input_ids'].shape}")
        print(f"  - Labels shape: {collated['labels'].shape}")
        
        return model, processor, collated
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_model_forward(model, collated):
    """Test model forward pass."""
    print("\n🔍 Testing model forward pass...")
    
    try:
        # Move to device
        device = next(model.parameters()).device
        print(f"✓ Model device: {device}")
        
        # Move batch to device
        batch = {}
        for key, value in collated.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
            else:
                batch[key] = value
        
        print(f"✓ Batch moved to device")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
        
        print(f"✓ Forward pass successful")
        print(f"  - Loss: {outputs.loss if hasattr(outputs, 'loss') else 'No loss'}")
        print(f"  - Output keys: {list(outputs.keys()) if hasattr(outputs, 'keys') else 'No keys'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model forward failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debugging function."""
    print("🚀 Starting CoT debugging...\n")
    
    # Test data loading
    model, processor, collated = test_data_loading()
    if model is None:
        print("❌ Cannot proceed without successful data loading")
        return
    
    # Test model forward
    forward_success = test_model_forward(model, collated)
    if not forward_success:
        print("❌ Model forward pass failed")
        return
    
    print("\n✅ All tests passed! The issue might be in the training loop itself.")

if __name__ == "__main__":
    main() 