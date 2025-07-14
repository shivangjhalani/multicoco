#!/usr/bin/env python3
"""
Test script to verify that the state_dict fix resolves the shared memory error
while maintaining embedding consistency.
"""

import os
import sys
import tempfile
import torch
import logging
from pathlib import Path

# Add the parent directory to the path so we can import multicoco as a package
sys.path.insert(0, str(Path(__file__).parent))

from multicoco.latent_wrapper import LatentWrapper
from multicoco.config import ExperimentConfig, ModelConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_state_dict_no_shared_memory():
    """Test that state_dict doesn't contain shared memory references."""
    
    print("=" * 50)
    print("Testing state_dict shared memory fix...")
    print("=" * 50)
    
    # Create a minimal config
    model_config = ModelConfig(
        model_name="OpenGVLab/InternVL2_5-1B",
        device="cpu",  # Use CPU to avoid GPU memory issues
        dtype="bfloat16"
    )
    config = ExperimentConfig(model=model_config)
    
    # Create LatentWrapper
    print("Creating LatentWrapper...")
    wrapper = LatentWrapper(config)
    wrapper.eval()  # Set to eval mode for consistency
    
    # Test 1: Check that embedding is a reference to base model embedding
    print("\n1. Testing embedding reference consistency...")
    base_embedding = wrapper._get_embedding_layer(wrapper.base_model)
    assert wrapper.embedding is base_embedding, "Embedding should be a reference to base model embedding"
    print("✓ Embedding is correctly referencing base model embedding")
    
    # Test 2: Get state_dict and check for shared memory issues
    print("\n2. Testing state_dict for shared memory...")
    state_dict = wrapper.state_dict()
    
    # Check that 'embedding' is NOT in the state_dict
    embedding_keys = [key for key in state_dict.keys() if 'embedding' in key and not key.startswith('base_model.')]
    print(f"Non-base_model embedding keys in state_dict: {embedding_keys}")
    assert len(embedding_keys) == 0, f"Found unexpected embedding keys: {embedding_keys}"
    print("✓ No duplicate embedding keys found in state_dict")
    
    # Test 3: Check that we can save and load the model without shared memory errors
    print("\n3. Testing save/load cycle...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = os.path.join(temp_dir, "test_model")
        
        # Try to save - this should not raise shared memory errors
        try:
            print(f"Saving model to {save_path}...")
            wrapper.save_pretrained(save_path)
            print("✓ Model saved successfully without shared memory errors")
        except Exception as e:
            print(f"✗ Error saving model: {e}")
            raise
        
        # Create a new wrapper and try to load
        print("Creating new wrapper for loading test...")
        wrapper2 = LatentWrapper(config)
        
        # Load the saved state
        try:
            print("Loading saved model...")
            # Load the base model first
            wrapper2.base_model = wrapper2.base_model.from_pretrained(save_path)
            # Reinitialize embedding reference
            wrapper2.embedding = wrapper2._get_embedding_layer(wrapper2.base_model)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
        
        # Test 4: Verify that loaded model has correct embedding reference
        print("\n4. Testing loaded model embedding consistency...")
        base_embedding2 = wrapper2._get_embedding_layer(wrapper2.base_model)
        assert wrapper2.embedding is base_embedding2, "Loaded model should have correct embedding reference"
        print("✓ Loaded model has correct embedding reference")
        
        # Test 5: Verify embedding weights are the same
        print("\n5. Testing embedding weight consistency...")
        original_weights = wrapper.embedding.weight
        loaded_weights = wrapper2.embedding.weight
        
        # They should have the same values (though different objects due to loading)
        weight_diff = torch.max(torch.abs(original_weights - loaded_weights)).item()
        print(f"Max weight difference: {weight_diff}")
        assert weight_diff < 1e-6, f"Embedding weights differ by {weight_diff}"
        print("✓ Embedding weights are consistent after save/load")

def test_manual_shared_memory_detection():
    """Manually test for shared memory in state_dict."""
    
    print("\n" + "=" * 50)
    print("Manual shared memory detection test...")
    print("=" * 50)
    
    # Create a minimal config
    model_config = ModelConfig(
        model_name="OpenGVLab/InternVL2_5-1B",
        device="cpu",
        dtype="bfloat16"
    )
    config = ExperimentConfig(model=model_config)
    
    # Create LatentWrapper
    wrapper = LatentWrapper(config)
    state_dict = wrapper.state_dict()
    
    # Manually check for shared memory
    print("Checking for shared memory in state_dict...")
    tensor_storage_map = {}
    shared_tensors = []
    
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            storage_ptr = tensor.storage().data_ptr()
            if storage_ptr in tensor_storage_map:
                shared_tensors.append((name, tensor_storage_map[storage_ptr]))
                print(f"SHARED MEMORY DETECTED: {name} shares storage with {tensor_storage_map[storage_ptr]}")
            else:
                tensor_storage_map[storage_ptr] = name
    
    if shared_tensors:
        print(f"✗ Found {len(shared_tensors)} shared memory instances")
        for tensor_pair in shared_tensors:
            print(f"  - {tensor_pair[0]} <-> {tensor_pair[1]}")
        return False
    else:
        print("✓ No shared memory detected in state_dict")
        return True

if __name__ == "__main__":
    print("Testing LatentWrapper state_dict shared memory fix")
    print("This test verifies that the embedding consistency fix doesn't cause shared memory errors")
    
    try:
        # Run the main test
        test_state_dict_no_shared_memory()
        
        # Run manual shared memory detection
        shared_memory_ok = test_manual_shared_memory_detection()
        
        if shared_memory_ok:
            print("\n" + "=" * 50)
            print("🎉 ALL TESTS PASSED! 🎉")
            print("The state_dict fix successfully resolves shared memory issues")
            print("while maintaining embedding consistency.")
            print("=" * 50)
        else:
            print("\n" + "=" * 50)
            print("❌ SHARED MEMORY ISSUES DETECTED")
            print("The fix needs further refinement.")
            print("=" * 50)
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
