#!/usr/bin/env python3
"""
Test script to verify that the state_dict fix resolves the shared memory error
while maintaining embedding consistency.
"""

import os
import sys
import tempfile
import torch
import torch.nn as nn
import logging
from pathlib import Path

# Add the parent directory to the path so we can import multicoco as a package
sys.path.insert(0, str(Path(__file__).parent))

from multicoco.latent_wrapper import LatentWrapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockTokenizer:
    """Mock tokenizer for testing purposes."""
    
    def __init__(self):
        self.vocab = {
            '<|latent|>': 1001,
            '<|start_latent|>': 1002,
            '<|end_latent|>': 1003,
            '<IMG_CONTEXT>': 1004,
            '<unk>': 0
        }
        self.unk_token_id = 0
        
    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)


class MockBaseModel(nn.Module):
    """Mock base model for testing purposes."""
    
    def __init__(self, vocab_size=1000, hidden_size=768):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.hidden_size = hidden_size
        
    def get_input_embeddings(self):
        return self.embed_tokens


def test_state_dict_no_shared_memory():
    """Test that state_dict doesn't contain shared memory references."""
    
    print("=" * 50)
    print("Testing state_dict shared memory fix...")
    print("=" * 50)
    
    # Create mock components
    print("Creating mock model and tokenizer...")
    base_model = MockBaseModel()
    tokenizer = MockTokenizer()
    
    # Create LatentWrapper
    print("Creating LatentWrapper...")
    wrapper = LatentWrapper(base_model, tokenizer)
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
    
    # Test 3: Manual shared memory detection
    print("\n3. Testing for shared memory manually...")
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
        raise AssertionError("Shared memory detected in state_dict!")
    else:
        print("✓ No shared memory detected in state_dict")
    
    # Test 4: Test save/load cycle
    print("\n4. Testing save/load cycle...")
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = os.path.join(temp_dir, "test_model.pt")
        
        # Save the state_dict
        try:
            print(f"Saving state_dict to {save_path}...")
            torch.save(wrapper.state_dict(), save_path)
            print("✓ State dict saved successfully")
        except Exception as e:
            print(f"✗ Error saving state_dict: {e}")
            raise
        
        # Load the state_dict
        try:
            print("Loading state_dict...")
            loaded_state_dict = torch.load(save_path)
            
            # Create new wrapper and load state
            new_base_model = MockBaseModel()
            new_wrapper = LatentWrapper(new_base_model, tokenizer)
            new_wrapper.load_state_dict(loaded_state_dict)
            print("✓ State dict loaded successfully")
        except Exception as e:
            print(f"✗ Error loading state_dict: {e}")
            raise
        
        # Test 5: Verify embedding consistency after load
        print("\n5. Testing embedding consistency after load...")
        new_base_embedding = new_wrapper._get_embedding_layer(new_wrapper.base_model)
        assert new_wrapper.embedding is new_base_embedding, "Loaded wrapper should have correct embedding reference"
        print("✓ Loaded wrapper has correct embedding reference")


def test_manual_shared_memory_detection():
    """Manually test for shared memory in state_dict."""
    
    print("\n" + "=" * 50)
    print("Manual shared memory detection test...")
    print("=" * 50)
    
    # Create mock components
    base_model = MockBaseModel()
    tokenizer = MockTokenizer()
    
    # Create LatentWrapper
    wrapper = LatentWrapper(base_model, tokenizer)
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
