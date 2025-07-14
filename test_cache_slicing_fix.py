#!/usr/bin/env python3
"""
Test script to verify that the KV cache slicing fix resolves the 
"past_key_values should be either a Cache object or None" error.
"""

import sys
import os
import torch
import logging

# Add the current directory to Python path
sys.path.append('/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_cache_slicing():
    """Test the new cache slicing implementation"""
    print("Testing KV cache slicing with new transformers API...")
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        
        # Test if Cache classes are available
        try:
            from transformers.cache_utils import DynamicCache, Cache
            print("✓ DynamicCache and Cache classes are available")
            has_cache_utils = True
        except ImportError:
            print("! DynamicCache and Cache classes not available (older transformers)")
            has_cache_utils = False
        
        # Create a mock LatentWrapper to test cache slicing
        class MockModel:
            def parameters(self):
                return [torch.tensor([1.0])]
                
            def get_input_embeddings(self):
                # Return a simple embedding layer
                return torch.nn.Embedding(1000, 64)
        
        mock_model = MockModel()
        # Set img_context_token_id on the model to avoid the missing attribute issue
        mock_model.img_context_token_id = 103
        
        # Create a mock tokenizer
        class MockTokenizer:
            def __init__(self):
                self.unk_token_id = 0
                self.pad_token_id = 1
                self.eos_token_id = 2
                
            def convert_tokens_to_ids(self, token):
                return {'<|latent|>': 50257, '<|start_latent|>': 50258, '<|end_latent|>': 50259, '<IMG_CONTEXT>': 103}.get(token, 0)
        
        mock_tokenizer = MockTokenizer()
        wrapper = LatentWrapper(mock_model, mock_tokenizer)
        
        # Test 1: Legacy cache format
        print("\n--- Test 1: Legacy cache format ---")
        batch_size, num_heads, seq_len, head_dim = 1, 8, 10, 64
        layers = 2
        
        legacy_cache = []
        for layer in range(layers):
            key = torch.randn(batch_size, num_heads, seq_len, head_dim)
            value = torch.randn(batch_size, num_heads, seq_len, head_dim)
            legacy_cache.append((key, value))
        
        print(f"Created legacy cache with {len(legacy_cache)} layers")
        
        # Test slicing
        compute_range = (3, 7)  # slice from position 0 to 3
        sliced_cache = wrapper._extract_kv_cache_slice(legacy_cache, compute_range)
        
        if sliced_cache is not None:
            print(f"✓ Cache slicing successful")
            print(f"  Original seq_len: {seq_len}")
            print(f"  Compute range: {compute_range}")
            print(f"  Sliced cache type: {type(sliced_cache)}")
            
            # Check sliced dimensions
            if hasattr(sliced_cache, 'key_cache'):
                # DynamicCache format
                print(f"  Sliced seq_len: {sliced_cache.key_cache[0].shape[2]}")
                print(f"  Expected seq_len: {compute_range[0]}")
                assert sliced_cache.key_cache[0].shape[2] == compute_range[0], "Cache slice length mismatch"
            else:
                # Legacy format
                print(f"  Sliced seq_len: {sliced_cache[0][0].shape[2]}")
                print(f"  Expected seq_len: {compute_range[0]}")
                assert sliced_cache[0][0].shape[2] == compute_range[0], "Cache slice length mismatch"
            
            print("✓ Cache slice dimensions are correct")
        else:
            print("✗ Cache slicing failed")
            return False
        
        # Test 2: DynamicCache format (if available)
        if has_cache_utils:
            print("\n--- Test 2: DynamicCache format ---")
            dynamic_cache = DynamicCache()
            
            for layer in range(layers):
                key = torch.randn(batch_size, num_heads, seq_len, head_dim)
                value = torch.randn(batch_size, num_heads, seq_len, head_dim)
                dynamic_cache.update(key, value, layer)
            
            print(f"Created DynamicCache with {len(dynamic_cache.key_cache)} layers")
            
            # Test slicing
            sliced_dynamic_cache = wrapper._extract_kv_cache_slice(dynamic_cache, compute_range)
            
            if sliced_dynamic_cache is not None:
                print(f"✓ DynamicCache slicing successful")
                print(f"  Sliced cache type: {type(sliced_dynamic_cache)}")
                print(f"  Sliced seq_len: {sliced_dynamic_cache.key_cache[0].shape[2]}")
                print(f"  Expected seq_len: {compute_range[0]}")
                assert sliced_dynamic_cache.key_cache[0].shape[2] == compute_range[0], "DynamicCache slice length mismatch"
                print("✓ DynamicCache slice dimensions are correct")
            else:
                print("✗ DynamicCache slicing failed")
                return False
        
        # Test 3: Validation
        print("\n--- Test 3: Cache validation ---")
        is_valid = wrapper._validate_kv_cache(sliced_cache)
        print(f"Cache validation result: {is_valid}")
        
        if is_valid:
            print("✓ Sliced cache validation passed")
        else:
            print("✗ Sliced cache validation failed")
            return False
        
        # Test 4: First pass (should return None)
        print("\n--- Test 4: First pass handling ---")
        first_pass_cache = wrapper._extract_kv_cache_slice(legacy_cache, (0, 5))
        if first_pass_cache is None:
            print("✓ First pass correctly returns None")
        else:
            print("✗ First pass should return None but got:", type(first_pass_cache))
            return False
        
        print("\n🎉 All cache slicing tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cache_slicing()
    if success:
        print("\n✅ Cache slicing fix is working correctly!")
        print("The model should now handle stage 1 latent processing without cache API errors.")
    else:
        print("\n❌ Cache slicing tests failed!")
        sys.exit(1)
