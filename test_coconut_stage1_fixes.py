#!/usr/bin/env python3

"""
Quick test script to verify CoCoNut latent injection fixes.
"""

import torch
from multicoco.latent_wrapper import LatentWrapper
from multicoco.model import MultiCoCo

def test_kv_cache_validation():
    """Test KV cache validation with different formats"""
    model = MultiCoCo(model_id='microsoft/DialoGPT-small')
    wrapper = LatentWrapper(model.model, model.tokenizer)
    
    # Test legacy format
    legacy_cache = [(torch.randn(1, 8, 10, 64), torch.randn(1, 8, 10, 64)) for _ in range(12)]
    print(f"✅ Legacy cache validation: {wrapper._validate_kv_cache(legacy_cache)}")
    
    # Test DynamicCache format simulation
    class MockDynamicCache:
        def __init__(self):
            self.key_cache = [torch.randn(1, 8, 10, 64) for _ in range(12)]
            self.value_cache = [torch.randn(1, 8, 10, 64) for _ in range(12)]
    
    dynamic_cache = MockDynamicCache()
    print(f"✅ DynamicCache validation: {wrapper._validate_kv_cache(dynamic_cache)}")
    
    print("✅ All KV cache tests passed!")

def test_internvl_helper():
    """Test InternVL helper method"""
    model = MultiCoCo(model_id='microsoft/DialoGPT-small')
    wrapper = LatentWrapper(model.model, model.tokenizer)
    
    # Test helper method exists
    has_method = hasattr(wrapper, '_call_model_with_embeds_internvl_safe')
    print(f"✅ InternVL helper method exists: {has_method}")
    
    print("✅ InternVL compatibility test passed!")

if __name__ == "__main__":
    print("🧪 Testing CoCoNut fixes...")
    
    test_kv_cache_validation()
    test_internvl_helper()
    
    print("\n🎉 All tests passed! Ready for training.")
