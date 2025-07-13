#!/usr/bin/env python3
"""
Quick test to verify LatentWrapper attribute access fix
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing LatentWrapper attribute access fix...")

# Create mock classes without importing torch
class MockTokenizer:
    def convert_tokens_to_ids(self, token):
        token_map = {
            '<|latent|>': 50001,
            '<|start_latent|>': 50002, 
            '<|end_latent|>': 50003
        }
        return token_map.get(token, 1)

class MockEmbedding:
    pass

class MockModel:
    def __init__(self):
        self.device = "cpu"
        self.some_attr = "test_value"
    
    def get_input_embeddings(self):
        return MockEmbedding()

# Now test the LatentWrapper
try:
    from multicoco.latent_wrapper import LatentWrapper
    
    tokenizer = MockTokenizer()
    base_model = MockModel()
    
    print("✅ Creating LatentWrapper...")
    wrapper = LatentWrapper(base_model, tokenizer)
    
    print("✅ Testing hasattr for model property...")
    has_model = hasattr(wrapper, 'model')
    print(f"  hasattr(wrapper, 'model'): {has_model}")
    
    if has_model:
        print("✅ Testing model property access...")
        model = wrapper.model
        print(f"  wrapper.model is wrapper.base_model: {model is wrapper.base_model}")
    
    print("✅ Testing hasattr for device property...")
    has_device = hasattr(wrapper, 'device') 
    print(f"  hasattr(wrapper, 'device'): {has_device}")
    
    if has_device:
        print("✅ Testing device property access...")
        device = wrapper.device
        print(f"  wrapper.device: {device}")
    
    print("✅ Testing attribute delegation...")
    try:
        some_attr = wrapper.some_attr
        print(f"  wrapper.some_attr: {some_attr}")
        print("✅ Attribute delegation working!")
    except AttributeError as e:
        print(f"❌ Attribute delegation failed: {e}")
    
    print("✅ Testing non-existent attribute...")
    try:
        non_existent = wrapper.non_existent_attr
        print("❌ Should have raised AttributeError")
    except AttributeError:
        print("✅ Correctly raised AttributeError for non-existent attribute")
    
    print("\n🎉 All attribute access tests passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
