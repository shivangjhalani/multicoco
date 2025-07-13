#!/usr/bin/env python3
"""
Test the fixed LatentWrapper attribute access.
"""

import sys
import os
import torch
import torch.nn as nn

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_fixed_wrapper():
    """Test the fixed wrapper"""
    print("Testing fixed LatentWrapper...")
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        
        # Create simple mock components
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 64)
                self.test_attr = "test_value"
                
            def get_input_embeddings(self):
                return self.embedding
                
            @property
            def device(self):
                return next(self.parameters()).device
        
        class MockTokenizer:
            def convert_tokens_to_ids(self, token):
                return {'<|latent|>': 10, '<|start_latent|>': 11, '<|end_latent|>': 12}.get(token, 2)
        
        # Test wrapper creation
        base_model = MockModel()
        tokenizer = MockTokenizer()
        wrapper = LatentWrapper(base_model, tokenizer)
        
        print("✅ LatentWrapper created successfully")
        
        # Test basic attribute access
        print(f"wrapper.base_model: {wrapper.base_model}")
        print(f"wrapper.tokenizer: {wrapper.tokenizer}")
        
        # Test property access
        print(f"wrapper.model: {wrapper.model}")
        print(f"wrapper.device: {wrapper.device}")
        
        # Test attribute delegation
        print(f"wrapper.test_attr: {wrapper.test_attr}")
        
        print("🎉 All tests passed! The fix is working.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_wrapper()
    sys.exit(0 if success else 1)
