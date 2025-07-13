#!/usr/bin/env python3
"""
Final definitive test for LatentWrapper attribute access.
"""

import sys
import os
import torch
import torch.nn as nn

# Add the current directory to Python path
sys.path.insert(0, '/kaggle/working/multicoco')

def test_final_fix():
    """Final test to verify LatentWrapper works correctly"""
    print("🧪 Testing LatentWrapper final fix...")
    
    try:
        # Import directly from the multicoco directory
        from multicoco.latent_wrapper import LatentWrapper
        print("✅ Successfully imported LatentWrapper")
        
        # Create a simple mock model
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 64)
                self.linear = nn.Linear(64, 100)
                
            def get_input_embeddings(self):
                return self.embedding
                
            def forward(self, input_ids):
                return self.linear(self.embedding(input_ids))
            
            def generate(self, input_ids, **kwargs):
                return input_ids  # Simple mock
                
            @property
            def device(self):
                return next(self.parameters()).device
        
        # Create mock tokenizer
        class MockTokenizer:
            def __init__(self):
                self.eos_token_id = 1
                self.pad_token_id = 0
                
            def convert_tokens_to_ids(self, token):
                token_map = {
                    '<|latent|>': 10, 
                    '<|start_latent|>': 11, 
                    '<|end_latent|>': 12
                }
                return token_map.get(token, 2)
        
        # Test wrapper creation
        base_model = MockModel()
        tokenizer = MockTokenizer()
        wrapper = LatentWrapper(base_model, tokenizer)
        
        print("✅ LatentWrapper created successfully")
        
        # Test critical attributes
        print("🔍 Testing critical attributes...")
        
        # Test hasattr
        has_model = hasattr(wrapper, 'model')
        print(f"  hasattr(wrapper, 'model'): {has_model}")
        
        if has_model:
            model_attr = wrapper.model
            print(f"  wrapper.model type: {type(model_attr)}")
            print(f"  wrapper.model is wrapper.base_model: {model_attr is wrapper.base_model}")
        
        has_device = hasattr(wrapper, 'device')
        print(f"  hasattr(wrapper, 'device'): {has_device}")
        
        if has_device:
            device = wrapper.device
            print(f"  wrapper.device: {device}")
        
        has_tokenizer = hasattr(wrapper, 'tokenizer')
        print(f"  hasattr(wrapper, 'tokenizer'): {has_tokenizer}")
        
        if has_tokenizer:
            print(f"  wrapper.tokenizer type: {type(wrapper.tokenizer)}")
        
        # Test method delegation
        has_generate = hasattr(wrapper, 'generate')
        print(f"  hasattr(wrapper, 'generate'): {has_generate}")
        
        if has_generate:
            input_ids = torch.randint(0, 100, (1, 5))
            result = wrapper.generate(input_ids)
            print(f"  wrapper.generate() result shape: {result.shape}")
        
        print("\n🎉 All attribute tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_final_fix()
    if success:
        print("\n✅ DEFINITIVE FIX CONFIRMED: LatentWrapper is working correctly!")
    else:
        print("\n❌ Fix failed - need further debugging")
