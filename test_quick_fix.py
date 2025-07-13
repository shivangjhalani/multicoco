#!/usr/bin/env python3
"""
Quick test to verify LatentWrapper attribute access is working correctly.
"""

import sys
import os
import torch
import torch.nn as nn

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_quick_fix():
    """Quick test to verify the attribute access fix"""
    print("Testing LatentWrapper attribute access fix...")
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        
        # Create a simple mock model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 64)
                self.linear = nn.Linear(64, 100)
                
            def get_input_embeddings(self):
                return self.embedding
                
            def forward(self, input_ids):
                embeds = self.embedding(input_ids)
                return self.linear(embeds)
                
            @property
            def device(self):
                return next(self.parameters()).device
        
        # Create a simple tokenizer
        class SimpleTokenizer:
            def __init__(self):
                self.eos_token_id = 1
                self.pad_token_id = 0
                
            def convert_tokens_to_ids(self, token):
                return {'<|latent|>': 10, '<|start_latent|>': 11, '<|end_latent|>': 12}.get(token, 2)
        
        # Test wrapper creation
        base_model = SimpleModel()
        tokenizer = SimpleTokenizer()
        wrapper = LatentWrapper(base_model, tokenizer)
        
        print("✅ LatentWrapper created successfully")
        
        # Test attribute access
        assert hasattr(wrapper, 'model'), "Should have model property"
        assert wrapper.model is wrapper.base_model, "Model property should return base_model"
        print("✅ Model property works")
        
        assert hasattr(wrapper, 'device'), "Should have device property"
        device = wrapper.device
        print(f"✅ Device property works: {device}")
        
        # Test method delegation
        embeddings = wrapper.get_input_embeddings()
        assert embeddings is not None, "Should delegate get_input_embeddings"
        print("✅ Method delegation works")
        
        # Test simple forward pass without latent tokens
        input_ids = torch.randint(0, 100, (2, 5))
        output = wrapper.forward(input_ids=input_ids)
        print(f"✅ Forward pass works, output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
        
        print("\n🎉 All basic tests passed! The attribute access fix is working.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_quick_fix()
    sys.exit(0 if success else 1)
