#!/usr/bin/env python3
"""
Test script to verify LatentWrapperV2 integration.
"""

import sys
import os
import torch
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multicoco.latent_wrapper_v2 import LatentWrapperV2
from multicoco.constants import COCONUT_SPECIAL_TOKENS

def test_latent_wrapper_v2_import():
    """Test that LatentWrapperV2 can be imported and initialized."""
    print("Testing LatentWrapperV2 import and initialization...")
    
    # Create a mock model and tokenizer
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(100, 64)
        
        def forward(self, input_ids, **kwargs):
            return type('obj', (object,), {'hidden_states': self.embed_tokens(input_ids)})
    
    # Initialize tokenizer with special tokens
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    tokenizer.add_special_tokens({'additional_special_tokens': COCONUT_SPECIAL_TOKENS})
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create mock model
    model = MockModel()
    
    # Test LatentWrapperV2 initialization
    try:
        wrapper = LatentWrapperV2(model, tokenizer)
        print("✓ LatentWrapperV2 initialized successfully")
        
        # Test that it has the expected methods
        assert hasattr(wrapper, 'multimodal_prep'), "Missing multimodal_prep method"
        assert hasattr(wrapper, 'latent_injection'), "Missing latent_injection method"
        assert hasattr(wrapper, 'forward'), "Missing forward method"
        print("✓ LatentWrapperV2 has expected methods")
        
        # Test token detection
        latent_start_id = tokenizer.convert_tokens_to_ids('<|start_latent|>')
        latent_end_id = tokenizer.convert_tokens_to_ids('<|end_latent|>')
        print(f"✓ Latent tokens detected: start_id={latent_start_id}, end_id={latent_end_id}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error initializing LatentWrapperV2: {e}")
        return False

def test_integration_compatibility():
    """Test that the new wrapper maintains compatibility with existing code."""
    print("\nTesting integration compatibility...")
    
    try:
        # Test import from run.py style
        from multicoco.latent_wrapper_v2 import LatentWrapperV2
        print("✓ LatentWrapperV2 can be imported from expected location")
        
        # Test that constructor signature matches old LatentWrapper
        import inspect
        sig = inspect.signature(LatentWrapperV2.__init__)
        params = list(sig.parameters.keys())
        expected_params = ['self', 'model', 'tokenizer']
        
        if params == expected_params:
            print("✓ Constructor signature matches expected format")
        else:
            print(f"✗ Constructor signature mismatch. Expected {expected_params}, got {params}")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Integration compatibility error: {e}")
        return False

if __name__ == "__main__":
    print("Testing LatentWrapperV2 Integration")
    print("=" * 50)
    
    success = True
    success &= test_latent_wrapper_v2_import()
    success &= test_integration_compatibility()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All LatentWrapperV2 integration tests passed!")
    else:
        print("✗ Some tests failed. Check the output above.")
    
    sys.exit(0 if success else 1)
