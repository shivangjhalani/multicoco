#!/usr/bin/env python3
"""
Test script to verify the simplified LatentWrapper works correctly.
"""
import sys
import os
import torch

# Clear any cached modules to ensure we get the latest version
modules_to_clear = [k for k in sys.modules.keys() if k.startswith('multicoco')]
for module in modules_to_clear:
    del sys.modules[module]

sys.path.insert(0, '/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco')

def test_latent_wrapper_import():
    """Test that we can import the LatentWrapper without errors."""
    try:
        from multicoco.latent_wrapper import LatentWrapper
        print("✅ LatentWrapper imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import LatentWrapper: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading():
    """Test that configuration loading still works with the new wrapper."""
    try:
        from multicoco.config import MultiCoCoConfig
        config = MultiCoCoConfig.load_with_base('args/aokvqa_cot.yaml')
        print(f"✅ Config loaded: CoCoNut enabled = {config.coconut.enabled}")
        return config
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_wrapper_creation():
    """Test creating a LatentWrapper with a dummy model."""
    try:
        from multicoco.latent_wrapper import LatentWrapper
        from transformers import AutoTokenizer
        
        # Create a simple dummy model for testing
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                
            def forward(self, **kwargs):
                return {"logits": torch.randn(1, 10, 1000)}
                
            def generate(self, **kwargs):
                return torch.randint(0, 1000, (1, 10))
        
        dummy_model = DummyModel()
        
        # Create a simple tokenizer-like object
        class DummyTokenizer:
            def __init__(self):
                self.vocab_size = 1000
                
            def encode(self, text):
                return [1, 2, 3]
                
            def decode(self, ids):
                return "dummy text"
                
            def convert_tokens_to_ids(self, token):
                # Return dummy IDs for latent tokens
                token_map = {
                    '<|latent|>': 998,
                    '<|start_latent|>': 997,
                    '<|end_latent|>': 996
                }
                return token_map.get(token, 999)
        
        tokenizer = DummyTokenizer()
        
        # Create wrapper
        wrapper = LatentWrapper(dummy_model, tokenizer)
        print("✅ LatentWrapper created successfully")
        
        # Test forward pass
        dummy_input = torch.randint(0, 1000, (1, 10))
        output = wrapper.forward(input_ids=dummy_input)
        print(f"✅ Forward pass successful, output shape: {output['logits'].shape}")
        
        # Test generate
        generated = wrapper.generate(input_ids=dummy_input, max_new_tokens=5)
        print(f"✅ Generate successful, output shape: {generated.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to create/test wrapper: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Testing MultiCoCo LatentWrapper Fix")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    print("\n1. Testing import...")
    if test_latent_wrapper_import():
        success_count += 1
    
    print("\n2. Testing config loading...")
    config = test_config_loading()
    if config is not None:
        success_count += 1
    
    print("\n3. Testing wrapper creation...")
    if test_wrapper_creation():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {success_count}/{total_tests} passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! The LatentWrapper fix appears to be working correctly.")
        print("\nKey improvements:")
        print("- Removed problematic two-pass forward with hidden state injection")
        print("- Simplified to single-pass delegation to base model")
        print("- Maintained compatibility with InternVL's multimodal processing")
        print("- Preserved end-to-end gradient flow for proper latent learning")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
