#!/usr/bin/env python3
"""
Test the InternVL3-1B API compatibility fix for prepare_inputs_for_multimodal.

This test verifies that our _prepare_inputs_for_multimodal_internvl method works correctly
and replaces the missing prepare_inputs_for_multimodal method.
"""

import torch
import sys
import os
sys.path.append('/home/ubuntu/multicoco')

from multicoco.model import MultiCoCo
from multicoco.latent_wrapper import LatentWrapper

def test_internvl_api_fix():
    """Test that the InternVL3-1B API fix works correctly"""
    print("🧪 Testing InternVL3-1B API compatibility fix...")
    
    try:
        # Initialize the model (this should work with our fixes)
        print("📦 Loading MultiCoCo model...")
        model = MultiCoCo(model_id="OpenGVLab/InternVL3-1B-Pretrained")
        print(f"✅ Model loaded successfully: {type(model.model)}")
        
        # Create LatentWrapper
        print("🎭 Creating LatentWrapper...")
        latent_special_tokens = ['<|start_latent|>', '<|latent|>', '<|end_latent|>']
        model.tokenizer.add_tokens(latent_special_tokens)
        wrapped_model = LatentWrapper(model.model, model.tokenizer)
        print(f"✅ LatentWrapper created successfully")
        
        # Test that the _prepare_inputs_for_multimodal_internvl method exists
        print("🔍 Checking API compatibility method...")
        assert hasattr(wrapped_model, '_prepare_inputs_for_multimodal_internvl'), \
            "_prepare_inputs_for_multimodal_internvl method not found"
        print("✅ API compatibility method found")
        
        # Test the method with dummy inputs
        print("🧮 Testing multimodal input preparation...")
        device = next(model.model.parameters()).device
        
        # Create dummy inputs
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
        
        # Test without image embeddings (should return text embeddings as-is)
        text_embeds = wrapped_model._prepare_inputs_for_multimodal_internvl(
            input_ids=input_ids,
            image_embeds=None
        )
        print(f"✅ Text-only processing works, output shape: {text_embeds.shape}")
        
        # Test with dummy image embeddings
        embed_dim = text_embeds.shape[-1]  # Get embedding dimension
        dummy_image_embeds = torch.randn(1, 10, embed_dim, device=device)  # 10 image tokens
        
        # This should work without the 'prepare_inputs_for_multimodal' error
        multimodal_embeds = wrapped_model._prepare_inputs_for_multimodal_internvl(
            input_ids=input_ids,
            image_embeds=dummy_image_embeds
        )
        print(f"✅ Multimodal processing works, output shape: {multimodal_embeds.shape}")
        
        print("\n🎉 InternVL3-1B API compatibility fix is working correctly!")
        print("✅ The 'prepare_inputs_for_multimodal' error should now be resolved")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_internvl_api_fix()
    sys.exit(0 if success else 1)
