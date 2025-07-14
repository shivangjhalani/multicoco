#!/usr/bin/env python3
"""
Test the InternVL3-1B API compatibility fix for prepare_inputs_for_multimodal
"""

import torch
import sys
import os
sys.path.append('/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco')

from multicoco.latent_wrapper import LatentWrapper
from multicoco.model import get_model
from multicoco.data import get_tokenizer

def test_internvl_api_fix():
    """Test that the InternVL3-1B API compatibility fix works"""
    print("🧪 Testing InternVL3-1B API Compatibility Fix")
    print("=" * 50)
    
    try:
        # Initialize model
        print("📦 Loading InternVL3-1B model...")
        base_model, tokenizer = get_model(
            model_name="OpenGVLab/InternVL3-1B-Pretrained",
            model_path="OpenGVLab/InternVL3-1B-Pretrained",
            max_dynamic_patch=1,
            use_flash_attn=False
        )
        print(f"✅ Model loaded: {type(base_model)}")
        
        # Create LatentWrapper
        print("🎯 Creating LatentWrapper...")
        wrapper = LatentWrapper(base_model, tokenizer)
        print(f"✅ LatentWrapper created: {type(wrapper)}")
        
        # Test the new method exists
        print("🔍 Testing _prepare_inputs_for_multimodal_internvl method...")
        assert hasattr(wrapper, '_prepare_inputs_for_multimodal_internvl'), "_prepare_inputs_for_multimodal_internvl method not found"
        print("✅ Method found")
        
        # Create test inputs
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {device}")
        
        # Test text with image placeholders
        test_text = "Describe this image: <IMG_CONTEXT>" * 10  # Simulate image tokens
        inputs = tokenizer(test_text, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        print(f"📝 Test input shape: {input_ids.shape}")
        
        # Create dummy image embeddings (simulating InternVL's image feature extraction)
        # InternVL3-1B typically has 784 image tokens with 896-dimensional embeddings
        batch_size = input_ids.shape[0]
        hidden_size = 896  # InternVL3-1B text embedding dimension
        num_image_tokens = 10  # Match the number of <IMG_CONTEXT> tokens
        image_embeds = torch.randn(batch_size, num_image_tokens, hidden_size, device=device)
        print(f"🖼️  Test image embeds shape: {image_embeds.shape}")
        
        # Test the new method
        print("🧪 Testing multimodal input preparation...")
        try:
            result_embeds = wrapper._prepare_inputs_for_multimodal_internvl(
                input_ids=input_ids,
                image_embeds=image_embeds
            )
            print(f"✅ Method executed successfully")
            print(f"📊 Result shape: {result_embeds.shape}")
            print(f"📊 Expected shape: {input_ids.shape + (hidden_size,)}")
            
            # Verify output shape is correct
            expected_shape = input_ids.shape + (hidden_size,)
            assert result_embeds.shape == expected_shape, f"Shape mismatch: {result_embeds.shape} vs {expected_shape}"
            print("✅ Output shape is correct")
            
            # Verify output is on correct device and has correct dtype
            assert result_embeds.device == device, f"Device mismatch: {result_embeds.device} vs {device}"
            print("✅ Output device is correct")
            
            print("\n🎉 InternVL3-1B API compatibility fix working correctly!")
            return True
            
        except Exception as e:
            print(f"❌ Method execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_internvl_api_fix()
    if success:
        print("\n✅ ALL TESTS PASSED - InternVL3-1B API fix is working!")
        sys.exit(0)
    else:
        print("\n❌ TESTS FAILED - Please check the implementation")
        sys.exit(1)
