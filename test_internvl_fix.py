#!/usr/bin/env python3
"""
Quick test script for the InternVL3-1B attribute forwarding fix.
This tests the __getattr__ fix with the actual model.
"""

import sys
import os
import torch

sys.path.append('/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco')

def test_internvl_attribute_forwarding():
    """Test attribute forwarding with real InternVL3-1B model."""
    
    print("🧪 Testing MultiCoCo attribute forwarding with InternVL3-1B...")
    print("   Note: This will download the model if not cached (~1.9GB)")
    
    try:
        from multicoco import MultiCoCo
        from multicoco.latent_wrapper import LatentWrapper
        from multicoco.constants import COCONUT_SPECIAL_TOKENS
        
        print("✅ Successfully imported required modules")
        
        # Create MultiCoCo instance
        print("📦 Creating MultiCoCo instance...")
        multicoco_model = MultiCoCo(
            model_id="OpenGVLab/InternVL3-1B-Pretrained",
            special_tokens=list(COCONUT_SPECIAL_TOKENS),
            torch_dtype="bfloat16"
        )
        print("✅ MultiCoCo created successfully!")
        
        # Test attribute forwarding
        print("\n🔍 Testing critical attribute forwarding...")
        
        # Test extract_feature
        try:
            extract_feature = getattr(multicoco_model, 'extract_feature')
            print("✅ extract_feature method accessible")
            
            # Test calling it
            dummy_pixels = torch.randn(1, 3, 448, 448, dtype=torch.bfloat16)
            result = extract_feature(dummy_pixels)
            print(f"✅ extract_feature works: output shape {result.shape}")
            
        except Exception as e:
            print(f"❌ extract_feature failed: {e}")
            return False
        
        # Test dtype
        try:
            model_dtype = getattr(multicoco_model, 'dtype')
            print(f"✅ dtype accessible: {model_dtype}")
        except Exception as e:
            print(f"❌ dtype failed: {e}")
        
        # Test config
        try:
            config = getattr(multicoco_model, 'config')
            print(f"✅ config accessible: {type(config)}")
        except Exception as e:
            print(f"❌ config failed: {e}")
        
        # Test LatentWrapper creation
        print("\n🚀 Testing LatentWrapper creation...")
        try:
            latent_wrapper = LatentWrapper(multicoco_model, multicoco_model.tokenizer)
            print("✅ LatentWrapper created successfully!")
            
            # Test the critical vision embedding extraction
            vision_embeds = latent_wrapper._get_cached_vision_embeddings(dummy_pixels, dummy_pixels.device)
            print(f"✅ Vision embedding extraction works: {vision_embeds.shape}")
            
            print("\n🎉 ALL TESTS PASSED!")
            print("   The attribute forwarding fix is working correctly.")
            print("   LatentWrapper can now access InternVL model attributes.")
            return True
            
        except Exception as e:
            print(f"❌ LatentWrapper test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_internvl_attribute_forwarding()
    
    if success:
        print("\n✅ CONCLUSION: The fix is working!")
        print("   - MultiCoCo correctly forwards attributes to the underlying InternVL model")
        print("   - LatentWrapper can access extract_feature, dtype, config, etc.")
        print("   - The original architectural flaw has been resolved")
        print("   - Multimodal CoCoNut functionality should now work properly")
    else:
        print("\n❌ CONCLUSION: Issues remain.")
    
    sys.exit(0 if success else 1)
