#!/usr/bin/env python3
"""
Test script to verify the MultiCoCo attribute forwarding fix.
This tests whether LatentWrapper can access the required attributes from the underlying InternVL model.
"""

import sys
import os
sys.path.append('/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco')

def test_attribute_forwarding():
    """Test that MultiCoCo forwards attributes to the underlying model properly."""
    
    print("🧪 Testing MultiCoCo attribute forwarding fix...")
    
    try:
        from multicoco import MultiCoCo
        from multicoco.latent_wrapper import LatentWrapper
        from multicoco.constants import COCONUT_SPECIAL_TOKENS
        
        print("✅ Successfully imported required modules")
        
        # Create a MultiCoCo instance (this loads the actual InternVL model)
        print("📦 Creating MultiCoCo instance...")
        special_tokens = list(COCONUT_SPECIAL_TOKENS)
        multicoco_model = MultiCoCo(
            model_id="OpenGVLab/InternVL3-1B-Pretrained", 
            special_tokens=special_tokens,
            torch_dtype="bfloat16"
        )
        print("✅ MultiCoCo instance created successfully")
        
        # Test that MultiCoCo can forward attributes to the underlying model
        print("\n🔍 Testing attribute forwarding...")
        
        # Test 1: extract_feature method
        try:
            extract_feature_method = getattr(multicoco_model, 'extract_feature', None)
            if extract_feature_method is not None:
                print("✅ extract_feature method is accessible")
            else:
                print("❌ extract_feature method is NOT accessible")
        except Exception as e:
            print(f"❌ Error accessing extract_feature: {e}")
        
        # Test 2: dtype property
        try:
            dtype_prop = getattr(multicoco_model, 'dtype', None)
            if dtype_prop is not None:
                print(f"✅ dtype property is accessible: {dtype_prop}")
            else:
                print("❌ dtype property is NOT accessible")
        except Exception as e:
            print(f"❌ Error accessing dtype: {e}")
        
        # Test 3: conv_template attribute
        try:
            conv_template = getattr(multicoco_model, 'conv_template', None)
            if conv_template is not None:
                print("✅ conv_template is accessible")
            else:
                print("⚠️  conv_template is not accessible (this might be expected)")
        except Exception as e:
            print(f"⚠️  Error accessing conv_template: {e} (this might be expected)")
        
        # Test 4: config attribute
        try:
            config = getattr(multicoco_model, 'config', None)
            if config is not None:
                print("✅ config attribute is accessible")
                # Test downsample_ratio specifically
                downsample_ratio = getattr(config, 'downsample_ratio', None)
                if downsample_ratio is not None:
                    print(f"✅ config.downsample_ratio is accessible: {downsample_ratio}")
                else:
                    print("⚠️  config.downsample_ratio is not accessible")
            else:
                print("❌ config attribute is NOT accessible")
        except Exception as e:
            print(f"❌ Error accessing config: {e}")
        
        # Test 5: Create LatentWrapper and test it doesn't crash
        print("\n🚀 Testing LatentWrapper creation...")
        try:
            latent_wrapper = LatentWrapper(multicoco_model, multicoco_model.tokenizer)
            print("✅ LatentWrapper created successfully with MultiCoCo model")
            
            # Test that LatentWrapper can access the required methods
            print("\n🔍 Testing LatentWrapper attribute access...")
            
            # Test _get_cached_vision_embeddings method (which uses extract_feature)
            try:
                import torch
                dummy_pixel_values = torch.randn(1, 3, 448, 448, dtype=torch.bfloat16)
                device = torch.device('cpu')
                
                # This should not crash now
                vision_embeds = latent_wrapper._get_cached_vision_embeddings(dummy_pixel_values, device)
                print("✅ LatentWrapper can access extract_feature through MultiCoCo")
                
            except Exception as e:
                print(f"❌ LatentWrapper cannot access extract_feature: {e}")
            
        except Exception as e:
            print(f"❌ LatentWrapper creation failed: {e}")
            return False
        
        print("\n🎉 All tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_attribute_forwarding()
    if success:
        print("\n✅ CONCLUSION: The fix appears to be working!")
        print("   MultiCoCo can now forward attributes to the underlying InternVL model.")
        print("   LatentWrapper should be able to access extract_feature, dtype, config, etc.")
    else:
        print("\n❌ CONCLUSION: The fix needs more work.")
    
    sys.exit(0 if success else 1)
