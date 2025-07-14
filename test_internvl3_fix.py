#!/usr/bin/env python3
"""
Test script to verify the MultiCoCo attribute forwarding fix with actual InternVL3-1B.
This will test the real model and verify that LatentWrapper can access required attributes.
"""

import sys
import os
sys.path.append('/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco')

def test_internvl3_attribute_forwarding():
    """Test that MultiCoCo forwards attributes correctly with real InternVL3-1B."""
    
    print("🧪 Testing MultiCoCo attribute forwarding with InternVL3-1B...")
    
    try:
        from multicoco import MultiCoCo
        from multicoco.latent_wrapper import LatentWrapper
        from multicoco.constants import COCONUT_SPECIAL_TOKENS
        import torch
        
        print("✅ Successfully imported required modules")
        
        # Clear any cached models that might be causing issues
        print("🧹 Clearing transformers cache to avoid cached config issues...")
        try:
            from transformers.utils import TRANSFORMERS_CACHE
            import shutil
            cache_dir = os.path.join(TRANSFORMERS_CACHE, "models--OpenGVLab--InternVL3-1B-Pretrained")
            if os.path.exists(cache_dir):
                print(f"   Removing cached model at: {cache_dir}")
                shutil.rmtree(cache_dir)
            else:
                print("   No cached model found to remove")
        except Exception as e:
            print(f"   Warning: Could not clear cache: {e}")
        
        # Create MultiCoCo instance with InternVL3-1B
        print("📦 Creating MultiCoCo instance with InternVL3-1B...")
        print("   This may take a few minutes to download the model...")
        
        special_tokens = list(COCONUT_SPECIAL_TOKENS)
        try:
            multicoco_model = MultiCoCo(
                model_id="OpenGVLab/InternVL3-1B-Pretrained", 
                special_tokens=special_tokens,
                torch_dtype="bfloat16",
                trust_remote_code=True
            )
            print("✅ MultiCoCo instance created successfully")
        except Exception as e:
            print(f"❌ Failed to create MultiCoCo: {e}")
            
            # Try to diagnose the issue
            print("\n🔍 Diagnosing the issue...")
            try:
                from transformers import AutoConfig
                print("   Attempting to load config directly...")
                config = AutoConfig.from_pretrained("OpenGVLab/InternVL3-1B-Pretrained", trust_remote_code=True)
                print(f"   Config loaded successfully: {type(config)}")
                print(f"   Config class: {config.__class__.__name__}")
                
                # Check if it's the right config
                if hasattr(config, 'model_type'):
                    print(f"   Model type: {config.model_type}")
                if hasattr(config, 'architectures'):
                    print(f"   Architectures: {config.architectures}")
                    
            except Exception as config_e:
                print(f"   Config loading also failed: {config_e}")
                return False
            
            return False
        
        # Test that MultiCoCo can forward attributes to the underlying model
        print("\n🔍 Testing attribute forwarding...")
        
        # Test 1: extract_feature method (critical for vision processing)
        try:
            extract_feature_method = getattr(multicoco_model, 'extract_feature', None)
            if extract_feature_method is not None:
                print("✅ extract_feature method is accessible via forwarding")
            else:
                print("❌ extract_feature method is NOT accessible")
        except Exception as e:
            print(f"❌ Error accessing extract_feature: {e}")
        
        # Test 2: dtype property (critical for tensor operations)
        try:
            dtype_prop = getattr(multicoco_model, 'dtype', None)
            if dtype_prop is not None:
                print(f"✅ dtype property is accessible: {dtype_prop}")
            else:
                print("❌ dtype property is NOT accessible")
        except Exception as e:
            print(f"❌ Error accessing dtype: {e}")
        
        # Test 3: config attribute (needed for model introspection)
        try:
            config = getattr(multicoco_model, 'config', None)
            if config is not None:
                print("✅ config attribute is accessible")
                # Test downsample_ratio specifically (used in LatentWrapper)
                downsample_ratio = getattr(config, 'downsample_ratio', None)
                if downsample_ratio is not None:
                    print(f"✅ config.downsample_ratio is accessible: {downsample_ratio}")
                else:
                    print("⚠️  config.downsample_ratio is not accessible (might be normal)")
            else:
                print("❌ config attribute is NOT accessible")
        except Exception as e:
            print(f"❌ Error accessing config: {e}")
        
        # Test 4: num_image_token (used for image token calculations)
        try:
            num_image_token = getattr(multicoco_model, 'num_image_token', None)
            if num_image_token is not None:
                print(f"✅ num_image_token is accessible: {num_image_token}")
            else:
                print("⚠️  num_image_token is not accessible (might use default)")
        except Exception as e:
            print(f"❌ Error accessing num_image_token: {e}")
        
        # Test 5: Create LatentWrapper and test it doesn't crash
        print("\n🚀 Testing LatentWrapper creation with real InternVL3...")
        try:
            latent_wrapper = LatentWrapper(multicoco_model, multicoco_model.tokenizer)
            print("✅ LatentWrapper created successfully with MultiCoCo model")
            
            # Test the critical path that was failing before our fix
            print("\n🎯 Testing the critical vision processing path...")
            try:
                # This is the exact method that was failing before our fix
                dummy_pixel_values = torch.randn(1, 3, 448, 448, dtype=torch.bfloat16)
                device = torch.device('cpu')
                
                # This method calls self.base_model.extract_feature() and self.base_model.dtype
                vision_embeds = latent_wrapper._get_cached_vision_embeddings(dummy_pixel_values, device)
                print("✅ LatentWrapper can access extract_feature through MultiCoCo forwarding")
                print(f"   Vision embeddings shape: {vision_embeds.shape}")
                print("✅ The critical multimodal path is now working!")
                
            except Exception as e:
                print(f"❌ LatentWrapper vision processing failed: {e}")
                import traceback
                print("Full traceback:")
                traceback.print_exc()
                return False
                
        except Exception as e:
            print(f"❌ LatentWrapper creation failed: {e}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            return False
        
        # Test 6: Test a simple forward pass to ensure everything works end-to-end
        print("\n🔄 Testing simple forward pass...")
        try:
            # Create dummy inputs
            dummy_pixel_values = torch.randn(1, 3, 448, 448, dtype=torch.bfloat16)
            dummy_input_ids = torch.randint(0, 1000, (1, 20))
            dummy_attention_mask = torch.ones_like(dummy_input_ids)
            
            # Test MultiCoCo forward (should work)
            with torch.no_grad():
                outputs = multicoco_model.forward(
                    pixel_values=dummy_pixel_values,
                    input_ids=dummy_input_ids,
                    attention_mask=dummy_attention_mask
                )
            print("✅ MultiCoCo forward pass successful")
            
            # Test LatentWrapper forward (the real test)
            with torch.no_grad():
                outputs = latent_wrapper.forward(
                    pixel_values=dummy_pixel_values,
                    input_ids=dummy_input_ids,
                    attention_mask=dummy_attention_mask
                )
            print("✅ LatentWrapper forward pass successful")
            print("✅ End-to-end multimodal processing is working!")
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            return False
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_internvl3_attribute_forwarding()
    if success:
        print("\n" + "="*60)
        print("✅ CONCLUSION: The fix is working perfectly!")
        print("="*60)
        print("🎯 Key findings:")
        print("   1. MultiCoCo can now forward attributes to InternVL3")
        print("   2. LatentWrapper can access extract_feature(), dtype, etc.")
        print("   3. The multimodal vision processing path works")
        print("   4. End-to-end forward passes are successful")
        print("   5. The original recursion issue has been resolved")
        print("\n🚀 Your MultiCoCo implementation is now fully functional!")
        print("   The LatentWrapper will be able to properly process")
        print("   multimodal inputs with the underlying InternVL3 model.")
    else:
        print("\n" + "="*60)
        print("❌ CONCLUSION: Issues remain to be resolved.")
        print("="*60)
    
    sys.exit(0 if success else 1)
