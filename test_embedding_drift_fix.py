#!/usr/bin/env python3
"""
Focused test to verify the embedding drift fix.
This test simulates the exact scenario that was causing problems.
"""

import sys
import os
sys.path.append('/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco')

def test_embedding_drift_scenario():
    """
    Test the exact scenario that was causing embedding drift:
    1. First pass uses original embedding (via prepare_inputs_for_multimodal)
    2. Second pass uses LatentWrapper embedding
    3. Verify they are the same object
    """
    print("🧪 Testing Embedding Drift Fix")
    print("=" * 50)
    
    try:
        from multicoco import MultiCoCo
        from multicoco.latent_wrapper import LatentWrapper
        from multicoco.constants import COCONUT_SPECIAL_TOKENS
        import torch
        
        # Create model
        print("📦 Creating model...")
        multicoco_model = MultiCoCo(
            model_id="OpenGVLab/InternVL3-1B-Pretrained",
            special_tokens=list(COCONUT_SPECIAL_TOKENS),
            torch_dtype="bfloat16"
        )
        
        latent_wrapper = LatentWrapper(multicoco_model, multicoco_model.tokenizer)
        print("✅ Model created")
        
        # Get the embedding that would be used in first pass (prepare_inputs_for_multimodal)
        print("\n🔍 Checking embedding layers...")
        
        # This is what happens in _first_pass_hidden_states
        test_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        
        # Simulate first pass embedding (what prepare_inputs_for_multimodal uses)
        first_pass_embeds = multicoco_model.model.prepare_inputs_for_multimodal(
            input_ids=test_input_ids,
            pixel_values=None,
            image_embeds=None
        )
        print(f"First pass embeddings shape: {first_pass_embeds.shape}")
        
        # Simulate second pass embedding (what LatentWrapper uses)
        second_pass_embeds = latent_wrapper.embedding(test_input_ids)
        print(f"Second pass embeddings shape: {second_pass_embeds.shape}")
        
        # CRITICAL TEST: Are they using the same underlying embedding weights?
        # Get the actual embedding layer that prepare_inputs_for_multimodal uses
        if hasattr(multicoco_model.model, 'language_model'):
            original_embedding = multicoco_model.model.language_model.get_input_embeddings()
        else:
            original_embedding = multicoco_model.model.get_input_embeddings()
        
        print(f"\nOriginal embedding layer: {type(original_embedding)} at {id(original_embedding)}")
        print(f"LatentWrapper embedding: {type(latent_wrapper.embedding)} at {id(latent_wrapper.embedding)}")
        
        # Check if they're the same object
        if latent_wrapper.embedding is original_embedding:
            print("✅ PASS: Both passes use the SAME embedding layer!")
            print("   No embedding drift will occur during training.")
            same_embedding = True
        else:
            print("❌ FAIL: Different embedding layers detected!")
            print("   This WILL cause embedding drift and break CoCoNut!")
            same_embedding = False
        
        # Additional verification: Check that the weights are identical
        print("\n🔧 Verifying weight consistency...")
        original_weights = original_embedding.weight
        wrapper_weights = latent_wrapper.embedding.weight
        
        if torch.equal(original_weights, wrapper_weights):
            print("✅ Embedding weights are identical")
        else:
            print("❌ Embedding weights differ!")
            return False
        
        # Test that changes propagate
        print("\n🔄 Testing parameter propagation...")
        original_value = original_embedding.weight[0, 0].clone()
        
        # Modify original
        with torch.no_grad():
            original_embedding.weight[0, 0] += 0.01
        
        # Check if change appears in wrapper
        if torch.equal(original_embedding.weight[0, 0], latent_wrapper.embedding.weight[0, 0]):
            print("✅ Parameter changes propagate correctly")
            propagation_works = True
        else:
            print("❌ Parameter changes don't propagate!")
            propagation_works = False
        
        # Restore original value
        with torch.no_grad():
            original_embedding.weight[0, 0] = original_value
        
        # Overall result
        success = same_embedding and propagation_works
        
        print("\n" + "=" * 50)
        if success:
            print("🎉 EMBEDDING DRIFT BUG FIXED!")
            print("✅ Both CoCoNut passes use the same embedding space")
            print("✅ No parameter divergence will occur during training")
        else:
            print("❌ EMBEDDING DRIFT BUG STILL EXISTS!")
            print("⚠️  Training will be unstable due to embedding divergence")
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_before_after_comparison():
    """
    Show the difference between the old (buggy) and new (fixed) behavior.
    """
    print("\n📊 Before/After Comparison")
    print("=" * 50)
    
    print("BEFORE (Buggy Implementation):")
    print("  1st pass: Uses original embedding from InternVL3")
    print("  2nd pass: Uses COPIED embedding in LatentWrapper")
    print("  Result: ❌ Embedding matrices diverge → Breaks CoCoNut")
    
    print("\nAFTER (Fixed Implementation):")
    print("  1st pass: Uses original embedding from InternVL3")
    print("  2nd pass: Uses SAME embedding (shared reference)")
    print("  Result: ✅ Same embedding space → CoCoNut works correctly")
    
    return True

if __name__ == "__main__":
    print("🔧 CRITICAL BUG FIX VERIFICATION")
    print("Testing: Embedding Drift in CoCoNut Training")
    print("=" * 60)
    
    success = test_embedding_drift_scenario()
    test_before_after_comparison()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ CRITICAL BUG SUCCESSFULLY FIXED!")
        print("🚀 CoCoNut training will now work correctly")
        print("📈 Expect stable and effective latent reasoning")
    else:
        print("❌ CRITICAL BUG NOT FIXED!")
        print("⚠️  CoCoNut training will remain unstable")
        print("🔧 Further investigation required")
    
    print("=" * 60)
    sys.exit(0 if success else 1)
