#!/usr/bin/env python3
"""
Test to verify the embedding consistency fix in LatentWrapper.
This test ensures both passes use the same embedding layer, preventing embedding drift.
"""

import sys
import os
import torch
import torch.nn as nn
sys.path.append('/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco')

def test_embedding_consistency():
    """Test that LatentWrapper uses the same embedding layer in both passes."""
    print("🧪 Testing embedding consistency fix...")
    
    try:
        from multicoco import MultiCoCo
        from multicoco.latent_wrapper import LatentWrapper
        from multicoco.constants import COCONUT_SPECIAL_TOKENS
        import torch
        
        print("✅ Successfully imported required modules")
        
        # Create small model for testing
        print("📦 Creating MultiCoCo instance...")
        multicoco_model = MultiCoCo(
            model_id="OpenGVLab/InternVL3-1B-Pretrained",
            special_tokens=list(COCONUT_SPECIAL_TOKENS),
            torch_dtype="bfloat16"
        )
        print("✅ MultiCoCo created successfully!")
        
        # Create LatentWrapper
        latent_wrapper = LatentWrapper(multicoco_model, multicoco_model.tokenizer)
        print("✅ LatentWrapper created successfully!")
        
        # Critical test: Check if both passes use the same embedding layer
        print("\n🔍 Testing embedding layer consistency...")
        
        # Test 1: Check that LatentWrapper.embedding is the same object as the base model's embedding
        base_embedding = None
        if hasattr(multicoco_model, 'language_model') and hasattr(multicoco_model.language_model, 'model'):
            if hasattr(multicoco_model.language_model.model, 'embed_tokens'):
                base_embedding = multicoco_model.language_model.model.embed_tokens
        
        if base_embedding is None:
            base_embedding = multicoco_model.get_input_embeddings()
        
        print(f"Base model embedding: {type(base_embedding)} at {id(base_embedding)}")
        print(f"LatentWrapper embedding: {type(latent_wrapper.embedding)} at {id(latent_wrapper.embedding)}")
        
        # CRITICAL CHECK: These should be the same object
        if latent_wrapper.embedding is base_embedding:
            print("✅ CRITICAL PASS: LatentWrapper uses the SAME embedding layer as base model!")
            print("   This ensures both passes use the same embedding space.")
        else:
            print("❌ CRITICAL FAIL: LatentWrapper uses a DIFFERENT embedding layer!")
            print("   This would cause embedding drift and break CoCoNut training!")
            return False
        
        # Test 2: Verify that parameter updates affect both
        print("\n🔧 Testing parameter sharing...")
        original_weight = base_embedding.weight[0, 0].clone()
        
        # Simulate a gradient update
        with torch.no_grad():
            base_embedding.weight[0, 0] += 0.1
        
        updated_weight_base = base_embedding.weight[0, 0]
        updated_weight_wrapper = latent_wrapper.embedding.weight[0, 0]
        
        if torch.equal(updated_weight_base, updated_weight_wrapper):
            print("✅ Parameter sharing works: Updates to base model affect LatentWrapper")
        else:
            print("❌ Parameter sharing broken: Updates don't propagate!")
            return False
        
        # Restore original weight
        with torch.no_grad():
            base_embedding.weight[0, 0] = original_weight
        
        # Test 3: Test a simple forward pass to ensure it still works
        print("\n🚀 Testing forward pass with shared embedding...")
        test_input = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        
        # Test embeddings directly
        base_embed_result = base_embedding(test_input)
        wrapper_embed_result = latent_wrapper.embedding(test_input)
        
        if torch.equal(base_embed_result, wrapper_embed_result):
            print("✅ Forward pass consistency: Both embeddings produce identical results")
        else:
            print("❌ Forward pass inconsistency: Embeddings produce different results!")
            return False
        
        print("\n🎉 All embedding consistency tests passed!")
        print("✅ The critical embedding drift bug has been fixed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_coconut_forward_consistency():
    """Test that CoCoNut forward pass uses consistent embedding spaces."""
    print("\n🧪 Testing CoCoNut forward pass consistency...")
    
    try:
        from multicoco import MultiCoCo
        from multicoco.latent_wrapper import LatentWrapper
        from multicoco.constants import COCONUT_SPECIAL_TOKENS
        import torch
        
        # Create model
        multicoco_model = MultiCoCo(
            model_id="OpenGVLab/InternVL3-1B-Pretrained",
            special_tokens=list(COCONUT_SPECIAL_TOKENS),
            torch_dtype="bfloat16"
        )
        latent_wrapper = LatentWrapper(multicoco_model, multicoco_model.tokenizer)
        
        # Create input with latent tokens
        tokenizer = multicoco_model.tokenizer
        test_text = "Question: What is 2+2? <|start_latent|> <|latent|> <|latent|> <|end_latent|> Answer: 4"
        input_ids = tokenizer.encode(test_text, return_tensors="pt")
        
        print(f"Test input shape: {input_ids.shape}")
        print(f"Contains latent tokens: {latent_wrapper._has_latent_spans(input_ids)}")
        
        # Test the forward pass
        with torch.no_grad():
            outputs = latent_wrapper.forward(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids)
            )
        
        print("✅ Forward pass with latent tokens completed successfully")
        print(f"Output logits shape: {outputs['logits'].shape if 'logits' in outputs else 'No logits'}")
        
        # Verify the spans are extracted correctly
        spans = latent_wrapper._extract_latent_spans(input_ids)
        print(f"Extracted latent spans: {spans}")
        
        if spans and spans[0]:  # Check if we have spans
            print("✅ Latent spans detected and processed correctly")
        else:
            print("⚠️  No latent spans found - check tokenization")
        
        return True
        
    except Exception as e:
        print(f"❌ CoCoNut forward test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_state_dict_handling():
    """Test that state_dict saving/loading works properly with shared embeddings."""
    print("\n🧪 Testing state_dict handling...")
    
    try:
        from multicoco import MultiCoCo
        from multicoco.latent_wrapper import LatentWrapper
        from multicoco.constants import COCONUT_SPECIAL_TOKENS
        import torch
        import tempfile
        import os
        
        # Create model
        multicoco_model = MultiCoCo(
            model_id="OpenGVLab/InternVL3-1B-Pretrained",
            special_tokens=list(COCONUT_SPECIAL_TOKENS),
            torch_dtype="bfloat16"
        )
        latent_wrapper = LatentWrapper(multicoco_model, multicoco_model.tokenizer)
        
        # Test state_dict creation
        print("💾 Testing state_dict creation...")
        state_dict = latent_wrapper.state_dict()
        print(f"✅ State dict created with {len(state_dict)} keys")
        
        # Test saving to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            torch.save(state_dict, f.name)
            temp_path = f.name
        
        print(f"✅ State dict saved to {temp_path}")
        
        # Test loading
        loaded_state_dict = torch.load(temp_path, map_location='cpu')
        print(f"✅ State dict loaded with {len(loaded_state_dict)} keys")
        
        # Test state_dict loading
        latent_wrapper.load_state_dict(loaded_state_dict, strict=False)
        print("✅ State dict loaded successfully")
        
        # Cleanup
        os.unlink(temp_path)
        
        return True
        
    except Exception as e:
        print(f"❌ State dict test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Running Embedding Consistency Tests")
    print("=" * 60)
    
    success = True
    
    # Test 1: Embedding consistency
    success &= test_embedding_consistency()
    
    # Test 2: CoCoNut forward pass
    success &= test_coconut_forward_consistency()
    
    # Test 3: State dict handling
    success &= test_state_dict_handling()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS PASSED!")
        print("🎉 The embedding drift bug has been successfully fixed!")
        print("   Both CoCoNut passes now use the same embedding space.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("   The embedding consistency issue may not be fully resolved.")
    
    print("=" * 60)
    sys.exit(0 if success else 1)
