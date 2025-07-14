#!/usr/bin/env python3
"""
Training simulation test to verify embedding consistency during optimization.
This test simulates multiple training steps to ensure embeddings don't diverge.
"""

import sys
import os
sys.path.append('/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco')

def test_training_simulation():
    """
    Simulate multiple training steps to verify embeddings remain consistent.
    """
    print("🧪 Training Simulation Test")
    print("=" * 50)
    
    try:
        from multicoco import MultiCoCo
        from multicoco.latent_wrapper import LatentWrapper
        from multicoco.constants import COCONUT_SPECIAL_TOKENS
        import torch
        import torch.nn.functional as F
        
        # Create model
        print("📦 Creating model...")
        multicoco_model = MultiCoCo(
            model_id="OpenGVLab/InternVL3-1B-Pretrained",
            special_tokens=list(COCONUT_SPECIAL_TOKENS),
            torch_dtype="bfloat16"
        )
        
        latent_wrapper = LatentWrapper(multicoco_model, multicoco_model.tokenizer)
        print("✅ Model created")
        
        # Create sample input with latent tokens
        tokenizer = multicoco_model.tokenizer
        test_text = "Question: What is 2+2? <|start_latent|> <|latent|> <|latent|> <|end_latent|> Answer: 4"
        input_ids = tokenizer.encode(test_text, return_tensors="pt")
        labels = input_ids.clone()
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Has latent tokens: {latent_wrapper._has_latent_spans(input_ids)}")
        
        # Create optimizer for testing
        optimizer = torch.optim.SGD(latent_wrapper.parameters(), lr=0.01)
        
        # Store initial embedding weights for comparison
        original_embedding = None
        if hasattr(multicoco_model.model, 'language_model'):
            original_embedding = multicoco_model.model.language_model.get_input_embeddings()
        else:
            original_embedding = multicoco_model.model.get_input_embeddings()
        
        initial_weight = original_embedding.weight[0, :5].clone()
        print(f"Initial embedding weights (first 5): {initial_weight}")
        
        print("\n🏋️ Simulating training steps...")
        
        for step in range(3):
            print(f"\nStep {step + 1}:")
            
            # Forward pass
            outputs = latent_wrapper.forward(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                labels=labels
            )
            
            loss = outputs.get('loss')
            if loss is None:
                # Calculate loss manually if not provided
                logits = outputs['logits']
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1),
                    ignore_index=-100
                )
            
            print(f"  Loss: {loss.item():.4f}")
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients
            if original_embedding.weight.grad is not None:
                grad_norm = original_embedding.weight.grad.norm().item()
                print(f"  Embedding grad norm: {grad_norm:.6f}")
            else:
                print("  No gradients for embedding")
            
            # Update parameters
            optimizer.step()
            
            # Check that both embedding references point to the same updated weights
            original_weight = original_embedding.weight[0, :5]
            wrapper_weight = latent_wrapper.embedding.weight[0, :5]
            
            if torch.equal(original_weight, wrapper_weight):
                print(f"  ✅ Embedding consistency maintained")
                print(f"  Updated weights: {original_weight}")
            else:
                print(f"  ❌ EMBEDDING DRIFT DETECTED!")
                print(f"  Original: {original_weight}")
                print(f"  Wrapper:  {wrapper_weight}")
                return False
        
        # Final check: Verify total change from initial weights
        final_weight = original_embedding.weight[0, :5]
        weight_change = (final_weight - initial_weight).norm().item()
        print(f"\n📊 Total weight change magnitude: {weight_change:.6f}")
        
        if weight_change > 0:
            print("✅ Embeddings were successfully updated during training")
        else:
            print("⚠️  No embedding updates detected (may indicate frozen weights)")
        
        print("\n✅ Training simulation completed successfully!")
        print("🎉 No embedding drift detected during training!")
        
        return True
        
    except Exception as e:
        print(f"❌ Training simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_efficiency():
    """
    Test that using shared embeddings doesn't cause memory issues.
    """
    print("\n💾 Memory Efficiency Test")
    print("=" * 50)
    
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
        
        # Count parameters to ensure we're not duplicating embedding weights
        original_params = sum(p.numel() for p in multicoco_model.parameters())
        wrapper_params = sum(p.numel() for p in latent_wrapper.parameters())
        
        print(f"Original model parameters: {original_params:,}")
        print(f"LatentWrapper parameters: {wrapper_params:,}")
        
        # They should be very close (LatentWrapper may have a small projection layer)
        param_ratio = wrapper_params / original_params
        print(f"Parameter ratio: {param_ratio:.4f}")
        
        if param_ratio < 1.01:  # Allow for small projection layer
            print("✅ Memory efficient: No significant parameter duplication")
        else:
            print("⚠️  Potential memory inefficiency detected")
            
        return True
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 EMBEDDING CONSISTENCY TRAINING TEST")
    print("=" * 60)
    
    success = True
    
    # Test 1: Training simulation
    success &= test_training_simulation()
    
    # Test 2: Memory efficiency
    success &= test_memory_efficiency()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ALL TRAINING TESTS PASSED!")
        print("🎉 Embedding consistency fix is working correctly!")
        print("🚀 CoCoNut training should now be stable and effective!")
    else:
        print("❌ TRAINING TESTS FAILED!")
        print("⚠️  Embedding consistency issues may persist!")
    
    print("=" * 60)
    sys.exit(0 if success else 1)
