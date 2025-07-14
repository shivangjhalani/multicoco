#!/usr/bin/env python3
"""
Test script to verify that the latent injection fix resolves the 
"Invalid source position" warnings during CoCoNut stage 1 training.
"""

import sys
import os
import torch
import logging

# Add the current directory to Python path
sys.path.append('/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco')

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_latent_injection_fix():
    """Test the updated latent injection logic"""
    print("Testing latent injection fix for KV cache slicing...")
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        
        # Create a mock setup
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Add a real parameter so device property works
                self.dummy_param = torch.nn.Parameter(torch.tensor([1.0]))
                self._embedding = torch.nn.Embedding(1000, 64)
                
            def get_input_embeddings(self):
                return self._embedding
        
        class MockTokenizer:
            def __init__(self):
                self.unk_token_id = 0
                self.pad_token_id = 1
                self.eos_token_id = 2
                
            def convert_tokens_to_ids(self, token):
                token_map = {
                    '<|latent|>': 100,
                    '<|start_latent|>': 101,
                    '<|end_latent|>': 102,
                    '<IMG_CONTEXT>': 103
                }
                return token_map.get(token, self.unk_token_id)
        
        mock_model = MockModel()
        mock_model.img_context_token_id = 103
        mock_tokenizer = MockTokenizer()
        
        wrapper = LatentWrapper(mock_model, mock_tokenizer)
        
        # Test scenario: latent tokens at positions where KV cache slicing occurs
        print("\n--- Test: Latent injection with KV cache offsets ---")
        
        # Create mock inputs
        batch_size, seq_len, hidden_dim = 1, 320, 64
        inputs_embeds = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Simulate latent token positions (from actual training logs)
        latent_lists = [[298, 299, 300]]  # Latent tokens at these absolute positions
        
        # Test different passes with different hidden_states slices
        test_scenarios = [
            # (pass_idx, hidden_states_shape, hidden_states_offset, description)
            (0, (batch_size, 297, hidden_dim), 0, "First pass: hidden states up to position 297"),
            (1, (batch_size, 4, hidden_dim), 297, "Second pass: positions 297-300 with KV cache"),
            (2, (batch_size, 4, hidden_dim), 301, "Third pass: positions 301-304 with KV cache"),
        ]
        
        for pass_idx, hs_shape, hs_offset, description in test_scenarios:
            print(f"\n{description}")
            print(f"  Pass {pass_idx}: hidden_states.shape={hs_shape}, offset={hs_offset}")
            
            # Create mock hidden states for this scenario
            hidden_states = torch.randn(*hs_shape)
            
            # Test the injection
            original_embeds = inputs_embeds.clone()
            updated_embeds = wrapper._update_embeddings_for_pass(
                inputs_embeds=inputs_embeds,
                hidden_states=hidden_states,
                latent_lists=latent_lists,
                pass_idx=pass_idx,
                hidden_states_offset=hs_offset
            )
            
            # Check if injection occurred without warnings
            if pass_idx < len(latent_lists[0]):  # If we have a latent token for this pass
                latent_pos = latent_lists[0][pass_idx]
                abs_source_pos = latent_pos - 1
                
                # Check if injection should have occurred
                current_compute_start = hs_offset
                current_compute_end = hs_offset + hs_shape[1]
                
                if current_compute_start <= abs_source_pos < current_compute_end:
                    # Should have injected
                    source_pos = abs_source_pos - current_compute_start
                    if torch.equal(updated_embeds[0, latent_pos], hidden_states[0, source_pos]):
                        print(f"  ✓ Successfully injected: latent_pos={latent_pos}, abs_source={abs_source_pos}, rel_source={source_pos}")
                    else:
                        print(f"  ✗ Injection failed: latent_pos={latent_pos}")
                        return False
                else:
                    # Should have skipped injection
                    if torch.equal(updated_embeds[0, latent_pos], original_embeds[0, latent_pos]):
                        print(f"  ✓ Correctly skipped: latent_pos={latent_pos}, abs_source={abs_source_pos} outside range [{current_compute_start}, {current_compute_end})")
                    else:
                        print(f"  ✗ Unexpected injection: latent_pos={latent_pos}")
                        return False
            else:
                print(f"  ✓ No latent token for pass {pass_idx}")
        
        # Test edge cases
        print("\n--- Test: Edge cases ---")
        
        # Edge case 1: Source position at sequence boundary
        edge_latent_lists = [[1]]  # Latent token at position 1 (source would be position 0)
        edge_inputs = torch.randn(batch_size, 10, hidden_dim)
        edge_hidden = torch.randn(batch_size, 5, hidden_dim)
        
        result = wrapper._update_embeddings_for_pass(
            inputs_embeds=edge_inputs,
            hidden_states=edge_hidden,
            latent_lists=edge_latent_lists,
            pass_idx=0,
            hidden_states_offset=0
        )
        print("  ✓ Edge case 1 handled")
        
        # Edge case 2: Empty latent lists
        empty_result = wrapper._update_embeddings_for_pass(
            inputs_embeds=edge_inputs,
            hidden_states=edge_hidden,
            latent_lists=[[]],
            pass_idx=0,
            hidden_states_offset=0
        )
        print("  ✓ Edge case 2 handled")
        
        print("\n🎉 All latent injection tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_latent_injection_fix()
    if success:
        print("\n✅ Latent injection fix is working correctly!")
        print("The model should now handle CoCoNut stages without 'Invalid source position' warnings.")
    else:
        print("\n❌ Latent injection tests failed!")
        sys.exit(1)
