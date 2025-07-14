#!/usr/bin/env python3
"""
Test script to understand the multimodal sequence length and compute range issue.
"""

import torch
import logging
import sys
import os

# Add project paths
sys.path.append('/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco')

# Set up logging to see INFO messages
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_compute_range_debug():
    """Test the compute range calculation with a realistic sequence"""
    
    # Mock model and tokenizer for testing
    class MockModel:
        def __init__(self):
            # Create a parameter so device property works
            self.dummy_param = torch.nn.Parameter(torch.randn(1))
            
        def parameters(self):
            yield self.dummy_param
    
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 2
            
        def convert_tokens_to_ids(self, token):
            token_map = {
                '<|latent|>': 50000,
                '<|start_latent|>': 50001, 
                '<|end_latent|>': 50002,
                '<IMG_CONTEXT>': 64003
            }
            return token_map.get(token, 1)  # Return 1 (unk) for unknown tokens
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        
        # Create wrapper with mock objects
        mock_model = MockModel()
        mock_tokenizer = MockTokenizer()
        
        # Override embedding to avoid errors
        mock_embedding = torch.nn.Embedding(100000, 64)
        wrapper = LatentWrapper(mock_model, mock_tokenizer)
        wrapper._embedding_ref = mock_embedding
        
        # Create a test sequence that matches the training scenario
        # Simulate a long sequence with latent tokens near the end
        seq_len = 320  # Similar to training
        input_ids = torch.arange(seq_len).unsqueeze(0)  # [1, 320]
        
        # Add latent spans at positions similar to the training warnings
        # Replace some positions with latent token IDs
        input_ids[0, 295] = 50001  # start_latent
        input_ids[0, 296] = 50000  # latent 
        input_ids[0, 297] = 50000  # latent
        input_ids[0, 298] = 50000  # latent
        input_ids[0, 299] = 50000  # latent
        input_ids[0, 300] = 50002  # end_latent
        
        input_ids[0, 305] = 50001  # start_latent
        input_ids[0, 306] = 50000  # latent
        input_ids[0, 307] = 50000  # latent
        input_ids[0, 308] = 50002  # end_latent
        
        print(f"Input sequence length: {input_ids.shape[1]}")
        print(f"Latent spans at: 295-300 and 305-308")
        
        # Extract spans like the wrapper does
        spans = wrapper._extract_latent_spans(input_ids)
        print(f"Extracted spans: {spans}")
        
        # Convert to latent lists
        latent_lists = wrapper._convert_spans_to_latent_lists(spans, input_ids.shape[1])
        print(f"Latent lists: {latent_lists}")
        
        # Calculate max latents
        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0
        print(f"Max latents: {max_n_latents}")
        
        # Calculate earliest latent position (this is where the issue might be)
        if max_n_latents > 0:
            earliest_latent_pos = min([pos for span_list in spans for start, end in span_list for pos in range(start + 1, end)]) if any(spans) else input_ids.shape[1]
            print(f"Earliest latent position: {earliest_latent_pos}")
            
            # This is the critical compute range
            compute_range = (0, min(earliest_latent_pos, input_ids.shape[1]))
            print(f"Initial compute range: {compute_range}")
            
            # Simulate the passes
            print(f"\nSimulating {max_n_latents} passes:")
            next_compute_range = compute_range
            
            for pass_idx in range(max_n_latents):
                print(f"\nPass {pass_idx}:")
                print(f"  Compute range: {next_compute_range}")
                print(f"  Hidden states would have shape: [1, {next_compute_range[1] - next_compute_range[0]}, hidden_dim]")
                print(f"  Hidden states offset: {next_compute_range[0]}")
                
                # Find latent tokens for this pass
                filling_indices = [
                    (instance_idx, mask_list[pass_idx])
                    for instance_idx, mask_list in enumerate(latent_lists)
                    if len(mask_list) > pass_idx
                ]
                
                print(f"  Latent tokens to inject: {filling_indices}")
                
                # Check if injections would be valid
                for instance_idx, token_idx in filling_indices:
                    source_pos = token_idx - 1 - next_compute_range[0]  # This is the critical calculation
                    hidden_states_len = next_compute_range[1] - next_compute_range[0]
                    
                    print(f"    Latent {token_idx}: source_pos={source_pos}, valid_range=[0, {hidden_states_len})")
                    if 0 <= source_pos < hidden_states_len:
                        print(f"      ✓ VALID injection")
                    else:
                        print(f"      ❌ INVALID injection - this would cause the warning!")
                
                # Update compute range for next pass
                next_compute_range = (
                    next_compute_range[1],
                    input_ids.shape[1] if pass_idx + 1 >= max_n_latents else next_compute_range[1] + 1
                )
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing compute range calculation for multimodal latent injection...")
    success = test_compute_range_debug()
    if success:
        print("\n✅ Test completed successfully")
    else:
        print("\n❌ Test failed")
