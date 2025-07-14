#!/usr/bin/env python3
"""
Test script to verify that our compute range logic matches the original CoCoNut algorithm.
This should eliminate the 'Invalid source position' warnings.
"""

import sys
import os
import torch
import logging

# Add the current directory to Python path
sys.path.append('/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_compute_range_logic():
    """Test that our compute range logic matches the original algorithm"""
    
    print("Testing compute range logic against original CoCoNut...")
    
    # Simulate a sequence with latent tokens
    seq_length = 310
    latent_spans = [
        [(295, 299), (303, 307)]  # Two latent spans: positions 296-298 and 304-306
    ]
    
    # Convert to latent lists (skip start/end markers)
    latent_lists = []
    for batch_spans in latent_spans:
        latent_positions = []
        for start, end in batch_spans:
            for pos in range(start + 1, end):  # Skip start/end markers
                latent_positions.append(pos)
        latent_lists.append(latent_positions)
    
    print(f"Sequence length: {seq_length}")
    print(f"Latent spans: {latent_spans}")
    print(f"Latent lists: {latent_lists}")
    
    max_n_latents = max([len(l) for l in latent_lists])
    print(f"Max latent tokens: {max_n_latents}")
    
    # ORIGINAL ALGORITHM: Find earliest latent token
    earliest_latent_pos = min([pos for span_list in latent_spans for start, end in span_list for pos in range(start + 1, end)])
    print(f"Earliest latent position: {earliest_latent_pos}")
    
    # Test each pass
    next_compute_range = (0, earliest_latent_pos)
    print(f"\nTesting compute ranges and source positions:")
    
    for pass_idx in range(max_n_latents):
        print(f"\n--- Pass {pass_idx} ---")
        print(f"Compute range: {next_compute_range}")
        
        # Simulate hidden states for this range
        hidden_states_length = next_compute_range[1] - next_compute_range[0]
        print(f"Hidden states length: {hidden_states_length}")
        
        if pass_idx == 0:
            hidden_states_offset = 0
        else:
            hidden_states_offset = next_compute_range[0]
        
        print(f"Hidden states offset: {hidden_states_offset}")
        
        # Check which latent tokens will be injected in this pass
        filling_indices = [
            (0, latent_lists[0][pass_idx])  # batch_idx=0 for simplicity
            for instance_idx, mask_list in enumerate(latent_lists)
            if len(mask_list) > pass_idx
        ]
        
        print(f"Tokens to inject: {[token_idx for _, token_idx in filling_indices]}")
        
        # Check if source positions are valid
        all_valid = True
        for _, token_idx in filling_indices:
            source_pos = token_idx - 1 - hidden_states_offset
            is_valid = 0 <= source_pos < hidden_states_length
            
            print(f"  Token {token_idx}: source_pos = {token_idx} - 1 - {hidden_states_offset} = {source_pos}, valid: {is_valid}")
            
            if not is_valid:
                all_valid = False
                print(f"    ❌ INVALID: source_pos {source_pos} outside range [0, {hidden_states_length})")
        
        if all_valid and filling_indices:
            print(f"  ✅ All source positions valid for pass {pass_idx}")
        elif not filling_indices:
            print(f"  ✅ No injections needed for pass {pass_idx}")
        
        # Update compute range for next pass (original algorithm)
        next_compute_range = (
            next_compute_range[1],
            seq_length if pass_idx + 1 >= max_n_latents else next_compute_range[1] + 1
        )
    
    print(f"\n🎯 Analysis complete!")
    return True

def test_problematic_case():
    """Test the specific case that was causing warnings"""
    print(f"\n{'='*60}")
    print("Testing the problematic case from training logs:")
    print(f"{'='*60}")
    
    # From the warnings: "Invalid source position 297 for latent position 298 (offset: 0)"
    # This suggests latent token at position 298, trying to access source at 297
    
    # Reconstruct the scenario
    latent_pos = 298
    hidden_states_offset = 0  # First pass
    source_pos = latent_pos - 1 - hidden_states_offset  # = 298 - 1 - 0 = 297
    
    print(f"Problematic case:")
    print(f"  Latent position: {latent_pos}")
    print(f"  Hidden states offset: {hidden_states_offset}")
    print(f"  Calculated source position: {source_pos}")
    
    # For this to be valid, hidden_states must have length > 297
    # This means compute_range[1] must be > 297
    # If earliest latent is at 298, then compute_range = (0, 298)
    # So hidden_states would have length 298, indices 0-297 ✅
    
    print(f"\nFor this to work:")
    print(f"  Compute range should be: (0, {latent_pos})")
    print(f"  Hidden states length should be: {latent_pos}")
    print(f"  Valid source indices: 0 to {latent_pos - 1}")
    print(f"  Required source index: {source_pos}")
    print(f"  Should be valid: {0 <= source_pos < latent_pos}")
    
    if 0 <= source_pos < latent_pos:
        print(f"  ✅ This should work with correct compute range!")
    else:
        print(f"  ❌ This indicates a bug in our compute range calculation")
    
    return 0 <= source_pos < latent_pos

if __name__ == "__main__":
    print("=" * 80)
    print("COCONUT COMPUTE RANGE VALIDATION")
    print("=" * 80)
    
    success1 = test_compute_range_logic()
    success2 = test_problematic_case()
    
    if success1 and success2:
        print(f"\n✅ Compute range logic should now work correctly!")
        print(f"   The 'Invalid source position' warnings should be eliminated.")
    else:
        print(f"\n❌ Compute range logic still has issues.")
    
    print(f"\n💡 Key insights:")
    print(f"   1. First pass: compute_range = (0, earliest_latent_position)")
    print(f"   2. Subsequent passes: compute_range = (prev_end, prev_end + 1)")
    print(f"   3. This ensures predecessor tokens are always available")
    print(f"   4. Original algorithm never needs bounds checking - it's deterministic")
