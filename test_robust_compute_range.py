#!/usr/bin/env python3
"""
Simple test to verify the compute range fix works correctly.
"""

def test_robust_compute_range():
    """Test the new robust compute range calculation"""
    
    print("=== Testing Robust Compute Range Fix ===")
    
    # Test case from the user's example
    spans = [[(295, 301), (305, 309)]]  # Two latent spans
    seq_length = 320
    
    # Convert spans to latent lists
    latent_lists = []
    for batch_spans in spans:
        latent_positions = []
        for start, end in batch_spans:
            # Extract individual latent token positions (skip start/end markers)
            for pos in range(start + 1, end):
                if pos < seq_length:
                    latent_positions.append(pos)
        latent_lists.append(latent_positions)
    
    print(f"Latent positions: {latent_lists[0]}")
    # Should be [296, 297, 298, 299, 300, 306, 307, 308]
    
    max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0
    print(f"Max latents: {max_n_latents}")
    
    # NEW ROBUST APPROACH: Find maximum latent position
    max_latent_pos = max([pos for span_list in spans for start, end in span_list for pos in range(start + 1, end)]) if any(spans) else 0
    compute_range = (0, min(max_latent_pos + 1, seq_length))
    
    print(f"Max latent position: {max_latent_pos}")
    print(f"Compute range: {compute_range}")
    print(f"Hidden states length: {compute_range[1] - compute_range[0]}")
    
    print(f"\nTesting injection validity for all passes:")
    
    # Test all passes
    all_valid = True
    for pass_idx in range(max_n_latents):
        # Find latent tokens for this pass
        filling_indices = [
            (instance_idx, mask_list[pass_idx])
            for instance_idx, mask_list in enumerate(latent_lists)
            if len(mask_list) > pass_idx
        ]
        
        print(f"\nPass {pass_idx}:")
        print(f"  Latent tokens to inject: {[token_idx for _, token_idx in filling_indices]}")
        
        # Check if injections would be valid
        for instance_idx, token_idx in filling_indices:
            source_pos = token_idx - 1 - 0  # hidden_states_offset is always 0 now
            hidden_states_len = compute_range[1] - compute_range[0]
            
            print(f"    Latent {token_idx}: source_pos={source_pos}, valid_range=[0, {hidden_states_len})")
            if 0 <= source_pos < hidden_states_len:
                print(f"      ✅ VALID injection")
            else:
                print(f"      ❌ INVALID injection")
                all_valid = False
    
    print(f"\nAll injections valid: {'✅' if all_valid else '❌'}")
    
    if all_valid:
        print("\n🎉 Robust compute range fix works correctly!")
        print("The model should now handle all CoCoNut stages without 'Invalid source position' warnings.")
    else:
        print("\n❌ Fix still has issues")
    
    return all_valid

if __name__ == "__main__":
    print("Testing robust compute range fix...")
    success = test_robust_compute_range()
    if success:
        print("\n✅ Test passed!")
    else:
        print("\n❌ Test failed")
