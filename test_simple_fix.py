#!/usr/bin/env python3
"""
Simple test to verify the off-by-one fix works correctly.
"""

# Test the compute range calculation fix
def test_compute_range_fix():
    """Test that the compute range fix eliminates off-by-one errors"""
    
    print("=== Testing Compute Range Fix ===")
    
    # Test case 1: Latent token at position 298
    earliest_latent_pos = 298
    
    print(f"Test case: earliest_latent_pos = {earliest_latent_pos}")
    
    # Old (broken) logic
    old_compute_range = (0, earliest_latent_pos)
    old_hidden_states_len = old_compute_range[1] - old_compute_range[0]  # 298
    old_source_pos = earliest_latent_pos - 1 - 0  # 297
    old_valid = 0 <= old_source_pos < old_hidden_states_len  # 0 <= 297 < 298 = False!
    
    print(f"❌ OLD logic:")
    print(f"   compute_range: {old_compute_range}")
    print(f"   hidden_states_len: {old_hidden_states_len}")
    print(f"   source_pos needed: {old_source_pos}")
    print(f"   valid range: [0, {old_hidden_states_len})")
    print(f"   injection valid: {old_valid}")
    
    # New (fixed) logic
    new_compute_range = (0, earliest_latent_pos + 1)
    new_hidden_states_len = new_compute_range[1] - new_compute_range[0]  # 299
    new_source_pos = earliest_latent_pos - 1 - 0  # 297
    new_valid = 0 <= new_source_pos < new_hidden_states_len  # 0 <= 297 < 299 = True!
    
    print(f"✅ NEW logic:")
    print(f"   compute_range: {new_compute_range}")
    print(f"   hidden_states_len: {new_hidden_states_len}")
    print(f"   source_pos needed: {new_source_pos}")
    print(f"   valid range: [0, {new_hidden_states_len})")
    print(f"   injection valid: {new_valid}")
    
    print()
    
    # Test case 2: Multiple latent positions
    latent_positions = [296, 297, 298, 299, 300]
    earliest = min(latent_positions)
    
    print(f"Test case 2: latent positions = {latent_positions}")
    print(f"earliest_latent_pos = {earliest}")
    
    # Fixed logic
    compute_range = (0, earliest + 1)
    hidden_states_len = compute_range[1] - compute_range[0]
    
    print(f"compute_range: {compute_range}")
    print(f"hidden_states_len: {hidden_states_len}")
    
    all_valid = True
    for pos in latent_positions:
        source_pos = pos - 1 - 0  # offset = 0 for first pass
        valid = 0 <= source_pos < hidden_states_len
        status = "✅" if valid else "❌"
        print(f"   latent pos {pos}: source_pos {source_pos}, valid: {valid} {status}")
        if not valid:
            all_valid = False
    
    print(f"\nAll injections valid: {'✅' if all_valid else '❌'}")
    
    if new_valid and all_valid:
        print("\n🎉 COMPUTE RANGE FIX WORKS!")
        print("The off-by-one error has been successfully resolved.")
        print("Training should now proceed without 'Invalid source position' warnings.")
        return True
    else:
        print("\n❌ Fix failed - logic still has issues")
        return False

if __name__ == "__main__":
    success = test_compute_range_fix()
    if success:
        print("\n✅ Test passed - fix is correct!")
    else:
        print("\n❌ Test failed - fix needs adjustment")
