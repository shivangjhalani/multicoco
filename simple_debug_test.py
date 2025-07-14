#!/usr/bin/env python3
"""
Simple test to understand the actual compute range issue without complex mocks.
"""

import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_training_warnings():
    """
    Analyze the actual warnings from training to understand the root cause.
    
    From the training log:
    WARNING - Pass 0: Invalid source position 297 for latent position 298 (offset: 0)
    WARNING - Pass 1: Invalid source position 3 for latent position 299 (offset: 295)
    
    This tells us:
    1. Pass 0: trying to inject at latent_pos=298, source_pos=297, offset=0
       - source_pos = 298 - 1 - 0 = 297
       - hidden_states.shape[1] must be <= 297 (since 297 is out of bounds)
    
    2. Pass 1: trying to inject at latent_pos=299, source_pos=3, offset=295  
       - source_pos = 299 - 1 - 295 = 3
       - hidden_states.shape[1] must be <= 3 (since 3 is out of bounds)
    """
    
    print("=== ANALYSIS OF TRAINING WARNINGS ===")
    print()
    
    # Scenario 1: Pass 0
    print("Pass 0 Analysis:")
    print("  Warning: Invalid source position 297 for latent position 298 (offset: 0)")
    print("  Formula: source_pos = token_idx - 1 - hidden_states_offset")
    print("  Calculation: source_pos = 298 - 1 - 0 = 297")
    print("  Issue: hidden_states.shape[1] <= 297, so position 297 is out of bounds")
    print("  This means the first forward pass didn't include token 298!")
    print()
    
    # Scenario 2: Pass 1  
    print("Pass 1 Analysis:")
    print("  Warning: Invalid source position 3 for latent position 299 (offset: 295)")
    print("  Formula: source_pos = token_idx - 1 - hidden_states_offset")
    print("  Calculation: source_pos = 299 - 1 - 295 = 3")
    print("  Issue: hidden_states.shape[1] <= 3, so position 3 is out of bounds")
    print("  This means the second forward pass only processed 4 tokens or less!")
    print()
    
    print("=== ROOT CAUSE IDENTIFIED ===")
    print("The compute range calculation is wrong!")
    print()
    print("Current logic:")
    print("  1. Find earliest_latent_pos (e.g., 298)")
    print("  2. first_compute_range = (0, earliest_latent_pos) = (0, 298)")
    print("  3. This gives hidden_states.shape[1] = 298")
    print("  4. But we need position 297 (298-1) to inject into position 298")
    print("  5. Position 297 is out of bounds in a tensor of length 298!")
    print()
    print("SOLUTION:")
    print("  The first compute range should include one extra token!")
    print("  first_compute_range = (0, earliest_latent_pos + 1)")
    print("  This ensures we have the predecessor token available for injection.")
    print()

def demonstrate_fix():
    """Demonstrate the simple fix needed"""
    print("=== DEMONSTRATING THE FIX ===")
    print()
    
    # Simulate the problematic scenario
    earliest_latent_pos = 298
    
    print("Current (BROKEN) logic:")
    current_range = (0, earliest_latent_pos)
    print(f"  earliest_latent_pos = {earliest_latent_pos}")
    print(f"  compute_range = {current_range}")
    print(f"  hidden_states.shape[1] = {current_range[1] - current_range[0]} = {current_range[1]}")
    print(f"  To inject at pos {earliest_latent_pos}, need source_pos = {earliest_latent_pos - 1}")
    print(f"  But max valid index in hidden_states = {current_range[1] - 1}")
    print(f"  Result: position {earliest_latent_pos - 1} is OUT OF BOUNDS!")
    print()
    
    print("Fixed (CORRECT) logic:")
    fixed_range = (0, earliest_latent_pos + 1)
    print(f"  earliest_latent_pos = {earliest_latent_pos}")
    print(f"  compute_range = {fixed_range}")
    print(f"  hidden_states.shape[1] = {fixed_range[1] - fixed_range[0]} = {fixed_range[1]}")
    print(f"  To inject at pos {earliest_latent_pos}, need source_pos = {earliest_latent_pos - 1}")
    print(f"  Max valid index in hidden_states = {fixed_range[1] - 1} = {fixed_range[1] - 1}")
    print(f"  Result: position {earliest_latent_pos - 1} is VALID!")
    print()
    
    print("✅ Simple fix: Change compute range from (0, earliest_latent_pos) to (0, earliest_latent_pos + 1)")

if __name__ == "__main__":
    print("Simple Debug Test: Understanding Compute Range Issue")
    print("=" * 60)
    print()
    
    analyze_training_warnings()
    demonstrate_fix()
    
    print("=" * 60)
    print("CONCLUSION: The fix is simple - add +1 to the first compute range!")
    print("This is a classic off-by-one error in array indexing.")
