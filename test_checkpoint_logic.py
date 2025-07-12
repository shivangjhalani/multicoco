#!/usr/bin/env python3
"""
Test script to verify checkpoint logic issues in the trainer.
"""

def test_checkpoint_numbering_consistency():
    """Test the checkpoint numbering logic to identify inconsistencies."""
    print("=== Testing Checkpoint Numbering Logic ===")
    
    # Simulate the current logic
    print("\n1. Current checkpoint saving logic:")
    for epoch in range(3):  # 0-indexed epochs from training loop
        checkpoint_name = f"epoch-{epoch}"
        display_epoch = epoch + 1
        print(f"  Training epoch {epoch} (displayed as epoch {display_epoch}) -> saves as '{checkpoint_name}'")
    
    print("\n2. Current checkpoint loading logic:")
    saved_checkpoints = ["epoch-0", "epoch-1", "epoch-2"]
    for checkpoint in saved_checkpoints:
        epoch_num = int(checkpoint.split('-')[1])
        next_epoch = epoch_num + 1
        print(f"  Loading '{checkpoint}' -> extracts epoch_num={epoch_num} -> returns next_epoch={next_epoch}")
    
    print("\n3. Issue Analysis:")
    print("  - If training completes 3 epochs (0, 1, 2), checkpoints saved: epoch-0, epoch-1, epoch-2")
    print("  - If resuming from epoch-2, next_epoch = 2 + 1 = 3")
    print("  - But training loop expects 0-based epochs, so it would try to resume at epoch 3")
    print("  - This creates an off-by-one error!")

def test_checkpoint_resume_scenarios():
    """Test various resume scenarios."""
    print("\n=== Testing Resume Scenarios ===")
    
    scenarios = [
        ("epoch-0", "After 1st epoch training"),
        ("epoch-1", "After 2nd epoch training"),  
        ("epoch-2", "After 3rd epoch training"),
    ]
    
    for checkpoint, description in scenarios:
        epoch_num = int(checkpoint.split('-')[1])
        next_epoch = epoch_num + 1
        print(f"\nScenario: {description}")
        print(f"  Checkpoint: {checkpoint}")
        print(f"  Current logic returns next_epoch: {next_epoch}")
        print(f"  Training loop will start from: range({next_epoch}, num_epochs)")
        
        # Show the problem
        if checkpoint == "epoch-2":
            print(f"  ❌ PROBLEM: If num_epochs=3, range(3,3) is empty - no training!")

if __name__ == "__main__":
    test_checkpoint_numbering_consistency()
    test_checkpoint_resume_scenarios()
