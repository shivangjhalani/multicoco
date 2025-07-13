#!/usr/bin/env python3
"""
Test script to verify that progressive curriculum preserves image fields.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multicoco.data import create_progressive_latent_dataset

def test_progressive_curriculum_preserves_fields():
    """Test that create_progressive_latent_dataset preserves all original fields."""
    print("Testing progressive curriculum field preservation...")
    
    # Create test data with all expected fields
    base_dataset = [
        {
            'image': 'test_image_1.jpg',
            'question': 'What is in the image?',
            'answer': 'A cat',
            'direct_answer': 'cat',
            'steps': ['I see an animal', 'It has whiskers', 'It must be a cat'],
            'other_field': 'should_be_preserved'
        },
        {
            'image': 'test_image_2.jpg',
            'question': 'What color is it?',
            'answer': 'Blue',
            'direct_answer': 'blue',
            'steps': ['Looking at the color', 'It appears blue'],
            'another_field': 'also_preserved'
        }
    ]
    
    # Test progressive curriculum
    processed_dataset = create_progressive_latent_dataset(
        scheduled_stage=1,
        base_dataset=base_dataset,
        c_thought=4,
        max_latent_stage=3,
        uniform_prob=0.0,
        pad_latent_to_max=False,
        no_cot=False
    )
    
    print(f"Original dataset size: {len(base_dataset)}")
    print(f"Processed dataset size: {len(processed_dataset)}")
    
    # Check that all original fields are preserved
    for i, (original, processed) in enumerate(zip(base_dataset, processed_dataset)):
        print(f"\nChecking sample {i}:")
        
        # Check that all original fields are present
        for key, value in original.items():
            if key not in processed:
                print(f"✗ Missing field '{key}' in processed sample")
                return False
            if processed[key] != value:
                print(f"✗ Field '{key}' changed from '{value}' to '{processed[key]}'")
                return False
            print(f"✓ Field '{key}' preserved: {value}")
        
        # Check that new fields are added
        expected_new_fields = ['reasoning', 'stage', 'n_latent_tokens', 'n_skip_steps']
        for field in expected_new_fields:
            if field not in processed:
                print(f"✗ Missing new field '{field}' in processed sample")
                return False
            print(f"✓ New field '{field}' added: {processed[field]}")
        
        # Specifically check image field is preserved
        if processed['image'] != original['image']:
            print(f"✗ Image field not preserved: {original['image']} != {processed['image']}")
            return False
        print(f"✓ Image field properly preserved: {processed['image']}")
        
        # Check that reasoning field contains latent tokens if stage > 0
        if processed['stage'] > 0:
            reasoning = processed['reasoning']
            if '<|start_latent|>' not in reasoning or '<|end_latent|>' not in reasoning:
                print(f"✗ Reasoning field missing latent tokens: {reasoning}")
                return False
            print(f"✓ Reasoning field contains latent tokens")
        
    print("\n✓ All progressive curriculum field preservation tests passed!")
    return True

if __name__ == "__main__":
    print("Testing Progressive Curriculum Field Preservation")
    print("=" * 60)
    
    success = test_progressive_curriculum_preserves_fields()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Progressive curriculum correctly preserves all fields!")
    else:
        print("✗ Progressive curriculum has field preservation issues.")
    
    sys.exit(0 if success else 1)
