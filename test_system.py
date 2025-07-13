#!/usr/bin/env python3
"""
Comprehensive test suite for the MultiCoCo system.
Tests the fixes for multimodal latent handling and ensures nothing is broken.
"""

import os
import sys
import torch
import json
import logging
from typing import Dict, List, Any
from PIL import Image
import tempfile

# Add the project root to path
sys.path.insert(0, os.path.abspath('.'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    try:
        from multicoco.constants import LATENT_TOKEN, START_LATENT_TOKEN, END_LATENT_TOKEN, IMAGE_TOKEN
        from multicoco.data import (
            create_progressive_latent_dataset, 
            _build_reasoning_text,
            _create_chat_formatted_texts,
            SupervisedDataset,
            collate_fn
        )
        from multicoco.latent_wrapper import LatentWrapper
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_reasoning_text_generation():
    """Test the fixed reasoning text generation."""
    print("\nTesting reasoning text generation...")
    
    from multicoco.data import _build_reasoning_text
    from multicoco.constants import LATENT_TOKEN, START_LATENT_TOKEN, END_LATENT_TOKEN
    
    # Test case 1: With latent tokens and remaining steps
    steps = ["First, I need to analyze the image.", "Then, I'll identify key objects.", "Finally, I'll answer the question."]
    reasoning = _build_reasoning_text(total_latent_tokens=6, steps=steps, n_skip_steps=1)
    
    expected_pattern = f"{START_LATENT_TOKEN} {LATENT_TOKEN} {LATENT_TOKEN} {LATENT_TOKEN} {LATENT_TOKEN} {LATENT_TOKEN} {LATENT_TOKEN} {END_LATENT_TOKEN} Then, I'll identify key objects. Finally, I'll answer the question."
    
    if reasoning == expected_pattern:
        print("✓ Reasoning text generation with latent tokens works correctly")
    else:
        print(f"✗ Reasoning text mismatch")
        print(f"Expected: {expected_pattern}")
        print(f"Got: {reasoning}")
        return False
    
    # Test case 2: No latent tokens
    reasoning_no_latent = _build_reasoning_text(total_latent_tokens=0, steps=steps, n_skip_steps=0)
    expected_no_latent = "First, I need to analyze the image. Then, I'll identify key objects. Finally, I'll answer the question."
    
    if reasoning_no_latent == expected_no_latent:
        print("✓ Reasoning text without latent tokens works correctly")
    else:
        print(f"✗ No-latent reasoning text mismatch")
        print(f"Expected: {expected_no_latent}")
        print(f"Got: {reasoning_no_latent}")
        return False
    
    # Test case 3: Only latent tokens, no remaining steps
    reasoning_only_latent = _build_reasoning_text(total_latent_tokens=3, steps=steps, n_skip_steps=100)
    expected_only_latent = f"{START_LATENT_TOKEN} {LATENT_TOKEN} {LATENT_TOKEN} {LATENT_TOKEN} {END_LATENT_TOKEN}"
    
    if reasoning_only_latent == expected_only_latent:
        print("✓ Reasoning text with only latent tokens works correctly")
        return True
    else:
        print(f"✗ Only-latent reasoning text mismatch")
        print(f"Expected: {expected_only_latent}")
        print(f"Got: {reasoning_only_latent}")
        return False

def test_progressive_dataset_creation():
    """Test progressive dataset creation."""
    print("\nTesting progressive dataset creation...")
    
    from multicoco.data import create_progressive_latent_dataset
    
    # Create mock dataset
    base_dataset = [
        {
            "image": "test_image1.jpg",
            "question": "What is in this image?",
            "answer": "A cat",
            "steps": ["I see an animal.", "It has fur and whiskers.", "It's a cat."]
        },
        {
            "image": "test_image2.jpg", 
            "question": "What color is the car?",
            "answer": "Red",
            "steps": ["I need to look at the car.", "The car appears to be red colored."]
        }
    ]
    
    # Test different stages
    for stage in [0, 1, 2]:
        try:
            processed = create_progressive_latent_dataset(
                scheduled_stage=stage,
                base_dataset=base_dataset,
                c_thought=2,
                max_latent_stage=3,
                uniform_prob=0.0,
                pad_latent_to_max=False,
                no_cot=False
            )
            
            if len(processed) != len(base_dataset):
                print(f"✗ Dataset length mismatch for stage {stage}")
                return False
                
            # Check that reasoning field is added
            for item in processed:
                if 'reasoning' not in item:
                    print(f"✗ Missing reasoning field for stage {stage}")
                    return False
                    
            print(f"✓ Stage {stage} dataset creation successful")
            
        except Exception as e:
            print(f"✗ Error in stage {stage} dataset creation: {e}")
            return False
    
    return True

def test_chat_formatting():
    """Test chat text formatting."""
    print("\nTesting chat formatting...")
    
    from multicoco.data import _create_chat_formatted_texts
    from multicoco.constants import IMAGE_TOKEN
    
    batch = [
        {
            "reasoning": "<|start_latent|> <|latent|> <|latent|> <|end_latent|> I can see this is a cat."
        }
    ]
    questions = ["What animal is this?"]
    answers = ["Cat"]
    
    try:
        full_texts, prompts = _create_chat_formatted_texts(batch, questions, answers)
        
        expected_prompt = f'<|im_start|>user\n{IMAGE_TOKEN}\nWhat animal is this?<|im_end|><|im_start|>assistant\n'
        expected_full = f'{expected_prompt}<|start_latent|> <|latent|> <|latent|> <|end_latent|> I can see this is a cat. The answer is Cat'
        
        if prompts[0] == expected_prompt and full_texts[0] == expected_full:
            print("✓ Chat formatting works correctly")
            return True
        else:
            print("✗ Chat formatting mismatch")
            print(f"Expected prompt: {expected_prompt}")
            print(f"Got prompt: {prompts[0]}")
            print(f"Expected full: {expected_full}")
            print(f"Got full: {full_texts[0]}")
            return False
            
    except Exception as e:
        print(f"✗ Error in chat formatting: {e}")
        return False

def create_test_data():
    """Create temporary test data files."""
    print("\nCreating test data...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create test images
    test_images_dir = os.path.join(temp_dir, "images")
    os.makedirs(test_images_dir, exist_ok=True)
    
    # Create simple test images
    for i in range(3):
        img = Image.new('RGB', (64, 64), color=(i*80, i*80, i*80))
        img.save(os.path.join(test_images_dir, f"test_{i}.jpg"))
    
    # Create test dataset JSON
    test_data = [
        {
            "image": "test_0.jpg",
            "question": "What do you see in this image?",
            "answer": "A gray square",
            "steps": ["I see a gray colored square.", "It appears to be uniform in color."]
        },
        {
            "image": "test_1.jpg", 
            "question": "What is the main color?",
            "answer": "Dark gray",
            "steps": ["Looking at the image.", "The dominant color is dark gray."]
        },
        {
            "image": "test_2.jpg",
            "question": "Describe this image.",
            "answer": "Light colored square",
            "steps": ["This appears to be a light colored square shape."]
        }
    ]
    
    data_file = os.path.join(temp_dir, "test_data.json")
    with open(data_file, 'w') as f:
        json.dump(test_data, f)
    
    print(f"✓ Test data created in {temp_dir}")
    return temp_dir, data_file, test_images_dir

def test_dataset_loading():
    """Test dataset loading with real files."""
    print("\nTesting dataset loading...")
    
    from multicoco.data import SupervisedDataset
    
    try:
        temp_dir, data_file, images_dir = create_test_data()
        
        # Test dataset creation
        dataset = SupervisedDataset(data_file, images_dir, test_limit=2)
        
        if len(dataset) != 2:
            print(f"✗ Dataset length incorrect: expected 2, got {len(dataset)}")
            return False
        
        # Test item access
        item = dataset[0]
        required_keys = ['image', 'question', 'answer']
        
        for key in required_keys:
            if key not in item:
                print(f"✗ Missing key {key} in dataset item")
                return False
        
        # Check image is PIL Image
        if not isinstance(item['image'], Image.Image):
            print(f"✗ Image is not PIL Image: {type(item['image'])}")
            return False
        
        print("✓ Dataset loading works correctly")
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ Error in dataset loading: {e}")
        return False

def test_latent_wrapper_basic():
    """Test basic latent wrapper functionality."""
    print("\nTesting latent wrapper basic functionality...")
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        from multicoco.constants import COCONUT_SPECIAL_TOKENS
        
        # This is a basic test to ensure the class can be instantiated
        # We can't test the full functionality without a real model
        print("✓ LatentWrapper can be imported")
        
        # Test span extraction (this doesn't require a model)
        if hasattr(LatentWrapper, '_extract_latent_spans'):
            print("✓ LatentWrapper has required methods")
            return True
        else:
            print("✗ LatentWrapper missing required methods")
            return False
            
    except Exception as e:
        print(f"✗ Error in latent wrapper test: {e}")
        return False

def test_constants():
    """Test that all required constants are defined."""
    print("\nTesting constants...")
    
    try:
        from multicoco.constants import (
            LATENT_TOKEN, START_LATENT_TOKEN, END_LATENT_TOKEN, 
            IMAGE_TOKEN, COCONUT_SPECIAL_TOKENS, DEFAULT_MAX_LENGTH
        )
        
        required_tokens = [LATENT_TOKEN, START_LATENT_TOKEN, END_LATENT_TOKEN, IMAGE_TOKEN]
        
        for token in required_tokens:
            if not isinstance(token, str) or len(token) == 0:
                print(f"✗ Invalid token: {token}")
                return False
        
        print("✓ All constants are properly defined")
        return True
        
    except ImportError as e:
        print(f"✗ Missing constants: {e}")
        return False

def test_error_handling():
    """Test error handling in various scenarios."""
    print("\nTesting error handling...")
    
    from multicoco.data import _build_reasoning_text
    
    try:
        # Test with empty steps
        result = _build_reasoning_text(0, [], 0)
        if result == "":
            print("✓ Empty input handling works")
        else:
            print(f"✗ Empty input handling failed: got '{result}'")
            return False
        
        # Test with negative values (should not crash)
        result = _build_reasoning_text(0, ["step1"], 10)  # n_skip_steps > len(steps)
        if result == "":  # Should skip all steps
            print("✓ Edge case handling works")
        else:
            print(f"✗ Edge case handling failed: got '{result}'")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error in error handling test: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("MULTICOCO SYSTEM TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Constants Test", test_constants),
        ("Reasoning Text Generation", test_reasoning_text_generation),
        ("Progressive Dataset Creation", test_progressive_dataset_creation),
        ("Chat Formatting", test_chat_formatting),
        ("Dataset Loading", test_dataset_loading),
        ("Latent Wrapper Basic", test_latent_wrapper_basic),
        ("Error Handling", test_error_handling),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print(f"{'-' * 40}")
        
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test_name} crashed: {e}")
            failed += 1
    
    print(f"\n{'=' * 60}")
    print("TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! The system is working correctly.")
        return True
    else:
        print(f"\n❌ {failed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
