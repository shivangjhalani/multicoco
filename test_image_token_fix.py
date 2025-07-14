#!/usr/bin/env python3
"""
Test script to verify that the image token count fix works correctly.
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to the path so we can import multicoco as a package
sys.path.insert(0, str(Path(__file__).parent))

from multicoco.image_tokens import get_model_image_token_count, get_tokenizer_image_token_count, validate_image_token_count

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class MockModel:
    """Mock model for testing image token count detection."""
    
    def __init__(self, config_type="internvl", num_tokens=None):
        self.config = MockConfig(config_type, num_tokens)
    
    @property
    def base_model(self):
        return self


class MockConfig:
    """Mock config for testing."""
    
    def __init__(self, config_type="internvl", num_tokens=None):
        self.config_type = config_type
        
        if config_type == "internvl":
            self._name_or_path = "OpenGVLab/InternVL3-1B"
            self.downsample_ratio = 0.25  # Typical value
            if num_tokens:
                self.num_image_token = num_tokens
        elif config_type == "direct":
            self.num_image_token = num_tokens or 784
        elif config_type == "vision_config":
            self.vision_config = MockVisionConfig()
            if num_tokens:
                self.num_image_token = num_tokens


class MockVisionConfig:
    """Mock vision config."""
    
    def __init__(self):
        self.image_size = 448
        self.patch_size = 14


class MockTokenizer:
    """Mock tokenizer with different configurations."""
    
    def __init__(self, has_num_tokens=False, num_tokens=256):
        if has_num_tokens:
            self.model = MockTokenizerModel(num_tokens)
    

class MockTokenizerModel:
    """Mock tokenizer model."""
    
    def __init__(self, num_tokens):
        self.num_image_token = num_tokens


def test_image_token_count_detection():
    """Test various methods of detecting image token counts."""
    
    print("=" * 60)
    print("Testing Image Token Count Detection Methods")
    print("=" * 60)
    
    # Test 1: Direct config.num_image_token
    print("\n1. Testing direct config.num_image_token")
    model1 = MockModel("direct", 784)
    count1 = get_model_image_token_count(model1)
    print(f"Expected: 784, Got: {count1}")
    assert count1 == 784, f"Expected 784, got {count1}"
    print("✓ Direct config detection works")
    
    # Test 2: InternVL calculation
    print("\n2. Testing InternVL downsample calculation")
    model2 = MockModel("internvl")
    count2 = get_model_image_token_count(model2)
    expected2 = int(32 * 32 * 0.25)  # 1024 * 0.25 = 256
    print(f"Expected: {expected2}, Got: {count2}")
    assert count2 == expected2, f"Expected {expected2}, got {count2}"
    print("✓ InternVL calculation works")
    
    # Test 3: Vision config calculation
    print("\n3. Testing vision config calculation")
    model3 = MockModel("vision_config")
    count3 = get_model_image_token_count(model3)
    expected3 = (448 // 14) * (448 // 14)  # 32 * 32 = 1024
    print(f"Expected: {expected3}, Got: {count3}")
    assert count3 == expected3, f"Expected {expected3}, got {count3}"
    print("✓ Vision config calculation works")
    
    # Test 4: Fallback
    print("\n4. Testing fallback value")
    model4 = MockModel("unknown")
    count4 = get_model_image_token_count(model4, fallback=512)
    print(f"Expected: 512, Got: {count4}")
    assert count4 == 512, f"Expected 512, got {count4}"
    print("✓ Fallback works")


def test_tokenizer_detection():
    """Test tokenizer-based detection."""
    
    print("\n" + "=" * 60)
    print("Testing Tokenizer Image Token Detection")
    print("=" * 60)
    
    # Test 1: Tokenizer with num_image_token
    print("\n1. Testing tokenizer with num_image_token")
    tokenizer1 = MockTokenizer(has_num_tokens=True, num_tokens=784)
    count1 = get_tokenizer_image_token_count(tokenizer1)
    print(f"Expected: 784, Got: {count1}")
    assert count1 == 784, f"Expected 784, got {count1}"
    print("✓ Tokenizer detection works")
    
    # Test 2: Tokenizer fallback
    print("\n2. Testing tokenizer fallback")
    tokenizer2 = MockTokenizer(has_num_tokens=False)
    count2 = get_tokenizer_image_token_count(tokenizer2, fallback=256)
    print(f"Expected: 256, Got: {count2}")
    assert count2 == 256, f"Expected 256, got {count2}"
    print("✓ Tokenizer fallback works")


def test_validation():
    """Test validation functions."""
    
    print("\n" + "=" * 60)
    print("Testing Image Token Count Validation")
    print("=" * 60)
    
    # Test 1: Correct count
    print("\n1. Testing correct token count")
    prompt1 = "<img>" + "<IMG_CONTEXT>" * 256 + "</img>Question here"
    result1 = validate_image_token_count(prompt1, None, 256)
    print(f"Expected: True, Got: {result1}")
    assert result1 == True, f"Expected True, got {result1}"
    print("✓ Correct count validation works")
    
    # Test 2: Incorrect count
    print("\n2. Testing incorrect token count")
    prompt2 = "<img>" + "<IMG_CONTEXT>" * 128 + "</img>Question here"
    result2 = validate_image_token_count(prompt2, None, 256)
    print(f"Expected: False, Got: {result2}")
    assert result2 == False, f"Expected False, got {result2}"
    print("✓ Incorrect count detection works")


def test_integration_scenario():
    """Test a realistic integration scenario."""
    
    print("\n" + "=" * 60)
    print("Testing Realistic Integration Scenario")
    print("=" * 60)
    
    # Scenario: InternVL3-1B model with proper token count
    print("\nScenario: InternVL3-1B model setup")
    
    # Create mock model and tokenizer
    model = MockModel("direct", 784)  # Assume we determined the real count is 784
    tokenizer = MockTokenizer(has_num_tokens=True, num_tokens=784)
    
    # Get counts from both sources
    model_count = get_model_image_token_count(model)
    tokenizer_count = get_tokenizer_image_token_count(tokenizer)
    
    print(f"Model reports: {model_count} tokens")
    print(f"Tokenizer reports: {tokenizer_count} tokens")
    
    # They should match
    assert model_count == tokenizer_count, f"Model and tokenizer counts don't match: {model_count} vs {tokenizer_count}"
    print("✓ Model and tokenizer counts match")
    
    # Create prompt with correct count
    img_context = "<IMG_CONTEXT>" * model_count
    prompt = f"<img>{img_context}</img>What do you see in this image?"
    
    print(f"Created prompt with {prompt.count('<IMG_CONTEXT>')} IMG_CONTEXT tokens")
    
    # Validate
    is_valid = validate_image_token_count(prompt, None, model_count)
    assert is_valid, "Prompt validation failed"
    print("✓ Prompt validation passed")
    
    print(f"\n✅ Integration test successful!")
    print(f"   - Model produces {model_count} image tokens")
    print(f"   - Prompt contains {prompt.count('<IMG_CONTEXT>')} IMG_CONTEXT tokens")
    print(f"   - No token count mismatch!")


if __name__ == "__main__":
    print("Testing Image Token Count Fix")
    print("This test verifies that the image token count detection and validation works correctly")
    
    try:
        test_image_token_count_detection()
        test_tokenizer_detection()
        test_validation()
        test_integration_scenario()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The image token count fix is working correctly.")
        print("This should resolve the token count mismatch issue that was causing")
        print("assertion failures and loss of visual information.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
