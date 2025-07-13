#!/usr/bin/env python3
"""
Simple test to verify our token consistency fixes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multicoco.constants import IMAGE_TOKEN, IMG_CONTEXT_TOKEN, PROMPT_TOKENS

def test_token_consistency():
    """Test that image tokens are consistent"""
    print("Testing token consistency...")
    
    # Test 1: IMAGE_TOKEN and IMG_CONTEXT_TOKEN should be the same
    assert IMAGE_TOKEN == IMG_CONTEXT_TOKEN, f"Token mismatch: IMAGE_TOKEN='{IMAGE_TOKEN}', IMG_CONTEXT_TOKEN='{IMG_CONTEXT_TOKEN}'"
    print(f"✓ Image tokens are consistent: {IMAGE_TOKEN}")
    
    # Test 2: IMAGE_TOKEN should be in PROMPT_TOKENS
    assert IMAGE_TOKEN in PROMPT_TOKENS, f"IMAGE_TOKEN '{IMAGE_TOKEN}' not in PROMPT_TOKENS: {PROMPT_TOKENS}"
    print(f"✓ IMAGE_TOKEN is in PROMPT_TOKENS: {PROMPT_TOKENS}")
    
    # Test 3: All expected tokens are in PROMPT_TOKENS
    expected_tokens = ['<|im_start|>', '<|im_end|>', '<img>']
    for token in expected_tokens:
        assert token in PROMPT_TOKENS, f"Expected token '{token}' not in PROMPT_TOKENS: {PROMPT_TOKENS}"
    print(f"✓ All expected prompt tokens are present")
    
    print("All token consistency tests passed! ✅")

if __name__ == "__main__":
    test_token_consistency()
