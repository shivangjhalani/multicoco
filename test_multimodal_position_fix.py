#!/usr/bin/env python3
"""
Test script for multimodal position handling in LatentWrapper.
Verifies that latent token injection uses correct source positions when image tokens are present.
"""

import torch
import logging
from multicoco.latent_wrapper import LatentWrapper

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MockModel:
    """Mock model for testing"""
    def __init__(self):
        self.img_context_token_id = 32001  # Mock IMG_CONTEXT token ID
        
    def get_input_embeddings(self):
        return MockEmbedding()

class MockEmbedding:
    """Mock embedding layer"""
    def __init__(self):
        self.weight = torch.randn(50000, 4096)  # Mock embedding weights
        
    def __call__(self, input_ids):
        # Return mock embeddings
        batch_size, seq_len = input_ids.shape
        return torch.randn(batch_size, seq_len, 4096)

class MockTokenizer:
    """Mock tokenizer for testing"""
    def __init__(self):
        self.unk_token_id = 0
        
    def convert_tokens_to_ids(self, token):
        token_map = {
            '<|latent|>': 32000,
            '<|start_latent|>': 32002,
            '<|end_latent|>': 32003,
            '<IMG_CONTEXT>': 32001
        }
        return token_map.get(token, self.unk_token_id)

def test_multimodal_position_calculation():
    """Test the position calculation helper method"""
    print("Testing multimodal position calculation...")
    
    # Create mock components
    model = MockModel()
    tokenizer = MockTokenizer()
    wrapper = LatentWrapper(model, tokenizer)
    
    # Test case 1: No image tokens
    input_ids = torch.tensor([[1, 2, 3, 32002, 32000, 32000, 32003, 4, 5]])
    
    # Test position 4 (first latent token after start marker)
    adjusted_pos = wrapper._calculate_adjusted_source_pos(4, input_ids, 0)
    expected_pos = 3  # 4 - 1 = 3 (no image tokens)
    assert adjusted_pos == expected_pos, f"Expected {expected_pos}, got {adjusted_pos}"
    print(f"✓ No image tokens: position 4 -> {adjusted_pos}")
    
    # Test case 2: Image tokens before latent span
    input_ids_with_img = torch.tensor([[1, 32001, 32001, 32002, 32000, 32000, 32003, 4, 5]])
    
    # Test position 4 (first latent token after start marker)
    adjusted_pos = wrapper._calculate_adjusted_source_pos(4, input_ids_with_img, 0)
    expected_pos = 1  # 4 - 1 - 2 = 1 (2 image tokens before position 4)
    assert adjusted_pos == expected_pos, f"Expected {expected_pos}, got {adjusted_pos}"
    print(f"✓ With image tokens: position 4 -> {adjusted_pos}")
    
    # Test position 5 (second latent token)
    adjusted_pos = wrapper._calculate_adjusted_source_pos(5, input_ids_with_img, 0)
    expected_pos = 2  # 5 - 1 - 2 = 2 (2 image tokens before position 5)
    assert adjusted_pos == expected_pos, f"Expected {expected_pos}, got {adjusted_pos}"
    print(f"✓ With image tokens: position 5 -> {adjusted_pos}")
    
    # Test case 3: Image tokens after latent span (should not affect calculation)
    input_ids_img_after = torch.tensor([[1, 2, 32002, 32000, 32000, 32003, 32001, 32001, 4]])
    
    # Test position 3 (first latent token after start marker)
    adjusted_pos = wrapper._calculate_adjusted_source_pos(3, input_ids_img_after, 0)
    expected_pos = 2  # 3 - 1 = 2 (no image tokens before position 3)
    assert adjusted_pos == expected_pos, f"Expected {expected_pos}, got {adjusted_pos}"
    print(f"✓ Image tokens after latent: position 3 -> {adjusted_pos}")
    
    print("All position calculation tests passed!")

def test_multimodal_latent_injection():
    """Test that latent injection works correctly with image tokens"""
    print("Testing multimodal latent injection...")
    
    # Create mock components
    model = MockModel()
    tokenizer = MockTokenizer()
    wrapper = LatentWrapper(model, tokenizer)
    
    # Create input with image tokens and latent span
    # Format: [text] [IMG] [IMG] [start] [latent] [latent] [end] [text]
    input_ids = torch.tensor([[1, 32001, 32001, 32002, 32000, 32000, 32003, 2]])
    batch_size, seq_len = input_ids.shape
    
    # Create mock last_hidden from previous pass (accounting for no image tokens in hidden states)
    # Hidden states should have length: seq_len - num_image_tokens = 8 - 2 = 6
    hidden_len = seq_len - 2  # Subtract 2 image tokens
    last_hidden = torch.randn(batch_size, hidden_len, 4096)
    
    # Define latent spans
    spans = [[(3, 6)]]  # start=3, end=6 (positions of start and end markers)
    
    try:
        # Test the embedding modification
        modified_embeds = wrapper._build_modified_embeddings_sequential(input_ids, spans, last_hidden)
        
        # Verify the shape is unchanged
        assert modified_embeds.shape == (batch_size, seq_len, 4096), f"Shape mismatch: {modified_embeds.shape}"
        print(f"✓ Modified embeddings shape: {modified_embeds.shape}")
        
        # The positions 4 and 5 should have been modified (latent tokens)
        # These should get hidden states from adjusted positions 1 and 2 respectively
        print("✓ Latent injection completed without errors")
        
    except Exception as e:
        print(f"✗ Error during latent injection: {e}")
        raise
    
    print("Multimodal latent injection test passed!")

def test_edge_cases():
    """Test edge cases for position calculation"""
    print("Testing edge cases...")
    
    model = MockModel()
    tokenizer = MockTokenizer()
    wrapper = LatentWrapper(model, tokenizer)
    
    # Test case: Negative adjusted position (should return 0)
    input_ids = torch.tensor([[32001, 32001, 32001, 32002, 32000]])
    adjusted_pos = wrapper._calculate_adjusted_source_pos(4, input_ids, 0)
    assert adjusted_pos == 0, f"Expected 0 for negative position, got {adjusted_pos}"
    print(f"✓ Negative position handling: {adjusted_pos}")
    
    # Test case: No img_context_token_id (should fall back to simple calculation)
    model.img_context_token_id = None
    wrapper2 = LatentWrapper(model, tokenizer)
    input_ids = torch.tensor([[1, 32001, 32001, 32002, 32000]])
    adjusted_pos = wrapper2._calculate_adjusted_source_pos(4, input_ids, 0)
    expected_pos = 3  # Simple calculation: 4 - 1
    assert adjusted_pos == expected_pos, f"Expected {expected_pos}, got {adjusted_pos}"
    print(f"✓ Fallback to simple calculation: {adjusted_pos}")
    
    print("Edge case tests passed!")

if __name__ == "__main__":
    test_multimodal_position_calculation()
    test_multimodal_latent_injection()
    test_edge_cases()
    print("\n🎉 All multimodal position handling tests passed!")
