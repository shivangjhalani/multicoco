#!/usr/bin/env python3
"""
Test script to verify the latent token injection algorithm fix.

This script tests that each latent token receives the hidden state from its 
immediate predecessor (pos-1), not from a fixed start-1 position.
"""

import torch
import torch.nn as nn
import sys
import os

# Add multicoco to path
sys.path.insert(0, '/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco')

from multicoco.latent_wrapper import LatentWrapper
from multicoco.constants import COCONUT_SPECIAL_TOKENS

class MockTokenizer:
    """Mock tokenizer for testing"""
    def __init__(self):
        self.eos_token_id = 2
        self.pad_token_id = 0
        # Create token mappings
        self.token_to_id = {
            '<|latent|>': 100,
            '<|start_latent|>': 101, 
            '<|end_latent|>': 102,
            '<IMG_CONTEXT>': 103
        }
        
    def convert_tokens_to_ids(self, token):
        return self.token_to_id.get(token, 999)  # Return 999 for unknown tokens

class MockEmbedding(nn.Module):
    """Mock embedding layer"""
    def __init__(self, vocab_size=1000, embed_dim=512):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, embed_dim))
        self.num_embeddings = vocab_size
        self.embedding_dim = embed_dim
        
    def forward(self, input_ids):
        return torch.embedding(self.weight, input_ids)

class MockModel(nn.Module):
    """Mock base model for testing"""
    def __init__(self):
        super().__init__()
        self.embedding = MockEmbedding()
        self.img_context_token_id = 103
        
    def get_input_embeddings(self):
        return self.embedding

def test_latent_injection_individual_tokens():
    """Test that each latent token gets hidden state from its immediate predecessor"""
    print("Testing individual latent token injection...")
    
    # Create mock components
    tokenizer = MockTokenizer()
    base_model = MockModel()
    
    # Create LatentWrapper
    wrapper = LatentWrapper(base_model, tokenizer)
    
    # Create test input with a latent span
    # Input: [regular_token, start_latent, latent1, latent2, latent3, end_latent, regular_token]
    input_ids = torch.tensor([[10, 101, 100, 100, 100, 102, 11]], dtype=torch.long)
    
    # Create mock hidden states with distinct values for each position
    batch_size, seq_len = input_ids.shape
    hidden_dim = 512
    last_hidden = torch.zeros(batch_size, seq_len, hidden_dim)
    
    # Set distinct hidden states for each position
    for pos in range(seq_len):
        last_hidden[0, pos, :] = float(pos + 1)  # Position 0 gets value 1.0, position 1 gets 2.0, etc.
    
    # Extract latent spans
    spans = wrapper._extract_latent_spans(input_ids)
    print(f"Extracted spans: {spans}")
    
    # Apply latent injection
    modified_embeds = wrapper._build_modified_embeddings_sequential(input_ids, spans, last_hidden)
    
    # Verify results
    print("Verification:")
    print(f"Input sequence: {input_ids[0].tolist()}")
    print("Expected behavior: Each latent token should get hidden state from pos-1")
    
    # Check each latent token position (positions 2, 3, 4 in our example)
    if len(spans[0]) > 0:
        start, end = spans[0][0]  # First span
        print(f"Span: start={start}, end={end}")
        
        for pos in range(start + 1, end):  # Skip start/end markers
            if pos < modified_embeds.shape[1]:
                # Get the embedding value (should match hidden state from pos-1)
                embedding_value = modified_embeds[0, pos, 0].item()  # First dimension value
                expected_value = float(pos)  # Since hidden states were set to pos+1, but we get from pos-1
                
                print(f"  Token at pos {pos}: embedding[0]={embedding_value:.1f}, expected from pos-1={expected_value:.1f}")
                
                # Check if injection worked correctly
                if abs(embedding_value - expected_value) < 0.1:
                    print(f"    ✓ CORRECT: Token at pos {pos} got hidden state from pos {pos-1}")
                else:
                    print(f"    ✗ ERROR: Token at pos {pos} should have value {expected_value:.1f} but got {embedding_value:.1f}")
                    return False
    
    print("✓ Individual latent token injection test PASSED")
    return True

def test_multiple_spans():
    """Test with multiple latent spans to ensure each is handled independently"""
    print("\nTesting multiple latent spans...")
    
    # Create mock components
    tokenizer = MockTokenizer()
    base_model = MockModel()
    wrapper = LatentWrapper(base_model, tokenizer)
    
    # Input with two spans: [token, start, lat1, lat2, end, token, start, lat3, end, token]
    input_ids = torch.tensor([[10, 101, 100, 100, 102, 11, 101, 100, 102, 12]], dtype=torch.long)
    
    batch_size, seq_len = input_ids.shape
    hidden_dim = 512
    last_hidden = torch.zeros(batch_size, seq_len, hidden_dim)
    
    # Set distinct values for testing
    for pos in range(seq_len):
        last_hidden[0, pos, :] = float(pos + 10)  # Values 10, 11, 12, ...
    
    spans = wrapper._extract_latent_spans(input_ids)
    print(f"Extracted spans: {spans}")
    
    modified_embeds = wrapper._build_modified_embeddings_sequential(input_ids, spans, last_hidden)
    
    # Verify each span
    for span_idx, (start, end) in enumerate(spans[0]):
        print(f"Span {span_idx}: start={start}, end={end}")
        for pos in range(start + 1, end):
            if pos < modified_embeds.shape[1]:
                embedding_value = modified_embeds[0, pos, 0].item()
                expected_value = float(pos + 10 - 1)  # pos-1 value from hidden states
                
                print(f"  Token at pos {pos}: embedding[0]={embedding_value:.1f}, expected={expected_value:.1f}")
                
                if abs(embedding_value - expected_value) < 0.1:
                    print(f"    ✓ CORRECT")
                else:
                    print(f"    ✗ ERROR")
                    return False
    
    print("✓ Multiple spans test PASSED")
    return True

def main():
    """Run all tests"""
    print("=== Testing Latent Token Injection Algorithm Fix ===\n")
    
    try:
        # Test individual token injection
        success1 = test_latent_injection_individual_tokens()
        
        # Test multiple spans
        success2 = test_multiple_spans()
        
        if success1 and success2:
            print("\n🎉 ALL TESTS PASSED!")
            print("The latent token injection algorithm fix is working correctly.")
            print("Each latent token now receives the hidden state from its immediate predecessor (pos-1).")
            return True
        else:
            print("\n❌ SOME TESTS FAILED!")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
