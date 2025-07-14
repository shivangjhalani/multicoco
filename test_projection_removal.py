#!/usr/bin/env python3
"""
Test script to verify the removal of dimension projection layers.

This script tests that:
1. No projection layers are created or used
2. Dimension mismatches are handled by skipping rather than projecting
3. Direct assignment is used for compatible dimensions
"""

import torch
import torch.nn as nn
import sys
import os

# Add multicoco to path
sys.path.insert(0, '/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco')

from multicoco.latent_wrapper import LatentWrapper

class MockTokenizer:
    """Mock tokenizer for testing"""
    def __init__(self):
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.token_to_id = {
            '<|latent|>': 100,
            '<|start_latent|>': 101, 
            '<|end_latent|>': 102,
            '<IMG_CONTEXT>': 103
        }
        
    def convert_tokens_to_ids(self, token):
        return self.token_to_id.get(token, 999)

class MockEmbedding(nn.Module):
    """Mock embedding layer with configurable dimensions"""
    def __init__(self, vocab_size=1000, embed_dim=512):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, embed_dim))
        self.num_embeddings = vocab_size
        self.embedding_dim = embed_dim
        
    def forward(self, input_ids):
        return torch.embedding(self.weight, input_ids)

class MockModel(nn.Module):
    """Mock base model for testing"""
    def __init__(self, embed_dim=512):
        super().__init__()
        self.embedding = MockEmbedding(embed_dim=embed_dim)
        self.img_context_token_id = 103
        
    def get_input_embeddings(self):
        return self.embedding

def test_no_projection_layers_created():
    """Test that no projection layers are created even with dimension mismatch"""
    print("Testing that no projection layers are created...")
    
    # Create mock components with mismatched dimensions
    tokenizer = MockTokenizer()
    base_model = MockModel(embed_dim=256)  # Different from hidden_dim
    
    # Create LatentWrapper
    wrapper = LatentWrapper(base_model, tokenizer)
    
    # Verify no projection layer initially
    has_projection_before = hasattr(wrapper, '_hidden_to_embed_proj')
    print(f"Has projection layer before: {has_projection_before}")
    
    # Create test input with latent span
    input_ids = torch.tensor([[10, 101, 100, 100, 102, 11]], dtype=torch.long)
    
    # Create hidden states with different dimension than embeddings
    batch_size, seq_len = input_ids.shape
    hidden_dim = 512  # Different from embed_dim=256
    last_hidden = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Extract spans and try to apply injection
    spans = wrapper._extract_latent_spans(input_ids)
    print(f"Extracted spans: {spans}")
    
    # This should handle dimension mismatch without creating projection layers
    modified_embeds = wrapper._build_modified_embeddings_sequential(input_ids, spans, last_hidden)
    
    # Verify no projection layer was created
    has_projection_after = hasattr(wrapper, '_hidden_to_embed_proj')
    print(f"Has projection layer after: {has_projection_after}")
    
    if not has_projection_before and not has_projection_after:
        print("✓ CORRECT: No projection layers created")
        return True
    else:
        print("✗ ERROR: Projection layer was created when it shouldn't be")
        return False

def test_dimension_compatibility_validation():
    """Test that compatible dimensions work with direct assignment"""
    print("\nTesting dimension compatibility validation...")
    
    # Create mock components with matching dimensions
    embed_dim = 512
    tokenizer = MockTokenizer()
    base_model = MockModel(embed_dim=embed_dim)
    
    wrapper = LatentWrapper(base_model, tokenizer)
    
    # Create test input
    input_ids = torch.tensor([[10, 101, 100, 100, 102, 11]], dtype=torch.long)
    
    # Create hidden states with same dimension as embeddings
    batch_size, seq_len = input_ids.shape
    hidden_dim = embed_dim  # Same as embed_dim
    last_hidden = torch.zeros(batch_size, seq_len, hidden_dim)
    
    # Set distinct values for verification
    for pos in range(seq_len):
        last_hidden[0, pos, :] = float(pos + 1)
    
    spans = wrapper._extract_latent_spans(input_ids)
    modified_embeds = wrapper._build_modified_embeddings_sequential(input_ids, spans, last_hidden)
    
    # Verify direct assignment worked
    if len(spans[0]) > 0:
        start, end = spans[0][0]
        for pos in range(start + 1, end):
            if pos < modified_embeds.shape[1]:
                # Check if the hidden state was directly assigned
                embedding_value = modified_embeds[0, pos, 0].item()
                expected_value = float(pos)  # From hidden state at pos-1
                
                if abs(embedding_value - expected_value) < 0.1:
                    print(f"  ✓ Direct assignment worked for pos {pos}")
                else:
                    print(f"  ✗ Direct assignment failed for pos {pos}")
                    return False
    
    print("✓ CORRECT: Compatible dimensions handled with direct assignment")
    return True

def test_dimension_mismatch_handling():
    """Test that dimension mismatches are handled by skipping, not projecting"""
    print("\nTesting dimension mismatch handling...")
    
    # Create mock components with incompatible dimensions
    tokenizer = MockTokenizer()
    base_model = MockModel(embed_dim=256)
    
    wrapper = LatentWrapper(base_model, tokenizer)
    
    # Create test input
    input_ids = torch.tensor([[10, 101, 100, 100, 102, 11]], dtype=torch.long)
    original_embeds = wrapper.embedding(input_ids).clone()
    
    # Create hidden states with different dimension
    batch_size, seq_len = input_ids.shape
    hidden_dim = 512  # Different from embed_dim=256
    last_hidden = torch.randn(batch_size, seq_len, hidden_dim)
    
    spans = wrapper._extract_latent_spans(input_ids)
    modified_embeds = wrapper._build_modified_embeddings_sequential(input_ids, spans, last_hidden)
    
    # With dimension mismatch, embeddings should remain unchanged (early return)
    embeddings_unchanged = torch.allclose(original_embeds, modified_embeds, atol=1e-6)
    
    if embeddings_unchanged:
        print("✓ CORRECT: Dimension mismatch handled by returning original embeddings")
        return True
    else:
        print("✗ ERROR: Embeddings were modified despite dimension mismatch")
        return False

def test_state_dict_no_projection_layers():
    """Test that state_dict methods don't save/load projection layers"""
    print("\nTesting state_dict methods exclude projection layers...")
    
    tokenizer = MockTokenizer()
    base_model = MockModel()
    wrapper = LatentWrapper(base_model, tokenizer)
    
    # Get state dict
    state_dict = wrapper.state_dict()
    
    # Check for any projection-related keys
    projection_keys = [key for key in state_dict.keys() if 'projection' in key or '_hidden_to_embed_proj' in key]
    
    if len(projection_keys) == 0:
        print("✓ CORRECT: No projection layer keys in state_dict")
        return True
    else:
        print(f"✗ ERROR: Found projection keys in state_dict: {projection_keys}")
        return False

def main():
    """Run all tests"""
    print("=== Testing Dimension Projection Layer Removal ===\n")
    
    try:
        # Test no projection layers are created
        success1 = test_no_projection_layers_created()
        
        # Test compatible dimensions work
        success2 = test_dimension_compatibility_validation()
        
        # Test incompatible dimensions are handled correctly
        success3 = test_dimension_mismatch_handling()
        
        # Test state_dict methods
        success4 = test_state_dict_no_projection_layers()
        
        if success1 and success2 and success3 and success4:
            print("\n🎉 ALL TESTS PASSED!")
            print("Dimension projection layers have been successfully removed.")
            print("Coconut's shared representation space assumption is now maintained.")
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
