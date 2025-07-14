#!/usr/bin/env python3
"""
Test script for iterative multi-pass processing implementation.
Verifies that the new multi-pass approach works correctly following original coconut algorithm.
"""

import os
import sys
import torch
import logging

# Add the multicoco directory to path
sys.path.insert(0, '/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco')

from multicoco.latent_wrapper import LatentWrapper

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_mock_model():
    """Create a minimal mock model for testing"""
    class MockEmbedding(torch.nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=768):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(vocab_size, hidden_size))
            self.num_embeddings = vocab_size
            self.embedding_dim = hidden_size
        
        def forward(self, input_ids):
            return torch.nn.functional.embedding(input_ids, self.weight)
    
    class MockModel(torch.nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=768):
            super().__init__()
            self.embed_tokens = MockEmbedding(vocab_size, hidden_size)
            self.hidden_size = hidden_size
            self.img_context_token_id = 32000  # Mock image context token ID
            
        def forward(self, inputs_embeds=None, attention_mask=None, past_key_values=None, 
                   output_hidden_states=False, use_cache=False, **kwargs):
            batch_size, seq_len, hidden_size = inputs_embeds.shape
            
            # Mock hidden states (just return the inputs with some transformation)
            hidden_states = inputs_embeds + 0.1 * torch.randn_like(inputs_embeds)
            
            # Mock logits
            logits = torch.randn(batch_size, seq_len, 1000)
            
            # Mock past_key_values for KV caching
            mock_kv = torch.randn(batch_size, 8, seq_len, 64)  # 8 heads, 64 dim per head
            past_key_values = [(mock_kv, mock_kv) for _ in range(12)]  # 12 layers
            
            # Create mock output
            class MockOutput:
                def __init__(self):
                    self.logits = logits
                    self.hidden_states = [hidden_states] if output_hidden_states else None
                    self.past_key_values = past_key_values if use_cache else None
                    self.loss = None
            
            return MockOutput()
        
        def get_input_embeddings(self):
            return self.embed_tokens
        
        def parameters(self):
            return self.embed_tokens.parameters()
    
    return MockModel()

def create_mock_tokenizer():
    """Create a minimal mock tokenizer"""
    class MockTokenizer:
        def __init__(self):
            self.vocab = {
                '<|start_latent|>': 1,
                '<|latent|>': 2, 
                '<|end_latent|>': 3,
                '<IMG_CONTEXT>': 32000,
                '<unk>': 0
            }
            self.unk_token_id = 0
            
        def convert_tokens_to_ids(self, token):
            return self.vocab.get(token, self.unk_token_id)
    
    return MockTokenizer()

def test_multi_pass_processing():
    """Test that multi-pass processing works correctly"""
    print("🧪 Testing iterative multi-pass processing...")
    
    # Create test components
    mock_model = create_mock_model()
    mock_tokenizer = create_mock_tokenizer()
    
    # Create LatentWrapper
    wrapper = LatentWrapper(mock_model, mock_tokenizer)
    
    # Create test input with latent spans
    # Format: normal_tokens <start_latent> <latent> <latent> <end_latent> more_tokens <start_latent> <latent> <end_latent>
    input_ids = torch.tensor([[
        10, 11, 12,  # normal tokens
        1, 2, 2, 3,  # first latent span: start, latent, latent, end  
        13, 14,      # normal tokens
        1, 2, 3,     # second latent span: start, latent, end
        15, 16       # normal tokens
    ]])
    
    attention_mask = torch.ones_like(input_ids)
    
    print(f"📝 Input shape: {input_ids.shape}")
    print(f"📝 Input IDs: {input_ids.tolist()}")
    
    # Test that spans are correctly extracted
    spans = wrapper._extract_latent_spans(input_ids)
    print(f"📝 Extracted spans: {spans}")
    
    # Expected spans: [(3, 6), (9, 11)] for the latent regions
    expected_spans = [[(3, 6), (9, 11)]]
    assert spans == expected_spans, f"Expected {expected_spans}, got {spans}"
    print("✅ Span extraction correct")
    
    # Test latent list conversion
    latent_lists = wrapper._convert_spans_to_latent_lists(spans, input_ids.shape[1])
    print(f"📝 Latent lists: {latent_lists}")
    
    # Expected latent positions: [4, 5] from first span, [10] from second span
    expected_latent_lists = [[4, 5, 10]]
    assert latent_lists == expected_latent_lists, f"Expected {expected_latent_lists}, got {latent_lists}"
    print("✅ Latent list conversion correct")
    
    # Test forward pass with multi-pass processing
    try:
        with torch.no_grad():
            outputs = wrapper(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        
        print(f"📝 Output logits shape: {outputs.logits.shape}")
        print("✅ Multi-pass forward pass successful")
        
        # Verify that output shape matches input
        assert outputs.logits.shape[:2] == input_ids.shape, f"Expected logits shape {input_ids.shape}, got {outputs.logits.shape[:2]}"
        print("✅ Output shape verification passed")
        
    except Exception as e:
        print(f"❌ Multi-pass forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test case with no latent tokens (should work without issues)
    print("\n🧪 Testing case with no latent tokens...")
    no_latent_input = torch.tensor([[10, 11, 12, 13, 14, 15]])
    no_latent_attention = torch.ones_like(no_latent_input)
    
    try:
        with torch.no_grad():
            outputs = wrapper(input_ids=no_latent_input, attention_mask=no_latent_attention, labels=no_latent_input)
        print("✅ No-latent case successful")
    except Exception as e:
        print(f"❌ No-latent case failed: {e}")
        return False
    
    print("\n🎉 All multi-pass processing tests passed!")
    return True

def test_embedding_update_logic():
    """Test the embedding update logic specifically"""
    print("\n🧪 Testing embedding update logic...")
    
    mock_model = create_mock_model() 
    mock_tokenizer = create_mock_tokenizer()
    wrapper = LatentWrapper(mock_model, mock_tokenizer)
    
    # Create test tensors
    batch_size, seq_len, hidden_size = 1, 8, 768
    inputs_embeds = torch.randn(batch_size, seq_len, hidden_size)
    hidden_states = torch.randn(batch_size, seq_len - 2, hidden_size)  # Shorter due to compute range
    
    # Test latent lists: positions [2, 5] need to be updated
    latent_lists = [[2, 5]]
    
    # Test pass 0 (should update position 2)
    print("📝 Testing pass 0...")
    updated_embeds_0 = wrapper._update_embeddings_for_pass(
        inputs_embeds, hidden_states, latent_lists, pass_idx=0, hidden_states_offset=0
    )
    
    # Position 2 should be updated with hidden state from position 1
    # Check that it's different from original
    original_token_2 = inputs_embeds[0, 2, :]
    updated_token_2 = updated_embeds_0[0, 2, :]
    assert not torch.allclose(original_token_2, updated_token_2), "Token 2 should be updated"
    print("✅ Pass 0 update successful")
    
    # Test pass 1 (should update position 5) 
    print("📝 Testing pass 1...")
    updated_embeds_1 = wrapper._update_embeddings_for_pass(
        updated_embeds_0, hidden_states, latent_lists, pass_idx=1, hidden_states_offset=0
    )
    
    # Position 5 should be updated
    token_5_before = updated_embeds_0[0, 5, :]
    token_5_after = updated_embeds_1[0, 5, :]
    assert not torch.allclose(token_5_before, token_5_after), "Token 5 should be updated"
    print("✅ Pass 1 update successful")
    
    print("✅ Embedding update logic tests passed!")
    return True

if __name__ == "__main__":
    print("🚀 Starting multi-pass processing tests...")
    
    success = True
    success &= test_multi_pass_processing()
    success &= test_embedding_update_logic()
    
    if success:
        print("\n🎉 All tests passed! Multi-pass processing implementation is working correctly.")
        exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        exit(1)
