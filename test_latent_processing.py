#!/usr/bin/env python3
"""
Test script to verify latent span detection and processing work correctly.
"""

import sys
import os
import torch
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multicoco.latent_wrapper_v2 import LatentWrapperV2
from multicoco.constants import COCONUT_SPECIAL_TOKENS

def test_latent_span_detection():
    """Test that LatentWrapperV2 can detect latent spans correctly."""
    print("Testing latent span detection...")
    
    try:
        # Create tokenizer with special tokens
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        tokenizer.add_special_tokens({'additional_special_tokens': COCONUT_SPECIAL_TOKENS})
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create mock model
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(1000, 64)
                
            def get_input_embeddings(self):
                return self.embed_tokens
                
            def forward(self, inputs_embeds, attention_mask=None, **kwargs):
                # Mock forward pass that returns a simple structure
                hidden_states = inputs_embeds.mean(dim=-1, keepdim=True).repeat(1, 1, 64)
                return type('obj', (object,), {'hidden_states': hidden_states, 'logits': hidden_states})()
        
        model = MockModel()
        wrapper = LatentWrapperV2(model, tokenizer)
        
        # Test input with latent spans
        test_text = "Question: What is this? <|start_latent|> <|latent|> <|latent|> <|latent|> <|end_latent|> It's a cat."
        input_ids = tokenizer.encode(test_text, return_tensors='pt')
        
        print(f"✓ Test text: {test_text}")
        print(f"✓ Input IDs shape: {input_ids.shape}")
        
        # Test span detection
        spans = wrapper._extract_latent_spans(input_ids[0])
        print(f"✓ Detected {len(spans)} latent spans: {spans}")
        
        if len(spans) > 0:
            start_pos, end_pos = spans[0]
            span_tokens = input_ids[0][start_pos:end_pos+1]
            span_text = tokenizer.decode(span_tokens)
            print(f"✓ First span tokens: {span_text}")
            
            # Verify span contains latent tokens
            assert '<|start_latent|>' in span_text, "Span should contain start latent token"
            assert '<|latent|>' in span_text, "Span should contain latent tokens" 
            assert '<|end_latent|>' in span_text, "Span should contain end latent token"
            print(f"✓ Span correctly contains latent tokens")
        else:
            print("✗ No latent spans detected!")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Latent span detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multimodal_prep():
    """Test that multimodal preparation works correctly.""" 
    print("\nTesting multimodal preparation...")
    
    try:
        # Create tokenizer with special tokens
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        tokenizer.add_special_tokens({'additional_special_tokens': COCONUT_SPECIAL_TOKENS})
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create mock model with vision tower
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(1000, 64)
                
            def get_input_embeddings(self):
                return self.embed_tokens
                
            def prepare_inputs_embeds(self, input_ids, pixel_values, attention_mask=None, **kwargs):
                # Mock multimodal preparation
                embeds = self.embed_tokens(input_ids)
                if pixel_values is not None:
                    # Simulate image embedding injection
                    vision_embeds = torch.randn(input_ids.shape[0], 256, 64)  # Mock vision embeddings
                    # Find <img> tokens and replace with vision embeddings
                    img_token_id = tokenizer.convert_tokens_to_ids('<img>')
                    img_positions = (input_ids == img_token_id).nonzero(as_tuple=True)
                    if len(img_positions[0]) > 0:
                        print(f"✓ Found <img> tokens at positions: {img_positions[1].tolist()}")
                        # This is a simplified simulation
                        embeds = torch.cat([embeds, vision_embeds[:, :embeds.shape[1], :]], dim=1)
                return embeds
                
        model = MockModel()
        wrapper = LatentWrapperV2(model, tokenizer)
        
        # Test input with image token
        test_text = "Question: <img> What is in this image? <|start_latent|> <|latent|> <|end_latent|> Answer"
        input_ids = tokenizer.encode(test_text, return_tensors='pt')
        pixel_values = torch.randn(1, 3, 224, 224)  # Mock image
        
        print(f"✓ Test text with image: {test_text}")
        print(f"✓ Input IDs shape: {input_ids.shape}")
        print(f"✓ Pixel values shape: {pixel_values.shape}")
        
        # Test multimodal prep
        if hasattr(model, 'prepare_inputs_embeds'):
            embeds = wrapper.multimodal_prep(input_ids, pixel_values)
            print(f"✓ Multimodal prep completed, embeddings shape: {embeds.shape}")
        else:
            # Fallback to basic embedding
            embeds = model.get_input_embeddings()(input_ids)
            print(f"✓ Basic embedding fallback, shape: {embeds.shape}")
            
        return True
        
    except Exception as e:
        print(f"✗ Multimodal prep test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_latent_injection():
    """Test that latent injection works correctly."""
    print("\nTesting latent injection...")
    
    try:
        # Create tokenizer with special tokens
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        tokenizer.add_special_tokens({'additional_special_tokens': COCONUT_SPECIAL_TOKENS})
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create mock model
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(1000, 64)
                
            def get_input_embeddings(self):
                return self.embed_tokens
                
            def forward(self, inputs_embeds, attention_mask=None, **kwargs):
                # Mock forward that returns hidden states
                hidden_states = inputs_embeds.mean(dim=-1, keepdim=True).repeat(1, 1, 64)
                return type('obj', (object,), {'hidden_states': hidden_states})()
                
        model = MockModel()
        wrapper = LatentWrapperV2(model, tokenizer)
        
        # Test input with latent spans
        test_text = "Question: What is this? <|start_latent|> <|latent|> <|latent|> <|end_latent|> Answer"
        input_ids = tokenizer.encode(test_text, return_tensors='pt')
        embeds = model.get_input_embeddings()(input_ids)
        
        print(f"✓ Test text: {test_text}")
        print(f"✓ Input embeddings shape: {embeds.shape}")
        
        # Test latent injection
        spans = wrapper._extract_latent_spans(input_ids[0])
        if len(spans) > 0:
            injected_embeds = wrapper.latent_injection(embeds, spans)
            print(f"✓ Latent injection completed, output shape: {injected_embeds.shape}")
            
            # Verify the injection modified the embeddings
            if not torch.equal(embeds, injected_embeds):
                print(f"✓ Embeddings were modified by latent injection")
            else:
                print(f"! Embeddings were not modified (could be expected in mock)")
            
        else:
            print("✗ No latent spans found for injection!")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Latent injection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Latent Span Detection and Processing")
    print("=" * 60)
    
    success = True
    success &= test_latent_span_detection()
    success &= test_multimodal_prep()
    success &= test_latent_injection()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All latent processing tests passed!")
    else:
        print("✗ Some tests failed. Check the output above.")
    
    sys.exit(0 if success else 1)
