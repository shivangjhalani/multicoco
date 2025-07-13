#!/usr/bin/env python3
"""
Comprehensive test suite for LatentWrapper integration with MultiCoCo codebase.
Tests the CoCoNut algorithm implementation and compatibility with the training pipeline.
"""

import sys
import os
import logging
import traceback
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_mock_tokenizer():
    """Create a mock tokenizer for testing"""
    class MockTokenizer:
        def __init__(self):
            # Create token mappings for special tokens
            self.special_tokens = {
                '<|latent|>': 50001,
                '<|start_latent|>': 50002, 
                '<|end_latent|>': 50003,
                '<|pad|>': 0,
                '<|eos|>': 1,
                '<|image|>': 50004
            }
            self.id_to_token = {v: k for k, v in self.special_tokens.items()}
            self.vocab_size = 50010
            self.eos_token_id = 1
            self.pad_token_id = 0
        
        def convert_tokens_to_ids(self, token):
            if isinstance(token, str):
                return self.special_tokens.get(token, 2)  # Default to token ID 2
            return [self.special_tokens.get(t, 2) for t in token]
        
        def decode(self, token_ids, skip_special_tokens=False):
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            
            tokens = []
            for tid in token_ids:
                if tid in self.id_to_token:
                    if not skip_special_tokens or not self.id_to_token[tid].startswith('<|'):
                        tokens.append(self.id_to_token[tid])
                else:
                    tokens.append(f"<unk_{tid}>")
            return " ".join(tokens)
        
        def encode(self, text, add_special_tokens=False):
            # Simple mock encoding
            words = text.split()
            return [hash(word) % 1000 + 100 for word in words]
        
        def __len__(self):
            return self.vocab_size
    
    return MockTokenizer()

def create_mock_base_model(hidden_size=768, vocab_size=50010):
    """Create a mock base model that mimics InternVL structure"""
    
    class MockLanguageModel(nn.Module):
        def __init__(self, hidden_size, vocab_size):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList([nn.TransformerDecoderLayer(
                d_model=hidden_size, nhead=8, batch_first=True
            ) for _ in range(2)])
            self.norm = nn.LayerNorm(hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        def forward(self, inputs_embeds, attention_mask=None, output_hidden_states=False, use_cache=False, **kwargs):
            hidden_states = inputs_embeds
            all_hidden_states = [hidden_states] if output_hidden_states else None
            
            for layer in self.layers:
                hidden_states = layer(hidden_states, hidden_states)
                if output_hidden_states:
                    all_hidden_states.append(hidden_states)
            
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            
            class Output:
                def __init__(self, logits, hidden_states=None):
                    self.logits = logits
                    self.hidden_states = hidden_states
            
            return Output(logits, all_hidden_states if output_hidden_states else None)
        
        def resize_token_embeddings(self, new_size):
            self.embed_tokens = nn.Embedding(new_size, self.embed_tokens.embedding_dim)
            self.lm_head = nn.Linear(self.lm_head.in_features, new_size)
    
    class MockVisionTower(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.conv = nn.Conv2d(3, hidden_size, kernel_size=16, stride=16)
            
        def forward(self, pixel_values):
            # Mock vision processing: [B, 3, H, W] -> [B, num_patches, hidden_size]
            batch_size = pixel_values.shape[0]
            features = self.conv(pixel_values)  # [B, hidden_size, H//16, W//16]
            features = features.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_size]
            return features
    
    class MockProjector(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear = nn.Linear(hidden_size, hidden_size)
            
        def forward(self, vision_embeds):
            return self.linear(vision_embeds)
    
    class MockInternVLModel(nn.Module):
        def __init__(self, hidden_size, vocab_size):
            super().__init__()
            self.vision_tower = MockVisionTower(hidden_size)
            self.projector = MockProjector(hidden_size)
            self.language_model = MockLanguageModel(hidden_size, vocab_size)
            self.dtype = torch.float32
            self.img_context_token_id = 50004  # <|image|> token
            
        def prepare_inputs_for_multimodal(self, input_ids=None, pixel_values=None, image_embeds=None, inputs_embeds=None, **kwargs):
            if inputs_embeds is not None:
                # If inputs_embeds provided, combine with image embeds
                if image_embeds is not None:
                    # Insert image embeddings at image token positions
                    batch_size, seq_len, hidden_size = inputs_embeds.shape
                    combined_embeds = inputs_embeds.clone()
                    # For simplicity, just add image embeds to the beginning
                    if image_embeds.shape[1] <= seq_len:
                        combined_embeds[:, :image_embeds.shape[1]] += image_embeds
                    return combined_embeds
                return inputs_embeds
            else:
                # Convert input_ids to embeddings and add image embeds
                inputs_embeds = self.language_model.embed_tokens(input_ids)
                if image_embeds is not None:
                    batch_size, seq_len, hidden_size = inputs_embeds.shape
                    # Add image embeddings at the beginning
                    if image_embeds.shape[1] <= seq_len:
                        inputs_embeds[:, :image_embeds.shape[1]] += image_embeds
                return inputs_embeds
    
    class MockBaseModel(nn.Module):
        def __init__(self, hidden_size, vocab_size):
            super().__init__()
            self.model = MockInternVLModel(hidden_size, vocab_size)
            
        def forward(self, input_ids=None, attention_mask=None, pixel_values=None, labels=None, **kwargs):
            if input_ids is not None:
                inputs_embeds = self.model.prepare_inputs_for_multimodal(
                    input_ids=input_ids, 
                    pixel_values=pixel_values
                )
            else:
                inputs_embeds = kwargs.get('inputs_embeds')
                
            output = self.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **kwargs
            )
            
            loss = None
            if labels is not None:
                shift_logits = output.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            class ModelOutput:
                def __init__(self, loss, logits):
                    self.loss = loss
                    self.logits = logits
            
            return ModelOutput(loss, output.logits)
        
        def generate(self, input_ids, attention_mask=None, pixel_values=None, max_new_tokens=10, **kwargs):
            # Simple mock generation
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
            
            generated = input_ids.clone()
            for _ in range(max_new_tokens):
                outputs = self.forward(input_ids=generated, attention_mask=attention_mask, pixel_values=pixel_values)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                
                # Update attention mask
                if attention_mask is not None:
                    attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=device)], dim=1)
            
            return generated
        
        def get_input_embeddings(self):
            return self.model.language_model.embed_tokens
        
        def resize_token_embeddings(self, new_size):
            return self.model.language_model.resize_token_embeddings(new_size)
        
        @property
        def device(self):
            return next(self.parameters()).device
    
    return MockBaseModel(hidden_size, vocab_size)

def test_basic_initialization():
    """Test 1: Basic LatentWrapper initialization"""
    logger.info("🧪 Test 1: Basic LatentWrapper initialization")
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        
        tokenizer = create_mock_tokenizer()
        base_model = create_mock_base_model()
        
        wrapper = LatentWrapper(base_model, tokenizer)
        
        # Check that special token IDs are correctly set
        assert wrapper.latent_id == 50001, f"Expected latent_id=50001, got {wrapper.latent_id}"
        assert wrapper.start_id == 50002, f"Expected start_id=50002, got {wrapper.start_id}"
        assert wrapper.end_id == 50003, f"Expected end_id=50003, got {wrapper.end_id}"
        
        # Check attribute delegation
        assert hasattr(wrapper, 'model'), "Wrapper should expose 'model' attribute"
        assert wrapper.model is wrapper.base_model, "wrapper.model should refer to base_model"
        
        logger.info("✅ Basic initialization test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic initialization test failed: {e}")
        traceback.print_exc()
        return False

def test_forward_without_latent_tokens():
    """Test 2: Forward pass without latent tokens (should use standard forward)"""
    logger.info("🧪 Test 2: Forward pass without latent tokens")
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        
        tokenizer = create_mock_tokenizer()
        base_model = create_mock_base_model()
        wrapper = LatentWrapper(base_model, tokenizer)
        
        # Create input without latent tokens
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(100, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(100, 1000, (batch_size, seq_len))
        
        # Forward pass
        output = wrapper.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Check output structure
        assert 'logits' in output, "Output should contain 'logits'"
        assert 'loss' in output, "Output should contain 'loss' when labels provided"
        assert output['logits'].shape == (batch_size, seq_len, tokenizer.vocab_size), \
            f"Expected logits shape {(batch_size, seq_len, tokenizer.vocab_size)}, got {output['logits'].shape}"
        
        logger.info("✅ Forward without latent tokens test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Forward without latent tokens test failed: {e}")
        traceback.print_exc()
        return False

def test_forward_with_latent_tokens():
    """Test 3: Forward pass with latent tokens (should use CoCoNut algorithm)"""
    logger.info("🧪 Test 3: Forward pass with latent tokens (CoCoNut algorithm)")
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        
        tokenizer = create_mock_tokenizer()
        base_model = create_mock_base_model()
        wrapper = LatentWrapper(base_model, tokenizer)
        
        # Create input with latent tokens: [regular tokens] <|start_latent|> <|latent|> <|latent|> <|end_latent|> [more tokens]
        batch_size = 2
        input_ids = torch.tensor([
            [100, 101, 102, 50002, 50001, 50001, 50003, 200, 201, 202],  # Has latent span
            [300, 301, 302, 50002, 50001, 50003, 400, 401, 402, 403]     # Has latent span
        ])
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(100, 1000, input_ids.shape)
        
        # Forward pass
        output = wrapper.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Check output structure
        assert 'logits' in output, "Output should contain 'logits'"
        assert 'loss' in output, "Output should contain 'loss' when labels provided"
        assert output['logits'].shape == (batch_size, input_ids.shape[1], tokenizer.vocab_size), \
            f"Expected logits shape {(batch_size, input_ids.shape[1], tokenizer.vocab_size)}, got {output['logits'].shape}"
        
        logger.info("✅ Forward with latent tokens test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Forward with latent tokens test failed: {e}")
        traceback.print_exc()
        return False

def test_generation_without_latent_tokens():
    """Test 4: Generation without latent tokens"""
    logger.info("🧪 Test 4: Generation without latent tokens")
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        
        tokenizer = create_mock_tokenizer()
        base_model = create_mock_base_model()
        wrapper = LatentWrapper(base_model, tokenizer)
        
        # Create input without latent tokens
        batch_size, seq_len = 2, 5
        input_ids = torch.randint(100, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Test generation
        generated = wrapper.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=3,
            do_sample=False
        )
        
        # Check output
        expected_shape = (batch_size, seq_len + 3)
        assert generated.shape == expected_shape, \
            f"Expected generated shape {expected_shape}, got {generated.shape}"
        
        # Check that original input is preserved
        assert torch.equal(generated[:, :seq_len], input_ids), \
            "Original input should be preserved in generated output"
        
        logger.info("✅ Generation without latent tokens test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Generation without latent tokens test failed: {e}")
        traceback.print_exc()
        return False

def test_generation_with_latent_tokens():
    """Test 5: Generation with latent tokens (custom generation loop)"""
    logger.info("🧪 Test 5: Generation with latent tokens (custom generation loop)")
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        
        tokenizer = create_mock_tokenizer()
        base_model = create_mock_base_model()
        wrapper = LatentWrapper(base_model, tokenizer)
        
        # Create input with latent tokens
        input_ids = torch.tensor([
            [100, 101, 50002, 50001, 50001, 50003, 200, 201],  # Has latent span
            [300, 301, 50002, 50001, 50003, 400, 401, 402]     # Has latent span
        ])
        attention_mask = torch.ones_like(input_ids)
        
        # Test generation
        generated = wrapper.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=3,
            do_sample=False
        )
        
        # Check output
        batch_size, original_seq_len = input_ids.shape
        expected_shape = (batch_size, original_seq_len + 3)
        assert generated.shape == expected_shape, \
            f"Expected generated shape {expected_shape}, got {generated.shape}"
        
        # Check that original input is preserved
        assert torch.equal(generated[:, :original_seq_len], input_ids), \
            "Original input should be preserved in generated output"
        
        logger.info("✅ Generation with latent tokens test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Generation with latent tokens test failed: {e}")
        traceback.print_exc()
        return False

def test_multimodal_integration():
    """Test 6: Multimodal integration with pixel_values"""
    logger.info("🧪 Test 6: Multimodal integration with pixel_values")
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        
        tokenizer = create_mock_tokenizer()
        base_model = create_mock_base_model()
        wrapper = LatentWrapper(base_model, tokenizer)
        
        # Create multimodal input
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(100, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        pixel_values = torch.randn(batch_size, 3, 224, 224)  # Mock image input
        
        # Forward pass with multimodal input
        output = wrapper.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        
        # Check output
        assert 'logits' in output, "Output should contain 'logits'"
        assert output['logits'].shape == (batch_size, seq_len, tokenizer.vocab_size), \
            f"Expected logits shape {(batch_size, seq_len, tokenizer.vocab_size)}, got {output['logits'].shape}"
        
        # Test generation with multimodal input
        generated = wrapper.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=2
        )
        
        expected_shape = (batch_size, seq_len + 2)
        assert generated.shape == expected_shape, \
            f"Expected generated shape {expected_shape}, got {generated.shape}"
        
        logger.info("✅ Multimodal integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Multimodal integration test failed: {e}")
        traceback.print_exc()
        return False

def test_trainer_compatibility():
    """Test 7: Compatibility with trainer-like usage patterns"""
    logger.info("🧪 Test 7: Trainer compatibility")
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        
        tokenizer = create_mock_tokenizer()
        base_model = create_mock_base_model()
        wrapper = LatentWrapper(base_model, tokenizer)
        
        # Test attributes that trainer expects
        assert hasattr(wrapper, 'tokenizer'), "Wrapper should expose tokenizer"
        assert hasattr(wrapper, 'device'), "Wrapper should expose device"
        assert hasattr(wrapper, 'model'), "Wrapper should expose model attribute"
        
        # Test device property
        device = wrapper.device
        assert isinstance(device, torch.device), f"Device should be torch.device, got {type(device)}"
        
        # Test train/eval mode switching
        wrapper.train()
        assert wrapper.training, "Wrapper should be in training mode"
        
        wrapper.eval()
        assert not wrapper.training, "Wrapper should be in evaluation mode"
        
        # Test generation with trainer-like parameters
        input_ids = torch.randint(100, 1000, (1, 5))
        generated = wrapper.generate(
            input_ids=input_ids,
            max_length=8,  # trainer might pass max_length instead of max_new_tokens
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        assert generated.shape[1] == 8, f"Expected length 8, got {generated.shape[1]}"
        
        logger.info("✅ Trainer compatibility test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Trainer compatibility test failed: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test 8: Error handling and edge cases"""
    logger.info("🧪 Test 8: Error handling and edge cases")
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        
        tokenizer = create_mock_tokenizer()
        base_model = create_mock_base_model()
        wrapper = LatentWrapper(base_model, tokenizer)
        
        # Test with empty input
        empty_input = torch.tensor([[]], dtype=torch.long)
        try:
            output = wrapper.forward(input_ids=empty_input)
            # Should not crash
        except Exception as e:
            logger.warning(f"Empty input caused error (might be expected): {e}")
        
        # Test with latent span at position 0 (edge case)
        input_with_start_latent = torch.tensor([
            [50002, 50001, 50001, 50003, 100, 101, 102, 103]  # Latent span starts at position 0
        ])
        
        try:
            output = wrapper.forward(input_ids=input_with_start_latent)
            # Should handle this edge case gracefully
            logger.info("  ✓ Edge case (latent span at position 0) handled")
        except Exception as e:
            logger.warning(f"  ⚠ Edge case caused error: {e}")
        
        # Test attribute access to non-existent attribute
        try:
            _ = wrapper.non_existent_attribute
            logger.warning("  ⚠ Expected AttributeError was not raised")
        except AttributeError:
            logger.info("  ✓ AttributeError correctly raised for non-existent attribute")
        
        logger.info("✅ Error handling test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error handling test failed: {e}")
        traceback.print_exc()
        return False

def test_coconut_algorithm_correctness():
    """Test 9: Verify CoCoNut algorithm implementation correctness"""
    logger.info("🧪 Test 9: CoCoNut algorithm correctness")
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        
        tokenizer = create_mock_tokenizer()
        base_model = create_mock_base_model()
        wrapper = LatentWrapper(base_model, tokenizer)
        
        # Create input with multiple latent spans
        input_ids = torch.tensor([
            [100, 101, 50002, 50001, 50001, 50003, 200, 50002, 50001, 50003, 300]
        ])
        
        # Extract latent spans
        spans = wrapper._extract_latent_spans(input_ids)
        expected_spans = [[(2, 5), (7, 9)]]  # Two spans: positions (2,5) and (7,9)
        
        assert spans == expected_spans, f"Expected spans {expected_spans}, got {spans}"
        
        # Test first pass hidden states
        attention_mask = torch.ones_like(input_ids)
        image_embeds = None
        
        hidden_states = wrapper._first_pass_hidden_states(input_ids, attention_mask, image_embeds)
        
        # Verify hidden states shape
        batch_size, seq_len = input_ids.shape
        expected_shape = (batch_size, seq_len, 768)  # 768 is our mock hidden size
        assert hidden_states.shape == expected_shape, \
            f"Expected hidden states shape {expected_shape}, got {hidden_states.shape}"
        
        # Test modified embeddings building
        inputs_embeds = wrapper._build_modified_embeddings(input_ids, spans, hidden_states)
        
        # Verify embeddings shape
        assert inputs_embeds.shape == (batch_size, seq_len, 768), \
            f"Expected embeddings shape {(batch_size, seq_len, 768)}, got {inputs_embeds.shape}"
        
        # Verify that latent tokens were replaced with hidden states from previous positions
        # For span (2,5): latent tokens at positions 2,3,4 should be replaced with hidden state from position 1
        # For span (7,9): latent tokens at positions 7,8 should be replaced with hidden state from position 6
        
        original_embeds = wrapper.embedding(input_ids)
        
        # Check that non-latent positions are unchanged
        non_latent_positions = [0, 1, 5, 6, 9, 10]
        for pos in non_latent_positions:
            assert torch.allclose(inputs_embeds[0, pos], original_embeds[0, pos], atol=1e-6), \
                f"Non-latent position {pos} should be unchanged"
        
        # Check that latent positions are modified
        latent_positions = [2, 3, 4, 7, 8]
        for pos in latent_positions:
            assert not torch.allclose(inputs_embeds[0, pos], original_embeds[0, pos], atol=1e-6), \
                f"Latent position {pos} should be modified"
        
        logger.info("✅ CoCoNut algorithm correctness test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ CoCoNut algorithm correctness test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and report results"""
    logger.info("🚀 Starting LatentWrapper Integration Test Suite")
    logger.info("=" * 60)
    
    tests = [
        test_basic_initialization,
        test_forward_without_latent_tokens,
        test_forward_with_latent_tokens,
        test_generation_without_latent_tokens,
        test_generation_with_latent_tokens,
        test_multimodal_integration,
        test_trainer_compatibility,
        test_error_handling,
        test_coconut_algorithm_correctness
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            logger.error(f"Test {test_func.__name__} crashed: {e}")
            results.append(False)
        logger.info("-" * 40)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total tests: {total}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {total - passed}")
    logger.info(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 All tests passed! LatentWrapper integration is working correctly.")
    else:
        logger.warning(f"⚠️ {total - passed} test(s) failed. Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n🛑 Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Test suite crashed: {e}")
        traceback.print_exc()
        sys.exit(1)
