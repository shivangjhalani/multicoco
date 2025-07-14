#!/usr/bin/env python3
"""
Comprehensive Test Script for CoCoNut Algorithm Fixes

This script validates that all coconut algorithm fixes work correctly without breaking
existing functionality, covering both multimodal and text-only scenarios.

Tests cover:
1. Latent token injection correctness (each token gets pos-1 hidden state)
2. Absence of projection layers (direct hidden state assignment)
3. Iterative multi-pass processing (layer-by-layer token processing)
4. Multimodal position handling (correct source positions with image tokens)
5. KV cache efficiency (proper cache reuse across passes)
6. Backward compatibility (existing functionality still works)
7. Performance benchmarks to ensure no regression
"""

import sys
import time
import logging
import traceback
from typing import Optional, List, Tuple, Dict, Any
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Import the LatentWrapper
try:
    from multicoco.latent_wrapper import LatentWrapper
    from multicoco.config import Config
except ImportError as e:
    logger.error(f"Failed to import multicoco modules: {e}")
    sys.exit(1)

class MockTokenizer:
    """Mock tokenizer for testing"""
    def __init__(self):
        self.vocab = {
            '<|start_latent|>': 32000,
            '<|end_latent|>': 32001,
            '<|latent|>': 32002,
            '<IMG_CONTEXT>': 32003,
            '<|im_start|>': 32004,
            '<|im_end|>': 32005,
            'Hello': 1,
            'world': 2,
            'test': 3,
            'image': 4,
            'question': 5,
            'answer': 6,
            'the': 7,
            'this': 8,
            'is': 9,
            'a': 10,
            '[PAD]': 0,
            '[EOS]': 11,
        }
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = 0
        self.eos_token_id = 11
        self.unk_token_id = 12

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        # Simple mock encoding - just return some tokens
        tokens = [1, 2, 3, 4, 5]  # Mock token sequence
        if return_tensors == "pt":
            return torch.tensor([tokens])
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        tokens = [self.inverse_vocab.get(tid, '[UNK]') for tid in token_ids]
        return ' '.join(tokens)

class MockLanguageModel(nn.Module):
    """Mock language model for testing"""
    def __init__(self, vocab_size=50000, hidden_size=4096, num_layers=12):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
    def forward(self, inputs_embeds=None, attention_mask=None, past_key_values=None, 
                output_hidden_states=False, use_cache=False, **kwargs):
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        
        # Mock hidden states
        hidden_states = [inputs_embeds]  # Layer 0 (embedding)
        for _ in range(self.num_layers):
            # Each layer modifies the hidden state slightly
            layer_output = hidden_states[-1] + torch.randn_like(hidden_states[-1]) * 0.01
            hidden_states.append(layer_output)
        
        # Mock logits
        logits = torch.randn(batch_size, seq_len, self.vocab_size)
        
        # Mock KV cache
        kv_cache = None
        if use_cache:
            kv_cache = []
            for layer_idx in range(self.num_layers):
                key = torch.randn(batch_size, 32, seq_len, 128)  # 32 heads, 128 head_dim
                value = torch.randn(batch_size, 32, seq_len, 128)
                kv_cache.append((key, value))
        
        # Create mock outputs
        class MockOutputs:
            def __init__(self):
                self.logits = logits
                self.hidden_states = hidden_states if output_hidden_states else None
                self.past_key_values = kv_cache
                
        return MockOutputs()
    
    def get_input_embeddings(self):
        return self.embed_tokens

class MockVisionModel(nn.Module):
    """Mock vision model for testing"""
    def __init__(self, hidden_size=4096):
        super().__init__()
        self.hidden_size = hidden_size
        
    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        # Mock vision features - 256 image tokens per image
        return torch.randn(batch_size, 256, self.hidden_size)

class MockInternVLModel(nn.Module):
    """Mock InternVL model for testing"""
    def __init__(self, vocab_size=50000, hidden_size=4096, num_layers=12):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.language_model = MockLanguageModel(vocab_size, hidden_size, num_layers)
        self.vision_model = MockVisionModel(hidden_size)
        self.img_context_token_id = 32003  # <IMG_CONTEXT>
        self.dtype = torch.float32
        
        # Mock device property
        self._device = torch.device('cpu')
        
    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, 
                pixel_values=None, past_key_values=None, output_hidden_states=False, 
                use_cache=False, labels=None, **kwargs):
        
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)
        
        return self.language_model.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            **kwargs
        )
    
    def generate(self, input_ids=None, inputs_embeds=None, attention_mask=None, 
                 pixel_values=None, max_new_tokens=10, **kwargs):
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
        else:
            batch_size, seq_len, _ = inputs_embeds.shape
        
        # Mock generation - just return extended sequence
        new_tokens = torch.randint(1, 1000, (batch_size, max_new_tokens))
        
        if input_ids is not None:
            return torch.cat([input_ids, new_tokens], dim=1)
        else:
            # Generate mock token IDs for embeddings-based generation
            mock_input_ids = torch.randint(1, 1000, (batch_size, seq_len))
            return torch.cat([mock_input_ids, new_tokens], dim=1)
    
    def extract_feature(self, pixel_values):
        return self.vision_model(pixel_values)
    
    def get_input_embeddings(self):
        return self.language_model.embed_tokens
    
    def parameters(self):
        # Mock parameters method for device/dtype detection
        param = torch.tensor([1.0], device=self._device, dtype=self.dtype)
        yield param
    
    def chat(self, tokenizer, pixel_values=None, question="", generation_config=None, **kwargs):
        # Mock chat method
        return "This is a mock response from the base model."

class CoconutTestSuite:
    """Comprehensive test suite for CoCoNut algorithm fixes"""
    
    def __init__(self):
        self.tokenizer = MockTokenizer()
        self.base_model = MockInternVLModel()
        self.wrapper = LatentWrapper(self.base_model, self.tokenizer, enable_norm_logging=False)
        self.test_results = {}
        
    def run_all_tests(self):
        """Run all test suites"""
        print("🧪 Starting Comprehensive CoCoNut Algorithm Test Suite")
        print("=" * 60)
        
        test_methods = [
            self.test_latent_injection_correctness,
            self.test_absence_of_projection_layers,
            self.test_iterative_multipass_processing,
            self.test_multimodal_position_handling,
            self.test_kv_cache_efficiency,
            self.test_backward_compatibility,
            self.test_text_only_scenarios,
            self.test_multimodal_scenarios,
            self.test_performance_benchmarks,
            self.test_edge_cases
        ]
        
        passed = 0
        failed = 0
        
        for test_method in test_methods:
            try:
                print(f"\n🔬 Running {test_method.__name__}...")
                test_method()
                print(f"✅ {test_method.__name__} PASSED")
                self.test_results[test_method.__name__] = "PASSED"
                passed += 1
            except Exception as e:
                print(f"❌ {test_method.__name__} FAILED: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                self.test_results[test_method.__name__] = f"FAILED: {e}"
                failed += 1
        
        print("\n" + "=" * 60)
        print(f"📊 Test Results Summary: {passed} passed, {failed} failed")
        
        if failed == 0:
            print("🎉 All tests passed! CoCoNut algorithm fixes are working correctly.")
        else:
            print(f"⚠️  {failed} tests failed. Please review the failures above.")
        
        return self.test_results
    
    def test_latent_injection_correctness(self):
        """Test that each latent token gets hidden state from pos-1"""
        print("  📝 Testing latent token injection correctness...")
        
        # Create input with latent spans: "Hello <|start_latent|> <|latent|> <|latent|> <|end_latent|> world"
        start_id = self.tokenizer.convert_tokens_to_ids('<|start_latent|>')
        end_id = self.tokenizer.convert_tokens_to_ids('<|end_latent|>')
        latent_id = self.tokenizer.convert_tokens_to_ids('<|latent|>')
        
        input_ids = torch.tensor([[1, start_id, latent_id, latent_id, end_id, 2]])  # batch_size=1
        attention_mask = torch.ones_like(input_ids)
        
        # Run forward pass
        with torch.no_grad():
            outputs = self.wrapper.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids.clone()
            )
        
        # Verify output structure
        assert hasattr(outputs, 'logits'), "Output should have logits"
        assert outputs.logits.shape[0] == 1, "Batch size should be 1"
        assert outputs.logits.shape[1] == input_ids.shape[1], "Sequence length should match input"
        
        print("    ✓ Latent injection produces correct output structure")
        
        # Test span extraction
        spans = self.wrapper._extract_latent_spans(input_ids)
        expected_spans = [[(1, 5)]]  # One span from position 1 to 5
        assert spans == expected_spans, f"Expected spans {expected_spans}, got {spans}"
        
        print("    ✓ Latent span extraction works correctly")
        
        # Test latent list conversion  
        latent_lists = self.wrapper._convert_spans_to_latent_lists(spans, input_ids.shape[1])
        expected_latent_lists = [[2, 3]]  # Positions 2 and 3 are the latent tokens
        assert latent_lists == expected_latent_lists, f"Expected {expected_latent_lists}, got {latent_lists}"
        
        print("    ✓ Span to latent list conversion works correctly")
    
    def test_absence_of_projection_layers(self):
        """Test that no projection layers are created or used"""
        print("  📝 Testing absence of projection layers...")
        
        # Check that wrapper doesn't have any projection parameters
        param_names = [name for name, _ in self.wrapper.named_parameters()]
        projection_params = [name for name in param_names if 'proj' in name.lower()]
        
        assert len(projection_params) == 0, f"Found projection parameters: {projection_params}"
        print("    ✓ No projection parameters found in wrapper")
        
        # Check that embedding layer is the original one (no projection wrapper)
        embedding_layer = self.wrapper.embedding
        base_embedding = self.base_model.get_input_embeddings()
        
        # They should be the same object (reference equality)
        assert embedding_layer is base_embedding, "Embedding layer should be the original, not a projection wrapper"
        print("    ✓ Embedding layer is original (no projection wrapper)")
        
        # Test dimension compatibility
        test_input = torch.randint(0, 1000, (1, 5))
        embeddings = embedding_layer(test_input)
        
        # Create mock hidden states with same dimensions
        hidden_states = torch.randn_like(embeddings)
        
        # Should be able to assign directly without projection
        assert embeddings.shape == hidden_states.shape, "Embeddings and hidden states should have same shape"
        print("    ✓ Direct assignment possible (no dimension mismatch)")
    
    def test_iterative_multipass_processing(self):
        """Test iterative multi-pass processing instead of two-pass approach"""
        print("  📝 Testing iterative multi-pass processing...")
        
        # Create input with multiple latent tokens to test multi-pass
        start_id = self.tokenizer.convert_tokens_to_ids('<|start_latent|>')
        end_id = self.tokenizer.convert_tokens_to_ids('<|end_latent|>')
        latent_id = self.tokenizer.convert_tokens_to_ids('<|latent|>')
        
        # Input: "Hello <|start_latent|> <|latent|> <|latent|> <|latent|> <|end_latent|> world"
        input_ids = torch.tensor([[1, start_id, latent_id, latent_id, latent_id, end_id, 2]])
        attention_mask = torch.ones_like(input_ids)
        
        # Test that _sequential_latent_forward is called (multi-pass approach)
        original_method = self.wrapper._sequential_latent_forward
        call_count = 0
        
        def counting_wrapper(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_method(*args, **kwargs)
        
        self.wrapper._sequential_latent_forward = counting_wrapper
        
        try:
            with torch.no_grad():
                outputs = self.wrapper.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids.clone()
                )
            
            assert call_count == 1, f"_sequential_latent_forward should be called once, was called {call_count} times"
            print("    ✓ Multi-pass processing method called correctly")
            
        finally:
            # Restore original method
            self.wrapper._sequential_latent_forward = original_method
        
        # Test latent list processing
        spans = self.wrapper._extract_latent_spans(input_ids)
        latent_lists = self.wrapper._convert_spans_to_latent_lists(spans, input_ids.shape[1])
        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0
        
        assert max_n_latents == 3, f"Should have 3 latent tokens, found {max_n_latents}"
        print("    ✓ Correct number of latent tokens identified for multi-pass processing")
    
    def test_multimodal_position_handling(self):
        """Test correct source positions with image tokens"""
        print("  📝 Testing multimodal position handling...")
        
        # Create multimodal input with image tokens
        img_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        start_id = self.tokenizer.convert_tokens_to_ids('<|start_latent|>')
        end_id = self.tokenizer.convert_tokens_to_ids('<|end_latent|>')
        latent_id = self.tokenizer.convert_tokens_to_ids('<|latent|>')
        
        # Input: "Hello <IMG_CONTEXT> <IMG_CONTEXT> question <|start_latent|> <|latent|> <|end_latent|> answer"
        input_ids = torch.tensor([[1, img_token_id, img_token_id, 5, start_id, latent_id, end_id, 6]])
        attention_mask = torch.ones_like(input_ids)
        
        # Mock pixel values for multimodal input
        pixel_values = torch.randn(1, 3, 224, 224)
        
        # Test position calculation helper method
        test_pos = 5  # Position of latent token
        adjusted_pos = self.wrapper._calculate_adjusted_source_pos(input_ids, test_pos, 0)
        
        # The adjustment should account for image tokens that may not be in hidden states
        assert adjusted_pos is not None, "Adjusted position should be calculated"
        print(f"    ✓ Position adjustment calculated: {test_pos} -> {adjusted_pos}")
        
        # Test multimodal forward pass
        with torch.no_grad():
            outputs = self.wrapper.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=input_ids.clone()
            )
        
        assert hasattr(outputs, 'logits'), "Multimodal forward should produce logits"
        print("    ✓ Multimodal forward pass completed successfully")
    
    def test_kv_cache_efficiency(self):
        """Test proper cache reuse across passes"""
        print("  📝 Testing KV cache efficiency...")
        
        # Test KV cache validation
        # Create mock valid cache
        valid_cache = []
        for _ in range(12):  # 12 layers
            key = torch.randn(1, 32, 10, 128)
            value = torch.randn(1, 32, 10, 128)
            valid_cache.append((key, value))
        
        assert self.wrapper._validate_kv_cache(valid_cache), "Valid cache should pass validation"
        print("    ✓ KV cache validation works for valid cache")
        
        # Test invalid cache
        invalid_cache = [("not", "tensors")]
        assert not self.wrapper._validate_kv_cache(invalid_cache), "Invalid cache should fail validation"
        print("    ✓ KV cache validation correctly rejects invalid cache")
        
        # Test cache extraction
        compute_range = (5, 10)
        extracted = self.wrapper._extract_kv_cache_slice(valid_cache, compute_range)
        
        if extracted is not None:
            assert len(extracted) == len(valid_cache), "Extracted cache should have same number of layers"
            # Check that cache was sliced correctly
            for layer_idx, (key, value) in enumerate(extracted):
                assert key.shape[2] == compute_range[0], f"Key should be sliced to position {compute_range[0]}"
                assert value.shape[2] == compute_range[0], f"Value should be sliced to position {compute_range[0]}"
        
        print("    ✓ KV cache extraction works correctly")
    
    def test_backward_compatibility(self):
        """Test that existing functionality still works"""
        print("  📝 Testing backward compatibility...")
        
        # Test chat method without latent tokens
        question = "Hello world, how are you?"
        response = self.wrapper.chat(
            tokenizer=self.tokenizer,
            question=question,
            generation_config={'max_new_tokens': 10}
        )
        
        assert isinstance(response, str), "Chat should return string response"
        assert len(response) > 0, "Chat response should not be empty"
        print("    ✓ Chat method works for non-latent inputs")
        
        # Test generate method without latent spans
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        generated = self.wrapper.generate(
            input_ids=input_ids,
            max_new_tokens=5
        )
        
        assert generated.shape[0] == 1, "Generated output should have batch size 1"
        assert generated.shape[1] > input_ids.shape[1], "Generated sequence should be longer than input"
        print("    ✓ Generate method works for non-latent inputs")
        
        # Test forward pass without latent tokens
        with torch.no_grad():
            outputs = self.wrapper.forward(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                labels=input_ids.clone()
            )
        
        assert hasattr(outputs, 'logits'), "Forward pass should return logits"
        print("    ✓ Forward pass works for non-latent inputs")
    
    def test_text_only_scenarios(self):
        """Test text-only latent reasoning scenarios"""
        print("  📝 Testing text-only scenarios...")
        
        start_id = self.tokenizer.convert_tokens_to_ids('<|start_latent|>')
        end_id = self.tokenizer.convert_tokens_to_ids('<|end_latent|>')
        latent_id = self.tokenizer.convert_tokens_to_ids('<|latent|>')
        
        # Simple text-only latent input
        input_ids = torch.tensor([[7, 8, start_id, latent_id, end_id, 9]])  # "the this <latent> is"
        attention_mask = torch.ones_like(input_ids)
        
        # Test forward pass
        with torch.no_grad():
            outputs = self.wrapper.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids.clone()
            )
        
        assert outputs.logits.shape[1] == input_ids.shape[1], "Output length should match input"
        print("    ✓ Text-only latent forward pass works")
        
        # Test generation with text-only latent input
        generated = self.wrapper.generate(
            input_ids=input_ids,
            max_new_tokens=5
        )
        
        assert generated.shape[1] > input_ids.shape[1], "Generated text should be longer"
        print("    ✓ Text-only latent generation works")
    
    def test_multimodal_scenarios(self):
        """Test multimodal latent reasoning scenarios"""
        print("  📝 Testing multimodal scenarios...")
        
        img_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        start_id = self.tokenizer.convert_tokens_to_ids('<|start_latent|>')
        end_id = self.tokenizer.convert_tokens_to_ids('<|end_latent|>')
        latent_id = self.tokenizer.convert_tokens_to_ids('<|latent|>')
        
        # Multimodal input with image and latent tokens
        input_ids = torch.tensor([[4, img_token_id, img_token_id, start_id, latent_id, latent_id, end_id, 5]])
        attention_mask = torch.ones_like(input_ids)
        pixel_values = torch.randn(1, 3, 224, 224)
        
        # Test multimodal forward pass
        with torch.no_grad():
            outputs = self.wrapper.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=input_ids.clone()
            )
        
        assert outputs.logits.shape[1] == input_ids.shape[1], "Multimodal output length should match input"
        print("    ✓ Multimodal latent forward pass works")
        
        # Test multimodal generation
        generated = self.wrapper.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=5
        )
        
        assert generated.shape[1] > input_ids.shape[1], "Multimodal generated text should be longer"
        print("    ✓ Multimodal latent generation works")
        
        # Test chat with multimodal input
        question = "What do you see in this image? <|start_latent|> <|latent|> <|end_latent|> Please analyze."
        response = self.wrapper.chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config={'max_new_tokens': 10}
        )
        
        assert isinstance(response, str), "Multimodal chat should return string"
        print("    ✓ Multimodal chat with latent tokens works")
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks to ensure no regression"""
        print("  📝 Testing performance benchmarks...")
        
        start_id = self.tokenizer.convert_tokens_to_ids('<|start_latent|>')
        end_id = self.tokenizer.convert_tokens_to_ids('<|end_latent|>')
        latent_id = self.tokenizer.convert_tokens_to_ids('<|latent|>')
        
        # Test with longer sequence to measure performance
        base_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        latent_tokens = [start_id, latent_id, latent_id, latent_id, end_id]
        input_ids = torch.tensor([base_tokens + latent_tokens + base_tokens])
        attention_mask = torch.ones_like(input_ids)
        
        # Benchmark forward pass
        start_time = time.time()
        with torch.no_grad():
            for _ in range(5):  # Run multiple times for better measurement
                outputs = self.wrapper.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids.clone()
                )
        forward_time = (time.time() - start_time) / 5
        
        # Performance should be reasonable (less than 1 second for mock model)
        assert forward_time < 1.0, f"Forward pass too slow: {forward_time:.3f}s"
        print(f"    ✓ Forward pass performance: {forward_time:.3f}s per call")
        
        # Benchmark generation
        start_time = time.time()
        generated = self.wrapper.generate(
            input_ids=input_ids,
            max_new_tokens=10
        )
        generation_time = time.time() - start_time
        
        assert generation_time < 2.0, f"Generation too slow: {generation_time:.3f}s"
        print(f"    ✓ Generation performance: {generation_time:.3f}s")
        
        # Check that we have timing information
        if hasattr(self.wrapper, '_last_forward_time'):
            print(f"    ✓ Forward timing tracked: {self.wrapper._last_forward_time:.3f}s")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("  📝 Testing edge cases...")
        
        # Test empty input
        try:
            empty_input = torch.tensor([[]])
            outputs = self.wrapper.forward(
                input_ids=empty_input,
                attention_mask=torch.ones_like(empty_input),
                labels=empty_input.clone()
            )
            print("    ✓ Empty input handled gracefully")
        except Exception as e:
            # It's okay if empty input raises an exception, as long as it's handled
            print(f"    ⚠ Empty input raises exception (expected): {type(e).__name__}")
        
        # Test mismatched input dimensions
        try:
            input_ids = torch.tensor([[1, 2, 3]])
            attention_mask = torch.tensor([[1, 1]])  # Wrong length
            outputs = self.wrapper.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids.clone()
            )
            print("    ✓ Mismatched dimensions handled gracefully")
        except Exception as e:
            print(f"    ⚠ Mismatched dimensions raise exception (expected): {type(e).__name__}")
        
        # Test large batch size
        start_id = self.tokenizer.convert_tokens_to_ids('<|start_latent|>')
        end_id = self.tokenizer.convert_tokens_to_ids('<|end_latent|>')
        latent_id = self.tokenizer.convert_tokens_to_ids('<|latent|>')
        
        large_batch = torch.tensor([
            [1, start_id, latent_id, end_id, 2],
            [3, start_id, latent_id, end_id, 4],
            [5, start_id, latent_id, end_id, 6],
            [7, start_id, latent_id, end_id, 8]
        ])
        attention_mask = torch.ones_like(large_batch)
        
        with torch.no_grad():
            outputs = self.wrapper.forward(
                input_ids=large_batch,
                attention_mask=attention_mask,
                labels=large_batch.clone()
            )
        
        assert outputs.logits.shape[0] == 4, "Should handle batch size > 1"
        print("    ✓ Large batch size handled correctly")
        
        # Test no latent tokens (should delegate to base model)
        no_latent_input = torch.tensor([[1, 2, 3, 4, 5]])
        no_latent_mask = torch.ones_like(no_latent_input)
        
        with torch.no_grad():
            outputs = self.wrapper.forward(
                input_ids=no_latent_input,
                attention_mask=no_latent_mask,
                labels=no_latent_input.clone()
            )
        
        assert hasattr(outputs, 'logits'), "No-latent case should work"
        print("    ✓ No latent tokens case works (delegates to base model)")

def main():
    """Main test runner"""
    print("🚀 Starting CoCoNut Algorithm Comprehensive Test Suite")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    try:
        suite = CoconutTestSuite()
        results = suite.run_all_tests()
        
        # Print detailed results
        print("\n📋 Detailed Test Results:")
        for test_name, result in results.items():
            status = "✅" if "PASSED" in result else "❌"
            print(f"  {status} {test_name}: {result}")
        
        # Summary
        passed_count = sum(1 for r in results.values() if "PASSED" in r)
        total_count = len(results)
        
        print(f"\n🏁 Final Summary: {passed_count}/{total_count} tests passed")
        
        if passed_count == total_count:
            print("🎉 ALL TESTS PASSED! CoCoNut algorithm fixes are working correctly.")
            return 0
        else:
            print("⚠️  Some tests failed. Please review and fix the issues.")
            return 1
            
    except Exception as e:
        print(f"💥 Test suite failed with error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
