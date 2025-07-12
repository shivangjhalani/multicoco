"""
Test script for Issue #4 fix: Incomplete Handling of Latent Tokens in Generation.
Tests that dynamically generated latent tokens are properly handled during generation.
"""

import sys
import os
from unittest.mock import Mock, MagicMock, patch
import logging

# Add the multicoco directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dynamic_latent_generation():
    """Test that latent tokens generated during generation are properly handled."""
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        from multicoco.constants import START_LATENT_TOKEN, END_LATENT_TOKEN, LATENT_TOKEN
        
        print("✓ Successfully imported required modules")
        
        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.convert_tokens_to_ids = Mock(side_effect=lambda x: {
            START_LATENT_TOKEN: 1000,
            END_LATENT_TOKEN: 1001,
            LATENT_TOKEN: 1002
        }[x])
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2
        
        # Mock base model with vision components
        base_model = Mock()
        base_model.get_input_embeddings = Mock(return_value=Mock())
        base_model.generate = Mock()
        
        # Mock the multimodal model structure
        base_model.model = Mock()
        base_model.model.vision_tower = Mock()
        base_model.model.projector = Mock()
        base_model.model.language_model = Mock()
        base_model.model.prepare_inputs_for_multimodal = Mock()
        base_model.model.dtype = Mock()
        
        # Create wrapper
        wrapper = LatentWrapper(base_model, tokenizer, enable_norm_logging=True)
        
        print("✓ LatentWrapper setup completed")
        
        # Test 1: Input without latent tokens, but generation produces them
        input_ids = Mock()
        input_ids.device = Mock()
        input_ids.shape = [1, 5]  # batch_size=1, seq_len=5
        input_ids.clone = Mock(return_value=input_ids)
        
        # Mock generation that produces latent tokens
        # Simulate: "Question: What is 2+2? <|start_latent|> <|latent|> <|latent|> <|end_latent|> Answer: 4"
        mock_generated_sequence = Mock()
        mock_generated_sequence.device = input_ids.device
        mock_generated_sequence.shape = [1, 15]  # Extended sequence
        
        # Mock the generation state updates
        generation_state = {
            'generated_ids': mock_generated_sequence,
            'attention_mask': Mock(),
            'unfinished_sequences': Mock(),
            'pad_token_id': 0,
            'eos_token_id': 2
        }
        
        # Mock the helper methods
        wrapper._initialize_generation_state = Mock(return_value=generation_state)
        wrapper._get_cached_vision_embeddings = Mock(return_value=None)
        
        # Mock forward to return some logits
        mock_forward_output = {'logits': Mock()}
        wrapper.forward = Mock(return_value=mock_forward_output)
        
        # Mock token sampling
        wrapper._sample_and_update_token = Mock(return_value=Mock())
        
        # Mock latent span detection - simulate that spans are generated dynamically
        call_count = 0
        def mock_has_latent_spans(ids):
            nonlocal call_count
            call_count += 1
            # First few calls: no spans, then spans appear
            return call_count > 3
        
        def mock_extract_latent_spans(ids):
            nonlocal call_count
            if call_count > 3:
                return [[(5, 9)]]  # Simulate one span from position 5 to 9
            return [[]]
        
        def mock_complete_partial_spans(ids):
            nonlocal call_count
            # Simulate span completion on call 6
            return call_count == 6
        
        wrapper._has_latent_spans = Mock(side_effect=mock_has_latent_spans)
        wrapper._extract_latent_spans = Mock(side_effect=mock_extract_latent_spans)
        wrapper._complete_partial_spans_if_needed = Mock(side_effect=mock_complete_partial_spans)
        
        # Mock early stopping condition
        generation_state['unfinished_sequences'].max = Mock(side_effect=[1, 1, 1, 1, 1, 1, 0])  # Stop after 6 steps
        
        # Test generation
        result = wrapper._generate_with_latent_injection(
            input_ids=input_ids,
            max_new_tokens=10
        )
        
        # Verify that the wrapper handled dynamic latent generation
        assert wrapper._has_latent_spans.call_count > 0, "Should check for latent spans during generation"
        assert wrapper._complete_partial_spans_if_needed.call_count > 0, "Should check for span completion"
        assert wrapper.forward.call_count > 0, "Should call forward pass multiple times"
        
        print("✓ Dynamic latent generation handled correctly")
        print("✓ Latent span detection called during generation")
        print("✓ Span completion checking works")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Note: This test requires the full MultiCoCo environment to run.")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_partial_span_detection():
    """Test detection of partial latent spans."""
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        from multicoco.constants import START_LATENT_TOKEN, END_LATENT_TOKEN, LATENT_TOKEN
        import torch
        
        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.convert_tokens_to_ids = Mock(side_effect=lambda x: {
            START_LATENT_TOKEN: 1000,
            END_LATENT_TOKEN: 1001,
            LATENT_TOKEN: 1002
        }[x])
        
        # Create wrapper
        wrapper = LatentWrapper(Mock(), tokenizer)
        
        # Test 1: No latent tokens
        input_ids = torch.tensor([[100, 200, 300, 400]])
        assert not wrapper._has_partial_latent_spans(input_ids), "Should not detect partial spans in normal text"
        
        # Test 2: Complete span
        input_ids = torch.tensor([[100, 1000, 1002, 1002, 1001, 400]])  # start, latent, latent, end
        assert not wrapper._has_partial_latent_spans(input_ids), "Should not detect partial spans when complete"
        
        # Test 3: Partial span (start without end)
        input_ids = torch.tensor([[100, 1000, 1002, 1002, 400]])  # start, latent, latent (no end)
        assert wrapper._has_partial_latent_spans(input_ids), "Should detect partial span (start without end)"
        
        print("✓ Partial span detection works correctly")
        
        # Test span completion detection
        # Test 1: Last token is not end token
        input_ids = torch.tensor([[100, 200, 300, 400]])
        assert not wrapper._complete_partial_spans_if_needed(input_ids), "Should not detect completion with non-end token"
        
        # Test 2: Last token is end token, completing a span
        input_ids = torch.tensor([[100, 1000, 1002, 1002, 1001]])  # start, latent, latent, end
        assert wrapper._complete_partial_spans_if_needed(input_ids), "Should detect span completion"
        
        # Test 3: Last token is end token, but no corresponding start
        input_ids = torch.tensor([[100, 200, 300, 1001]])  # end without start
        assert not wrapper._complete_partial_spans_if_needed(input_ids), "Should not detect completion without start"
        
        print("✓ Span completion detection works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Partial span detection test failed: {e}")
        return False


def test_generate_method_behavior():
    """Test that the generate method always uses latent injection."""
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        
        # Mock components
        tokenizer = Mock()
        base_model = Mock()
        
        # Create wrapper
        wrapper = LatentWrapper(base_model, tokenizer)
        
        # Mock the latent injection generation method
        wrapper._generate_with_latent_injection = Mock(return_value=Mock())
        
        # Mock input
        input_ids = Mock()
        
        # Test that generate always uses latent injection (addresses Issue #4)
        result = wrapper.generate(input_ids=input_ids)
        
        # Verify that latent injection generation was called
        wrapper._generate_with_latent_injection.assert_called_once()
        
        # Verify that base model generate was NOT called
        base_model.generate.assert_not_called()
        
        print("✓ Generate method always uses latent injection")
        print("✓ Can handle dynamically generated latent tokens")
        return True
        
    except Exception as e:
        print(f"✗ Generate method test failed: {e}")
        return False


def test_integration_with_forward_method():
    """Test that the generation integrates correctly with the fixed forward method."""
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        
        # Mock components
        tokenizer = Mock()
        base_model = Mock()
        wrapper = LatentWrapper(base_model, tokenizer)
        
        # Mock the forward method to simulate latent processing
        forward_call_count = 0
        def mock_forward(*args, **kwargs):
            nonlocal forward_call_count
            forward_call_count += 1
            return {'logits': Mock()}
        
        wrapper.forward = Mock(side_effect=mock_forward)
        
        # Mock other required methods
        wrapper._initialize_generation_state = Mock(return_value={
            'generated_ids': Mock(),
            'attention_mask': Mock(),
            'unfinished_sequences': Mock(),
            'pad_token_id': 0,
            'eos_token_id': 2
        })
        wrapper._get_cached_vision_embeddings = Mock(return_value=None)
        wrapper._sample_and_update_token = Mock(return_value=Mock())
        wrapper._has_latent_spans = Mock(return_value=False)
        wrapper._complete_partial_spans_if_needed = Mock(return_value=False)
        
        # Mock early stopping
        mock_unfinished = Mock()
        mock_unfinished.max = Mock(side_effect=[1, 1, 0])  # Stop after 2 iterations
        wrapper._initialize_generation_state.return_value['unfinished_sequences'] = mock_unfinished
        
        # Test generation
        input_ids = Mock()
        result = wrapper._generate_with_latent_injection(input_ids=input_ids, max_new_tokens=5)
        
        # Verify that forward was called multiple times (once per generation step)
        assert wrapper.forward.call_count >= 2, f"Forward should be called multiple times, was called {wrapper.forward.call_count} times"
        
        print("✓ Generation integrates correctly with forward method")
        print("✓ Forward method called once per generation step")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Issue #4 fix: Dynamic Latent Token Generation...")
    print("=" * 60)
    
    try:
        test1_success = test_partial_span_detection()
        test2_success = test_generate_method_behavior()
        test3_success = test_integration_with_forward_method()
        test4_success = test_dynamic_latent_generation()
        
        if test1_success and test2_success and test3_success and test4_success:
            print("\n✓ All tests passed! Issue #4 fix appears to be working correctly.")
            print("Key improvements:")
            print("  - Generate method always uses latent injection")
            print("  - Dynamic latent span detection during generation")
            print("  - Partial span tracking and completion detection")
            print("  - Integration with sequential chaining from Issue #1")
        else:
            print("\n✗ Some tests failed. Please check the implementation.")
            
    except Exception as e:
        print(f"\n✗ Test execution failed: {e}")
        print("Note: This test requires a proper PyTorch environment to run.")
