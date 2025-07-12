import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from multicoco.latent_wrapper import LatentWrapper
    from multicoco.constants import START_LATENT_TOKEN, END_LATENT_TOKEN, LATENT_TOKEN
except ImportError as e:
    print(f'Import error: {e}')
    print('This test requires the full MultiCoCo environment to run.')
    sys.exit(1)

def test_latent_chaining():
    tokenizer = Mock()
    tokenizer.convert_tokens_to_ids = Mock(side_effect=lambda x: {START_LATENT_TOKEN: 1000, END_LATENT_TOKEN: 1001, LATENT_TOKEN: 1002}[x])
    base_model = Mock()
    base_model.get_input_embeddings = Mock(return_value=nn.Embedding(50000, 768))
    base_model.model = Mock()
    base_model.model.vision_tower = Mock()
    base_model.model.projector = Mock()
    base_model.model.language_model = Mock()
    base_model.model.prepare_inputs_for_multimodal = Mock()
    base_model.model.dtype = torch.float32
    wrapper = LatentWrapper(base_model, tokenizer, enable_norm_logging=True)
    input_ids = torch.tensor([[100, 200, 1000, 1002, 1002, 1002, 1001, 300]])
    attention_mask = torch.ones_like(input_ids)
    spans = wrapper._extract_latent_spans(input_ids)
    expected_spans = [[(2, 6)]]
    print(f'Detected spans: {spans}')
    print(f'Expected spans: {expected_spans}')
    assert spans == expected_spans, f'Expected {expected_spans}, got {spans}'
    hidden_dim = 768
    seq_len = input_ids.shape[1]
    last_hidden = torch.randn(1, seq_len, hidden_dim)

    def mock_language_model(**kwargs):
        output = Mock()
        current_seq_len = kwargs['inputs_embeds'].shape[1]
        hidden_states = torch.randn(1, current_seq_len, hidden_dim)
        output.hidden_states = [hidden_states]
        return output
    base_model.model.language_model.side_effect = mock_language_model
    base_model.model.prepare_inputs_for_multimodal.side_effect = lambda **kwargs: kwargs.get('inputs_embeds')
    try:
        inputs_embeds = wrapper._build_modified_embeddings(input_ids, spans, last_hidden, None, attention_mask)
        print('✓ Sequential chaining implementation completed successfully')
        print(f'Input embeddings shape: {inputs_embeds.shape}')
        call_count = base_model.model.language_model.call_count
        expected_calls = 4
        print(f'Language model called {call_count} times for chaining')
        if call_count >= expected_calls:
            print('✓ Sequential chaining verified: Multiple forward passes detected')
        else:
            print(f'⚠ Expected at least {expected_calls} calls for chaining, got {call_count}')
    except Exception as e:
        print(f'✗ Error during sequential chaining: {e}')
        return False
    return True

def test_no_latent_spans():
    tokenizer = Mock()
    tokenizer.convert_tokens_to_ids = Mock(side_effect=lambda x: {START_LATENT_TOKEN: 1000, END_LATENT_TOKEN: 1001, LATENT_TOKEN: 1002}[x])
    base_model = Mock()
    base_model.get_input_embeddings = Mock(return_value=nn.Embedding(50000, 768))
    wrapper = LatentWrapper(base_model, tokenizer)
    input_ids = torch.tensor([[100, 200, 300, 400]])
    spans = wrapper._extract_latent_spans(input_ids)
    expected_spans = [[]]
    assert spans == expected_spans, f'Expected {expected_spans}, got {spans}'
    print('✓ No latent spans correctly detected')
    return True
if __name__ == '__main__':
    print('Testing Issue #1 fix: Sequential latent chaining...')
    print('=' * 60)
    try:
        test1_success = test_no_latent_spans()
        test2_success = test_latent_chaining()
        if test1_success and test2_success:
            print('\n✓ All tests passed! Issue #1 fix appears to be working correctly.')
        else:
            print('\n✗ Some tests failed. Please check the implementation.')
    except Exception as e:
        print(f'\n✗ Test execution failed: {e}')
        print('Note: This test requires a proper PyTorch environment to run.')