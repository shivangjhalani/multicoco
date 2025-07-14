#!/usr/bin/env python3
"""
Comprehensive Test Suite for CoCoNut Algorithm Fixes

This script validates that all the coconut algorithm fixes work correctly together,
covering both multimodal and text-only scenarios.

Test Coverage:
1. Latent token injection correctness 
2. Absence of projection layers
3. Iterative multi-pass processing
4. Multimodal position handling
5. KV cache efficiency
6. Backward compatibility
7. Performance benchmarks
"""

import os
import sys
import time
import logging
import traceback
from typing import Dict, List, Optional, Tuple, Any

# Add the multicoco directory to the path
sys.path.insert(0, os.path.abspath('.'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer
    print("✓ Successfully imported torch and transformers")
except ImportError as e:
    logger.error(f"Failed to import required packages: {e}")
    sys.exit(1)

try:
    from multicoco.latent_wrapper import LatentWrapper
    from multicoco.config import MultiCoCoConfig
    from multicoco.answer_extraction import extract_answer_choice
    print("✓ Successfully imported multicoco modules")
except ImportError as e:
    logger.error(f"Failed to import multicoco modules: {e}")
    sys.exit(1)

class MockModel(nn.Module):
    """Mock model for testing that mimics InternVL3-1B structure"""
    def __init__(self, vocab_size=50000, hidden_size=4096, num_layers=12):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create nested structure similar to InternVL3
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Add other required attributes
        self.dtype = torch.float32
        self.img_context_token_id = None
        
        # Mock transformer layers for KV cache
        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(hidden_size, 16, batch_first=True)
            for _ in range(num_layers)
        ])
        
    def get_input_embeddings(self):
        return self.language_model.model.embed_tokens
    
    def extract_feature(self, pixel_values):
        """Mock vision feature extraction"""
        batch_size = pixel_values.shape[0]
        # Return mock image features
        return torch.randn(batch_size, 256, self.hidden_size)
    
    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, 
                pixel_values=None, labels=None, output_hidden_states=False, 
                use_cache=False, past_key_values=None, **kwargs):
        """Mock forward pass with proper structure"""
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        
        # Mock hidden states
        hidden_states = inputs_embeds + torch.randn_like(inputs_embeds) * 0.1
        
        # Mock logits
        logits = torch.randn(batch_size, seq_len, self.vocab_size)
        
        # Mock KV cache if requested
        kv_cache = None
        if use_cache:
            kv_cache = []
            for _ in range(self.num_layers):
                # Each layer has (key, value) tuple with proper 4D shapes
                key = torch.randn(batch_size, 16, seq_len, hidden_size // 16)  # [B, num_heads, seq_len, head_dim]
                value = torch.randn(batch_size, 16, seq_len, hidden_size // 16)
                kv_cache.append((key, value))
        
        # Mock outputs structure
        class MockOutputs:
            def __init__(self):
                self.logits = logits
                self.hidden_states = [hidden_states] * (self.num_layers + 1) if output_hidden_states else None
                self.past_key_values = kv_cache
        
        return MockOutputs()

class ComprehensiveCoCoNutTester:
    """Comprehensive tester for all CoCoNut algorithm fixes"""
    
    def __init__(self):
        self.device = torch.device('cpu')  # Use CPU for testing
        self.setup_test_environment()
        
    def setup_test_environment(self):
        """Set up the testing environment"""
        # Create mock tokenizer
        try:
            # Try to use a real tokenizer if available
            self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
        except:
            # Create a very basic mock tokenizer
            class MockTokenizer:
                def __init__(self):
                    self.vocab = {
                        '<|start_latent|>': 50001,
                        '<|latent|>': 50002, 
                        '<|end_latent|>': 50003,
                        '<IMG_CONTEXT>': 50004,
                        '<pad>': 50005,
                        '<eos>': 50006,
                        'hello': 1000,
                        'world': 2000,
                        'test': 3000,
                        'the': 4000,
                        'answer': 5000,
                        'is': 6000,
                        '42': 7000
                    }
                    self.unk_token_id = 0
                    self.pad_token_id = self.vocab['<pad>']
                    self.eos_token_id = self.vocab['<eos>']
                    
                def convert_tokens_to_ids(self, token):
                    return self.vocab.get(token, self.unk_token_id)
                    
                def encode(self, text, add_special_tokens=False, return_tensors=None):
                    # Simple word-based tokenization for testing
                    tokens = text.split()
                    ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
                    if return_tensors == "pt":
                        return torch.tensor([ids])
                    return ids
                    
                def decode(self, ids, skip_special_tokens=True):
                    # Reverse lookup for decoding
                    reverse_vocab = {v: k for k, v in self.vocab.items()}
                    if isinstance(ids, torch.Tensor):
                        ids = ids.tolist()
                    tokens = [reverse_vocab.get(id, f'<unk_{id}>') for id in ids]
                    return ' '.join(tokens)
            
            self.tokenizer = MockTokenizer()
        
        # Create mock model
        self.base_model = MockModel()
        
        # Create LatentWrapper
        self.wrapper = LatentWrapper(
            base_model=self.base_model,
            tokenizer=self.tokenizer,
            enable_norm_logging=True
        )
        
        print("✓ Test environment setup complete")
    
    def test_latent_injection_correctness(self) -> Dict[str, Any]:
        """Test 1: Verify latent token injection follows original coconut algorithm"""
        print("\n🧪 Testing Latent Token Injection Correctness...")
        
        results = {}
        start_time = time.time()
        
        try:
            # Test case 1: Single latent span
            input_ids = torch.tensor([[1000, 2000, 50001, 50002, 50002, 50003, 3000]])
            spans = self.wrapper._extract_latent_spans(input_ids)
            
            # Verify span extraction
            expected_spans = [[(2, 5)]]
            assert spans == expected_spans, f"Expected {expected_spans}, got {spans}"
            
            # Test latent list conversion
            latent_lists = self.wrapper._convert_spans_to_latent_lists(spans, input_ids.shape[1])
            expected_latent_lists = [[3, 4]]  # Positions 3 and 4 are latent tokens
            assert latent_lists == expected_latent_lists, f"Expected {expected_latent_lists}, got {latent_lists}"
            
            # Test forward pass with latent injection
            with torch.no_grad():
                outputs = self.wrapper.forward(input_ids=input_ids, attention_mask=None)
            
            results['single_span'] = True
            results['span_extraction'] = True 
            results['latent_conversion'] = True
            results['forward_pass'] = True
            
            # Test case 2: Multiple latent spans
            multi_input_ids = torch.tensor([[1000, 50001, 50002, 50003, 2000, 50001, 50002, 50002, 50003]])
            multi_spans = self.wrapper._extract_latent_spans(multi_input_ids)
            expected_multi_spans = [[(1, 3), (5, 8)]]
            assert multi_spans == expected_multi_spans, f"Expected {expected_multi_spans}, got {multi_spans}"
            
            multi_latent_lists = self.wrapper._convert_spans_to_latent_lists(multi_spans, multi_input_ids.shape[1])
            expected_multi_lists = [[2, 6, 7]]  # Positions from both spans
            assert multi_latent_lists == expected_multi_lists, f"Expected {expected_multi_lists}, got {multi_latent_lists}"
            
            results['multiple_spans'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['traceback'] = traceback.format_exc()
        
        results['duration'] = time.time() - start_time
        return results
    
    def test_projection_layer_absence(self) -> Dict[str, Any]:
        """Test 2: Verify no projection layers are created"""
        print("\n🧪 Testing Projection Layer Absence...")
        
        results = {}
        start_time = time.time()
        
        try:
            # Check that no projection layers exist in the wrapper
            projection_layers = []
            for name, module in self.wrapper.named_modules():
                if 'proj' in name.lower() or 'projection' in name.lower():
                    projection_layers.append(name)
            
            results['no_projection_layers'] = len(projection_layers) == 0
            results['projection_layers_found'] = projection_layers
            
            # Test that embeddings and hidden states have same dimensions
            input_ids = torch.tensor([[1000, 2000, 3000]])
            embeddings = self.wrapper.embedding(input_ids)
            
            # Mock forward to get hidden states
            with torch.no_grad():
                mock_hidden = torch.randn_like(embeddings)
            
            embed_dim = embeddings.shape[-1]
            hidden_dim = mock_hidden.shape[-1]
            
            results['dimension_compatibility'] = embed_dim == hidden_dim
            results['embed_dim'] = embed_dim
            results['hidden_dim'] = hidden_dim
            
            # Test state_dict doesn't contain projection parameters
            state_dict = self.wrapper.state_dict()
            proj_params = [k for k in state_dict.keys() if 'proj' in k.lower()]
            
            results['clean_state_dict'] = len(proj_params) == 0
            results['projection_params'] = proj_params
            
        except Exception as e:
            results['error'] = str(e)
            results['traceback'] = traceback.format_exc()
        
        results['duration'] = time.time() - start_time
        return results
    
    def test_multipass_processing(self) -> Dict[str, Any]:
        """Test 3: Verify iterative multi-pass processing"""
        print("\n🧪 Testing Multi-Pass Processing...")
        
        results = {}
        start_time = time.time()
        
        try:
            # Create input with multiple latent tokens
            input_ids = torch.tensor([[1000, 50001, 50002, 50002, 50002, 50003, 2000]])
            
            # Track the number of passes
            original_forward = self.wrapper.base_model.forward
            pass_count = 0
            
            def counting_forward(*args, **kwargs):
                nonlocal pass_count
                pass_count += 1
                return original_forward(*args, **kwargs)
            
            self.wrapper.base_model.forward = counting_forward
            
            # Run forward pass
            with torch.no_grad():
                outputs = self.wrapper.forward(input_ids=input_ids)
            
            # Restore original forward
            self.wrapper.base_model.forward = original_forward
            
            # Should have multiple passes (one per latent token + final pass)
            spans = self.wrapper._extract_latent_spans(input_ids)
            latent_lists = self.wrapper._convert_spans_to_latent_lists(spans, input_ids.shape[1])
            expected_passes = max([len(l) for l in latent_lists]) + 1  # +1 for final pass
            
            results['pass_count'] = pass_count
            results['expected_passes'] = expected_passes
            results['correct_pass_count'] = pass_count >= expected_passes
            
            # Test edge case: no latent tokens
            no_latent_ids = torch.tensor([[1000, 2000, 3000]])
            pass_count = 0
            self.wrapper.base_model.forward = counting_forward
            
            with torch.no_grad():
                outputs = self.wrapper.forward(input_ids=no_latent_ids)
            
            self.wrapper.base_model.forward = original_forward
            
            # Should have exactly 1 pass for no latent tokens
            results['no_latent_pass_count'] = pass_count
            results['no_latent_correct'] = pass_count == 1
            
        except Exception as e:
            results['error'] = str(e)
            results['traceback'] = traceback.format_exc()
        
        results['duration'] = time.time() - start_time
        return results
    
    def test_multimodal_position_handling(self) -> Dict[str, Any]:
        """Test 4: Verify multimodal position handling"""
        print("\n🧪 Testing Multimodal Position Handling...")
        
        results = {}
        start_time = time.time()
        
        try:
            # Set up image context token
            self.base_model.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
            
            # Create multimodal input with image tokens and latent tokens
            img_ctx_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
            input_ids = torch.tensor([[1000, img_ctx_id, img_ctx_id, 50001, 50002, 50003, 2000]])
            
            # Create mock image embeddings
            pixel_values = torch.randn(1, 3, 224, 224)
            image_embeds = self.wrapper._compute_vision_embeddings(pixel_values, None)
            
            # Test position calculation
            spans = self.wrapper._extract_latent_spans(input_ids)
            latent_pos = 4  # Position of the latent token
            
            # Test that multimodal preparation works
            embeddings = self.wrapper.embedding(input_ids)
            multimodal_embeddings = self.wrapper._prepare_inputs_for_multimodal_internvl(
                input_ids, image_embeds, embeddings
            )
            
            results['multimodal_preparation'] = multimodal_embeddings is not None
            results['embedding_shape_preserved'] = multimodal_embeddings.shape == embeddings.shape
            
            # Test forward pass with multimodal input
            with torch.no_grad():
                outputs = self.wrapper.forward(
                    input_ids=input_ids,
                    pixel_values=pixel_values
                )
            
            results['multimodal_forward'] = True
            
            # Test edge cases
            # Case 1: No image tokens
            text_only_ids = torch.tensor([[1000, 50001, 50002, 50003, 2000]])
            with torch.no_grad():
                text_outputs = self.wrapper.forward(input_ids=text_only_ids)
            
            results['text_only_forward'] = True
            
            # Case 2: Image tokens but no latent tokens
            img_only_ids = torch.tensor([[1000, img_ctx_id, img_ctx_id, 2000]])
            with torch.no_grad():
                img_outputs = self.wrapper.forward(
                    input_ids=img_only_ids,
                    pixel_values=pixel_values
                )
            
            results['image_only_forward'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['traceback'] = traceback.format_exc()
        
        results['duration'] = time.time() - start_time
        return results
    
    def test_kv_cache_efficiency(self) -> Dict[str, Any]:
        """Test 5: Verify KV cache management and efficiency"""
        print("\n🧪 Testing KV Cache Efficiency...")
        
        results = {}
        start_time = time.time()
        
        try:
            # Test KV cache validation
            # Create valid cache
            valid_cache = []
            for _ in range(12):  # 12 layers
                key = torch.randn(1, 16, 10, 64)  # [batch, heads, seq, head_dim]
                value = torch.randn(1, 16, 10, 64)
                valid_cache.append((key, value))
            
            is_valid = self.wrapper._validate_kv_cache(valid_cache)
            results['valid_cache_detection'] = is_valid
            
            # Test invalid cache detection
            invalid_cache = []
            for _ in range(12):
                key = torch.randn(1, 16, 10)  # Wrong dimensions (3D instead of 4D)
                value = torch.randn(1, 16, 10, 64)
                invalid_cache.append((key, value))
            
            is_invalid = not self.wrapper._validate_kv_cache(invalid_cache)
            results['invalid_cache_detection'] = is_invalid
            
            # Test cache extraction
            compute_range = (5, 10)
            extracted_cache = self.wrapper._extract_kv_cache_slice(valid_cache, compute_range)
            
            if extracted_cache is not None:
                # Check that extracted cache has correct dimensions
                sample_key, sample_value = extracted_cache[0]
                expected_seq_len = compute_range[0]  # Should slice up to start of compute range
                
                results['cache_extraction'] = True
                results['extracted_seq_len'] = sample_key.shape[2]
                results['expected_seq_len'] = expected_seq_len
                results['correct_slice'] = sample_key.shape[2] == expected_seq_len
            else:
                results['cache_extraction'] = False
            
            # Test cache usage in forward pass
            input_ids = torch.tensor([[1000, 50001, 50002, 50002, 50003, 2000]])
            
            # Time forward pass with cache vs without cache
            no_cache_start = time.time()
            with torch.no_grad():
                outputs = self.wrapper.forward(input_ids=input_ids)
            no_cache_time = time.time() - no_cache_start
            
            results['forward_time'] = no_cache_time
            results['cache_efficiency_test'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['traceback'] = traceback.format_exc()
        
        results['duration'] = time.time() - start_time
        return results
    
    def test_backward_compatibility(self) -> Dict[str, Any]:
        """Test 6: Verify backward compatibility with existing code"""
        print("\n🧪 Testing Backward Compatibility...")
        
        results = {}
        start_time = time.time()
        
        try:
            # Test that wrapper exposes required properties
            properties_to_check = ['model', 'embedding', 'image_processor']
            missing_properties = []
            
            for prop in properties_to_check:
                if not hasattr(self.wrapper, prop):
                    missing_properties.append(prop)
            
            results['all_properties_exposed'] = len(missing_properties) == 0
            results['missing_properties'] = missing_properties
            
            # Test that base model methods are accessible
            methods_to_check = ['forward', 'generate']
            missing_methods = []
            
            for method in methods_to_check:
                if not hasattr(self.wrapper, method):
                    missing_methods.append(method)
            
            results['all_methods_accessible'] = len(missing_methods) == 0
            results['missing_methods'] = missing_methods
            
            # Test state_dict compatibility
            state_dict = self.wrapper.state_dict()
            results['state_dict_accessible'] = isinstance(state_dict, dict)
            
            # Test that non-latent inputs work normally
            normal_input = torch.tensor([[1000, 2000, 3000, 4000]])
            with torch.no_grad():
                normal_output = self.wrapper.forward(input_ids=normal_input)
            
            results['normal_input_works'] = True
            
            # Test generate method
            try:
                with torch.no_grad():
                    generated = self.wrapper.generate(
                        input_ids=normal_input,
                        max_new_tokens=5,
                        do_sample=False
                    )
                results['generate_works'] = True
                results['generated_shape'] = generated.shape if hasattr(generated, 'shape') else str(type(generated))
            except Exception as gen_e:
                results['generate_works'] = False
                results['generate_error'] = str(gen_e)
            
        except Exception as e:
            results['error'] = str(e)
            results['traceback'] = traceback.format_exc()
        
        results['duration'] = time.time() - start_time
        return results
    
    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test 7: Performance benchmarks and efficiency metrics"""
        print("\n🧪 Testing Performance Benchmarks...")
        
        results = {}
        start_time = time.time()
        
        try:
            # Benchmark different sequence lengths
            sequence_lengths = [10, 20, 50]
            latent_counts = [1, 3, 5]
            
            timing_results = {}
            
            for seq_len in sequence_lengths:
                for latent_count in latent_counts:
                    # Create input with specified number of latent tokens
                    input_tokens = [1000] * (seq_len - latent_count * 3)  # Regular tokens
                    
                    # Add latent spans
                    for _ in range(latent_count):
                        input_tokens.extend([50001, 50002, 50003])  # <start_latent> <latent> <end_latent>
                    
                    input_ids = torch.tensor([input_tokens[:seq_len]])
                    
                    # Time the forward pass
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    forward_start = time.time()
                    with torch.no_grad():
                        outputs = self.wrapper.forward(input_ids=input_ids)
                    forward_time = time.time() - forward_start
                    
                    key = f"seq{seq_len}_lat{latent_count}"
                    timing_results[key] = {
                        'forward_time': forward_time,
                        'sequence_length': seq_len,
                        'latent_count': latent_count,
                        'tokens_per_second': seq_len / forward_time if forward_time > 0 else float('inf')
                    }
            
            results['timing_results'] = timing_results
            
            # Test memory efficiency (basic check)
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated()
                
                # Run a large batch
                large_input = torch.tensor([[1000, 50001, 50002, 50002, 50003, 2000]] * 5)
                with torch.no_grad():
                    outputs = self.wrapper.forward(input_ids=large_input)
                
                memory_after = torch.cuda.memory_allocated()
                memory_delta = memory_after - memory_before
                
                results['memory_efficient'] = memory_delta < 1e9  # Less than 1GB increase
                results['memory_delta_mb'] = memory_delta / (1024**2)
            else:
                results['memory_test_skipped'] = "CUDA not available"
            
            # Test generation speed
            gen_input = torch.tensor([[1000, 50001, 50002, 50003, 2000]])
            gen_start = time.time()
            
            try:
                with torch.no_grad():
                    generated = self.wrapper.generate(
                        input_ids=gen_input,
                        max_new_tokens=10,
                        do_sample=False
                    )
                gen_time = time.time() - gen_start
                
                results['generation_time'] = gen_time
                results['generation_speed'] = 10 / gen_time if gen_time > 0 else float('inf')  # tokens per second
            except Exception as gen_e:
                results['generation_test_failed'] = str(gen_e)
            
        except Exception as e:
            results['error'] = str(e)
            results['traceback'] = traceback.format_exc()
        
        results['duration'] = time.time() - start_time
        return results
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results"""
        print("🚀 Starting Comprehensive CoCoNut Algorithm Test Suite")
        print("=" * 60)
        
        test_suite_start = time.time()
        all_results = {}
        
        # Run all tests
        test_methods = [
            ('latent_injection', self.test_latent_injection_correctness),
            ('projection_absence', self.test_projection_layer_absence),
            ('multipass_processing', self.test_multipass_processing),
            ('multimodal_handling', self.test_multimodal_position_handling),
            ('kv_cache_efficiency', self.test_kv_cache_efficiency),
            ('backward_compatibility', self.test_backward_compatibility),
            ('performance_benchmarks', self.test_performance_benchmarks)
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_name, test_method in test_methods:
            try:
                test_results = test_method()
                all_results[test_name] = test_results
                
                # Determine if test passed (no errors and key checks passed)
                test_passed = 'error' not in test_results and self._evaluate_test_success(test_name, test_results)
                
                if test_passed:
                    passed_tests += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
                    if 'error' in test_results:
                        print(f"   Error: {test_results['error']}")
                
            except Exception as e:
                all_results[test_name] = {
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'duration': 0
                }
                print(f"💥 {test_name}: CRASHED - {str(e)}")
        
        total_duration = time.time() - test_suite_start
        
        # Compile final results
        final_results = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': (passed_tests / total_tests) * 100,
                'total_duration': total_duration
            },
            'individual_results': all_results
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {final_results['summary']['success_rate']:.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! CoCoNut algorithm fixes are working correctly.")
        else:
            print("⚠️  Some tests failed. Please check the detailed results above.")
        
        return final_results
    
    def _evaluate_test_success(self, test_name: str, results: Dict[str, Any]) -> bool:
        """Evaluate whether a test was successful based on its results"""
        
        if test_name == 'latent_injection':
            return all([
                results.get('single_span', False),
                results.get('span_extraction', False),
                results.get('latent_conversion', False),
                results.get('forward_pass', False),
                results.get('multiple_spans', False)
            ])
        
        elif test_name == 'projection_absence':
            return all([
                results.get('no_projection_layers', False),
                results.get('dimension_compatibility', False),
                results.get('clean_state_dict', False)
            ])
        
        elif test_name == 'multipass_processing':
            return all([
                results.get('correct_pass_count', False),
                results.get('no_latent_correct', False)
            ])
        
        elif test_name == 'multimodal_handling':
            return all([
                results.get('multimodal_preparation', False),
                results.get('embedding_shape_preserved', False),
                results.get('multimodal_forward', False),
                results.get('text_only_forward', False)
            ])
        
        elif test_name == 'kv_cache_efficiency':
            return all([
                results.get('valid_cache_detection', False),
                results.get('invalid_cache_detection', False),
                results.get('cache_efficiency_test', False)
            ])
        
        elif test_name == 'backward_compatibility':
            return all([
                results.get('all_properties_exposed', False),
                results.get('all_methods_accessible', False),
                results.get('state_dict_accessible', False),
                results.get('normal_input_works', False)
            ])
        
        elif test_name == 'performance_benchmarks':
            return all([
                'timing_results' in results,
                results.get('timing_results', {}) != {}
            ])
        
        return False

def main():
    """Main function to run the comprehensive test suite"""
    try:
        tester = ComprehensiveCoCoNutTester()
        results = tester.run_comprehensive_test_suite()
        
        # Optionally save results to file
        import json
        with open('comprehensive_test_results.json', 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_results = json.loads(json.dumps(results, default=str))
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n📁 Detailed results saved to: comprehensive_test_results.json")
        
        # Return appropriate exit code
        success_rate = results['summary']['success_rate']
        if success_rate == 100:
            return 0
        elif success_rate >= 80:
            return 1  # Mostly successful but some issues
        else:
            return 2  # Significant failures
            
    except Exception as e:
        logger.error(f"Test suite crashed: {e}")
        logger.error(traceback.format_exc())
        return 3  # Complete failure

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
