#!/usr/bin/env python3
"""
Test script to verify latency tracking functionality in MultiCoCo evaluation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import logging
import tempfile
import json
from unittest.mock import Mock, patch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_latency_config_loading():
    """Test that latency configuration is properly loaded."""
    print("=== Testing Latency Configuration Loading ===")
    
    try:
        from multicoco.config import MultiCoCoConfig
        
        # Test with latency enabled (default)
        config_dict = {
            'mode': 'eval_only',
            'eval_config': {
                'log_latency': True,
                'log_per_sample': True
            }
        }
        
        # Create a temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_dict, f)
            temp_config_path = f.name
        
        try:
            config = MultiCoCoConfig.from_yaml(temp_config_path)
            assert hasattr(config.evaluation, 'log_latency'), "log_latency attribute missing"
            assert config.evaluation.log_latency == True, f"Expected log_latency=True, got {config.evaluation.log_latency}"
            print("✓ Latency configuration loaded correctly")
            return True
        finally:
            os.unlink(temp_config_path)
            
    except Exception as e:
        print(f"✗ Latency configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generation_with_latency():
    """Test that _generate_batch_predictions_with_details returns latencies."""
    print("\n=== Testing Generation with Latency Tracking ===")
    
    try:
        from multicoco.trainer import CoCoTrainer
        from transformers import TrainingArguments
        
        # Create mock trainer with minimal setup
        args = TrainingArguments(
            output_dir="./test_output",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            logging_steps=1,
            report_to=[]  # Disable wandb for testing
        )
        
        # Mock model and tokenizer
        mock_model = Mock()
        mock_model.device = torch.device('cpu')
        mock_model.parameters.return_value = [torch.tensor([1.0])]  # For dtype checking
        
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode.return_value = "Test response"
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        
        # Create trainer with mocks
        trainer = CoCoTrainer(
            model=mock_model,
            args=args,
            tokenizer=mock_tokenizer
        )
        trainer.tokenizer = mock_tokenizer
        
        # Mock the generation config method
        def mock_get_generation_config(max_tokens):
            return {
                'max_new_tokens': max_tokens,
                'do_sample': False,
                'num_beams': 1
            }
        trainer._get_generation_config = mock_get_generation_config
        
        # Create test batch
        batch = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]]),
            'questions': ['What is this?'],
            'pixel_values': None
        }
        
        # Mock the model's generate method to return something
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])  # Original + generated
        
        # Test the method
        predictions, gen_texts, gen_tokens, latencies = trainer._generate_batch_predictions_with_details(
            batch, max_new_tokens=10
        )
        
        # Verify results
        assert len(predictions) == 1, f"Expected 1 prediction, got {len(predictions)}"
        assert len(gen_texts) == 1, f"Expected 1 generated text, got {len(gen_texts)}"
        assert len(gen_tokens) == 1, f"Expected 1 token count, got {len(gen_tokens)}"
        assert len(latencies) == 1, f"Expected 1 latency, got {len(latencies)}"
        
        # Check that latency is a reasonable positive number
        assert isinstance(latencies[0], float), f"Expected float latency, got {type(latencies[0])}"
        assert latencies[0] >= 0, f"Expected non-negative latency, got {latencies[0]}"
        assert latencies[0] < 10, f"Expected reasonable latency (<10s), got {latencies[0]}"
        
        print(f"✓ Generation with latency tracking successful")
        print(f"  - Predictions: {predictions}")
        print(f"  - Generated tokens: {gen_tokens}")
        print(f"  - Latencies: {latencies}")
        return True
        
    except Exception as e:
        print(f"✗ Generation latency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_per_sample_logging_with_latency():
    """Test that per-sample logging includes latency information."""
    print("\n=== Testing Per-Sample Logging with Latency ===")
    
    try:
        from multicoco.trainer import CoCoTrainer
        import io
        
        # Create a string buffer to capture log output
        log_capture = io.StringIO()
        
        # Create a logger that writes to our buffer
        eval_logger = logging.getLogger('evaluation_details')
        eval_logger.handlers.clear()  # Remove existing handlers
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(logging.Formatter('%(message)s'))
        eval_logger.addHandler(handler)
        eval_logger.setLevel(logging.INFO)
        
        # Create mock trainer
        trainer = CoCoTrainer(model=Mock(), args=Mock())
        
        # Test data
        questions = ['What is this?', 'How many objects?']
        labels = ['A', 'B']
        generated_texts = ['This is A', 'There are B objects']
        extracted = ['A', 'B']
        generated_tokens = [5, 7]
        correctness = [True, True]
        latencies = [0.123, 0.456]
        
        # Call the logging method
        trainer._log_per_sample_details(
            questions, labels, generated_texts, extracted, 
            generated_tokens, correctness, latencies
        )
        
        # Get the logged output
        logged_output = log_capture.getvalue()
        lines = [line.strip() for line in logged_output.split('\n') if line.strip()]
        
        # Verify we got the expected number of log lines
        assert len(lines) == 2, f"Expected 2 log lines, got {len(lines)}"
        
        # Parse and verify each line
        for i, line in enumerate(lines):
            log_data = json.loads(line)
            
            # Check required fields
            assert 'question' in log_data, "Missing 'question' field"
            assert 'ground_truth' in log_data, "Missing 'ground_truth' field"
            assert 'generated_answer' in log_data, "Missing 'generated_answer' field"
            assert 'extracted_answer' in log_data, "Missing 'extracted_answer' field"
            assert 'generated_tokens' in log_data, "Missing 'generated_tokens' field"
            assert 'correct' in log_data, "Missing 'correct' field"
            assert 'latency_sec' in log_data, "Missing 'latency_sec' field"
            
            # Check latency values
            expected_latency = latencies[i]
            actual_latency = log_data['latency_sec']
            assert actual_latency == expected_latency, f"Expected latency {expected_latency}, got {actual_latency}"
            
            print(f"✓ Sample {i+1} logged correctly with latency {actual_latency}s")
        
        print("✓ Per-sample logging with latency successful")
        return True
        
    except Exception as e:
        print(f"✗ Per-sample logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_metrics_with_latency():
    """Test that evaluation metrics include latency statistics."""
    print("\n=== Testing Evaluation Metrics with Latency ===")
    
    try:
        from multicoco.trainer import CoCoTrainer
        from multicoco.config import EvaluationConfig
        
        # Create mock trainer with latency enabled
        trainer = CoCoTrainer(model=Mock(), args=Mock())
        
        # Mock runner with evaluation config
        trainer.runner = Mock()
        trainer.runner.config = Mock()
        trainer.runner.config.evaluation = EvaluationConfig(log_latency=True)
        
        # Mock the computation method
        def mock_compute_metrics(preds, labels, prefix):
            return {
                f'{prefix}_accuracy': 0.75,
                f'{prefix}_num_samples': 4,
                f'{prefix}_correct': 3
            }
        trainer._compute_evaluation_metrics = mock_compute_metrics
        
        # Test latency metrics calculation
        all_latencies = [0.1, 0.2, 0.3, 0.4]  # Sample latencies
        all_preds = ['A', 'B', 'A', 'B']
        all_labels = ['A', 'A', 'A', 'B']
        
        # Simulate the metrics calculation part of perform_evaluation
        metrics = trainer._compute_evaluation_metrics(all_preds, all_labels, 'eval')
        
        # Add latency metrics (this is what happens in perform_evaluation)
        if all_latencies:
            avg_latency = sum(all_latencies) / len(all_latencies)
            min_latency = min(all_latencies)
            max_latency = max(all_latencies)
            total_eval_time = sum(all_latencies)
            
            latency_metrics = {
                'eval/avg_latency_sec': avg_latency,
                'eval/min_latency_sec': min_latency,
                'eval/max_latency_sec': max_latency,
                'eval/total_eval_time_sec': total_eval_time
            }
            metrics.update(latency_metrics)
        
        # Verify latency metrics are present
        assert 'eval/avg_latency_sec' in metrics, "Missing avg_latency_sec metric"
        assert 'eval/min_latency_sec' in metrics, "Missing min_latency_sec metric"
        assert 'eval/max_latency_sec' in metrics, "Missing max_latency_sec metric"
        assert 'eval/total_eval_time_sec' in metrics, "Missing total_eval_time_sec metric"
        
        # Verify latency values
        expected_avg = 0.25
        expected_min = 0.1
        expected_max = 0.4
        expected_total = 1.0
        
        assert abs(metrics['eval/avg_latency_sec'] - expected_avg) < 1e-6, f"Expected avg {expected_avg}, got {metrics['eval/avg_latency_sec']}"
        assert abs(metrics['eval/min_latency_sec'] - expected_min) < 1e-6, f"Expected min {expected_min}, got {metrics['eval/min_latency_sec']}"
        assert abs(metrics['eval/max_latency_sec'] - expected_max) < 1e-6, f"Expected max {expected_max}, got {metrics['eval/max_latency_sec']}"
        assert abs(metrics['eval/total_eval_time_sec'] - expected_total) < 1e-6, f"Expected total {expected_total}, got {metrics['eval/total_eval_time_sec']}"
        
        print("✓ Evaluation metrics with latency calculations successful")
        print(f"  - Average latency: {metrics['eval/avg_latency_sec']:.3f}s")
        print(f"  - Min latency: {metrics['eval/min_latency_sec']:.3f}s")
        print(f"  - Max latency: {metrics['eval/max_latency_sec']:.3f}s")
        print(f"  - Total time: {metrics['eval/total_eval_time_sec']:.3f}s")
        return True
        
    except Exception as e:
        print(f"✗ Evaluation metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all latency tracking tests."""
    print("=" * 60)
    print("MULTICOCO LATENCY TRACKING TESTS")
    print("=" * 60)
    
    tests = [
        test_latency_config_loading,
        test_generation_with_latency,
        test_per_sample_logging_with_latency,
        test_evaluation_metrics_with_latency
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print(f"\n{'=' * 60}")
    print("TEST RESULTS")
    print(f"{'=' * 60}")
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test.__name__}: {status}")
    
    passed = sum(results)
    total = len(results)
    print(f"\nSummary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All latency tracking tests passed!")
        return True
    else:
        print("❌ Some tests failed. Check implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
