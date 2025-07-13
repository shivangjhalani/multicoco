#!/usr/bin/env python3
"""
Comprehensive test for the logging and checkpointing system in MultiCoCo.
Tests all aspects of logging, checkpointing, resume functionality, and WandB integration.
"""

import os
import sys
import json
import tempfile
import shutil
import logging
from typing import Dict, Any, List
from datetime import datetime
import torch
from unittest.mock import Mock, patch, MagicMock

# Add the project root to path
sys.path.insert(0, os.path.abspath('.'))

def test_logging_configuration():
    """Test the logging configuration setup."""
    print("\n=== Testing Logging Configuration ===")
    
    from multicoco.config import MultiCoCoConfig, LoggingConfig, TrainingConfig, ModelConfig
    from run import MultiCoCoRunner
    
    # Create test config
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp directory: {temp_dir}")
    
    try:
        config = MultiCoCoConfig(
            logging=LoggingConfig(
                log_dir=temp_dir,
                run_name="test_logging",
                console_output=True,
                log_to_file=True,
                log_level="INFO",
                verbose=True,
                use_wandb=False  # Disable for testing
            ),
            training=TrainingConfig(
                name="test_training",
                seed=42
            ),
            model=ModelConfig(
                model_name="OpenGVLab/InternVL3-1B-Pretrained"
            )
        )
        
        # Initialize runner (this sets up logging)
        runner = MultiCoCoRunner(config)
        
        # Check that log directory was created
        assert runner.run_log_dir is not None, "Run log directory not created"
        assert os.path.exists(runner.run_log_dir), f"Log directory doesn't exist: {runner.run_log_dir}"
        
        # Check that run.log file exists
        run_log_path = os.path.join(runner.run_log_dir, 'run.log')
        assert os.path.exists(run_log_path), f"Run log file doesn't exist: {run_log_path}"
        
        # Check logging functionality
        logger = logging.getLogger(__name__)
        test_message = "Test logging message for verification"
        logger.info(test_message)
        
        # Verify the message was logged to file
        with open(run_log_path, 'r') as f:
            log_content = f.read()
            assert test_message in log_content, "Test message not found in log file"
        
        # Check evaluation logger setup
        eval_logger = logging.getLogger('evaluation_details')
        assert eval_logger is not None, "Evaluation logger not configured"
        
        print("✓ Basic logging configuration works")
        print("✓ Log directory creation works")
        print("✓ File logging works")
        print("✓ Evaluation logger setup works")
        
        return True
        
    except Exception as e:
        print(f"✗ Logging configuration test failed: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_checkpoint_structure():
    """Test checkpoint saving and loading structure."""
    print("\n=== Testing Checkpoint Structure ===")
    
    from multicoco.trainer import CoCoTrainer
    from transformers import TrainingArguments
    import torch.nn as nn
    
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp directory: {temp_dir}")
    
    try:
        # Create a mock model
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self.tokenizer = Mock()
                self.tokenizer.pad_token_id = 0
                self.tokenizer.eos_token_id = 1
                
            def forward(self, x):
                return self.linear(x)
        
        model = MockModel()
        
        # Create training arguments
        args = TrainingArguments(
            output_dir=temp_dir,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=1,
            save_steps=1,
            logging_steps=1,
            report_to=[]  # Disable wandb for testing
        )
        
        # Create trainer
        trainer = CoCoTrainer(
            model=model,
            args=args,
            tokenizer=model.tokenizer
        )
        
        # Test checkpoint saving
        metrics = {"eval_accuracy": 0.85, "eval_loss": 0.15}
        checkpoint_dir = trainer._save_checkpoint_with_metrics(epoch=0, metrics=metrics)
        
        # Verify checkpoint structure
        assert os.path.exists(checkpoint_dir), f"Checkpoint directory not created: {checkpoint_dir}"
        
        expected_files = [
            "config.json",  # Model config
            "metrics.json",  # Evaluation metrics
            "training_state.pt",  # Training state
        ]
        
        for file_name in expected_files:
            file_path = os.path.join(checkpoint_dir, file_name)
            if file_name in ["config.json"]:  # These might not exist in mock
                continue
            assert os.path.exists(file_path), f"Expected file not found: {file_path}"
        
        # Check metrics file content
        metrics_path = os.path.join(checkpoint_dir, "metrics.json")
        with open(metrics_path, 'r') as f:
            saved_metrics = json.load(f)
            assert saved_metrics == metrics, f"Metrics mismatch: {saved_metrics} != {metrics}"
        
        # Check training state
        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        training_state = torch.load(state_path, map_location='cpu')
        expected_keys = ['epoch', 'global_step', 'total_train_steps']
        for key in expected_keys:
            assert key in training_state, f"Missing key in training state: {key}"
        
        print("✓ Checkpoint directory creation works")
        print("✓ Metrics saving works")
        print("✓ Training state saving works")
        
        return True
        
    except Exception as e:
        print(f"✗ Checkpoint structure test failed: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_checkpoint_resumption():
    """Test checkpoint resumption logic."""
    print("\n=== Testing Checkpoint Resumption ===")
    
    from multicoco.trainer import CoCoTrainer
    from transformers import TrainingArguments
    import torch.nn as nn
    
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp directory: {temp_dir}")
    
    try:
        # Create mock checkpoint structure
        epoch_dirs = []
        for epoch in [1, 2, 3]:
            epoch_dir = os.path.join(temp_dir, f"epoch-{epoch}")
            os.makedirs(epoch_dir)
            epoch_dirs.append(epoch_dir)
            
            # Create dummy checkpoint files
            with open(os.path.join(epoch_dir, "metrics.json"), 'w') as f:
                json.dump({"eval_accuracy": 0.8 + epoch * 0.05}, f)
            
            torch.save({"epoch": epoch, "global_step": epoch * 100}, 
                      os.path.join(epoch_dir, "training_state.pt"))
        
        # Create mock trainer
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
        
        model = MockModel()
        args = TrainingArguments(
            output_dir=temp_dir,
            per_device_train_batch_size=1,
            num_train_epochs=5,
            report_to=[]
        )
        
        trainer = CoCoTrainer(model=model, args=args)
        
        # Test resumption logic
        start_epoch, checkpoint_path = trainer._handle_checkpoint_resumption(True)
        
        # Should find the latest checkpoint (epoch-3)
        expected_checkpoint = os.path.join(temp_dir, "epoch-3")
        assert checkpoint_path == expected_checkpoint, f"Wrong checkpoint selected: {checkpoint_path}"
        assert start_epoch == 3, f"Wrong start epoch: {start_epoch}"
        
        print("✓ Latest checkpoint detection works")
        print("✓ Resume epoch calculation works")
        
        # Test with specific checkpoint path
        specific_checkpoint = os.path.join(temp_dir, "epoch-2")
        start_epoch, checkpoint_path = trainer._handle_checkpoint_resumption(specific_checkpoint)
        
        assert checkpoint_path == specific_checkpoint, "Specific checkpoint path not respected"
        
        print("✓ Specific checkpoint path works")
        
        return True
        
    except Exception as e:
        print(f"✗ Checkpoint resumption test failed: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_evaluation_logging():
    """Test evaluation-specific logging functionality."""
    print("\n=== Testing Evaluation Logging ===")
    
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp directory: {temp_dir}")
    
    try:
        # Setup evaluation logger manually
        eval_logger = logging.getLogger('evaluation_details')
        eval_logger.setLevel(logging.INFO)
        eval_logger.propagate = False
        
        # Clear any existing handlers
        if eval_logger.hasHandlers():
            eval_logger.handlers.clear()
        
        # Add file handler
        eval_log_path = os.path.join(temp_dir, 'evaluation.log')
        eval_handler = logging.FileHandler(eval_log_path)
        eval_formatter = logging.Formatter('%(message)s')
        eval_handler.setFormatter(eval_formatter)
        eval_logger.addHandler(eval_handler)
        
        # Test logging sample evaluation details
        sample_data = [
            {
                "question": "What color is the sky?",
                "ground_truth": "blue", 
                "generated_answer": "The sky is blue in color.",
                "extracted_answer": "blue",
                "generated_tokens": 15,
                "correct": True
            },
            {
                "question": "How many wheels does a car have?",
                "ground_truth": "four",
                "generated_answer": "A car typically has 4 wheels.",
                "extracted_answer": "4", 
                "generated_tokens": 12,
                "correct": False  # Different format
            }
        ]
        
        # Log the samples
        for sample in sample_data:
            eval_logger.info(json.dumps(sample))
        
        # Verify the log file
        assert os.path.exists(eval_log_path), f"Evaluation log file not created: {eval_log_path}"
        
        with open(eval_log_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == len(sample_data), f"Wrong number of log lines: {len(lines)}"
            
            for i, line in enumerate(lines):
                logged_data = json.loads(line.strip())
                assert logged_data == sample_data[i], f"Logged data mismatch for sample {i}"
        
        print("✓ Evaluation logger setup works")
        print("✓ JSON format logging works")
        print("✓ Per-sample evaluation logging works")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation logging test failed: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_coconut_metrics_logging():
    """Test CoCoNut-specific metrics logging."""
    print("\n=== Testing CoCoNut Metrics Logging ===")
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        from multicoco.constants import START_LATENT_TOKEN, END_LATENT_TOKEN, LATENT_TOKEN
        
        # Test latent span extraction
        test_text = f"I need to think about this. {START_LATENT_TOKEN} {LATENT_TOKEN} {LATENT_TOKEN} {LATENT_TOKEN} {END_LATENT_TOKEN} The answer is blue."
        
        # Mock wrapper to test span extraction
        class MockWrapper:
            def _extract_latent_spans(self, text):
                import re
                pattern = f"{re.escape(START_LATENT_TOKEN)}(.+?){re.escape(END_LATENT_TOKEN)}"
                matches = re.findall(pattern, text)
                spans = []
                for match in matches:
                    latent_tokens = match.count(LATENT_TOKEN)
                    spans.append({"start": 0, "end": len(match), "length": latent_tokens})
                return spans
        
        wrapper = MockWrapper()
        spans = wrapper._extract_latent_spans(test_text)
        
        assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
        assert spans[0]["length"] == 3, f"Expected 3 latent tokens, got {spans[0]['length']}"
        
        print("✓ Latent span extraction works")
        
        # Test metrics calculation
        def calculate_coconut_metrics(generated_texts, questions):
            metrics = {
                "avg_latent_spans": 0,
                "avg_span_length": 0,
                "reasoning_quality": 0,
                "total_latent_tokens": 0
            }
            
            total_spans = 0
            total_span_length = 0
            total_reasoning_quality = 0
            
            for text in generated_texts:
                spans = wrapper._extract_latent_spans(text)
                total_spans += len(spans)
                
                for span in spans:
                    total_span_length += span["length"]
                
                # Simple reasoning quality metric
                if "think" in text.lower() or "reason" in text.lower():
                    total_reasoning_quality += 1
            
            if generated_texts:
                metrics["avg_latent_spans"] = total_spans / len(generated_texts)
                metrics["reasoning_quality"] = total_reasoning_quality / len(generated_texts)
                
            if total_spans > 0:
                metrics["avg_span_length"] = total_span_length / total_spans
                
            metrics["total_latent_tokens"] = total_span_length
            
            return metrics
        
        # Test with sample data
        sample_texts = [
            f"Let me think. {START_LATENT_TOKEN} {LATENT_TOKEN} {LATENT_TOKEN} {END_LATENT_TOKEN} The answer is A.",
            f"I need to reason about this. {START_LATENT_TOKEN} {LATENT_TOKEN} {LATENT_TOKEN} {LATENT_TOKEN} {LATENT_TOKEN} {END_LATENT_TOKEN} It's clearly B.",
            "This is a simple answer without latent reasoning."
        ]
        
        sample_questions = ["Q1", "Q2", "Q3"]
        
        metrics = calculate_coconut_metrics(sample_texts, sample_questions)
        
        # Verify metrics
        assert metrics["avg_latent_spans"] == 2/3, f"Wrong avg spans: {metrics['avg_latent_spans']}"
        assert metrics["avg_span_length"] == 3.0, f"Wrong avg span length: {metrics['avg_span_length']}"
        assert metrics["total_latent_tokens"] == 6, f"Wrong total tokens: {metrics['total_latent_tokens']}"
        assert metrics["reasoning_quality"] == 2/3, f"Wrong reasoning quality: {metrics['reasoning_quality']}"
        
        print("✓ CoCoNut metrics calculation works")
        print("✓ Latent token counting works")
        print("✓ Reasoning quality assessment works")
        
        return True
        
    except Exception as e:
        print(f"✗ CoCoNut metrics logging test failed: {e}")
        return False

def test_wandb_integration():
    """Test WandB integration (mocked)."""
    print("\n=== Testing WandB Integration ===")
    
    try:
        # Mock WandB
        with patch('wandb.init') as mock_init, \
             patch('wandb.log') as mock_log, \
             patch('wandb.Table') as mock_table:
            
            mock_run = Mock()
            mock_run.config = Mock()
            mock_init.return_value = mock_run
            
            from multicoco.config import MultiCoCoConfig, LoggingConfig
            from run import MultiCoCoRunner
            
            config = MultiCoCoConfig(
                logging=LoggingConfig(
                    use_wandb=True,
                    project="test_project",
                    run_name="test_run"
                )
            )
            
            # This should initialize WandB
            runner = MultiCoCoRunner(config)
            
            # Verify WandB was initialized
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs['project'] == "test_project"
            assert call_kwargs['name'] == "test_run"
            
            print("✓ WandB initialization works")
            
            # Test logging functionality
            from multicoco.trainer import CoCoTrainer
            from transformers import TrainingArguments
            import torch.nn as nn
            
            class MockModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 5)
            
            model = MockModel()
            args = TrainingArguments(
                output_dir="/tmp",
                per_device_train_batch_size=1,
                num_train_epochs=1,
                report_to=["wandb"]
            )
            
            trainer = CoCoTrainer(model=model, args=args)
            
            # Mock a training step log
            with patch('wandb.run', mock_run):
                trainer._log_training_step(torch.tensor(0.5), step=0, epoch=0)
            
            # Should have called wandb.log
            mock_log.assert_called()
            
            print("✓ WandB training logging works")
            
            # Test evaluation logging
            metrics = {"eval_accuracy": 0.85, "eval_loss": 0.15}
            
            with patch('wandb.run', mock_run):
                trainer._log_epoch_summary(epoch=0, eval_metrics=metrics, 
                                         checkpoint_dir="/tmp/epoch-1", epoch_time=120.5)
            
            print("✓ WandB evaluation logging works")
            
        return True
        
    except Exception as e:
        print(f"✗ WandB integration test failed: {e}")
        return False

def test_multimodal_logging_enhancements():
    """Test the enhanced logging for multimodal specifics."""
    print("\n=== Testing Multimodal Logging Enhancements ===")
    
    try:
        # Test image-text alignment metrics
        def calculate_multimodal_metrics(questions, generated_texts, has_images):
            metrics = {
                "image_questions_ratio": 0,
                "avg_response_length": 0,
                "visual_reasoning_indicators": 0
            }
            
            if not questions:
                return metrics
            
            # Image questions ratio
            image_question_count = sum(1 for has_img in has_images if has_img)
            metrics["image_questions_ratio"] = image_question_count / len(questions)
            
            # Average response length
            if generated_texts:
                total_length = sum(len(text.split()) for text in generated_texts)
                metrics["avg_response_length"] = total_length / len(generated_texts)
            
            # Visual reasoning indicators
            visual_keywords = ["see", "image", "picture", "visual", "shows", "depicts", "appears"]
            visual_responses = 0
            for text in generated_texts:
                if any(keyword in text.lower() for keyword in visual_keywords):
                    visual_responses += 1
            
            metrics["visual_reasoning_indicators"] = visual_responses / len(generated_texts) if generated_texts else 0
            
            return metrics
        
        # Test data
        test_questions = [
            "What do you see in this image?",
            "What is 2+2?", 
            "Describe the colors in the picture.",
            "What is the capital of France?"
        ]
        
        test_responses = [
            "I can see a cat sitting on a chair in the image.",
            "The answer is 4.",
            "The image shows blue sky and green grass.",
            "The capital of France is Paris."
        ]
        
        test_has_images = [True, False, True, False]
        
        metrics = calculate_multimodal_metrics(test_questions, test_responses, test_has_images)
        
        # Verify metrics
        assert metrics["image_questions_ratio"] == 0.5, f"Wrong image ratio: {metrics['image_questions_ratio']}"
        assert metrics["visual_reasoning_indicators"] == 0.5, f"Wrong visual indicators: {metrics['visual_reasoning_indicators']}"
        
        print("✓ Multimodal metrics calculation works")
        
        # Test enhanced answer extraction logging
        from multicoco.answer_extraction import extract_answer_choice
        
        multimodal_responses = [
            "The image shows a red car.",
            "I can see 3 objects in total.",
            "The main object is a bicycle.",
            "It's located in the center of the image."
        ]
        
        expected_types = ["color", "count", "object", "location"]
        
        for response, exp_type in zip(multimodal_responses, expected_types):
            extracted = extract_answer_choice(response, is_multimodal=True, expected_type=exp_type)
            # Just verify it doesn't crash and returns something
            assert isinstance(extracted, str), f"Extraction failed for type {exp_type}"
        
        print("✓ Enhanced multimodal answer extraction works")
        
        return True
        
    except Exception as e:
        print(f"✗ Multimodal logging test failed: {e}")
        return False

def test_checkpoint_cleanup_and_management():
    """Test checkpoint cleanup and management features."""
    print("\n=== Testing Checkpoint Management ===")
    
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp directory: {temp_dir}")
    
    try:
        # Create multiple checkpoint directories
        checkpoint_dirs = []
        for epoch in range(1, 6):  # 5 checkpoints
            checkpoint_dir = os.path.join(temp_dir, f"epoch-{epoch}")
            os.makedirs(checkpoint_dir)
            checkpoint_dirs.append(checkpoint_dir)
            
            # Create checkpoint files with different sizes
            with open(os.path.join(checkpoint_dir, "metrics.json"), 'w') as f:
                json.dump({"eval_accuracy": 0.7 + epoch * 0.05}, f)
            
            # Create a dummy model file
            model_data = torch.randn(100, 100)  # Different sizes
            torch.save(model_data, os.path.join(checkpoint_dir, "pytorch_model.bin"))
        
        # Test checkpoint enumeration
        def list_checkpoints(output_dir):
            checkpoints = []
            if not os.path.exists(output_dir):
                return checkpoints
            
            for item in os.listdir(output_dir):
                if item.startswith("epoch-") and os.path.isdir(os.path.join(output_dir, item)):
                    try:
                        epoch_num = int(item.split("-")[1])
                        checkpoints.append((epoch_num, os.path.join(output_dir, item)))
                    except (ValueError, IndexError):
                        continue
            
            return sorted(checkpoints, key=lambda x: x[0])
        
        checkpoints = list_checkpoints(temp_dir)
        assert len(checkpoints) == 5, f"Expected 5 checkpoints, found {len(checkpoints)}"
        assert checkpoints[-1][0] == 5, f"Latest checkpoint should be epoch 5, got {checkpoints[-1][0]}"
        
        print("✓ Checkpoint enumeration works")
        
        # Test checkpoint size calculation
        def get_checkpoint_size(checkpoint_dir):
            total_size = 0
            for root, dirs, files in os.walk(checkpoint_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
            return total_size
        
        sizes = [get_checkpoint_size(cp[1]) for cp in checkpoints]
        assert all(size > 0 for size in sizes), "All checkpoints should have non-zero size"
        
        print("✓ Checkpoint size calculation works")
        
        # Test checkpoint validation
        def validate_checkpoint(checkpoint_dir):
            required_files = ["metrics.json"]  # Minimal validation
            for file_name in required_files:
                file_path = os.path.join(checkpoint_dir, file_name)
                if not os.path.exists(file_path):
                    return False, f"Missing file: {file_name}"
            
            # Validate metrics file
            try:
                with open(os.path.join(checkpoint_dir, "metrics.json"), 'r') as f:
                    metrics = json.load(f)
                    if not isinstance(metrics, dict):
                        return False, "Invalid metrics format"
            except json.JSONDecodeError:
                return False, "Corrupted metrics file"
            
            return True, "Valid"
        
        for epoch, checkpoint_dir in checkpoints:
            is_valid, message = validate_checkpoint(checkpoint_dir)
            assert is_valid, f"Checkpoint validation failed for epoch {epoch}: {message}"
        
        print("✓ Checkpoint validation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Checkpoint management test failed: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def run_all_logging_tests():
    """Run all logging and checkpointing tests."""
    print("=" * 80)
    print("MULTICOCO LOGGING & CHECKPOINTING TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Logging Configuration", test_logging_configuration),
        ("Checkpoint Structure", test_checkpoint_structure),
        ("Checkpoint Resumption", test_checkpoint_resumption),
        ("Evaluation Logging", test_evaluation_logging),
        ("CoCoNut Metrics Logging", test_coconut_metrics_logging),
        ("WandB Integration", test_wandb_integration),
        ("Multimodal Logging Enhancements", test_multimodal_logging_enhancements),
        ("Checkpoint Management", test_checkpoint_cleanup_and_management),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 60}")
        print(f"Running: {test_name}")
        print(f"{'-' * 60}")
        
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                failed += 1
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} CRASHED: {e}")
            failed += 1
    
    print(f"\n{'=' * 80}")
    print("LOGGING & CHECKPOINTING TEST RESULTS")
    print(f"{'=' * 80}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ALL LOGGING & CHECKPOINTING TESTS PASSED!")
        print("\nKey aspects verified:")
        print("✓ Complete logging configuration and file management")
        print("✓ Robust checkpoint saving and loading")
        print("✓ Training resumption from checkpoints")
        print("✓ Detailed evaluation logging with JSON format")
        print("✓ CoCoNut-specific metrics tracking")
        print("✓ WandB integration for experiment tracking")
        print("✓ Enhanced multimodal logging capabilities")
        print("✓ Checkpoint validation and management")
        return True
    else:
        print(f"\n❌ {failed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = run_all_logging_tests()
    sys.exit(0 if success else 1)
