#!/usr/bin/env python3
"""
Test script to verify evaluation logging functionality.
"""

import json
import logging
import os
import tempfile
from pathlib import Path

# Setup a test logger to simulate the evaluation_details logger behavior
def test_evaluation_logging():
    """Test the evaluation logging setup"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing logging in: {temp_dir}")
        
        # Test 1: JSON-only logging (no timestamp)
        eval_logger = logging.getLogger('test_evaluation_details')
        eval_logger.setLevel(logging.INFO)
        eval_logger.propagate = False
        
        # Clear any existing handlers
        if eval_logger.hasHandlers():
            eval_logger.handlers.clear()
        
        # Create JSON-only formatter (no timestamp)
        json_formatter = logging.Formatter('%(message)s')
        
        # Test eval-only mode - create evaluation.log
        eval_log_path = os.path.join(temp_dir, 'evaluation.log')
        eval_handler = logging.FileHandler(eval_log_path)
        eval_handler.setFormatter(json_formatter)
        eval_logger.addHandler(eval_handler)
        
        # Simulate logging some evaluation details
        test_sample = {
            "question": "What does the yellow sign advise you to watch for? The choices are 0 : pedestrians, 1 : speedbumps, 2 : dogs, 3 : deer",
            "ground_truth": "0",
            "generated_answer": "0 : pedestrians",
            "extracted_answer": "0",
            "generated_tokens": 3,
            "correct": True
        }
        
        eval_logger.info(json.dumps(test_sample))
        
        # Test 2: Epoch-specific logging
        # Clear handlers and setup epoch-specific
        eval_logger.handlers.clear()
        
        epoch = 0
        epoch_eval_log_path = os.path.join(temp_dir, f'evaluation_epoch_{epoch + 1}.log')
        epoch_eval_handler = logging.FileHandler(epoch_eval_log_path)
        epoch_eval_handler.setFormatter(json_formatter)
        eval_logger.addHandler(epoch_eval_handler)
        
        # Simulate logging for epoch
        eval_logger.info(json.dumps(test_sample))
        
        # Verify files were created and contain correct content
        print("✓ Testing evaluation.log")
        assert os.path.exists(eval_log_path), "evaluation.log was not created"
        
        with open(eval_log_path, 'r') as f:
            content = f.read().strip()
            # Verify it's just JSON without timestamp
            assert content.startswith('{"question"'), f"Expected JSON content, got: {content[:50]}"
            parsed = json.loads(content)
            assert parsed["correct"] == True, "JSON parsing failed"
        
        print("✓ Testing evaluation_epoch_1.log")
        assert os.path.exists(epoch_eval_log_path), "evaluation_epoch_1.log was not created"
        
        with open(epoch_eval_log_path, 'r') as f:
            content = f.read().strip()
            # Verify it's just JSON without timestamp
            assert content.startswith('{"question"'), f"Expected JSON content, got: {content[:50]}"
            parsed = json.loads(content)
            assert parsed["correct"] == True, "JSON parsing failed"
        
        print("✓ All tests passed!")
        print(f"evaluation.log contains: {Path(eval_log_path).read_text()[:100]}...")
        print(f"evaluation_epoch_1.log contains: {Path(epoch_eval_log_path).read_text()[:100]}...")

if __name__ == "__main__":
    test_evaluation_logging()
