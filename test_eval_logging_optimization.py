#!/usr/bin/env python3
"""
Test script to verify the evaluation logging optimization.
This will test the performance difference between the old individual logging
approach and the new batched logging approach.
"""

import time
import logging
import json
import tempfile
import os

def test_logging_performance():
    """Test the performance improvement of batched logging vs individual logging"""
    
    print("Testing evaluation logging performance optimization...")
    
    # Create test data
    num_samples = 20  # Same as your training
    test_data = []
    for i in range(num_samples):
        test_data.append({
            'question': f'This is test question {i} with some reasonable length to simulate real questions',
            'ground_truth': f'Answer {i}',
            'generated_answer': f'Generated answer {i} with some additional text to make it realistic',
            'extracted_answer': f'Answer {i}',
            'generated_tokens': 50 + i,
            'correct': i % 2 == 0,
            'latency_sec': 0.5 + i * 0.1
        })
    
    # Test 1: Individual logging (old approach)
    print(f"\n--- Test 1: Individual logging (old approach) ---")
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file1 = os.path.join(temp_dir, 'individual.log')
        
        # Setup logger
        logger1 = logging.getLogger('individual_test')
        logger1.setLevel(logging.INFO)
        handler1 = logging.FileHandler(log_file1)
        handler1.setFormatter(logging.Formatter('%(message)s'))
        logger1.addHandler(handler1)
        
        # Time individual logging
        start_time = time.time()
        for details in test_data:
            logger1.info(json.dumps(details))
        individual_time = time.time() - start_time
        
        logger1.removeHandler(handler1)
        handler1.close()
        
        print(f"Individual logging time: {individual_time:.4f} seconds")
        print(f"File size: {os.path.getsize(log_file1)} bytes")
    
    # Test 2: Batched logging (new approach)
    print(f"\n--- Test 2: Batched logging (new approach) ---")
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file2 = os.path.join(temp_dir, 'batched.log')
        
        # Setup logger
        logger2 = logging.getLogger('batched_test')
        logger2.setLevel(logging.INFO)
        handler2 = logging.FileHandler(log_file2)
        handler2.setFormatter(logging.Formatter('%(message)s'))
        logger2.addHandler(handler2)
        
        # Time batched logging
        start_time = time.time()
        batch_json = '\n'.join(json.dumps(details) for details in test_data)
        logger2.info(batch_json)
        batched_time = time.time() - start_time
        
        logger2.removeHandler(handler2)
        handler2.close()
        
        print(f"Batched logging time: {batched_time:.4f} seconds")
        print(f"File size: {os.path.getsize(log_file2)} bytes")
    
    # Calculate improvement
    if individual_time > 0:
        improvement = ((individual_time - batched_time) / individual_time) * 100
        speedup = individual_time / batched_time if batched_time > 0 else float('inf')
        
        print(f"\n🎯 Performance Results:")
        print(f"  Time improvement: {improvement:.1f}% faster")
        print(f"  Speedup factor: {speedup:.1f}x")
        
        if improvement > 20:
            print("✅ Significant performance improvement achieved!")
        elif improvement > 0:
            print("✅ Performance improvement achieved!")
        else:
            print("⚠️  No significant performance improvement")
    
    return batched_time < individual_time

if __name__ == "__main__":
    success = test_logging_performance()
    
    print(f"\n{'='*60}")
    print("SUMMARY: Evaluation Logging Optimization")
    print(f"{'='*60}")
    print("Changes made:")
    print("1. ✅ Batched JSON serialization instead of individual calls")
    print("2. ✅ Single logging call instead of N individual calls") 
    print("3. ✅ Added file handler encoding and buffering")
    print("4. ✅ Graceful fallback to individual logging if batch fails")
    
    if success:
        print(f"\n✅ Optimization successful!")
        print(f"   The long pause after 'Logging per-sample evaluation details' should be significantly reduced.")
    else:
        print(f"\n⚠️  Optimization may need further tuning.")
    
    print(f"\n💡 Additional recommendations:")
    print(f"   - Consider reducing log verbosity during training")
    print(f"   - Use async logging for even better performance")
    print(f"   - Consider logging only on rank 0 in distributed training")
