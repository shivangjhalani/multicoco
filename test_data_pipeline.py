#!/usr/bin/env python3
"""
Test script to verify the entire data pipeline works end-to-end.
"""

import sys
import os
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multicoco.data import SupervisedDataset
from multicoco.config import CoconutConfig
from multicoco.constants import IMAGE_TOKEN, COCONUT_SPECIAL_TOKENS

def test_data_pipeline_end_to_end():
    """Test the complete data pipeline from dataset to model input."""
    print("Testing end-to-end data pipeline...")
    
    # Create a simple test dataset file
    test_data = [
        {
            "image": "test_image_1.jpg",
            "question": "What is in the image?",
            "answer": "A cat",
            "direct_answer": "cat",
            "steps": ["I see an animal", "It has whiskers", "It must be a cat"]
        },
        {
            "image": "test_image_2.jpg", 
            "question": "What color is it?",
            "answer": "Blue",
            "direct_answer": "blue",
            "steps": ["Looking at the color", "It appears blue"]
        }
    ]
    
    import json
    test_file = "/tmp/test_aokvqa.json"
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    
    try:
        # Test with Coconut configuration
        coconut_config = CoconutConfig(
            enabled=True,
            n_latent_tokens=4,
            progressive_stages=[1, 2],
            current_stage=1
        )
        
        # Create dataset with progressive curriculum
        dataset = SupervisedDataset(
            data_path=test_file,
            data_dir="/tmp"  # Dummy path
        )
        
        # Apply coconut progressive curriculum
        dataset.apply_progressive_curriculum(
            scheduled_stage=coconut_config.current_stage,
            c_thought=coconut_config.n_latent_tokens,
            max_latent_stage=max(coconut_config.progressive_stages),
            uniform_prob=0.0,
            pad_latent_to_max=False,
            no_cot=False
        )
        
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Test dataset access
        sample = dataset[0]
        print(f"✓ Sample accessed successfully")
        
        # Verify all expected fields are present
        expected_fields = ['image', 'question', 'reasoning', 'answer']
        for field in expected_fields:
            assert field in sample, f"Missing field '{field}' in sample"
        print(f"✓ All expected fields present: {expected_fields}")
        
        # Verify reasoning contains latent tokens
        reasoning = sample['reasoning']
        assert '<|start_latent|>' in reasoning, "Missing start latent token"
        assert '<|latent|>' in reasoning, "Missing latent tokens"
        assert '<|end_latent|>' in reasoning, "Missing end latent token"
        print(f"✓ Reasoning contains latent tokens: {reasoning[:100]}...")
        
        # Test collate function with DataLoader
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        tokenizer.add_special_tokens({'additional_special_tokens': COCONUT_SPECIAL_TOKENS})
        tokenizer.pad_token = tokenizer.eos_token
        
        # Override collate_fn for testing
        def test_collate_fn(batch):
            from multicoco.data import collate_fn
            return collate_fn(batch, tokenizer)
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=test_collate_fn,
            shuffle=False
        )
        
        # Test batch processing
        batch = next(iter(dataloader))
        print(f"✓ Batch processed successfully")
        
        # Verify batch structure
        expected_batch_keys = ['pixel_values', 'input_ids', 'attention_mask', 'labels']
        for key in expected_batch_keys:
            assert key in batch, f"Missing batch key '{key}'"
        print(f"✓ Batch has expected keys: {list(batch.keys())}")
        
        # Verify batch shapes
        batch_size = len(test_data)
        print(f"✓ Batch size: {batch_size}")
        print(f"✓ Input IDs shape: {batch['input_ids'].shape}")
        print(f"✓ Attention mask shape: {batch['attention_mask'].shape}")
        print(f"✓ Labels shape: {batch['labels'].shape}")
        
        # Check that IMAGE_TOKEN is properly handled
        input_text = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
        assert IMAGE_TOKEN in input_text, f"IMAGE_TOKEN '{IMAGE_TOKEN}' not found in input text"
        print(f"✓ IMAGE_TOKEN '{IMAGE_TOKEN}' found in tokenized input")
        
        # Check that latent tokens are preserved in tokenization
        assert '<|start_latent|>' in input_text, "Start latent token lost in tokenization"
        assert '<|latent|>' in input_text, "Latent tokens lost in tokenization"
        assert '<|end_latent|>' in input_text, "End latent token lost in tokenization"
        print(f"✓ Latent tokens preserved in tokenization")
        
        return True
        
    except Exception as e:
        print(f"✗ Data pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)

def test_latent_token_detection():
    """Test that the tokenizer properly detects latent tokens."""
    print("\nTesting latent token detection...")
    
    try:
        from transformers import AutoTokenizer
        
        # Create tokenizer with special tokens
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        tokenizer.add_special_tokens({'additional_special_tokens': COCONUT_SPECIAL_TOKENS})
        tokenizer.pad_token = tokenizer.eos_token
        
        # Test text with latent tokens
        test_text = "Question: What is this? <|start_latent|> <|latent|> <|latent|> <|end_latent|> Answer: A cat"
        
        # Tokenize
        tokens = tokenizer.tokenize(test_text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # Check that latent tokens have proper IDs
        latent_start_id = tokenizer.convert_tokens_to_ids('<|start_latent|>')
        latent_token_id = tokenizer.convert_tokens_to_ids('<|latent|>')
        latent_end_id = tokenizer.convert_tokens_to_ids('<|end_latent|>')
        
        assert latent_start_id in token_ids, "Start latent token not found in token IDs"
        assert latent_token_id in token_ids, "Latent token not found in token IDs"
        assert latent_end_id in token_ids, "End latent token not found in token IDs"
        
        print(f"✓ Latent token IDs: start={latent_start_id}, latent={latent_token_id}, end={latent_end_id}")
        print(f"✓ All latent tokens properly detected in tokenization")
        
        return True
        
    except Exception as e:
        print(f"✗ Latent token detection test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing MultiCoCo Data Pipeline")
    print("=" * 60)
    
    success = True
    success &= test_data_pipeline_end_to_end()
    success &= test_latent_token_detection()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All data pipeline tests passed!")
    else:
        print("✗ Some tests failed. Check the output above.")
    
    sys.exit(0 if success else 1)
