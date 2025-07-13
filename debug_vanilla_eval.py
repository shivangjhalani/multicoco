#!/usr/bin/env python3
"""
Debug script to isolate the shape mismatch issue in vanilla evaluation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import logging
from PIL import Image
import tempfile
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_model_loading():
    """Test basic model loading without our wrapper."""
    print("=== Testing Basic Model Loading ===")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
        
        model_name = "OpenGVLab/InternVL3-1B-Pretrained"
        
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        print(f"Loading image processor: {model_name}")
        image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        print("✓ Basic model loading successful")
        print(f"Model config: {type(model.config)}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        
        return model, tokenizer, image_processor
        
    except Exception as e:
        print(f"✗ Basic model loading failed: {e}")
        return None, None, None

def test_simple_inference():
    """Test simple inference with the model."""
    print("\n=== Testing Simple Inference ===")
    
    try:
        model, tokenizer, image_processor = test_basic_model_loading()
        if model is None:
            return False
        
        # Create a simple test image
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # Prepare simple inputs
        question = "What do you see in this image?"
        
        print("Processing inputs...")
        
        # Try the chat interface
        try:
            generation_config = {
                'max_new_tokens': 32,
                'do_sample': False,
                'num_beams': 1
            }
            
            pixel_values = image_processor(test_image, return_tensors='pt')['pixel_values']
            print(f"Image tensor shape: {pixel_values.shape}")
            
            # Use model.chat method directly
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values.to(model.dtype),
                question=question,
                generation_config=generation_config
            )
            
            print(f"✓ Simple inference successful")
            print(f"Response: {response}")
            return True
            
        except Exception as e:
            print(f"✗ Simple inference failed: {e}")
            print(f"Error type: {type(e)}")
            
            # Try to get more details about the error
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"✗ Test setup failed: {e}")
        return False

def test_our_wrapper():
    """Test our MultiCoCo wrapper."""
    print("\n=== Testing Our MultiCoCo Wrapper ===")
    
    try:
        from multicoco.model import MultiCoCo
        
        # Test with no special tokens (vanilla mode)
        model = MultiCoCo(
            model_id="OpenGVLab/InternVL3-1B-Pretrained",
            special_tokens=[],  # No special tokens for vanilla
            torch_dtype="bfloat16",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print("✓ MultiCoCo wrapper initialization successful")
        
        # Create test data
        test_image = Image.new('RGB', (224, 224), color='blue')
        question = "What color is this image?"
        
        # Test our wrapper's functionality
        pixel_values = model.image_processor(test_image, return_tensors='pt')['pixel_values']
        print(f"Processed image shape: {pixel_values.shape}")
        print(f"Image dtype: {pixel_values.dtype}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        print(f"Model device: {model.device}")
        
        # Ensure dtype and device consistency
        pixel_values = pixel_values.to(dtype=next(model.parameters()).dtype, device=model.device)
        print(f"Corrected image dtype: {pixel_values.dtype}")
        print(f"Corrected image device: {pixel_values.device}")
        
        generation_config = {
            'max_new_tokens': 32,
            'do_sample': False,
            'num_beams': 1
        }
        
        # Try the chat method through our wrapper
        response = model.model.chat(
            tokenizer=model.tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config=generation_config
        )
        
        print(f"✓ MultiCoCo wrapper inference successful")
        print(f"Response: {response}")
        return True
        
    except Exception as e:
        print(f"✗ MultiCoCo wrapper test failed: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return False

def main():
    """Run all debug tests."""
    print("=" * 60)
    print("MULTICOCO VANILLA EVALUATION DEBUG")
    print("=" * 60)
    
    # Test basic model loading
    success1 = test_basic_model_loading() is not None
    
    # Test simple inference
    success2 = test_simple_inference()
    
    # Test our wrapper
    success3 = test_our_wrapper()
    
    print(f"\n{'=' * 60}")
    print("DEBUG RESULTS")
    print(f"{'=' * 60}")
    print(f"Basic Loading: {'✓' if success1 else '✗'}")
    print(f"Simple Inference: {'✓' if success2 else '✗'}")
    print(f"MultiCoCo Wrapper: {'✓' if success3 else '✗'}")
    
    if all([success1, success2, success3]):
        print("\n🎉 All debug tests passed! The issue might be in the evaluation pipeline.")
    else:
        print(f"\n❌ Some debug tests failed. This helps isolate the issue.")
    
    return all([success1, success2, success3])

if __name__ == "__main__":
    main()
