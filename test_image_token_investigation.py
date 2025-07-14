#!/usr/bin/env python3
"""
Test script to investigate the image token count mismatch issue.
This script will test what the actual num_image_token should be for InternVL3-1B.
"""

import sys
import torch
from pathlib import Path

# Add the parent directory to the path so we can import multicoco as a package
sys.path.insert(0, str(Path(__file__).parent))

def test_internvl_image_token_count():
    """Test what the actual image token count should be for InternVL3-1B."""
    print("=" * 60)
    print("Testing InternVL3-1B Image Token Count")
    print("=" * 60)
    
    try:
        # Try to load the InternVL3-1B model to inspect its config
        from transformers import AutoModel, AutoTokenizer
        
        model_name = "OpenGVLab/InternVL3-1B-Pretrained"
        print(f"Loading model: {model_name}")
        
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"✓ Tokenizer loaded")
        
        # Load model
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cpu"  # Use CPU to avoid GPU memory issues
        )
        print(f"✓ Model loaded")
        
        # Check various possible sources of num_image_token
        print("\n" + "=" * 40)
        print("Investigating num_image_token sources:")
        print("=" * 40)
        
        # 1. Direct attribute on model
        if hasattr(model, 'num_image_token'):
            print(f"✓ model.num_image_token = {model.num_image_token}")
        else:
            print("✗ model.num_image_token not found")
        
        # 2. Config attribute
        if hasattr(model, 'config') and hasattr(model.config, 'num_image_token'):
            print(f"✓ model.config.num_image_token = {model.config.num_image_token}")
        else:
            print("✗ model.config.num_image_token not found")
        
        # 3. Vision config
        if hasattr(model, 'config') and hasattr(model.config, 'vision_config'):
            vision_config = model.config.vision_config
            print(f"✓ Found vision_config")
            
            if hasattr(vision_config, 'image_size'):
                print(f"  - vision_config.image_size = {vision_config.image_size}")
            if hasattr(vision_config, 'patch_size'):
                print(f"  - vision_config.patch_size = {vision_config.patch_size}")
            
            # Calculate expected tokens
            if hasattr(vision_config, 'image_size') and hasattr(vision_config, 'patch_size'):
                image_size = vision_config.image_size
                patch_size = vision_config.patch_size
                expected_tokens = (image_size // patch_size) ** 2
                print(f"  - Expected base tokens: ({image_size}//{patch_size})^2 = {expected_tokens}")
        else:
            print("✗ model.config.vision_config not found")
        
        # 4. Downsample ratio
        if hasattr(model, 'config') and hasattr(model.config, 'downsample_ratio'):
            downsample_ratio = model.config.downsample_ratio
            print(f"✓ model.config.downsample_ratio = {downsample_ratio}")
            
            # Calculate tokens with downsample ratio
            if hasattr(model.config, 'vision_config'):
                vision_config = model.config.vision_config
                if hasattr(vision_config, 'image_size') and hasattr(vision_config, 'patch_size'):
                    image_size = vision_config.image_size
                    patch_size = vision_config.patch_size
                    base_tokens = (image_size // patch_size) ** 2
                    final_tokens = int(base_tokens * (downsample_ratio ** 2))
                    print(f"  - With downsample: {base_tokens} * {downsample_ratio}^2 = {final_tokens}")
        else:
            print("✗ model.config.downsample_ratio not found")
        
        # 5. Test actual vision output
        print("\n" + "=" * 40)
        print("Testing actual vision output:")
        print("=" * 40)
        
        # Create a dummy image to test
        import torch
        dummy_image = torch.randn(1, 3, 448, 448, dtype=torch.bfloat16)
        
        try:
            # Use the model's extract_feature method if available
            if hasattr(model, 'extract_feature'):
                print("Testing model.extract_feature()...")
                with torch.inference_mode():
                    vision_embeds = model.extract_feature(dummy_image)
                print(f"✓ Vision embeddings shape: {vision_embeds.shape}")
                print(f"✓ Actual vision token count: {vision_embeds.shape[1]}")
            else:
                print("✗ model.extract_feature() not available")
        except Exception as e:
            print(f"✗ Error testing vision output: {e}")
        
        # 6. Check tokenizer
        print("\n" + "=" * 40)
        print("Checking tokenizer:")
        print("=" * 40)
        
        if hasattr(tokenizer, 'model') and hasattr(tokenizer.model, 'num_image_token'):
            print(f"✓ tokenizer.model.num_image_token = {tokenizer.model.num_image_token}")
        else:
            print("✗ tokenizer.model.num_image_token not found")
        
        # 7. Check IMG_CONTEXT token
        img_context_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        if img_context_id != tokenizer.unk_token_id:
            print(f"✓ <IMG_CONTEXT> token id: {img_context_id}")
        else:
            print("✗ <IMG_CONTEXT> token not found in vocabulary")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("This might be due to missing dependencies or network issues.")
        return False

def test_mismatch_simulation():
    """Simulate the mismatch between hardcoded 256 and actual token count."""
    print("\n" + "=" * 60)
    print("Simulating Token Count Mismatch")
    print("=" * 60)
    
    # Common scenarios
    scenarios = [
        ("Hardcoded fallback", 256),
        ("InternVL3-1B actual (estimated)", 784),  # 28x28 patches  
        ("Alternative calculation", 1024),  # 32x32 patches
    ]
    
    for name, count in scenarios:
        print(f"\n{name}: {count} tokens")
        prompt_length = len("<img>" + "<IMG_CONTEXT>" * count + "</img>Question here")
        print(f"  - Prompt with {count} IMG_CONTEXT tokens: {prompt_length} chars")
        print(f"  - Memory impact: ~{count * 4 * 1024} bytes (assuming 1024-dim embeddings)")
    
    print(f"\n⚠️  Mismatch impact:")
    print(f"   - If prompt has 256 tokens but model produces 784:")
    print(f"     → assertion failure or silent truncation")
    print(f"     → loss of {784-256} = 528 vision tokens ({(784-256)/784*100:.1f}% visual info lost)")

if __name__ == "__main__":
    print("InternVL3-1B Image Token Count Investigation")
    print("This script investigates the image token count mismatch issue.")
    
    # Test 1: Try to load actual model and check config
    model_loaded = test_internvl_image_token_count()
    
    # Test 2: Simulate the mismatch scenario
    test_mismatch_simulation()
    
    print("\n" + "=" * 60)
    if model_loaded:
        print("✅ Investigation complete. Check the actual token counts above.")
    else:
        print("⚠️  Could not load model, but mismatch simulation completed.")
    print("Use the findings to implement the proper fix.")
    print("=" * 60)
