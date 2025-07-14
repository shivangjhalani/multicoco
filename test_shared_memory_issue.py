#!/usr/bin/env python3
"""
Test script to reproduce and fix the shared memory issue in LatentWrapper.

The issue occurs when saving the model because LatentWrapper's embedding layer
and the base model's embedding layer reference the same tensor, causing
safetensors to warn about duplicate memory on disk.

Error:
    Some tensors share memory, this will lead to duplicate memory on disk 
    and potential differences when loading them again: 
    [{'embedding.weight', 'base_model.model.language_model.model.embed_tokens.weight'}].
"""

import sys
import os
import torch
import tempfile
from pathlib import Path

# Add project path
sys.path.append('/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco')

from transformers import AutoTokenizer, AutoModel
from multicoco.latent_wrapper import LatentWrapper

def test_shared_memory_issue():
    """Test to reproduce the shared memory issue"""
    print("=" * 80)
    print("TESTING SHARED MEMORY ISSUE")
    print("=" * 80)
    
    # Load model and tokenizer
    model_name = "OpenGVLab/InternVL3-1B-Pretrained"
    print(f"Loading model: {model_name}")
    
    try:
        model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        print("✓ Base model loaded successfully")
        
        # Create LatentWrapper
        wrapper = LatentWrapper(model, tokenizer)
        print("✓ LatentWrapper created successfully")
        
        # Check if embeddings share memory
        print("\n" + "=" * 40)
        print("MEMORY SHARING ANALYSIS")
        print("=" * 40)
        
        base_embedding = model.model.language_model.model.embed_tokens
        wrapper_embedding = wrapper.embedding
        
        print(f"Base model embedding: {type(base_embedding)}")
        print(f"Wrapper embedding: {type(wrapper_embedding)}")
        print(f"Same object (id): {id(base_embedding) == id(wrapper_embedding)}")
        print(f"Weights share memory: {base_embedding.weight.data_ptr() == wrapper_embedding.weight.data_ptr()}")
        print(f"Weights equal: {torch.equal(base_embedding.weight, wrapper_embedding.weight)}")
        
        # Test saving - this should trigger the error
        print("\n" + "=" * 40)
        print("TESTING MODEL SAVE")
        print("=" * 40)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_model"
            save_path.mkdir(exist_ok=True)
            
            try:
                print("Attempting to save model with torch.save...")
                torch.save(wrapper.state_dict(), save_path / "wrapper_state.pt")
                print("✓ torch.save succeeded")
            except Exception as e:
                print(f"✗ torch.save failed: {e}")
            
            try:
                print("Attempting to save model with safetensors...")
                from safetensors.torch import save_file
                save_file(wrapper.state_dict(), save_path / "wrapper_state.safetensors")
                print("✓ safetensors save succeeded")
            except Exception as e:
                print(f"✗ safetensors save failed: {e}")
        
        return wrapper, model, tokenizer
        
    except Exception as e:
        print(f"✗ Error in test: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_fix_shared_memory():
    """Test the fix for shared memory issue"""
    print("\n" + "=" * 80)
    print("TESTING SHARED MEMORY FIX")
    print("=" * 80)
    
    # Load model and tokenizer
    model_name = "OpenGVLab/InternVL3-1B-Pretrained"
    print(f"Loading model: {model_name}")
    
    try:
        model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        
        # Create LatentWrapper
        wrapper = LatentWrapper(model, tokenizer)
        
        print("\n" + "=" * 40)
        print("APPLYING FIX")
        print("=" * 40)
        
        # Fix 1: Create independent embedding copy
        print("Creating independent embedding layer...")
        original_embedding = wrapper.embedding
        
        # Create a new embedding layer with copied weights
        new_embedding = torch.nn.Embedding(
            num_embeddings=original_embedding.num_embeddings,
            embedding_dim=original_embedding.embedding_dim,
            padding_idx=original_embedding.padding_idx,
            max_norm=original_embedding.max_norm,
            norm_type=original_embedding.norm_type,
            scale_grad_by_freq=original_embedding.scale_grad_by_freq,
            sparse=original_embedding.sparse,
            dtype=original_embedding.weight.dtype,
            device=original_embedding.weight.device
        )
        
        # Copy weights but make them independent
        with torch.no_grad():
            new_embedding.weight.copy_(original_embedding.weight)
        
        # Replace the embedding in wrapper
        wrapper.embedding = new_embedding
        
        # Verify the fix
        print("\n" + "=" * 40)
        print("VERIFYING FIX")
        print("=" * 40)
        
        base_embedding = model.model.language_model.model.embed_tokens
        wrapper_embedding = wrapper.embedding
        
        print(f"Same object (id): {id(base_embedding) == id(wrapper_embedding)}")
        print(f"Weights share memory: {base_embedding.weight.data_ptr() == wrapper_embedding.weight.data_ptr()}")
        print(f"Weights equal: {torch.equal(base_embedding.weight, wrapper_embedding.weight)}")
        
        # Test saving after fix
        print("\n" + "=" * 40)
        print("TESTING SAVE AFTER FIX")
        print("=" * 40)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "fixed_model"
            save_path.mkdir(exist_ok=True)
            
            try:
                print("Attempting to save fixed model with safetensors...")
                from safetensors.torch import save_file
                save_file(wrapper.state_dict(), save_path / "wrapper_state_fixed.safetensors")
                print("✓ safetensors save succeeded!")
                
                # Test loading
                print("Testing model reload...")
                from safetensors.torch import load_file
                loaded_state = load_file(save_path / "wrapper_state_fixed.safetensors")
                print(f"✓ Model reloaded successfully! Keys: {len(loaded_state)}")
                
                return True
                
            except Exception as e:
                print(f"✗ Save after fix failed: {e}")
                return False
        
    except Exception as e:
        print(f"✗ Error in fix test: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_fixed_latent_wrapper():
    """Create a version of LatentWrapper that avoids shared memory"""
    print("\n" + "=" * 80)
    print("CREATING FIXED LATENT WRAPPER")
    print("=" * 80)
    
    # This will be the content for the fixed version
    fixed_code = '''
def _get_embedding_layer(self, model):
    """Get the correct embedding layer from potentially nested model structure with independent copy"""
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        # InternVL3 structure: model.language_model.model.embed_tokens
        original_embedding = model.language_model.model.embed_tokens
    elif hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        # InternVL structure: model.model.language_model.model.embed_tokens
        original_embedding = model.model.language_model.model.embed_tokens
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        # Direct access: model.model.embed_tokens  
        original_embedding = model.model.embed_tokens
    elif hasattr(model, 'get_input_embeddings'):
        # Fallback: use get_input_embeddings method
        original_embedding = model.get_input_embeddings()
    else:
        # Last resort: try to find embed_tokens attribute
        for attr_name in ['embed_tokens', 'embeddings', 'word_embeddings']:
            if hasattr(model, attr_name):
                original_embedding = getattr(model, attr_name)
                break
        else:
            raise AttributeError(f"Could not find embedding layer in model: {type(model)}")
    
    # CRITICAL FIX: Create independent embedding copy to avoid shared memory issues
    # This prevents the "Some tensors share memory" error when saving
    new_embedding = torch.nn.Embedding(
        num_embeddings=original_embedding.num_embeddings,
        embedding_dim=original_embedding.embedding_dim,
        padding_idx=original_embedding.padding_idx,
        max_norm=original_embedding.max_norm,
        norm_type=original_embedding.norm_type,
        scale_grad_by_freq=original_embedding.scale_grad_by_freq,
        sparse=original_embedding.sparse,
        dtype=original_embedding.weight.dtype,
        device=original_embedding.weight.device
    )
    
    # Copy weights but make them independent
    with torch.no_grad():
        new_embedding.weight.copy_(original_embedding.weight)
    
    return new_embedding
'''
    
    print("Fixed embedding layer method:")
    print(fixed_code)
    return fixed_code

if __name__ == "__main__":
    print("Testing shared memory issue in LatentWrapper...")
    
    # Test 1: Reproduce the issue
    wrapper, model, tokenizer = test_shared_memory_issue()
    
    if wrapper is not None:
        # Test 2: Apply and test the fix
        success = test_fix_shared_memory()
        
        if success:
            print("\n" + "=" * 80)
            print("✓ SHARED MEMORY ISSUE SUCCESSFULLY RESOLVED!")
            print("=" * 80)
            
            # Show the fixed code
            create_fixed_latent_wrapper()
            
        else:
            print("\n" + "=" * 80)
            print("✗ Fix unsuccessful - needs further investigation")
            print("=" * 80)
    
    print("\nTest complete.")
