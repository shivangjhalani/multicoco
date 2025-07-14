#!/usr/bin/env python3
"""
Fix script to resolve shared memory issue in LatentWrapper.

This script applies the fix to the LatentWrapper's _get_embedding_layer method
to prevent shared memory issues when saving the model.
"""

import sys
import os
from pathlib import Path

def apply_fix_to_latent_wrapper():
    """Apply the shared memory fix to LatentWrapper"""
    print("Applying shared memory fix to LatentWrapper...")
    
    latent_wrapper_path = Path("./multicoco/latent_wrapper.py")
    
    if not latent_wrapper_path.exists():
        print(f"Error: LatentWrapper file not found at {latent_wrapper_path}")
        return False
    
    # Read the current file
    with open(latent_wrapper_path, 'r') as f:
        content = f.read()
    
    # Define the original problematic method
    original_method = '''    def _get_embedding_layer(self, model):
        """Get the correct embedding layer from potentially nested model structure"""
        if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
            # InternVL3 structure: model.language_model.model.embed_tokens
            return model.language_model.model.embed_tokens
        elif hasattr(model, 'model') and hasattr(model.model, 'language_model'):
            # InternVL structure: model.model.language_model.model.embed_tokens
            return model.model.language_model.model.embed_tokens
        elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            # Direct access: model.model.embed_tokens  
            return model.model.embed_tokens
        elif hasattr(model, 'get_input_embeddings'):
            # Fallback: use get_input_embeddings method
            return model.get_input_embeddings()
        else:
            # Last resort: try to find embed_tokens attribute
            for attr_name in ['embed_tokens', 'embeddings', 'word_embeddings']:
                if hasattr(model, attr_name):
                    return getattr(model, attr_name)
            raise AttributeError(f"Could not find embedding layer in model: {type(model)}")'''
    
    # Define the fixed method
    fixed_method = '''    def _get_embedding_layer(self, model):
        """Get the correct embedding layer from potentially nested model structure with independent copy"""
        # First, find the original embedding layer
        original_embedding = None
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
        import torch.nn as nn
        import torch
        
        new_embedding = nn.Embedding(
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
        
        return new_embedding'''
    
    # Check if the original method exists
    if original_method not in content:
        print("Warning: Original method not found exactly. Trying to locate and replace...")
        # Try to find the method signature and replace more flexibly
        import re
        
        # Find the _get_embedding_layer method
        pattern = r'    def _get_embedding_layer\(self, model\):.*?(?=    def |\Z)'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            print("Found _get_embedding_layer method, replacing...")
            content = content.replace(match.group(0), fixed_method)
        else:
            print("Error: Could not find _get_embedding_layer method to replace")
            return False
    else:
        # Direct replacement
        print("Found exact method match, replacing...")
        content = content.replace(original_method, fixed_method)
    
    # Write the fixed content back
    with open(latent_wrapper_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Successfully applied shared memory fix to {latent_wrapper_path}")
    return True

def verify_fix():
    """Verify that the fix has been applied correctly"""
    print("Verifying the fix...")
    
    latent_wrapper_path = Path("/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco/multicoco/latent_wrapper.py")
    
    with open(latent_wrapper_path, 'r') as f:
        content = f.read()
    
    # Check for the fix indicators
    fix_indicators = [
        "# CRITICAL FIX: Create independent embedding copy",
        "new_embedding = nn.Embedding(",
        "new_embedding.weight.copy_(original_embedding.weight)"
    ]
    
    all_found = True
    for indicator in fix_indicators:
        if indicator in content:
            print(f"✓ Found: {indicator}")
        else:
            print(f"✗ Missing: {indicator}")
            all_found = False
    
    if all_found:
        print("✓ All fix indicators found - fix appears to be applied correctly")
    else:
        print("✗ Some fix indicators missing - fix may not be complete")
    
    return all_found

def create_test_script():
    """Create a simple test script to verify the fix works"""
    test_script_content = '''#!/usr/bin/env python3
"""
Simple test to verify the shared memory fix works.
"""

import sys
sys.path.append('.')

def test_no_shared_memory():
    """Test that embeddings no longer share memory"""
    try:
        from transformers import AutoTokenizer, AutoModel
        from multicoco.latent_wrapper import LatentWrapper
        import torch
        
        print("Loading model...")
        model_name = "OpenGVLab/InternVL3-1B-Pretrained"
        model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        
        print("Creating LatentWrapper...")
        wrapper = LatentWrapper(model, tokenizer)
        
        print("Checking memory sharing...")
        base_embedding = model.model.language_model.model.embed_tokens
        wrapper_embedding = wrapper.embedding
        
        shared_memory = base_embedding.weight.data_ptr() == wrapper_embedding.weight.data_ptr()
        same_values = torch.equal(base_embedding.weight, wrapper_embedding.weight)
        
        print(f"Embeddings share memory: {shared_memory}")
        print(f"Embedding values are equal: {same_values}")
        
        if not shared_memory and same_values:
            print("✓ SUCCESS: Embeddings have same values but don't share memory!")
            return True
        else:
            print("✗ FAILURE: Fix not working correctly")
            return False
            
    except Exception as e:
        print(f"✗ Error in test: {e}")
        return False

if __name__ == "__main__":
    test_no_shared_memory()
'''
    
    test_script_path = Path("./test_fix.py")
    with open(test_script_path, 'w') as f:
        f.write(test_script_content)
    
    print(f"✓ Created test script at {test_script_path}")
    return test_script_path

if __name__ == "__main__":
    print("=" * 60)
    print("FIXING SHARED MEMORY ISSUE IN LATENTWRAPPER")
    print("=" * 60)
    
    # Apply the fix
    if apply_fix_to_latent_wrapper():
        # Verify the fix
        if verify_fix():
            # Create test script
            test_script_path = create_test_script()
            
            print("\n" + "=" * 60)
            print("FIX APPLIED SUCCESSFULLY!")
            print("=" * 60)
            print(f"You can test the fix by running: python {test_script_path}")
            print("\nThe fix creates independent embedding layers to prevent")
            print("shared memory issues when saving the model.")
        else:
            print("\n✗ Fix verification failed")
    else:
        print("\n✗ Failed to apply fix")
