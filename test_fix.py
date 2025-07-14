#!/usr/bin/env python3
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
        # Find the base embedding layer using the same logic as LatentWrapper
        base_embedding = None
        if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
            base_embedding = model.language_model.model.embed_tokens
        elif hasattr(model, 'model') and hasattr(model.model, 'language_model'):
            base_embedding = model.model.language_model.model.embed_tokens
        elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            base_embedding = model.model.embed_tokens
        elif hasattr(model, 'get_input_embeddings'):
            base_embedding = model.get_input_embeddings()
        else:
            for attr_name in ['embed_tokens', 'embeddings', 'word_embeddings']:
                if hasattr(model, attr_name):
                    base_embedding = getattr(model, attr_name)
                    break
        
        if base_embedding is None:
            print("✗ Could not find base embedding layer")
            return False
            
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
