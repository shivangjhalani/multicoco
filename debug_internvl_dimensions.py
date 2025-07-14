#!/usr/bin/env python3
"""
Debug script to investigate InternVL3-1B dimension mismatch and find the correct projection layer.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import sys
import os

# Add multicoco to path
sys.path.insert(0, '/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco')

def main():
    print("=== InternVL3-1B Dimension Analysis ===")
    
    # Load the model
    print("Loading InternVL3-1B model...")
    model = AutoModel.from_pretrained(
        "OpenGVLab/InternVL3-1B-Pretrained", 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "OpenGVLab/InternVL3-1B-Pretrained", 
        trust_remote_code=True, 
        use_fast=False
    )
    
    print("\n=== Model Architecture Analysis ===")
    print(f"Model type: {type(model)}")
    print(f"Model config keys: {list(model.config.__dict__.keys())}")
    
    # Analyze vision and language components
    print(f"\n=== Vision Model Analysis ===")
    vision_model = model.vision_model
    print(f"Vision model type: {type(vision_model)}")
    print(f"Vision hidden size: {model.config.vision_config.hidden_size}")
    
    print(f"\n=== Language Model Analysis ===")
    language_model = model.language_model
    print(f"Language model type: {type(language_model)}")
    print(f"Language hidden size: {model.config.llm_config.hidden_size}")
    print(f"Language embedding size: {language_model.model.embed_tokens.embedding_dim}")
    
    # Analyze the mlp1 projector (this is key!)
    print(f"\n=== MLP1 Projector Analysis ===")
    mlp1 = model.mlp1
    print(f"MLP1 type: {type(mlp1)}")
    print(f"MLP1 structure:")
    for i, layer in enumerate(mlp1):
        print(f"  [{i}] {layer}")
        if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
            print(f"      Input: {layer.in_features}, Output: {layer.out_features}")
    
    # Test with dummy inputs to trace dimensions
    print(f"\n=== Dimension Tracing ===")
    
    # Create dummy inputs
    dummy_text_ids = torch.randint(0, 1000, (1, 10))  # Small vocab range to avoid issues
    dummy_image = torch.randn(1, 3, 448, 448).to(torch.bfloat16)
    
    print(f"Dummy text input shape: {dummy_text_ids.shape}")
    print(f"Dummy image input shape: {dummy_image.shape}")
    
    with torch.no_grad():
        # Test vision model
        print(f"\n--- Vision Processing ---")
        vision_output = vision_model(dummy_image)
        vision_features = vision_output.last_hidden_state
        print(f"Vision features shape: {vision_features.shape}")
        
        # Test vision features through mlp1
        print(f"\n--- Vision Feature Projection ---")
        # The vision features need to be processed through mlp1
        # First, let's see what mlp1 expects
        vision_features_flat = vision_features.view(-1, vision_features.shape[-1])  # Flatten spatial dims
        print(f"Vision features flattened shape: {vision_features_flat.shape}")
        
        try:
            projected_vision = mlp1(vision_features_flat)
            print(f"Projected vision features shape: {projected_vision.shape}")
        except Exception as e:
            print(f"Error projecting vision features: {e}")
        
        # Test language model embeddings
        print(f"\n--- Language Model Processing ---")
        text_embeddings = language_model.model.embed_tokens(dummy_text_ids)
        print(f"Text embeddings shape: {text_embeddings.shape}")
        
        # Test language model forward to get hidden states
        print(f"\n--- Language Model Hidden States ---")
        lm_output = language_model.model(input_ids=dummy_text_ids, output_hidden_states=True)
        hidden_states = lm_output.hidden_states[-1]  # Last layer
        print(f"Language model hidden states shape: {hidden_states.shape}")
        
        # Test all hidden layers to see if there's a dimension change
        print(f"\n--- All Hidden Layer Shapes ---")
        for i, hs in enumerate(lm_output.hidden_states):
            print(f"Layer {i}: {hs.shape}")
        
        # Test full model forward (this is what should work)
        print(f"\n--- Full Model Forward ---")
        try:
            full_output = model(pixel_values=dummy_image, input_ids=dummy_text_ids)
            print(f"Full model output logits shape: {full_output.logits.shape}")
        except Exception as e:
            print(f"Error in full model forward: {e}")
    
    # Analyze extract_feature method
    print(f"\n=== Extract Feature Method Analysis ===")
    if hasattr(model, 'extract_feature'):
        try:
            with torch.no_grad():
                extracted_features = model.extract_feature(dummy_image)
                print(f"Extracted features shape: {extracted_features.shape}")
                print(f"Extracted features dtype: {extracted_features.dtype}")
        except Exception as e:
            print(f"Error in extract_feature: {e}")
    else:
        print("extract_feature method not found")
    
    # Look for other projection layers
    print(f"\n=== All Model Components ===")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"Linear layer '{name}': {module.in_features} -> {module.out_features}")
    
    print(f"\n=== Key Findings ===")
    print(f"Vision hidden size: {model.config.vision_config.hidden_size}")
    print(f"Language hidden size: {model.config.llm_config.hidden_size}")
    print(f"Embedding size: {language_model.model.embed_tokens.embedding_dim}")
    
    # Identify the correct projection path
    print(f"\n=== Projection Path Recommendation ===")
    vision_dim = model.config.vision_config.hidden_size
    lang_dim = model.config.llm_config.hidden_size
    
    if vision_dim != lang_dim:
        print(f"✓ Vision-language dimension mismatch detected: {vision_dim} != {lang_dim}")
        print(f"✓ mlp1 projector should handle: {vision_dim} -> {lang_dim}")
        print(f"✓ Use model.mlp1 for projection in LatentWrapper")
    else:
        print(f"✓ Vision and language dimensions match: {vision_dim}")
    
    return model, tokenizer

if __name__ == "__main__":
    main()
