#!/usr/bin/env python3
"""
Debug script to trace the exact dimension flow in InternVL3-1B and find the root cause
of the dimension mismatch in CoCoNut latent injection.
"""
import torch
import logging
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_internvl_dimensions():
    """Debug the dimension flow in InternVL3-1B"""
    print("=== InternVL3-1B Dimension Debugging ===")
    
    # Load model
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
    
    print("\n=== Key Dimensions ===")
    print(f"Vision hidden size: {model.vision_model.config.hidden_size}")
    print(f"Language hidden size: {model.language_model.config.hidden_size}")
    print(f"Language embedding size: {model.language_model.model.embed_tokens.embedding_dim}")
    
    # Create test inputs
    test_text = "What is this image?"
    input_ids = tokenizer.encode(test_text, return_tensors="pt")
    dummy_image = torch.randn(1, 3, 448, 448, dtype=torch.bfloat16)
    
    print(f"\nTest input_ids shape: {input_ids.shape}")
    print(f"Test image shape: {dummy_image.shape}")
    
    # Test vision processing
    print("\n=== Vision Processing Debug ===")
    with torch.no_grad():
        # Extract vision features using model's method
        print("Calling model.extract_feature(pixel_values)...")
        vision_features = model.extract_feature(dummy_image)
        print(f"Vision features shape: {vision_features.shape}")
        print(f"Vision features dtype: {vision_features.dtype}")
        
        # Test vision model directly
        print("\nCalling vision_model directly...")
        vision_outputs = model.vision_model(dummy_image, output_hidden_states=True)
        raw_vision = vision_outputs.last_hidden_state
        print(f"Raw vision output shape: {raw_vision.shape}")
        
        # Test mlp1 projector
        print("\nTesting mlp1 projector...")
        # The extract_feature method should handle the projection
        
    # Test language model embeddings
    print("\n=== Language Model Debug ===")
    with torch.no_grad():
        # Get embeddings
        embeddings = model.language_model.model.embed_tokens(input_ids)
        print(f"Language embeddings shape: {embeddings.shape}")
        print(f"Language embeddings dtype: {embeddings.dtype}")
        
        # Get hidden states
        lang_outputs = model.language_model.model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = lang_outputs.hidden_states[-1]
        print(f"Language hidden states shape: {hidden_states.shape}")
        print(f"Language hidden states dtype: {hidden_states.dtype}")
    
    # Test multimodal processing
    print("\n=== Multimodal Processing Debug ===")
    with torch.no_grad():
        try:
            # Add image tokens to text
            img_context_token = "<IMG_CONTEXT>"
            num_img_tokens = 256
            multimodal_text = f"<img>{'<IMG_CONTEXT>' * num_img_tokens}</img>{test_text}"
            multimodal_ids = tokenizer.encode(multimodal_text, return_tensors="pt")
            print(f"Multimodal input_ids shape: {multimodal_ids.shape}")
            
            # Set img_context_token_id
            img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
            model.img_context_token_id = img_context_token_id
            print(f"IMG_CONTEXT token ID: {img_context_token_id}")
            
            # Get text embeddings
            text_embeds = model.language_model.model.embed_tokens(multimodal_ids)
            print(f"Text embeddings shape: {text_embeds.shape}")
            
            # Simulate the image token replacement logic
            vision_features = model.extract_feature(dummy_image)
            print(f"Vision features to inject shape: {vision_features.shape}")
            
            # Find image token positions
            img_positions = (multimodal_ids == img_context_token_id)
            num_img_positions = img_positions.sum().item()
            print(f"Number of IMG_CONTEXT positions: {num_img_positions}")
            
            # Check dimension compatibility
            if vision_features.shape[-1] == text_embeds.shape[-1]:
                print("✓ Vision features and text embeddings have compatible dimensions")
            else:
                print(f"✗ Dimension mismatch: vision={vision_features.shape[-1]}, text={text_embeds.shape[-1]}")
                
        except Exception as e:
            print(f"Error in multimodal processing: {e}")
    
    print("\n=== CoCoNut Dimension Analysis ===")
    with torch.no_grad():
        try:
            # Test the exact scenario from CoCoNut
            simple_text = "Test <|start-latent|> <|latent|> <|latent|> <|end-latent|> done"
            simple_ids = tokenizer.encode(simple_text, return_tensors="pt")
            print(f"CoCoNut test input shape: {simple_ids.shape}")
            
            # Get embeddings (what CoCoNut uses)
            coconut_embeds = model.language_model.model.embed_tokens(simple_ids)
            print(f"CoCoNut embeddings shape: {coconut_embeds.shape}")
            
            # Get hidden states (what CoCoNut wants to inject)
            coconut_hidden = model.language_model.model(input_ids=simple_ids, output_hidden_states=True)
            coconut_hidden_states = coconut_hidden.hidden_states[-1]
            print(f"CoCoNut hidden states shape: {coconut_hidden_states.shape}")
            
            # Check if they match
            if coconut_embeds.shape[-1] == coconut_hidden_states.shape[-1]:
                print("✓ Embeddings and hidden states have same dimension - CoCoNut should work!")
            else:
                print(f"✗ CoCoNut dimension mismatch: embeds={coconut_embeds.shape[-1]}, hidden={coconut_hidden_states.shape[-1]}")
                print("This explains the test failure!")
                
                # Suggest fix
                print("\n=== Suggested Fix ===")
                print("The issue is that InternVL3-1B uses Qwen2 language model internally.")
                print("Hidden states might be from a different layer or need projection.")
                print("Need to check the exact layer structure...")
                
                # Inspect the language model layers
                print(f"Language model type: {type(model.language_model)}")
                print(f"Language model layers: {list(model.language_model.named_children())}")
                
        except Exception as e:
            print(f"Error in CoCoNut analysis: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_internvl_dimensions()
