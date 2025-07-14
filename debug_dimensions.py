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
        try:
            # Skip the potentially hanging extract_feature for now
            print("Skipping model.extract_feature() - testing vision_model directly...")
            
            # Test vision model directly first
            print("Calling vision_model directly...")
            vision_outputs = model.vision_model(dummy_image, output_hidden_states=True)
            raw_vision = vision_outputs.last_hidden_state
            print(f"Raw vision output shape: {raw_vision.shape}")
            
            # Test mlp1 projector manually
            print("\nTesting mlp1 projector manually...")
            if hasattr(model, 'mlp1'):
                print(f"MLP1 input expected: {model.mlp1[0].normalized_shape}")
                # Flatten vision features for MLP1
                B, L, C = raw_vision.shape
                # Vision model output needs to be flattened and projected
                flattened_vision = raw_vision.view(B, -1)  # Flatten spatial dimensions
                print(f"Flattened vision shape: {flattened_vision.shape}")
                
                # The mlp1 expects 4096 input, but we have different dimensions
                # This explains the dimension mismatch!
                print(f"MLP1 expected input: 4096, got: {flattened_vision.shape[-1]}")
            else:
                print("No mlp1 found in model")
                
        except Exception as e:
            print(f"Error in vision processing: {e}")
            import traceback
            traceback.print_exc()
        
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
            # Test the exact scenario from CoCoNut with special tokens
            # First add the special tokens to the tokenizer
            special_tokens = ['<|start-latent|>', '<|latent|>', '<|end-latent|>']
            tokenizer.add_tokens(special_tokens)
            
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
                
            # CRITICAL: Test with the actual MultiCoCo wrapper to see where 2048 comes from
            print("\n=== Testing with MultiCoCo wrapper ===")
            try:
                # Import MultiCoCo to see what happens
                import sys
                sys.path.append('/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco')
                from multicoco.model import MultiCoCo
                from multicoco.constants import COCONUT_SPECIAL_TOKENS
                
                print("Creating MultiCoCo model...")
                multicoco_model = MultiCoCo(
                    model_id='OpenGVLab/InternVL3-1B-Pretrained',
                    special_tokens=COCONUT_SPECIAL_TOKENS,
                    torch_dtype='bfloat16'
                )
                
                # Test the same scenario
                test_text = "Test <|start-latent|> <|latent|> <|latent|> <|end-latent|> done"
                test_ids = multicoco_model.tokenizer.encode(test_text, return_tensors="pt")
                print(f"MultiCoCo test input shape: {test_ids.shape}")
                
                # Get embeddings from MultiCoCo
                multicoco_embeds = multicoco_model.language_model.model.embed_tokens(test_ids)
                print(f"MultiCoCo embeddings shape: {multicoco_embeds.shape}")
                
                # Get hidden states from MultiCoCo
                multicoco_hidden = multicoco_model.language_model.model(input_ids=test_ids, output_hidden_states=True)
                multicoco_hidden_states = multicoco_hidden.hidden_states[-1]
                print(f"MultiCoCo hidden states shape: {multicoco_hidden_states.shape}")
                
                # Check for the 2048 dimension
                if multicoco_hidden_states.shape[-1] == 2048:
                    print("🔍 FOUND IT! Hidden states are 2048 dimensions in MultiCoCo")
                    print("This means MultiCoCo is using a different model configuration!")
                    
                    # Check if there's a different language model path being used
                    print(f"MultiCoCo language model type: {type(multicoco_model.language_model)}")
                    print(f"Raw model language model type: {type(model.language_model)}")
                    
                    # Check configs
                    print(f"MultiCoCo LM hidden size: {multicoco_model.language_model.config.hidden_size}")
                    print(f"Raw model LM hidden size: {model.language_model.config.hidden_size}")
                    
                elif multicoco_hidden_states.shape[-1] == 896:
                    print("Hidden states are 896 dimensions in MultiCoCo - same as raw model")
                    print("The 2048 dimension must come from somewhere else...")
                else:
                    print(f"MultiCoCo hidden states have unexpected dimension: {multicoco_hidden_states.shape[-1]}")
                
            except Exception as e:
                print(f"Error testing MultiCoCo: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"Error in CoCoNut analysis: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_internvl_dimensions()
