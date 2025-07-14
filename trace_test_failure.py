#!/usr/bin/env python3
"""
Trace the exact test failure scenario to find where 2048 dimension comes from.
"""
import torch
import logging
from multicoco.model import MultiCoCo
from multicoco.latent_wrapper import LatentWrapper
from multicoco.constants import COCONUT_SPECIAL_TOKENS

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def trace_test_failure():
    """Replicate the exact failing test scenario"""
    print("=== Tracing Test Failure ===")
    
    # Initialize exactly like the test does
    print("Initializing MultiCoCo model...")
    model = MultiCoCo(
        model_id='OpenGVLab/InternVL3-1B-Pretrained',
        special_tokens=COCONUT_SPECIAL_TOKENS,
        torch_dtype='bfloat16'
    )
    
    print("Initializing LatentWrapper...")
    latent_model = LatentWrapper(model, model.tokenizer, enable_norm_logging=False)
    
    # Create the exact test input that fails
    test_text = "Question: What is 2+2? <|start-latent|> <|latent|> <|latent|> <|end-latent|> Answer: 4"
    input_ids = model.tokenizer.encode(test_text, return_tensors="pt")
    print(f"Test input_ids shape: {input_ids.shape}")
    print(f"Test tokens: {model.tokenizer.convert_ids_to_tokens(input_ids[0])}")
    
    # Check token dimensions
    embeddings = latent_model.embedding(input_ids)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings last dim: {embeddings.shape[-1]}")
    
    # Test the first pass (where the error occurs)
    print("\n=== Testing First Pass (where error likely occurs) ===")
    try:
        with torch.no_grad():
            # Extract latent spans
            spans = latent_model._extract_latent_spans(input_ids)
            print(f"Latent spans: {spans}")
            
            # Get hidden states (this is where the error likely happens)
            print("Getting first pass hidden states...")
            hidden_states = latent_model._first_pass_hidden_states(input_ids, None, None)
            print(f"Hidden states shape: {hidden_states.shape}")
            print(f"Hidden states last dim: {hidden_states.shape[-1]}")
            
            # Test the exact assignment that fails
            print("\n=== Testing Direct Assignment ===")
            inputs_embeds = latent_model.embedding(input_ids).clone()
            print(f"inputs_embeds shape: {inputs_embeds.shape}")
            
            # Try the direct assignment that should fail
            batch_idx = 0
            pos = 4  # Some position within latent span
            source_pos = 3  # Previous position
            
            print(f"Trying to assign hidden_states[{batch_idx}, {source_pos}] to inputs_embeds[{batch_idx}, {pos}]")
            print(f"Source shape: {hidden_states[batch_idx, source_pos].shape}")
            print(f"Target shape: {inputs_embeds[batch_idx, pos].shape}")
            
            if hidden_states[batch_idx, source_pos].shape != inputs_embeds[batch_idx, pos].shape:
                print("❌ SHAPE MISMATCH FOUND!")
                print(f"Cannot assign {hidden_states[batch_idx, source_pos].shape} to {inputs_embeds[batch_idx, pos].shape}")
            else:
                print("✅ Shapes match, assignment should work")
                inputs_embeds[batch_idx, pos] = hidden_states[batch_idx, source_pos]
                print("Assignment successful!")
                
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        
        # If error contains dimension info, extract it
        error_str = str(e)
        if "2048" in error_str and "896" in error_str:
            print("\n🔍 Found the 2048 vs 896 mismatch!")
            print("The error is likely in the language model structure.")
            
            # Investigate the language model more deeply
            print("\n=== Deep Language Model Investigation ===")
            print(f"Model type: {type(model.model)}")
            print(f"Language model type: {type(model.model.language_model)}")
            
            # Check if there are multiple language model layers
            if hasattr(model.model.language_model, 'model'):
                inner_model = model.model.language_model.model
                print(f"Inner language model type: {type(inner_model)}")
                
                # Check hidden size at different levels
                if hasattr(inner_model.config, 'hidden_size'):
                    print(f"Inner model hidden_size: {inner_model.config.hidden_size}")
                    
            # Check the actual output dimensions from language model
            with torch.no_grad():
                print("Testing language model directly...")
                lang_out = model.model.language_model.model(input_ids=input_ids, output_hidden_states=True)
                print(f"Language model output type: {type(lang_out)}")
                print(f"Hidden states available: {len(lang_out.hidden_states)}")
                for i, hs in enumerate(lang_out.hidden_states):
                    print(f"  Layer {i}: {hs.shape}")

if __name__ == "__main__":
    trace_test_failure()
