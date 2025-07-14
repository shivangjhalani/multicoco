#!/usr/bin/env python3
"""
Test script to debug multimodal sequence length mismatch in latent injection.
"""

import torch
import logging
import sys
import os

# Add project paths
sys.path.append('/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco')
sys.path.append('/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco/InternVL/internvl_chat')

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_sequence_length_mismatch():
    """Test that shows the sequence length issue in multimodal processing"""
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        
        # Load a small InternVL model for testing
        model_path = "OpenGVLab/InternVL2_5-1B"
        
        print("Loading model...")
        wrapper = LatentWrapper(model_path)
        print(f"Model loaded successfully!")
        print(f"Model type: {type(wrapper.base_model)}")
        
        # Create test inputs with latent tokens
        # Format: text + latent span + more text + image tokens
        test_input = "<s>Question: What do you see? <start_latent>analyze<end_latent> <IMG_CONTEXT></s>"
        
        # Simulate tokenization (we'll create fake token ids for testing)
        # This simulates what happens in real usage
        fake_input_ids = torch.tensor([[1, 100, 200, 50001, 300, 50002, 400, 64003, 2]], dtype=torch.long)  # 9 tokens
        print(f"Original input_ids shape: {fake_input_ids.shape}")
        
        # Create fake image embeddings (simulating what extract_feature would return)
        # InternVL typically expands 1 image token to many embedding vectors
        image_embed_length = 256  # Typical for InternVL
        fake_image_embeds = torch.randn(1, image_embed_length, 4096)  # [1, 256, 4096]
        print(f"Image embeds shape: {fake_image_embeds.shape}")
        
        # Test the multimodal input preparation
        print("\nTesting multimodal input preparation...")
        
        # Get text embeddings first
        text_embeds = wrapper.embedding(fake_input_ids)
        print(f"Text embeddings shape: {text_embeds.shape}")
        
        # Prepare multimodal inputs
        multimodal_embeds = wrapper._prepare_inputs_for_multimodal_internvl(
            fake_input_ids, fake_image_embeds, text_embeds
        )
        print(f"Multimodal embeddings shape: {multimodal_embeds.shape}")
        
        # This is the problem: multimodal_embeds.shape[1] != fake_input_ids.shape[1]
        if multimodal_embeds.shape[1] != fake_input_ids.shape[1]:
            print(f"\n❌ SEQUENCE LENGTH MISMATCH DETECTED!")
            print(f"   input_ids length: {fake_input_ids.shape[1]}")
            print(f"   multimodal_embeds length: {multimodal_embeds.shape[1]}")
            print(f"   Difference: {multimodal_embeds.shape[1] - fake_input_ids.shape[1]}")
            print("\nThis mismatch causes the 'Invalid source position' warnings!")
            
            # Show what happens in the compute range logic
            print(f"\nCompute range logic analysis:")
            print(f"  Original algorithm assumes sequence length = {fake_input_ids.shape[1]}")
            print(f"  But actual hidden states will have length = {multimodal_embeds.shape[1]}")
            print(f"  When injecting latent tokens, source_pos calculations will be wrong!")
            
        else:
            print(f"✅ No sequence length mismatch - shapes match")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing multimodal sequence length mismatch...")
    success = test_sequence_length_mismatch()
    if success:
        print("\n✅ Test completed successfully")
    else:
        print("\n❌ Test failed")
