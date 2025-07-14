#!/usr/bin/env python3

import logging
import torch
from multicoco.model import MultiCoCo
from multicoco.latent_wrapper import LatentWrapper
from multicoco.constants import COCONUT_SPECIAL_TOKENS

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_forward_issue():
    """Debug the forward pass issue"""
    try:
        logger.info("Creating MultiCoCo model...")
        model = MultiCoCo(
            model_id="microsoft/DialoGPT-small",
            special_tokens=COCONUT_SPECIAL_TOKENS.copy(),
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        logger.info("Wrapping with LatentWrapper...")
        wrapper = LatentWrapper(model, model.tokenizer)
        
        tokenizer = model.tokenizer
        test_text = "Question: What is this? <|start_latent|><|latent|><|latent|><|end_latent|> Answer: A test."
        encoded = tokenizer(test_text, return_tensors='pt', padding=True, truncation=True)
        
        logger.info(f"Input IDs shape: {encoded['input_ids'].shape}")
        logger.info(f"Input IDs: {encoded['input_ids']}")
        logger.info(f"Attention mask shape: {encoded['attention_mask'].shape}")
        
        # Check wrapper forward arguments
        logger.info("Testing wrapper forward with explicit arguments...")
        with torch.no_grad():
            outputs = wrapper(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask']
            )
        
        logger.info("✓ Forward pass successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_forward_issue()
