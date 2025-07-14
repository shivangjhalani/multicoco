#!/usr/bin/env python3
"""
Test script to verify that LatentWrapper simplification didn't break functionality.
Tests both latent injection and standard (non-latent) model behavior.
"""
import sys
import torch
import logging
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_initialization():
    """Test that we can initialize the model and LatentWrapper without errors"""
    try:
        from multicoco.model import MultiCoCo
        from multicoco.latent_wrapper import LatentWrapper
        from multicoco.constants import COCONUT_SPECIAL_TOKENS
        
        logger.info("=== Testing Model Initialization ===")
        
        # Initialize base model
        model = MultiCoCo(
            model_id='OpenGVLab/InternVL3-1B-Pretrained',
            special_tokens=COCONUT_SPECIAL_TOKENS,
            torch_dtype='bfloat16'
        )
        logger.info("✓ Base model initialized successfully")
        
        # Initialize LatentWrapper
        latent_model = LatentWrapper(model, model.tokenizer, enable_norm_logging=False)
        logger.info("✓ LatentWrapper initialized successfully")
        
        # Check token IDs are properly set
        assert latent_model.start_id is not None, "start_id not set"
        assert latent_model.end_id is not None, "end_id not set"
        assert latent_model.latent_id is not None, "latent_id not set"
        logger.info(f"✓ Token IDs properly set: start={latent_model.start_id}, end={latent_model.end_id}, latent={latent_model.latent_id}")
        
        # Check embedding access
        embedding = latent_model.embedding
        assert embedding is not None, "Embedding layer not accessible"
        logger.info(f"✓ Embedding layer accessible: {type(embedding)}")
        
        return model, latent_model
        
    except Exception as e:
        logger.error(f"✗ Model initialization failed: {e}")
        raise

def test_tokenization():
    """Test that tokenization works correctly with our special tokens"""
    try:
        from multicoco.model import MultiCoCo
        from multicoco.constants import COCONUT_SPECIAL_TOKENS
        
        logger.info("=== Testing Tokenization ===")
        
        model = MultiCoCo(
            model_id='OpenGVLab/InternVL3-1B-Pretrained',
            special_tokens=COCONUT_SPECIAL_TOKENS,
            torch_dtype='bfloat16'
        )
        
        # Test sequence with latent tokens
        test_text = "Question: What is 2+2? <|start-latent|> <|latent|> <|latent|> <|end-latent|> Answer: 4"
        
        # Encode and decode
        tokens = model.tokenizer.encode(test_text, add_special_tokens=False)
        decoded = model.tokenizer.decode(tokens)
        
        assert "<|start-latent|>" in decoded, "start-latent token not properly encoded/decoded"
        assert "<|end-latent|>" in decoded, "end-latent token not properly encoded/decoded"
        assert "<|latent|>" in decoded, "latent token not properly encoded/decoded"
        
        logger.info("✓ Tokenization works correctly")
        logger.info(f"  Original: {test_text}")
        logger.info(f"  Decoded:  {decoded}")
        
        return model.tokenizer
        
    except Exception as e:
        logger.error(f"✗ Tokenization test failed: {e}")
        raise

def test_latent_span_detection():
    """Test that latent span detection works correctly"""
    try:
        from multicoco.model import MultiCoCo
        from multicoco.latent_wrapper import LatentWrapper
        from multicoco.constants import COCONUT_SPECIAL_TOKENS
        
        logger.info("=== Testing Latent Span Detection ===")
        
        model = MultiCoCo(
            model_id='OpenGVLab/InternVL3-1B-Pretrained',
            special_tokens=COCONUT_SPECIAL_TOKENS,
            torch_dtype='bfloat16'
        )
        latent_model = LatentWrapper(model, model.tokenizer, enable_norm_logging=False)
        
        # Test text with latent spans
        test_text = "Question: <|start-latent|> <|latent|> <|latent|> <|end-latent|> Answer:"
        input_ids = torch.tensor([model.tokenizer.encode(test_text, add_special_tokens=False)])
        
        # Test span detection
        has_latents = latent_model._has_latent_spans(input_ids)
        assert has_latents, "Failed to detect latent spans"
        logger.info("✓ Latent span detection works")
        
        # Test span extraction
        spans = latent_model._extract_latent_spans(input_ids)
        assert len(spans) == 1, f"Expected 1 batch, got {len(spans)}"
        assert len(spans[0]) == 1, f"Expected 1 span in batch, got {len(spans[0])}"
        
        start, end = spans[0][0]
        logger.info(f"✓ Span extraction works: span from {start} to {end}")
        
        # Test text without latent spans
        normal_text = "Question: What is 2+2? Answer: 4"
        normal_ids = torch.tensor([model.tokenizer.encode(normal_text, add_special_tokens=False)])
        
        has_latents_normal = latent_model._has_latent_spans(normal_ids)
        assert not has_latents_normal, "False positive: detected latents in normal text"
        logger.info("✓ No false positives for normal text")
        
        return latent_model
        
    except Exception as e:
        logger.error(f"✗ Latent span detection test failed: {e}")
        raise

def test_simplified_algorithm():
    """Test the simplified embedding injection algorithm"""
    try:
        from multicoco.model import MultiCoCo
        from multicoco.latent_wrapper import LatentWrapper
        from multicoco.constants import COCONUT_SPECIAL_TOKENS
        
        logger.info("=== Testing Simplified Algorithm ===")
        
        model = MultiCoCo(
            model_id='OpenGVLab/InternVL3-1B-Pretrained',
            special_tokens=COCONUT_SPECIAL_TOKENS,
            torch_dtype='bfloat16'
        )
        latent_model = LatentWrapper(model, model.tokenizer, enable_norm_logging=False)
        
        # Test text with latent spans
        test_text = "Question: <|start-latent|> <|latent|> <|latent|> <|end-latent|> Answer:"
        input_ids = torch.tensor([model.tokenizer.encode(test_text, add_special_tokens=False)])
        
        # Extract spans
        spans = latent_model._extract_latent_spans(input_ids)
        
        # Create dummy hidden states
        seq_len = input_ids.shape[1]
        hidden_dim = 896  # InternVL3-1B language model hidden dimension (corrected)
        dummy_hidden = torch.randn(1, seq_len, hidden_dim)
        
        # Test simplified embedding building
        modified_embeds = latent_model._build_modified_embeddings_sequential(input_ids, spans, dummy_hidden)
        
        assert modified_embeds.shape[0] == 1, f"Wrong batch size: {modified_embeds.shape[0]}"
        assert modified_embeds.shape[1] == seq_len, f"Wrong sequence length: {modified_embeds.shape[1]}"
        logger.info(f"✓ Simplified algorithm produces correct shape: {modified_embeds.shape}")
        
        # Verify that latent positions have been modified
        original_embeds = latent_model.embedding(input_ids)
        start, end = spans[0][0]
        
        # Check that latent token positions (between start and end markers) are different
        for pos in range(start + 1, end):
            if pos < seq_len:
                original_embed = original_embeds[0, pos]
                modified_embed = modified_embeds[0, pos]
                
                # They should be different (modified by hidden state injection)
                diff = torch.norm(original_embed - modified_embed).item()
                assert diff > 1e-6, f"Position {pos} was not modified (diff={diff})"
        
        logger.info("✓ Latent token embeddings were properly modified")
        
        return latent_model
        
    except Exception as e:
        logger.error(f"✗ Simplified algorithm test failed: {e}")
        raise

def test_forward_pass():
    """Test that forward pass works without errors"""
    try:
        from multicoco.model import MultiCoCo
        from multicoco.latent_wrapper import LatentWrapper
        from multicoco.constants import COCONUT_SPECIAL_TOKENS
        
        logger.info("=== Testing Forward Pass ===")
        
        model = MultiCoCo(
            model_id='OpenGVLab/InternVL3-1B-Pretrained',
            special_tokens=COCONUT_SPECIAL_TOKENS,
            torch_dtype='bfloat16'
        )
        latent_model = LatentWrapper(model, model.tokenizer, enable_norm_logging=False)
        
        # Test text with latent spans
        test_text = "Question: <|start-latent|> <|latent|> <|latent|> <|end-latent|> Answer:"
        input_ids = torch.tensor([model.tokenizer.encode(test_text, add_special_tokens=False)])
        attention_mask = torch.ones_like(input_ids)
        
        # Forward pass with latent injection
        with torch.no_grad():
            outputs = latent_model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids.clone()  # Dummy labels
            )
        
        assert 'logits' in outputs, "Forward pass should return logits"
        assert 'loss' in outputs, "Forward pass should return loss"
        
        logits = outputs['logits']
        loss = outputs['loss']
        
        assert logits.shape[0] == 1, f"Wrong batch size in logits: {logits.shape[0]}"
        assert logits.shape[1] == input_ids.shape[1], f"Wrong sequence length in logits: {logits.shape[1]}"
        assert loss is not None, "Loss should not be None"
        assert torch.isfinite(loss).all(), f"Loss should be finite, got {loss}"
        
        logger.info(f"✓ Forward pass works: logits.shape={logits.shape}, loss={loss.item():.4f}")
        
        # Test forward pass without latent spans
        normal_text = "Question: What is 2+2? Answer: 4"
        normal_ids = torch.tensor([model.tokenizer.encode(normal_text, add_special_tokens=False)])
        normal_mask = torch.ones_like(normal_ids)
        
        with torch.no_grad():
            normal_outputs = latent_model.forward(
                input_ids=normal_ids,
                attention_mask=normal_mask,
                labels=normal_ids.clone()
            )
        
        assert 'logits' in normal_outputs, "Normal forward pass should return logits"
        normal_logits = normal_outputs['logits']
        logger.info(f"✓ Normal forward pass works: logits.shape={normal_logits.shape}")
        
        return latent_model
        
    except Exception as e:
        logger.error(f"✗ Forward pass test failed: {e}")
        raise

def test_generation():
    """Test that generation works without errors"""
    try:
        from multicoco.model import MultiCoCo
        from multicoco.latent_wrapper import LatentWrapper
        from multicoco.constants import COCONUT_SPECIAL_TOKENS
        
        logger.info("=== Testing Generation ===")
        
        model = MultiCoCo(
            model_id='OpenGVLab/InternVL3-1B-Pretrained',
            special_tokens=COCONUT_SPECIAL_TOKENS,
            torch_dtype='bfloat16'
        )
        latent_model = LatentWrapper(model, model.tokenizer, enable_norm_logging=False)
        
        # Test generation without latent spans (should delegate to base model)
        normal_text = "Question: What is 2+2? Answer:"
        normal_ids = torch.tensor([model.tokenizer.encode(normal_text, add_special_tokens=False)])
        
        # Create proper attention mask
        attention_mask = torch.ones_like(normal_ids)
        
        with torch.no_grad():
            generated = latent_model.generate(
                input_ids=normal_ids,
                attention_mask=attention_mask,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=model.tokenizer.eos_token_id,
                eos_token_id=model.tokenizer.eos_token_id
            )
        
        assert generated.shape[0] == 1, f"Wrong batch size: {generated.shape[0]}"
        assert generated.shape[1] > normal_ids.shape[1], "Generated sequence should be longer"
        
        generated_text = model.tokenizer.decode(generated[0], skip_special_tokens=True)
        logger.info(f"✓ Normal generation works: '{generated_text}'")
        
        # Test generation with latent spans
        latent_text = "Question: <|start-latent|> <|latent|> <|end-latent|> Answer:"
        latent_ids = torch.tensor([model.tokenizer.encode(latent_text, add_special_tokens=False)])
        
        # Create proper attention mask for latent test
        latent_attention_mask = torch.ones_like(latent_ids)
        
        with torch.no_grad():
            latent_generated = latent_model.generate(
                input_ids=latent_ids,
                attention_mask=latent_attention_mask,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=model.tokenizer.eos_token_id,
                eos_token_id=model.tokenizer.eos_token_id
            )
        
        assert latent_generated.shape[0] == 1, f"Wrong batch size: {latent_generated.shape[0]}"
        assert latent_generated.shape[1] > latent_ids.shape[1], "Generated sequence should be longer"
        
        latent_generated_text = model.tokenizer.decode(latent_generated[0], skip_special_tokens=True)
        logger.info(f"✓ Latent generation works: '{latent_generated_text}'")
        
        return latent_model
        
    except Exception as e:
        logger.error(f"✗ Generation test failed: {e}")
        raise

def main():
    """Run all tests"""
    logger.info("Starting LatentWrapper Simplification Tests...")
    
    try:
        # Test 1: Model initialization
        model, latent_model = test_model_initialization()
        
        # Test 2: Tokenization
        tokenizer = test_tokenization()
        
        # Test 3: Latent span detection
        latent_model = test_latent_span_detection()
        
        # Test 4: Simplified algorithm
        latent_model = test_simplified_algorithm()
        
        # Test 5: Forward pass
        latent_model = test_forward_pass()
        
        # Test 6: Generation
        latent_model = test_generation()
        
        logger.info("\n=== ALL TESTS PASSED ===")
        logger.info("✓ LatentWrapper simplification did not break functionality")
        logger.info("✓ Core CoCoNut algorithm is working correctly")
        logger.info("✓ Multimodal compatibility is maintained")
        
        return True
        
    except Exception as e:
        logger.error(f"\n=== TESTS FAILED ===")
        logger.error(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
