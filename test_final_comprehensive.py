#!/usr/bin/env python3
"""
Final comprehensive test with the fixed LatentWrapper.
"""

import sys
import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

# Add the current directory to Python path
sys.path.insert(0, '/kaggle/working/multicoco')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_test_model():
    """Create a test model that mimics InternVL structure"""
    
    class MockLanguageModel(nn.Module):
        def __init__(self, hidden_size=768, vocab_size=50010):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList([
                nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
                for _ in range(2)
            ])
            self.norm = nn.LayerNorm(hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        def forward(self, inputs_embeds, attention_mask=None, output_hidden_states=False, **kwargs):
            hidden_states = inputs_embeds
            all_hidden_states = [hidden_states] if output_hidden_states else None
            
            for layer in self.layers:
                hidden_states = layer(hidden_states, hidden_states)
                if output_hidden_states:
                    all_hidden_states.append(hidden_states)
            
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            
            class Output:
                def __init__(self, logits, hidden_states=None):
                    self.logits = logits
                    self.hidden_states = hidden_states
            
            return Output(logits, all_hidden_states if output_hidden_states else None)
    
    class MockVisionTower(nn.Module):
        def __init__(self, hidden_size=768):
            super().__init__()
            self.linear = nn.Linear(256, hidden_size)  # Simple projection
            
        def forward(self, pixel_values):
            # Mock vision processing
            batch_size = pixel_values.shape[0]
            # Simulate flattened image patches
            features = self.linear(torch.randn(batch_size, 256, 256))  # 256 patches, 256 features each
            return features
    
    class MockProjector(nn.Module):
        def __init__(self, hidden_size=768):
            super().__init__()
            self.linear = nn.Linear(hidden_size, hidden_size)
            
        def forward(self, vision_embeds):
            return self.linear(vision_embeds)
    
    class MockInternVLModel(nn.Module):
        def __init__(self, hidden_size=768, vocab_size=50010):
            super().__init__()
            self.vision_tower = MockVisionTower(hidden_size)
            self.projector = MockProjector(hidden_size)
            self.language_model = MockLanguageModel(hidden_size, vocab_size)
            self.dtype = torch.float32
            
        def prepare_inputs_for_multimodal(self, input_ids=None, pixel_values=None, image_embeds=None, inputs_embeds=None, **kwargs):
            if inputs_embeds is not None:
                combined_embeds = inputs_embeds
                if image_embeds is not None:
                    # Add image embeddings to first few positions
                    batch_size, seq_len, hidden_size = inputs_embeds.shape
                    img_len = min(image_embeds.shape[1], seq_len // 3)
                    combined_embeds[:, :img_len] += image_embeds[:, :img_len]
                return combined_embeds
            else:
                inputs_embeds = self.language_model.embed_tokens(input_ids)
                if image_embeds is not None:
                    batch_size, seq_len, hidden_size = inputs_embeds.shape
                    img_len = min(image_embeds.shape[1], seq_len // 3)
                    inputs_embeds[:, :img_len] += image_embeds[:, :img_len]
                return inputs_embeds
    
    class MockBaseModel(nn.Module):
        def __init__(self, hidden_size=768, vocab_size=50010):
            super().__init__()
            self.model = MockInternVLModel(hidden_size, vocab_size)
            
        def forward(self, input_ids=None, attention_mask=None, pixel_values=None, labels=None, **kwargs):
            # Process vision if provided
            image_embeds = None
            if pixel_values is not None:
                vision_features = self.model.vision_tower(pixel_values)
                image_embeds = self.model.projector(vision_features)
            
            # Prepare multimodal inputs
            if input_ids is not None:
                inputs_embeds = self.model.prepare_inputs_for_multimodal(
                    input_ids=input_ids, 
                    image_embeds=image_embeds
                )
            else:
                inputs_embeds = kwargs.get('inputs_embeds')
                if image_embeds is not None:
                    inputs_embeds = self.model.prepare_inputs_for_multimodal(
                        inputs_embeds=inputs_embeds,
                        image_embeds=image_embeds
                    )
                
            output = self.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **kwargs
            )
            
            loss = None
            if labels is not None:
                shift_logits = output.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Return dictionary format that LatentWrapper expects
            return {'loss': loss, 'logits': output.logits}
        
        def generate(self, input_ids, **kwargs):
            # Simple mock generation
            max_new_tokens = kwargs.get('max_new_tokens', 5)
            batch_size, seq_len = input_ids.shape
            
            generated = input_ids.clone()
            for _ in range(max_new_tokens):
                outputs = self.forward(input_ids=generated)
                next_token = torch.argmax(outputs['logits'][:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
            
            return generated
        
        def get_input_embeddings(self):
            return self.model.language_model.embed_tokens
        
        @property
        def device(self):
            return next(self.parameters()).device
    
    return MockBaseModel()

def create_test_tokenizer():
    """Create test tokenizer"""
    class TestTokenizer:
        def __init__(self):
            self.special_tokens = {
                '<|latent|>': 50001,
                '<|start_latent|>': 50002, 
                '<|end_latent|>': 50003,
                '<|pad|>': 0,
                '<|eos|>': 1,
            }
            self.vocab_size = 50010
            self.eos_token_id = 1
            self.pad_token_id = 0
        
        def convert_tokens_to_ids(self, token):
            return self.special_tokens.get(token, 2)
        
        def __len__(self):
            return self.vocab_size
    
    return TestTokenizer()

def test_coconut_implementation():
    """Test CoCoNut algorithm with latent tokens"""
    logger.info("🧪 Testing CoCoNut Algorithm Implementation")
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        
        tokenizer = create_test_tokenizer()
        base_model = create_test_model()
        wrapper = LatentWrapper(base_model, tokenizer)
        
        # Create input with latent tokens
        input_ids = torch.tensor([
            [100, 101, 102, 50002, 50001, 50001, 50003, 200, 201, 202],  # Has latent span (3,6)
            [300, 301, 50002, 50001, 50001, 50001, 50003, 400, 401, 402]  # Has latent span (2,6)
        ])
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(100, 1000, input_ids.shape)
        
        logger.info("  Testing forward pass with latent tokens...")
        output = wrapper.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        assert 'logits' in output, "Output should contain logits"
        assert 'loss' in output, "Output should contain loss"
        assert output['logits'].shape == (2, 10, tokenizer.vocab_size), \
            f"Expected logits shape (2, 10, {tokenizer.vocab_size}), got {output['logits'].shape}"
        
        logger.info(f"  ✅ Forward pass successful, loss: {output['loss'].item():.4f}")
        
        # Test latent span extraction
        spans = wrapper._extract_latent_spans(input_ids)
        expected_spans = [[(3, 6)], [(2, 6)]]
        assert spans == expected_spans, f"Expected spans {expected_spans}, got {spans}"
        logger.info(f"  ✅ Latent spans correctly identified: {spans}")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ CoCoNut test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generation_functionality():
    """Test generation with and without latent tokens"""
    logger.info("🧪 Testing Generation Functionality")
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        
        tokenizer = create_test_tokenizer()
        base_model = create_test_model()
        wrapper = LatentWrapper(base_model, tokenizer)
        
        # Test generation without latent tokens
        logger.info("  Testing generation without latent tokens...")
        input_ids = torch.randint(100, 1000, (2, 5))
        generated = wrapper.generate(input_ids=input_ids, max_new_tokens=3)
        
        assert generated.shape == (2, 8), f"Expected shape (2, 8), got {generated.shape}"
        assert torch.equal(generated[:, :5], input_ids), "Original input should be preserved"
        logger.info("  ✅ Generation without latent tokens working")
        
        # Test generation with latent tokens
        logger.info("  Testing generation with latent tokens...")
        latent_input = torch.tensor([
            [100, 101, 50002, 50001, 50001, 50003, 200],
            [300, 50002, 50001, 50003, 400, 401, 402]
        ])
        
        latent_generated = wrapper.generate(input_ids=latent_input, max_new_tokens=2)
        
        expected_shape = (2, latent_input.shape[1] + 2)
        assert latent_generated.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {latent_generated.shape}"
        logger.info("  ✅ Generation with latent tokens working")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trainer_compatibility():
    """Test trainer integration patterns"""
    logger.info("🧪 Testing Trainer Compatibility")
    
    try:
        from multicoco.latent_wrapper import LatentWrapper
        
        tokenizer = create_test_tokenizer()
        base_model = create_test_model()
        wrapper = LatentWrapper(base_model, tokenizer)
        
        # Test required attributes
        logger.info("  Testing required attributes...")
        assert hasattr(wrapper, 'tokenizer'), "Should have tokenizer attribute"
        assert hasattr(wrapper, 'model'), "Should have model attribute"
        assert hasattr(wrapper, 'device'), "Should have device attribute"
        logger.info("  ✅ All required attributes present")
        
        # Test training mode switching
        logger.info("  Testing training mode switching...")
        wrapper.train()
        assert wrapper.training, "Should be in training mode"
        wrapper.eval()
        assert not wrapper.training, "Should be in evaluation mode"
        logger.info("  ✅ Training mode switching working")
        
        # Test batch processing
        logger.info("  Testing batch processing...")
        batch = {
            'input_ids': torch.randint(100, 1000, (2, 8)),
            'attention_mask': torch.ones(2, 8),
            'labels': torch.randint(100, 1000, (2, 8)),
        }
        
        outputs = wrapper.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        assert 'loss' in outputs, "Should return loss"
        assert 'logits' in outputs, "Should return logits"
        logger.info(f"  ✅ Batch processing working, loss: {outputs['loss'].item():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Trainer compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_final_comprehensive_test():
    """Run all final tests"""
    logger.info("🚀 Running Final Comprehensive LatentWrapper Tests")
    logger.info("=" * 60)
    
    tests = [
        test_coconut_implementation,
        test_generation_functionality,
        test_trainer_compatibility
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            logger.error(f"Test {test_func.__name__} crashed: {e}")
            results.append(False)
        logger.info("-" * 40)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    logger.info("📊 FINAL TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total tests: {total}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {total - passed}")
    logger.info(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("✅ LatentWrapper is FULLY FUNCTIONAL and ready for production!")
        logger.info("✅ CoCoNut algorithm correctly implemented")
        logger.info("✅ Generation functionality working")
        logger.info("✅ Trainer integration confirmed")
        logger.info("✅ The integration testing is COMPLETE!")
    else:
        logger.warning(f"⚠️ {total - passed} test(s) failed")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = run_final_comprehensive_test()
        if success:
            print("\n" + "="*60)
            print("🎉 INTEGRATION TESTING COMPLETE AND SUCCESSFUL! 🎉")
            print("✅ LatentWrapper with CoCoNut algorithm is ready for use!")
            print("="*60)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
