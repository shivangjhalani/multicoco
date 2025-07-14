#!/usr/bin/env python3
"""
Validation script to test token encoding/decoding after standardization fix.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from multicoco.constants import COCONUT_SPECIAL_TOKENS, START_LATENT_TOKEN, END_LATENT_TOKEN, LATENT_TOKEN
from multicoco.model import MultiCoCo

def test_token_standardization():
    """Test that the corrected tokens work properly with the tokenizer."""
    print("=== Token Standardization Validation ===")
    
    # Print the corrected token names
    print(f"START_LATENT_TOKEN: {START_LATENT_TOKEN}")
    print(f"END_LATENT_TOKEN: {END_LATENT_TOKEN}")
    print(f"LATENT_TOKEN: {LATENT_TOKEN}")
    print(f"COCONUT_SPECIAL_TOKENS: {COCONUT_SPECIAL_TOKENS}")
    
    # Verify that tokens use hyphens instead of underscores
    assert START_LATENT_TOKEN == '<|start-latent|>', f"Expected <|start-latent|>, got {START_LATENT_TOKEN}"
    assert END_LATENT_TOKEN == '<|end-latent|>', f"Expected <|end-latent|>, got {END_LATENT_TOKEN}"
    assert LATENT_TOKEN == '<|latent|>', f"Expected <|latent|>, got {LATENT_TOKEN}"
    
    print("✓ Token names correctly use hyphens")
    
    try:
        # Initialize model with special tokens
        model = MultiCoCo(special_tokens=COCONUT_SPECIAL_TOKENS)
        tokenizer = model.tokenizer
        
        # Test tokenization of each special token
        for token in COCONUT_SPECIAL_TOKENS:
            token_id = tokenizer.convert_tokens_to_ids(token)
            decoded_token = tokenizer.convert_ids_to_tokens(token_id)
            
            print(f"Token: {token}")
            print(f"  ID: {token_id}")
            print(f"  Decoded: {decoded_token}")
            
            # Verify token was properly added (not unknown)
            assert token_id != tokenizer.unk_token_id, f"Token {token} was not properly added to tokenizer"
            assert decoded_token == token, f"Token {token} roundtrip failed: {decoded_token}"
        
        print("✓ All special tokens encode/decode correctly")
        
        # Test a simple latent reasoning sequence
        test_sequence = f"Question: What is 2+2? {START_LATENT_TOKEN} {LATENT_TOKEN} {LATENT_TOKEN} {END_LATENT_TOKEN} Answer: 4"
        encoded = tokenizer.encode(test_sequence, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        
        print(f"Test sequence: {test_sequence}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
        
        # Verify the sequence contains our special tokens
        assert START_LATENT_TOKEN in decoded, f"Start token missing from decoded sequence"
        assert END_LATENT_TOKEN in decoded, f"End token missing from decoded sequence"
        assert LATENT_TOKEN in decoded, f"Latent token missing from decoded sequence"
        
        print("✓ Latent reasoning sequence tokenizes correctly")
        print("=== Token Standardization: PASSED ===")
        return True
        
    except Exception as e:
        print(f"✗ Token standardization test failed: {e}")
        print("=== Token Standardization: FAILED ===")
        return False

if __name__ == "__main__":
    success = test_token_standardization()
    sys.exit(0 if success else 1)
