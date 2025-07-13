#!/usr/bin/env python3
"""
Simple test to verify LatentWrapper basic structure and imports.
"""

import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_import_and_basic_structure():
    """Test that we can import the module and check basic structure"""
    print("Testing LatentWrapper import and basic structure...")
    
    try:
        # Test import
        from multicoco.latent_wrapper import LatentWrapper
        print("✅ Successfully imported LatentWrapper")
        
        # Check class structure
        assert hasattr(LatentWrapper, '__init__'), "Should have __init__ method"
        assert hasattr(LatentWrapper, '__getattr__'), "Should have __getattr__ method"
        assert hasattr(LatentWrapper, 'forward'), "Should have forward method"
        assert hasattr(LatentWrapper, 'generate'), "Should have generate method"
        print("✅ Class has required methods")
        
        # Check properties
        assert hasattr(LatentWrapper, 'model'), "Should have model property"
        assert hasattr(LatentWrapper, 'device'), "Should have device property"
        print("✅ Class has required properties")
        
        print("\n🎉 Basic structure test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_constants_import():
    """Test that constants are available"""
    print("Testing constants import...")
    
    try:
        from multicoco.constants import COCONUT_SPECIAL_TOKENS
        print(f"✅ COCONUT_SPECIAL_TOKENS: {COCONUT_SPECIAL_TOKENS}")
        
        assert isinstance(COCONUT_SPECIAL_TOKENS, (list, tuple)), "Should be a list or tuple"
        assert len(COCONUT_SPECIAL_TOKENS) > 0, "Should not be empty"
        print("✅ Constants test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Constants test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Running basic LatentWrapper tests (no torch required)")
    print("=" * 50)
    
    test1 = test_import_and_basic_structure()
    print("-" * 30)
    test2 = test_constants_import()
    
    if test1 and test2:
        print("\n🎉 All basic tests passed!")
        print("The LatentWrapper module structure is correct.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)
