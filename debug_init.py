#!/usr/bin/env python3

"""
Debug script to understand the MultiCoCo initialization issue
"""

import sys
import os

# Add the multicoco directory to the path
sys.path.append('/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco')

def debug_attribute_access():
    """Debug what happens during MultiCoCo initialization"""
    
    print("🔍 Debugging MultiCoCo initialization...")
    
    try:
        from multicoco.model import MultiCoCo
        from multicoco.constants import COCONUT_SPECIAL_TOKENS
        
        print("✅ Successfully imported MultiCoCo")
        
        # Let's add some debug prints to understand what's happening
        
        class DebugMultiCoCo(MultiCoCo):
            def __init__(self, *args, **kwargs):
                print("🐛 Starting MultiCoCo.__init__")
                super().__init__(*args, **kwargs)
                print("🐛 Finished MultiCoCo.__init__")
            
            def __getattr__(self, name):
                print(f"🐛 __getattr__ called for attribute: {name}")
                print(f"🐛 self.__dict__.keys(): {list(self.__dict__.keys())}")
                
                # Check if 'model' is in __dict__ and what it is
                if 'model' in self.__dict__:
                    print(f"🐛 self.__dict__['model'] exists: {type(self.__dict__['model'])}")
                else:
                    print("🐛 'model' not in self.__dict__")
                
                return super().__getattr__(name)
        
        # Try to create the debug version
        print("📦 Creating DebugMultiCoCo instance...")
        multicoco_model = DebugMultiCoCo(
            model_id="OpenGVLab/InternVL2_5-1B",
            special_tokens=list(COCONUT_SPECIAL_TOKENS),
            torch_dtype="bfloat16"
        )
        
        print("✅ SUCCESS: DebugMultiCoCo created successfully!")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_attribute_access()
