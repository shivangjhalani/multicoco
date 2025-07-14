#!/usr/bin/env python3
"""
Debug script to understand why self.model assignment is failing
"""

import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor

# Add multicoco to path
sys.path.append('/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco')

from multicoco.model import MultiCoCo
from multicoco.constants import COCONUT_SPECIAL_TOKENS

class SuperDebugMultiCoCo(MultiCoCo):
    def __init__(self, *args, **kwargs):
        print("🐛 Starting SuperDebugMultiCoCo.__init__")
        super(MultiCoCo, self).__init__()  # Call nn.Module.__init__
        
        special_tokens = kwargs.get('special_tokens', []) or []
        model_id = args[0] if args else kwargs.get('model_id', 'OpenGVLab/InternVL2_5-1B')
        
        try:
            print("🐛 About to call _initialize_components")
            model, tokenizer, image_processor = self._initialize_components(
                model_id, None, None, None, special_tokens, 'bfloat16', True, True
            )
            
            print(f"🐛 Components initialized:")
            print(f"   model type: {type(model)}")
            print(f"   tokenizer type: {type(tokenizer)}")
            print(f"   image_processor type: {type(image_processor)}")
            
            print("🐛 About to assign self.tokenizer")
            self.tokenizer = tokenizer
            print(f"🐛 self.tokenizer assigned: {'tokenizer' in self.__dict__}")
            
            print("🐛 About to assign self.image_processor")
            self.image_processor = image_processor
            print(f"🐛 self.image_processor assigned: {'image_processor' in self.__dict__}")
            
            print("🐛 About to assign self.model")
            print(f"🐛 Before assignment - self.__dict__.keys(): {list(self.__dict__.keys())}")
            
            # Try to debug what happens during assignment
            object.__setattr__(self, 'model', model)
            
            print(f"🐛 After assignment - self.__dict__.keys(): {list(self.__dict__.keys())}")
            print(f"🐛 'model' in self.__dict__: {'model' in self.__dict__}")
            
            if 'model' in self.__dict__:
                print(f"🐛 self.__dict__['model'] type: {type(self.__dict__['model'])}")
                print("🐛 Model assignment SUCCESS!")
                
                print("🐛 Testing model access...")
                try:
                    test_model = self.model
                    print(f"🐛 Direct model access works: {type(test_model)}")
                except Exception as e:
                    print(f"🐛 Direct model access failed: {e}")
                    
            else:
                print("🐛 Model assignment FAILED - not in __dict__!")
                
        except Exception as e:
            print(f"🐛 Exception during init: {e}")
            import traceback
            traceback.print_exc()

def test_assignment_debug():
    print("🧪 Testing SuperDebugMultiCoCo assignment...")
    
    try:
        debug_model = SuperDebugMultiCoCo(
            model_id="OpenGVLab/InternVL2_5-1B",
            special_tokens=list(COCONUT_SPECIAL_TOKENS),
            torch_dtype="bfloat16"
        )
        print("✅ Model created successfully!")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_assignment_debug()
