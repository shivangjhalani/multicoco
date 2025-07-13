#!/usr/bin/env python3

import sys
import os
import importlib

# Force clear any cached modules
if 'multicoco.config' in sys.modules:
    del sys.modules['multicoco.config']
if 'multicoco' in sys.modules:
    del sys.modules['multicoco']

sys.path.insert(0, '/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco')

# Fresh import
from multicoco.config import MultiCoCoConfig, TrainingConfig

# Check TrainingConfig fields
print("TrainingConfig fields:")
import dataclasses
for field in dataclasses.fields(TrainingConfig):
    print(f"  {field.name}: {field.type}")

# Check MultiCoCoConfig fields  
print("\nMultiCoCoConfig fields:")
for field in dataclasses.fields(MultiCoCoConfig):
    print(f"  {field.name}: {field.type}")

try:
    # Test loading the config
    config = MultiCoCoConfig.load_with_base('args/aokvqa_cot.yaml')
    print('\n✅ Configuration loaded successfully!')
    
    # Test basic fields
    print(f'Batch size: {config.training.batch_size}')
    
    # Test new fields
    print(f'Has max_grad_norm: {hasattr(config.training, "max_grad_norm")}')
    print(f'Has generation: {hasattr(config, "generation")}')
    
    if hasattr(config.training, 'max_grad_norm'):
        print(f'Max grad norm: {config.training.max_grad_norm}')
    else:
        print('Max grad norm: NOT_FOUND')
        
    if hasattr(config.training, 'lr_scheduler_type'):
        print(f'LR scheduler: {config.training.lr_scheduler_type}')
    else:
        print('LR scheduler: NOT_FOUND')
    
    # Test generation config
    if hasattr(config, 'generation'):
        print(f'Generation config type: {type(config.generation)}')
        print(f'Generation config: {config.generation}')
    else:
        print('Generation config not found')

except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
