#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/home/shivang/shivang/projs/cdsaml/kaggle/scratch/multicoco')

from multicoco.config import MultiCoCoConfig

try:
    # Test loading the config
    config = MultiCoCoConfig.load_with_base('args/aokvqa_cot.yaml')
    print('✅ Configuration loaded successfully!')
    
    # Test basic fields
    print(f'Batch size: {config.training.batch_size}')
    
    # Test new fields
    print(f'Has max_grad_norm: {hasattr(config.training, "max_grad_norm")}')
    if hasattr(config.training, 'max_grad_norm'):
        print(f'Max grad norm: {config.training.max_grad_norm}')
    else:
        print('Max grad norm: NOT_FOUND')
        
    print(f'Has lr_scheduler_type: {hasattr(config.training, "lr_scheduler_type")}')
    if hasattr(config.training, 'lr_scheduler_type'):
        print(f'LR scheduler: {config.training.lr_scheduler_type}')
    else:
        print('LR scheduler: NOT_FOUND')
    
    # Test generation config
    if hasattr(config, 'generation'):
        print(f'Generation temperature: {config.generation.get("temperature", "not set")}')
        print(f'Generation max tokens: {config.generation.get("max_new_tokens", "not set")}')
    else:
        print('Generation config not found')
    
    # Test all fields
    print('\nAll TrainingConfig fields:')
    for field_name in dir(config.training):
        if not field_name.startswith('_'):
            value = getattr(config.training, field_name)
            if not callable(value):
                print(f'  {field_name}: {value}')

except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
