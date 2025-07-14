#!/usr/bin/env python3
# MultiCoCo Code-Level Memory Issue Detector
# This script helps identify code-level memory leaks in the MultiCoCo implementation

import os
import sys
import argparse
import logging
import traceback
import gc
import torch
from torch.utils.data import DataLoader
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_gpu_memory():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / (1024**3),  # GB
            "reserved": torch.cuda.memory_reserved() / (1024**3),    # GB
            "max_allocated": torch.cuda.max_memory_allocated() / (1024**3)  # GB
        }
    return {"allocated": 0, "reserved": 0, "max_allocated": 0}

def check_model_initialization(config_file):
    """Test model initialization for memory leaks."""
    logger.info("Testing model initialization...")
    
    try:
        # Dynamically import required modules to avoid errors if not testing full functionality
        from multicoco.model import MultiCoCo
        from multicoco.config import MultiCoCoConfig, TrainingMode
        
        # Load configuration file
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Record memory before model creation
        before_mem = get_gpu_memory()
        logger.info(f"Memory before model init: {before_mem['allocated']:.2f} GB allocated")
        
        # Create model configuration
        model_config = MultiCoCoConfig.from_dict(config_dict)
        
        # Initialize model
        model = MultiCoCo(
            model_id=model_config.model.model_name, 
            config_id=model_config.model.config_id,
            tokenizer_id=model_config.model.tokenizer_id,
            image_processor_id=model_config.model.image_processor_id,
            special_tokens=model_config.coconut.special_tokens if model_config.coconut.enabled else None,
            torch_dtype=model_config.model.torch_dtype,
            trust_remote_code=model_config.model.trust_remote_code,
            low_cpu_mem_usage=model_config.model.low_cpu_mem_usage
        )
        
        # Move model to GPU
        if torch.cuda.is_available():
            model.cuda()
            torch.cuda.synchronize()  # Make sure CUDA operations are completed
            
        # Record memory after model creation
        torch.cuda.synchronize()
        after_mem = get_gpu_memory()
        logger.info(f"Memory after model init: {after_mem['allocated']:.2f} GB allocated")
        logger.info(f"Difference: {after_mem['allocated'] - before_mem['allocated']:.2f} GB")
        
        # Additional model info
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {num_params:,}")
        logger.info(f"Model dtype: {next(model.parameters()).dtype}")
        
        # Check if this is a coconut model and if we need to wrap it
        if model_config.coconut.enabled and model_config.training.mode == TrainingMode.COCONUT_TRAIN:
            logger.info("Testing LatentWrapper initialization...")
            
            from multicoco.latent_wrapper import LatentWrapper
            
            # Record memory before wrapper creation
            before_wrap_mem = get_gpu_memory()
            
            # Create wrapper
            wrapped_model = LatentWrapper(model, model.tokenizer)
            
            # Move model to GPU
            if torch.cuda.is_available():
                wrapped_model.cuda()
                torch.cuda.synchronize()
                
            # Record memory after wrapper creation
            after_wrap_mem = get_gpu_memory()
            logger.info(f"Memory after wrapper init: {after_wrap_mem['allocated']:.2f} GB allocated")
            logger.info(f"Wrapper overhead: {after_wrap_mem['allocated'] - before_wrap_mem['allocated']:.2f} GB")
        
        return True, model
        
    except Exception as e:
        logger.error(f"Error during model initialization: {str(e)}")
        traceback.print_exc()
        return False, None

def check_data_loading(config_file, model=None):
    """Test data loading for memory leaks."""
    logger.info("Testing data loading...")
    
    try:
        # Dynamically import required modules
        from multicoco.data import SupervisedDataset, collate_fn
        from multicoco.config import MultiCoCoConfig
        
        # Load configuration
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        config = MultiCoCoConfig.from_dict(config_dict)
        
        # Record memory before data loading
        before_mem = get_gpu_memory()
        logger.info(f"Memory before data loading: {before_mem['allocated']:.2f} GB allocated")
        
        # Create dataset
        dataset = SupervisedDataset(
            data_path=config.data.train_data_path,
            tokenizer=model.tokenizer if model else None,
            image_processor=model.image_processor if model else None,
            max_length=config.training.max_length,
            stage=0,  # Start with stage 0
            coconut_config=config.coconut
        )
        
        # Record memory after dataset creation
        after_dataset_mem = get_gpu_memory()
        logger.info(f"Memory after dataset creation: {after_dataset_mem['allocated']:.2f} GB allocated")
        logger.info(f"Dataset overhead: {after_dataset_mem['allocated'] - before_mem['allocated']:.2f} GB")
        
        # Create data loader with a small batch size
        safe_batch_size = 1
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=safe_batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Test loading a single batch
        logger.info(f"Testing loading a single batch (size={safe_batch_size})...")
        batch = next(iter(dataloader))
        
        # Record memory after batch loading
        after_batch_mem = get_gpu_memory()
        logger.info(f"Memory after batch loading: {after_batch_mem['allocated']:.2f} GB allocated")
        logger.info(f"Batch overhead: {after_batch_mem['allocated'] - after_dataset_mem['allocated']:.2f} GB")
        
        # Print batch keys and shapes
        logger.info("Batch contents:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"  {k}: {v.shape}")
            elif v is None:
                logger.info(f"  {k}: None")
            else:
                logger.info(f"  {k}: {type(v)}")
                
        return True, dataset, dataloader
        
    except Exception as e:
        logger.error(f"Error during data loading: {str(e)}")
        traceback.print_exc()
        return False, None, None

def check_forward_pass(config_file, model=None, dataloader=None):
    """Test forward pass for memory leaks."""
    logger.info("Testing forward pass...")
    
    if model is None or dataloader is None:
        logger.error("Model or dataloader not provided")
        return False
        
    try:
        # Get a batch
        batch = next(iter(dataloader))
        
        # Move batch to GPU
        if torch.cuda.is_available():
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        
        # Record memory before forward pass
        before_mem = get_gpu_memory()
        logger.info(f"Memory before forward pass: {before_mem['allocated']:.2f} GB allocated")
        
        # Make sure model is in eval mode to avoid storing gradients
        model.eval()
        
        # Forward pass with no grad
        with torch.no_grad():
            outputs = model(**batch)
        
        # Force CUDA synchronization
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        # Record memory after forward pass
        after_mem = get_gpu_memory()
        logger.info(f"Memory after forward pass: {after_mem['allocated']:.2f} GB allocated")
        logger.info(f"Forward pass overhead: {after_mem['allocated'] - before_mem['allocated']:.2f} GB")
        
        # Run garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # Check memory after garbage collection
        after_gc_mem = get_gpu_memory()
        logger.info(f"Memory after garbage collection: {after_gc_mem['allocated']:.2f} GB allocated")
        logger.info(f"Memory freed by GC: {after_mem['allocated'] - after_gc_mem['allocated']:.2f} GB")
        
        # Check if there's a memory leak
        if after_gc_mem['allocated'] > before_mem['allocated'] + 0.1:  # 0.1 GB threshold
            logger.warning(f"Potential memory leak detected! {after_gc_mem['allocated'] - before_mem['allocated']:.2f} GB not freed after GC")
        else:
            logger.info("No significant memory leak detected in forward pass")
            
        return True
        
    except Exception as e:
        logger.error(f"Error during forward pass: {str(e)}")
        traceback.print_exc()
        return False

def check_latent_wrapper_logic(model=None):
    """Check the LatentWrapper logic for issues."""
    if model is None:
        logger.error("Model not provided")
        return False
        
    try:
        logger.info("Checking LatentWrapper implementation...")
        
        # Check if model is LatentWrapper
        from multicoco.latent_wrapper import LatentWrapper
        if not isinstance(model, LatentWrapper):
            logger.info("Model is not a LatentWrapper, skipping this check")
            return True
            
        # Check embedding reference handling
        logger.info("Checking embedding reference handling...")
        
        # Check if _embedding_ref is properly set
        if not hasattr(model, '_embedding_ref'):
            logger.error("LatentWrapper doesn't have _embedding_ref attribute")
            return False
            
        # Check embedding sharing issue - a common cause of memory problems
        base_embed = model.base_model.get_input_embeddings()
        wrapper_embed_ref = getattr(model, '_embedding_ref')
        
        logger.info(f"Base model embedding: {base_embed}")
        logger.info(f"Wrapper embedding ref: {wrapper_embed_ref}")
        
        if base_embed is not wrapper_embed_ref:
            logger.error("Embedding reference mismatch! This can cause memory issues.")
            return False
        else:
            logger.info("Embedding references match correctly")
            
        # Check for KV cache handling in generation
        logger.info("Checking generation code for KV cache issues...")
        import inspect
        gen_code = inspect.getsource(model._generate_with_latent_injection)
        
        # Check for indicators of potential issues
        if "use_cache" not in gen_code:
            logger.warning("No 'use_cache' parameter found in generation code")
        if "past_key_values" not in gen_code:
            logger.warning("No 'past_key_values' handling found in generation code")
            
        return True
        
    except Exception as e:
        logger.error(f"Error during LatentWrapper check: {str(e)}")
        traceback.print_exc()
        return False

def check_model_code_issues():
    """Check for potential code issues in the MultiCoCo implementation."""
    logger.info("Checking for potential code issues...")
    
    try:
        # Import necessary modules
        import multicoco
        import inspect
        
        # Check for common issues in different modules
        
        # 1. Check LatentWrapper implementation
        from multicoco.latent_wrapper import LatentWrapper
        latent_wrapper_code = inspect.getsource(LatentWrapper)
        
        # Check for tensor cloning/copying
        issues_found = 0
        
        if "inputs_embeds.clone()" in latent_wrapper_code:
            logger.warning("Unnecessary tensor cloning in LatentWrapper may increase memory usage")
            issues_found += 1
            
        # 2. Check trainer implementation for memory issues
        from multicoco.trainer import CoCoTrainer
        trainer_code = inspect.getsource(CoCoTrainer)
        
        if "gc.collect()" not in trainer_code:
            logger.warning("CoCoTrainer doesn't explicitly call garbage collection between steps/epochs")
            issues_found += 1
            
        # 3. Check for proper cleaning of CUDA cache
        if "torch.cuda.empty_cache()" not in trainer_code:
            logger.warning("CoCoTrainer doesn't explicitly clear CUDA cache between steps/epochs")
            issues_found += 1
            
        # 4. Check model.py for issues
        from multicoco.model import MultiCoCo
        model_code = inspect.getsource(MultiCoCo)
        
        if "_ensure_dtype_consistency" not in model_code:
            logger.warning("No explicit dtype consistency check in MultiCoCo")
            issues_found += 1
            
        # Report findings
        if issues_found > 0:
            logger.info(f"Found {issues_found} potential code issues that could affect memory usage")
        else:
            logger.info("No obvious code issues found")
            
        return True
        
    except Exception as e:
        logger.error(f"Error during code check: {str(e)}")
        traceback.print_exc()
        return False

def check_train_step(config_file, model=None, dataloader=None):
    """Simulate a training step to check for memory issues."""
    logger.info("Testing train step...")
    
    if model is None or dataloader is None:
        logger.error("Model or dataloader not provided")
        return False
        
    try:
        # Prepare for training
        model.train()
        
        # Get a batch
        batch = next(iter(dataloader))
        
        # Move batch to GPU
        if torch.cuda.is_available():
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        
        # Create a simple optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        # Record memory before training step
        before_mem = get_gpu_memory()
        logger.info(f"Memory before train step: {before_mem['allocated']:.2f} GB allocated")
        
        # Forward pass (with gradient computation)
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
        
        # Record memory after forward pass
        after_forward_mem = get_gpu_memory()
        logger.info(f"Memory after forward pass: {after_forward_mem['allocated']:.2f} GB allocated")
        logger.info(f"Forward pass overhead: {after_forward_mem['allocated'] - before_mem['allocated']:.2f} GB")
        
        # Backward pass
        loss.backward()
        
        # Record memory after backward pass
        after_backward_mem = get_gpu_memory()
        logger.info(f"Memory after backward pass: {after_backward_mem['allocated']:.2f} GB allocated")
        logger.info(f"Backward pass overhead: {after_backward_mem['allocated'] - after_forward_mem['allocated']:.2f} GB")
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Record memory after optimizer step
        after_optim_mem = get_gpu_memory()
        logger.info(f"Memory after optimizer step: {after_optim_mem['allocated']:.2f} GB allocated")
        logger.info(f"Optimizer step overhead: {after_optim_mem['allocated'] - after_backward_mem['allocated']:.2f} GB")
        
        # Run garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # Check memory after garbage collection
        after_gc_mem = get_gpu_memory()
        logger.info(f"Memory after garbage collection: {after_gc_mem['allocated']:.2f} GB allocated")
        logger.info(f"Memory freed by GC: {after_optim_mem['allocated'] - after_gc_mem['allocated']:.2f} GB")
        
        # Check if there's a memory leak
        if after_gc_mem['allocated'] > before_mem['allocated'] + 0.1:  # 0.1 GB threshold
            logger.warning(f"Potential memory leak detected! {after_gc_mem['allocated'] - before_mem['allocated']:.2f} GB not freed after GC")
        else:
            logger.info("No significant memory leak detected in training step")
            
        return True
        
    except Exception as e:
        logger.error(f"Error during train step: {str(e)}")
        traceback.print_exc()
        return False

def fix_common_issues(config_file):
    """Apply fixes for common MultiCoCo memory issues."""
    logger.info(f"Applying fixes to config file: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Output file name
        output_file = f"{config_file.rsplit('.', 1)[0]}_fixed.yaml"
        
        # Fix 1: Add or enable gradient checkpointing
        if 'gradient_checkpointing' not in config_dict:
            config_dict['gradient_checkpointing'] = True
            logger.info("✅ Added gradient_checkpointing: true")
        elif not config_dict.get('gradient_checkpointing'):
            config_dict['gradient_checkpointing'] = True
            logger.info("✅ Enabled gradient_checkpointing")
        
        # Fix 2: Reduce batch size if needed
        if config_dict.get('batch_size', 16) > 8:
            orig_batch = config_dict.get('batch_size', 16)
            config_dict['batch_size'] = 8
            logger.info(f"✅ Reduced batch_size from {orig_batch} to 8")
        
        # Fix 3: Add or adjust gradient accumulation steps
        if 'gradient_accumulation_steps' not in config_dict:
            config_dict['gradient_accumulation_steps'] = 2
            logger.info("✅ Added gradient_accumulation_steps: 2")
        
        # Fix 4: Set mixed precision training
        if 'bf16' not in config_dict:
            config_dict['bf16'] = True
            logger.info("✅ Added bf16: true for mixed precision training")
            
        # Fix 5: Add max_length if not present to control sequence length
        if 'max_length' not in config_dict.get('training', {}):
            if 'training' not in config_dict:
                config_dict['training'] = {}
            config_dict['training']['max_length'] = 512
            logger.info("✅ Added max_length: 512 to control sequence length")
            
        # Fix 6: Optimize eval batch size
        if 'eval_batch_size' not in config_dict:
            config_dict['eval_batch_size'] = 4
            logger.info("✅ Added eval_batch_size: 4")
        elif config_dict.get('eval_batch_size', 64) > 4:
            orig_eval_batch = config_dict.get('eval_batch_size')
            config_dict['eval_batch_size'] = 4
            logger.info(f"✅ Reduced eval_batch_size from {orig_eval_batch} to 4")
            
        # Fix 7: Add eval accumulation steps if not present
        if 'eval_accumulation_steps' not in config_dict:
            config_dict['eval_accumulation_steps'] = 2
            logger.info("✅ Added eval_accumulation_steps: 2")
            
        # Fix 8: Add torch compile setting
        if 'torch_compile' not in config_dict.get('model', {}):
            if 'model' not in config_dict:
                config_dict['model'] = {}
            config_dict['model']['torch_compile'] = False  # Default to False to be safe
            logger.info("✅ Added torch_compile setting (default: false)")
            
        # Save fixed config
        with open(output_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
            
        logger.info(f"✅ Saved fixed config to {output_file}")
        logger.info(f"Run training with: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nnodes 1 --nproc_per_node 1 run.py {output_file}")
        
        return True, output_file
        
    except Exception as e:
        logger.error(f"Error applying fixes: {str(e)}")
        traceback.print_exc()
        return False, None

def find_memory_issues(config_file=None):
    """Main function to find and fix memory issues."""
    logger.info("=" * 80)
    logger.info("Running MultiCoCo memory issue detector")
    logger.info("=" * 80)
    
    if config_file is None or not os.path.exists(config_file):
        logger.error(f"Config file not found: {config_file}")
        return False
    
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, running in CPU mode")
    
    try:
        # Step 1: Check for code issues
        check_model_code_issues()
        
        # Step 2: Initialize model and check for issues
        model_init_ok, model = check_model_initialization(config_file)
        if not model_init_ok:
            logger.error("Model initialization failed, applying fixes...")
            fix_success, fixed_config = fix_common_issues(config_file)
            return False
        
        # Step 3: Check data loading
        data_load_ok, dataset, dataloader = check_data_loading(config_file, model)
        if not data_load_ok:
            logger.error("Data loading failed, applying fixes...")
            fix_success, fixed_config = fix_common_issues(config_file)
            return False
        
        # Step 4: Check forward pass
        forward_ok = check_forward_pass(config_file, model, dataloader)
        if not forward_ok:
            logger.error("Forward pass failed, applying fixes...")
            fix_success, fixed_config = fix_common_issues(config_file)
            return False
        
        # Step 5: Check training step
        train_ok = check_train_step(config_file, model, dataloader)
        if not train_ok:
            logger.error("Training step failed, applying fixes...")
            fix_success, fixed_config = fix_common_issues(config_file)
            return False
        
        # Step 6: Check LatentWrapper logic if applicable
        from multicoco.latent_wrapper import LatentWrapper
        if isinstance(model, LatentWrapper):
            wrapper_ok = check_latent_wrapper_logic(model)
            if not wrapper_ok:
                logger.error("LatentWrapper issues found, applying fixes...")
                fix_success, fixed_config = fix_common_issues(config_file)
                return False
        
        # All checks passed
        logger.info("=" * 80)
        logger.info("All checks passed successfully")
        logger.info("=" * 80)
        
        # Still provide fixed config with optimizations
        logger.info("Generating optimized config with best practices...")
        fix_success, fixed_config = fix_common_issues(config_file)
        
        return True
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='MultiCoCo Memory Issue Detector')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--fix-only', action='store_true', help='Only apply fixes without running tests')
    args = parser.parse_args()
    
    if args.fix_only:
        if args.config:
            fix_success, fixed_config = fix_common_issues(args.config)
            if fix_success:
                logger.info(f"Successfully created optimized config: {fixed_config}")
                return 0
            else:
                logger.error("Failed to create optimized config")
                return 1
        else:
            logger.error("Config file path is required with --fix-only")
            return 1
    
    if not args.config:
        logger.error("Config file path is required")
        return 1
        
    success = find_memory_issues(args.config)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
