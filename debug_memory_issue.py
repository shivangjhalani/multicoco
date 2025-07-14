#!/usr/bin/env python3
"""
Diagnostic script for debugging CUDA out-of-memory issues in MultiCoCo training.
This script analyzes memory usage and potential issues that may be causing OOM errors.

Usage:
    python debug_memory_issue.py [--config CONFIG_PATH] [--check-env-only]

For best results, run this script before attempting to run the training command.
"""

import argparse
import gc
import os
import sys
import yaml
from typing import Dict, Optional, Any, List

# Set environment variable before importing PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    import torch
    import torch.utils.checkpoint as checkpoint_module
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
    
    # Import our MultiCoCo specific modules
    from multicoco.model import MultiCoCo
    from multicoco.data import SupervisedDataset, collate_fn
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure all required packages are installed.")
    sys.exit(1)

def print_section(title):
    """Print a section title with decorative formatting."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def check_cuda_info():
    """Check CUDA availability and display info."""
    print_section("CUDA Information")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available! Please check your installation.")
        return False
    
    print(f"✅ CUDA is available")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
    print(f"Device Count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nDevice {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    return True

def check_memory_usage():
    """Check current GPU memory usage."""
    print_section("Current GPU Memory Usage")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory check.")
        return
    
    # Force garbage collection first
    gc.collect()
    torch.cuda.empty_cache()
    
    for i in range(torch.cuda.device_count()):
        print(f"\nDevice {i} Memory Stats:")
        print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
        
        # Check for fragmentation (reserved but not allocated)
        reserved = torch.cuda.memory_reserved(i)
        allocated = torch.cuda.memory_allocated(i)
        fragmented = reserved - allocated
        print(f"  Fragmented: {fragmented / 1024**3:.2f} GB")
        
        if fragmented > 1 * 1024**3:  # More than 1 GB fragmentation
            print("  ⚠️ High memory fragmentation detected! This could be causing OOM errors.")

    # Try to print detailed memory stats
    try:
        print("\nDetailed Memory Summary:")
        print(torch.cuda.memory_summary())
    except Exception as e:
        print(f"Could not get detailed memory summary: {e}")

def test_model_loading(config_path: Optional[str] = None):
    """Test loading the model to check memory usage."""
    print_section("Testing Model Loading")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping model loading test.")
        return

    # Default settings if no config provided
    model_name = "OpenGVLab/InternVL3-1B-Pretrained"
    torch_dtype = "bfloat16"
    
    # Load config file if provided
    if config_path:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                model_config = config.get('model', {})
                model_name = model_config.get('model_name', model_name)
                torch_dtype = model_config.get('torch_dtype', torch_dtype)
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default settings")
    
    # Print initial memory usage
    print(f"Memory before model loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)
    
    try:
        print(f"Loading model: {model_name} with dtype: {torch_dtype}")
        
        # Record memory usage during loading steps
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"Memory after tokenizer: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        print(f"Memory after image_processor: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Test with gradient checkpointing enabled (memory efficient)
        print("Loading model with gradient checkpointing enabled...")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.use_cache = False
        
        # Try loading with lower precision first
        model = MultiCoCo(
            model_name, 
            config_id=model_name,
            tokenizer_id=model_name,
            image_processor_id=model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Check memory after loading
        print(f"Memory after model loaded: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Check model parameters
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model Parameters: {param_count:,} ({param_count / 1000000:.2f}M)")
        
        # Enable gradient checkpointing if available
        if hasattr(model.model, 'gradient_checkpointing_enable'):
            model.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        
        # Test memory with a small forward pass
        print("\nTesting small forward pass...")
        model = model.cuda()
        print(f"Memory after model to CUDA: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print("✅ Model loading test completed successfully")
        
    except Exception as e:
        print(f"❌ Error during model loading: {e}")
        import traceback
        traceback.print_exc()

def check_environment_variables():
    """Check for proper environment variables that affect PyTorch memory management."""
    print_section("Environment Variables Check")
    
    # Check for common environment variables that affect PyTorch memory usage
    important_vars = [
        'PYTORCH_CUDA_ALLOC_CONF',
        'CUDA_VISIBLE_DEVICES',
        'CUDA_LAUNCH_BLOCKING',
        'CUBLAS_WORKSPACE_CONFIG'
    ]
    
    for var in important_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var} = {value}")
        else:
            if var == 'PYTORCH_CUDA_ALLOC_CONF':
                print(f"⚠️ {var} is not set! Recommended: 'expandable_segments:True'")
            else:
                print(f"ℹ️ {var} is not set")
    
    # Check if we're in a conda environment
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        print(f"Running in conda environment: {conda_prefix}")
    else:
        print("Not running in a conda environment")

def check_config_file(config_path: str):
    """Analyze the configuration file for potential issues."""
    print_section(f"Analyzing Config File: {config_path}")
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check for batch size issues
        batch_size = config.get('batch_size', 16)
        print(f"Batch Size: {batch_size}")
        
        if batch_size > 8 and torch.cuda.is_available():
            print(f"⚠️ Large batch size ({batch_size}) may cause OOM with multimodal models")
            print("   Recommendation: Try reducing batch size to 8 or lower")
        
        # Check for gradient accumulation
        grad_accum = config.get('gradient_accumulation_steps', 1)
        print(f"Gradient Accumulation Steps: {grad_accum}")
        
        if grad_accum == 1 and batch_size > 8:
            print("⚠️ Consider using gradient_accumulation_steps > 1 with large batch size")
        
        # Check for gradient checkpointing
        if not config.get('gradient_checkpointing', False):
            print("⚠️ Gradient checkpointing is not enabled. This could help reduce memory usage.")
        else:
            print("✅ Gradient checkpointing is enabled")
            
        # Generation settings
        gen_config = config.get('generation', {})
        max_new_tokens = gen_config.get('max_new_tokens', 256)
        print(f"Max New Tokens: {max_new_tokens}")
        
        if max_new_tokens > 256:
            print("⚠️ Large max_new_tokens value may require more memory during inference")
            
    except Exception as e:
        print(f"Error analyzing config file: {e}")

def suggest_fixes():
    """Suggest potential fixes for OOM issues."""
    print_section("Suggested Fixes for OOM Issues")
    
    suggestions = [
        "1. Set environment variable: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "2. Reduce batch_size in your config file (try 4 or 8)",
        "3. Increase gradient_accumulation_steps to compensate for smaller batch size",
        "4. Enable gradient checkpointing in your model",
        "5. Restart your environment to clear any lingering GPU memory",
        "6. Ensure no other processes are using GPU memory",
        "7. Try running with PyTorch compiled model: add torch_compile: True to your config",
        "8. Free more CPU memory to reduce system pressure on VRAM paging",
        "9. Update your CUDA and PyTorch versions to the latest compatible versions"
    ]
    
    for suggestion in suggestions:
        print(suggestion)
        
    # Special suggestion for memory fragmentation
    if torch.cuda.is_available():
        reserved = torch.cuda.memory_reserved(0)
        allocated = torch.cuda.memory_allocated(0)
        if (reserved - allocated) > 2 * 1024**3:  # More than 2GB fragmentation
            print("\n⚠️ HIGH MEMORY FRAGMENTATION DETECTED!")
            print("This is likely your issue. Try the following in your code:")
            print("  - Add torch.cuda.empty_cache() calls after large operations")
            print("  - Use del variable_name to explicitly delete large tensors")
            print("  - Call gc.collect() after deleting variables")
            print("  - IMPORTANT: Use PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

def main():
    parser = argparse.ArgumentParser(description="Debug CUDA OOM issues in MultiCoCo")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--check-env-only", action="store_true", 
                        help="Only check environment, skip model loading tests")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(" MULTICOCO MEMORY DIAGNOSTICS TOOL ".center(80, "="))
    print("=" * 80)
    
    # Always check environment variables and CUDA info
    check_environment_variables()
    has_cuda = check_cuda_info()
    
    if has_cuda:
        check_memory_usage()
    
    if args.config:
        check_config_file(args.config)
    
    if has_cuda and not args.check_env_only:
        test_model_loading(args.config)
        check_memory_usage()  # Check again after model tests
    
    suggest_fixes()
    
    print("\nDiagnostics completed. If issues persist, please share this output with the team.")

if __name__ == "__main__":
    main()
