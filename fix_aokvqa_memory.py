#!/usr/bin/env python3
"""
Quick fix for CUDA OOM error in MultiCoCo training with aokvqa_cot.yaml.
This script specifically addresses the observed error in the logs.

The key issue appears to be memory fragmentation, which can be fixed by:
1. Setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
2. Adjusting batch size and gradient accumulation
3. Enabling gradient checkpointing 
"""

import os
import sys
import yaml
import subprocess
import argparse
from pathlib import Path

def print_colored(text, color="white"):
    """Print colored text to console."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "end": "\033[0m"
    }
    print(f"{colors.get(color, colors['white'])}{text}{colors['end']}")

def optimize_config(config_path, output_path=None):
    """
    Optimize the config file to prevent OOM errors.
    
    Args:
        config_path: Path to the original config file
        output_path: Path to save the optimized config (if None, will use a default)
    
    Returns:
        Path to the optimized config file
    """
    if not os.path.exists(config_path):
        print_colored(f"Error: Config file not found: {config_path}", "red")
        sys.exit(1)
    
    # Default output path
    if output_path is None:
        output_path = str(Path(config_path).with_name(f"{Path(config_path).stem}_optimized.yaml"))
    
    print_colored(f"Reading config: {config_path}", "blue")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply optimizations
    print_colored("Applying memory optimizations...", "blue")
    
    # 1. Adjust batch size
    original_batch_size = config.get('batch_size', 16)
    new_batch_size = min(original_batch_size, 8)  # Max 8 for multimodal
    config['batch_size'] = new_batch_size
    print_colored(f"✓ Batch size: {original_batch_size} → {new_batch_size}", "green")
    
    # 2. Set gradient accumulation
    original_grad_accum = config.get('gradient_accumulation_steps', 1)
    if original_batch_size > new_batch_size:
        # Maintain effective batch size
        new_grad_accum = max(original_grad_accum, original_batch_size // new_batch_size)
    else:
        new_grad_accum = max(original_grad_accum, 2)  # Minimum 2
    
    config['gradient_accumulation_steps'] = new_grad_accum
    print_colored(f"✓ Gradient accumulation: {original_grad_accum} → {new_grad_accum}", "green")
    
    # 3. Enable gradient checkpointing
    config['gradient_checkpointing'] = True
    print_colored(f"✓ Enabled gradient checkpointing", "green")
    
    # 4. Optionally enable mixed precision
    if 'bf16' not in config:
        # If model supports bfloat16, use it
        config['bf16'] = True
        print_colored(f"✓ Enabled bfloat16 mixed precision", "green")
    
    # 5. Optimize eval settings
    config['eval_batch_size'] = min(config.get('eval_batch_size', 64), 32)
    config['eval_accumulation_steps'] = max(config.get('eval_accumulation_steps', 1), 2)
    print_colored(f"✓ Optimized eval batch settings", "green")
    
    # Save the optimized config
    print_colored(f"Saving optimized config to: {output_path}", "blue")
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return output_path

def run_optimized_training(config_path, num_nodes=1, num_procs=1, **kwargs):
    """Run the training with optimized memory settings."""
    # Set environment variable for memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Build the torchrun command
    cmd = [
        "torchrun",
        "--nnodes", str(num_nodes),
        "--nproc_per_node", str(num_procs),
        "run.py", 
        config_path
    ]
    
    # Print command
    print_colored("\n" + "="*80, "purple")
    print_colored(" RUNNING OPTIMIZED TRAINING ", "purple")
    print_colored("="*80, "purple")
    print_colored("Command: " + " ".join(cmd), "cyan")
    print_colored("Environment: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "cyan")
    print_colored("="*80 + "\n", "purple")
    
    # Execute the command
    try:
        subprocess.run(cmd, check=True, env=os.environ)
    except subprocess.CalledProcessError as e:
        print_colored(f"Error running training: {e}", "red")
        return False
    except KeyboardInterrupt:
        print_colored("Training interrupted by user", "yellow")
        return False
    
    return True

def clear_gpu_memory():
    """Clear GPU memory caches."""
    print_colored("Clearing GPU memory...", "blue")
    
    # Try to force CUDA to release memory
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print_colored("✓ CUDA cache cleared using torch.cuda.empty_cache()", "green")
    except ImportError:
        print_colored("PyTorch not available, skipping CUDA cache clearing", "yellow")
    
    # Try to run nvidia-smi to reset GPU stats
    try:
        subprocess.run(["nvidia-smi", "-r"], check=False)
        print_colored("✓ Attempted nvidia-smi reset", "green")
    except:
        pass  # Ignore if nvidia-smi fails
    
    # Force garbage collection
    import gc
    gc.collect()
    print_colored("✓ Python garbage collection performed", "green")

def main():
    parser = argparse.ArgumentParser(description="Fix CUDA OOM errors in MultiCoCo training")
    parser.add_argument("--config", type=str, default="args/aokvqa_cot.yaml",
                        help="Path to config file (default: args/aokvqa_cot.yaml)")
    parser.add_argument("--output", type=str, 
                        help="Path to save optimized config (default: *_optimized.yaml)")
    parser.add_argument("--run", action="store_true",
                        help="Run the training after optimization")
    parser.add_argument("--num-procs", type=int, default=1,
                        help="Number of processes for torchrun (default: 1)")
    parser.add_argument("--clear-memory", action="store_true",
                        help="Clear GPU memory before running")
                        
    args = parser.parse_args()
    
    print_colored("\n" + "="*80, "blue")
    print_colored(" MULTICOCO MEMORY OPTIMIZATION ", "blue")
    print_colored("="*80 + "\n", "blue")
    
    # Clear memory if requested
    if args.clear_memory:
        clear_gpu_memory()
    
    # Optimize the config
    optimized_config = optimize_config(args.config, args.output)
    
    print_colored("\n" + "="*80, "green")
    print_colored(" OPTIMIZATION COMPLETE ", "green")
    print_colored("="*80, "green")
    print_colored(f"Optimized config saved to: {optimized_config}", "cyan")
    
    if args.run:
        print_colored("\nStarting training with optimized settings...", "blue")
        run_optimized_training(optimized_config, num_procs=args.num_procs)
    else:
        print_colored("\nTo run training with optimized settings:", "yellow")
        print_colored(f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nnodes 1 --nproc_per_node {args.num_procs} run.py {optimized_config}", "cyan")
    
    print_colored("\nOptimization process completed.", "green")

if __name__ == "__main__":
    main()
