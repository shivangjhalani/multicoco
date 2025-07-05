# Set environment variable to disable Flash Attention 2 before importing transformers
import os
os.environ["TRANSFORMERS_NO_FLASH_ATTENTION_2"] = "1"

import argparse
import yaml
import sys
import torch
import wandb

# The previous monkey-patch for _flash_supports_window_size is no longer needed
# and has been removed. The environment variable is the correct way to disable it.

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from copy import copy

# Add the correct path for the internvl package to the system path.
# This ensures that imports within the Hugging Face cached scripts can find the local modules.
internvl_chat_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'internvl', 'internvl_chat'))
if internvl_chat_path not in sys.path:
    sys.path.insert(0, internvl_chat_path)

from multicoco.data import MultiCoCoDataset, DataCollatorForInternVL
from multicoco.model import MultiCoCo
from multicoco.trainer import Trainer

def setup(rank, world_size):
    """Initializes the distributed environment."""
    # os.environ['MASTER_ADDR'] = 'localhost' # This is now handled by torchrun
    # os.environ['MASTER_PORT'] = '12355'   # This is now handled by torchrun
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Cleans up the distributed environment."""
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="MultiCoCo Training Script")
    parser.add_argument('config', type=str, help='Path to the YAML config file')
    cli_args = parser.parse_args()

    with open(cli_args.config, 'r') as f:
        args = yaml.safe_load(f)

    # DDP Setup
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        setup(rank, world_size)
    
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Model
    # Determine model path and special tokens based on the config.
    # If cot or coconut flags are set, use the local patched model and custom tokens.
    # Otherwise, use the model_id from the config for a vanilla run.
    if args.get('cot') or args.get('coconut'):
        model_path = os.path.abspath('local_internvl_model')
        hub_id = args['model_id'] # The original Hub ID for configs/tokenizer
        latent_tokens = {"start": "<|start-latent|>", "end": "<|end-latent|>", "latent": "<|latent|>"}
        special_tokens = list(latent_tokens.values())
    else:
        model_path = args['model_id']
        hub_id = args['model_id'] # For vanilla, local and hub IDs are the same
        latent_tokens = {}
        special_tokens = []

    # -- Initialize Tokenizer and Model
    if args.get('only_eval', False):
        model = MultiCoCo(args['load_model_path'])
        tokenizer = model.tokenizer
    else:
        special_tokens = ['<thought>', '<start_thought>', '<end_thought>']
        model = MultiCoCo(args['model_id'], special_tokens=special_tokens)
        tokenizer = model.tokenizer
        
        # Add special tokens to args to be accessible in the trainer
        args['thought_token_id'] = tokenizer.convert_tokens_to_ids('<thought>')
        args['start_thought_id'] = tokenizer.convert_tokens_to_ids('<start_thought>')
        args['end_thought_id'] = tokenizer.convert_tokens_to_ids('<end_thought>')

    # -- Load Model from Checkpoint if Provided
    if args.get('load_model_path', None) and not args.get('only_eval', False):
        print(f"Loading model from {args['load_model_path']}")
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    tokenizer = model.module.tokenizer if hasattr(model, 'module') else model.tokenizer

    # Data
    hf_model = (model.module if hasattr(model, 'module') else model).model
    image_processor = model.image_processor if not hasattr(model, 'module') else model.module.image_processor
    collator = DataCollatorForInternVL(
        tokenizer=tokenizer,
        model=hf_model,
        image_processor=image_processor
    )

    # Always create val_loader
    val_dataset = MultiCoCoDataset(data_path=args['val_path'], data_dir=args['data_dir'])
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=args['batch_size_training'],
        sampler=val_sampler,
        collate_fn=collator
    )

    # Conditionally create train_loader
    train_loader = None
    if not args.get('only_eval', False):
        train_dataset = MultiCoCoDataset(data_path=args['train_path'], data_dir=args['data_dir'])
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=args['batch_size_training'],
            sampler=train_sampler,
            collate_fn=collator,
            shuffle=(train_sampler is None) # Shuffle only if not using DDP
        )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    # Initialize wandb for logging
    wandb_run = None
    if not args.get('debug', False) and not args.get('only_eval', False) and rank == 0:
        wandb_run = wandb.init(
            project=args.get('project', 'multicoco'),
            name=args.get('name', 'default-run'),
            config=args
        )
        text_table = wandb.Table(columns=["step", "text"])
    else:
        text_table = None

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        wandb_run=wandb_run,
        text_table=text_table
    )

    # Start training or evaluation
    if args.get('only_eval', False):
        print("--- Starting Evaluation Only ---")
        val_acc = trainer.evaluate()
        if rank == 0:
            print(f"Final Validation Accuracy: {val_acc:.4f}")
    else:
        trainer.train()

    if world_size > 1:
        cleanup()

if __name__ == "__main__":
    main()
