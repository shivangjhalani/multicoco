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
    
    # -- DDP Setup
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1
    if is_ddp:
        rank = int(os.environ["LOCAL_RANK"])
        setup(rank, world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Initialize Tokenizer and Model
    load_path = args.get('load_model_path')
    model_id = args['model_id']
    is_eval_only = args.get('only_eval', False)
    
    special_tokens = []
    # Add special tokens if we are training, or if we are evaluating a model that used them (cot/coconut)
    if not is_eval_only or args.get('cot') or args.get('coconut'):
        special_tokens = ['<thought>', '<start_thought>', '<end_thought>']

    # If load_path is a checkpoint file, we must load the base model first
    if load_path and os.path.isfile(load_path):
        print(f"Initializing from base model '{model_id}' to load checkpoint '{load_path}'")
        model = MultiCoCo(model_id, special_tokens=special_tokens).to(device)
        print(f"Loading checkpoint weights from file: {load_path}")
        checkpoint = torch.load(load_path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        # If load_path is a directory or None, use it or model_id as the primary source
        primary_path = load_path if load_path else model_id
        print(f"Initializing model from '{primary_path}'")
        model = MultiCoCo(primary_path, special_tokens=special_tokens).to(device)

    tokenizer = model.tokenizer
    
    if not is_eval_only:
        # Add special tokens to args to be accessible in the trainer for training
        args['thought_token_id'] = tokenizer.convert_tokens_to_ids('<thought>')
        args['start_thought_id'] = tokenizer.convert_tokens_to_ids('<start_thought>')
        args['end_thought_id'] = tokenizer.convert_tokens_to_ids('<end_thought>')

    # -- DDP Model
    if is_ddp:
        model = DDP(model, device_ids=[rank])
    
    # -- Collator
    # The collator needs access to the model and tokenizer, which might be wrapped in DDP
    hf_model = (model.module if hasattr(model, 'module') else model).model
    image_processor = model.image_processor if not hasattr(model, 'module') else model.module.image_processor
    collator = DataCollatorForInternVL(
        tokenizer=tokenizer,
        model=hf_model,
        image_processor=image_processor
    )

    # -- DataLoaders
    train_dataset = MultiCoCoDataset(
        data_path=args['train_path'] if not is_eval_only else None,
        data_dir=args['data_dir']
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if is_ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args['batch_size_training'],
        sampler=train_sampler,
        collate_fn=collator,
        shuffle=(train_sampler is None) # Shuffle only if not using DDP
    )

    # Always create val_loader
    val_dataset = MultiCoCoDataset(data_path=args['val_path'], data_dir=args['data_dir'])
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank) if is_ddp else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.get('batch_size_evaluation', 1),
        sampler=val_sampler,
        collate_fn=collator
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
        optimizer=None if is_eval_only else optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        wandb_run=wandb_run if rank == 0 else None,
        text_table=text_table if rank == 0 else None
    )

    # Start training or evaluation
    if is_eval_only:
        print("--- Starting Evaluation Only ---")
        val_acc = trainer.evaluate()
        if rank == 0:
            print(f"Final Validation Accuracy: {val_acc:.4f}")
    else:
        trainer.train()

    # -- Cleanup
    if is_ddp:
        cleanup()

if __name__ == "__main__":
    main()
