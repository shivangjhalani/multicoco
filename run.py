import argparse
import yaml
import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from multicoco.data import MultiCoCoDataset, DataCollatorForMultiCoCo
from multicoco.model import MultiCoCo
from multicoco.trainer import Trainer

def setup(rank, world_size):
    """Initializes the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
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
    model = MultiCoCo(args).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    tokenizer_path = model.module.tokenizer.name_or_path if hasattr(model, 'module') else model.tokenizer.name_or_path

    # Data
    train_dataset = MultiCoCoDataset(data_path=args['train_path'], data_dir=args['data_dir'])
    val_dataset = MultiCoCoDataset(data_path=args['val_path'], data_dir=args['data_dir'])
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    
    collator = DataCollatorForMultiCoCo(tokenizer_path=tokenizer_path)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args['batch_size_training'], 
        sampler=train_sampler, 
        collate_fn=collator,
        shuffle=(train_sampler is None) # Shuffle only if not using DDP
    )
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size_training'], collate_fn=collator)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args
    )

    # Start training
    trainer.train()

    if world_size > 1:
        cleanup()

if __name__ == "__main__":
    main()
