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

from multicoco.data import SupervisedDataset, DataCollatorForCoCo
from multicoco.model import MultiCoCo
from multicoco.trainer import CoCoTrainer
from transformers import TrainingArguments

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

    unwrapped_model = model.module if hasattr(model, 'module') else model
    tokenizer = unwrapped_model.tokenizer
    image_processor = unwrapped_model.image_processor
    
    if not is_eval_only:
        # Add special tokens to args to be accessible in the trainer for training
        args['thought_token_id'] = tokenizer.convert_tokens_to_ids('<thought>')
        args['start_thought_id'] = tokenizer.convert_tokens_to_ids('<start_thought>')
        args['end_thought_id'] = tokenizer.convert_tokens_to_ids('<end_thought>')

    # -- DDP Model
    if is_ddp:
        model = DDP(model, device_ids=[rank])
    
    # -- Collator
    collator = DataCollatorForCoCo(
        tokenizer=tokenizer,
        image_processor=image_processor,
        cot=args.get('cot', False)
    )

    # -- DataLoaders
    train_loader = None
    if not is_eval_only:
        train_dataset = SupervisedDataset(
            data_path=args['train_path'],
            data_dir=args['data_dir']
        )
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args['batch_size_training'],
            sampler=train_sampler,
            collate_fn=collator,
            shuffle=(train_sampler is None) # Shuffle only if not using DDP
        )

    # Always create val_loader
    val_dataset = SupervisedDataset(
        data_path=args['val_path'],
        data_dir=args['data_dir']
    )
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
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

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=args.get('save_path', './results'),
        num_train_epochs=args.get('epochs_per_stage', 1),
        per_device_train_batch_size=args.get('batch_size_training', 1),
        per_device_eval_batch_size=args.get('batch_size_evaluation', 1),
        gradient_accumulation_steps=args.get('gradient_accumulation_steps', 1),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.get('lr', 5e-5),
        weight_decay=args.get('weight_decay', 0.01),
        warmup_steps=args.get('warmup_steps', 500),
        logging_dir=args.get('log_dir', './logs'),
        logging_steps=10,
        do_train=not is_eval_only,
        do_eval=True,
        bf16=args.get('bf16', False),
        report_to="wandb" if wandb_run else "none",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        remove_unused_columns=False, # We need to keep original_questions and answers
        label_names=["labels"], # Explicitly tell the trainer what the labels are
        deepspeed=args.get('deepspeed_config')
    )
    
    # Add custom args to training_args that our custom trainer needs
    training_args.log_dir = args.get('log_dir', 'logs')
    training_args.eval_config = {'coconut': args.get('coconut', False), 'cot': args.get('cot', False)}
    
    # Add CoCoNut specific parameters to training_args
    training_args.c_thought = args.get('c_thought', 0)
    training_args.max_latent_stage = args.get('max_latent_stage', 0)
    
    # Add special token IDs if we're using CoCoNut or CoT
    if args.get('coconut', False) or args.get('cot', False):
        training_args.thought_token_id = args.get('thought_token_id')
        training_args.start_thought_id = args.get('start_thought_id') 
        training_args.end_thought_id = args.get('end_thought_id')


    # Trainer
    trainer = CoCoTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if not is_eval_only else None,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collator
    )

    # Start training or evaluation
    if is_eval_only:
        print("--- Starting Evaluation Only ---")
        metrics = trainer.evaluate()
        if rank == 0:
            print(f"Final Validation Metrics: {metrics}")
    else:
        trainer.train()

    # -- Cleanup
    if is_ddp:
        cleanup()

if __name__ == "__main__":
    main()
