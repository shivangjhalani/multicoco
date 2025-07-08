#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import yaml
from PIL import Image
from torch.utils.data import DataLoader


# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import TrainingArguments, AutoImageProcessor, AutoTokenizer
from multicoco.data import SupervisedDataset, collate_fn
from multicoco.model import MultiCoCo
from multicoco.trainer import CoCoTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_collate_function(tokenizer, image_processor):
    """Create a collate function with the tokenizer and image processor."""
    def collate_wrapper(batch):
        return collate_fn(batch, tokenizer, image_processor)
    return collate_wrapper


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='Path to the configuration file')
    config_file = parser.parse_args().config_file
    
    # Load config
    args = load_config(config_file)
    logger.info(f"Loaded config from {config_file}")
    logger.info(f"Config: {args}")
    
    # Set up output directory
    output_dir = args.get('output_dir', './output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model components
    logger.info("Initializing model components...")
    
    # Load tokenizer and image processor
    model_name = args.get('model_name', 'OpenGVLab/InternVL3-1B-Pretrained')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # Add special tokens if doing CoCoNut training
    special_tokens = []
    if args.get('coconut', False):
        special_tokens = ['<|thought|>', '<|start_thought|>', '<|end_thought|>']
        tokenizer.add_tokens(special_tokens)
    
    # Initialize model
    model = MultiCoCo(
        model_id=model_name,
        special_tokens=special_tokens
    )
    
    # Resize token embeddings if we added special tokens
    if special_tokens:
        model.model.resize_token_embeddings(len(tokenizer))
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create training arguments
    # Conditionally disable wandb for evaluation only
    eval_only = args.get('eval_only', False)
    report_to = None if eval_only else "wandb"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.get('num_epochs', 3),
        per_device_train_batch_size=args.get('batch_size', 2),
        per_device_eval_batch_size=args.get('eval_batch_size', 2),
        gradient_accumulation_steps=args.get('gradient_accumulation_steps', 1),
        learning_rate=args.get('learning_rate', 1e-5),
        warmup_steps=args.get('warmup_steps', 500),
        logging_steps=args.get('logging_steps', 10),
        save_steps=args.get('save_steps', 500),
        eval_steps=args.get('eval_steps', 500),
        eval_strategy=args.get('eval_strategy', 'steps'),
        save_strategy=args.get('save_strategy', 'steps'),
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        bf16=True if torch.cuda.is_available() else False,
        report_to=report_to,  # Enable wandb for training, disable for eval
    )
    
    # Add custom args to training_args that our custom trainer needs
    training_args.log_dir = args.get('log_dir', 'logs')
    training_args.eval_config = {'coconut': args.get('coconut', False), 'cot': args.get('cot', False)}
    
    # Add CoCoNut specific parameters to training_args
    training_args.c_thought = args.get('c_thought', 0)
    training_args.max_latent_stage = args.get('max_latent_stage', 0)
    
    # Add special token IDs if we're using CoCoNut or CoT
    if args.get('coconut', False) or args.get('cot', False):
        training_args.thought_token_id = tokenizer.convert_tokens_to_ids('<|thought|>') if '<|thought|>' in tokenizer.get_vocab() else None
        training_args.start_thought_id = tokenizer.convert_tokens_to_ids('<|start_thought|>') if '<|start_thought|>' in tokenizer.get_vocab() else None
        training_args.end_thought_id = tokenizer.convert_tokens_to_ids('<|end_thought|>') if '<|end_thought|>' in tokenizer.get_vocab() else None
    
    # Load dataset
    logger.info("Loading dataset...")
    
    train_dataset = None
    eval_dataset = None
    
    if 'train_data_path' in args:
        train_dataset = SupervisedDataset(
            data_path=args['train_data_path'],
            data_dir=args.get('data_dir', '')
        )
        logger.info(f"Loaded {len(train_dataset)} training samples")
    
    if 'val_data_path' in args:
        eval_dataset = SupervisedDataset(
            data_path=args['val_data_path'],
            data_dir=args.get('data_dir', '')
        )
        logger.info(f"Loaded {len(eval_dataset)} validation samples")
    
    # Create data collator
    data_collator = create_collate_function(tokenizer, image_processor)
    
    # Initialize trainer
    trainer = CoCoTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Check if we should train or just evaluate
    if args.get('eval_only', False):
        logger.info("--- Starting Evaluation Only ---")
        if eval_dataset is None:
            logger.error("No evaluation dataset provided!")
            return
        
        metrics = trainer.evaluate()
        logger.info(f"Evaluation metrics: {metrics}")
    else:
        logger.info("--- Starting Training ---")
        if train_dataset is None:
            logger.error("No training dataset provided!")
            return
        
        # Start training
        trainer.train()
        
        # Save final model
        trainer.save_model()
        logger.info(f"Training completed. Model saved to {output_dir}")
        
        # Final evaluation if eval dataset provided
        if eval_dataset is not None:
            logger.info("--- Starting Final Evaluation ---")
            metrics = trainer.evaluate()
            logger.info(f"Final evaluation metrics: {metrics}")


if __name__ == "__main__":
    main()
