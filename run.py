# Set environment variable to disable Flash Attention 2 before importing transformers
import os
os.environ["TRANSFORMERS_NO_FLASH_ATTENTION_2"] = "1"

import sys
from dataclasses import dataclass, field
from typing import Optional
import torch

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoProcessor,
)

from multicoco.data import SupervisedDataset, DataCollatorForCoCo
from multicoco.model import MultiCoCo
from multicoco.trainer import CoCoTrainer

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to the model checkpoint."})
    load_model_path: Optional[str] = field(default=None, metadata={"help": "Path to a specific model checkpoint to load."})

@dataclass
class DataArguments:
    data_dir: str = field(default="data/", metadata={"help": "The directory where the data is stored."})
    train_data_path: Optional[str] = field(default=None, metadata={"help": "Path to the training data."})
    eval_data_path: Optional[str] = field(default=None, metadata={"help": "Path to the evaluation data."})
    cot: bool = field(default=False, metadata={"help": "Whether to use Chain-of-Thought prompting."})
    coconut: bool = field(default=False, metadata={"help": "Whether to use Coconut-style prompting."})


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_hacls()

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )

    model = MultiCoCo(model_args.model_name_or_path)
    model.to(torch.bfloat16)

    train_dataset = SupervisedDataset(
        data_args,
        processor=processor
    )
    eval_dataset = SupervisedDataset(
        data_args,
        processor=processor,
        is_eval=True
    )

    data_collator = DataCollatorForCoCo(processor=processor, data_args=data_args)

    # Add custom args to training_args that our custom trainer needs
    training_args.eval_config = {'coconut': data_args.coconut, 'cot': data_args.cot}

    trainer = CoCoTrainer(
        model=model,
        processor=processor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(output_dir=training_args.output_dir)
    elif training_args.do_eval:
        trainer.evaluate()


if __name__ == "__main__":
    main()
