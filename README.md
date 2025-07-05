# MultiCoCo

MultiCoCo training framework with wandb logging support.

## Features

- Staged training with CoT and Coconut models
- Distributed training with DDP support
- Weights & Biases (wandb) logging for experiment tracking
- Evaluation on A-OKVQA dataset

## Wandb Logging

The framework now includes comprehensive wandb logging similar to the coconut project:

### Logged Metrics

- **Training metrics**: Loss, stage, epoch, and step information
- **Validation metrics**: Accuracy and loss after each epoch
- **Training data**: Sample input/output pairs for debugging (first step only)

### Configuration

Wandb logging is controlled by the following config parameters:

- `project`: wandb project name (default: "multicoco")
- `name`: experiment name for the run
- `debug`: Set to `true` to disable wandb logging
- `only_eval`: Set to `true` to disable wandb logging during evaluation-only runs

### Example Config

```yaml
project: multicoco
name: aokvqa-coconut-experiment
debug: false
only_eval: false
# ... other training parameters
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Login to wandb (required for logging):
```bash
wandb login
```

3. Run training:
```bash
torchrun --nnodes 1 --nproc_per_node 8 run.py args/aokvqa_cot.yaml
```

## Training Pipeline

1. **Stage 0 (CoT)**: Train with chain-of-thought reasoning
2. **Stage 1-N (Coconut)**: Progressive latent space training

Wandb will automatically log metrics for each stage and epoch, making it easy to track training progress and compare experiments.

## Requirements

- PyTorch
- Transformers
- wandb
- PIL
- tqdm
- accelerate
- datasets

See `requirements.txt` for complete dependencies. 