# MultiCoCo: Multimodal Chain-of-Continuous-Thought

MultiCoCo is a comprehensive training framework that extends the original [CoCoNut (Chain-of-Continuous-Thought)](https://arxiv.org/abs/2412.06769) methodology to multimodal models, specifically InternVL3-1B for Visual Question Answering.

## Overview

MultiCoCo implements the progressive curriculum learning approach from the original CoCoNut paper, enabling multimodal models to reason in continuous latent space through a multi-stage training process.

### Key Features

- **Progressive Curriculum Learning**: Stage-by-stage replacement of reasoning steps with latent tokens
- **Multimodal Support**: Extended CoCoNut methodology for vision-language models (InternVL3-1B)
- **Two-Phase Training**: Separate CoT training and CoCoNut multi-stage training commands
- **Distributed Training**: Full DDP support for multi-GPU training
- **Wandb Integration**: Comprehensive experiment tracking and logging
- **Evaluation Suite**: Support for A-OKVQA dataset evaluation

## Training Methodology

MultiCoCo follows the original CoCoNut methodology with two distinct phases:

### Phase 1: Chain-of-Thought (CoT) Training
```bash
torchrun --nnodes 1 --nproc_per_node 1 run.py args/aokvqa_cot.yaml
```

This phase trains the model with full reasoning chains:
```
Question: What is this animal?
Reasoning: I can see this is a four-legged animal with stripes. The black and white stripe pattern is characteristic of zebras.
Answer: Zebra
```

### Phase 2: CoCoNut Multi-Stage Training
```bash
torchrun --nnodes 1 --nproc_per_node 1 run.py args/aokvqa_coconut.yaml
```

This phase implements progressive curriculum learning:

- **Stage 1**: Replace 1st reasoning step with latent tokens
  ```
  Question: What is this animal?
  <latent> The black and white stripe pattern is characteristic of zebras.
  Answer: Zebra
  ```

- **Stage 2**: Replace 2nd reasoning step with additional latent tokens
  ```
  Question: What is this animal?
  <latent> <latent>
  Answer: Zebra
  ```

- **Stage N**: Complete latent reasoning
  ```
  Question: What is this animal?
  <latent> <latent> <latent>
  Answer: Zebra
  ```

## Progressive Curriculum Learning

The core innovation is the progressive replacement algorithm:

```python
# Stage calculation
scheduled_stage = epoch // epochs_per_stage

# Progressive replacement logic
n_skip_steps = scheduled_stage      # Skip this many reasoning steps
n_latent_tokens = scheduled_stage   # Replace with this many latent tokens

# Build sequence
tokens = (
    question + 
    [latent_token] * (n_latent_tokens * c_thought) +
    remaining_reasoning_steps +
    answer
)
```

### Configuration Parameters

- `c_thought`: Number of continuous thoughts per reasoning step (default: 1)
- `epochs_per_stage`: Number of epochs to train each stage (default: 5)
- `max_latent_stage`: Maximum number of latent stages (default: 6)
- `reset_optimizer`: Whether to reset optimizer between stages (default: true)
- `uniform_prob`: Probability to mix data from other stages (default: 0.0)
- `pad_latent_to_max`: Whether to pad latent tokens to max stage (default: false)

## Installation and Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Login to Wandb** (required for logging):
   ```bash
   wandb login
   ```

3. **Prepare Data**: Place A-OKVQA dataset in `data/` directory
   ```
   data/
   ├── aokvqa_train.json
   ├── aokvqa_validation.json
   └── [images]/
   ```

## Training Pipeline

### Step 1: CoT Training (Stage 0)
```bash
torchrun --nnodes 1 --nproc_per_node 1 run.py args/aokvqa_cot.yaml
```

**Purpose**: Train the model with full chain-of-thought reasoning to establish baseline reasoning capabilities.

**Expected Output**: A CoT-trained model saved in `checkpoints/aokvqa_cot/`

### Step 2: CoCoNut Multi-Stage Training
```bash
torchrun --nnodes 1 --nproc_per_node 1 run.py args/aokvqa_coconut.yaml
```

**Purpose**: Progressive curriculum learning that gradually replaces reasoning steps with latent tokens.

**Configuration**: Update `load_model_path` in the config to point to the best CoT checkpoint.

**Training Process**:
- Automatically calculates training stages based on `epochs_per_stage`
- Applies progressive curriculum to dataset for each stage
- Optionally resets optimizer between stages
- Saves checkpoints after each epoch

### Step 3: Evaluation
```bash
torchrun --nnodes 1 --nproc_per_node 1 run.py args/aokvqa_coconut_eval.yaml
```

**Purpose**: Evaluate the trained CoCoNut model on the validation set.

## Configuration Files

### CoT Training (`args/aokvqa_cot.yaml`)
```yaml
# Stage 0: Chain-of-Thought Training
mode: "cot_train"
model_name: "OpenGVLab/InternVL3-1B-Pretrained"
output_dir: "checkpoints/aokvqa_cot"
num_epochs: 10
eval_config:
  cot: true
  coconut: false
```

### CoCoNut Training (`args/aokvqa_coconut.yaml`)
```yaml
# CoCoNut Multi-Stage Training
mode: "coconut_train"
load_model_path: "checkpoints/aokvqa_cot"  # CoT checkpoint
output_dir: "checkpoints/aokvqa_coconut"
num_epochs: 50

coconut:
  enabled: true
  c_thought: 1
  epochs_per_stage: 5
  max_latent_stage: 6
  reset_optimizer: true

eval_config:
  cot: false
  coconut: true
```

### Evaluation (`args/aokvqa_coconut_eval.yaml`)
```yaml
# CoCoNut Evaluation Only
mode: "eval_only"
load_model_path: "checkpoints/aokvqa_coconut"  # Best CoCoNut checkpoint
only_eval: true

eval_config:
  cot: false
  coconut: true
```

## Wandb Logging

MultiCoCo provides comprehensive experiment tracking:

### Logged Metrics
- **Training**: Loss, stage information, learning rate, epoch progress
- **Validation**: Accuracy, loss, stage-specific metrics
- **Stage Tracking**: Current stage, stage epoch, progressive curriculum status
- **System**: Training time, memory usage, checkpoint information

### Configuration
```yaml
project: "multicoco"
name: "aokvqa-coconut-experiment"
debug: false  # Set to true to disable wandb logging
```

## Model Architecture

**Base Model**: InternVL3-1B-Pretrained
- **Vision**: InternViT-300M-448px-V2_5
- **Language**: Qwen2.5-0.5B
- **Total Parameters**: ~896M
- **Architecture**: ViT-MLP-LLM with dynamic patch integration

**Latent Space Enhancement**: Added `<latent>` tokens to vocabulary for continuous thought representation.

## Key Differences from Original CoCoNut

| Aspect | Original CoCoNut | MultiCoCo |
|--------|------------------|-----------|
| **Modality** | Text-only (GPT-2, Llama) | Multimodal (InternVL3-1B) |
| **Dataset** | GSM8K, ProntoQA, ProsQA | A-OKVQA (Visual QA) |
| **Input** | Text questions | Image + Text questions |
| **Model Size** | Up to 7B parameters | 896M parameters |
| **Training** | Text-based reasoning | Visual reasoning with image understanding |

## Data Format

Expected JSON format for training data:
```json
[
  {
    "image": "path/to/image.jpg",
    "question": "What is this object?",
    "steps": [
      "I can see this is a metallic object.",
      "It has a cylindrical shape with a spout.",
      "This appears to be a teapot based on its design."
    ],
    "answer": "Teapot"
  }
]
```

## Evaluation Metrics

- **Accuracy**: Exact match accuracy for answer extraction
- **Stage Performance**: Performance tracking across different latent stages
- **Latent Efficiency**: Reasoning capability with reduced explicit steps

## Troubleshooting

### Common Issues

1. **CUDA Memory Issues**
   ```yaml
   batch_size: 4  # Reduce batch size
   gradient_accumulation_steps: 4  # Increase accumulation
   ```

2. **Missing `<latent>` Token**
   - Automatically added during CoCoNut training initialization
   - Check tokenizer vocabulary size in logs

3. **Checkpoint Loading Issues**
   - Ensure `load_model_path` points to valid checkpoint directory
   - Check that CoT training completed successfully

4. **Wandb Connection Issues**
   ```bash
   wandb offline  # For offline mode
   # or set debug: true in config
   ```

## Performance Expectations

Based on the original CoCoNut methodology:
- **CoT Training**: Establishes baseline reasoning (~40% accuracy on GSM8K equivalent)
- **CoCoNut Training**: Progressive improvement through latent reasoning
- **Final Performance**: Comparable or improved accuracy with latent space reasoning

## Contributing

This implementation follows the original CoCoNut paper methodology:
- [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)
- Original CoCoNut codebase: [GitHub](https://github.com/facebookresearch/coconut)

## Requirements

See `requirements.txt` for complete dependencies:
- PyTorch
- Transformers
- InternVL models
- Wandb
- PIL/Pillow
- tqdm
- accelerate

## License

This project builds upon the original CoCoNut methodology and InternVL models. Please refer to their respective licenses for usage terms. 