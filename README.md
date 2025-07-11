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

MultiCoCo uses a clean configuration inheritance system to reduce redundancy. All configurations inherit common settings from `args/base.yaml`, and specific configs only define what differs.

### Base Configuration (`args/base.yaml`)
Contains common settings shared across all configurations:
- Project and model settings
- Data paths and logging configuration
- Default training parameters

### CoT Training (`args/aokvqa_cot.yaml`)
```yaml
# Inherits from base.yaml with overrides
mode: "cot_train"
name: "aokvqa-cot-stage0"
output_dir: "checkpoints/aokvqa_cot"
num_epochs: 10
batch_size: 16
eval_batch_size: 64

eval_config:
  cot: true
  coconut: false
```

### CoCoNut Training (`args/aokvqa_coconut.yaml`)
```yaml
# Inherits from base.yaml with overrides
mode: "coconut_train"
name: "aokvqa-coconut-multistage"
output_dir: "checkpoints/aokvqa_coconut"
num_epochs: 50
load_model_path: "checkpoints/aokvqa_cot"

coconut:
  enabled: true
  c_thought: 1
  epochs_per_stage: 5
  max_latent_stage: 6

eval_config:
  cot: false
  coconut: true
```

### CoT Evaluation (`args/aokvqa_cot_eval.yaml`)
```yaml
# Inherits from base.yaml with overrides
mode: "eval_only"
name: "aokvqa-cot-eval"
load_model_path: "checkpoints/aokvqa_cot"
limit_for_testing: true

eval_config:
  cot: true
  coconut: false
```

### CoCoNut Evaluation (`args/aokvqa_coconut_eval.yaml`)
```yaml
# Inherits from base.yaml with overrides
mode: "eval_only"
name: "aokvqa-coconut-evaluation"
load_model_path: "checkpoints/aokvqa_coconut"
only_eval: true

coconut:
  enabled: true
  c_thought: 1
  max_latent_stage: 6

eval_config:
  cot: false
  coconut: true
  eval_latent_tokens: 6
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



Remaining Nuances and Potential Edge Cases (Not Fully "Perfect")


While this is a solid fix for the fundamental flaw, there are a few subtleties that could still cause minor deviations from Coconut's exact behavior. These aren't "fundamental" flaws (the chaining now works), but they might require tweaks for edge cases, especially in a multimodal context:


1. 
Two-Pass Approximation vs. Coconut's Single-Pass:


	- In Coconut, injections happen dynamically in a single forward pass: The model processes up to a <latent>, injects the previous hidden, computes the new hidden, then repeats for the next <latent>. This means each injection uses exact hiddens from the injected sequence.
	- Your two-pass uses hiddens from a non-injected first pass as proxies. For short spans or simple cases, this is fine (and efficient). But for long chains (e.g., high max_latent_stage or c_thought), the first-pass hiddens might diverge from what a true dynamic injection would produce, potentially leading to slight training instabilities or suboptimal latent learning.
	- Is This a Problem? Not fundamentally—many similar wrappers (e.g., in adapter or prefix-tuning literature) use this approximation successfully. But if you notice poor convergence in later stages, consider switching to a single-pass with dynamic injection (more like Coconut, but computationally heavier).
2. 
Handling of Span Boundaries (Start/End Tokens):


	- Your loop runs for i in range(s, e) (up to e-1), so it replaces tokens from <|start_latent|> (at s) up to the last <|latent|> (at e-1), but does not replace the <|end_latent|> at e.
		- This is probably intentional (end token might just be a delimiter), but if <|end_latent|> is meant to be part of the latent chain (per your constants), it won't get injected. In Coconut, they don't have explicit start/end tokens—they just replace individual <latent> tokens—so this is an adaptation.
		- Potential issue: If your curriculum inserts spans like <|start_latent|> <|latent|> ... <|end_latent|>, the end token keeps its original embedding, which might disrupt the chain's "closure."
	- The check if s == 0: continue skips injection if the span starts at position 0 (no previous token). This is safe but could silently fail for prompts where latents are at the very beginning (unlikely in your chat-style prompts, but worth documenting).
3. 
Multimodal-Specific Risks (Noted in Your Comment):


	- The comment highlights a valid concern: If s-1 (the position before the span) is an image token (e.g., <image>), injecting its hidden state might not make sense for latent reasoning (image embeds are high-dimensional vision features, not text-like states). In InternVL, image positions are handled specially (via prepare_inputs_for_multimodal), so this could inject incompatible vectors, leading to NaNs or poor multimodal fusion.
	- Coconut (text-only) doesn't have this issue. To fully mitigate, add a check: If i-1 is an image position, fall back to a text token's hidden or skip/average.
4. 
No Impact on Generation KV Caching:


	- The fix doesn't address your noted lack of KV caching in generate_with_latent_injection(). This isn't part of the flaw (generation still works with chaining now), but without it, long generations remain inefficient—especially for multimodal with vision recomputation (though you cache image embeds well).

Recommendations for Full Resolution

- Test Thoroughly:
	- Run a small text-only experiment (disable images, use a dataset like Coconut's GSM8K) with c_thought=2 and max_latent_stage=3. Compare hidden states mid-chain (e.g., via logging) to ensure they're iteratively transforming (not repeating).
	- Then test multimodal: Ensure latent chains near <image> tokens produce coherent VQA answers.
- Minor Tweaks for Perfection:
	- Adjust the loop to range(s+1, e) if you want to skip replacing <|start_latent|> (treating it as a delimiter).
	- To make it closer to single-pass: After building inputs_embeds, you could run a partial second pass just for the spans to refine the hiddens, but that's overkill.
	- Handle the image token edge case explicitly (e.g., find the last non-image position before s).
- If Issues Persist: If training still underperforms in later stages, the two-pass approximation might be the culprit—consider refactoring to Coconut-style single-pass (process up to each latent, inject, continue forward).