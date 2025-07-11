# WandB Integration in MultiCoCo

This document explains the comprehensive WandB logging integration in MultiCoCo, implemented following the manual approach used in the coconut codebase.

## Overview

MultiCoCo now includes comprehensive experiment tracking with Weights & Biases (WandB) for:
- Hyperparameter logging and comparison
- Training metrics and loss tracking
- Evaluation results with sample visualizations
- CoCoNut stage-specific progressive metrics
- Model and dataset artifact management
- System metrics monitoring (GPU usage, memory)

## Configuration

### Base Configuration

WandB settings are configured in YAML files under the `wandb_*` keys:

```yaml
# WandB Configuration
wandb_project: "multicoco-research"
wandb_entity: null  # Set to your wandb team/entity if applicable
wandb_tags: ["cot", "aokvqa", "baseline"]  # Tags for organizing runs
wandb_group: "aokvqa-cot-training"  # Group related runs together
wandb_notes: "Chain-of-thought training baseline for AOK-VQA dataset"
```

### Per-Experiment Configuration

Each experiment config inherits from `base.yaml` and customizes WandB settings:

- **CoT Training** (`aokvqa_cot.yaml`): Tagged with `["cot", "aokvqa", "baseline"]`
- **CoCoNut Training** (`aokvqa_coconut.yaml`): Tagged with `["coconut", "aokvqa", "latent-reasoning", "multimodal"]`
- **Evaluation** (`*_eval.yaml`): Tagged with `["evaluation"]`

## Implementation Details

### Manual WandB Integration

Following the coconut pattern, MultiCoCo uses **manual WandB initialization** instead of HuggingFace's automatic integration:

```python
# In MultiCoCoRunner.__init__()
wandb_run = wandb.init(
    project=log_config.wandb_project,
    entity=log_config.wandb_entity,
    name=training_config.name,
    group=log_config.wandb_group,
    tags=log_config.wandb_tags,
    notes=log_config.wandb_notes,
    config=self.config.to_dict(),
    settings=wandb.Settings(
        _stats_sample_rate_seconds=10  # System metrics every 10 seconds
    )
)
```

### Logging Components

#### 1. Training Metrics

```python
# Automatic logging in trainer.log()
wandb.log({
    "train/epoch_loss": avg_loss,
    "train/epoch": epoch + 1,
    "train/total_steps": self.total_train_steps
})
```

#### 2. Evaluation Results with Sample Tables

```python
# In evaluation_loop()
sample_table = wandb.Table(columns=["Question", "Ground Truth", "Prediction", "Correct"])
for i in range(min(50, len(predictions))):
    correct = predictions[i].strip() == labels[i].strip()
    sample_table.add_data(questions[i][:200], labels[i], predictions[i], correct)

wandb.log({"eval/samples": copy(sample_table)})
```

#### 3. CoCoNut Progressive Training

```python
# Stage information
wandb.log({
    "coconut/stage": stage,
    "coconut/latent_tokens": stage * c_thought,
    "coconut/max_latent_stage": max_latent_stage,
    "coconut/dataset_size": dataset_size
})

# Stage-specific evaluation metrics
wandb.log({
    "coconut/stage_epoch": stage_epoch + 1,
    "coconut/current_stage": current_stage,
    "coconut/eval_accuracy": accuracy
})
```

#### 4. Artifacts for Reproducibility

```python
# Model checkpoints
artifact = wandb.Artifact(f"checkpoint-epoch-{epoch}", type="model")
artifact.add_dir(checkpoint_dir)
wandb.log_artifact(artifact)

# Datasets
train_artifact = wandb.Artifact("train_dataset", type="dataset")
train_artifact.add_file(train_data_path)
wandb.log_artifact(train_artifact)
```

#### 5. Model and System Information

```python
# Model metadata
wandb.log({
    "model/source": source_info,
    "model/dtype": torch_dtype,
    "model/total_parameters": total_params,
    "training/mode": training_mode,
    "coconut/enabled": coconut_enabled
})
```

## WandB Dashboard Features

Your WandB dashboard will include:

### Panels and Visualizations

1. **Metrics Trends**
   - Training loss over epochs/steps
   - Evaluation accuracy progression
   - CoCoNut stage accuracy comparison

2. **Sample Tables**
   - Interactive Q&A samples with correctness
   - Image visualization (for multimodal samples)
   - Filtering by correctness/stage

3. **Hyperparameter Parallel Coordinates**
   - Compare runs across different configurations
   - Identify optimal hyperparameter combinations

4. **System Metrics**
   - GPU memory usage during training
   - CPU utilization
   - Real-time system monitoring

5. **CoCoNut-Specific Plots**
   - Latent token progression across stages
   - Compression ratio analysis
   - Stage-wise accuracy improvements

### Artifacts

- **Models**: Checkpoints for each epoch
- **Datasets**: Training and evaluation data files
- **Generation Examples**: Text samples from each stage

## Usage Examples

### Basic Training with WandB

```bash
# Training with WandB enabled (default)
torchrun --nnodes 1 --nproc_per_node 1 run.py args/aokvqa_cot.yaml

# Disable WandB
# Edit the config file to set: use_wandb: false
```

### Custom WandB Configuration

```yaml
# In your config file
use_wandb: true
wandb_project: "my-research-project"
wandb_entity: "my-team"
wandb_tags: ["experiment-1", "high-lr"]
wandb_group: "ablation-studies"
wandb_notes: "Testing higher learning rates for CoCoNut training"
```

### Using WandB Sweeps

Create a `sweep.yaml` file:

```yaml
program: run.py
method: grid
metric:
  name: eval/accuracy
  goal: maximize
parameters:
  generation.temperature:
    values: [0.7, 0.8, 1.0]
  coconut.epochs_per_stage:
    values: [3, 5]
```

Run the sweep:

```bash
wandb sweep sweep.yaml
wandb agent <sweep_id>
```

## Utility Functions

MultiCoCo includes utility functions in `wandb_utils.py`:

```python
from multicoco.wandb_utils import (
    log_wandb_samples,
    log_coconut_stage_metrics,
    log_model_parameters,
    finish_wandb_run
)

# Log custom samples
log_wandb_samples(questions, labels, predictions, images=images)

# Log CoCoNut metrics
log_coconut_stage_metrics(stage=2, latent_tokens=6, dataset_size=1000, accuracy=0.85)

# Log model info
log_model_parameters(model)
```

## Best Practices

### Run Organization

1. **Use Groups**: Group related experiments (e.g., "aokvqa-coconut-ablation")
2. **Tag Systematically**: Use consistent tags like `["coconut", "cot", "baseline", "multimodal"]`
3. **Descriptive Names**: Use clear run names like `"aokvqa-coconut-6stages-lr1e4"`

### Performance

1. **Sample Limits**: Limit sample tables to 50 samples to avoid memory issues
2. **Artifact Size**: Be mindful of large checkpoint artifacts
3. **System Metrics**: Enable system monitoring for resource analysis

### Privacy

1. **Sensitive Data**: Anonymize any sensitive information in samples
2. **Entity Settings**: Use appropriate wandb_entity for team/organization

## Troubleshooting

### Common Issues

1. **No Logs Appearing**
   - Check `use_wandb: true` in config
   - Verify `wandb login` authentication
   - Check network connectivity

2. **Artifact Upload Failures**
   - Verify file paths exist
   - Check disk space and upload limits
   - Use artifact filters for large directories

3. **Multimodal Image Issues**
   - Ensure PIL images are properly formatted
   - Check image file sizes
   - Verify image paths are accessible

### Debug Mode

Disable WandB for quick debugging:

```yaml
debug: true  # This will automatically disable wandb
# OR
use_wandb: false
```

## Comparison with Coconut

MultiCoCo's WandB integration follows the same manual approach as coconut but adds:

- **Multimodal Support**: Image logging in sample tables
- **Progressive Curriculum**: CoCoNut stage-specific metrics
- **Enhanced Artifacts**: Automatic model and dataset versioning
- **Comprehensive Config**: Hierarchical YAML configuration
- **Utility Functions**: Reusable logging helpers

This provides far more insights than coconut's basic epoch logging while maintaining the same manual control philosophy. 