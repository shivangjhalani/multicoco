# MultiCoCo Wandb Metrics Enhancement

This document describes the comprehensive wandb logging metrics that have been added to the MultiCoCo project to match and extend the logging capabilities found in the CoCoNut project.

## New Wandb Metrics Added

### 1. Training Metrics (Similar to CoCoNut)
- `train/batch_loss`: Per-batch training loss
- `train/step`: Training step counter
- `train/epoch`: Current epoch number
- `train/global_step`: Global step counter across all epochs
- `train/learning_rate`: Current learning rate
- `train/grad_norm`: Gradient norm (when gradient clipping is enabled)
- `train/gradient_accumulation_steps`: Gradient accumulation configuration
- `train/epoch_avg_loss`: Average loss per epoch
- `train/steps_per_epoch`: Number of steps per epoch

### 2. Evaluation Metrics (Enhanced from CoCoNut)
- `eval/acc`: Accuracy score (matching CoCoNut's eval/acc)
- `eval/cot_em`: Chain-of-Thought exact match (matching CoCoNut's eval/cot_em)
- `eval/loss`: Validation loss
- `eval/total_samples`: Total number of evaluation samples
- `eval/correct_predictions`: Number of correct predictions
- `eval/sample_generations`: Table of sample generations for qualitative analysis

### 3. CoCoNut-Specific Stage Metrics (New)
- `coconut/current_stage`: Current latent replacement stage
- `coconut/stage_epoch`: Epoch within the current stage
- `coconut/stage_progress`: Progress within the current stage (0.0 to 1.0)
- `coconut/latent_replacement_ratio`: Ratio of latent tokens to total stages
- `coconut/max_latent_stage`: Maximum number of latent stages
- `coconut/epochs_per_stage`: Epochs per stage configuration
- `coconut/c_thought`: Number of latent tokens per reasoning step
- `coconut/total_stages`: Total number of stages in the curriculum
- `coconut/uniform_prob`: Uniform probability configuration
- `coconut/stage_transition`: Stage transition events
- `coconut/epoch_in_stage`: Current epoch within stage
- `coconut/latent_tokens_count`: Current number of latent tokens

### 4. Best Model Tracking (Similar to CoCoNut)
- `best/accuracy`: Best accuracy achieved
- `best/epoch`: Epoch where best accuracy was achieved
- `best/checkpoint`: Path to best model checkpoint
- `best/{metric}`: Best value for each evaluation metric

### 5. Epoch Summary Metrics
- `epoch/number`: Epoch number
- `epoch/time_seconds`: Time taken for the epoch
- `epoch/checkpoint_dir`: Checkpoint directory path
- `epoch/{metric}`: Epoch-specific evaluation metrics

### 6. Training Summary (Final)
- `summary/best_accuracy`: Final best accuracy
- `summary/best_epoch`: Final best epoch
- `summary/total_train_steps`: Total training steps completed
- `summary/best_checkpoint`: Final best checkpoint path

### 7. Data Inspection Tables (Similar to CoCoNut)
- `train/data_samples`: Table showing tokenized training data samples
- `eval/sample_generations`: Table showing evaluation sample generations

## Enhanced Features Beyond CoCoNut

### 1. Comprehensive Configuration Logging
- Complete model configuration details
- Training hyperparameters
- CoCoNut-specific settings

### 2. Enhanced Tagging System
- Automatic tagging based on training mode (cot, coconut)
- Stage-specific tags for CoCoNut training
- Model size and configuration tags

### 3. Qualitative Analysis Support
- Generation sample tables with ground truth comparison
- Token-level data inspection
- Stage progression visualization

### 4. Distributed Training Support
- Proper metric aggregation across multiple GPUs
- World process zero logging to avoid duplicates
- Gradient norm tracking across distributed setup

## Usage

The enhanced wandb logging is automatically enabled when:
1. `use_wandb: true` is set in the configuration
2. The wandb package is installed
3. The process is running on rank 0 (for distributed training)

All metrics are logged automatically during training and evaluation phases, providing comprehensive experiment tracking similar to the original CoCoNut implementation while adding MultiCoCo-specific enhancements.

## Configuration

The logging behavior can be controlled through the configuration:

```yaml
logging:
  use_wandb: true
  project: "multicoco"
  run_name: "experiment_name"
  log_to_file: true
  verbose: false
```

## Comparison with CoCoNut

| CoCoNut Metric | MultiCoCo Equivalent | Enhancement |
|---|---|---|
| `train/loss` | `train/batch_loss` | ✅ Added epoch and step info |
| `train/step` | `train/step` | ✅ Enhanced with global_step |
| `train/epoch` | `train/epoch` | ✅ Direct match |
| `eval/acc` | `eval/acc` | ✅ Direct match |
| `eval/cot_em` | `eval/cot_em` | ✅ Enhanced calculation |
| `eval/loss` | `eval/loss` | ✅ Direct match |
| `data_table` | `train/data_samples` | ✅ Enhanced structure |
| - | `coconut/*` | ✅ New stage tracking |
| - | `best/*` | ✅ Enhanced best tracking |
| - | `epoch/*` | ✅ New epoch summaries |

This implementation provides full feature parity with CoCoNut's wandb logging while adding significant enhancements for MultiCoCo's specific multimodal and multi-stage training requirements.
