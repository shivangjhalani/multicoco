# WandB Integration for MultiCoCo

This document summarizes the comprehensive Weights & Biases (WandB) integration that has been implemented in the MultiCoCo codebase for advanced experiment tracking and research insights.

## Overview

The WandB integration provides comprehensive experiment tracking with:
- **Hyperparameter logging**: Complete configuration tracking
- **Metric visualization**: Training/validation accuracy, loss trends  
- **Sample tables**: Q&A predictions with correctness analysis
- **Artifacts**: Model checkpoints and datasets for reproducibility
- **CoCoNut-specific tracking**: Stage progression and latent token metrics
- **System monitoring**: GPU usage and performance metrics

## Features Implemented

### 1. Configuration-Based WandB Control

**Files Modified:**
- `args/base.yaml` - Added WandB project settings
- `args/aokvqa_coconut.yaml` - CoCoNut-specific tags and grouping
- `multicoco/config.py` - Enhanced LoggingConfig with WandB fields

**New Configuration Options:**
```yaml
# Logging Configuration
use_wandb: true
wandb_project: "multicoco-research"
wandb_entity: null  # Set to your team name if applicable  
wandb_tags: ["coconut", "multimodal"]  # List of tags
wandb_group: "aokvqa-coconut-progressive"  # Group runs
```

### 2. Automatic WandB Initialization

**Files Modified:**
- `run.py` - WandB initialization in MultiCoCoRunner

**Features:**
- Automatic WandB init with complete hyperparameter logging
- Custom metric definitions (train/loss, eval/accuracy)
- Model source and configuration tracking
- Distributed training support (rank 0 only)

### 3. Enhanced Training Metrics

**Files Modified:**
- `trainer.py` - Comprehensive logging in CoCoTrainer

**Training Metrics:**
- Per-epoch loss and accuracy tracking
- Real-time step-level metrics
- CoCoNut stage progression logging
- Latent token count tracking

### 4. Evaluation Insights

**Features:**
- **Sample Tables**: Up to 50 Q&A samples with predictions and correctness
- **Metric Tracking**: Accuracy, loss, and custom metrics
- **Distributed Evaluation**: Proper aggregation across multiple GPUs
- **Multimodal Support**: Ready for image logging (when available)

### 5. Artifacts Management

**Features:**
- **Model Checkpoints**: Automatic versioning per epoch
- **Dataset Artifacts**: Training and evaluation data tracking
- **Reproducibility**: Complete artifact lineage for experiments

### 6. Research Utilities

**Files Created:**
- `utils.py` - Enhanced with `log_wandb_samples()` function

**Utilities:**
- `log_wandb_samples()`: Custom function for detailed sample logging
- Multimodal image support (when images are available)
- Configurable sample limits and table names

### 7. Hyperparameter Sweeps

**Files Created:**
- `sweep.yaml` - WandB Sweeps configuration

**Sweep Parameters:**
- Generation temperature optimization
- CoCoNut epochs per stage tuning  
- Learning rate and batch size grid search
- Gradient accumulation optimization

## Usage Instructions

### 1. Authentication
```bash
wandb login
```

### 2. Basic Training
```bash
python run.py args/aokvqa_coconut.yaml
```

### 3. Disable WandB (for quick tests)
```yaml
# In your config file
use_wandb: false
```

### 4. Run Hyperparameter Sweeps
```bash
wandb sweep sweep.yaml
wandb agent <sweep_id>
```

### 5. Test Integration
```bash
python test_wandb_integration.py
```

## Dashboard Insights

Your WandB dashboard will show:

### Metrics Panels
- **Training Loss**: Real-time loss tracking per epoch
- **Evaluation Accuracy**: Validation performance trends
- **CoCoNut Stages**: Progression through latent token stages

### Tables
- **Evaluation Samples**: Question-answer pairs with predictions
- **Correctness Analysis**: Success/failure breakdown by sample

### Artifacts
- **Model Checkpoints**: Versioned model states per epoch
- **Datasets**: Training and evaluation data for reproducibility

### System Metrics (auto-enabled)
- GPU utilization and memory usage
- Training speed and throughput

## Configuration Examples

### Research Project Setup
```yaml
wandb_project: "multicoco-ablation-study"
wandb_tags: ["coconut", "multimodal", "ablation"]
wandb_group: "latent-tokens-experiment"
```

### Production Training
```yaml
wandb_project: "multicoco-production"
wandb_tags: ["production", "final-model"]
wandb_group: "v2.0-release"
```

### Quick Debugging
```yaml
use_wandb: false  # Disable for fast iteration
```

## Files Modified/Created

### Configuration Files
- ✅ `args/base.yaml` - Base WandB settings
- ✅ `args/aokvqa_coconut.yaml` - CoCoNut-specific configuration

### Core Implementation  
- ✅ `multicoco/config.py` - Enhanced LoggingConfig class
- ✅ `run.py` - WandB initialization and dataset artifacts
- ✅ `trainer.py` - Comprehensive training/evaluation logging
- ✅ `utils.py` - Research utility functions

### New Files
- ✅ `sweep.yaml` - Hyperparameter optimization configuration
- ✅ `test_wandb_integration.py` - Integration validation script
- ✅ `WANDB_INTEGRATION_README.md` - This documentation

### Dependencies
- ✅ `requirements.txt` - Already included `wandb` dependency

## Best Practices

### Experiment Organization
- Use **tags** for filtering experiments (e.g., ["baseline", "ablation"])
- Use **groups** for related runs (e.g., "coconut-stages-5-epochs")
- Use descriptive **run names** that include key hyperparameters

### Performance Considerations
- WandB adds minimal overhead (~1-2% training time)
- Sample tables are limited to 50 samples to prevent UI slowdowns
- Artifacts are logged only on rank 0 in distributed training

### Privacy and Security
- Anonymize sensitive data before logging
- Use `wandb_entity` for team/organization access control
- Consider private projects for proprietary datasets

## Troubleshooting

### Common Issues

**No logs appearing:**
- Check `use_wandb: true` in configuration
- Verify `wandb login` authentication
- Ensure rank 0 process in distributed training

**Import errors:**
- Install wandb: `pip install wandb`
- Check internet connectivity for WandB API

**Large artifact uploads:**
- Monitor disk space for checkpoint artifacts
- Consider filtering large temporary files
- Use artifact retention policies for storage management

**Multimodal logging:**
- Images automatically logged when available in dataset
- Supports PIL Images and numpy arrays
- Memory-efficient batch processing

## Integration Benefits

### For Researchers
- **Comprehensive Tracking**: Every hyperparameter and metric logged
- **Visualization**: Interactive charts for loss, accuracy trends
- **Comparison**: Side-by-side run comparison across configurations
- **Reproducibility**: Complete artifact and configuration versioning

### For Development
- **Debugging**: Sample tables reveal model prediction patterns
- **Optimization**: Automated hyperparameter sweeps
- **Collaboration**: Shared dashboards for team insights
- **Documentation**: Automatic experiment documentation

### For CoCoNut Research
- **Stage Tracking**: Monitor progression through latent token stages  
- **Curriculum Analysis**: Visualize curriculum learning effectiveness
- **Token Utilization**: Track latent token usage patterns
- **Performance Comparison**: Compare different stage configurations

## Next Steps

1. **Run Authentication**: `wandb login`
2. **Start Training**: `python run.py args/aokvqa_coconut.yaml`
3. **Monitor Dashboard**: Check WandB web interface for real-time metrics
4. **Experiment**: Try different configurations and compare results
5. **Optimize**: Use sweep.yaml for automated hyperparameter tuning

The WandB integration is now fully operational and ready for research use. All logging is handled automatically based on your configuration settings, providing maximum insights with minimal overhead. 