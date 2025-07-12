# Issue #2 Fix Summary: Multi-Stage Training Loop in CoCoNut Mode

## Problem Description
The original implementation had several critical issues in the `_run_coconut_mode` method:

1. **Missing Trainer Initialization**: `self.trainer` was never created in coconut mode
2. **Incorrect Training Loop**: `self.trainer.train()` was called inside a loop (once per epoch), which restarts the entire training process
3. **No Dataset Updates**: Stage transitions logged metrics but didn't actually update the dataset curriculum

## Solution Implemented

### 1. Fixed `run.py` - `_run_coconut_mode` method
**File**: `/multicoco/run.py`

**Changes**:
- Added `self.create_trainer()` call before training starts
- Removed the epoch loop and multiple `train()` calls  
- Now calls `self.trainer.train()` only once
- Simplified the method to focus on setup and final evaluation

**Before**:
```python
def _run_coconut_mode(self) -> Dict[str, float]:
    # No trainer creation
    for epoch in range(total_epochs):
        # ... stage calculation logic ...
        self.trainer.train()  # ❌ Called multiple times, trainer doesn't exist
    # ... evaluation ...
```

**After**:
```python
def _run_coconut_mode(self) -> Dict[str, float]:
    self.create_trainer()  # ✅ Initialize trainer once
    # ... logging setup ...
    self.trainer.train()   # ✅ Single call to train()
    return self.trainer.perform_evaluation(log_per_sample=True)
```

### 2. Enhanced `trainer.py` - `CoCoTrainer` class  
**File**: `/multicoco/trainer.py`

**Changes**:
- Overrode `train()` method to detect CoCoNut mode and handle stage transitions
- Added `_train_with_coconut_stages()` method for multi-stage training
- Added `_update_for_stage()` method to apply progressive curriculum
- Added stage transition logging and metrics

**Key Methods Added**:

#### `train()` method override:
```python
def train(self, resume_from_checkpoint=None, **kwargs):
    is_coconut_mode = hasattr(self.args, 'epochs_per_stage') and hasattr(self.args, 'max_latent_stage')
    
    if is_coconut_mode:
        return self._train_with_coconut_stages(resume_from_checkpoint, **kwargs)
    else:
        return self._train_standard(resume_from_checkpoint, **kwargs)
```

#### `_train_with_coconut_stages()` method:
- Handles the main training loop with stage transitions
- Calculates current stage: `current_stage = min(epoch // self.args.epochs_per_stage, self.args.max_latent_stage)`
- Calls `_update_for_stage()` when stage changes
- Refreshes dataloader after dataset updates

#### `_update_for_stage()` method:
- Applies progressive curriculum to training dataset
- Optionally resets optimizer for new stage
- Logs stage transitions to wandb

## Key Improvements

### ✅ Fixed Training Loop Architecture
- **Before**: Multiple calls to `train()` in a loop (incorrect HuggingFace Trainer usage)
- **After**: Single call to `train()` with internal epoch loop handling stage transitions

### ✅ Proper Trainer Initialization  
- **Before**: No `create_trainer()` call in coconut mode
- **After**: Trainer properly initialized before training starts

### ✅ Progressive Curriculum Application
- **Before**: Commented out dataset updates (`# self._update_dataset_for_stage(stage)`)
- **After**: Active curriculum updates via `train_dataset.apply_progressive_curriculum()`

### ✅ Dataloader Refresh
- **Before**: Stale dataloader with unchanged dataset
- **After**: Dataloader refreshed after dataset curriculum updates

### ✅ Stage-based Optimizer Reset
- **Before**: No optimizer management between stages  
- **After**: Optional optimizer reset for each stage via `reset_optimizer` config

## Testing

Created comprehensive test in `test_issue2_fix.py` that verifies:
- Trainer is properly initialized 
- `train()` is called exactly once (not in a loop)
- Stage transitions work correctly
- Progressive curriculum is applied
- CoCoNut parameters are properly set

## Integration with Issue #1 Fix

This fix works seamlessly with Issue #1 (Sequential Latent Chaining):
- Stage transitions update the dataset with proper latent token curriculum
- The fixed latent wrapper processes the progressively updated datasets
- Multi-pass latent chaining works with the multi-stage training

## Compatibility

- **Backward Compatible**: Standard CoT training (non-CoCoNut mode) unchanged
- **CoCoNut Mode**: Now properly functional with multi-stage curriculum
- **Logging**: Enhanced wandb integration for stage tracking

## Expected Results

With this fix, CoCoNut training should now:
1. ✅ Execute without crashing (trainer properly initialized)
2. ✅ Apply progressive curriculum (datasets updated per stage)  
3. ✅ Follow proper HuggingFace Trainer patterns (single `train()` call)
4. ✅ Enable true multi-stage latent reasoning learning
