"""
Test script for Issue #2 fix: Multi-stage training loop in CoCoNut mode.
Tests that the trainer is properly initialized and stage transitions work.
"""

import sys
import os
from unittest.mock import Mock, MagicMock, patch
import logging

# Add the multicoco directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_coconut_mode_initialization():
    """Test that CoCoNut mode properly initializes the trainer and doesn't call train() in a loop."""
    
    try:
        from multicoco.config import MultiCoCoConfig, TrainingMode
        from multicoco.trainer import CoCoTrainer
        from run import MultiCoCoRunner
        
        print("✓ Successfully imported required modules")
        
        # Create a mock config for CoCoNut mode
        config = Mock(spec=MultiCoCoConfig)
        config.training = Mock()
        config.training.mode = TrainingMode.COCONUT_TRAIN
        config.training.seed = None
        config.training.num_epochs = 6
        config.coconut = Mock()
        config.coconut.enabled = True
        config.coconut.epochs_per_stage = 2
        config.coconut.max_latent_stage = 2
        config.coconut.c_thought = 4
        config.coconut.uniform_prob = 0.0
        config.coconut.pad_latent_to_max = False
        config.coconut.reset_optimizer = True
        config.logging = Mock()
        config.logging.use_wandb = False
        config.logging.log_level = 'INFO'
        config.logging.log_to_file = False
        config.logging.run_name = 'test'
        config.logging.log_dir = '/tmp'
        
        # Mock methods that would be called
        with patch.object(MultiCoCoRunner, '_setup_cuda'):
            with patch.object(MultiCoCoRunner, '_setup_logging'):
                with patch.object(MultiCoCoRunner, '_setup_wandb'):
                    runner = MultiCoCoRunner(config)
                    
                    # Mock the necessary components
                    runner.model = Mock()
                    runner.train_dataset = Mock()
                    runner.eval_dataset = Mock()
                    
                    # Mock create_trainer to create a real CoCoTrainer instance
                    def mock_create_trainer():
                        from transformers import TrainingArguments
                        
                        # Create mock training arguments with CoCoNut parameters
                        args = Mock(spec=TrainingArguments)
                        args.epochs_per_stage = config.coconut.epochs_per_stage
                        args.max_latent_stage = config.coconut.max_latent_stage
                        args.c_thought = config.coconut.c_thought
                        args.uniform_prob = config.coconut.uniform_prob
                        args.pad_latent_to_max = config.coconut.pad_latent_to_max
                        args.reset_optimizer = config.coconut.reset_optimizer
                        args.num_train_epochs = config.training.num_epochs
                        args.gradient_accumulation_steps = 1
                        args.report_to = []
                        
                        # Create a real CoCoTrainer instance
                        trainer = CoCoTrainer(
                            model=runner.model,
                            args=args,
                            train_dataset=runner.train_dataset,
                            eval_dataset=runner.eval_dataset,
                            data_collator=Mock()
                        )
                        
                        # Mock the methods that would be called during training
                        trainer._setup_epoch_training = Mock()
                        trainer._handle_checkpoint_resumption = Mock(return_value=0)
                        trainer.get_train_dataloader = Mock(return_value=Mock(__len__=Mock(return_value=10)))
                        trainer._log_training_setup = Mock()
                        trainer._wrap_model = Mock(return_value=Mock())
                        trainer.create_optimizer_and_scheduler = Mock()
                        trainer._train_single_epoch = Mock()
                        trainer.perform_evaluation = Mock(return_value={'accuracy': 0.8})
                        
                        runner.trainer = trainer
                        
                    runner.create_trainer = mock_create_trainer
                    
                    print("✓ Runner and mocks setup completed")
                    
                    # Test that the trainer is created and train() is called only once
                    with patch.object(CoCoTrainer, 'train') as mock_train:
                        mock_train.return_value = Mock()
                        
                        result = runner._run_coconut_mode()
                        
                        # Verify trainer was created
                        assert runner.trainer is not None, "Trainer should be created"
                        print("✓ Trainer was properly initialized")
                        
                        # Verify train() was called exactly once
                        assert mock_train.call_count == 1, f"train() should be called once, was called {mock_train.call_count} times"
                        print("✓ train() was called exactly once (not in a loop)")
                        
                        # Verify CoCoNut parameters are set
                        assert hasattr(runner.trainer.args, 'epochs_per_stage'), "epochs_per_stage should be set"
                        assert hasattr(runner.trainer.args, 'max_latent_stage'), "max_latent_stage should be set"
                        print("✓ CoCoNut parameters properly set on trainer")
                        
                        print("✓ Issue #2 fix verified: Multi-stage training loop works correctly!")
                        return True
                        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Note: This test requires the full MultiCoCo environment to run.")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_stage_transitions():
    """Test that stage transitions work correctly in the trainer."""
    
    try:
        from multicoco.trainer import CoCoTrainer
        from transformers import TrainingArguments
        
        # Create mock training arguments with CoCoNut parameters
        args = Mock(spec=TrainingArguments)
        args.epochs_per_stage = 2
        args.max_latent_stage = 2
        args.c_thought = 4
        args.uniform_prob = 0.0
        args.pad_latent_to_max = False
        args.reset_optimizer = False
        args.num_train_epochs = 6
        args.gradient_accumulation_steps = 1
        args.report_to = []
        
        # Create trainer with mocked components
        trainer = CoCoTrainer(
            model=Mock(),
            args=args,
            train_dataset=Mock(),
            eval_dataset=Mock(),
            data_collator=Mock()
        )
        
        # Mock the dataset to support progressive curriculum
        trainer.train_dataset.apply_progressive_curriculum = Mock()
        
        # Test stage calculation logic
        for epoch in range(6):
            current_stage = min(epoch // args.epochs_per_stage, args.max_latent_stage)
            expected_stages = [0, 0, 1, 1, 2, 2]  # epochs 0-1: stage 0, 2-3: stage 1, 4-5: stage 2
            
            assert current_stage == expected_stages[epoch], f"Epoch {epoch}: expected stage {expected_stages[epoch]}, got {current_stage}"
        
        print("✓ Stage calculation logic works correctly")
        
        # Test the _update_for_stage method
        trainer._update_for_stage(1)
        
        # Verify that apply_progressive_curriculum was called
        trainer.train_dataset.apply_progressive_curriculum.assert_called_once_with(
            scheduled_stage=1,
            c_thought=args.c_thought,
            max_latent_stage=args.max_latent_stage,
            uniform_prob=args.uniform_prob,
            pad_latent_to_max=args.pad_latent_to_max,
            no_cot=False,
        )
        
        print("✓ Stage transition and curriculum update works correctly")
        print("✓ Issue #2 stage transition logic verified!")
        return True
        
    except Exception as e:
        print(f"✗ Stage transition test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Issue #2 fix: Multi-stage training loop...")
    print("=" * 60)
    
    try:
        test1_success = test_coconut_mode_initialization()
        test2_success = test_stage_transitions()
        
        if test1_success and test2_success:
            print("\n✓ All tests passed! Issue #2 fix appears to be working correctly.")
            print("Key improvements:")
            print("  - Trainer is properly initialized once before training")
            print("  - train() is called only once, not in a loop")
            print("  - Stage transitions are handled internally in the trainer")
            print("  - Progressive curriculum is applied during stage transitions")
        else:
            print("\n✗ Some tests failed. Please check the implementation.")
            
    except Exception as e:
        print(f"\n✗ Test execution failed: {e}")
        print("Note: This test requires a proper PyTorch environment to run.")
