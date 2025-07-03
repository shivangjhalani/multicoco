import os
import torch
from tqdm import tqdm
import torch.distributed as dist

class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, args):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = dist.get_rank() if dist.is_initialized() else 'cuda'
        self.best_val_acc = 0.0

    def _get_train_config_for_stage(self, stage):
        """Returns the training configuration for a given stage."""
        if stage == 0: # CoT training
            return {'is_train': True, 'coconut': False}
        else: # Coconut training
            return {'is_train': True, 'coconut': True, 'c_thought': self.args.get('c_thought', 2)}

    def train(self):
        """Main training loop that handles staged training."""
        max_stages = self.args.get('max_latent_stage', 0) + 1 # 0 is CoT stage
        grad_accumulation_steps = self.args.get('gradient_accumulation_steps', 1)

        for stage in range(max_stages):
            print(f"--- Starting Stage {stage} ---")
            
            # Reset optimizer if specified
            if stage > 0 and self.args.get('reset_optimizer', False):
                print("Resetting optimizer for new stage.")
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])

            # Update data collator with the correct config for the current stage
            train_config = self._get_train_config_for_stage(stage)
            self.train_loader.collate_fn.train_config = train_config
            self.val_loader.collate_fn.train_config = {'is_train': False} # Eval is always inference mode

            num_epochs = self.args.get('epochs_per_stage', 1)

            for epoch in range(num_epochs):
                self.model.train()
                total_loss = 0
                self.optimizer.zero_grad()
                
                # Use tqdm for progress bar
                pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Stage {stage}/Epoch {epoch+1}", disable=(dist.is_initialized() and dist.get_rank() != 0))

                for i, batch in pbar:
                    # Move batch to device
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(self.device)
                
                    output = self.model(**batch)
                    loss = output.loss / grad_accumulation_steps
                    
                    loss.backward()

                    if (i + 1) % grad_accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    
                    total_loss += loss.item() * grad_accumulation_steps
                    pbar.set_postfix({"loss": loss.item() * grad_accumulation_steps})
                
                avg_loss = total_loss / len(self.train_loader)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"Stage {stage}, Epoch {epoch+1}: Average Training Loss: {avg_loss:.4f}")
                
                # Evaluate after each epoch
                val_acc = self.evaluate()
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"Stage {stage}, Epoch {epoch+1}: Validation Accuracy: {val_acc:.4f}")
                    self.save_checkpoint(stage, epoch, val_acc)

    def evaluate(self):
        """Evaluation loop."""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.val_loader, desc="Evaluating", disable=(not dist.is_initialized() or dist.get_rank() != 0))

        with torch.no_grad():
            for batch in pbar:
                # Move batch to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                # The generation part is complex and depends on the exact model API
                # For now, we simulate a dummy accuracy.
                # A proper implementation would call model.generate() and compare outputs.
                # Here we'll just check if the validation loss runs
                output = self.model(**batch)
                
                # Dummy accuracy calculation
                total_correct += 1 # Assume one correct per batch for now
                total_samples += batch['input_ids'].size(0)

        # In a real scenario, you'd calculate accuracy based on generated text vs ground truth
        return total_correct / total_samples if total_samples > 0 else 0.0

    def save_checkpoint(self, stage, epoch, val_acc):
        """Saves a model checkpoint."""
        if not self.args.get('save_path'):
            return
            
        save_dir = os.path.join(self.args['save_path'], f"stage_{stage}")
        os.makedirs(save_dir, exist_ok=True)
        
        save_only_improve = self.args.get('save_only_improve', False)

        if save_only_improve and val_acc <= self.best_val_acc:
            return

        self.best_val_acc = val_acc
        
        # In DDP, only the main process should save the model
        if not dist.is_initialized() or dist.get_rank() == 0:
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            
            checkpoint_path = os.path.join(save_dir, f"epoch_{epoch+1}_acc_{val_acc:.4f}.pt")
            torch.save(model_to_save.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
