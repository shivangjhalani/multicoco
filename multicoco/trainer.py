import os
import torch
from tqdm import tqdm
import torch.distributed as dist
import inspect
import re

def parse_choices(question_text):
    """Extracts choices from the question text."""
    choices_match = re.search(r'The choices are (.+)', question_text)
    if not choices_match:
        return []
    
    choices_str = choices_match.group(1)
    # This regex handles "0 : choice, 1 : another"
    return [c.strip() for c in re.split(r'\d+\s*:\s*', choices_str) if c]

def is_answer_correct(generated_text, ground_truth_answers, choices):
    """
    Checks if the generated text contains a correct answer.
    It checks for the ground truth answer text itself, or the index of the correct answer.
    """
    generated_text_lower = generated_text.lower()
    
    for gt_ans in ground_truth_answers:
        gt_ans_lower = gt_ans.lower()

        # 1. Direct match of the ground truth answer text
        if gt_ans_lower in generated_text_lower:
            return True

        # 2. Check if the generated text contains the choice text corresponding to the answer
        try:
            # gt_ans is often the index as a string, like '3'
            choice_index = int(gt_ans)
            if 0 <= choice_index < len(choices):
                choice_text = choices[choice_index].lower()
                if choice_text in generated_text_lower:
                    return True
        except (ValueError, IndexError):
            # If gt_ans is not an index, or index is out of bounds, this will fail.
            # It might be the full answer text, which is already checked above.
            continue
            
    return False


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
        total_correct = torch.tensor([0.0]).to(self.device)
        total_samples = torch.tensor([0.0]).to(self.device)
        
        all_results = []
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0

        # The collator needs access to the tokenizer for decoding
        tokenizer = self.val_loader.collate_fn.tokenizer

        pbar = tqdm(self.val_loader, desc="Evaluating", disable=(dist.is_initialized() and dist.get_rank() != 0))

        with torch.no_grad():
            for batch in pbar:
                # We get prompts directly from the new collator
                prompts = batch.pop("prompts")
                original_answers = batch.pop("answers")
                original_questions = batch.pop("original_questions")

                # Move pixel_values to device
                pixel_values = batch['pixel_values'].to(self.device)
                
                model_to_eval = self.model.module if hasattr(self.model, 'module') else self.model

                # Tokenize prompts
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=tokenizer.model_max_length)
                input_ids = inputs.input_ids.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device)

                # Generate outputs
                outputs = model_to_eval.model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=100,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                # Decode and compare
                generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for i, gen_text in enumerate(generated_texts):
                    # Clean up the generated text by removing the prompt part
                    prompt_part = prompts[i].replace('<img>' * self.val_loader.collate_fn.num_image_tokens + '\n', '').strip()
                    answer_text = gen_text.replace(prompt_part, '').strip()

                    # Extract choices from the original question to help with parsing
                    choices = parse_choices(original_questions[i])
                    
                    is_correct = is_answer_correct(answer_text, original_answers[i], choices)
                    if is_correct:
                        total_correct += 1

                    all_results.append({
                        "question": original_questions[i],
                        "generated_answer": answer_text,
                        "ground_truth": original_answers[i],
                        "correct": is_correct
                    })
                
                total_samples += len(original_answers)

        # In DDP, gather results from all processes to the main process
        if dist.is_initialized():
            gathered_results = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_results, all_results)
            if is_main_process:
                # Flatten the list of lists
                all_results = [item for sublist in gathered_results for item in sublist]

        # Log results on the main process
        if is_main_process:
            with open('evaluation.log', 'w') as f:
                for res in all_results:
                    f.write("----------------------------------------\n")
                    f.write(f"Question: {res['question']}\n")
                    f.write(f"Generated Answer: {res['generated_answer']}\n")
                    f.write(f"Ground Truth Answers: {res['ground_truth']}\n")
                    f.write(f"Correct: {'Yes' if res['correct'] else 'No'}\n")
                    f.write("----------------------------------------\n\n")

        # Aggregate results from all processes in DDP for accuracy calculation
        if dist.is_initialized():
            dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

        return (total_correct / total_samples).item() if total_samples > 0 else 0.0

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
