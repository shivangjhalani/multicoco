import os
import torch
from tqdm import tqdm
import torch.distributed as dist
import inspect
import re
from copy import copy

class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, args, wandb_run=None, text_table=None):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = dist.get_rank() if dist.is_initialized() else 'cuda'
        self.best_val_acc = 0.0
        self.wandb_run = wandb_run
        self.text_table = text_table
        self.total_train_steps = 0

    def _get_train_config_for_stage(self, stage):
        """Returns the training configuration for a given stage."""
        if stage == 0: # CoT training
            return {'is_train': True, 'coconut': False}
        else: # Coconut training
            return {'is_train': True, 'coconut': True, 'c_thought': self.args.get('c_thought', 2)}

    def _get_eval_config(self):
        """Returns the evaluation configuration based on args."""
        cot_mode = self.args.get('cot', False)
        coconut_mode = self.args.get('coconut', False)
        
        if coconut_mode:
            return {'is_train': False, 'coconut': True}
        elif cot_mode:
            return {'is_train': False, 'coconut': False, 'cot': True}
        else:
            return {'is_train': False, 'coconut': False, 'cot': False}

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
            self.val_loader.collate_fn.train_config = self._get_eval_config()

            num_epochs = self.args.get('epochs_per_stage', 1)

            for epoch in range(num_epochs):
                self.model.train()
                total_loss = 0
                self.optimizer.zero_grad()
                
                # Use tqdm for progress bar
                pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Stage {stage}/Epoch {epoch+1}", disable=(dist.is_initialized() and dist.get_rank() != 0))

                for i, batch in pbar:
                    # Log training data on first step if wandb is enabled
                    if i == 0 and self.wandb_run and self.text_table and (not dist.is_initialized() or dist.get_rank() == 0):
                        print("logging training data")
                        text_str = ""
                        if 'input_ids' in batch:
                            cur_bs = len(batch["input_ids"])
                            for data_idx in range(min(cur_bs, 2)):  # Log max 2 samples to avoid overwhelming logs
                                for token_idx in range(min(len(batch["input_ids"][data_idx]), 50)):  # Log max 50 tokens per sample
                                    if 'labels' in batch:
                                        text_str += (
                                            str(batch["input_ids"][data_idx][token_idx].item())
                                            + " "
                                            + str(batch["labels"][data_idx][token_idx].item())
                                            + " "
                                            + self.train_loader.collate_fn.tokenizer.decode(
                                                batch["input_ids"][data_idx][token_idx]
                                            )
                                            + "\n"
                                        )
                                text_str += "====" * 10 + "\n"
                        self.text_table.add_data(self.total_train_steps, text_str)
                        self.wandb_run.log({"data_table": copy(self.text_table)})

                    # Move batch to device
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(self.device)
                
                    # Remove fields that are not model inputs before passing to the model
                    batch.pop("answers", None)
                    batch.pop("original_questions", None)
                    batch.pop("steps", None)

                    output = self.model(**batch)
                    loss = output.loss / grad_accumulation_steps
                    
                    loss.backward()

                    if (i + 1) % grad_accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    
                    total_loss += loss.item() * grad_accumulation_steps
                    pbar.set_postfix({"loss": loss.item() * grad_accumulation_steps})
                    
                    # Log training metrics to wandb
                    if self.wandb_run and (not dist.is_initialized() or dist.get_rank() == 0):
                        log_dict = {
                            "train/stage": stage,
                            "train/epoch": epoch + 1,
                            "train/step": self.total_train_steps,
                            "train/loss": loss.item() * grad_accumulation_steps,
                        }
                        self.wandb_run.log(log_dict)
                    
                    self.total_train_steps += 1
                
                avg_loss = total_loss / len(self.train_loader)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"Stage {stage}, Epoch {epoch+1}: Average Training Loss: {avg_loss:.4f}")
                
                # Evaluate after each epoch
                val_acc = self.evaluate()
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"Stage {stage}, Epoch {epoch+1}: Validation Accuracy: {val_acc:.4f}")
                    
                    # Log validation metrics to wandb
                    if self.wandb_run:
                        log_dict = {
                            "eval/stage": stage,
                            "eval/epoch": epoch + 1,
                            "eval/acc": val_acc,
                            "eval/loss": avg_loss,
                        }
                        self.wandb_run.log(log_dict)
                    
                    self.save_checkpoint(stage, epoch, val_acc)

    def format_question_for_mode(self, question: str, mode: str) -> str:
        """
        Format the question based on evaluation mode.
        
        Args:
            question: The original question with choices
            mode: "vanilla", "cot", or "coconut"
            
        Returns:
            Formatted question string
        """
        if mode == "cot":
            return f"{question}\n\nPlease think step by step and provide your reasoning, then give your final answer as a number (0, 1, 2, or 3)."
        else:
            return f"{question}\n\nPlease answer with only the number (0, 1, 2, or 3) corresponding to the correct choice."

    def extract_answer_choice(self, response: str, mode: str = "vanilla") -> str:
        """
        Extract the answer choice from the model response.
        
        Args:
            response: Model response string
            mode: "vanilla", "cot", or "coconut"
            
        Returns:
            Extracted answer choice or empty string if not found
        """
        # For CoT mode, look for the final answer after reasoning
        if mode == "cot":
            # Look for patterns like "Therefore, the answer is 2" or "The answer is 2"
            final_answer_patterns = [
                r'(?:therefore|thus|so),?\s+(?:the\s+)?answer\s+is\s+([0-3])',
                r'(?:final|my)\s+answer\s+is\s+([0-3])',
                r'answer:\s*([0-3])',
                r'the\s+answer\s+is\s+([0-3])'
            ]
            
            for pattern in final_answer_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    return match.group(1)
        
        # Look for single digits in the response (works for all modes)
        single_digits = re.findall(r'\b[0-3]\b', response)
        if single_digits:
            return single_digits[-1]  # Take the last one for CoT (likely the final answer)
        
        # Try to find answer patterns
        answer_patterns = [
            r'answer\s*:?\s*([0-3])',
            r'choice\s*:?\s*([0-3])',
            r'option\s*:?\s*([0-3])',
            r'([0-3])\s*:',
            r'^\s*([0-3])\s*$'
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ""

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        tokenizer = self.val_loader.collate_fn.tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

    def preprocess_image_for_eval(self, image_path: str):
        """
        Preprocess image for evaluation using the same logic as the standalone script.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed pixel values tensor or None if image can't be loaded
        """
        try:
            from PIL import Image
            
            # Load image and convert to RGB
            image = Image.open(image_path).convert('RGB')
            
            # Get the underlying model that has the dynamic_preprocess method
            model_to_eval = self.model.module if hasattr(self.model, 'module') else self.model
            if hasattr(model_to_eval, 'model'):
                underlying_model = model_to_eval.model
            else:
                underlying_model = model_to_eval
            
            # Use dynamic preprocessing if available
            if hasattr(underlying_model, 'dynamic_preprocess'):
                pixel_values_list, _ = underlying_model.dynamic_preprocess([image], image_size=underlying_model.config.image_size)
                pixel_values = torch.cat(pixel_values_list, dim=0)
            else:
                # Fallback to manual preprocessing
                pixel_values = self._manual_dynamic_preprocess(image, underlying_model)
            
            return pixel_values
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None

    def _manual_dynamic_preprocess(self, image, model, input_size: int = 448, max_num: int = 12):
        """Manual implementation of dynamic preprocessing."""
        # Get image processor from the data collator
        image_processor = self.val_loader.collate_fn.image_processor
        
        # Get transform from image processor
        if hasattr(image_processor, 'transforms'):
            transform = image_processor.transforms
        else:
            # Create a basic transform if not available
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Get original dimensions
        width, height = image.size
        aspect_ratio = width / height
        
        # Calculate target dimensions
        if aspect_ratio > 1:
            target_width = input_size * max_num
            target_height = int(target_width / aspect_ratio)
        else:
            target_height = input_size * max_num
            target_width = int(target_height * aspect_ratio)
        
        # Ensure dimensions are multiples of input_size
        target_width = (target_width // input_size) * input_size
        target_height = (target_height // input_size) * input_size
        
        # Resize image
        resized_img = image.resize((target_width, target_height))
        
        # Split into blocks
        processed_images = []
        for i in range((target_width // input_size) * (target_height // input_size)):
            box = (
                (i % (target_width // input_size)) * input_size,
                (i // (target_width // input_size)) * input_size,
                ((i % (target_width // input_size)) + 1) * input_size,
                ((i // (target_width // input_size)) + 1) * input_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        
        # Add thumbnail if multiple blocks
        if len(processed_images) != 1:
            thumbnail_img = image.resize((input_size, input_size))
            processed_images.append(thumbnail_img)
        
        # Apply transforms and stack
        pixel_values = [transform(img) for img in processed_images]
        pixel_values = torch.stack(pixel_values)
        
        # Convert to bfloat16 and move to correct device to match model
        pixel_values = pixel_values.to(device=self.device, dtype=torch.bfloat16)
        
        return pixel_values

    def generate_answer(self, pixel_values: torch.Tensor, questions: list[str]) -> tuple[list[str], list[int]]:
        """
        Generate answers for a batch of questions.
        
        Args:
            pixel_values: Tensor of preprocessed image pixels for the batch
            questions: List of question strings for the batch
            
        Returns:
            A tuple containing a list of response strings and a list of token counts
        """
        try:
            # Get the underlying model and tokenizer
            model_to_eval = self.model.module if hasattr(self.model, 'module') else self.model
            
            # The `batch_chat` method is on the MultiCoCo model wrapper, not the underlying HF model
            underlying_model = model_to_eval
            
            tokenizer = self.val_loader.collate_fn.tokenizer
            
            # Check if model and tokenizer are available
            if underlying_model is None or tokenizer is None:
                raise ValueError("Model and tokenizer must be loaded before generating answers")
            
            # Use the model's chat method for inference with same config as standalone script
            generation_config = dict(
                max_new_tokens=100, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Ensure pixel_values are on the same device as model
            model_device = next(underlying_model.model.parameters()).device
            if pixel_values.device != model_device:
                pixel_values = pixel_values.to(model_device)
            
            # Use the model's batch_chat method for batch inference
            responses = underlying_model.batch_chat(
                tokenizer,
                pixel_values,
                questions,
                generation_config
            )
            
            token_counts = [self.count_tokens(resp) for resp in responses]
            
            return responses, token_counts
            
        except Exception:
            import traceback
            traceback.print_exc()
            return ["" for _ in questions], [0 for _ in questions]

    def evaluate(self):
        """Evaluation loop with CoT support and token counting."""
        self.model.eval()
        
        total_correct = torch.tensor([0.0]).to(self.device)
        total_samples = torch.tensor([0.0]).to(self.device)
        total_tokens = torch.tensor([0.0]).to(self.device)
        
        all_results = []
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0

        # Determine evaluation mode
        eval_config = self._get_eval_config()
        cot_mode = eval_config.get('cot', False)
        coconut_mode = eval_config.get('coconut', False)
        
        if coconut_mode:
            mode_name = "coconut"
        elif cot_mode:
            mode_name = "cot"
        else:
            mode_name = "vanilla"
        
        pbar = tqdm(self.val_loader, desc=f"Evaluating ({mode_name})", disable=(not is_main_process))

        with torch.no_grad():
            for batch in pbar:
                # Move batch to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                pixel_values = batch['pixel_values']
                questions = batch['original_questions']
                correct_answers = batch['answers']
                steps_list = batch.get('steps', [[] for _ in range(len(questions))])

                # Format questions for the current mode
                formatted_questions = [self.format_question_for_mode(q, mode_name) for q in questions]

                # Generate answers in batch
                raw_responses, token_counts = self.generate_answer(pixel_values, formatted_questions)

                for i in range(len(questions)):
                    raw_response = raw_responses[i]
                    token_count = token_counts[i]
                    correct_answer = str(correct_answers[i])
                    question = questions[i]
                    steps = steps_list[i]

                    # Extract answer choice
                    extracted_answer = self.extract_answer_choice(raw_response, mode_name)
                    
                    # Check correctness
                    is_correct = extracted_answer == correct_answer

                    if is_correct:
                        total_correct += 1

                    total_tokens += token_count

                    all_results.append({
                        "question": question,
                        "steps": steps,
                        "generated_answer": raw_response,
                        "extracted_answer": extracted_answer,
                        "ground_truth": correct_answer,
                        "correct": is_correct,
                        "tokens_generated": token_count,
                        "mode": mode_name
                    })
                
                total_samples += len(questions)

        # In DDP, gather results from all processes to the main process
        if dist.is_initialized():
            gathered_results = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_results, all_results)
            if is_main_process:
                # Flatten the list of lists
                all_results = [item for sublist in gathered_results for item in sublist]

        # Log results on the main process
        if is_main_process:
            avg_tokens = (total_tokens / total_samples).item() if total_samples > 0 else 0
            accuracy = (total_correct / total_samples).item() if total_samples > 0 else 0
            
            log_filename = f'evaluation_{mode_name}.log'
            with open(log_filename, 'w') as f:
                f.write(f"InternVL3-1B A-OKVQA Evaluation Log ({mode_name.upper()} mode)\n")
                f.write(f"Total samples: {len(all_results)}\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"Average tokens generated: {avg_tokens:.2f}\n")
                f.write("="*80 + "\n\n")
                
                for i, res in enumerate(all_results, 1):
                    f.write(f"Sample {i}:\n")
                    f.write("----------------------------------------\n")
                    f.write(f"Question: {res['question']}\n")
                    if res['steps'] and mode_name == "cot":
                        f.write(f"Ground Truth Reasoning: {' '.join(res['steps'])}\n")
                    f.write(f"Generated Answer: {res['generated_answer']}\n")
                    f.write(f"Extracted Answer: {res['extracted_answer']}\n")
                    f.write(f"Ground Truth Answer: {res['ground_truth']}\n")
                    f.write(f"Tokens Generated: {res['tokens_generated']}\n")
                    f.write(f"Correct: {'Yes' if res['correct'] else 'No'}\n")
                    f.write("----------------------------------------\n\n")
            
            print(f"Evaluation complete. Results saved to {log_filename}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Average tokens generated: {avg_tokens:.2f}")

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
