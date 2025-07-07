import os
import torch
from tqdm import tqdm
import torch.distributed as dist
import inspect
import re
from copy import copy
from PIL import Image
import torchvision.transforms as T

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
                
                    if self.args.get('bf16'):
                        batch['pixel_values'] = batch['pixel_values'].to(torch.bfloat16)

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
        Returns the question directly, as the conversation template is handled
        by the batch_chat method.
        
        Args:
            question: The original question with choices
            mode: "vanilla", "cot", or "coconut" (unused, but kept for consistency)
            
        Returns:
            The original question string.
        """
        # The batch_chat method in model.py wraps this content in the appropriate
        # conversation template. We provide only the raw question here to avoid
        # nested prompt confusion.
        return question

    def extract_answer_choice(self, response: str, mode: str = "vanilla") -> str:
        """
        Extract the answer choice from the model response.
        
        Args:
            response: Model response string
            mode: "vanilla", "cot", or "coconut"
            
        Returns:
            Extracted answer choice or empty string if not found
        """
        # More robust pattern for CoT to find the final answer
        if mode == "cot":
            match = re.search(r'the final answer is.*?([0-3])', response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # General pattern to find a digit, improved to be more specific
        general_patterns = [
            r'answer is ([0-3])',
            r'answer: ([0-3])',
            r'is: ([0-3])',
            r'is ([0-3])'
        ]
        
        for pattern in general_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback to finding the last single digit in the response
        single_digits = re.findall(r'\b[0-3]\b', response)
        if single_digits:
            return single_digits[-1]

        return ""

    def evaluate(self):
        """Main evaluation loop."""
        self.model.eval()
        correct = 0
        total = 0
        total_tokens = 0
        
        eval_config = self._get_eval_config()
        mode = "coconut" if eval_config.get('coconut') else "cot" if eval_config.get('cot') else "vanilla"

        log_dir = self.args.get('log_dir', 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file_path = os.path.join(log_dir, f'evaluation_{mode}.log')
        
        with open(log_file_path, 'w') as log_file:
            log_file.write(f"InternVL3-1B A-OKVQA Evaluation Log ({mode.upper()} mode)\n")
            
            pbar = tqdm(self.val_loader, desc=f"Evaluating ({mode.upper()})", disable=(dist.is_initialized() and dist.get_rank() != 0))

            for batch in pbar:
                pixel_values = batch.pop("pixel_values").to(self.device)
                input_ids = batch.pop("input_ids").to(self.device)
                attention_mask = batch.pop("attention_mask").to(self.device)
                image_flags = batch.pop("image_flags").to(self.device)
                original_questions = batch.pop("original_questions")
                ground_truths = batch.pop("answers")

                if self.args.get('bf16'):
                    pixel_values = pixel_values.to(torch.bfloat16)

                with torch.no_grad():
                    model_to_eval = self.model.module if hasattr(self.model, 'module') else self.model
                    
                    generation_config = {
                        'max_new_tokens': self.args.get('max_new_tokens', 500),
                        'temperature': 0.0,
                        'do_sample': False,
                        'pad_token_id': model_to_eval.tokenizer.pad_token_id,
                        'eos_token_id': model_to_eval.tokenizer.eos_token_id,
                    }

                    outputs = model_to_eval.generate(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        image_flags=image_flags,
                        **generation_config
                    )
                    
                    generated_texts = model_to_eval.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for i, gen_text in enumerate(generated_texts):
                    question = original_questions[i]
                    gt_answers = ground_truths[i]
                    
                    # The generated text is the full completion, remove the prompt part
                    prompt_len = len(model_to_eval.tokenizer.decode(input_ids[i], skip_special_tokens=True))
                    generated_answer = gen_text[prompt_len:].strip()

                    extracted_answer = self.extract_answer_choice(generated_answer, mode)
                    is_correct = extracted_answer in gt_answers

                    if is_correct:
                        correct += 1
                    total += 1
                    
                    tokens_generated = len(outputs[i]) - len(input_ids[i])
                    total_tokens += tokens_generated
                    
                    log_file.write("="*80 + "\n\n")
                    log_file.write(f"Sample {total}:\n")
                    log_file.write("-"*40 + "\n")
                    log_file.write(f"Question: {question}\n")
                    log_file.write(f"Generated Answer: {generated_answer}\n")
                    log_file.write(f"Extracted Answer: {extracted_answer}\n")
                    log_file.write(f"Ground Truth Answer: {gt_answers}\n")
                    log_file.write(f"Tokens Generated: {tokens_generated}\n")
                    log_file.write(f"Correct: {'Yes' if is_correct else 'No'}\n")
                    log_file.write("-"*40 + "\n\n")

            accuracy = correct / total if total > 0 else 0
            avg_tokens = total_tokens / total if total > 0 else 0
            
            summary = (
                f"\n{'='*80}\n"
                f"Evaluation Summary ({mode.upper()} mode)\n"
                f"Total samples: {total}\n"
                f"Accuracy: {accuracy:.4f}\n"
                f"Average tokens generated: {avg_tokens:.2f}\n"
                f"{'='*80}\n"
            )
            log_file.write(summary)
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(summary)

        return accuracy

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
            transform = T.Compose([
                T.Resize((input_size, input_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
