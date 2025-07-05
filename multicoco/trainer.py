import os
import torch
from tqdm import tqdm
import torch.distributed as dist
import inspect
import re

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
                
                    # Remove fields that are not model inputs before passing to the model
                    batch.pop("answers", None)
                    batch.pop("original_questions", None)

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

    def format_question(self, question: str) -> str:
        """
        Format the question for the model.
        
        Args:
            question: The original question with choices
            
        Returns:
            Formatted question string
        """
        # The question already contains the choices in the format:
        # "What is in the motorcyclist's mouth? The choices are 0 : toothpick, 1 : food, 2 : popsicle stick, 3 : cigarette"
        
        # Add instruction to make it clear we want a single number
        formatted_question = f"{question}\n\nPlease answer with only the number (0, 1, 2, or 3) corresponding to the correct choice."
        
        return formatted_question

    def extract_answer_choice(self, response: str) -> str:
        """
        Extract the answer choice (0, 1, 2, or 3) from the model response.
        
        Args:
            response: Model response string
            
        Returns:
            Extracted answer choice or empty string if not found
        """
        # Look for single digits in the response
        
        # First, try to find explicit single digit answers
        single_digits = re.findall(r'\b[0-3]\b', response)
        if single_digits:
            return single_digits[0]
        
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
        
        # If no clear answer found, return empty string
        return ""

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
            
            # Get the model to use for preprocessing
            model_to_eval = self.model.module if hasattr(self.model, 'module') else self.model
            
            # Access the underlying model that has the dynamic_preprocess method
            if hasattr(model_to_eval, 'model'):
                underlying_model = model_to_eval.model
            else:
                underlying_model = model_to_eval
            
            # Use the model's dynamic preprocessing if available
            if hasattr(underlying_model, 'dynamic_preprocess'):
                pixel_values_list, _ = underlying_model.dynamic_preprocess(
                    [image], 
                    image_size=underlying_model.config.image_size
                )
                pixel_values = torch.cat(pixel_values_list, dim=0)
                # Ensure bfloat16 dtype and correct device to match model
                pixel_values = pixel_values.to(device=self.device, dtype=torch.bfloat16)
                return pixel_values
            else:
                # Fallback to manual dynamic preprocessing
                return self._manual_dynamic_preprocess(image, underlying_model)
            
        except Exception as e:
            print(f"Error loading/preprocessing image {image_path}: {e}")
            return None

    def _manual_dynamic_preprocess(self, image, model, input_size: int = 448, max_num: int = 12):
        """
        Manual dynamic preprocessing for InternVL3 model.
        """
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        
        # Build transform
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        # Dynamic preprocessing logic
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        # Generate target ratios
        target_ratios = set(
            (i, j) for n in range(1, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= 1)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        # Find closest aspect ratio
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = orig_width * orig_height
        
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * input_size * input_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        
        # Resize and split image
        target_width = input_size * best_ratio[0]
        target_height = input_size * best_ratio[1]
        blocks = best_ratio[0] * best_ratio[1]
        
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        
        for i in range(blocks):
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

    def generate_answer(self, pixel_values: torch.Tensor, question: str) -> str:
        """
        Generate answer using model.chat() exactly like the standalone script.
        
        Args:
            pixel_values: Preprocessed pixel values tensor
            question: Formatted question string
            
        Returns:
            Generated answer string
        """
        try:
            # Get the model and tokenizer
            model_to_eval = self.model.module if hasattr(self.model, 'module') else self.model
            
            # Access the underlying model that has the chat method
            if hasattr(model_to_eval, 'model'):
                underlying_model = model_to_eval.model
            else:
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
            model_device = next(underlying_model.parameters()).device
            if pixel_values.device != model_device:
                pixel_values = pixel_values.to(model_device)
            
            # Use the model's chat method with preprocessed pixel values
            response = underlying_model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
                history=None,
                return_history=False
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return ""

    def evaluate(self):
        """Evaluation loop using the same logic as the standalone script."""
        self.model.eval()
        
        total_correct = torch.tensor([0.0]).to(self.device)
        total_samples = torch.tensor([0.0]).to(self.device)
        
        all_results = []
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0

        # Create a simple dataset from the validation loader
        # We need to iterate through the raw data, not the processed batches
        dataset = self.val_loader.dataset
        
        pbar = tqdm(range(len(dataset)), desc="Evaluating", disable=(dist.is_initialized() and dist.get_rank() != 0))

        with torch.no_grad():
            for idx in pbar:
                # Get raw sample data
                sample = dataset[idx]
                
                # Extract information
                image_path = sample['image'] if isinstance(sample['image'], str) else sample['image']
                question = sample['question']
                
                # Handle different answer formats
                if 'answers' in sample:
                    correct_answer = str(sample['answers'][0]) if isinstance(sample['answers'], list) else str(sample['answers'])
                else:
                    correct_answer = str(sample['answer'])
                
                # Preprocess image using the same logic as standalone script
                pixel_values = self.preprocess_image_for_eval(image_path)
                if pixel_values is None:
                    all_results.append({
                        "question": question,
                        "generated_answer": "",
                        "extracted_answer": "",
                        "ground_truth": correct_answer,
                        "correct": False
                    })
                    total_samples += 1
                    continue
                
                # Format question using the same logic as standalone script
                formatted_question = self.format_question(question)
                
                # Generate answer using model.chat()
                raw_response = self.generate_answer(pixel_values, formatted_question)
                
                # Extract answer choice using the same logic as standalone script
                extracted_answer = self.extract_answer_choice(raw_response)
                
                # Check correctness
                is_correct = extracted_answer == correct_answer

                if is_correct:
                    total_correct += 1

                all_results.append({
                    "question": question,
                    "generated_answer": raw_response,
                    "extracted_answer": extracted_answer,
                    "ground_truth": correct_answer,
                    "correct": is_correct
                })
                
                total_samples += 1

        # In DDP, gather results from all processes to the main process
        if dist.is_initialized():
            gathered_results = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_results, all_results)
            if is_main_process:
                # Flatten the list of lists
                all_results = [item for sublist in gathered_results for item in sublist]

        # Log results on the main process with the same format as the standalone script
        if is_main_process:
            with open('evaluation.log', 'w') as f:
                f.write("InternVL3-1B A-OKVQA Evaluation Log\n")
                f.write(f"Total samples: {len(all_results)}\n")
                f.write("="*80 + "\n\n")
                
                for i, res in enumerate(all_results, 1):
                    f.write(f"Sample {i}:\n")
                    f.write("----------------------------------------\n")
                    f.write(f"Question: {res['question']}\n")
                    f.write(f"Generated Answer: {res['generated_answer']}\n")
                    f.write(f"Extracted Answer: {res['extracted_answer']}\n")
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
