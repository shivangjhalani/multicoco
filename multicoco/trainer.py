import torch
from tqdm import tqdm
import torch.distributed as dist
import inspect
import re


def extract_answer_from_model_output(generated_text: str, choices: list):
    """
    Extracts the answer from the model's generated text for a multiple-choice question.

    This function tries to find the answer in a hierarchical manner:
    1. It checks if the text of any of the choices appears in the generated output.
    2. If not, it looks for a unique digit in the text that corresponds to a choice index.

    Args:
        generated_text: The output from the model.
        choices: A list of the possible answer strings.

    Returns:
        The index of the chosen answer as a string, or None if no answer can be found.
    """
    # 1. Check for the full text of a choice in the generated answer
    for i, choice_text in enumerate(choices):
        if re.search(r'\b' + re.escape(choice_text) + r'\b', generated_text, re.IGNORECASE):
            return str(i)

    # 2. If no text match, look for a digit corresponding to a choice
    # Find all digits in the string
    found_digits = re.findall(r'\d', generated_text)
    
    # Filter for digits that are valid choice indices
    valid_indices = [d for d in found_digits if int(d) < len(choices)]
    
    # If there is exactly one valid digit, return it
    if len(valid_indices) == 1:
        return valid_indices[0]
    
    # Advanced search for patterns like "the answer is 2"
    match = re.search(r'(?:the final answer is|the answer is|the correct answer is|choice is)\s*:?\s*(\d)', generated_text, re.IGNORECASE)
    if match:
        digit = match.group(1)
        if int(digit) < len(choices):
            return digit
            
    # As a last resort, return the last valid digit if any exist
    if valid_indices:
        return valid_indices[-1]

    return None


class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, args):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args

    def train(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            # Move all tensor data to the device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.args.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch} Average Loss: {avg_loss}")
        return avg_loss

    def evaluate(self):
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        
        model_to_eval = self.model.module if hasattr(self.model, 'module') else self.model
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        tokenizer = self.val_loader.collate_fn.tokenizer
        
        # Prevent repetitive warning
        if hasattr(model_to_eval.model.config, 'pad_token_id'):
            model_to_eval.model.config.pad_token_id = tokenizer.eos_token_id

        pbar = tqdm(self.val_loader, desc="Evaluating", disable=(not is_main_process))

        with torch.no_grad():
            for batch in pbar:
                batch.pop('labels', None)
                
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.args.device)

                generate_kwargs = {
                    'pixel_values': batch['pixel_values'],
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                    'do_sample': False,
                    'num_beams': 1,
                    'max_new_tokens': 150,
                }
                
                generate_args_spec = inspect.signature(model_to_eval.model.generate).parameters
                if 'image_flags' in generate_args_spec and 'image_flags' in batch:
                    generate_kwargs['image_flags'] = batch['image_flags']

                outputs = model_to_eval.model.generate(**generate_kwargs)

                generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                for i, (generated_text, gt_answers) in enumerate(zip(generated_texts, batch['answers'])):
                    if i >= len(batch['original_questions']):
                        break
                    
                    with open('evaluation.log', 'a') as f:
                        question = batch['original_questions'][i]
                        
                        choices_list = batch['choices'][i]
                        choices_str = ", ".join([f"{idx}: {choice}" for idx, choice in enumerate(choices_list)])
                        predicted_answer = extract_answer_from_model_output(generated_text, choices_list)
                        is_correct = predicted_answer in gt_answers if predicted_answer is not None else False

                        if is_correct:
                            correct_predictions += 1
                        
                        f.write('----------------------------------------\n')
                        f.write(f"Question: {question}{' The choices are ' + choices_str}\n")
                        f.write(f"Generated Answer: {generated_text.strip()}\n")
                        f.write(f"Ground Truth Answers: {gt_answers}\n")
                        f.write(f"Correct: {'Yes' if is_correct else 'No'}\n")
                
                total_predictions += len(batch['original_questions'])

        if dist.is_initialized():
            total_correct_tensor = torch.tensor(correct_predictions).to(self.args.device)
            total_preds_tensor = torch.tensor(total_predictions).to(self.args.device)
            dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_preds_tensor, op=dist.ReduceOp.SUM)
            correct_predictions = total_correct_tensor.item()
            total_predictions = total_preds_tensor.item()
            
        if is_main_process:
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            print(f"Evaluation Accuracy: {accuracy * 100:.2f}%")
            return accuracy
        return None