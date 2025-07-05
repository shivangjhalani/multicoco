import torch
from tqdm import tqdm
import torch.distributed as dist
import inspect
import re


def extract_final_answer_digit(s: str):
    """
    Extracts the digit that follows the phrase 'The final answer is: '.
    
    Args:
        s: The input string.
    
    Returns:
        The digit as a string, or None if the pattern is not found.
    """
    match = re.search(r'The final answer is:\s*(\d)', s)
    if match:
        return match.group(1)
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
                    batch[k] = v.to(self.args['device'])
            
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

        pbar = tqdm(self.val_loader, desc="Evaluating", disable=(not is_main_process))

        with torch.no_grad():
            for batch in pbar:
                batch.pop('labels', None)
                
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.args['device'])

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
                    
                    is_mcq = batch['is_mcq'][i]

                    with open('evaluation.log', 'a') as f:
                        question = batch['original_questions'][i]
                        choices_str = batch['choices_str'][i]
                        
                        if is_mcq:
                            predicted_answer = extract_final_answer_digit(generated_text)
                            is_correct = predicted_answer in gt_answers if predicted_answer is not None else False
                        else:
                            # For non-MCQ, check if any ground truth answer is in the generated text
                            is_correct = any(gt.lower() in generated_text.lower() for gt in gt_answers)

                        if is_correct:
                            correct_predictions += 1
                        
                        f.write('----------------------------------------\n')
                        f.write(f"Question: {question}{' Choices: ' + choices_str if is_mcq else ''}\n")
                        f.write(f"Generated Answer: {generated_text.strip()}\n")
                        f.write(f"Ground Truth Answers: {gt_answers}\n")
                        f.write(f"Correct: {'Yes' if is_correct else 'No'}\n")
                
                total_predictions += len(batch['original_questions'])

        if dist.is_initialized():
            total_correct_tensor = torch.tensor(correct_predictions).to(self.args['device'])
            total_preds_tensor = torch.tensor(total_predictions).to(self.args['device'])
            dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_preds_tensor, op=dist.ReduceOp.SUM)
            correct_predictions = total_correct_tensor.item()
            total_predictions = total_preds_tensor.item()
            
        if is_main_process:
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            print(f"Evaluation Accuracy: {accuracy * 100:.2f}%")
            return accuracy
        return None