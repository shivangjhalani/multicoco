import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, Sequence
import json
# from multicoco.conversation import get_conv_template # No longer needed

class MultiCoCoDataset(Dataset):
    def __init__(self, data_path, data_dir):
        self.data = []
        self.data_dir = data_dir
        if data_path and os.path.exists(data_path):
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        
        # Temporary: Slice the dataset to only use the first 10 examples for quick evaluation.
        # Remove this line to use the full dataset again.
        if data_path and "val" in data_path: # Apply only to validation set
            self.data = self.data[:20]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "image": os.path.join(self.data_dir, item["image"]),
            "question": item["question"],
            "answer": item["answer"],
            "answers": item.get("answers", [item.get("answer")]),
            "steps": item.get("steps", [])  # Chain of thought steps
        }

class DataCollatorForInternVL(object):
    def __init__(self, tokenizer, model, image_processor):
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.train_config = {'is_train': True}  # Default to training mode
        self.thought_token_id = tokenizer.convert_tokens_to_ids('<thought>')
        self.start_thought_id = tokenizer.convert_tokens_to_ids('<start_thought>')
        self.end_thought_id = tokenizer.convert_tokens_to_ids('<end_thought>')

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        is_train = self.train_config.get('is_train', True)
        is_coconut = self.train_config.get('coconut', False)
        
        if is_coconut:
            return self.prepare_coconut_batch(instances, is_train)

        return self.prepare_cot_batch(instances, is_train)
        
    def prepare_cot_batch(self, instances: Sequence[Dict], is_train: bool) -> Dict[str, torch.Tensor]:
        images = [Image.open(ins.pop('image')).convert('RGB') for ins in instances]
        answers = [ins.pop('answers') for ins in instances]
        original_questions = [ins['question'] for ins in instances]
        steps = [ins.pop('steps', []) for ins in instances]

        # Process images
        pixel_values = self.image_processor(images=images, return_tensors="pt")['pixel_values'].to(torch.bfloat16)

        all_input_ids = []
        all_labels = []

        for i, ins in enumerate(instances):
            question = ins['question']
            
            # For CoT, combine reasoning steps with the final answer
            answer_text = " ".join(steps[i]) + " " + ins['answer'] if steps[i] else ins['answer']

            # Simple concatenation for base model
            full_text = f"{question} {answer_text}{self.tokenizer.eos_token}"
            
            # Tokenize the full sequence
            input_ids = self.tokenizer(full_text, return_tensors="pt", max_length=self.tokenizer.model_max_length, truncation=True).input_ids[0]
            labels = input_ids.clone()
            
            if is_train:
                # Mask the question part, only train on the answer
                question_tokenized = self.tokenizer(question, return_tensors="pt", max_length=self.tokenizer.model_max_length, truncation=True).input_ids[0]
                question_len = len(question_tokenized)
                labels[:question_len] = -100
            else:
                # For evaluation, no labels are needed
                labels[:] = -100

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        # Pad the sequences
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_labels = torch.nn.utils.rnn.pad_sequence(all_labels, batch_first=True, padding_value=-100)
        attention_mask = padded_input_ids.ne(self.tokenizer.pad_token_id)

        return {
            'pixel_values': pixel_values,
            'input_ids': padded_input_ids,
            'attention_mask': attention_mask,
            'labels': padded_labels,
            'answers': answers,
            'original_questions': original_questions
        }

    def prepare_coconut_batch(self, instances: Sequence[Dict], is_train: bool) -> Dict[str, torch.Tensor]:
        images = [Image.open(ins.pop('image')).convert('RGB') for ins in instances]
        answers = [ins.pop('answers')for ins in instances]
        original_questions = [ins['question'] for ins in instances]
        c_thought = self.train_config.get('c_thought', 1)

        pixel_values = self.image_processor(images=images, return_tensors="pt")['pixel_values'].to(torch.bfloat16)

        all_input_ids = []
        all_labels = []

        for i, ins in enumerate(instances):
            question = ins['question']
            answer_text = ins['answer']

            # Construct input with latent tokens
            question_with_thoughts = (
                f"{question} "
                f"<start_thought>{'<thought>' * c_thought}<end_thought>"
            )

            full_text = f"{question_with_thoughts} {answer_text}{self.tokenizer.eos_token}"
            
            input_ids = self.tokenizer(full_text, return_tensors="pt", max_length=self.tokenizer.model_max_length, truncation=True).input_ids[0]
            labels = input_ids.clone()

            if is_train:
                # Mask everything before the answer
                question_part_tokenized = self.tokenizer(question_with_thoughts, return_tensors="pt", max_length=self.tokenizer.model_max_length, truncation=True).input_ids[0]
                question_len = len(question_part_tokenized)
                labels[:question_len] = -100
            else:
                labels[:] = -100

            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_labels = torch.nn.utils.rnn.pad_sequence(all_labels, batch_first=True, padding_value=-100)
        attention_mask = padded_input_ids.ne(self.tokenizer.pad_token_id)

        return {
            'pixel_values': pixel_values,
            'input_ids': padded_input_ids,
            'attention_mask': attention_mask,
            'labels': padded_labels,
            'answers': answers,
            'original_questions': original_questions
        }