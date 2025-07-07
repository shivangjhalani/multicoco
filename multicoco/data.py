import os
import json
from typing import Dict, Sequence

import torch
from torch.utils.data import Dataset
from PIL import Image


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, data_dir: str):
        super(SupervisedDataset, self).__init__()
        with open(data_path, 'r') as f:
            self.data = json.load(f)[:20]  # limit for testing
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.data[i]
        image_file = item['image']
        image_path = os.path.join(self.data_dir, image_file)
        try:
            image = Image.open(image_path).convert('RGB')
            # For InternVL, we don't manually add <image> tokens
            # The model will handle image processing internally
            question = item['question']
            answer = item.get('answer', item.get('direct_answer', ''))
            
            return {
                'image': image,
                'question': question,
                'answer': answer
            }
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a default item if image loading fails
            return {
                'image': Image.new('RGB', (224, 224), color=(0, 0, 0)),
                'question': "What is in this image?",
                'answer': "I cannot see the image."
            }


def collate_fn(batch, tokenizer, image_processor):
    """
    Collate function for the SupervisedDataset.
    """
    # Separate the batch into components
    images = [item['image'] for item in batch]
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    
    # Process images for InternVL
    pixel_values = image_processor(images, return_tensors='pt')['pixel_values']
    
    # For InternVL, we need to create proper conversation format
    # The model expects text input without manual <image> tokens
    # During generation, the model will handle image placement
    
    # For training, we create input-output pairs
    input_texts = []
    for question in questions:
        # Simple question format - let the model handle image processing
        input_texts.append(question)
    
    # Tokenize questions (input)
    input_encodings = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt',
        add_special_tokens=True
    )
    
    # Tokenize answers (targets)
    target_encodings = tokenizer(
        answers,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors='pt',
        add_special_tokens=True
    )
    
    # For training, we need to create labels
    # We'll create a simple format where we concatenate question and answer
    full_texts = []
    for question, answer in zip(questions, answers):
        # Create training text: question + answer
        full_text = f"{question} {answer}"
        full_texts.append(full_text)
    
    full_encodings = tokenizer(
        full_texts,
        padding=True,
        truncation=True,
        max_length=768,
        return_tensors='pt',
        add_special_tokens=True
    )
    
    # Create labels for training - mask the question part
    labels = full_encodings['input_ids'].clone()
    
    for i, (question, answer) in enumerate(zip(questions, answers)):
        # Tokenize just the question to know where to mask
        question_tokens = tokenizer(question, add_special_tokens=False)['input_ids']
        question_len = len(question_tokens)
        
        # Mask question tokens in labels (set to -100 to ignore in loss)
        if question_len < labels.shape[1]:
            labels[i, :question_len] = -100
    
    return {
        'pixel_values': pixel_values,
        'input_ids': full_encodings['input_ids'],
        'attention_mask': full_encodings['attention_mask'],
        'labels': labels,
        'questions': questions,  # Preserve for evaluation
        'answers': answers       # Preserve for evaluation
    }