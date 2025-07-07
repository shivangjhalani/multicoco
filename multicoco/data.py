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
            # For InternVL, we need to properly format the conversation
            # The question should include the <image> token where the image should be placed
            conversation = f"<image>\n{item['question']}"
            answer = item.get('answer', item.get('direct_answer', ''))
            
            return {
                'image': image,
                'conversation': conversation,
                'answer': answer
            }
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a default item if image loading fails
            return {
                'image': Image.new('RGB', (224, 224), color=(0, 0, 0)),
                'conversation': "<image>\nWhat is in this image?",
                'answer': "I cannot see the image."
            }


def collate_fn(batch, tokenizer, image_processor):
    """
    Collate function for the SupervisedDataset.
    """
    # Separate the batch into components
    images = [item['image'] for item in batch]
    conversations = [item['conversation'] for item in batch]
    answers = [item['answer'] for item in batch]
    
    # Process images
    # For InternVL, we need to process images to get pixel_values
    pixel_values = image_processor(images, return_tensors='pt')['pixel_values']
    
    # Process text for InternVL conversation format
    # InternVL expects a specific conversation format
    input_texts = []
    target_texts = []
    
    for conversation, answer in zip(conversations, answers):
        # For training, we need to format as conversation
        # InternVL format: conversation includes <image> token, followed by assistant response
        input_text = conversation
        target_text = answer
        
        input_texts.append(input_text)
        target_texts.append(target_text)
    
    # Tokenize inputs and targets
    input_encodings = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    target_encodings = tokenizer(
        target_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # For InternVL, we need to create labels for the full sequence
    # Create full conversation text for proper training
    full_conversations = []
    for conv, ans in zip(conversations, answers):
        # Create a complete conversation format that InternVL expects
        full_conv = f"{conv}\n{ans}"
        full_conversations.append(full_conv)
    
    # Tokenize full conversations for labels
    full_encodings = tokenizer(
        full_conversations,
        padding=True,
        truncation=True,
        max_length=1024,  # longer for full conversation
        return_tensors='pt'
    )
    
    # Create labels - mask input tokens, only train on response tokens
    labels = full_encodings['input_ids'].clone()
    
    # For each item in the batch, mask the input part
    for i, (conv, ans) in enumerate(zip(conversations, answers)):
        # Tokenize just the input part to know where to mask
        input_only = tokenizer(conv, add_special_tokens=False)['input_ids']
        input_len = len(input_only)
        
        # Mask the input tokens in labels (set to -100)
        if input_len < labels.shape[1]:
            labels[i, :input_len] = -100
    
    return {
        'pixel_values': pixel_values,
        'input_ids': full_encodings['input_ids'],
        'attention_mask': full_encodings['attention_mask'],
        'labels': labels,
        'questions': conversations,  # Preserve original questions for evaluation
        'answers': answers           # Preserve original answers for evaluation
    }