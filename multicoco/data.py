import json
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from .utils import load_image

class MultiCoCoDataset(Dataset):
    def __init__(self, data_path, data_dir):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.data_dir, item['image'])
        question = item['question']
        steps = item.get('steps', [])
        answer = item['answer']
        return {
            "image_path": image_path,
            "question": question,
            "steps": steps,
            "answer": answer
        }

class DataCollatorForMultiCoCo:
    def __init__(self, tokenizer, train_config=None):
        self.tokenizer = tokenizer
        self.train_config = train_config or {}
        self.max_length = self.train_config.get('max_length', 2048)
        self.training = self.train_config.get('is_train', False)
        self.image_token = "<img>"
        self.latent_tokens = {"start": "<|start-latent|>", "end": "<|end-latent|>", "latent": "<|latent|>"}

    def __call__(self, batch):
        texts = []
        images = []
        
        for item in batch:
            # Process the conversation to create proper image token format
            conversation = item['conversations']
            text = ""
            
            for turn in conversation:
                if turn['from'] == 'human':
                    # Check if this turn has an image
                    if 'image' in item and item['image'] is not None:
                        # Format with proper image tokens: <img> + 255 <IMG_CONTEXT> tokens
                        img_context_tokens = "<IMG_CONTEXT>" * 255
                        image_placeholder = f"<img>{img_context_tokens}"
                        text += f"<|im_start|>user\n{image_placeholder}{turn['value']}<|im_end|>\n"
                        images.append(item['image'])
                    else:
                        text += f"<|im_start|>user\n{turn['value']}<|im_end|>\n"
                elif turn['from'] == 'gpt':
                    text += f"<|im_start|>assistant\n{turn['value']}<|im_end|>\n"
            
            texts.append(text)
        
        # Ensure all batches have the same number of images (pad with None if needed)
        max_images = max(len([img for img in images if img is not None]), 1)
        while len(images) < len(batch):
            images.append(None)
        
        # Tokenize texts
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        
        # Process images
        pixel_values = []
        for img in images:
            if img is not None:
                # Load and preprocess image
                processed_img = load_image(img, max_num=1)
                pixel_values.append(processed_img)
            else:
                # Create dummy image if None
                dummy_img = torch.zeros((3, 448, 448))
                pixel_values.append(dummy_img)
        
        if pixel_values:
            pixel_values = torch.stack(pixel_values)
        else:
            pixel_values = torch.zeros((len(batch), 3, 448, 448))
        
        # Create labels for training (shift input_ids by one position)
        if self.training:
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            labels = None

        # Create image_flags to indicate presence of images
        image_flags = torch.ones(input_ids.size(0), 1, dtype=torch.long)

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'image_flags': image_flags,
        }
