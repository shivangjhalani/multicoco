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
                        # Use single <img> token - let model handle internal mapping
                        text += f"<img>\n{turn['value']}\n"
                        images.append(item['image'])
                    else:
                        text += f"{turn['value']}\n"
                elif turn['from'] == 'gpt':
                    text += f"{turn['value']}\n"
            
            texts.append(text.strip())
        
        # If no images in batch, pad with None
        while len(images) < len(texts):
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
        for image in images:
            if image is not None:
                pixel_values.append(image)
            else:
                # Create dummy image for text-only samples
                pixel_values.append(torch.zeros(3, 448, 448))
        
        pixel_values = torch.stack(pixel_values)
        
        # Create labels for training
        if self.training:
            labels = input_ids.clone()
            # Mask non-assistant tokens (simple approach - mask everything except assistant responses)
            labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            labels = None

        # Image flags: simple batch-level flag indicating presence of images
        image_flags = torch.ones(input_ids.size(0), 1, dtype=torch.long)

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'image_flags': image_flags,
        }
