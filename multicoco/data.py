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
            # Load image if present
            image = None
            if 'image_path' in item and item['image_path']:
                try:
                    image = load_image(item['image_path'], max_num=1)
                    images.append(image)
                except Exception as e:
                    print(f"Warning: Could not load image {item['image_path']}: {e}")
                    image = None
                    images.append(None)
            else:
                images.append(None)
            
            # Create text prompt
            question = item.get('question', '')
            steps = item.get('steps', [])
            answer = item.get('answer', '')
            
            # Format text with image token if image exists
            if image is not None:
                image_placeholder = '<img>' + '<IMG_CONTEXT>' * 255
                text = f"{image_placeholder}\nQuestion: {question}\nAnswer: {answer}"
            else:
                text = f"Question: {question}\nAnswer: {answer}"
                
            texts.append(text)
        
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
        
        # Create image_flags
        image_flags = torch.zeros_like(input_ids, dtype=torch.bool)
        img_token_id = self.tokenizer.convert_tokens_to_ids('<img>')
        img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        
        for i in range(input_ids.shape[0]):
            image_flags[i] = (input_ids[i] == img_token_id) | (input_ids[i] == img_context_token_id)

        # Process images - ensure we have same batch size
        pixel_values = []
        for image in images:
            if image is not None:
                pixel_values.append(image)
            else:
                # Create dummy image for text-only samples
                pixel_values.append(torch.zeros(3, 448, 448))
        
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.squeeze(1)
        
        # Create labels for training
        if self.training:
            labels = input_ids.clone()
            # Mask non-assistant tokens (simple approach - mask everything except assistant responses)
            labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            labels = None

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'image_flags': image_flags,
        }
