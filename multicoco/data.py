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
        # The new collator will handle the conversation formatting.
        # We just need to pass the raw parts.
        conversations = item.get('conversations', [])
        return {
            "image_path": image_path,
            "conversations": conversations
        }

class DataCollatorForMultiCoCo:
    def __init__(self, tokenizer, train_config=None):
        self.tokenizer = tokenizer
        self.train_config = train_config or {}
        self.max_length = self.train_config.get('max_length', 2048)
        self.training = self.train_config.get('is_train', True) # Default to training

    def __call__(self, batch):
        pixel_values_list = []
        input_ids_list = []
        labels_list = []
        
        img_token_id = self.tokenizer.convert_tokens_to_ids('<img>')
        img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        image_placeholder_tokens = torch.tensor(
            [img_token_id] + [img_context_token_id] * 255, 
            dtype=torch.long
        )
        
        for item in batch:
            image_path = item.get('image_path')
            conversations = item.get('conversations', [])
            
            # Load image
            if image_path:
                try:
                    pixel_values = load_image(image_path, max_num=1)
                    pixel_values_list.append(pixel_values)
                except Exception as e:
                    print(f"Warning: Could not load image {image_path}, skipping sample: {e}")
                    continue
            
            # Process conversation
            current_input_ids = []
            current_labels = []
            
            for i, turn in enumerate(conversations):
                role = turn.get('from')
                value = turn.get('value', '')

                # Add image placeholder tokens before the first user message
                if role == 'human' and i == 0:
                    current_input_ids.append(image_placeholder_tokens)
                    current_labels.append(torch.full(image_placeholder_tokens.shape, -100, dtype=torch.long))

                # Tokenize message value
                # We add a space before the value for the tokenizer.
                # This is a common practice for sentencepiece-based tokenizers.
                value_ids = self.tokenizer(f" {value}", add_special_tokens=False).input_ids
                value_ids = torch.tensor(value_ids, dtype=torch.long)
                
                current_input_ids.append(value_ids)

                if role == 'human':
                    # Mask human prompts
                    current_labels.append(torch.full(value_ids.shape, -100, dtype=torch.long))
                elif role == 'gpt':
                    # Do not mask assistant responses
                    current_labels.append(value_ids.clone())
            
            input_ids_list.append(torch.cat(current_input_ids))
            labels_list.append(torch.cat(current_labels))

        # Pad sequences
        # Manually pad because we are dealing with lists of tensors
        max_len = max(len(ids) for ids in input_ids_list)
        
        padded_input_ids = torch.full((len(input_ids_list), max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        padded_labels = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(input_ids_list), max_len), dtype=torch.long)
        
        for i, ids in enumerate(input_ids_list):
            len_ids = len(ids)
            padded_input_ids[i, :len_ids] = ids
            padded_labels[i, :len_ids] = labels_list[i]
            attention_mask[i, :len_ids] = 1

        # Stack pixel values
        pixel_values = torch.cat(pixel_values_list, dim=0)
        
        # Create image_flags
        batch_size = pixel_values.shape[0]
        image_flags = torch.ones(batch_size, 1, dtype=torch.bool)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': padded_input_ids,
            'attention_mask': attention_mask,
            'labels': padded_labels,
            'image_flags': image_flags,
        }
