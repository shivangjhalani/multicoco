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
        self.training = self.train_config.get('is_train', True)  # Default to training

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

            pixel_values = None
            if image_path:
                try:
                    pixel_values = load_image(image_path, max_num=1)
                except Exception as e:
                    print(f"Warning: Could not load image {image_path}, skipping sample: {e}")
                    continue
            else:
                continue

            question = item.get('question', '')
            answer = item.get('answer', '')
            conversations = [
                {'from': 'human', 'value': f"\nQuestion: {question}\nAnswer:"},
                {'from': 'gpt', 'value': f" {answer}"}
            ]

            current_input_ids = []
            current_labels = []

            current_input_ids.append(image_placeholder_tokens)
            current_labels.append(torch.full(image_placeholder_tokens.shape, -100, dtype=torch.long))

            for turn in conversations:
                role = turn.get('from')
                value = turn.get('value', '')

                value_ids = self.tokenizer(value, add_special_tokens=False).input_ids
                value_ids = torch.tensor(value_ids, dtype=torch.long)

                current_input_ids.append(value_ids)

                if role == 'human':
                    current_labels.append(torch.full(value_ids.shape, -100, dtype=torch.long))
                elif role == 'gpt':
                    current_labels.append(value_ids.clone())

            pixel_values_list.append(pixel_values)
            input_ids_list.append(torch.cat(current_input_ids))
            labels_list.append(torch.cat(current_labels))

        if not pixel_values_list:
            return {
                'pixel_values': torch.empty(0, 3, 448, 448, dtype=torch.bfloat16),
                'input_ids': torch.empty(0, 0, dtype=torch.long),
                'attention_mask': torch.empty(0, 0, dtype=torch.long),
                'labels': torch.empty(0, 0, dtype=torch.long),
                'image_flags': torch.empty(0, 1, dtype=torch.bool),
            }

        max_len = max(len(ids) for ids in input_ids_list)
        if self.max_length and max_len > self.max_length:
            max_len = self.max_length

        padded_input_ids = torch.full((len(input_ids_list), max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        padded_labels = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(input_ids_list), max_len), dtype=torch.long)

        for i, ids in enumerate(input_ids_list):
            len_ids = min(len(ids), max_len)
            padded_input_ids[i, :len_ids] = ids[:len_ids]
            padded_labels[i, :len_ids] = labels_list[i][:len_ids]
            attention_mask[i, :len_ids] = 1

        pixel_values = torch.cat(pixel_values_list, dim=0)
        batch_size = pixel_values.shape[0]
        image_flags = torch.ones(batch_size, 1, dtype=torch.bool)

        return {
            'pixel_values': pixel_values,
            'input_ids': padded_input_ids,
            'attention_mask': attention_mask,
            'labels': padded_labels,
            'image_flags': image_flags,
        }
