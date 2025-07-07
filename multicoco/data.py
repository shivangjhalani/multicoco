import os
from typing import Dict, Sequence

import torch
from torch.utils.data import Dataset
from PIL import Image
import json


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, data_dir: str):
        super(SupervisedDataset, self).__init__()
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # Load the data item
        item = self.data[i]

        # Process the image
        image_file = item['image']
        image_path = os.path.join(self.data_dir, image_file)
        try:
            image = Image.open(image_path).convert('RGB')
        except (FileNotFoundError, OSError) as e:
            print(f"Warning: Could not open image file {image_path}. Skipping. Error: {e}")
            # Return the next item to avoid crashing the whole batch
            return self.__getitem__((i + 1) % len(self))
            
        return dict(
            image=image,
            question=item['question'],
            answer=item['answer']
        )


class DataCollatorForCoCo(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer, model, image_processor):
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        pixel_values_list, input_ids_list, labels_list = [], [], []

        for instance in instances:
            image = instance['image']
            question = instance['question']
            answer = instance['answer']
            
            # Preprocess each sample
            image_token_len = self.model.num_image_token
            prompt = "<IMG_CONTEXT>" * image_token_len + " " + question

            # Tokenize prompt and answer
            input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids
            labels = self.tokenizer(answer, return_tensors='pt').input_ids
            
            # Process image
            pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
            
            pixel_values_list.append(pixel_values)
            input_ids_list.append(input_ids)
            labels_list.append(labels)

        # Pad the sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [ids.squeeze(0) for ids in input_ids_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [l.squeeze(0) for l in labels_list],
            batch_first=True,
            padding_value=-100 # Use -100 to ignore padding in loss calculation
        )
        
        # Create attention mask
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        
        # Concatenate pixel values
        pixel_values = torch.cat(pixel_values_list, dim=0)
        
        # Create image_flags
        image_flags = torch.ones(pixel_values.shape[0], dtype=torch.long)

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            image_flags=image_flags,
        )