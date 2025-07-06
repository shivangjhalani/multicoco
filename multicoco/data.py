import os
import json
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from PIL import Image


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer,
                 image_processor: transformers.ProcessorMixin, data_dir: str):
        super(SupervisedDataset, self).__init__()
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.data[i]
        image_file = item['image']
        image_folder = self.data_dir
        
        # This is where the error was happening before. We need to handle the image path correctly.
        if image_folder is not None:
            image_file = os.path.join(image_folder, image_file)

        image = Image.open(image_file).convert('RGB')
        # Preprocess the image
        image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']

        return dict(
            image=image,
            question=item['conversations'][0]['value'],
            answer=item['conversations'][1]['value'],
        )


@dataclass
class DataCollatorForCoCo(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    model: transformers.PreTrainedModel

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        questions = [instance['question'] for instance in instances]
        answers = [instance['answer'] for instance in instances]
        images = [instance['image'] for instance in instances]
        images = torch.cat(images, dim=0)

        all_input_ids = []
        all_labels = []

        for q, a in zip(questions, answers):
            # The base model expects a specific number of `<IMG_CONTEXT>` tokens
            image_token_str = '<IMG_CONTEXT>' * self.model.num_image_token
            prompt = image_token_str + q

            # The full sequence includes the answer for the language model to learn
            full_text = prompt + ' ' + a + self.tokenizer.eos_token

            # Tokenize the full sequence
            tokenized_full = self.tokenizer(
                full_text,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )
            input_ids = tokenized_full.input_ids[0]

            # Tokenize the prompt separately to determine its length for masking
            tokenized_prompt = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )
            prompt_len = tokenized_prompt.input_ids[0].ne(self.tokenizer.pad_token_id).sum().item()

            # Create labels, masking the prompt part
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        # Pad the sequences to form a batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            all_labels, batch_first=True, padding_value=-100
        )

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            pixel_values=images,
            image_flags=torch.ones(images.shape[0], 1) # Signal that all samples have an image
        )
        return batch