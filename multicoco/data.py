import json
import os
import sys
from typing import Sequence, Dict
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# Add the InternVL chat model path to the system path
sys.path.append('InternVL/internvl_chat')

from .utils import load_image
from internvl.conversation import get_conv_template
from internvl.train.dataset import dynamic_preprocess, build_transform
from PIL import Image

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

class DataCollatorForInternVL(object):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.conv_template = model.conv_template
        self.ignore_label_token_id = -100
        self.image_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.image_token = '<IMG_CONTEXT>'
        self.img_start_token_id = self.tokenizer.eos_token_id # InternVL uses eos_token_id as a start token for images
        self.img_end_token_id = self.tokenizer.eos_token_id # and for the end token as well

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_num_patches = []

        for i, ins in enumerate(instances):
            # 1. Load and process image
            image = Image.open(ins['image_path']).convert('RGB')
            processed_images = dynamic_preprocess(
                image,
                min_num=self.model.config.min_dynamic_patch,
                max_num=self.model.config.max_dynamic_patch,
                image_size=self.model.config.force_image_size,
                use_thumbnail=self.model.config.use_thumbnail
            )
            num_patches = len(processed_images)
            transform = build_transform(is_train=True, input_size=self.model.config.force_image_size)
            pixel_values = [transform(img) for img in processed_images]
            pixel_values = torch.stack(pixel_values)

            all_pixel_values.append(pixel_values)
            all_num_patches.append(num_patches)
            
            # 2. Construct conversation and tokenize
            question = '<img>\n' + ins['question']
            answer = ' '.join(ins['steps']) + ' ' + ins['answer'] if ins.get('steps') else ins['answer']

            # Get the prompt for the user turn to calculate its length for masking
            conv_user = self.conv_template.copy()
            conv_user.append_message(conv_user.roles[0], question)
            conv_user.append_message(conv_user.roles[1], None)
            prompt = conv_user.get_prompt()
            
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)

            # Find the placeholder token and replace it with the actual image tokens
            img_placeholder_id = self.tokenizer.convert_tokens_to_ids('<img>')
            try:
                placeholder_idx = prompt_ids.index(img_placeholder_id)
            except ValueError:
                raise ValueError("The '<img>' token was not found in the question.")

            # Insert a fixed number of image tokens, corresponding to the model's configuration
            num_image_tokens = self.model.config.num_image_token
            image_tokens = [self.image_token_id] * num_image_tokens
            
            # Replace placeholder in prompt_ids
            prompt_ids_with_image = prompt_ids[:placeholder_idx] + image_tokens + prompt_ids[placeholder_idx+1:]

            # Create final input_ids and labels
            input_ids = torch.tensor(prompt_ids_with_image + answer_ids, dtype=torch.long)
            labels = torch.tensor([self.ignore_label_token_id] * len(prompt_ids_with_image) + answer_ids, dtype=torch.long)
            
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        # 3. Pad the batch
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            all_labels, batch_first=True, padding_value=self.ignore_label_token_id)

        # 4. Create attention mask and image flags
        attention_mask = padded_input_ids.ne(self.tokenizer.pad_token_id)
        image_flags = (padded_input_ids == self.image_token_id).long()

        return {
            'pixel_values': torch.cat(all_pixel_values, dim=0),
            'input_ids': padded_input_ids,
            'attention_mask': attention_mask,
            'labels': padded_labels,
            'image_flags': image_flags,
        } 