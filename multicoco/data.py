import json
import os
import sys
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# Add the InternVL chat model path to the system path
sys.path.append('InternVL/internvl_chat')

from .utils import load_image
from internvl.conversation import get_conv_template
from internvl.train.dataset import dynamic_preprocess
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

class DataCollatorForInternVL:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.conv_style = model.config.conv_style
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.image_token_id = self.tokenizer.convert_tokens_to_ids('<img>')
        self.img_start_token_id = self.tokenizer.eos_token_id # InternVL uses eos_token_id as a start token for images
        self.img_end_token_id = self.tokenizer.eos_token_id # and for the end token as well

    def __call__(self, batch):
        pixel_values_list, input_ids_list, labels_list, num_patches_list = [], [], [], []

        for item in batch:
            # 1. Load and process image
            image = Image.open(item['image_path']).convert('RGB')
            pixel_values, num_patches = dynamic_preprocess(
                image,
                min_num=self.model.config.min_dynamic_patch,
                max_num=self.model.config.max_dynamic_patch,
                image_size=self.model.config.force_image_size,
                use_thumbnail=self.model.config.use_thumbnail
            )
            pixel_values_list.append(pixel_values)
            num_patches_list.append(num_patches)
            
            # 2. Construct conversation and tokenize
            question = item['question']
            steps = ' '.join(item['steps'])
            answer = item['answer']
            
            conv = get_conv_template(self.conv_style)
            conv.append_message(conv.roles[0], question + '\\n<image>')
            conv.append_message(conv.roles[1], steps + ' ' + answer)
            
            # Tokenize conversation turns
            human_prompt = conv.messages[0][1]
            gpt_response = conv.messages[1][1]
            
            human_token_ids = self.tokenizer(human_prompt, add_special_tokens=False).input_ids
            gpt_token_ids = self.tokenizer(gpt_response, add_special_tokens=False).input_ids

            # Find and replace the image placeholder token
            image_placeholder_index = human_token_ids.index(self.image_token_id)
            
            # Create the image token sequence
            image_tokens = [self.img_start_token_id] + [self.img_context_token_id] * num_patches + [self.img_end_token_id]

            # Replace the placeholder with the actual image tokens
            input_ids = human_token_ids[:image_placeholder_index] + image_tokens + human_token_ids[image_placeholder_index+1:] + gpt_token_ids
            
            # Create labels, masking out the human prompt and image tokens
            labels = ([-100] * len(human_token_ids[:image_placeholder_index] + image_tokens + human_token_ids[image_placeholder_index+1:])) + gpt_token_ids
            
            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        # 3. Pad the batch
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100)

        attention_mask = padded_input_ids.ne(self.tokenizer.pad_token_id)

        return {
            'pixel_values': torch.stack(pixel_values_list),
            'input_ids': padded_input_ids,
            'labels': padded_labels,
            'attention_mask': attention_mask,
            'num_patches_list': num_patches_list
        } 