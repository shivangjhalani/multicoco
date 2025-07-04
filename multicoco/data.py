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
        
        # Get the conversation template from the model
        conv = self.conv_template.copy()
        
        # Prepare lists for batching
        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_num_patches = []

        for i, ins in enumerate(instances):
            # 1. Load and process image
            image = Image.open(ins['image_path']).convert('RGB')
            pixel_values = dynamic_preprocess(
                image,
                min_num=self.model.config.min_dynamic_patch,
                max_num=self.model.config.max_dynamic_patch,
                image_size=self.model.config.force_image_size,
                use_thumbnail=self.model.config.use_thumbnail
            )
            num_patches = pixel_values.shape[0]
            all_pixel_values.append(pixel_values)
            all_num_patches.append(num_patches)
            
            # 2. Construct conversation and tokenize
            question = ins['question']
            steps = ' '.join(ins['steps'])
            answer = ins['answer']
            
            # Format the conversation
            conv.messages = []
            conv.append_message(conv.roles[0], question)  # User's turn
            conv.append_message(conv.roles[1], None) # Assistant's turn (will be filled with answer)
            prompt = conv.get_prompt()

            # Prepend the beginning-of-sequence token if necessary
            if not prompt.startswith(self.tokenizer.bos_token):
                prompt = self.tokenizer.bos_token + prompt

            # Tokenize the prompt
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

            # --- Handle Answer and Labels ---
            # Now, let's prepare the full sequence with the answer for label creation
            full_conv = self.conv_template.copy()
            full_conv.messages = []
            full_conv.append_message(full_conv.roles[0], question)
            full_conv.append_message(full_conv.roles[1], answer)
            full_text = full_conv.get_prompt()
            if not full_text.startswith(self.tokenizer.bos_token):
                full_text = self.tokenizer.bos_token + full_text

            full_input_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
            
            # Create labels: mask out the prompt part
            labels = [self.ignore_label_token_id] * len(prompt_ids) + full_input_ids[len(prompt_ids):]
            
            # The final input_ids for the model is the full sequence
            input_ids = torch.tensor(full_input_ids, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
            
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        # 3. Pad the batch
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            all_labels, batch_first=True, padding_value=self.ignore_label_token_id)

        attention_mask = padded_input_ids.ne(self.tokenizer.pad_token_id)

        return {
            'pixel_values': torch.cat(all_pixel_values, dim=0),
            'input_ids': padded_input_ids,
            'labels': padded_labels,
            'attention_mask': attention_mask,
            'num_patches_list': all_num_patches
        } 