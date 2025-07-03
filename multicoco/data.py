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
    def __init__(self, tokenizer_path, train_config=None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)
        self.train_config = train_config or {}
        self.image_token = "<image>"
        self.latent_tokens = {"start": "<|start-latent|>", "end": "<|end-latent|>", "latent": "<|latent|>"}

    def __call__(self, features):
        is_train = self.train_config.get('is_train', False)
        is_coconut = self.train_config.get('coconut', False)
        c_thought = self.train_config.get('c_thought', 0)
        
        pixel_values_list = [load_image(f['image_path']) for f in features]
        pixel_values = torch.cat(pixel_values_list)
        num_patches_list = [p.size(0) for p in pixel_values_list]

        full_prompts = []
        for feature in features:
            question = f"{self.image_token}\n{feature['question']}"
            
            if is_train:
                # Chain of Thought or Coconut format
                answer_steps = feature['steps'] + [feature['answer']]
                if is_coconut and c_thought > 0:
                    # Add latent thoughts between steps
                    latent_thought_str = f"{self.latent_tokens['start']}{self.latent_tokens['latent'] * c_thought}{self.latent_tokens['end']}"
                    reasoning = latent_thought_str + latent_thought_str.join(answer_steps)
                else:
                    # Standard CoT
                    reasoning = "".join(answer_steps)
                
                full_prompt = f"Question: {question}\nAnswer: {reasoning}{self.tokenizer.eos_token}"
            else:
                # For validation/inference, just the question
                full_prompt = f"Question: {question}\nAnswer: "
            
            full_prompts.append(full_prompt)

        # Tokenize prompts
        self.tokenizer.padding_side = 'left'
        prompt_inputs = self.tokenizer(full_prompts, return_tensors="pt", padding=True)
        
        input_ids = prompt_inputs.input_ids
        attention_mask = prompt_inputs.attention_mask

        # Create labels, masking out the prompt part
        if is_train:
            labels = input_ids.clone()
            # Find where the answer starts for each item in the batch
            for i, prompt in enumerate(full_prompts):
                answer_marker = "Answer: "
                answer_start_index = prompt.find(answer_marker)
                if answer_start_index != -1:
                    # Tokenize the prompt part to find its length
                    prompt_part = prompt[:answer_start_index + len(answer_marker)]
                    prompt_token_len = len(self.tokenizer(prompt_part, add_special_tokens=False).input_ids)
                    # Mask out the prompt tokens
                    labels[i, :prompt_token_len] = -100
            
            # Mask out padding tokens
            labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            labels = None

        image_flags = torch.ones(input_ids.size(0), 1, dtype=torch.long)

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'num_patches_list': num_patches_list,
            'image_flags': image_flags,
        }
