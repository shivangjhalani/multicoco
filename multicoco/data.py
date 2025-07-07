import os
import json
from typing import Dict, Sequence

import torch
from torch.utils.data import Dataset
from PIL import Image


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, data_dir: str):
        super(SupervisedDataset, self).__init__()
        with open(data_path, 'r') as f:
            self.data = json.load(f)[:20]
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.data[i]
        image_file = item['image']
        image_path = os.path.join(self.data_dir, image_file)
        try:
            image = Image.open(image_path).convert('RGB')
        except (FileNotFoundError, OSError) as e:
            print(f"Warning: Could not open image file {image_path}. Skipping. Error: {e}")
            return self.__getitem__((i + 1) % len(self))
        
        rationale = item.get('rationale', '') 
        return dict(image=image, question=item['question'], answer=item['answer'], rationale=rationale)


class DataCollatorForCoCo(object):
    def __init__(self, tokenizer, image_processor, cot=False):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.cot = cot

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        full_conversations_text = []
        prompts_for_len_check = []
        
        questions = [instance['question'] for instance in instances]
        answers = [instance['answer'] for instance in instances]
        original_questions = [instance.get('original_question') for instance in instances]
        question_ids = [instance.get('question_id') for instance in instances]

        for instance in instances:
            question = instance['question']
            answer = instance['answer']
            
            user_content_str = f"<img>\n{question}"

            if self.cot:
                user_content_str += " Let's think step by step."
                full_answer = instance.get('rationale', '') + f" The answer is {answer}"
            else:
                user_content_str += " The answer is"
                full_answer = answer

            full_messages = [{'role': 'user', 'content': user_content_str}, {'role': 'assistant', 'content': full_answer}]
            full_conv_str = self.tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
            full_conversations_text.append(full_conv_str + self.tokenizer.eos_token)

            prompt_messages = [{'role': 'user', 'content': user_content_str}]
            prompt_str = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            prompts_for_len_check.append(prompt_str)

        data = self.tokenizer(text=full_conversations_text, return_tensors="pt", padding=True)
        
        images = [instance['image'] for instance in instances]
        image_data = self.image_processor(images=images, return_tensors="pt")
        data['pixel_values'] = image_data['pixel_values']

        image_token_id = self.tokenizer.convert_tokens_to_ids('<img>')
        image_flags = (data['input_ids'] == image_token_id).long()
        data['image_flags'] = image_flags
        
        prompt_tokenized = self.tokenizer(text=prompts_for_len_check, return_tensors="pt", padding=True)
        prompt_lengths = prompt_tokenized.attention_mask.sum(dim=1)

        labels = data['input_ids'].clone()
        for i in range(len(labels)):
            labels[i, :prompt_lengths[i]] = -100
        
        labels[data['input_ids'] == self.tokenizer.pad_token_id] = -100
        data['labels'] = labels

        data['question_ids'] = question_ids
        data['questions'] = questions
        data['answers'] = answers
        data['original_questions'] = original_questions
        data['num_items_in_batch'] = len(instances)

        return data