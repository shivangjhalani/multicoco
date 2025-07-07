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
    """Collate examples for supervised fine-tuning by applying the model's chat template."""

    def __init__(self, tokenizer, cot=False):
        self.processor = tokenizer # In our case, the tokenizer is the processor
        self.cot = cot

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        images = [instance['image'] for instance in instances]
        prompts_for_len_check = []
        full_conversations = []

        for instance in instances:
            question = instance['question']
            answer = instance['answer']
            
            if self.cot:
                full_answer = instance.get('rationale', '') + f" The answer is {answer}"
                user_content = [{"type": "image"}, {"type": "text", "text": f"{question} Let's think step by step."}]
            else: # Vanilla
                full_answer = answer
                user_content = [{"type": "image"}, {"type": "text", "text": f"{question} The answer is"}]

            # Messages for the full conversation (prompt + response)
            full_messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": full_answer}]}
            ]
            full_conversations.append(self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False))

            # Messages for just the prompt, to calculate its length for masking labels
            prompt_messages = [
                {"role": "user", "content": user_content}
            ]
            # `add_generation_prompt=True` adds the 'assistant' role to prime the model for generation,
            # which is what we need to mask correctly.
            prompts_for_len_check.append(self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True))

        # Tokenize full conversations for model input
        batch = self.processor(text=full_conversations, images=images, padding=True, return_tensors='pt')
        
        # Tokenize prompts to get their length
        prompt_tokenized = self.processor(text=prompts_for_len_check, padding=True, return_tensors='pt')
        prompt_lengths = torch.sum(prompt_tokenized.attention_mask, dim=1)

        # Create labels and mask out the prompt part
        labels = batch.input_ids.clone()
        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, :prompt_len] = -100
        
        # Also mask padding in labels
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        batch['labels'] = labels

        # Keep original questions and answers for evaluation
        batch['original_questions'] = [instance['question'] for instance in instances]
        batch['answers'] = [instance['answer'] for instance in instances]

        return batch