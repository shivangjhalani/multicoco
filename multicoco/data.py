import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, Sequence
import json
from multicoco.conversation import get_conv_template

class MultiCoCoDataset(Dataset):
    def __init__(self, data_path, data_dir):
        if data_path:
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = []
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "image": os.path.join(self.data_dir, item["image"]),
            "question": item["question"],
            "answer": item["answer"],
            "answers": item.get("answers", [item.get("answer")])
        }

class DataCollatorForInternVL(object):
    def __init__(self, tokenizer, model, image_processor):
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.image_token_id = tokenizer.convert_tokens_to_ids('<img>')
        self.num_image_tokens = model.config.num_image_token
        self.train_config = {}

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        images = [Image.open(ins.pop('image')).convert('RGB') for ins in instances]
        answers = [ins.pop('answers') for ins in instances]
        original_questions = [ins['question'] for ins in instances]

        if hasattr(self.model, 'dynamic_preprocess'):
            pixel_values_list, _ = self.model.dynamic_preprocess(images, image_size=self.model.config.image_size)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        else:
            pixel_values = self.image_processor(images=images, return_tensors="pt")['pixel_values'].to(torch.bfloat16)

        all_input_ids = []
        all_labels = []

        is_eval = not self.train_config.get('is_train', True)

        for ins in instances:
            question = '<img>' * self.num_image_tokens + '\n' + ins['question']
            if is_eval:
                question += "\n\nAnswer with the option number only."
            
            answer = ins['answer']
            conv = get_conv_template(self.model.conv_template)
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            conv.append_message(roles["human"], question)
            
            if not is_eval:
                conv.append_message(roles["gpt"], answer)
            else:
                conv.append_message(roles["gpt"], None)

            conversation = conv.get_prompt()

            input_ids = self.tokenizer(conversation, return_tensors="pt", padding="longest", max_length=self.tokenizer.model_max_length, truncation=True).input_ids[0]
            labels = input_ids.clone()

            if not is_eval:
                sep = conv.sep + conv.roles[1] + ": "
                parts = conversation.split(sep)
                if len(parts) > 1:
                    parts[0] += sep
                    round_len = len(self.tokenizer(parts[0], add_special_tokens=False).input_ids)
                    instruction_len = len(self.tokenizer(parts[0] + parts[1], add_special_tokens=False).input_ids)
                    labels[:round_len] = -100
                    if len(labels) > instruction_len:
                        labels[instruction_len:] = -100
            else:
                labels[:] = -100
            
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_labels = torch.nn.utils.rnn.pad_sequence(all_labels, batch_first=True, padding_value=-100)
        attention_mask = padded_input_ids.ne(self.tokenizer.pad_token_id)
        image_flags = (padded_input_ids == self.image_token_id).long()

        return {
            'pixel_values': pixel_values,
            'input_ids': padded_input_ids,
            'attention_mask': attention_mask,
            'labels': padded_labels,
            'image_flags': image_flags,
            'answers': answers,
            'original_questions': original_questions
        }