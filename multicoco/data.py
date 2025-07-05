import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, Sequence
import json
from multicoco.conversation import get_conv_template


class MultiCoCoDataset(Dataset):
    def __init__(self, data_path, data_dir, is_eval=False):
        if data_path:
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = []
        
        if is_eval: # Apply only to validation/test set for quick evaluation
            self.data = self.data[:20]

        self.data_dir = data_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        choices_str = ", ".join([f"{i} : {choice}" for i, choice in enumerate(item['choices'])])
        answer = item.get("answer")
        if answer is not None and answer in item['choices']:
            answers = [str(item['choices'].index(answer))]
        else:
            answers = []

        return {
            "image": os.path.join(self.data_dir, item["image"]),
            "question": item["question"],
            "choices": item.get("choices", []),
            "choices_str": choices_str,
            "answer": answer,
            "answers": answers,
        }


class DataCollatorForInternVL(object):
    def __init__(self, tokenizer, model, image_processor):
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.image_token_id = tokenizer.convert_tokens_to_ids('<img>')
        self.num_image_tokens = model.config.num_image_token

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        is_eval = 'answer' not in instances[0] or instances[0]['answer'] is None

        images = [Image.open(ins.pop('image')).convert('RGB') for ins in instances]
        
        if hasattr(self.model, 'dynamic_preprocess'):
            pixel_values_list, _ = self.model.dynamic_preprocess(images, image_size=self.model.config.image_size)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        else:
            pixel_values = self.image_processor(images=images, return_tensors="pt")['pixel_values'].to(torch.bfloat16)

        all_input_ids = []
        all_labels = []

        conv = get_conv_template(self.model.conv_template)
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        
        for ins in instances:
            question_part = f"{ins['question']}\nChoices: {ins['choices_str']}"

            full_prompt = f"{'<img>' * self.num_image_tokens}\n{question_part}"
            
            conv.messages = []
            conv.append_message(roles["human"], full_prompt)
            
            if not is_eval:
                answer_for_training = str(ins['choices'].index(ins['answer']))
                conv.append_message(roles["gpt"], answer_for_training)
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
                    labels[:round_len] = -100
            else:
                labels[:] = -100
            
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_labels = torch.nn.utils.rnn.pad_sequence(all_labels, batch_first=True, padding_value=-100)
        attention_mask = padded_input_ids.ne(self.tokenizer.pad_token_id)
        image_flags = (padded_input_ids == self.image_token_id).long()
        
        return_dict = {
            'pixel_values': pixel_values,
            'input_ids': padded_input_ids,
            'attention_mask': attention_mask,
            'image_flags': image_flags,
            'answers': [ins['answers'] for ins in instances],
            'original_questions': [ins['question'] for ins in instances],
            'choices_str': [ins['choices_str'] for ins in instances],
            'choices': [ins['choices'] for ins in instances],
        }

        if not is_eval:
            return_dict['labels'] = padded_labels
            
        return return_dict