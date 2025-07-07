import os
import json
from typing import Dict, Sequence

import torch
from torch.utils.data import Dataset
from PIL import Image
import transformers


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, data_dir: str, cot=False, coconut=False):
        super(SupervisedDataset, self).__init__()
        self.data = json.load(open(data_path))
        self.data_dir = data_dir
        self.cot = cot
        self.coconut = coconut

        if self.coconut:
            # For the coconut stage, we need to load the rationales from the CoT stage.
            # We assume they are saved in a predictable location.
            cot_predictions_path = "multicoco/aokvqa-cot/all_results.json"
            if not os.path.exists(cot_predictions_path):
                raise FileNotFoundError(
                    f"Coconut stage requires CoT predictions, but file not found at: {cot_predictions_path}"
                )
            
            cot_preds = json.load(open(cot_predictions_path))
            # Create a mapping from question_id to rationale for quick lookup
            self.rationales = {item['question_id']: item['prediction'] for item in cot_preds}


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.data[i]
        image_path = os.path.join(self.data_dir, item['image_path'])
        image = Image.open(image_path).convert('RGB')
        
        question = item['question']
        
        if self.coconut:
            question_id = item['question_id']
            rationale = self.rationales.get(question_id, "") # Get CoT rationale
            # Prepend the rationale to the question for the coconut stage
            question = f"{question} {rationale}"

        return {
            'image': image,
            'question': question,
            'answer': item.get('direct_answers', [''])[0],
            'question_id': item.get('question_id'),
            'original_question': item['question'],
            'rationale': item.get('rationale', '')
        }


class DataCollatorForCoCo(object):
    def __init__(self, tokenizer, image_processor, cot=False):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.cot = cot

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        # This collator is now significantly different. It manually constructs the input tensors.
        # We are no longer using `apply_chat_template`.

        image_token_id = self.tokenizer.convert_tokens_to_ids('<img>')
        im_start_token_id = self.tokenizer.bos_token_id
        im_end_token_id = self.tokenizer.eos_token_id
        
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        # Keep metadata to pass through
        questions = []
        answers = []
        original_questions = []
        question_ids = []

        for instance in instances:
            question = instance['question']
            answer = instance['answer']

            if self.cot:
                prompt_text = f"{question} Let's think step by step."
                full_answer_text = instance.get('rationale', '') + f" The answer is {answer}"
            else:
                prompt_text = f"{question} The answer is"
                full_answer_text = answer
                
            prompt_tokens = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
            answer_tokens = self.tokenizer(full_answer_text, add_special_tokens=False).input_ids

            # [BOS] 256 * <img> <prompt_text> <answer_text> [EOS]
            input_ids = [im_start_token_id] + [image_token_id] * 256 + prompt_tokens + answer_tokens + [im_end_token_id]
            
            # Labels: mask out everything that isn't the answer
            labels = [-100] * (1 + 256 + len(prompt_tokens)) + answer_tokens + [im_end_token_id]

            batch_input_ids.append(torch.tensor(input_ids))
            batch_labels.append(torch.tensor(labels))
            batch_attention_mask.append(torch.ones(len(input_ids), dtype=torch.long))

            # Store metadata
            questions.append(question)
            answers.append(answer)
            original_questions.append(instance.get('original_question'))
            question_ids.append(instance.get('question_id'))

        # Pad the batches
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            batch_labels, batch_first=True, padding_value=-100
        )
        padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
            batch_attention_mask, batch_first=True, padding_value=0
        )

        # Create image_flags
        image_flags = (padded_input_ids == image_token_id).long()

        # Process images
        images = [instance['image'] for instance in instances]
        image_data = self.image_processor(images=images, return_tensors="pt")

        return {
            'input_ids': padded_input_ids,
            'labels': padded_labels,
            'attention_mask': padded_attention_mask,
            'pixel_values': image_data['pixel_values'],
            'image_flags': image_flags,
            'question_ids': question_ids,
            'questions': questions,
            'answers': answers,
            'original_questions': original_questions,
            'num_items_in_batch': len(instances)
        }