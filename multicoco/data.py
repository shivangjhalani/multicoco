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

        # Temporary: Slice the dataset to only use the first 10 examples for quick evaluation.
        # Remove this line to use the full dataset again.
        if "val" in data_path: # Apply only to validation set
            self.data = self.data[:30]
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "image": os.path.join(self.data_dir, item["image"]),
            "question": item["question"],
            "answer": item["answer"],
            "answers": item.get("answers", [item.get("answer")]),
            "steps": item.get("steps", [])  # Chain of thought steps
        }

class DataCollatorForInternVL(object):
    def __init__(self, tokenizer, model, image_processor):
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.train_config = {'is_train': True}  # Default to training mode
        self.thought_token_id = tokenizer.convert_tokens_to_ids('<thought>')
        self.start_thought_id = tokenizer.convert_tokens_to_ids('<start_thought>')
        self.end_thought_id = tokenizer.convert_tokens_to_ids('<end_thought>')

    def format_question_for_eval(self, question: str, mode: str = "vanilla") -> str:
        """
        Format the question for evaluation based on the mode.
        
        Args:
            question: The original question with choices
            mode: "vanilla", "cot", or "coconut"
            
        Returns:
            Formatted question string
        """
        if mode == "cot":
            # For CoT, ask for reasoning before the answer
            formatted_question = f"{question}\n\nPlease think step by step and provide your reasoning, then give your final answer as a number (0, 1, 2, or 3)."
        else:
            # For vanilla and coconut, ask for direct answer
            formatted_question = f"{question}\n\nPlease answer with only the number (0, 1, 2, or 3) corresponding to the correct choice."
        return formatted_question

    def format_cot_answer(self, steps: list, final_answer: str) -> str:
        """
        Format the chain of thought answer for training.
        
        Args:
            steps: List of reasoning steps
            final_answer: The final answer
            
        Returns:
            Formatted answer string with reasoning
        """
        if not steps:
            return final_answer
        
        # Combine steps into reasoning
        reasoning = " ".join(steps)
        return f"{reasoning} Therefore, the answer is {final_answer}."

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        is_train = self.train_config.get('is_train', True)
        is_coconut = self.train_config.get('coconut', False)
        
        # In coconut mode, we construct a different kind of input
        if is_coconut:
            return self.prepare_coconut_batch(instances)

        # Otherwise, proceed with the standard CoT/vanilla batch preparation
        return self.prepare_cot_batch(instances, is_train)
        
    def prepare_cot_batch(self, instances: Sequence[Dict], is_train: bool) -> Dict[str, torch.Tensor]:
        images = [Image.open(ins.pop('image')).convert('RGB') for ins in instances]
        answers = [ins.pop('answers') for ins in instances]
        original_questions = [ins['question'] for ins in instances]
        steps = [ins.pop('steps', []) for ins in instances]

        if hasattr(self.model, 'dynamic_preprocess'):
            pixel_values_list, _ = self.model.dynamic_preprocess(images, image_size=self.model.config.image_size)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        else:
            pixel_values = self.image_processor(images=images, return_tensors="pt")['pixel_values'].to(torch.bfloat16)

        all_input_ids = []
        all_labels = []

        for i, ins in enumerate(instances):
            base_question = ins['question']
            
            # Determine the mode and format accordingly
            is_train = self.train_config.get('is_train', True)
            is_coconut = self.train_config.get('coconut', False)
            
            if is_train:
                if is_coconut:
                    # CoCoNUt training: Use original question, answer only (no reasoning)
                    formatted_question = base_question
                    answer = ins['answer']
                else:
                    # CoT training: Use original question, include reasoning steps
                    formatted_question = base_question
                    answer = self.format_cot_answer(steps[i], ins['answer'])
            else:
                # Evaluation mode: Format based on the mode
                eval_mode = "coconut" if is_coconut else ("cot" if steps[i] else "vanilla")
                formatted_question = self.format_question_for_eval(base_question, eval_mode)
                answer = ins['answer']
            
            question = '<img>' * self.model.config.num_image_token + '\n' + formatted_question
            conv = get_conv_template(self.model.conv_template)
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            conv.append_message(roles["human"], question)
            conv.append_message(roles["gpt"], answer)
            conversation = conv.get_prompt()

            input_ids = self.tokenizer(conversation, return_tensors="pt", padding="longest", max_length=self.tokenizer.model_max_length, truncation=True).input_ids[0]
            labels = input_ids.clone()

            if is_train:
                # For training, mask the instruction part
                sep = conv.sep + conv.roles[1] + ": "
                parts = conversation.split(sep)
                if len(parts) > 1:
                    parts[0] += sep
                    round_len = len(self.tokenizer(parts[0], add_special_tokens=False).input_ids)
                    instruction_len = len(self.tokenizer(parts[0] + parts[1], add_special_tokens=False).input_ids)
                    labels[:round_len] = -100
                    
                    if is_coconut:
                        # For CoCoNUt, also mask reasoning part if present, only train on final answer
                        # This is a simplified approach - in practice you'd want more sophisticated parsing
                        labels[instruction_len:] = -100
            else:
                # For evaluation, we don't need labels
                labels[:] = -100
            
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_labels = torch.nn.utils.rnn.pad_sequence(all_labels, batch_first=True, padding_value=-100)
        attention_mask = padded_input_ids.ne(self.tokenizer.pad_token_id)
        image_flags = (padded_input_ids == self.tokenizer.convert_tokens_to_ids('<img>')).long()

        return {
            'pixel_values': pixel_values,
            'input_ids': padded_input_ids,
            'attention_mask': attention_mask,
            'labels': padded_labels,
            'image_flags': image_flags,
            'answers': answers,
            'original_questions': original_questions,
            'steps': steps
        }

    def prepare_coconut_batch(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        images = [Image.open(ins.pop('image')).convert('RGB') for ins in instances]
        answers = [ins.pop('answers') for ins in instances]
        original_questions = [ins['question'] for ins in instances]
        
        c_thought = self.train_config.get('c_thought', 1)

        if hasattr(self.model, 'dynamic_preprocess'):
            pixel_values_list, _ = self.model.dynamic_preprocess(images, image_size=self.model.config.image_size)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        else:
            pixel_values = self.image_processor(images=images, return_tensors="pt")['pixel_values'].to(torch.bfloat16)

        all_input_ids = []
        all_labels = []

        for i, ins in enumerate(instances):
            question = ins['question']
            answer = ins['answer']
            
            # Construct the input with latent thought tokens
            question_with_thoughts = (
                '<img>' * self.model.config.num_image_token + '\n' +
                question +
                ' ' +
                '<start_thought>' +
                '<thought>' * c_thought +
                '<end_thought>'
            )
            
            conv = get_conv_template(self.model.conv_template)
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            conv.append_message(roles["human"], question_with_thoughts)
            conv.append_message(roles["gpt"], answer)
            conversation = conv.get_prompt()

            input_ids = self.tokenizer(conversation, return_tensors="pt", padding="longest", max_length=self.tokenizer.model_max_length, truncation=True).input_ids[0]
            labels = input_ids.clone()
            
            # Mask everything except the final answer
            sep = conv.sep + conv.roles[1] + ": "
            parts = conversation.split(sep)
            if len(parts) > 1:
                instruction_len = len(self.tokenizer(parts[0] + sep, add_special_tokens=False).input_ids)
                labels[:instruction_len] = -100

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_labels = torch.nn.utils.rnn.pad_sequence(all_labels, batch_first=True, padding_value=-100)
        attention_mask = padded_input_ids.ne(self.tokenizer.pad_token_id)
        image_flags = (padded_input_ids == self.tokenizer.convert_tokens_to_ids('<img>')).long()
        
        return {
            'pixel_values': pixel_values,
            'input_ids': padded_input_ids,
            'attention_mask': attention_mask,
            'labels': padded_labels,
            'image_flags': image_flags,
            'answers': answers,
            'original_questions': original_questions,
            'steps': [ins['steps'] for ins in instances]
        }