import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import os
from copy import deepcopy

class SupervisedDataset(Dataset):
    def __init__(self, data_args, processor, is_eval=False):
        super(SupervisedDataset, self).__init__()
        self.data_path = data_args.eval_data_path if is_eval else data_args.train_data_path
        self.data = json.load(open(self.data_path))
        self.data_dir = data_args.data_dir
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        image_file = os.path.join(self.data_dir, item['image'])
        image = Image.open(image_file).convert('RGB')
        
        question = item['question']
        answer = item.get('rationale', item.get('answer', '')) # For CoT, rationale is the answer

        # Construct the conversation in the format required by apply_chat_template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        return {
            "image": image,
            "messages": messages,
            "answer": answer,
            # Pass metadata for evaluation
            "original_question": item['question'],
            "original_answer": item.get('answer', ''),
            "question_id": item.get('question_id', idx) # Use index as fallback qid
        }

class DataCollatorForCoCo(object):
    def __init__(self, processor, data_args):
        self.processor = processor
        self.cot = data_args.cot

    def __call__(self, instances):
        # Separate images and conversations
        images = [ins.pop("image") for ins in instances]
        
        # We need to add the assistant's response to the conversation for training
        conversations = []
        for ins in instances:
            conv = deepcopy(ins['messages'])
            conv.append({"role": "assistant", "content": [{"type": "text", "text": ins['answer']}]})
            conversations.append(conv)
        
        # Use processor to handle both image processing and text tokenization
        # This will create all necessary inputs including pixel_values, input_ids, attention_mask,
        # and the model-specific image_flags.
        # Padding is handled automatically.
        inputs = self.processor.apply_chat_template(
            conversations,
            images=images,
            tokenize=True,
            padding=True,
            return_tensors="pt"
        )

        # Create labels for language modeling. We need to mask the prompt part.
        # The prompt is the conversation up to the assistant's turn.
        # We can get the prompt length by tokenizing the conversation without the assistant's part.
        
        # Create a copy of input_ids for labels
        labels = inputs['input_ids'].clone()

        # To mask the prompt, we find where the assistant's response starts.
        # The chat template adds special tokens. A common pattern is `...user...<end_of_turn>assistant...`
        # We can find the start of the assistant's turn by finding the token ids for `<|im_start|>assistant`.
        # However, a simpler and more robust way is to tokenize the user part of the conversation separately.
        
        prompt_lengths = []
        for i in range(len(instances)):
            # Get the user-only part of the conversation
            user_conv = instances[i]['messages']
            
            # Tokenize just the prompt part to find its length
            # Note: The image is not needed here as we only need the text token length
            prompt_inputs = self.processor.apply_chat_template(
                user_conv,
                images=None, 
                tokenize=True,
                add_generation_prompt=True # This is key to get the tokens that prompt the assistant
            )
            prompt_len = len(prompt_inputs['input_ids'])
            prompt_lengths.append(prompt_len)
            
            # Mask the prompt tokens
            labels[i, :prompt_len] = -100

        inputs['labels'] = labels
        inputs['prompt_lengths'] = torch.tensor(prompt_lengths)
        
        # Pass metadata through for the evaluation loop
        if 'question_id' in instances[0]:
            inputs['question_ids'] = [ins['question_id'] for ins in instances]
            inputs['original_questions'] = [ins['original_question'] for ins in instances]
            inputs['original_answers'] = [ins['original_answer'] for ins in instances]

        return inputs