import os
import json
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from PIL import Image


def preprocess(image, question, answer, image_processor, tokenizer):
    """
    Preprocesses a single data sample.

    Args:
        image (PIL.Image): The input image.
        question (str): The question text.
        answer (str): The answer text.
        image_processor: The Hugging Face image processor.
        tokenizer: The Hugging Face tokenizer.

    Returns:
        A dictionary containing the processed data.
    """
    # Prepare the text prompt
    image_token_len = image_processor.num_image_tokens
    prompt = "<img>" * image_token_len + " " + question

    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids

    # Process the image
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    
    # The 'labels' are the tokenized answer
    # This is what the model will learn to predict
    labels = tokenizer(answer, return_tensors='pt').input_ids if answer is not None else None

    return {
        "input_ids": input_ids.squeeze(0),
        "pixel_values": pixel_values.squeeze(0),
        "labels": labels.squeeze(0) if labels is not None else None,
        "image_flags": torch.tensor([1], dtype=torch.long)
    }

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer,
                 image_processor: transformers.ProcessorMixin, data_dir: str):
        super(SupervisedDataset, self).__init__()
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # Load the data item
        item = self.data[i]

        # Process the image
        image_file = item['image']
        image_path = os.path.join(self.data_dir, image_file)
        try:
            image = Image.open(image_path).convert('RGB')
        except (FileNotFoundError, OSError) as e:
            print(f"Warning: Could not open image file {image_path}. Skipping. Error: {e}")
            # Return the next item to avoid crashing the whole batch
            return self.__getitem__((i + 1) % len(self))
            
        # Extract question and answer from the correct keys
        question = item['question']
        answer = item.get('answer') # Use .get for safety, might not be present

        # Preprocess using the shared function
        return preprocess(
            image=image,
            question=question,
            answer=answer,
            image_processor=self.image_processor,
            tokenizer=self.tokenizer
        )


@dataclass
class DataCollatorForCoCo(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    model: transformers.PreTrainedModel

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        questions = [instance['question'] for instance in instances]
        answers = [instance['answer'] for instance in instances]
        images = [instance['image'] for instance in instances]
        images = torch.cat(images, dim=0)

        all_input_ids = []
        all_labels = []

        for q, a in zip(questions, answers):
            # The base model expects a specific number of `<IMG_CONTEXT>` tokens
            image_token_str = '<IMG_CONTEXT>' * self.model.num_image_token
            prompt = image_token_str + q

            # The full sequence includes the answer for the language model to learn
            full_text = prompt + ' ' + a + self.tokenizer.eos_token

            # Tokenize the full sequence
            tokenized_full = self.tokenizer(
                full_text,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )
            input_ids = tokenized_full.input_ids[0]

            # Tokenize the prompt separately to determine its length for masking
            tokenized_prompt = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )
            prompt_len = tokenized_prompt.input_ids[0].ne(self.tokenizer.pad_token_id).sum().item()

            # Create labels, masking the prompt part
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        # Pad the sequences to form a batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            all_labels, batch_first=True, padding_value=-100
        )

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            pixel_values=images,
            image_flags=torch.ones(images.shape[0], 1) # Signal that all samples have an image
        )
        return batch