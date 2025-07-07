import torch
import json
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. Load Model and Processors ---
    model_path = 'OpenGVLab/InternVL3-1B-Pretrained'
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # The image processor must be loaded separately for the pretrained model
    image_processor = CLIPImageProcessor.from_pretrained(model_path)

    # --- 2. Manually set the image context token ID ---
    # This was a key finding from our previous debugging
    IMG_CONTEXT_TOKEN_ID = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    model.img_context_token_id = IMG_CONTEXT_TOKEN_ID
    print(f"Set model.img_context_token_id to: {model.img_context_token_id}")

    # --- 3. Load a single data sample ---
    with open('data/aokvqa_test.json', 'r') as f:
        aokvqa_data = json.load(f)
    
    # Let's take the first sample
    sample = aokvqa_data[0]
    image_path = f"{sample['image']}"
    question = sample['question']
    
    image = Image.open(image_path).convert('RGB')

    # --- 4. Replicate the data processing logic ---
    # This mimics the logic in data.py and trainer.py
    
    # From data.py: Build the prompt
    # The base model requires the <IMG_CONTEXT> token to know where to place the image.
    # The vision tower produces 256 tokens, so we need 256 placeholders.
    image_token_placeholder = '<IMG_CONTEXT>' * 256
    prompt = f"{image_token_placeholder}\n{question} The answer is"
    
    # Process image and text
    pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
    pixel_values = pixel_values.to(torch.bfloat16).to(device)
    
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    print(f"--- Input ---")
    print(f"Image Path: {image_path}")
    print(f"Question: {question}")
    print(f"Full Prompt being tokenized: '{prompt}'")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Pixel Values shape: {pixel_values.shape}")
    print(f"Pixel Values dtype: {pixel_values.dtype}")
    
    # --- 5. Run Generation ---
    with torch.no_grad():
        outputs = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids), # Simple attention mask for one sample
            max_new_tokens=100,
            do_sample=False,
        )

    # --- 6. Decode and Print Output ---
    # The previous slice was incorrect. It seems this model's generate function
    # returns *only* the new tokens, not the full sequence.
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    print(f"\n--- Output ---")
    print(f"Raw output tokens shape: {outputs.shape}")
    print(f"Generated Text: '{generated_text}'")


if __name__ == "__main__":
    main() 