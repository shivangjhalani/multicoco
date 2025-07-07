import torch
from PIL import Image
import requests
from multicoco.model import MultiCoCo

def main():
    # Load a sample image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    # Initialize the model, tokenizer, and image processor
    model_id = 'OpenGVLab/InternVL3-1B-Pretrained'
    model = MultiCoCo(model_id).to(torch.bfloat16).to("cuda")
    tokenizer = model.tokenizer
    image_processor = model.image_processor
    
    model.eval()

    # --- Test Input Construction ---
    question = "What is in the image?"
    
    # The model expects 256 image tokens
    image_placeholder = '<img>' * 256
    prompt_text = f"{image_placeholder}{question}"
    
    print(f"Full prompt text (first 100 chars): {prompt_text[:100]}...")

    # Tokenize the text
    inputs = tokenizer(text=prompt_text, return_tensors="pt")
    
    # Process the image
    image_inputs = image_processor(images=image, return_tensors="pt")
    
    # Combine inputs
    inputs['pixel_values'] = image_inputs['pixel_values'].to(torch.bfloat16)
    
    # Create image_flags
    image_token_id = tokenizer.convert_tokens_to_ids('<img>')
    inputs['image_flags'] = (inputs['input_ids'] == image_token_id).long()

    # Move all inputs to the correct device
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    print("\n--- Tensor Shapes Going into Model ---")
    for name, tensor in inputs.items():
        print(f"{name}: {tensor.shape}")
        
    print(f"Number of image flags set to 1: {torch.sum(inputs['image_flags'])}")

    # --- Forward Pass ---
    print("\n--- Running Forward Pass ---")
    try:
        with torch.no_grad():
            outputs = model.model(**inputs, output_hidden_states=True)
        print("Forward pass successful!")
        print("Logits shape:", outputs.logits.shape)
    except Exception as e:
        print(f"Forward pass failed: {e}")

if __name__ == "__main__":
    main() 