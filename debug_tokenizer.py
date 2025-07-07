import torch
from multicoco.model import MultiCoCo

def main():
    print("--- Initializing Model and Tokenizer ---")
    model = MultiCoCo(model_id='OpenGVLab/InternVL3-1B-Pretrained')
    tokenizer = model.tokenizer
    
    print("\n--- Tokenizer Info ---")
    print(f"Tokenizer class: {tokenizer.__class__}")
    print(f"Image token: '<img>', ID: {tokenizer.convert_tokens_to_ids('<img>')}")
    
    # --- Test Case ---
    image_token_placeholder = '<img>' * 256
    question = "This is a test question."
    prompt = f"{image_token_placeholder}\n{question}"
    
    print("\n--- Testing Tokenization ---")
    print(f"Prompt starts with: {prompt[:30]}...")
    
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    
    # Count the number of image tokens
    image_token_id = 32000
    image_token_count = (input_ids == image_token_id).sum().item()
    
    print(f"\nTotal tokens in sequence: {input_ids.shape[1]}")
    print(f"Number of '<img>' tokens found: {image_token_count}")
    
    if image_token_count == 256:
        print("\n✅ SUCCESS: The tokenizer produced exactly 256 '<img>' tokens.")
    else:
        print(f"\n❌ FAILURE: Expected 256 '<img>' tokens, but found {image_token_count}.")

if __name__ == "__main__":
    main() 