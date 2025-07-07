import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import os

# --- Configuration ---
MODEL_ID = "OpenGVLab/InternVL3-1B-Pretrained"
# Use an image we know exists from the project examples
IMAGE_PATH = "internvl/internvl_chat/examples/image1.jpg" 
QUESTION = "What is in this image?"

# --- Main Script ---
def debug_generate():
    """
    A self-contained script to debug the model's generate function with a single image.
    """
    print(f"Loading model: {MODEL_ID}")
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # The image processor is attached to the loaded model object
    image_processor = model.image_processor

    print(f"Loading image from: {IMAGE_PATH}")
    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: Image not found at {IMAGE_PATH}")
        print("Please ensure the path is correct relative to the 'multicoco' directory.")
        return
        
    image = Image.open(IMAGE_PATH).convert('RGB')

    # --- Data Preparation (mimicking DataCollatorForCoCo) ---
    print("Preparing data...")
    
    # 1. Create the prompt with the special image tokens
    image_token_len = model.num_image_token
    prompt = "<IMG_CONTEXT>" * image_token_len + " " + QUESTION
    
    # 2. Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    
    # 3. Process the image
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    
    # 4. Create the attention mask and image flags
    attention_mask = torch.ones_like(input_ids)
    image_flags = torch.ones(pixel_values.shape[0], dtype=torch.long)
    
    # --- Move to GPU and set correct dtype ---
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    image_flags = image_flags.cuda()

    # --- Print Shapes and Dtypes for Debugging ---
    print("\n--- Tensor Details ---")
    print(f"pixel_values: shape={pixel_values.shape}, dtype={pixel_values.dtype}, device={pixel_values.device}")
    print(f"input_ids:    shape={input_ids.shape}, dtype={input_ids.dtype}, device={input_ids.device}")
    print(f"attention_mask: shape={attention_mask.shape}, dtype={attention_mask.dtype}, device={attention_mask.device}")
    print(f"image_flags:  shape={image_flags.shape}, dtype={image_flags.dtype}, device={image_flags.device}")
    print("----------------------\n")

    # --- Call Generate ---
    print("Calling model.generate...")
    try:
        outputs = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            max_new_tokens=100,
            do_sample=False
        )
        print("SUCCESS: model.generate() completed.")
        print("Output:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
    except AssertionError as e:
        print(f"FAILURE: Caught AssertionError: {e}")
        print("This confirms the issue lies in how the inputs are prepared or passed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    debug_generate() 