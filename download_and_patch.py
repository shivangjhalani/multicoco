import os
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_and_patch_model():
    # Define paths
    hub_model_id = "OpenGVLab/InternVL3-1B"
    local_model_dir = "local_internvl_model"
    patched_file_path = "multicoco/InternVL/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py"
    
    # 1. Download and save the model from the Hub
    print(f"Downloading model '{hub_model_id}' from the Hub...")
    model = AutoModelForCausalLM.from_pretrained(hub_model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(hub_model_id, trust_remote_code=True)

    print(f"Saving model and tokenizer to '{local_model_dir}'...")
    model.save_pretrained(local_model_dir)
    tokenizer.save_pretrained(local_model_dir)
    
    # 2. Find the target directory for the patch
    # The target file is inside a nested directory created by save_pretrained, often with a hash name.
    # We need to find the correct subdirectory.
    target_dir = None
    for root, dirs, files in os.walk(local_model_dir):
        if "modeling_internvl_chat.py" in files:
            target_dir = root
            break
            
    if not target_dir:
        print("Error: Could not find the target directory for patching.")
        return

    # 3. Apply the patch
    destination_path = os.path.join(target_dir, "modeling_internvl_chat.py")
    print(f"Patching file at: {destination_path}")
    shutil.copyfile(patched_file_path, destination_path)
    
    print("Patch applied successfully.")
    print(f"Model is ready to be used from '{local_model_dir}'.")

if __name__ == "__main__":
    download_and_patch_model() 