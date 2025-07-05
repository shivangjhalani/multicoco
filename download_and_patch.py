import os
import shutil
import subprocess
import sys
import inspect

# Use a relative path for the patch file to make it more robust
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the InternVL chat model path to the system path
# This is necessary because the model's remote code has local imports.
sys.path.insert(0, os.path.join(script_dir, 'InternVL', 'internvl_chat'))

# Install dependencies
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Installing required libraries: transformers, torch, accelerate")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch", "accelerate"])
    from transformers import AutoModelForCausalLM, AutoTokenizer


def download_and_patch_model():
    # Define paths
    hub_model_id = "OpenGVLab/InternVL3-1B"
    local_model_dir = "local_internvl_model"
    
    # Use a relative path for the patch file to make it more robust
    script_dir = os.path.dirname(os.path.abspath(__file__))
    patched_file_path = os.path.join(script_dir, "InternVL", "internvl_chat", "internvl", "model", "internvl_chat", "modeling_internvl_chat.py")
    
    # 1. Download and save the model from the Hub
    print(f"Downloading model '{hub_model_id}' from the Hub...")
    model = AutoModelForCausalLM.from_pretrained(
        hub_model_id, 
        trust_remote_code=True,
        torch_dtype='auto',
        low_cpu_mem_usage=True
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(hub_model_id, trust_remote_code=True)

    # 2. Find and patch the model's source files in the cache
    try:
        # --- Patch modeling file ---
        modeling_class = model.__class__
        modeling_file_path = inspect.getfile(modeling_class)
        patched_modeling_file_path = os.path.join(script_dir, "InternVL", "internvl_chat", "internvl", "model", "internvl_chat", "modeling_internvl_chat.py")
        
        print(f"Patching modeling file at: {modeling_file_path}")
        shutil.copyfile(patched_modeling_file_path, modeling_file_path)
        print("Modeling file patched successfully in cache.")

        # --- Patch configuration file ---
        config_class = model.config.__class__
        config_file_path = inspect.getfile(config_class)
        patched_config_file_path = os.path.join(script_dir, "InternVL", "internvl_chat", "internvl", "model", "internvl_chat", "configuration_internvl_chat.py")

        print(f"Patching configuration file at: {config_file_path}")
        shutil.copyfile(patched_config_file_path, config_file_path)
        print("Configuration file patched successfully in cache.")

        # --- Reload the model to apply patches ---
        print("Reloading model to apply patches...")
        # We need to unload the old modules to force a re-import
        if modeling_class.__module__ in sys.modules:
            del sys.modules[modeling_class.__module__]
        if config_class.__module__ in sys.modules:
            del sys.modules[config_class.__module__]

        model = AutoModelForCausalLM.from_pretrained(
            hub_model_id, 
            trust_remote_code=True,
            torch_dtype='auto',
            low_cpu_mem_usage=True
        ).cuda()
        tokenizer = AutoTokenizer.from_pretrained(hub_model_id, trust_remote_code=True)
        print("Model reloaded successfully.")

    except Exception as e:
        print(f"Error: Could not find and patch the model files dynamically: {e}")
        print("Aborting.")
        return

    # 3. Save the patched model and tokenizer
    print(f"Saving model and tokenizer to '{local_model_dir}'...")
    model.save_pretrained(local_model_dir)
    tokenizer.save_pretrained(local_model_dir)
    
    # 4. (Debug) List the files in the output directory
    print(f"\n--- Contents of '{local_model_dir}' ---")
    for root, dirs, files in os.walk(local_model_dir):
        for name in files:
            print(os.path.join(root, name))
        for name in dirs:
            print(os.path.join(root, name))
    print("--------------------------------------\n")

    print(f"Model is ready to be used from '{local_model_dir}'.")

if __name__ == "__main__":
    download_and_patch_model() 