import os
import argparse
from huggingface_hub import HfApi, HfFolder, create_repo

def main():
    """
    Uploads the contents of the script's directory to a Hugging Face repository.
    """
    parser = argparse.ArgumentParser(description="Upload checkpoints to Hugging Face Hub.")
    parser.add_argument("--repo_id", type=str, default="ThefirstM/checkpoints", help="The Hugging Face repository ID.")
    args = parser.parse_args()

    # The script should be in the directory with the files to upload.
    local_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Preparing to upload contents of '{local_dir}' to '{args.repo_id}'")

    # Check for login token
    if HfFolder.get_token() is None:
        print("Hugging Face token not found.")
        print("Please log in using 'huggingface-cli login' or by setting the HUGGING_FACE_HUB_TOKEN environment variable.")
        return

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
        print(f"Repository '{args.repo_id}' created or already exists.")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    api = HfApi()

    # Get the name of the script file to ignore it during upload.
    script_name = os.path.basename(__file__)
    
    ignore_patterns = [script_name, "README.md"]

    try:
        print(f"Uploading files to '{args.repo_id}'...")
        api.upload_folder(
            folder_path=local_dir,
            repo_id=args.repo_id,
            repo_type="model",
            ignore_patterns=ignore_patterns
        )
        print("Upload complete!")
    except Exception as e:
        print(f"An error occurred during upload: {e}")

if __name__ == "__main__":
    main() 