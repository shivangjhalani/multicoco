import os
import argparse
import fnmatch
from huggingface_hub import HfApi, HfFolder, create_repo

def list_files_to_upload(local_dir, ignore_patterns):
    """
    Lists files in a directory that will be uploaded, excluding ignored files.
    """
    files_to_upload = []
    for root, _, filenames in os.walk(local_dir):
        for filename in filenames:
            # Check if the file should be ignored
            if any(fnmatch.fnmatch(filename, pattern) for pattern in ignore_patterns):
                continue
            
            # Get the relative path to be shown to the user
            relative_path = os.path.relpath(os.path.join(root, filename), local_dir)
            files_to_upload.append(relative_path)
    return files_to_upload

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

    # Get the name of the script file to ignore it during upload.
    script_name = os.path.basename(__file__)
    
    ignore_patterns = [script_name, "README.md"]

    # List files for user confirmation
    files_to_upload = list_files_to_upload(local_dir, ignore_patterns)

    if not files_to_upload:
        print("No files to upload.")
        return

    print("\nThe following files will be uploaded:")
    for file_path in sorted(files_to_upload):
        print(f"- {file_path}")
    print()

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