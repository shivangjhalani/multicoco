# Upload Checkpoints to Hugging Face Hub

This folder contains a script to upload its contents to a Hugging Face Hub repository.

## Prerequisites

1.  **Install `huggingface_hub`:**
    ```bash
    pip install huggingface_hub
    ```

2.  **Login to Hugging Face:**
    You need to authenticate with your Hugging Face account. You can do this by running the following command and entering your token:
    ```bash
    huggingface-cli login
    ```
    Alternatively, you can set the `HUGGING_FACE_HUB_TOKEN` environment variable.

## Usage

1.  Place all your checkpoint files in this directory.
2.  Run the script:
    ```bash
    python checkpoints/upload_checkpoints.py
    ```
3.  By default, it will upload to `ThefirstM/checkpoints`. To upload to a different repository, use the `--repo_id` argument:
    ```bash
    python checkpoints/upload_checkpoints.py --repo_id "your-username/your-repo-name"
    ``` 