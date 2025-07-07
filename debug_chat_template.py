from transformers import AutoProcessor

def main():
    model_id = 'OpenGVLab/InternVL3-1B-Pretrained'
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        # The tokenizer is the object that holds the chat template
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        
        print("--- Chat Template ---")
        print(tokenizer.chat_template)
        print("---------------------\n")

        # Let's test how to apply it with a multimodal message structure
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is in this image?"},
                ],
            }
        ]
        
        print("--- Applying Template with Multimodal Content ---")
        try:
            # We don't tokenize here, just render the string
            rendered_string = tokenizer.apply_chat_template(messages, tokenize=False)
            print("Successfully rendered template:")
            print(rendered_string)
        except Exception as e:
            print(f"Failed to apply template: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 