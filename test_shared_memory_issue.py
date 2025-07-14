import torch
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer

class LatentWrapper:
    """
    A wrapper class for models to handle image token insertion and manage
    embedding layers, specifically addressing shared memory issues during saving.
    """
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Initializes the LatentWrapper.

        Args:
            model: The pre-trained model (e.g., InternVL3-1B-Pretrained).
            tokenizer: The tokenizer corresponding to the model.
        """
        self.model = model
        self.tokenizer = tokenizer

        # Get the image context token ID from the model if available,
        # otherwise infer from tokenizer or set to None.
        self.img_context_token_id = getattr(model, 'img_context_token_id', None)
        if self.img_context_token_id is None:
            # Attempt to find a suitable image context token if not directly available
            if hasattr(tokenizer, 'img_context_token') and tokenizer.img_context_token is not None:
                self.img_context_token_id = tokenizer.convert_tokens_to_ids(tokenizer.img_context_token)
            elif '<image>' in tokenizer.vocab: # Common placeholder
                self.img_context_token_id = tokenizer.vocab['<image>']
            elif '<img>' in tokenizer.vocab: # Another common placeholder
                self.img_context_token_id = tokenizer.vocab['<img>']
            else:
                print("Warning: Could not determine img_context_token_id. Image token insertion might not work as expected.")

        # Initialize the embedding layer using the fixed method
        self.embedding = self._get_embedding_layer(model)

        # Ensure the model's embedding layer is also updated if it's the same instance
        # This is crucial if the original model's embedding was directly referenced
        # and we want the model to use the independent copy for its forward pass.
        # However, the primary fix is ensuring `wrapper.embedding` is independent for saving.
        # If model.model.language_model.model.embed_tokens is directly assigned,
        # then the fix in _get_embedding_layer handles the wrapper's reference.
        # For the base model to use the new embedding, its reference would need to be updated.
        # This wrapper focuses on ensuring its own `embedding` attribute is independent.

    def _get_embedding_layer(self, model):
        """
        Get the correct embedding layer from potentially nested model structure
        and create an independent copy to avoid shared memory issues.

        This prevents the "Some tensors share memory" error when saving.
        """
        original_embedding = None
        # Try common paths to find the embedding layer
        if hasattr(model, 'language_model') and hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'embed_tokens'):
            # InternVL3 structure: model.language_model.model.embed_tokens
            original_embedding = model.language_model.model.embed_tokens
        elif hasattr(model, 'model') and hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'model') and hasattr(model.model.language_model.model, 'embed_tokens'):
            # InternVL structure: model.model.language_model.model.embed_tokens
            original_embedding = model.model.language_model.model.embed_tokens
        elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            # Direct access: model.model.embed_tokens
            original_embedding = model.model.embed_tokens
        elif hasattr(model, 'get_input_embeddings'):
            # Fallback: use get_input_embeddings method
            original_embedding = model.get_input_embeddings()
        else:
            # Last resort: try to find embed_tokens attribute
            for attr_name in ['embed_tokens', 'embeddings', 'word_embeddings']:
                if hasattr(model, attr_name):
                    original_embedding = getattr(model, attr_name)
                    break
            if original_embedding is None:
                raise AttributeError(f"Could not find embedding layer in model: {type(model)}")

        # CRITICAL FIX: Create independent embedding copy to avoid shared memory issues
        # This prevents the "Some tensors share memory" error when saving
        new_embedding = torch.nn.Embedding(
            num_embeddings=original_embedding.num_embeddings,
            embedding_dim=original_embedding.embedding_dim,
            padding_idx=original_embedding.padding_idx,
            max_norm=original_embedding.max_norm,
            norm_type=original_embedding.norm_type,
            scale_grad_by_freq=original_embedding.scale_grad_by_freq,
            sparse=original_embedding.sparse,
            dtype=original_embedding.weight.dtype,
            device=original_embedding.weight.device
        )

        # Copy weights but make them independent
        with torch.no_grad():
            new_embedding.weight.copy_(original_embedding.weight)

        return new_embedding

    def insert_img_tokens(self, prompt: str) -> str:
        """
        Inserts the appropriate image context tokens into the prompt.

        Args:
            prompt: The original prompt string containing '<img></img>' placeholders.

        Returns:
            The prompt with '<img></img>' replaced by actual image context tokens.
        """
        if self.img_context_token_id is None:
            print("Error: img_context_token_id is not set. Cannot insert image tokens.")
            return prompt

        img_token = self.tokenizer.convert_ids_to_tokens(self.img_context_token_id)
        # Assuming <img></img> represents a single image token or a sequence
        # For simplicity, replacing <img></img> with a single img_token.
        # If multiple tokens are needed per image, this logic would need to be expanded.
        return prompt.replace('<img></img>', img_token)

    def image_processor(self):
        """
        Placeholder for the image processing logic.
        This method would typically handle image pre-processing before feeding to the model.
        """
        print("Image processing logic placeholder.")
        # Add actual image processing implementation here
        pass

    # You might want to add a state_dict method if you plan to save the wrapper's state directly
    def state_dict(self, *args, **kwargs):
        """
        Returns the state dictionary of the wrapper, including its independent embedding layer.
        """
        state = self.model.state_dict(*args, **kwargs)
        # Override the embedding weight with the wrapper's independent embedding
        # This ensures the independent copy is saved.
        # You might need to adjust the key based on how the base model's embedding is named.
        embedding_key = None
        if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'model'):
            embedding_key = 'language_model.model.embed_tokens.weight'
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            embedding_key = 'model.embed_tokens.weight'
        elif hasattr(self.model, 'embed_tokens'):
            embedding_key = 'embed_tokens.weight'

        if embedding_key and embedding_key in state:
            state[embedding_key] = self.embedding.weight
        else:
            # Fallback if the key isn't found, try to find a similar one
            for key in state.keys():
                if 'embed_tokens.weight' in key or 'embeddings.weight' in key:
                    state[key] = self.embedding.weight
                    print(f"Warning: Used fallback key '{key}' for embedding weight in state_dict.")
                    break
            else:
                print("Warning: Could not find a suitable key for embedding weight in state_dict. "
                      "The independent embedding might not be saved correctly.")

        return state

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Loads the state dictionary into the wrapper and its model.
        """
        # Load into the base model first
        self.model.load_state_dict(state_dict, *args, **kwargs)

        # Then, ensure the wrapper's independent embedding is updated
        embedding_key = None
        if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'model'):
            embedding_key = 'language_model.model.embed_tokens.weight'
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            embedding_key = 'model.embed_tokens.weight'
        elif hasattr(self.model, 'embed_tokens'):
            embedding_key = 'embed_tokens.weight'

        if embedding_key and embedding_key in state_dict:
            self.embedding.weight.data.copy_(state_dict[embedding_key])
        else:
            for key in state_dict.keys():
                if 'embed_tokens.weight' in key or 'embeddings.weight' in key:
                    self.embedding.weight.data.copy_(state_dict[key])
                    print(f"Warning: Used fallback key '{key}' for embedding weight when loading state_dict.")
                    break
            else:
                print("Warning: Could not find a suitable key for embedding weight when loading state_dict. "
                      "The independent embedding might not be loaded correctly.")
