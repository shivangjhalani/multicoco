import torch
from transformers import LogitsProcessor

class ForceDigitsLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.allowed_token_ids = []
        for i in range(10):
            # For each digit, get its token ID.
            # We must handle cases where a digit might not be a single token.
            # `tokenize` gives us the string representation, e.g., ['2'] or [' ', '2']
            # `convert_tokens_to_ids` gives us the integer ID.
            tokens = tokenizer.tokenize(str(i))
            if len(tokens) == 1:
                 token_id = tokenizer.convert_tokens_to_ids(tokens[0])
                 self.allowed_token_ids.append(token_id)
            else:
                # This handles cases for tokenizers that add a prefix (like ' ')
                # to numbers. We look for the token corresponding to the digit itself.
                for token in tokens:
                    if str(i) in token:
                        token_id = tokenizer.convert_tokens_to_ids(token)
                        self.allowed_token_ids.append(token_id)
                        break

        if not self.allowed_token_ids:
            raise ValueError("Could not find any digit tokens in the tokenizer vocabulary.")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Create a mask for the entire vocabulary, initially allowing nothing
        mask = torch.full_like(scores, -float('inf'))
        
        # Set the scores for allowed digit tokens to 0 (or their original score)
        mask[:, self.allowed_token_ids] = 0
        
        # Apply the mask to the original scores
        return scores + mask 