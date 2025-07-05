import torch
from transformers import LogitsProcessor


class SingleDigitLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.allowed_tokens = ["0", "1", "2", "3"]
        self.allowed_token_ids = self.tokenizer.convert_tokens_to_ids(self.allowed_tokens)
        print(f"Allowed token IDs: {self.allowed_token_ids}")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # This processor should only be applied at the first generation step.
        if input_ids.shape[-1] > 1:
            return scores

        mask = torch.full(scores.shape, -float("inf"), device=scores.device)
        mask[:, self.allowed_token_ids] = 0
        return scores + mask 