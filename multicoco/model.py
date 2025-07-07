import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor

class MultiCoCo(nn.Module):
    def __init__(self, model_id, **kwargs):
        super(MultiCoCo, self).__init__()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            **kwargs
        )

    def forward(self, *args, **kwargs):
        """
        The DataCollator now prepares all arguments in the format expected by the model,
        so we can just pass them through.
        """
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """
        Pass the generation call directly to the underlying model.
        """
        return self.model.generate(*args, **kwargs)
