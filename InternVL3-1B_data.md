```python
import torch
import torch.distributed
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel

model = AutoModel.from_pretrained("OpenGVLab/InternVL3-1B", trust_remote_code=True, torch_dtype=torch.bfloat16, use_flash_attn=True, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B", trust_remote_code=True, use_fast=False)
```

```python
print(type(model))
print(dir(model)) # This will list all attributes and methods
```

InternVLChatModel(
  (vision_model): InternVisionModel(
    (embeddings): InternVisionEmbeddings(
      (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14))
    )
    (encoder): InternVisionEncoder(
      (layers): ModuleList(
        (0): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): Identity()
          (drop_path2): Identity()
        )
        (1): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): DropPath(drop_prob=0.004)
          (drop_path2): DropPath(drop_prob=0.004)
        )
        (2): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): DropPath(drop_prob=0.009)
          (drop_path2): DropPath(drop_prob=0.009)
        )
        (3): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): DropPath(drop_prob=0.013)
          (drop_path2): DropPath(drop_prob=0.013)
        )
        (4): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): DropPath(drop_prob=0.017)
          (drop_path2): DropPath(drop_prob=0.017)
        )
        (5): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): DropPath(drop_prob=0.022)
          (drop_path2): DropPath(drop_prob=0.022)
        )
        (6): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): DropPath(drop_prob=0.026)
          (drop_path2): DropPath(drop_prob=0.026)
        )
        (7): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): DropPath(drop_prob=0.031)
          (drop_path2): DropPath(drop_prob=0.031)
        )
        (8): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): DropPath(drop_prob=0.035)
          (drop_path2): DropPath(drop_prob=0.035)
        )
        (9): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): DropPath(drop_prob=0.039)
          (drop_path2): DropPath(drop_prob=0.039)
        )
        (10): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): DropPath(drop_prob=0.044)
          (drop_path2): DropPath(drop_prob=0.044)
        )
        (11): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): DropPath(drop_prob=0.048)
          (drop_path2): DropPath(drop_prob=0.048)
        )
        (12): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): DropPath(drop_prob=0.052)
          (drop_path2): DropPath(drop_prob=0.052)
        )
        (13): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): DropPath(drop_prob=0.056)
          (drop_path2): DropPath(drop_prob=0.056)
        )
        (14): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): DropPath(drop_prob=0.061)
          (drop_path2): DropPath(drop_prob=0.061)
        )
        (15): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): DropPath(drop_prob=0.065)
          (drop_path2): DropPath(drop_prob=0.065)
        )
        (16): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): DropPath(drop_prob=0.069)
          (drop_path2): DropPath(drop_prob=0.069)
        )
        (17): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): DropPath(drop_prob=0.074)
          (drop_path2): DropPath(drop_prob=0.074)
        )
        (18): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): DropPath(drop_prob=0.078)
          (drop_path2): DropPath(drop_prob=0.078)
        )
        (19): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): DropPath(drop_prob=0.083)
          (drop_path2): DropPath(drop_prob=0.083)
        )
        (20): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): DropPath(drop_prob=0.087)
          (drop_path2): DropPath(drop_prob=0.087)
        )
        (21): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): DropPath(drop_prob=0.091)
          (drop_path2): DropPath(drop_prob=0.091)
        )
        (22): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): DropPath(drop_prob=0.096)
          (drop_path2): DropPath(drop_prob=0.096)
        )
        (23): InternVisionEncoderLayer(
          (attn): InternAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): InternMLP(
            (act): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (drop_path1): DropPath(drop_prob=0.100)
          (drop_path2): DropPath(drop_prob=0.100)
        )
      )
    )
  )
  (language_model): Qwen2ForCausalLM(
    (model): Qwen2Model(
      (embed_tokens): Embedding(151674, 896)
      (layers): ModuleList(
        (0-23): 24 x Qwen2DecoderLayer(
          (self_attn): Qwen2Attention(
            (q_proj): Linear(in_features=896, out_features=896, bias=True)
            (k_proj): Linear(in_features=896, out_features=128, bias=True)
            (v_proj): Linear(in_features=896, out_features=128, bias=True)
            (o_proj): Linear(in_features=896, out_features=896, bias=False)
          )
          (mlp): Qwen2MLP(
            (gate_proj): Linear(in_features=896, out_features=4864, bias=False)
            (up_proj): Linear(in_features=896, out_features=4864, bias=False)
            (down_proj): Linear(in_features=4864, out_features=896, bias=False)
            (act_fn): SiLU()
          )
          (input_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
          (post_attention_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
        )
      )
      (norm): Qwen2RMSNorm((896,), eps=1e-06)
      (rotary_emb): Qwen2RotaryEmbedding()
    )
    (lm_head): Linear(in_features=896, out_features=151674, bias=False)
  )
  (mlp1): Sequential(
    (0): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
    (1): Linear(in_features=4096, out_features=896, bias=True)
    (2): GELU(approximate='none')
    (3): Linear(in_features=896, out_features=896, bias=True)
  )
)

```python
print(model.config)
```

InternVLChatConfig {
  "architectures": [
    "InternVLChatModel"
  ],
  "auto_map": {
    "AutoConfig": "configuration_internvl_chat.InternVLChatConfig",
    "AutoModel": "modeling_internvl_chat.InternVLChatModel",
    "AutoModelForCausalLM": "modeling_internvl_chat.InternVLChatModel"
  },
  "downsample_ratio": 0.5,
  "dynamic_image_size": true,
  "force_image_size": 448,
  "hidden_size": 896,
  "image_fold": null,
  "llm_config": {
    "_name_or_path": "./pretrained/Qwen2.5-32B-Instruct",
    "architectures": [
      "Qwen2ForCausalLM"
    ],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151643,
    "hidden_act": "silu",
    "hidden_size": 896,
    "initializer_range": 0.02,
    "intermediate_size": 4864,
    "layer_types": [
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention"
    ],
    "max_position_embeddings": 32768,
    "max_window_layers": 70,
    "model_type": "qwen2",
    "moe_config": null,
    "num_attention_heads": 14,
    "num_hidden_layers": 24,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
      "factor": 2.0,
      "rope_type": "dynamic",
      "type": "dynamic"
    },
    "rope_theta": 1000000.0,
    "sliding_window": null,
    "torch_dtype": "bfloat16",
    "use_bfloat16": true,
    "use_cache": false,
    "use_sliding_window": false,
    "vocab_size": 151674
  },
  "max_dynamic_patch": 12,
  "min_dynamic_patch": 1,
  "model_type": "internvl_chat",
  "output_attentions": false,
  "pad2square": false,
  "ps_version": "v2",
  "select_layer": -1,
  "system_message": null,
  "template": "internvl2_5",
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": null,
  "use_backbone_lora": 0,
  "use_llm_lora": 0,
  "use_thumbnail": true,
  "vision_config": {
    "_name_or_path": "OpenGVLab/InternViT-6B-448px-V1-5",
    "architectures": [
      "InternVisionModel"
    ],
    "attention_dropout": 0.0,
    "auto_map": {
      "AutoConfig": "configuration_intern_vit.InternVisionConfig",
      "AutoModel": "modeling_intern_vit.InternVisionModel"
    },
    "capacity_factor": 1.2,
    "drop_path_rate": 0.1,
    "dropout": 0.0,
    "eval_capacity_factor": 1.4,
    "hidden_act": "gelu",
    "hidden_size": 1024,
    "image_size": 448,
    "initializer_factor": 0.1,
    "initializer_range": 1e-10,
    "intermediate_size": 4096,
    "laux_allreduce": "all_nodes",
    "layer_norm_eps": 1e-06,
    "model_type": "intern_vit_6b",
    "moe_coeff_ratio": 0.5,
    "moe_intermediate_size": 768,
    "moe_output_scale": 4.0,
    "noisy_gate_policy": "RSample_before",
    "norm_type": "layer_norm",
    "num_attention_heads": 16,
    "num_channels": 3,
    "num_experts": 8,
    "num_hidden_layers": 24,
    "num_routed_experts": 4,
    "num_shared_experts": 4,
    "patch_size": 14,
    "qk_normalization": false,
    "qkv_bias": true,
    "shared_expert_intermediate_size": 3072,
    "torch_dtype": "bfloat16",
    "use_bfloat16": true,
    "use_flash_attn": false,
    "use_moe": false,
    "use_residual": true,
    "use_rts": false,
    "use_weighted_residual": false
  }
}

```python
print(tokenizer)
```

Qwen2Tokenizer(name_or_path='OpenGVLab/InternVL3-1B', vocab_size=151643, model_max_length=12288, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'eos_token': '<|im_end|>', 'pad_token': '<|endoftext|>', 'additional_special_tokens': ['<|im_start|>', '<|im_end|>', '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>', '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>']}, clean_up_tokenization_spaces=False, added_tokens_decoder={
	151643: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151644: AddedToken("<|im_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151645: AddedToken("<|im_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151646: AddedToken("<|object_ref_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151647: AddedToken("<|object_ref_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151648: AddedToken("<|box_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151649: AddedToken("<|box_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151650: AddedToken("<|quad_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151651: AddedToken("<|quad_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151652: AddedToken("<|vision_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151653: AddedToken("<|vision_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151654: AddedToken("<|vision_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151655: AddedToken("<|image_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151656: AddedToken("<|video_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151657: AddedToken("<tool_call>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151658: AddedToken("</tool_call>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151659: AddedToken("<|fim_prefix|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151660: AddedToken("<|fim_middle|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151661: AddedToken("<|fim_suffix|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151662: AddedToken("<|fim_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151663: AddedToken("<|repo_name|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151664: AddedToken("<|file_sep|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151665: AddedToken("<img>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151666: AddedToken("</img>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151667: AddedToken("<IMG_CONTEXT>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151668: AddedToken("<quad>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151669: AddedToken("</quad>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151670: AddedToken("<ref>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151671: AddedToken("</ref>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151672: AddedToken("<box>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151673: AddedToken("</box>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
}
)


```python
# Inspect model attributes for image input format information
print("Model attributes potentially related to image input:")
for attr_name in dir(model.config.vision_config):
    if "image" in attr_name or "patch" in attr_name or "size" in attr_name:
        try:
            print(f"{attr_name}: {getattr(model.config.vision_config, attr_name)}")
        except:
            pass
```

Model attributes potentially related to image input:
__sizeof__: <built-in method __sizeof__ of InternVisionConfig object at 0x7803f86bc250>
chunk_size_feed_forward: 0
cross_attention_hidden_size: None
encoder_no_repeat_ngram_size: 0
hidden_size: 1024
image_size: 448
intermediate_size: 4096
moe_intermediate_size: 768
no_repeat_ngram_size: 0
patch_size: 14
shared_expert_intermediate_size: 3072



```python
import torch

# Create dummy inputs based on the model configuration
image_size = model.config.vision_config.image_size
patch_size = model.config.vision_config.patch_size
hidden_size = model.config.vision_config.hidden_size
llm_hidden_size = model.config.llm_config.hidden_size
max_position_embeddings = model.config.llm_config.max_position_embeddings
vocab_size = model.config.llm_config.vocab_size

# Dummy image input (batch_size, num_channels, height, width)
# Cast to bfloat16 to match model's expected type
dummy_image_input = torch.randn(1, 3, image_size, image_size).to(torch.bfloat16)
print(f"Dummy image input shape: {dummy_image_input.shape}")
print(f"Dummy image input dtype: {dummy_image_input.dtype}")


# Dummy text input (batch_size, sequence_length)
dummy_text_input = torch.randint(0, vocab_size, (1, 10))
print(f"Dummy text input shape: {dummy_text_input.shape}")

# Dummy attention mask (batch_size, sequence_length)
dummy_attention_mask = torch.ones(1, 10)
print(f"Dummy attention mask shape: {dummy_attention_mask.shape}")

# You can also create dummy inputs for other potential arguments like token_type_ids if needed,
# but these are the most common for this type of model.

# To see intermediate tensor shapes, you would need to forward the dummy inputs through the model
# and potentially add hooks or inspect the model's forward method. This can be complex and depends
# on the specific model architecture.

# Example of forwarding through the vision model to see the output shape
with torch.no_grad():
    vision_output = model.vision_model(dummy_image_input)
    print(f"Vision model output shape (last hidden state): {vision_output.last_hidden_state.shape}")

# Example of forwarding through the language model to see the output shape
# Note: This requires matching the dimensions and types correctly based on the model's forward method
# and the output of the vision model if you want to combine them.
# This is a simplified example and might need adjustments based on the exact model implementation.
with torch.no_grad():
    # In a multimodal model, the text input might be combined with the vision output
    # Let's create a dummy combined input shape based on potential concatenation
    # This is a simplification - the actual combination depends on the model's forward method
    dummy_combined_input = torch.randn(1, vision_output.last_hidden_state.shape[1] + dummy_text_input.shape[1], llm_hidden_size)
    print(f"Dummy combined input shape (example): {dummy_combined_input.shape}")

    # To get the language model output, you'd typically pass the combined input and attention mask
    # The exact method call depends on the model.
    # For demonstration, let's assume a method like 'generate' or 'forward' that takes inputs
    # print(f"Language model output shape: ...") # This would require calling the model's forward method with correctly formatted inputs
```

Dummy image input shape: torch.Size([1, 3, 448, 448])
Dummy image input dtype: torch.bfloat16
Dummy text input shape: torch.Size([1, 10])
Dummy attention mask shape: torch.Size([1, 10])
Vision model output shape (last hidden state): torch.Size([1, 1025, 1024])
Dummy combined input shape (example): torch.Size([1, 1035, 896])

