from .llava import LlavaAdapter
MODEL_REGISTRY={'llava15-7b':'llava-hf/llava-1.5-7b-hf'}
def create_adapter(alias, config, device='cuda', dtype='bfloat16', attn_implementation=None, load_in_4bit=False, **kwargs):
    if alias not in MODEL_REGISTRY: raise ValueError('Only the main paper backbone is included')
    return LlavaAdapter(MODEL_REGISTRY[alias], config, device=device, dtype=dtype, attn_implementation=attn_implementation, load_in_4bit=load_in_4bit, **kwargs)
