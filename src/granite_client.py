"""Granite's native template and EOS with the shared audited tool-call protocol."""

import json
from pathlib import Path

from .client import InfrastructureError
from .transformers_client import TransformersClient


def validate_granite_config(config):
    if (config.get('model_type') != 'granitemoehybrid'
            or not config.get('layer_types')
            or set(config['layer_types']) != {'attention'}
            or config.get('num_local_experts') != 0):
        raise ValueError('Use Granite-4.0-Micro, the dense attention-only instruct checkpoint')


class GraniteClient(TransformersClient):
    """Reuse response auditing/parsing; do not change the original Qwen client."""

    def __init__(self, model_dir, *, max_tokens=512):
        model_dir = Path(model_dir)
        validate_granite_config(json.loads((model_dir / 'config.json').read_text(encoding='utf-8')))
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        if not torch.cuda.is_available():
            raise InfrastructureError('CUDA is unavailable; run this pilot in the GPU WSL environment')
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float16, device_map={'': 'cuda:0'},
            low_cpu_mem_usage=True, attn_implementation='sdpa', local_files_only=True,
        ).eval()
        if any(p.dtype != torch.float16 for p in self.model.parameters()):
            raise InfrastructureError('Granite did not load entirely as FP16')
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        eos = self.tokenizer.eos_token_id
        if (type(eos) is not int or eos < 0
                or self.tokenizer.eos_token != '<|end_of_text|>'
                or self.model.config.eos_token_id != eos):
            raise InfrastructureError('Granite tokenizer and model EOS differ')
        self.stop_ids = [eos]
        self.generation = GenerationConfig(
            do_sample=False, max_new_tokens=max_tokens, repetition_penalty=1.0,
            eos_token_id=self.stop_ids, pad_token_id=self.tokenizer.pad_token_id, use_cache=True,
        )
        self.max_tokens = max_tokens
