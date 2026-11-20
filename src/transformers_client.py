"""Run the official Qwen checkpoint through the common single-action protocol."""

import json
import re

from .client import InfrastructureError, ModelOutputError, parse_chat_response


def qwen_response(text, *, finish_reason, usage):
    """Retain raw text while translating Qwen tool tags to OpenAI tool calls."""
    payload = {"choices": [{"finish_reason": finish_reason, "message": {"role": "assistant", "content": text}}],
               "usage": usage, "generated_text": text}
    if finish_reason == "length":
        parse_chat_response(payload, accept_plain_call=True)
    message = payload["choices"][0]["message"]
    content = text.strip()
    if content.startswith("<think>"):
        if "</think>" not in content:
            raise ModelOutputError("Unclosed reasoning block", kind="unparsed_tool_call", payload=payload)
        content = content.split("</think>", 1)[1].strip()
    matches = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL))
    if matches:
        remainder = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL).strip()
        if "<tool_call>" in remainder or "</tool_call>" in remainder:
            raise ModelOutputError("Incomplete tool markup", kind="unparsed_tool_call", payload=payload)
        calls = []
        for match in matches:
            try:
                function = json.loads(match.group(1))
            except json.JSONDecodeError as exc:
                raise ModelOutputError("Tool call is not valid JSON", payload=payload) from exc
            if not isinstance(function, dict) or set(function) != {"name", "arguments"}:
                raise ModelOutputError("Tool call needs name and arguments", payload=payload)
            calls.append({"type": "function", "function": function})
        message.update(content=remainder, tool_calls=calls)
    else:
        message["content"] = content
    action = parse_chat_response(payload, accept_plain_call=True)
    return {"action": action, "usage": usage, "raw_response": payload,
            "finish_reason": finish_reason, "attempted_tool_call_count": int(action["type"] == "tool_call")}


class TransformersClient:
    def __init__(self, model_dir, *, max_tokens=512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        if not torch.cuda.is_available():
            raise InfrastructureError("CUDA is unavailable in the Transformers environment")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float16, device_map={"": "cuda:0"},
            low_cpu_mem_usage=True, attn_implementation="sdpa", local_files_only=True,
        ).eval()
        if self.model.config.model_type != "qwen3":
            raise InfrastructureError("Expected the official Qwen3 architecture")
        if any(p.dtype != torch.float16 for p in self.model.parameters()):
            raise InfrastructureError("The official checkpoint did not load entirely as FP16")
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.stop_ids = list(dict.fromkeys(
            [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|endoftext|>")]
        ))
        if any(type(i) is not int or i < 0 for i in self.stop_ids):
            raise InfrastructureError("Missing Qwen stop token IDs")
        self.generation = GenerationConfig(
            do_sample=False, max_new_tokens=max_tokens, repetition_penalty=1.0,
            eos_token_id=self.stop_ids, pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        self.max_tokens = max_tokens

    def chat(self, messages, tools):
        prompt = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True,
            enable_thinking=False, parallel_tool_calls=False,
        )
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda:0")
        prompt_length = inputs["input_ids"].shape[1]
        if prompt_length + self.max_tokens > 8192:
            raise InfrastructureError("Prompt and output allowance exceed the common 8192-token context")
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=self.generation)
        ids = output[0, prompt_length:].tolist()
        stopped = bool(ids and ids[-1] in self.stop_ids)
        text = self.tokenizer.decode(ids[:-1] if stopped else ids, skip_special_tokens=False)
        usage = {"prompt_tokens": prompt_length, "completion_tokens": len(ids),
                 "total_tokens": prompt_length + len(ids)}
        audit = {"rendered_prompt": prompt, "prompt_token_ids": inputs["input_ids"][0].tolist(),
                 "generated_token_ids": ids}
        try:
            response = qwen_response(text, finish_reason="stop" if stopped else "length", usage=usage)
        except ModelOutputError as exc:
            exc.payload["input_audit"] = audit
            raise
        response["raw_response"]["input_audit"] = audit
        return response
