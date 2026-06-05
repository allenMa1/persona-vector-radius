from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator

from .config import RunConfig


@dataclass
class SteeringConfig:
    vector: object
    beta: float
    sigma: float
    layer: int
    hook_point: str = "block_output"


class ModelRuntime:
    def __init__(self, config: RunConfig):
        self.config = config
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install torch and transformers before loading the local model."
            ) from exc

        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available on this machine. Run model stages on the A100 box."
            )

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(self.config.dtype)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {self.config.dtype}")

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id, trust_remote_code=True
        )
        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if self.config.device == "cuda":
            load_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            **load_kwargs,
        )
        if self.config.device != "cuda":
            self.model.to(self.config.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        raise AttributeError("Could not locate decoder layers on this model.")

    @property
    def input_device(self):
        return next(self.model.parameters()).device

    def format_prompt(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def make_messages(self, system_instruction: str, question: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": question},
        ]

    def generate_response(
        self,
        messages: list[dict[str, str]],
        steering: SteeringConfig | None = None,
    ) -> str:
        torch = self.torch
        prompt = self.format_prompt(messages)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.input_device
        )
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.config.do_sample:
            generation_kwargs.update(
                {"temperature": self.config.temperature, "top_p": self.config.top_p}
            )
        with torch.no_grad(), self._steering_hook(steering):
            output_ids = self.model.generate(input_ids, **generation_kwargs)
        new_tokens = output_ids[0, input_ids.shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def response_mean_activations(
        self,
        messages: list[dict[str, str]],
        response_text: str,
    ):
        """Return [num_layers, d_model] block-output means over response tokens."""
        torch = self.torch
        prompt = self.format_prompt(messages)
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids
        response_ids = self.tokenizer(
            response_text, return_tensors="pt", add_special_tokens=False
        ).input_ids
        if response_ids.shape[1] == 0:
            raise ValueError("Response text tokenized to zero tokens.")

        input_ids = torch.cat([prompt_ids, response_ids], dim=1).to(self.input_device)
        response_start = prompt_ids.shape[1]
        captures: dict[int, object] = {}
        handles = []

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                response_hidden = hidden[:, response_start:, :]
                if response_hidden.shape[1] == 0:
                    raise ValueError("Response token slice is empty.")
                captures[layer_idx] = (
                    response_hidden.mean(dim=1).squeeze(0).detach().float().cpu()
                )
                return output

            return hook

        for idx, layer in enumerate(self.layers):
            handles.append(layer.register_forward_hook(make_hook(idx)))
        try:
            with torch.no_grad():
                self.model(input_ids=input_ids)
        finally:
            for handle in handles:
                handle.remove()
        if len(captures) != len(self.layers):
            raise RuntimeError(
                f"Captured {len(captures)} layers but expected {len(self.layers)}."
            )
        return torch.stack([captures[i] for i in range(len(self.layers))], dim=0)

    @contextlib.contextmanager
    def _steering_hook(self, steering: SteeringConfig | None) -> Iterator[None]:
        if steering is None:
            yield
            return
        if steering.hook_point != "block_output":
            raise NotImplementedError(
                "Only block_output residual steering is implemented in v1."
            )
        torch = self.torch
        layer = self.layers[steering.layer]
        vector = steering.vector
        if not torch.is_tensor(vector):
            vector = torch.tensor(vector)
        delta = vector * float(steering.beta) * float(steering.sigma)

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            local_delta = delta.to(device=hidden.device, dtype=hidden.dtype)
            edited = hidden.clone()
            edited[:, -1, :] = edited[:, -1, :] + local_delta
            if isinstance(output, tuple):
                return (edited,) + output[1:]
            return edited

        handle = layer.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()
