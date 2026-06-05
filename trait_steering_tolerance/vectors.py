from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from .artifacts import TraitArtifact
from .config import RunConfig
from .extraction import NEUTRAL_SYSTEM
from .runtime import ModelRuntime


@dataclass
class VectorSpec:
    trait_id: str
    vector_type: str
    vector: object
    sigma: float


def load_trait_vector(config: RunConfig, trait_id: str, layer: int | None = None):
    import torch

    layer = config.steering_layer if layer is None else layer
    path = config.vectors_dir / f"{trait_id}_response_avg_diff.pt"
    payload = torch.load(path, map_location="cpu")
    vectors = payload["vectors"].float()
    vector = vectors[layer]
    norm = vector.norm()
    if norm.item() == 0:
        raise ValueError(f"Zero vector for {trait_id} layer {layer}.")
    return vector / norm


def make_random_unit_vector(reference_vector, seed: int):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    vector = torch.randn(reference_vector.shape, generator=generator)
    return vector / vector.norm()


def compute_normalization_stats(
    config: RunConfig,
    runtime: ModelRuntime,
    artifacts: dict[str, TraitArtifact],
) -> None:
    import torch

    stats: dict[str, Any] = {}
    baseline_cache: dict[str, list[object]] = {}

    for trait_id in config.traits:
        artifact = artifacts[trait_id]
        activations = []
        for question in tqdm(
            artifact.eval_questions, desc=f"baseline activations {trait_id}"
        ):
            messages = runtime.make_messages(NEUTRAL_SYSTEM, question)
            response = runtime.generate_response(messages)
            z = runtime.response_mean_activations(messages, response)[
                config.steering_layer
            ]
            activations.append(z)
        baseline_cache[trait_id] = activations

    for trait_id in config.traits:
        v_hat = load_trait_vector(config, trait_id)
        acts = torch.stack(baseline_cache[trait_id], dim=0).float()
        centered = acts - acts.mean(dim=0, keepdim=True)
        projections = centered @ v_hat.float()
        sigma = projections.std(unbiased=False).item()
        if sigma <= 1e-8:
            sigma = 1.0
        trait_seed = config.seed + 1009 + sum(ord(char) for char in trait_id)
        random_hat = make_random_unit_vector(v_hat, seed=trait_seed)
        random_projections = centered @ random_hat.float()
        random_sigma = random_projections.std(unbiased=False).item()
        if random_sigma <= 1e-8:
            random_sigma = 1.0
        stats[trait_id] = {
            "trait_sigma": sigma,
            "random_sigma": random_sigma,
            "trait_v_hat": v_hat,
            "random_v_hat": random_hat,
            "num_baseline_prompts": len(baseline_cache[trait_id]),
        }

    torch.save(stats, config.normalization_path)


def load_vector_specs(config: RunConfig) -> dict[tuple[str, str], VectorSpec]:
    import torch

    stats = torch.load(config.normalization_path, map_location="cpu")
    specs: dict[tuple[str, str], VectorSpec] = {}
    for trait_id, payload in stats.items():
        specs[(trait_id, "trait")] = VectorSpec(
            trait_id=trait_id,
            vector_type="trait",
            vector=payload["trait_v_hat"],
            sigma=float(payload["trait_sigma"]),
        )
        specs[(trait_id, "random")] = VectorSpec(
            trait_id=trait_id,
            vector_type="random",
            vector=payload["random_v_hat"],
            sigma=float(payload["random_sigma"]),
        )
    return specs
