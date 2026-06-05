from __future__ import annotations

from typing import Any

from tqdm import tqdm

from .artifacts import TraitArtifact
from .config import RunConfig
from .diagnostics import repetition_rate, token_length
from .extraction import NEUTRAL_SYSTEM
from .io_utils import write_jsonl
from .judge import JudgeClient
from .runtime import ModelRuntime, SteeringConfig
from .vectors import load_vector_specs


def run_evaluation(
    config: RunConfig,
    runtime: ModelRuntime,
    judge: JudgeClient,
    artifacts: dict[str, TraitArtifact],
) -> None:
    specs = load_vector_specs(config)
    records: list[dict[str, Any]] = []

    for trait_id in config.traits:
        artifact = artifacts[trait_id]
        for vector_type in ["trait", "random"]:
            spec = specs[(trait_id, vector_type)]
            for beta in tqdm(
                config.beta_grid, desc=f"eval {trait_id}/{vector_type}"
            ):
                steering = SteeringConfig(
                    vector=spec.vector,
                    beta=float(beta),
                    sigma=spec.sigma,
                    layer=config.steering_layer,
                    hook_point=config.hook_point,
                )
                for prompt_id, question in enumerate(artifact.eval_questions):
                    messages = runtime.make_messages(NEUTRAL_SYSTEM, question)
                    response = runtime.generate_response(messages, steering=steering)
                    trait = judge.score_trait(artifact, question, response)
                    coherence = judge.score_coherence(question, response)
                    records.append(
                        {
                            "model_id": config.model_id,
                            "trait_id": trait_id,
                            "vector_type": vector_type,
                            "beta": beta,
                            "prompt_id": prompt_id,
                            "question": question,
                            "response": response,
                            "trait_score": trait.score,
                            "trait_reason": trait.reason,
                            "coherence_score": coherence.score,
                            "coherence_reason": coherence.reason,
                            "length": token_length(response),
                            "repetition_rate": repetition_rate(response),
                        }
                    )
    write_jsonl(config.eval_records_path, records)

