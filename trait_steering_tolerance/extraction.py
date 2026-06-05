from __future__ import annotations

from collections import defaultdict
from typing import Any

from tqdm import tqdm

from .artifacts import TraitArtifact
from .config import RunConfig
from .io_utils import read_json, read_jsonl, write_jsonl
from .judge import JudgeClient
from .runtime import ModelRuntime


NEUTRAL_SYSTEM = "You are a helpful, honest, and concise assistant."


def build_extraction_jobs(artifact: TraitArtifact) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for idx, question in enumerate(artifact.extraction_questions):
        instr_idx = idx % len(artifact.positive_instructions)
        pair_id = f"{artifact.trait_id}_pair_{idx:03d}"
        jobs.append(
            {
                "trait_id": artifact.trait_id,
                "pair_id": pair_id,
                "polarity": "positive",
                "instruction_idx": instr_idx,
                "system_instruction": artifact.positive_instructions[instr_idx],
                "question": question,
            }
        )
        jobs.append(
            {
                "trait_id": artifact.trait_id,
                "pair_id": pair_id,
                "polarity": "negative",
                "instruction_idx": instr_idx,
                "system_instruction": artifact.negative_instructions[instr_idx],
                "question": question,
            }
        )
    return jobs


def generate_extraction_records(
    config: RunConfig,
    runtime: ModelRuntime,
    artifacts: dict[str, TraitArtifact],
    traits: list[str] | None = None,
    append: bool = False,
) -> None:
    target_traits = traits or config.traits
    records: list[dict[str, Any]] = []
    if append:
        records = [
            record
            for record in read_jsonl(config.extraction_records_path)
            if record.get("trait_id") not in set(target_traits)
        ]
    for trait_id in target_traits:
        for job in tqdm(build_extraction_jobs(artifacts[trait_id]), desc=f"extract {trait_id}"):
            messages = runtime.make_messages(job["system_instruction"], job["question"])
            response = runtime.generate_response(messages)
            records.append(
                {
                    **job,
                    "model_id": config.model_id,
                    "response": response,
                }
            )
    write_jsonl(config.extraction_records_path, records)


def judge_extraction_records(
    config: RunConfig,
    judge: JudgeClient,
    artifacts: dict[str, TraitArtifact],
    traits: list[str] | None = None,
    append: bool = False,
) -> None:
    target_traits = set(traits or config.traits)
    judged: list[dict[str, Any]] = []
    if append:
        judged = [
            record
            for record in read_jsonl(config.judged_extraction_records_path)
            if record.get("trait_id") not in target_traits
        ]
    records = [
        record
        for record in read_jsonl(config.extraction_records_path)
        if record.get("trait_id") in target_traits
    ]
    for record in tqdm(records, desc="judge extraction"):
        artifact = artifacts[record["trait_id"]]
        trait = judge.score_trait(artifact, record["question"], record["response"])
        coherence = judge.score_coherence(record["question"], record["response"])
        judged.append(
            {
                **record,
                "trait_score": trait.score,
                "trait_reason": trait.reason,
                "coherence_score": coherence.score,
                "coherence_reason": coherence.reason,
            }
        )
    write_jsonl(config.judged_extraction_records_path, judged)


def retained_pairs(config: RunConfig) -> dict[str, list[dict[str, dict[str, Any]]]]:
    records = read_jsonl(config.judged_extraction_records_path)
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[(record["trait_id"], record["pair_id"])][record["polarity"]] = record

    retained: dict[str, list[dict[str, dict[str, Any]]]] = defaultdict(list)
    for (trait_id, _pair_id), pair in grouped.items():
        pos = pair.get("positive")
        neg = pair.get("negative")
        if pos is None or neg is None:
            continue
        if pos["coherence_score"] < config.coherence_threshold:
            continue
        if neg["coherence_score"] < config.coherence_threshold:
            continue
        if pos["trait_score"] <= config.trait_positive_threshold:
            continue
        if neg["trait_score"] >= config.trait_negative_threshold:
            continue
        retained[trait_id].append({"positive": pos, "negative": neg})
    return retained


def extract_trait_vectors(
    config: RunConfig,
    runtime: ModelRuntime,
    traits: list[str] | None = None,
) -> None:
    import torch

    target_traits = traits or config.traits
    retained = retained_pairs(config)
    metadata_path = config.vectors_dir / "vector_metadata.json"
    metadata: dict[str, Any] = read_json(metadata_path) if metadata_path.exists() else {}
    for trait_id in target_traits:
        pairs = retained.get(trait_id, [])
        if len(pairs) < config.min_pairs_per_trait:
            raise RuntimeError(
                f"Only retained {len(pairs)} pairs for {trait_id}; "
                f"minimum is {config.min_pairs_per_trait}."
            )
        pos_sum = None
        neg_sum = None
        for pair in tqdm(pairs, desc=f"vectors {trait_id}"):
            pos = pair["positive"]
            neg = pair["negative"]
            pos_messages = runtime.make_messages(
                pos["system_instruction"], pos["question"]
            )
            neg_messages = runtime.make_messages(
                neg["system_instruction"], neg["question"]
            )
            pos_z = runtime.response_mean_activations(pos_messages, pos["response"])
            neg_z = runtime.response_mean_activations(neg_messages, neg["response"])
            pos_sum = pos_z if pos_sum is None else pos_sum + pos_z
            neg_sum = neg_z if neg_sum is None else neg_sum + neg_z
        mean_pos = pos_sum / len(pairs)
        mean_neg = neg_sum / len(pairs)
        vectors = mean_pos - mean_neg
        out_path = config.vectors_dir / f"{trait_id}_response_avg_diff.pt"
        torch.save(
            {
                "trait_id": trait_id,
                "model_id": config.model_id,
                "hook_point": config.hook_point,
                "vectors": vectors,
                "mean_pos": mean_pos,
                "mean_neg": mean_neg,
                "num_pairs": len(pairs),
            },
            out_path,
        )
        metadata[trait_id] = {
            "num_pairs": len(pairs),
            "path": str(out_path),
            "shape": list(vectors.shape),
        }
    from .io_utils import write_json

    write_json(config.vectors_dir / "vector_metadata.json", metadata)
