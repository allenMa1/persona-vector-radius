from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, read_json, write_json


DEFAULT_BETA_GRID = [-8, -5, -3, -2, -1, 0, 1, 2, 3, 5, 8]


@dataclass
class RunConfig:
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"
    traits: list[str] = field(
        default_factory=lambda: ["sycophancy", "hallucination", "evil"]
    )
    steering_layer: int = 20
    hook_point: str = "block_output"
    beta_grid: list[float] = field(default_factory=lambda: list(DEFAULT_BETA_GRID))
    max_new_tokens: int = 180
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    coherence_threshold: int = 70
    trait_positive_threshold: int = 50
    trait_negative_threshold: int = 50
    min_pairs_per_trait: int = 4
    judge_model: str = "openai/gpt-4.1-mini"
    artifact_generator_model: str = "anthropic/claude-3.7-sonnet"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    cache_dir: str = "cache"
    artifact_dir: str = "artifacts"
    run_dir: str = "runs/qwen25_7b_layer20_response"

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir)

    @property
    def artifact_path(self) -> Path:
        return Path(self.artifact_dir)

    @property
    def run_path(self) -> Path:
        return Path(self.run_dir)

    @property
    def extraction_records_path(self) -> Path:
        return self.run_path / "extraction_records.jsonl"

    @property
    def judged_extraction_records_path(self) -> Path:
        return self.run_path / "extraction_records_judged.jsonl"

    @property
    def vectors_dir(self) -> Path:
        return self.run_path / "vectors"

    @property
    def normalization_path(self) -> Path:
        return self.run_path / "normalization.pt"

    @property
    def eval_records_path(self) -> Path:
        return self.run_path / "eval_records.jsonl"

    @property
    def analysis_dir(self) -> Path:
        return self.run_path / "analysis"

    def ensure_dirs(self) -> None:
        for path in [
            self.cache_path,
            self.artifact_path,
            self.run_path,
            self.vectors_dir,
            self.analysis_dir,
        ]:
            ensure_dir(path)


def load_config(path: str | Path | None = None) -> RunConfig:
    if path is None:
        return RunConfig()
    data = read_json(path)
    return RunConfig(**data)


def write_default_config(path: str | Path) -> None:
    write_json(path, RunConfig())


def config_summary(config: RunConfig) -> dict[str, Any]:
    return {
        "model_id": config.model_id,
        "traits": config.traits,
        "steering_layer": config.steering_layer,
        "hook_point": config.hook_point,
        "beta_grid": config.beta_grid,
        "run_dir": config.run_dir,
    }

