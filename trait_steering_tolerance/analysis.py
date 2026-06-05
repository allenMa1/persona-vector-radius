from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import RunConfig
from .io_utils import ensure_dir, read_jsonl, write_json


def contiguous_interval_around_zero(
    beta_values: list[float],
    coherence_values: list[float],
    threshold: float,
) -> tuple[float | None, float | None, float]:
    pairs = sorted(zip(beta_values, coherence_values), key=lambda item: item[0])
    betas = [float(pair[0]) for pair in pairs]
    ok = [float(pair[1]) >= threshold for pair in pairs]
    if 0.0 not in betas:
        return None, None, 0.0
    idx = betas.index(0.0)
    if not ok[idx]:
        return None, None, 0.0
    left = idx
    right = idx
    while left - 1 >= 0 and ok[left - 1]:
        left -= 1
    while right + 1 < len(ok) and ok[right + 1]:
        right += 1
    beta_low = betas[left]
    beta_high = betas[right]
    return beta_low, beta_high, beta_high - beta_low


def analyze_results(config: RunConfig) -> None:
    import numpy as np
    import pandas as pd

    ensure_dir(config.analysis_dir)
    records = read_jsonl(config.eval_records_path)
    if not records:
        raise RuntimeError(f"No eval records found at {config.eval_records_path}.")
    df = pd.DataFrame(records)
    grouped = (
        df.groupby(["trait_id", "vector_type", "beta"], as_index=False)
        .agg(
            trait_score_mean=("trait_score", "mean"),
            coherence_score_mean=("coherence_score", "mean"),
            repetition_rate_mean=("repetition_rate", "mean"),
            length_mean=("length", "mean"),
        )
        .sort_values(["trait_id", "vector_type", "beta"])
    )
    grouped.to_csv(config.analysis_dir / "score_curves.csv", index=False)

    diameters: list[dict[str, Any]] = []
    for (trait_id, vector_type), subdf in grouped.groupby(["trait_id", "vector_type"]):
        beta_low, beta_high, diameter = contiguous_interval_around_zero(
            subdf["beta"].tolist(),
            subdf["coherence_score_mean"].tolist(),
            config.coherence_threshold,
        )
        if len(subdf) >= 2:
            slope = float(
                np.polyfit(subdf["beta"].to_numpy(), subdf["trait_score_mean"].to_numpy(), 1)[0]
            )
        else:
            slope = 0.0
        diameters.append(
            {
                "trait_id": trait_id,
                "vector_type": vector_type,
                "beta_low": beta_low,
                "beta_high": beta_high,
                "diameter": diameter,
                "trait_score_slope": slope,
            }
        )
    diameter_df = pd.DataFrame(diameters)
    diameter_df.to_csv(config.analysis_dir / "diameters.csv", index=False)

    failures = (
        df[df["coherence_score"] < config.coherence_threshold]
        .sort_values(["trait_id", "vector_type", "beta", "coherence_score"])
        .groupby(["trait_id", "vector_type"])
        .head(3)
    )
    failures.to_csv(config.analysis_dir / "coherence_failure_examples.csv", index=False)
    write_json(
        config.analysis_dir / "summary.json",
        {
            "num_records": int(len(df)),
            "coherence_threshold": config.coherence_threshold,
            "diameters": diameters,
        },
    )
    _write_cosine_similarity(config)
    _plot_curves(config, grouped)


def _write_cosine_similarity(config: RunConfig) -> None:
    try:
        import pandas as pd
        import torch
    except ImportError:
        return

    vectors: dict[str, object] = {}
    for path in sorted(config.vectors_dir.glob("*_response_avg_diff.pt")):
        payload = torch.load(path, map_location="cpu")
        trait_id = str(payload.get("trait_id", path.name.replace("_response_avg_diff.pt", "")))
        matrix = payload["vectors"].float()
        if config.steering_layer >= matrix.shape[0]:
            continue
        vector = matrix[config.steering_layer]
        norm = vector.norm()
        if norm.item() == 0:
            continue
        vectors[trait_id] = vector / norm
    if len(vectors) < 2:
        return
    trait_ids = sorted(vectors)
    rows = []
    for left in trait_ids:
        row = {"trait_id": left}
        for right in trait_ids:
            row[right] = float((vectors[left] @ vectors[right]).item())
        rows.append(row)
    cosine_df = pd.DataFrame(rows)
    cosine_df.to_csv(config.analysis_dir / "trait_cosine_similarity.csv", index=False)


def _plot_curves(config: RunConfig, grouped) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        return

    for metric in ["trait_score_mean", "coherence_score_mean"]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for (trait_id, vector_type), subdf in grouped.groupby(["trait_id", "vector_type"]):
            label = f"{trait_id}/{vector_type}"
            ax.plot(subdf["beta"], subdf[metric], marker="o", label=label)
        if metric == "coherence_score_mean":
            ax.axhline(
                config.coherence_threshold,
                color="black",
                linestyle="--",
                linewidth=1,
                label="coherence threshold",
            )
        ax.set_xlabel("normalized beta")
        ax.set_ylabel(metric.replace("_", " "))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(Path(config.analysis_dir) / f"{metric}.png", dpi=180)
        plt.close(fig)

    diameter_path = Path(config.analysis_dir) / "diameters.csv"
    if diameter_path.exists():
        diameters = pd.read_csv(diameter_path)
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [
            f"{row.trait_id}\n{row.vector_type}" for row in diameters.itertuples()
        ]
        ax.bar(labels, diameters["diameter"])
        ax.set_ylabel("usable beta interval diameter")
        ax.tick_params(axis="x", labelrotation=30)
        fig.tight_layout()
        fig.savefig(Path(config.analysis_dir) / "diameters.png", dpi=180)
        plt.close(fig)

    cosine_path = Path(config.analysis_dir) / "trait_cosine_similarity.csv"
    if cosine_path.exists():
        cosine_df = pd.read_csv(cosine_path)
        trait_ids = cosine_df["trait_id"].tolist()
        matrix = cosine_df[trait_ids].to_numpy()
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(trait_ids)), trait_ids, rotation=45, ha="right")
        ax.set_yticks(range(len(trait_ids)), trait_ids)
        fig.colorbar(image, ax=ax, label="cosine similarity")
        fig.tight_layout()
        fig.savefig(Path(config.analysis_dir) / "trait_cosine_similarity.png", dpi=180)
        plt.close(fig)
