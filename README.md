# Persona Vector Radius

This repo contains an ARENA 3.0 capstone project on persona-vector steering tolerance.

The core experiment asks:

```text
How far can we steer a model along a learned trait/persona direction before its outputs stop being coherent?
```

The project is anchored on the Anthropic Persona Vectors setup: construct contrastive trait vectors from positive/negative persona-conditioned responses, apply inference-time activation steering, and evaluate whether behavior changes before coherence collapses.

## Main Result

Model:

```text
Qwen/Qwen2.5-7B-Instruct
```

Steering site:

```text
decoder layer 20, block output / residual stream, response-token steering
```

Main traits:

```text
sycophancy
hallucination
evil
```

The wide beta sweep found that same-normalization random control directions stayed coherent across the full tested range:

```text
random controls: diameter 128 over beta in [-64, 64]
```

Learned trait vectors had much smaller coherent steering intervals:

```text
hallucination: diameter 16
evil:          diameter 24
sycophancy:    diameter 40
```

Interpretation: random residual-stream directions were not enough to break coherence under this normalization, while learned trait directions were higher-sensitivity behavioral directions.

## Method

For each trait, the project builds contrastive extraction prompts:

```text
positive trait instruction + question
negative trait instruction + same question
```

The local model generates responses. An OpenRouter judge filters for examples where the positive response expresses the trait, the negative response does not, and both are coherent.

Trait vectors are computed from response-token activations:

```text
z_i[layer] = mean_response_token_activation_i[layer]

v_trait[layer] =
    mean_i(z_positive_i[layer])
    - mean_j(z_negative_j[layer])
```

Steering uses normalized beta units:

```text
hidden_state[:, -1, :] += beta * sigma * v_hat
```

where:

```text
v_hat = v_trait / ||v_trait||
sigma = std((a_baseline - mean_baseline) dot v_hat)
```

The main metric is the largest contiguous beta interval containing `0` where mean judged coherence is at least `70`:

```text
diameter = beta_high - beta_low
```

## Repository Layout

```text
trait_steering_tolerance/   Python package and CLI
configs/                    Run configs
artifacts/                  Trait prompt/rubric artifacts
runs/                       Committed experiment outputs
notebooks/                  Analysis notebook and generated figures
docs/                       Remote/A100 run notes
```

Important files:

```text
notebooks/qwen7_results_analysis.ipynb
runs/qwen25_7b_layer20_response/analysis/diameters.csv
runs/qwen25_7b_layer20_response/analysis/score_curves.csv
runs/qwen25_7b_layer20_response/eval_records.jsonl
```

## Setup

Install locally or on a GPU box:

```bash
pip install -e .
```

For OpenRouter-backed artifact generation and judging:

```bash
export OPENROUTER_API_KEY="..."
```

Do not commit API keys. A local `.env` file is ignored by git.

For remote GPU execution, see:

```text
docs/a100_runbook.md
```

## Running The Core Pipeline

Smoke tests:

```bash
python -m trait_steering_tolerance.cli --config configs/default.json smoke-openrouter
python -m trait_steering_tolerance.cli --config configs/default.json smoke-model
```

Extraction and vector construction:

```bash
python -m trait_steering_tolerance.cli --config configs/default.json write-builtin-artifacts
python -m trait_steering_tolerance.cli --config configs/default.json generate-extraction
python -m trait_steering_tolerance.cli --config configs/default.json judge-extraction
python -m trait_steering_tolerance.cli --config configs/default.json extract-vectors
python -m trait_steering_tolerance.cli --config configs/default.json normalize
```

Wide beta sweep and analysis:

```bash
python -m trait_steering_tolerance.cli --config configs/qwen25_7b_wide.json evaluate
python -m trait_steering_tolerance.cli --config configs/qwen25_7b_wide.json analyze
```

Outputs are written to:

```text
runs/qwen25_7b_layer20_response/
```

## Analysis

The main analysis notebook is:

```text
notebooks/qwen7_results_analysis.ipynb
```

It loads committed result files and exports slide-ready plots to:

```text
notebooks/figures/
```

The notebook covers:

```text
Qwen7 wide result
diameter ratios vs random controls
trait-score and coherence curves
fine boundary sweeps
failure examples
trait-vector cosine similarities
optional Qwen3 / broader geometry outputs if present
```

No GPU or API key is needed to run the notebook.

## Presentation

The 5-minute Beamer slide deck is:

```text
slides/persona_vector_radius.tex
```

It uses figures exported by the analysis notebook from:

```text
notebooks/figures/
```

Compile with:

```bash
cd slides
pdflatex persona_vector_radius.tex
pdflatex persona_vector_radius.tex
```

## Additional Experiments

Implemented configs include:

```text
configs/qwen25_7b_fine_hallucination.json
configs/qwen25_7b_fine_evil.json
configs/qwen25_7b_fine_sycophancy.json
configs/qwen25_3b_wide.json
configs/qwen25_7b_trait_geometry.json
```

Fine sweeps probe boundary points near the wide-sweep coherence failures. Because their grids omit beta `0`, their automatic `diameters.csv` should not be interpreted as official diameter estimates. Use them as boundary-refinement data.

The trait-geometry config extracts vectors for additional safety/style traits and computes cosine similarities only. It does not run beta sweeps.

## Limitations

- Trait artifacts are hand-authored or generated artifacts, not the exact Anthropic paper artifacts.
- Coherence and trait strength are LLM-judge metrics.
- The main result uses one model, one layer, and one hook point.
- Retained contrastive pair counts are small.
- Projection/downstream activation diagnostics are not part of the main result.
- Fine sweeps refine boundaries but do not replace the formal beta-0-containing diameter definition.
