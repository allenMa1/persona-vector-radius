# Trait Steering Tolerance

This project measures how far trait-specific persona vectors can be used for activation steering before model outputs lose coherence.

The non-stretch run uses:

- `Qwen/Qwen2.5-7B-Instruct`
- traits: `sycophancy`, `hallucination`, `evil`
- layer `20`
- response-token steering
- OpenRouter only for artifact generation/judging
- local A100 for model generation, activation extraction, and steering

## Setup

On the A100 box:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
export OPENROUTER_API_KEY="..."
```

PowerShell equivalent:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
$env:OPENROUTER_API_KEY="..."
```

Do not commit or paste the API key. The code only reads `OPENROUTER_API_KEY`.

## Pipeline

Write built-in artifacts:

```bash
tst --config configs/default.json write-builtin-artifacts
```

Smoke-test API and local model:

```bash
tst --config configs/default.json smoke-openrouter
tst --config configs/default.json smoke-model
```

Run extraction:

```bash
tst --config configs/default.json generate-extraction
tst --config configs/default.json judge-extraction
tst --config configs/default.json extract-vectors
```

Normalize steering strength:

```bash
tst --config configs/default.json normalize
```

Run beta sweep and analyze:

```bash
tst --config configs/default.json evaluate
tst --config configs/default.json analyze
```

Outputs are written under `runs/qwen25_7b_layer20_response/`.

For the local/A100 split, use [docs/a100_runbook.md](docs/a100_runbook.md).

## What Each Stage Does

`generate-extraction` asks local Qwen to answer trait-positive and trait-negative prompts.

`judge-extraction` scores those responses for trait expression and coherence through OpenRouter.

`extract-vectors` replays retained transcripts, averages response-token activations at every layer, and saves:

```text
v_trait[layer] = mean(pos_response_activations[layer]) - mean(neg_response_activations[layer])
```

`normalize` estimates baseline projection standard deviation for each trait and creates a same-dimension random control vector.

`evaluate` applies response steering:

```text
hidden_state[:, -1, :] += beta * sigma_trait * v_hat
```

at layer `20`, then judges trait expression and coherence.

`analyze` creates:

- `score_curves.csv`
- `diameters.csv`
- `trait_cosine_similarity.csv`
- coherence failure examples
- score and diameter plots

## Main Metric

For each trait/control, the usable interval is the largest contiguous beta range containing `0` where mean coherence score is at least `70`.

```text
diameter = beta_high - beta_low
```

## A100 Notes

This local workstation does not need to run Qwen. Copy the repo to the A100 box, install dependencies there, set `OPENROUTER_API_KEY`, and run the CLI stages.

If Hugging Face model access is gated or blocked, authenticate on the A100 box before running `smoke-model`.

## Stretch Slots

- Add `Llama-3.1-8B-Instruct` by creating a second config.
- Add the remaining paper traits by adding them to config and running `generate-artifacts` for traits without built-in artifacts.
- Add benign/style traits such as `humor` or `politeness`.
- Sweep layers `[12, 16, 20, 24]`.
- Add downstream projection logging during steered generation.

Cosine similarity between all extracted trait vectors is already computed during `analyze`.
