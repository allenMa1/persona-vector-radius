# A100 Runbook

This project has two execution environments:

- local workstation: code, configs, docs, cached-output analysis
- A100 box: Qwen/Llama generation, activation extraction, vector computation, steered evaluation

The split exists because the local workstation should not load `Qwen/Qwen2.5-7B-Instruct`.

## Local Workstation

Use local for:

- editing code and configs
- writing artifacts by hand
- reviewing cached JSONL/CSV/plots copied back from the A100
- running `tst analyze` if `runs/` has already been copied back
- optional OpenRouter smoke test if the local network/key are available

Avoid local for:

- `smoke-model`
- `generate-extraction`
- `extract-vectors`
- `normalize`
- `evaluate`

Those stages load the local model or run many model forwards.

## A100 Box

Copy the repo to the A100 box, then run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
export OPENROUTER_API_KEY="..."
```

Then:

```bash
tst --config configs/default.json write-builtin-artifacts
tst --config configs/default.json smoke-openrouter
tst --config configs/default.json smoke-model
tst --config configs/default.json generate-extraction
tst --config configs/default.json judge-extraction
tst --config configs/default.json extract-vectors
tst --config configs/default.json normalize
tst --config configs/default.json evaluate
tst --config configs/default.json analyze
```

If a stage fails, fix that stage and rerun it. Previous stages write cached files under `runs/`.

## Rerunning One Failed Trait

If one trait fails extraction filtering, keep the successful traits and replace only that trait:

```bash
tst --config configs/default.json write-builtin-artifacts
tst --config configs/default.json generate-extraction --traits evil --append
tst --config configs/default.json judge-extraction --traits evil --append
tst --config configs/default.json extract-vectors --traits evil
```

Vector extraction uses `extraction_coherence_threshold` from config, while final diameter analysis uses `coherence_threshold`. This lets extraction keep legible contrastive examples without weakening the final coherence metric.

Then continue:

```bash
tst --config configs/default.json normalize
tst --config configs/default.json evaluate
tst --config configs/default.json analyze
```

## Interface Between Machines

The important portable outputs are:

```text
artifacts/*.json
runs/<run_name>/extraction_records*.jsonl
runs/<run_name>/vectors/*.pt
runs/<run_name>/normalization.pt
runs/<run_name>/eval_records.jsonl
runs/<run_name>/analysis/*
```

After the A100 run, copy `runs/` back to local for inspection and writeup.

## Seven-Trait Stretch

The code is trait-config driven. To run all seven paper traits:

1. Add trait ids to `configs/default.json`.
2. For traits without built-in artifacts, run:

```bash
tst --config configs/default.json generate-artifacts --traits humorous apathetic optimistic impolite
```

3. Rerun the same extraction, normalization, evaluation, and analysis stages.

The analysis stage automatically writes:

```text
runs/<run_name>/analysis/trait_cosine_similarity.csv
runs/<run_name>/analysis/trait_cosine_similarity.png
```

for all available vectors in `runs/<run_name>/vectors/`.

## API Key Handling

Set the key only in the shell:

```bash
export OPENROUTER_API_KEY="..."
```

Do not place it in configs, source files, notebooks, or logs.
