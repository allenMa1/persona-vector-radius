from __future__ import annotations

import argparse
import sys
from .artifacts import (
    generate_artifact_via_llm,
    load_artifacts,
    write_builtin_artifacts,
)
from .config import config_summary, load_config, write_default_config
from .io_utils import write_json
from .llm_client import OpenRouterClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tst")
    parser.add_argument("--config", default="configs/default.json")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("write-default-config")
    sub.add_parser("write-builtin-artifacts")
    sub.add_parser("smoke-openrouter")
    sub.add_parser("smoke-model")
    generate_extraction = sub.add_parser("generate-extraction")
    generate_extraction.add_argument("--traits", nargs="*", default=None)
    generate_extraction.add_argument("--append", action="store_true")

    judge_extraction = sub.add_parser("judge-extraction")
    judge_extraction.add_argument("--traits", nargs="*", default=None)
    judge_extraction.add_argument("--append", action="store_true")

    extract_vectors = sub.add_parser("extract-vectors")
    extract_vectors.add_argument("--traits", nargs="*", default=None)
    extract_vectors.add_argument("--skip-failed", action="store_true")
    sub.add_parser("normalize")
    sub.add_parser("evaluate")
    sub.add_parser("analyze")
    sub.add_parser("analyze-geometry")

    gen_artifacts = sub.add_parser("generate-artifacts")
    gen_artifacts.add_argument("--traits", nargs="*", default=None)

    args = parser.parse_args(argv)
    if args.command == "write-default-config":
        write_default_config(args.config)
        print(f"Wrote {args.config}")
        return 0

    config = load_config(args.config)
    config.ensure_dirs()
    write_json(config.run_path / "run_manifest.json", config_summary(config))

    if args.command == "write-builtin-artifacts":
        write_builtin_artifacts(config)
        print(f"Wrote built-in artifacts to {config.artifact_path}")
        return 0

    if args.command == "generate-artifacts":
        traits = args.traits or config.traits
        for trait_id in traits:
            artifact = generate_artifact_via_llm(config, trait_id)
            print(f"Wrote generated artifact for {artifact.trait_id}")
        return 0

    if args.command == "smoke-openrouter":
        client = OpenRouterClient(
            model=config.judge_model,
            base_url=config.openrouter_base_url,
        )
        text = client.complete_text(
            [
                {"role": "system", "content": "Return only the word ok."},
                {"role": "user", "content": "Smoke test."},
            ],
            max_tokens=32,
        )
        print(text)
        return 0

    if args.command == "smoke-model":
        from .runtime import ModelRuntime

        runtime = ModelRuntime(config)
        response = runtime.generate_response(
            runtime.make_messages("You are a helpful assistant.", "Answer in one sentence: what is 2+2?")
        )
        print(response)
        return 0

    if args.command == "generate-extraction":
        from .extraction import generate_extraction_records
        from .runtime import ModelRuntime

        artifacts = load_artifacts(config)
        runtime = ModelRuntime(config)
        generate_extraction_records(
            config, runtime, artifacts, traits=args.traits, append=args.append
        )
        return 0

    if args.command == "judge-extraction":
        from .extraction import judge_extraction_records
        from .judge import JudgeClient

        artifacts = load_artifacts(config)
        judge = JudgeClient(config)
        judge_extraction_records(
            config, judge, artifacts, traits=args.traits, append=args.append
        )
        return 0

    if args.command == "extract-vectors":
        from .extraction import extract_trait_vectors
        from .runtime import ModelRuntime

        runtime = ModelRuntime(config)
        extract_trait_vectors(
            config, runtime, traits=args.traits, skip_failed=args.skip_failed
        )
        return 0

    if args.command == "normalize":
        from .runtime import ModelRuntime
        from .vectors import compute_normalization_stats

        artifacts = load_artifacts(config)
        runtime = ModelRuntime(config)
        compute_normalization_stats(config, runtime, artifacts)
        return 0

    if args.command == "evaluate":
        from .judge import JudgeClient
        from .runtime import ModelRuntime

        artifacts = load_artifacts(config)
        runtime = ModelRuntime(config)
        judge = JudgeClient(config)
        evaluate_msg = "This stage can be expensive; confirm beta grid before running."
        print(evaluate_msg, file=sys.stderr)
        from .evaluation import run_evaluation

        run_evaluation(config, runtime, judge, artifacts)
        return 0

    if args.command == "analyze":
        from .analysis import analyze_results

        analyze_results(config)
        return 0

    if args.command == "analyze-geometry":
        from .analysis import analyze_geometry

        analyze_geometry(config)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
