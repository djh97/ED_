from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agentic_system import EDOrchestrationAgent, RuleBasedEDBaseline, build_evaluation_scenarios  # noqa: E402
from app.experiment import ExperimentConfig, run_matched_experiment, write_experiment_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the matched, reproducible ED experiment pipeline.")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--scenario-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--count", type=int, default=180)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of independent configuration/run jobs to execute concurrently.",
    )
    parser.add_argument("--output-dir", default=str(ROOT / "evaluation_outputs"))
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Checkpoint directory. Defaults to <output-dir>/checkpoints.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse validated scenario checkpoints from an interrupted run.",
    )
    parser.add_argument(
        "--allow-checkpoint-writer-upgrade",
        action="store_true",
        help=(
            "Allow a recorded checkpoint-writer-only source revision while still requiring "
            "identical models, prompts, scenarios, seeds, runs, and configurations."
        ),
    )
    parser.add_argument(
        "--offline-smoke-test",
        action="store_true",
        help="Exercise the pipeline with deterministic runners; results are not LLM study results.",
    )
    args = parser.parse_args()

    config = ExperimentConfig(
        runs=args.runs,
        scenario_seed=args.scenario_seed,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        concurrency=args.concurrency,
    )
    scenarios = build_evaluation_scenarios(count=args.count, seed=args.scenario_seed)
    if args.offline_smoke_test:
        factories = {
            "full_dynamic": lambda _: RuleBasedEDBaseline(),
            "all_tools": lambda _: RuleBasedEDBaseline(),
            "non_agentic": lambda _: RuleBasedEDBaseline(),
        }
    else:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("The real experiment requires OPENAI_API_KEY. Use --offline-smoke-test only for pipeline verification.")
        factories = {
            "full_dynamic": lambda _: EDOrchestrationAgent(
                use_llm_summary=False,
                use_llm_input=False,
                enable_dynamic_tool_selection=True,
            ),
            "all_tools": lambda _: EDOrchestrationAgent(
                use_llm_summary=False,
                use_llm_input=False,
                enable_dynamic_tool_selection=False,
            ),
            "without_safety_validation": lambda _: EDOrchestrationAgent(
                use_llm_summary=False,
                use_llm_input=False,
                enable_dynamic_tool_selection=True,
                enable_safety_validation=False,
            ),
            "without_follow_up_tracking": lambda _: EDOrchestrationAgent(
                use_llm_summary=False,
                use_llm_input=False,
                enable_dynamic_tool_selection=True,
                enable_follow_up_tracking=False,
            ),
            "non_agentic": lambda _: RuleBasedEDBaseline(),
        }
    checkpoint_dir = args.checkpoint_dir or str(Path(args.output_dir) / "checkpoints")
    result = run_matched_experiment(
        scenarios,
        factories,
        config,
        checkpoint_dir=checkpoint_dir,
        resume=args.resume,
        allow_checkpoint_writer_upgrade=args.allow_checkpoint_writer_upgrade,
    )
    paths = write_experiment_outputs(result, args.output_dir)
    paths["checkpoint_index"] = str(Path(checkpoint_dir) / "checkpoint_index.json")
    print(json.dumps({"manifest": result["manifest"], "outputs": paths}, indent=2))


if __name__ == "__main__":
    main()
