# Reproducibility Guide

## Evaluated environment

- Python: 3.12.1
- FastAPI: 0.137.1
- Uvicorn: 0.49.0
- Pydantic: 2.13.4
- HTTPX: 0.28.1
- OpenAI Python SDK: 1.109.1
- Matplotlib: 3.11.1
- Model: `gpt-5-nano`
- Prompt schema: `dynamic-tools-v1`

Install the exact versions with:

```bash
python -m pip install -r requirements.lock
```

## Preserved experiment design

- 180 clinically inspired synthetic scenarios
- six hand-written core scenarios and 174 seeded mixed scenarios
- five matched configurations
- five repeated runs per configuration
- 4,500 retained scenario records
- scenario seed: 2026
- 10,000 scenario-cluster bootstrap samples
- bootstrap seed: 2026
- failure policy: retain and score failures
- narrative summary generation disabled in the benchmark
- structured ED inputs used in the benchmark

The five configurations are `full_dynamic`, `all_tools`, `without_safety_validation`, `without_follow_up_tracking`, and `non_agentic`.

## Live-model reproduction

Set `OPENAI_API_KEY` locally, then run:

```bash
python evaluation/run_reproducible_experiment.py --runs 5 --count 180 --scenario-seed 2026 --bootstrap-samples 10000 --bootstrap-seed 2026 --concurrency 10 --output-dir evaluation_outputs/main
```

The runner checkpoints each scenario locally and supports `--resume`. Checkpoints are intentionally excluded from this public repository because the preserved scenario-level result file contains all 4,500 scored records needed to inspect the reported analysis.

Hosted model behavior and API latency can change. Therefore, a new live-model run may not reproduce the preserved responses byte for byte even when the code, prompt schema, and seeds are unchanged.

## Preserved reader-facing results

- `results/scenario_results.jsonl`: all 4,500 retained scenario/configuration/run records, including predicted actions, targets, selected tools, safety-validation outcomes, and scored metrics.
- `results/experiment_manifest.json`: run configuration, model roles, seeds, environment, and command.
- `results/summary_metrics.csv`: point estimates and 95% scenario-cluster bootstrap intervals.
- `results/bootstrap_intervals.csv`: paired configuration differences and confidence intervals.
- `results/failure_report.csv`: retained execution failures and classifications.
- `results/tool_selection_report.csv`: selected and skipped adapters for every evaluated run.
- `results/empirical_safety_performance.csv`: bounded validation and correction outcomes.
- `results/weighted_error_sensitivity.csv`: the 27-combination weighted-error sensitivity analysis.
- `results/benchmark_figure_counts.csv`: source counts for Figure 3.

The large nested `experiment_results.json`, individual scenario checkpoint files, live console logs, and background-run metadata are excluded because they duplicate the included reader-facing records or expose execution history that is not required to verify the reported results.

## Regenerate the figures

```bash
python evaluation/plot_figure3.py --output-dir figures
python evaluation/plot_figure4.py results/summary_metrics.csv --output figures/figure4.png
```

Figure 3 is regenerated from `build_evaluation_scenarios(count=180, seed=2026)`. Figure 4 is regenerated from the preserved summary table.

## Scope of the evidence

The benchmark uses rule-derived software labels and synthetic ED scenarios. It evaluates software behavior and internal conformance, not clinical validity, patient outcomes, or operational effectiveness in a live ED. The benchmark oracle and safety validator use separate patient-risk calculations; their target sets agree for the 180 reported scenarios, but equivalence is not guaranteed by construction.

