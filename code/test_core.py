from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from h2exp.core import aggregate_metrics, generate_runtime_table, generate_scenarios, selector_policy  # noqa: E402


def test_selector_pipeline_runs() -> None:
    cfg = {
        "sweep_params": {
            "dimension": ["1d"],
            "domain_type": ["real"],
            "size_bucket": ["small"],
            "batch_size": [1],
            "epsilon_scale": [1.0],
            "memory_budget_scale": [1.0],
            "scenario_prior_profile": ["uniform"],
        }
    }
    scenarios = generate_scenarios(cfg)
    methods = [
        "global_winner_static_policy",
        "per_dimension_static_policy",
        "unconstrained_latency_argmin",
        "confidence_unaware_feasible_selector",
        "random_feasible_policy",
        "h2_selector",
    ]
    runtime = generate_runtime_table(scenarios, methods, [7, 17])
    picks = selector_policy(runtime)
    metrics = aggregate_metrics(runtime, picks)

    assert not runtime.empty
    assert len(picks) == 2
    assert "constrained_expected_latency_ms" in metrics.columns
