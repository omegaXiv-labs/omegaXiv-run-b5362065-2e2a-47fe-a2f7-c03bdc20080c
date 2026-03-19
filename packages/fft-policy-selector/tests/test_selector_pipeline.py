from __future__ import annotations

from pathlib import Path
import tempfile

from fft_policy_selector import ScenarioConditionedSelector, run_sympy_checks, selector_policy


def test_pipeline_and_selector_exports() -> None:
    config = {
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
    selector = ScenarioConditionedSelector(
        baselines=[
            "global_winner_static_policy",
            "per_dimension_static_policy",
            "unconstrained_latency_argmin",
            "confidence_unaware_feasible_selector",
            "random_feasible_policy",
        ],
        seeds=[7, 17],
    )

    outputs = selector.run(config)
    picks = selector_policy(outputs.runtime)

    assert not outputs.runtime.empty
    assert not outputs.selector_metrics.empty
    assert len(picks) == 2
    assert "constrained_regret_ms_vs_global" in outputs.selector_metrics.columns

    with tempfile.TemporaryDirectory() as tmp_dir:
        report = Path(tmp_dir) / "symbolic_report.md"
        result = run_sympy_checks(report)
        assert report.exists()
        assert bool(result["decomposition_ok"])
