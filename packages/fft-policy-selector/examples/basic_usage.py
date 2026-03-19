from __future__ import annotations

from fft_policy_selector import ScenarioConditionedSelector


def main() -> None:
    config = {
        "sweep_params": {
            "dimension": ["1d", "2d"],
            "domain_type": ["real", "complex"],
            "size_bucket": ["small", "medium"],
            "batch_size": [1, 8],
            "epsilon_scale": [1.0],
            "memory_budget_scale": [1.0],
            "scenario_prior_profile": ["uniform", "inference_heavy"],
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
        seeds=[7, 17, 29],
    )
    outputs = selector.run(config)

    print("rows_runtime:", len(outputs.runtime))
    print("rows_metrics:", len(outputs.selector_metrics))
    print(outputs.selector_metrics.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
