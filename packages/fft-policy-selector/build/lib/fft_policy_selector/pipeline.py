from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from .analysis import (
    confirmatory_stratified_check,
    finite_sample_bound,
    policy_class_dominance,
    summarize_with_ci,
)
from .core import aggregate_metrics, generate_runtime_table, generate_scenarios, selector_policy


@dataclass(frozen=True)
class BenchmarkOutputs:
    runtime: pd.DataFrame
    selector_choices: pd.DataFrame
    selector_metrics: pd.DataFrame
    ci_summary: pd.DataFrame
    policy_dominance: pd.DataFrame
    finite_sample_bound: pd.DataFrame
    confirmatory_stratified_gain: pd.DataFrame


class ScenarioConditionedSelector:
    """Primary reusable implementation of the scenario-conditioned selector methodology."""

    def __init__(self, baselines: Sequence[str], seeds: Sequence[int]) -> None:
        self._baselines = list(baselines)
        self._seeds = [int(seed) for seed in seeds]

    def run(self, config: Mapping[str, object]) -> BenchmarkOutputs:
        scenarios = generate_scenarios(config)
        methods = self._baselines + ["h2_selector"]

        runtime = generate_runtime_table(scenarios, methods=methods, seeds=self._seeds)
        selector_choices = selector_policy(runtime)
        selector_metrics = aggregate_metrics(runtime, selector_choices)
        ci_summary = summarize_with_ci(selector_metrics)
        dominance = policy_class_dominance(runtime, selector_choices)
        bounds = finite_sample_bound(
            runtime,
            method_cardinality=max(1, len(methods)),
            scenario_cardinality=max(1, len(scenarios)),
            delta=0.05,
        )
        confirm = confirmatory_stratified_check(runtime, selector_choices)

        return BenchmarkOutputs(
            runtime=runtime,
            selector_choices=selector_choices,
            selector_metrics=selector_metrics,
            ci_summary=ci_summary,
            policy_dominance=dominance,
            finite_sample_bound=bounds,
            confirmatory_stratified_gain=confirm,
        )
