"""Reusable selector APIs for scenario-conditioned Fourier method selection."""

from .analysis import (
    CIResult,
    bootstrap_ci,
    confirmatory_stratified_check,
    finite_sample_bound,
    policy_class_dominance,
    summarize_with_ci,
)
from .core import (
    Scenario,
    aggregate_metrics,
    generate_runtime_table,
    generate_scenarios,
    selector_policy,
)
from .pipeline import BenchmarkOutputs, ScenarioConditionedSelector
from .symbolic import run_sympy_checks

__all__ = [
    "BenchmarkOutputs",
    "CIResult",
    "Scenario",
    "ScenarioConditionedSelector",
    "aggregate_metrics",
    "bootstrap_ci",
    "confirmatory_stratified_check",
    "finite_sample_bound",
    "generate_runtime_table",
    "generate_scenarios",
    "policy_class_dominance",
    "run_sympy_checks",
    "selector_policy",
    "summarize_with_ci",
]
