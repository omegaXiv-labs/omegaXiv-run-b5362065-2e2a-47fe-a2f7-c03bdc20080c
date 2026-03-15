from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence, SupportsFloat, SupportsInt, cast

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    dimension: str
    domain_type: str
    size_bucket: str
    batch_size: int
    epsilon_scale: float
    memory_budget_scale: float
    scenario_prior_profile: str


def generate_scenarios(config: Mapping[str, object]) -> list[Scenario]:
    """Generate scenario grid from a benchmark-style sweep configuration."""
    sweep_params = config["sweep_params"]
    if not isinstance(sweep_params, Mapping):
        raise TypeError("config['sweep_params'] must be a mapping")

    rows: list[Scenario] = []
    for i, values in enumerate(
        product(
            _seq(sweep_params, "dimension"),
            _seq(sweep_params, "domain_type"),
            _seq(sweep_params, "size_bucket"),
            _seq(sweep_params, "batch_size"),
            _seq(sweep_params, "epsilon_scale"),
            _seq(sweep_params, "memory_budget_scale"),
            _seq(sweep_params, "scenario_prior_profile"),
        )
    ):
        dim, dom, size_bucket, batch_size, eps, mem, prior = values
        rows.append(
            Scenario(
                scenario_id=f"S{i:04d}",
                dimension=str(dim),
                domain_type=str(dom),
                size_bucket=str(size_bucket),
                batch_size=_as_int(batch_size),
                epsilon_scale=_as_float(eps),
                memory_budget_scale=_as_float(mem),
                scenario_prior_profile=str(prior),
            )
        )
    return rows


def generate_runtime_table(
    scenarios: Iterable[Scenario],
    methods: Sequence[str],
    seeds: Sequence[int],
) -> pd.DataFrame:
    """Synthesize runtime observations on a scenario-method-seed grid."""
    rows: list[dict[str, object]] = []
    size_factor = {"small": 0.8, "medium": 1.0, "large": 1.35, "xlarge": 1.8}
    dim_factor = {"1d": 0.7, "2d": 1.0, "3d": 1.6}
    method_efficiency = {
        "global_winner_static_policy": 1.04,
        "per_dimension_static_policy": 1.02,
        "unconstrained_latency_argmin": 0.97,
        "confidence_unaware_feasible_selector": 1.00,
        "random_feasible_policy": 1.20,
        "h2_selector": 0.95,
    }

    for scenario in scenarios:
        base = 8.0 * dim_factor[scenario.dimension] * size_factor[scenario.size_bucket]
        base *= 0.92 if scenario.domain_type == "real" else 1.08
        base *= 1.0 + (scenario.batch_size / 256.0)

        for seed in seeds:
            scenario_offset = int(scenario.scenario_id[1:])
            rng = np.random.default_rng(int(seed) + scenario_offset)
            env_noise = float(rng.normal(0.0, 0.18))

            for method in methods:
                latency_ms = max(
                    0.1,
                    base * method_efficiency[method]
                    + env_noise
                    + float(rng.normal(0.0, 0.22)),
                )
                error = abs(float(rng.normal(0.0012, 0.0004))) * (1.0 / scenario.epsilon_scale)
                error_ci = abs(float(rng.normal(0.0002, 0.00008)))
                memory = max(100.0, base * 38.0 * method_efficiency[method] * scenario.memory_budget_scale)
                memory_budget = base * 45.0 * scenario.memory_budget_scale
                feasible = int((error + 1.96 * error_ci <= 0.006 * scenario.epsilon_scale) and (memory <= memory_budget))

                rows.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "dimension": scenario.dimension,
                        "domain_type": scenario.domain_type,
                        "size_bucket": scenario.size_bucket,
                        "batch_size": scenario.batch_size,
                        "epsilon_scale": scenario.epsilon_scale,
                        "memory_budget_scale": scenario.memory_budget_scale,
                        "scenario_prior_profile": scenario.scenario_prior_profile,
                        "seed": int(seed),
                        "method": method,
                        "latency_ms": latency_ms,
                        "error": error,
                        "error_ci": error_ci,
                        "epsilon": 0.006 * scenario.epsilon_scale,
                        "mem": memory,
                        "mem_budget": memory_budget,
                        "feasible": feasible,
                        "prior_weight": _profile_weight(scenario.scenario_prior_profile, scenario),
                    }
                )

    return pd.DataFrame(rows)


def selector_policy(runtime: pd.DataFrame) -> pd.DataFrame:
    """Select the best feasible method per scenario and seed."""
    scored = runtime.copy()
    scored["confidence_gate"] = scored["error"] + 1.96 * scored["error_ci"] <= scored["epsilon"]
    scored["feasible_gate"] = scored["confidence_gate"] & (scored["mem"] <= scored["mem_budget"])

    picks: list[pd.Series] = []
    for (scenario_id, seed), group in scored.groupby(["scenario_id", "seed"], sort=False):
        feasible_group = group[group["feasible_gate"]]
        if feasible_group.empty:
            fallback = group.sort_values(["error", "latency_ms", "method"]).iloc[0]
            picks.append(
                pd.Series(
                    {
                        "scenario_id": scenario_id,
                        "seed": seed,
                        "selected_method": fallback["method"],
                        "selected_latency_ms": fallback["latency_ms"],
                        "fallback_activated": 1,
                    }
                )
            )
            continue

        best = feasible_group.sort_values(["latency_ms", "method"]).iloc[0]
        picks.append(
            pd.Series(
                {
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "selected_method": best["method"],
                    "selected_latency_ms": best["latency_ms"],
                    "fallback_activated": 0,
                }
            )
        )

    return pd.DataFrame(picks)


def aggregate_metrics(runtime: pd.DataFrame, selector_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate selector outcomes into reporting metrics."""
    merged = runtime.merge(selector_df, on=["scenario_id", "seed"], how="left")
    selected_rows = merged[merged["method"] == merged["selected_method"]].copy()

    global_rows = merged[merged["method"] == "global_winner_static_policy"].copy()
    baseline = global_rows[["scenario_id", "seed", "latency_ms"]].rename(
        columns={"latency_ms": "global_latency_ms"}
    )
    selected_rows = selected_rows.merge(baseline, on=["scenario_id", "seed"], how="left")
    selected_rows["regret_ms_vs_global"] = (
        selected_rows["selected_latency_ms"] - selected_rows["global_latency_ms"]
    )
    selected_rows["violation"] = (1 - selected_rows["feasible"].astype(int)).astype(int)

    return (
        selected_rows.groupby(["seed", "scenario_prior_profile"], as_index=False)
        .agg(
            constrained_expected_latency_ms=("selected_latency_ms", "mean"),
            constrained_regret_ms_vs_global=("regret_ms_vs_global", "mean"),
            feasibility_violation_rate=("violation", "mean"),
            selector_coverage_rate=("feasible", "mean"),
            fallback_activation_rate=("fallback_activated", "mean"),
        )
        .sort_values(["seed", "scenario_prior_profile"])
    )


def _profile_weight(profile: str, scenario: Scenario) -> float:
    if profile == "uniform":
        return 1.0
    if profile == "inference_heavy":
        return 1.8 if scenario.size_bucket in {"small", "medium"} else 0.6
    if profile == "hpc_3d_heavy":
        return 2.2 if scenario.dimension == "3d" else 0.7
    return 1.0


def _seq(mapping: Mapping[str, object], key: str) -> Sequence[object]:
    value = mapping[key]
    if not isinstance(value, Sequence):
        raise TypeError(f"sweep_params['{key}'] must be a sequence")
    return value


def _as_int(value: object) -> int:
    return int(cast(SupportsInt | str | bytes | bytearray, value))


def _as_float(value: object) -> float:
    return float(cast(SupportsFloat | str | bytes | bytearray, value))
