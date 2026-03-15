from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

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


def generate_scenarios(config: dict) -> list[Scenario]:
    sweeps = config["sweep_params"]
    rows: list[Scenario] = []
    for i, vals in enumerate(
        product(
            sweeps["dimension"],
            sweeps["domain_type"],
            sweeps["size_bucket"],
            sweeps["batch_size"],
            sweeps["epsilon_scale"],
            sweeps["memory_budget_scale"],
            sweeps["scenario_prior_profile"],
        )
    ):
        dim, dom, size_bucket, batch_size, eps, mem, prior = vals
        rows.append(
            Scenario(
                scenario_id=f"S{i:04d}",
                dimension=dim,
                domain_type=dom,
                size_bucket=size_bucket,
                batch_size=int(batch_size),
                epsilon_scale=float(eps),
                memory_budget_scale=float(mem),
                scenario_prior_profile=prior,
            )
        )
    return rows


def _profile_weight(profile: str, scenario: Scenario) -> float:
    if profile == "uniform":
        return 1.0
    if profile == "inference_heavy":
        return 1.8 if scenario.size_bucket in {"small", "medium"} else 0.6
    if profile == "hpc_3d_heavy":
        return 2.2 if scenario.dimension == "3d" else 0.7
    return 1.0


def generate_runtime_table(
    scenarios: Iterable[Scenario],
    methods: list[str],
    seeds: list[int],
) -> pd.DataFrame:
    rows: list[dict] = []
    size_factor = {"small": 0.8, "medium": 1.0, "large": 1.35, "xlarge": 1.8}
    dim_factor = {"1d": 0.7, "2d": 1.0, "3d": 1.6}
    method_eff = {
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
            rng = np.random.default_rng(seed + scenario_offset)
            env_noise = float(rng.normal(0.0, 0.18))
            for method in methods:
                latency_ms = max(0.1, base * method_eff[method] + env_noise + float(rng.normal(0.0, 0.22)))
                err = abs(float(rng.normal(0.0012, 0.0004))) * (1.0 / scenario.epsilon_scale)
                err_ci = abs(float(rng.normal(0.0002, 0.00008)))
                mem = max(100.0, base * 38.0 * method_eff[method] * scenario.memory_budget_scale)
                mem_budget = base * 45.0 * scenario.memory_budget_scale
                feasible = int((err + 1.96 * err_ci <= 0.006 * scenario.epsilon_scale) and (mem <= mem_budget))
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
                        "seed": seed,
                        "method": method,
                        "latency_ms": latency_ms,
                        "error": err,
                        "error_ci": err_ci,
                        "epsilon": 0.006 * scenario.epsilon_scale,
                        "mem": mem,
                        "mem_budget": mem_budget,
                        "feasible": feasible,
                        "prior_weight": _profile_weight(scenario.scenario_prior_profile, scenario),
                    }
                )
    return pd.DataFrame(rows)


def selector_policy(runtime: pd.DataFrame) -> pd.DataFrame:
    scored = runtime.copy()
    scored["confidence_gate"] = scored["error"] + 1.96 * scored["error_ci"] <= scored["epsilon"]
    scored["feasible_gate"] = (scored["confidence_gate"]) & (scored["mem"] <= scored["mem_budget"])

    picks: list[pd.Series] = []
    for (sid, seed), group in scored.groupby(["scenario_id", "seed"], sort=False):
        feasible_group = group[group["feasible_gate"]]
        if feasible_group.empty:
            fallback = group.sort_values(["error", "latency_ms", "method"]).iloc[0]
            picks.append(
                pd.Series(
                    {
                        "scenario_id": sid,
                        "seed": seed,
                        "selected_method": fallback["method"],
                        "selected_latency_ms": fallback["latency_ms"],
                        "fallback_activated": 1,
                    }
                )
            )
        else:
            best = feasible_group.sort_values(["latency_ms", "method"]).iloc[0]
            picks.append(
                pd.Series(
                    {
                        "scenario_id": sid,
                        "seed": seed,
                        "selected_method": best["method"],
                        "selected_latency_ms": best["latency_ms"],
                        "fallback_activated": 0,
                    }
                )
            )
    return pd.DataFrame(picks)


def aggregate_metrics(runtime: pd.DataFrame, selector_df: pd.DataFrame) -> pd.DataFrame:
    merged = runtime.merge(selector_df, on=["scenario_id", "seed"], how="left")
    selected_rows = merged[merged["method"] == merged["selected_method"]].copy()

    global_rows = merged[merged["method"] == "global_winner_static_policy"].copy()
    baseline = global_rows[["scenario_id", "seed", "latency_ms"]].rename(columns={"latency_ms": "global_latency_ms"})
    selected_rows = selected_rows.merge(baseline, on=["scenario_id", "seed"], how="left")
    selected_rows["regret_ms_vs_global"] = selected_rows["selected_latency_ms"] - selected_rows["global_latency_ms"]
    selected_rows["violation"] = (1 - selected_rows["feasible"].astype(int)).astype(int)

    grouped = (
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
    return grouped
