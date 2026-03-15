from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CIResult:
    mean: float
    low: float
    high: float


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> CIResult:
    """Compute bootstrap mean confidence interval."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("values must be non-empty")

    rng = np.random.default_rng(seed)
    means: list[float] = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))

    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return CIResult(mean=float(np.mean(values)), low=low, high=high)


def summarize_with_ci(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize selector metrics with profile-level confidence intervals."""
    rows: list[dict[str, Any]] = []
    for profile, group in metrics_df.groupby("scenario_prior_profile"):
        for metric in [
            "constrained_expected_latency_ms",
            "constrained_regret_ms_vs_global",
            "feasibility_violation_rate",
            "selector_coverage_rate",
            "fallback_activation_rate",
        ]:
            ci = bootstrap_ci(group[metric].to_numpy(), seed=int(group["seed"].iloc[0]))
            rows.append(
                {
                    "scenario_prior_profile": profile,
                    "metric": metric,
                    "mean": ci.mean,
                    "ci95_low": ci.low,
                    "ci95_high": ci.high,
                    "std": float(group[metric].std(ddof=1)),
                }
            )
    return pd.DataFrame(rows)


def policy_class_dominance(runtime: pd.DataFrame, selector_pick: pd.DataFrame) -> pd.DataFrame:
    """Compute selector-vs-global risk dominance metrics."""
    merged = runtime.merge(selector_pick, on=["scenario_id", "seed"], how="left")
    selector_rows = merged[merged["method"] == merged["selected_method"]].copy()
    selector_risk = selector_rows.groupby("seed", as_index=False)["latency_ms"].mean().rename(
        columns={"latency_ms": "risk_selector_RZ_hat_ms"}
    )

    global_rows = merged[merged["method"] == "global_winner_static_policy"]
    global_risk = global_rows.groupby("seed", as_index=False)["latency_ms"].mean().rename(
        columns={"latency_ms": "risk_global_RG_hat_ms"}
    )

    joined = selector_risk.merge(global_risk, on="seed")
    joined["delta_global_ms"] = joined["risk_global_RG_hat_ms"] - joined["risk_selector_RZ_hat_ms"]
    return joined


def finite_sample_bound(
    runtime: pd.DataFrame,
    method_cardinality: int,
    scenario_cardinality: int,
    delta: float = 0.05,
    sample_count: int = 32,
) -> pd.DataFrame:
    """Estimate finite-sample robust regret bound diagnostics."""
    rows = runtime[runtime["method"].isin(["h2_selector", "global_winner_static_policy"])].copy()
    by_seed = rows.groupby(["seed", "scenario_id", "method"], as_index=False)["latency_ms"].mean()

    pivot = by_seed.pivot_table(index=["seed", "scenario_id"], columns="method", values="latency_ms").reset_index()
    pivot["empirical_regret_ms"] = (
        pivot["h2_selector"] - pivot["global_winner_static_policy"]
    ).clip(lower=0.0)

    u_s = rows.groupby("scenario_id")["latency_ms"].max().rename("U_s")
    pivot = pivot.merge(u_s, on="scenario_id", how="left")
    radius = pivot["U_s"] * np.sqrt(
        np.log((2 * method_cardinality * scenario_cardinality) / delta) / (2 * sample_count)
    )
    pivot["theoretical_bound_ms"] = 2.0 * radius
    pivot["bound_satisfied"] = (pivot["empirical_regret_ms"] <= pivot["theoretical_bound_ms"]).astype(int)
    return pivot


def confirmatory_stratified_check(runtime: pd.DataFrame, selector_pick: pd.DataFrame) -> pd.DataFrame:
    """Compute stratified post-selection gain diagnostics."""
    merged = runtime.merge(selector_pick, on=["scenario_id", "seed"], how="left")
    selected = merged[merged["method"] == merged["selected_method"]].copy()
    global_rows = merged[merged["method"] == "global_winner_static_policy"].copy()
    baseline = global_rows[["scenario_id", "seed", "latency_ms"]].rename(
        columns={"latency_ms": "global_latency_ms"}
    )
    selected = selected.merge(baseline, on=["scenario_id", "seed"], how="left")
    selected["delta"] = selected["global_latency_ms"] - selected["latency_ms"]

    return (
        selected.groupby(["dimension", "size_bucket"], as_index=False)
        .agg(mean_gain_ms=("delta", "mean"), std_gain_ms=("delta", "std"), n=("delta", "count"))
        .sort_values(["dimension", "size_bucket"])
    )
