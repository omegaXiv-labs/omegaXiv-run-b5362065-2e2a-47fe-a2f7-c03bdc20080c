from __future__ import annotations

from pathlib import Path

import fitz
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns


def apply_theme() -> None:
    sns.set_theme(style="whitegrid", palette="colorblind", context="talk")


def save_selector_panels(metrics_df: pd.DataFrame, ci_df: pd.DataFrame, out_pdf: Path) -> None:
    apply_theme()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    sns.barplot(
        data=metrics_df,
        x="scenario_prior_profile",
        y="constrained_expected_latency_ms",
        estimator="mean",
        errorbar=("ci", 95),
        ax=axes[0],
    )
    axes[0].set_title("Constrained Latency by Prior Profile")
    axes[0].set_xlabel("Scenario Prior Profile")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].legend(handles=[Patch(facecolor="#4c72b0", label="Mean Across Seeds")], loc="upper right")

    sns.barplot(
        data=metrics_df,
        x="scenario_prior_profile",
        y="constrained_regret_ms_vs_global",
        estimator="mean",
        errorbar=("ci", 95),
        ax=axes[1],
    )
    axes[1].axhline(0.0, color="black", linewidth=1.2, label="Parity")
    axes[1].set_title("Regret vs Global Static Policy")
    axes[1].set_xlabel("Scenario Prior Profile")
    axes[1].set_ylabel("Regret (ms)")
    axes[1].legend(
        handles=[
            Patch(facecolor="#4c72b0", label="Mean Regret"),
            Line2D([0], [0], color="black", linewidth=1.2, label="Parity"),
        ],
        loc="upper right",
    )

    tmp = ci_df[ci_df["metric"] == "feasibility_violation_rate"]
    sns.lineplot(
        data=tmp,
        x="scenario_prior_profile",
        y="mean",
        marker="o",
        ax=axes[2],
        label="Mean Violation Rate",
    )
    for idx, row in tmp.reset_index(drop=True).iterrows():
        lbl = "95% CI" if idx == 0 else None
        axes[2].plot([idx, idx], [row["ci95_low"], row["ci95_high"]], color="black", linewidth=2, label=lbl)
    axes[2].set_title("Feasibility Violation Rate (95% CI)")
    axes[2].set_xlabel("Scenario Prior Profile")
    axes[2].set_ylabel("Violation Rate")
    axes[2].legend(loc="upper right")

    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


def save_dominance_panels(dominance_df: pd.DataFrame, out_pdf: Path) -> None:
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    sns.violinplot(data=dominance_df, y="delta_global_ms", ax=axes[0], color="#72B6A1")
    axes[0].axhline(0.0, color="black", linewidth=1.2, label="Weak dominance boundary")
    axes[0].set_title("Policy-Class Dominance Gap")
    axes[0].set_xlabel("Selector vs Global Class")
    axes[0].set_ylabel("Delta_global (ms)")
    axes[0].legend(
        handles=[
            Patch(facecolor="#72B6A1", label="Delta Distribution"),
            Line2D([0], [0], color="black", linewidth=1.2, label="Weak Dominance Boundary"),
        ],
        loc="upper right",
    )

    sns.scatterplot(data=dominance_df, x="risk_selector_RZ_hat_ms", y="risk_global_RG_hat_ms", hue="seed", ax=axes[1])
    lo = min(dominance_df["risk_selector_RZ_hat_ms"].min(), dominance_df["risk_global_RG_hat_ms"].min())
    hi = max(dominance_df["risk_selector_RZ_hat_ms"].max(), dominance_df["risk_global_RG_hat_ms"].max())
    axes[1].plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.2, label="Parity")
    axes[1].set_title("Risk Comparison by Seed")
    axes[1].set_xlabel("Selector Risk R_Z (ms)")
    axes[1].set_ylabel("Global Risk R_G (ms)")
    axes[1].legend(loc="upper left")

    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


def save_bound_panels(bound_df: pd.DataFrame, out_pdf: Path) -> None:
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    sns.scatterplot(data=bound_df, x="theoretical_bound_ms", y="empirical_regret_ms", hue="seed", ax=axes[0])
    lim = max(bound_df["theoretical_bound_ms"].max(), bound_df["empirical_regret_ms"].max())
    axes[0].plot([0, lim], [0, lim], linestyle="--", color="black", linewidth=1.2, label="Bound frontier")
    axes[0].set_title("Empirical Regret vs Theoretical Bound")
    axes[0].set_xlabel("Theoretical Bound (ms)")
    axes[0].set_ylabel("Empirical Regret (ms)")
    axes[0].legend(loc="upper left")

    sat = bound_df.groupby("seed", as_index=False)["bound_satisfied"].mean()
    sns.barplot(data=sat, x="seed", y="bound_satisfied", hue="seed", ax=axes[1])
    axes[1].axhline(0.95, color="black", linestyle=":", linewidth=1.2, label="95% target")
    axes[1].set_title("Bound Satisfaction Rate")
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Satisfaction Rate")
    axes[1].legend(title="Seed", loc="lower right")

    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


def save_assumption_panels(assumption_df: pd.DataFrame, boundary_df: pd.DataFrame, out_pdf: Path) -> None:
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    sns.barplot(data=assumption_df, x="check_name", y="pass_rate", hue="check_name", ax=axes[0])
    axes[0].axhline(0.99, color="black", linestyle=":", linewidth=1.2, label="99% target")
    axes[0].set_title("Assumption Guard Pass Rates")
    axes[0].set_xlabel("Assumption Check")
    axes[0].set_ylabel("Pass Rate")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(title="Check", loc="lower right")

    sns.heatmap(
        boundary_df.pivot(index="predicted", columns="actual", values="count"),
        annot=True,
        fmt=".0f",
        cmap="Blues",
        cbar=False,
        ax=axes[1],
    )
    axes[1].set_title("Counterexample Detection Matrix")
    axes[1].set_xlabel("Actual Class")
    axes[1].set_ylabel("Predicted Class")

    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


def verify_pdf_readability(pdf_path: Path) -> dict:
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    text = page.get_text("text")
    score = {
        "path": str(pdf_path),
        "pages": doc.page_count,
        "width": pix.width,
        "height": pix.height,
        "has_text": bool(text.strip()),
    }
    doc.close()
    return score
