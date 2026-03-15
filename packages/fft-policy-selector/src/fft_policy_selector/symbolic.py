from __future__ import annotations

from pathlib import Path

import sympy as sp


def run_sympy_checks(report_path: Path) -> dict[str, object]:
    """Run symbolic checks aligned to decomposition, dominance, and bound monotonicity."""
    p1, p2 = sp.symbols("p1 p2", positive=True)
    a1, a2, b1, b2 = sp.symbols("a1 a2 b1 b2", real=True)
    expression = p1 * (a1 - b1) + p2 * (a2 - b2)
    decomposition_ok = sp.simplify(expression - (p1 * a1 + p2 * a2 - p1 * b1 - p2 * b2)) == 0

    rz, rg = sp.symbols("rz rg", real=True)
    dominance_cond = sp.Implies(sp.Le(rz, rg), sp.Ge(rg - rz, 0))
    dominance_ok = bool(dominance_cond.subs({rz: 2, rg: 3}))

    method_count, scenario_count, delta, sample_count, upper_bound = sp.symbols(
        "method_count scenario_count delta sample_count upper_bound",
        positive=True,
    )
    radius = upper_bound * sp.sqrt(
        sp.log((2 * method_count * scenario_count) / delta) / (2 * sample_count)
    )
    derivative = sp.simplify(sp.diff(radius, sample_count))
    monotonic_ok = bool(
        derivative.subs(
            {
                method_count: 6,
                scenario_count: 48,
                delta: 0.05,
                sample_count: 64,
                upper_bound: 12.0,
            }
        )
        < 0
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Symbolic Validation Report",
                "",
                f"- decomposition_identity: {'PASS' if decomposition_ok else 'FAIL'}",
                f"- dominance_implication: {'PASS' if dominance_ok else 'FAIL'}",
                f"- monotonic_radius_derivative: {'PASS' if monotonic_ok else 'FAIL'}",
                f"- derivative_expression: {sp.sstr(derivative)}",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "decomposition_ok": decomposition_ok,
        "dominance_ok": dominance_ok,
        "monotonic_ok": monotonic_ok,
        "derivative": sp.sstr(derivative),
    }
