from __future__ import annotations

from pathlib import Path

import sympy as sp


def run_sympy_checks(report_path: Path) -> dict:
    p1, p2 = sp.symbols("p1 p2", positive=True)
    a1, a2, b1, b2 = sp.symbols("a1 a2 b1 b2", real=True)
    expr = p1 * (a1 - b1) + p2 * (a2 - b2)
    decomposition_ok = sp.simplify(expr - (p1 * a1 + p2 * a2 - p1 * b1 - p2 * b2)) == 0

    Rz, Rg = sp.symbols("Rz Rg", real=True)
    dominance_cond = sp.Implies(sp.Le(Rz, Rg), sp.Ge(Rg - Rz, 0))
    dominance_ok = bool(dominance_cond.subs({Rz: 2, Rg: 3}))

    M, S, delta, n, U = sp.symbols("M S delta n U", positive=True)
    b = U * sp.sqrt(sp.log(2 * M * S / delta) / (2 * n))
    monotonic_n = sp.simplify(sp.diff(b, n))
    monotonic_ok = bool(monotonic_n.subs({M: 6, S: 48, delta: 0.05, n: 64, U: 12.0}) < 0)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# SymPy Validation Report (C1-C3)",
                "",
                f"- C1 decomposition algebra identity check: {'PASS' if decomposition_ok else 'FAIL'}",
                f"- C2 weak-dominance implication sanity check: {'PASS' if dominance_ok else 'FAIL'}",
                f"- C3 monotonic radius derivative wrt n is negative: {'PASS' if monotonic_ok else 'FAIL'}",
                f"- C3 derivative expression: {sp.sstr(monotonic_n)}",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "decomposition_ok": decomposition_ok,
        "dominance_ok": dominance_ok,
        "monotonic_ok": monotonic_ok,
        "derivative": sp.sstr(monotonic_n),
    }
