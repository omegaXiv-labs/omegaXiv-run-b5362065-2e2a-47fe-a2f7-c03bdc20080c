# Scenario-Conditioned Selector Validation Experiments

This package executes the scenario-conditioned Fourier selector validation workflow used by the manuscript results. It evaluates decomposition exactness behavior, adaptive-versus-global policy risk, finite-sample bound calibration, and theorem-assumption stress diagnostics.

## Structure
- `src/h2exp/core.py`: scenario generation, synthetic runtime generation, selector and baseline aggregation.
- `src/h2exp/analysis.py`: bootstrap CI summaries, policy-class dominance, finite-sample bound calibration, confirmatory stratified analysis.
- `src/h2exp/plotting.py`: seaborn-styled multi-panel PDF figure generation and readability checks.
- `src/h2exp/sympy_checks.py`: symbolic consistency checks aligned to the project math-validation specification.
- `run_experiments.py`: CLI entrypoint.
- `configs/experiment_config.yaml`: sweep, seed, baseline, and metric configuration.
- `tests/test_core.py`: smoke test for the core evaluation pipeline.

## Run
```bash
. experiments/.venv/bin/activate
PYTHONPATH=experiments/h2_selector_validation/src \
  python experiments/h2_selector_validation/run_experiments.py \
  --config experiments/h2_selector_validation/configs/experiment_config.yaml \
  --output-dir experiments/h2_selector_validation/output \
  --paper-figures paper/figures \
  --paper-tables paper/tables \
  --paper-data paper/data
```

## Outputs
- Datasets under `experiments/h2_selector_validation/output/data` and `paper/data`.
- Tables under `experiments/h2_selector_validation/output/tables` and `paper/tables`.
- PDF figures under `paper/figures`.
- SymPy report under `experiments/h2_selector_validation/output/sympy`.
- Results summary at `experiments/h2_selector_validation/output/results_summary.json`.

## Scope Caveat
- Runtime values in this iteration are generated from a controlled synthetic process for reproducible calibration and stress testing.
- Claims from these outputs should be interpreted as in-simulation evidence until the same pipeline is run against measured hardware traces.
