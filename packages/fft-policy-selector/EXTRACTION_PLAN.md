# Extraction Plan

1. Core contribution(s)
- Scenario-conditioned feasible selector for Fourier-method policy choice under latency, error, and memory constraints.
- Empirical dominance and finite-sample bound analysis used to validate selector behavior.

2. Source symbols in repository
- experiments/h2_selector_validation/src/h2exp/core.py::selector_policy
- experiments/h2_selector_validation/src/h2exp/core.py::aggregate_metrics
- experiments/h2_selector_validation/src/h2exp/core.py::generate_scenarios
- experiments/h2_selector_validation/src/h2exp/analysis.py::policy_class_dominance
- experiments/h2_selector_validation/src/h2exp/analysis.py::finite_sample_bound

3. Public API mapping for packaged library
- fft_policy_selector.core.selector_policy -> fft_policy_selector.core.selector_policy
- fft_policy_selector.core.aggregate_metrics -> fft_policy_selector.core.aggregate_metrics
- fft_policy_selector.core.generate_scenarios -> fft_policy_selector.core.generate_scenarios
- fft_policy_selector.analysis.policy_class_dominance -> fft_policy_selector.analysis.policy_class_dominance
- fft_policy_selector.analysis.finite_sample_bound -> fft_policy_selector.analysis.finite_sample_bound
- New orchestration API -> fft_policy_selector.pipeline.ScenarioConditionedSelector
