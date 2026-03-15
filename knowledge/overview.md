# Literature Overview: Modern Fourier Transformation Methods

This distillation synthesizes the collected corpus (S01-S42) around the practical goal defined in user input: reproducible, scenario-specific selection of state-of-the-art Fourier pipelines rather than a single global winner. The central pattern across the literature is that algorithmic complexity classes are necessary but not sufficient for performance prediction. Across modern workloads, observed outcomes are dominated by interactions among transform decomposition, planning strategy, memory hierarchy, communication topology, and numeric precision constraints.

## 1) Foundational algorithms and enduring complexity assumptions

The foundational thread starts with Cooley-Tukey (S01), which formalized the divide-and-conquer FFT and established the transition from direct DFT cost to O(N log N) recurrence under factorizable sizes. Tutorial and historical surveys (S22, S23) reinforce two durable points: first, no single radix/factorization dominates all size regimes; second, implementation details can invert expected rankings from pure arithmetic counts.

Cross-paper equation-level consensus in this category is strong. The canonical DFT form appears repeatedly (S01, S02, S10, S15, S16, S20), and the key recurrence relation in S01 remains the theoretical anchor. However, there is also a consistent limitation: foundational works model arithmetic operations more directly than cache, memory bandwidth, synchronization, and launch overhead. This gap motivates later software-architecture papers.

## 2) Autotuning and program generation for portability

FFTW and SPIRAL families (S02-S05), extended by FFTX-IRIS (S11) and the SPIRAL codebase record (S35), collectively shift FFT research from static algorithm choice to adaptive synthesis/selection. S02/S03 place empirical planning at the center, using runtime plan search and codelets, while S04/S05 model transform identities in a DSL and generate implementation candidates. S11 extends this idea into heterogeneous runtimes where generation and scheduling are both dynamic.

Equation-level contrast is clear:
- S03 frames planner objectives as empirical cost minimization over candidate plans.
- S04/S05 frame decomposition as structured algebra and search over generated formulas.
- S11 adds a runtime-level objective balancing generation and execution costs on heterogeneous devices.

Assumption-level consensus: architecture-specific behavior is too strong for one fixed decomposition strategy, so empirical or generated specialization is necessary. Contradiction-level nuance: these works differ on where adaptation should occur. FFTW-style adaptation happens mostly in planner/runtime selection over prebuilt components; SPIRAL-style adaptation pushes deeper into code synthesis from symbolic decompositions. In practice this creates a methodological gap for benchmarking: if a benchmark compares “library A vs library B” without accounting for planning budget and autotuning warm-up, results are often confounded by setup policy rather than steady-state kernel quality.

## 3) Distributed and exascale FFT: communication as the dominant term

The most consistent cross-paper conclusion in the corpus appears in distributed 3D FFT works (S06-S09, S24, S36, S37, S41, S42): communication dominates at scale. S06 introduces practical pencil-decomposition scaling; S07 formalizes communication complexity implications; S08 and S42 analyze collectives and bandwidth behavior on multi-GPU systems; S09 (heFFTe) and S24 (PFFT context) provide implementation evidence; S36 proposes an alternative block tensor-matrix DFT path to reduce unfavorable all-to-all behaviors in specific networks; S37 shows exascale pseudo-spectral workloads where FFT remains central; S41 adds mixed-precision communication compression and normalization.

Equation-level consensus:
- Decomposition of total runtime into local compute and communication terms appears repeatedly (S06, S09, S42).
- Latency-bandwidth models (alpha-beta style) recur in S07, S08, S36.

Assumption-level consensus:
- At high node/GPU counts, transpose/collective costs dominate local FFT kernels.
- Process-grid/decomposition choices materially affect scaling.

Contradictions and boundary conditions:
- S06/S24/S09 usually motivate FFT-centered distributed pipelines with improved decomposition/collectives.
- S36 indicates regimes where an algorithm with higher arithmetic burden can win because it changes communication structure.
- S41 argues mixed-precision exchange plus normalization can improve throughput while keeping acceptable accuracy, but this depends on workload tolerance and spectral characteristics.

Methodological gap exposed by these contradictions: many published comparisons are platform-specific and difficult to transfer. Network topology, MPI implementation, process placement, and collective algorithm details are often non-portable. Therefore, a benchmark claiming broad superiority must separate algorithmic effects from runtime/network policy effects, matching the user guardrails.

## 4) GPU specialization and emerging accelerator mappings

Recent implementation-centric papers and docs (S10, S25, S26, S27, S28, S31, S39, S40) broaden the landscape beyond CPU-centric assumptions. S10 (VkFFT) argues cross-API, open-source GPU FFT can approach or exceed vendor alternatives on selected cases. Vendor docs (S25-S28) clarify capability matrices and version-sensitive optimizations. S39 pushes matrix-core-aware multidimensional FFT kernels and reports gains on MI250 against selected baselines. S40 studies NPU-oriented mappings for edge SoCs and energy trade-offs.

Consensus:
- Hardware mapping details (tensor/matrix cores, memory layouts, instruction sets, runtime versions) strongly influence ranking.
- Capability support is non-uniform across libraries, especially for dimension/precision combinations.

Contradictions:
- S10 and S39 both report strong results against incumbent libraries, but they are evaluated on different hardware/software stacks and with different workload slices.
- Vendor docs provide feature claims but do not provide neutral cross-vendor methodology.

Methodological gap:
- There is no broadly accepted 2024-2026 matched-hardware, matched-software, statistically stable, cross-vendor FFT benchmark covering both accuracy and systems metrics across 1D/2D/3D, real/complex, and mixed precision.

## 5) NUFFT and nonuniform workloads as a distinct regime

NUFFT papers/software (S12-S14, plus code artifacts S33-S34) consistently show that nonuniform transforms should not be treated as a simple extension of uniform FFT benchmarks. S13 develops min-max interpolation criteria for error control. S12 introduces exponential-of-semicircle kernel choices and demonstrates practical speed/accuracy behavior. S14 provides broad software support for nonequispaced transforms.

Equation-level comparison:
- S13 emphasizes interpolation optimization objective (worst-case error minimization).
- S12 emphasizes kernel definition and implementation behavior under tolerances.
- S14 emphasizes software breadth and transform variants.

Consensus:
- Tolerance, kernel width, oversampling, and point distribution are first-class experimental variables.

Methodological gap:
- Uniform FFT benchmark protocols cannot be reused unchanged for NUFFT. Separate normalization and fairness rules are required, including tolerance alignment and distribution-controlled datasets.

## 6) ML-era Fourier operators and long-sequence workloads

ML papers (S15-S20) are not library benchmarks, but they materially influence benchmark scope. S15 (FNO) and S16/S17 (Fourier token mixers) show Fourier-domain operators as scalable alternatives in model design. S18/S19/S20 broaden this with long-sequence architectures and efficient convolution/token mixing paths, including FlashFFTConv (S20), where system-level kernel engineering can dominate end-to-end gains.

Cross-paper implication:
- If benchmarking aims to guide modern ML practitioners, workload classes must include long 1D batched transforms and convolution-like patterns, not only square scientific grids.

Contradiction and trade-off:
- Some works achieve speed from architectural simplification (potential quality shifts), while others maintain mathematical objectives but optimize kernels/runtime. This distinction matters for attribution in benchmark conclusions.

## 7) Reproducibility and evidence quality across the corpus

Open code artifacts (S30-S35) improve reproducibility potential, and vendor docs (S25-S28) provide authoritative capability constraints. Yet reproducibility remains uneven due to missing standardized reporting for:
- planning/autotuning budget and warm-up policy,
- process/grid decomposition and communication configuration,
- exact runtime/library versions and compiler flags,
- statistical treatment of run-to-run variability,
- precision-specific error targets and acceptance thresholds.

The corpus repeatedly supports a decomposition-oriented benchmark protocol: report compute kernels, planning, and communication components separately; report numerical fidelity with explicit tolerance and scaling choices; and scope claims to tested environments.

## Consensus map

High-confidence consensus across sources:
1. FFT algorithmic complexity advantage is foundational but insufficient to predict real performance (S01-S03, S22-S23).
2. Autotuning or generated specialization is central for portability (S02-S05, S11).
3. Communication dominates distributed 3D FFT at large scale (S06-S09, S41-S42).
4. Hardware/runtime/version details can change rankings materially (S10, S25-S28, S39-S40).
5. NUFFT requires separate experimental treatment due to different error-control mechanisms (S12-S14).

## Contradiction map

Primary contradictions that should drive research design:
1. FFT-centric distributed approaches versus communication-altered alternatives like S3DFT in specific network regimes (S09 vs S36).
2. Cross-platform open implementations (S10) versus vendor-tuned stacks (S25-S28): superiority claims depend on workload slices and software versions.
3. Mixed-precision speedups (S41) versus strict numerical fidelity requirements in sensitive workloads.

## Methodological gaps and distillation outcome

The strongest unresolved gap is not lack of algorithms; it is lack of controlled, reproducible, cross-stack evidence that disentangles algorithm choice from implementation/runtime/network confounders. For the user’s stated objective, the distilled literature indicates that the right output is a workload-conditioned recommendation matrix with confidence intervals and explicit scope labels, not a single “best FFT library.”

A robust next-phase methodology should therefore include:
- workload taxonomy partitions (uniform FFT vs NUFFT; 1D/2D/3D; real/complex; batch structure; precision mode),
- standardized preprocessing and normalization,
- explicit plan/warm-up accounting,
- communication-aware distributed reporting,
- scenario-specific ranking under both performance and fidelity constraints.

This aligns directly with user constraints and with the strongest cross-paper evidence from S02-S03, S06-S09, S12-S14, S25-S28, S39-S42.
