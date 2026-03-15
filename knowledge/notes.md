# Knowledge Notes: Modern Fourier Transformation Benchmarking

## Scope and corpus map

This acquisition pass targets practical selection of state-of-the-art Fourier implementations under workload and hardware constraints. The corpus intentionally combines:

- Foundational algorithm papers (S01, S22, S23)
- Autotuning/code-generation systems (S02, S03, S04, S05, S11, S35)
- Distributed/HPC FFT performance papers (S06, S07, S08, S09, S24)
- NUFFT method+software papers (S12, S13, S14)
- Recent ML systems that rely on FFT primitives (S15-S20)
- Current vendor and project documentation for implementation constraints (S25-S34)

The benchmark objective in user intent is not a new Fourier theory, but reproducible and scenario-specific method selection. The corpus therefore emphasizes both numerical definitions and systems effects.

## Core formal definitions to anchor all benchmark tracks

Primary transform definitions and complexity baselines:

- DFT definition appears across S01, S02, S10, S15, S16, S20:
  - `X_k = sum_{n=0}^{N-1} x_n exp(-2*pi*i*k*n/N)`
- FFT recursive cost relation from S01:
  - `T(N) = 2T(N/2) + O(N) => O(N log N)`
- Parallel FFT decomposition model from S06/S09:
  - `T_total = T_local_fft + T_transpose/collective`
- Communication performance model in distributed settings from S07/S08:
  - `T_comm = alpha*m + beta*V`
- NUFFT approximation objective from S13 and kernel strategy from S12:
  - interpolation/kernel design controls error-speed tradeoff

Benchmark implication:

- Throughput-only ranking is insufficient. Runtime must be decomposed into compute kernels, planning, and communication terms (S03, S07, S08, S09).
- Accuracy checks must include round-trip and tolerance-constrained error metrics for each precision mode (S12, S13, S21).

## Algorithmic and software-generation lineage

S01 established asymptotic gains but did not model modern memory hierarchies.
S22 and S23 document why no single FFT factorization dominates across all dimensions and hardware.
S02/S03 and S04/S05 then operationalize this result in software terms:

- S02/S03 (FFTW): planner search over decomposition/codelet space
- S04/S05 (SPIRAL): DSL + synthesis + search for architecture-adapted kernels
- S11 (FFTX-IRIS): extends this logic to heterogeneous runtime scheduling and dynamic generation

Cross-source agreement:

- Performance portability requires empirical or generated specialization (S03, S05, S11).
- Planning/autotuning overhead is acceptable only when amortized over repeated calls (S02, S03).

Difference:

- FFTW relies on runtime planning against prebuilt codelets, whereas SPIRAL/FFTX more explicitly generate kernel structures from mathematical identities.

## Distributed FFT bottlenecks and exascale lessons

S06/S07/S08/S09/S24 converge on a dominant conclusion: communication and data movement govern large 3D FFT outcomes.

Shared claims:

- Pencil decomposition scales better than slab decomposition for large process counts (S06, S24).
- Collective implementation details can shift observed performance dramatically on GPU clusters (S08).
- Exascale-friendly FFT needs explicit communication modeling and overlap design (S07, S09).

Practical benchmark implications for this project:

- Separate single-device microbenchmarks from distributed transpose/collective benchmarks.
- Log network topology, MPI implementation/version, and collective algorithm configuration as first-class metadata.
- Use matrixed reporting by decomposition and process-grid shape rather than one aggregate number.

## NUFFT-specific insights for expanded workload coverage

S12, S13, S14 show that nonuniform transforms are operationally distinct from standard FFT:

- kernel/interpolation choice defines fidelity-performance frontier
- tolerance and point distribution materially affect performance ranking

Cross-source similarity:

- All NUFFT implementations encode approximation controls (kernel width/oversampling/tolerance) that must be normalized for fair comparisons.

Difference:

- S13 focuses optimization criterion (min-max error), S12 emphasizes practical implementation performance and memory behavior, S14 emphasizes software breadth and transform variants.

Benchmark implication:

- If NUFFT enters workload suite, treat it as separate benchmark family with its own hyperparameter normalization protocol.

## ML-era FFT usage and systems pressure

S15-S20 are not pure FFT-library papers, but they matter for modern ML workloads that consume FFT primitives heavily.

Shared pattern:

- FFT-based token/convolution mixing transforms long-context model scaling economics.
- Real gains arise only when kernel/system implementation is aligned with hardware pathways (tensor cores, memory fusion) as shown in S20.

Important contrast:

- Algorithmic substitution (S16 FNet) provides speed gains with possible quality tradeoff.
- Kernel-level systems optimization (S20 FlashFFTConv) shows large throughput gains without changing the top-level convolution identity.

Benchmark implication:

- Project should include batched long-1D convolution-like workloads to represent ML pipelines, not only classical square-grid FFT cases.

## Implementation inventory and reproducibility signals

Current reusable code/documentation candidates:

- CPU/general: FFTW (S29/S30), SPIRAL (S35)
- Distributed/HPC: heFFTe (S32), PFFT (S24)
- GPU/cross-platform: VkFFT (S31), cuFFT docs (S25), rocFFT docs (S26)
- NUFFT: FINUFFT (S33), cuFINUFFT (S34)
- Intel CPU/GPU stack: oneMKL docs/release notes (S27, S28)

Reproducibility signals observed across this corpus:

- Most projects provide code and CI-visible history (GitHub repos: S30-S35).
- Vendor docs provide explicit supported transform-type matrices and precision constraints (S25-S28).
- Primary HPC papers report decomposition and communication strategy, enabling protocol reproduction (S06-S09).

## Coverage quality and gaps

Strengths:

- Strong primary coverage for FFT algorithmic foundations, distributed exascale FFT, NUFFT, and modern ML FFT consumers.
- Includes recent (2023+) sources for recency-sensitive implementation decisions.

Gaps to close in later phases:

- More peer-reviewed direct head-to-head comparisons between cuFFT/rocFFT/oneMKL/VkFFT on identical hardware generations.
- Precision-mode-specific studies focused on BF16/FP16/TF32 error behavior for FFT/IFFT round-trip.
- Standardized statistical protocol references for confidence-interval reporting in systems benchmarks.

## Immediate implications for downstream phases

For knowledge distillation and experiment design, this corpus supports a benchmark taxonomy with orthogonal factors:

- Workload shape: 1D/2D/3D, uniform vs nonuniform
- Data type: real/complex
- Precision: FP64/FP32/FP16/BF16/TF32 where available
- Scale: single-device vs distributed
- Objective: latency, throughput, memory footprint, round-trip error

Claims should remain explicitly bounded to tested environments, consistent with user constraints and S06-S09 communication findings.


## Retry Iteration Delta (Deterministic Regeneration)

This retry pass incorporated additional primary sources (S36-S42) focused on recent distributed/GPU FFT systems and implementation studies (2022-2025).

Depth coverage delta versus prior recovery warning:
- key_equations: now populated for 32/35 primary sources
- assumptions: now populated for 31/35 primary sources
- contributions: now populated for 35/35 primary sources
- claims: now populated for 35/35 primary sources
- conclusions: now populated for 35/35 primary sources
- limitations: now populated for 35/35 primary sources
- future_work: now populated for 31/35 primary sources

Newly added sources improve recency and systems coverage for communication-aware, mixed-precision, and matrix-core/NPU-oriented FFT implementations.
