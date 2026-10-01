# SMALL training measurement record

These measurements accompany [PR #60](https://github.com/lisu188/jexperiments/pull/60) and the [implementation walkthrough](../../BLOG.md#small-specializing-tiny-networks-without-changing-the-default-engine). They target `2 → 8 → 8 → 8 → 1`; correctness tests separately cover the complete 120-shape family. Results describe the stated host, workload and API boundary, not a general CPU/GPU crossover rule.

## Results and provenance

| Record | Source revision | Host / scope | Files |
|---|---|---|---|
| Local measurements | `b987d7b5048beaf255fe8ec05500d3104b4696da` | i5-14400F, RTX 4060 Ti, Windows 11, OpenJDK 27+35-2325; interactive-host contention | [Notes](LOCAL.md), [distributions, identities and input hashes](local-results.json) |
| CI CPU/native measurements | `7965204e0ca7fda5aaa232fe4dbaae4c229a4f82` | EPYC 7763 VM, four logical processors, Windows Server 2025, Temurin 27+35 | [Notes](CI.md), [distributions and provenance](ci-results.json) |
| Local generated instructions | `b987d7b5048beaf255fe8ec05500d3104b4696da` | C2, AVX2/FMA enabled, maximum vector size 32 bytes | [Inspection notes](ASSEMBLY.md), [method summaries](assembly.json), [masked-store comparison](cohort-mask-comparison.json) |

The CI reporting revision changes benchmark tooling, not the kernels measured locally. The packaged CUDA source SHA-256 is `05b4b2a92de0d0ae91574528b7666b81b3b4fa29bb8170d3824abaf5d8143a27`; PTX SHA-256 is `637451e024c8c6b06028ab69c6aad345de0c86878648c8ed79571a5b1ca5a330`. Driver 616.92 reported CUDA Driver API 13040 and compute capability 8.9. The PTX was built ahead of time with CUDA 13.0.2; runtime fusion does not require NVRTC.

The local native DLL hash is `d398d1a8c310f809a72ccba651e925d29d8b8dad019f0a6d303cdfa51e7fa6e3`. CI rebuilt the same source with its installed MSVC toolchain and records its own artifact hash. Native is an internal benchmark candidate only: **both environments reject its promotion gate**. Its one-worker cohort must beat the strongest permitted JVM baseline, including four-worker reference execution where configured. No public native backend was added.

The [first CI measurement run](https://github.com/lisu188/jexperiments/actions/runs/36776029263) completed native acceptance, all timed forks and report validation. Its job status was failure because the expected negative native-gate exit code leaked through PowerShell. Commit `e0f7de6` makes successful, valid negative evidence return zero; invalid evidence and execution failures remain fatal. The [follow-up workflow run](https://github.com/lisu188/jexperiments/actions/runs/36777731915) verifies this reporting correction. The first run's observations are retained as their own dataset; they are not silently replaced by later measurements.

## Interpretation

- CI JMH FP64 256-bit SMALL averaged 120.636 µs per public epoch, versus 186.070 µs for reference matrix and 193.216 µs for reference scalar. The observed 1.54× mean ratio does not reach the 2× CPU stretch target. All scalar/128-/256-bit and FP32 variants, score errors and iteration distributions are retained.
- The fastest path depends on the API: legacy CPU wins the measured five-epoch bulk calls; SMALL 256-bit wins 64-epoch single-model chunk cases; the parallel reference wins four/32-model FP64 cohorts. A specialized implementation is not automatically the strongest baseline.
- Real GPU acceptance passed, but desktop fork drift prevents a clean 10× GPU claim. FP32 cohorts showed a promising observed result; FP64 cohorts regressed against the parallel CPU baseline. Keep precision, worker count, lifecycle and publication boundaries identical when comparing.
- All four math variants converged in the same 6/32 paired seeds by 10k epochs and all 32/32 by 100k. The smaller-budget failures remain censored observations; reported time-to-target statistics condition on convergence.
- Generated C2 code contains packed FMA at both widths and precisions. FP64 128-bit cohort updates also contain allocation/helper paths; default FP64 256-bit and both FP32 widths avoid those paths in the captured update methods. Scalar output and exact exponential work remain expected.

## Measurement boundaries

CLI totals include session open, training, validation/publication inside the session and close. Fresh model/data construction and the independent numerical oracle are outside that timing. `--retained-device true` retains the owned CUDA service across rounds and excludes its final destruction from each close. Open/train/close distributions are recorded separately.

Single-epoch/public-epoch measurements preserve immediate publication. Explicit `CHUNK` measurements may amortize publication while retaining the requested scoring, cancellation and epoch boundaries. Cohort timing includes packing and host checkpoints. No comparison treats unpublished epochs as completed work.

JMH measures ten public epochs per invocation and normalizes per epoch. Invocation setup/teardown time is excluded, but allocation profiling includes fixture allocations. JMH p95 here describes the distribution of iteration-average scores, not the tail latency of individual epochs. CLI p95 describes repeated whole workload calls. First-candidate-session calls exclude JVM startup and run after construction of the CPU oracle; process wall time also includes harness/report work. Neither is labeled pristine JVM startup.

Local CLI: three independent JVMs, 100 warmups/30 rounds for five-epoch bulk and one-epoch calls; 20/20 for 64-epoch chunks; 10/15 for cohorts. CI uses the same bulk/chunk counts and 10/20 for cohorts. JMH uses three forks, five one-second warmups and five one-second measurements in both environments. The cuBLAS comparison has only one fork and remains outside the multi-fork qualification. EXACT is the throughput default; the paired-seed fixture evaluates FAST separately.

## Reproduce

This section describes the historical revisions listed above; its native tasks and kernels are no longer part of the current TensorFlow application. The original report aggregator and regression tests are retained unchanged in [analysis](analysis/aggregate_small_benchmarks.py). To analyze matching historical reports, run `python3 jneuro/benchmarks/2026-09-30-small/analysis/aggregate_small_benchmarks.py` with the original report arguments.

Use Java 27 and the existing Gradle wrapper. CPU checks need no CUDA toolkit or native DLL. The [BLOG commands](../../BLOG.md#reproducing-the-comparisons) select each boundary explicitly. Run at least three fresh JVMs, retain the environment sidecars and do not merge data across different source/runtime/hardware configurations.

```text
./gradlew :jneuro:check :jneuro:jacocoTestReport
./gradlew :jneuro:guiCheck
./gradlew :jneuro:gpuCheck
./gradlew :jneuro:nativeSmallCheck -PneuroSmallNative=C:/path/to/jneuro-small.dll
```

`guiCheck` operates real windows and requires a display; Linux CI uses Xvfb and Openbox. The last two tasks require the actual requested hardware/library and fail if absent. Ordinary tests, real GPU tests and native acceptance establish different claims and are reported independently.

For the bounded CI experiment, dispatch **JNeuro native SMALL experiment** with `benchmarks=true`. It builds with existing MSVC/Java/Gradle toolchains, checks the real DLL and runs direct Java commands after Gradle exits. Its artifact includes commands, class/library/source provenance, numerical checks and raw rounds. The runner validates every engine has three identified JVM forks. [`aggregate_small_benchmarks.py`](analysis/aggregate_small_benchmarks.py) treats missing evidence as a failed gate and includes median plus p95 against the fastest applicable FP64 baseline. The eight-engine JMH matrix is validated separately.

For instruction inspection, use HotSpot diagnostic `CompileCommand=print` for the concrete `SmallCpuTraining` and `SmallCpuCohort` methods, retain their complete C2 byte ranges, and pass the output to `jneuro/tools/inspect_small_assembly.py`. `objdump` decodes bytes even when a local hsdis plugin is absent. Never infer packed training instructions merely from `simdBits` metadata or JVM flags.

## Retained evidence

Published JSON files retain compact distributions, fork identities and raw-input hashes. Local raw reports, complete instruction captures, diagnostic failures and command/provenance files remain in `/tmp/jneuro-small-evidence`; they are not a portable dependency of the application. CI artifacts have finite retention, so this directory retains the compact measured results independently. No datasets, CUDA SDK installation or opaque compiled native library is required to read this record.

Local acceptance at the implementation revision: 265 ordinary tests, zero failures/errors and one optional BLAS skip; 97.0% JNeuro line coverage; 16 actual CUDA/cuBLAS tests and two actual native tests, all passing. PR checks additionally require the repository build, blog/container build and the current real-window GUI inventory. At `e0f7de6`, all 25 real-window tests passed and covered **127/127 GUI paths**. Only successful current GUI tests count toward that independent 90% gate. [Validation metadata](validation.json) retains the report counts, coverage and source revisions.
