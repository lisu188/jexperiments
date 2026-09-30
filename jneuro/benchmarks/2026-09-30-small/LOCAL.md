# Local SMALL results — contention-sensitive observations

Source `b987d7b`; Windows 11, Java 27, Intel Family 6 Model 183; RTX 4060 Ti. These local measurements are separate from CI and the earlier UNC-DLL diagnostic pass. Reports are schema-complete and numerical checks passed for their measured trajectories. External desktop workload and large fork drift prevent unqualified performance claims.

The native candidate **fails the retention gate on all nine required warmed workloads**. The gate uses total open + training + close time and the fastest measured applicable FP64 JVM engine per workload; native must improve the median by at least 20% without a p95 regression on every case.

| Warm workload | Strongest FP64 JVM | JVM median / p95 ms | Native median / p95 ms | Native/JVM median |
| --- | --- | ---: | ---: | ---: |
| `cohort:2x8x8x8x1:s128:e64:b16:m32:w4:FP64:EXACT` | REFERENCE_PARALLEL | 91.336 / 126.942 | 121.422 / 154.058 | 1.329 |
| `cohort:2x8x8x8x1:s128:e64:b64:m1:w4:FP64:EXACT` | SMALL_SEQUENTIAL | 3.097 / 4.288 | 8.363 / 11.887 | 2.700 |
| `cohort:2x8x8x8x1:s128:e64:b64:m32:w4:FP64:EXACT` | REFERENCE_PARALLEL | 95.347 / 121.797 | 124.341 / 150.584 | 1.304 |
| `cohort:2x8x8x8x1:s128:e64:b64:m4:w4:FP64:EXACT` | REFERENCE_PARALLEL | 6.418 / 16.592 | 9.802 / 10.706 | 1.527 |
| `json:2x8x8x8x1:s128:e1:b64:EPOCH:EXACT:retained=true` | SMALL_256_FP64 | 0.108 / 0.194 | 2.203 / 3.111 | 20.317 |
| `json:2x8x8x8x1:s128:e5:b16:MINIBATCH:EXACT:retained=true` | CPU_LEGACY | 0.685 / 8.981 | 18.407 / 28.637 | 26.870 |
| `json:2x8x8x8x1:s128:e5:b64:MINIBATCH:EXACT:retained=true` | CPU_LEGACY | 0.671 / 10.514 | 12.153 / 24.029 | 18.099 |
| `json:2x8x8x8x1:s128:e64:b16:CHUNK:EXACT:retained=true` | SMALL_256_FP64 | 27.392 / 51.879 | 56.797 / 88.571 | 2.074 |
| `json:2x8x8x8x1:s128:e64:b64:CHUNK:EXACT:retained=true` | SMALL_256_FP64 | 20.270 / 39.760 | 55.281 / 76.290 | 2.727 |

Cohort scope is 64 epochs/model, 128 samples/model, requested maximum four CPU workers. Native uses one worker; REFERENCE_PARALLEL can use four. That difference remains visible and does not partition the retention gate. A four-model native cohort also misses the 20% requirement against the strongest single-worker JVM engine: 9.802 ms versus SMALL_SEQUENTIAL 11.596 ms (only about 15.5% lower observed median).

## Drift and GPU observations

- Single-model MINIBATCH batch 64 fused FP64 CUDA fork total medians: 44.364, 1.937, 53.360 ms (27.55× max/min); FP32: 31.711, 1.648, 49.126 ms (29.81×). These are not clean speedup comparisons.
- CHUNK batch 64 SMALL_256_FP64 fork medians: 5.927, 28.927, 27.387 ms (4.88×). Thus the variation affects CPU as well as GPU paths.
- For 32 models and batch 64, observed FP64 total median/p95: REFERENCE_PARALLEL 95.347/121.797 ms, SMALL_CPU 128.309/157.981 ms, SMALL_CUDA 103.026/192.069 ms.
- Separate 32-model FP32 batch-64 observations: SMALL_CPU 73.944/90.218 ms, SMALL_CUDA 22.747/32.943 ms, SMALL_SEQUENTIAL 160.772/191.739 ms. These runs occurred separately from FP64 and do not establish a controlled cross-precision speedup.
- Both batch 16 and 64 process sixteen eight-sample tiles per 128-sample epoch. Batch 64 has fewer update phases; no extra tile arithmetic explains its occasional larger timings. Clock/power, scheduling and publication effects are not isolated.

## Cold call and one-fork cuBLAS observations

Cold call totals include session open + five-epoch MINIBATCH training + close; zero warmups, one sample/JVM, three JVMs, batch 64. They exclude JVM launch and fresh model/data construction. The harness also trains a CPU reference before each measured candidate, even with zero configured warmups; shared CPU code can already be initialized. These are first candidate-session calls, not pristine-JVM startup measurements.

| Engine | Cold median ms | Cold p95 ms (three observations) |
| --- | ---: | ---: |
| CPU | 3.407 | 3.479 |
| CUDA | 228.432 | 231.411 |
| NATIVE_FP64 | 65.898 | 65.900 |
| SMALL_256_FP64 | 34.276 | 36.821 |
| SMALL_CUDA_FP64 | 204.409 | 211.382 |

cuBLAS is kept outside the main aggregate: one JVM, three warmups, nine rounds, five epochs. Batch 16 CPU/FP64-cuBLAS/FP32-cuBLAS median totals were 0.633/608.296/622.064 ms; batch 64 was 0.549/680.411/679.103 ms. This backend compiles/allocates/transfers within training calls; the data cannot characterize steady-state kernel throughput.

## Paired convergence

All runs use the same 32 paired seeds, XOR topology 2→8→8→8→1, target RMSE 0.05 and checks every 25 epochs. Times below are conditional on convergence; failures remain in the denominator.

| Budget | Variant | Converged / expected | Censored | Conditional median / p95 ms | Conditional median / p95 epochs |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10000 | FP64/EXACT | 6 / 32 | 26 | 16.908 / 33.670 | 7100 / 8225 |
| 10000 | FP64/FAST | 6 / 32 | 26 | 16.422 / 30.685 | 7100 / 8225 |
| 10000 | FP32/EXACT | 6 / 32 | 26 | 16.091 / 20.409 | 7100 / 8225 |
| 10000 | FP32/FAST | 6 / 32 | 26 | 16.346 / 27.413 | 7100 / 8225 |
| 100000 | FP64/EXACT | 32 / 32 | 0 | 60.988 / 251.844 | 16237.5 / 53150 |
| 100000 | FP64/FAST | 32 / 32 | 0 | 63.186 / 257.933 | 16237.5 / 53275 |
| 100000 | FP32/EXACT | 32 / 32 | 0 | 62.156 / 310.315 | 16237.5 / 53150 |
| 100000 | FP32/FAST | 32 / 32 | 0 | 57.092 / 259.802 | 16237.5 / 53275 |

The same seeds converge or remain censored for all four variants at both budgets. This small XOR fixture supports the recorded quality comparison only; timing differences remain host-sensitive.

## JMH public epoch

Eight engines, three JVM forks each, five one-second warmups and five one-second measurements; EXACT, batch 64, 128 samples. Scores are microseconds per epoch after OperationsPerInvocation(10). The percentile column summarizes iteration-average scores, **not** per-epoch tail latency. Setup and close are excluded from method timing.

| Engine | Mean µs/epoch ± JMH score error | Median / p95 iteration-average µs | Fork median max/min | GC B/epoch including fixture |
| --- | ---: | ---: | ---: | ---: |
| REFERENCE_SCALAR | 266.439 ± 7.760 | 267.650 / 278.035 | 1.049 | 4911.9 |
| REFERENCE_MATRIX | 261.077 ± 78.562 | 245.145 / 523.182 | 1.113 | 7714.5 |
| SMALL_SCALAR | 218.376 ± 8.902 | 214.325 / 231.775 | 1.080 | 13500.8 |
| SMALL_128 | 182.567 ± 15.275 | 176.971 / 219.821 | 1.138 | 13495.6 |
| SMALL_256 | 179.358 ± 18.422 | 172.075 / 221.269 | 1.183 | 13494.0 |
| SMALL_FP32_SCALAR | 212.898 ± 6.980 | 213.396 / 229.046 | 1.036 | 13029.6 |
| SMALL_FP32_128 | 168.092 ± 17.830 | 166.988 / 217.257 | 1.165 | 13026.7 |
| SMALL_FP32_256 | 163.033 ± 8.388 | 165.597 / 176.096 | 1.104 | 13040.1 |

**GC caveat:** the profiler includes Level.Invocation fixture allocations (new model, data and session) even though fixture time is excluded from the benchmark method. B/epoch is not a training-only allocation measurement. JMH fork data is kept separate from CLI totals and the native retention gate.

The JSON retains all engine median/p95 distributions, every per-fork summary and identity, requested/actual worker distinctions, paired comparisons and input hashes. Raw local reports are retained in `/tmp/jneuro-small-evidence/bench/`; their hashes are in `local-results.json`. No CI results were mixed in; no source files or benchmark inputs were changed.
