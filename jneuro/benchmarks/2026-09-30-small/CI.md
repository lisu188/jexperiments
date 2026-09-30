# CI SMALL engine measurements — source 7965204

These results come only from the first downloaded CI artifact. No local measurement or later CI rerun is pooled. The measurements and numerical checks completed; the workflow ended with exit 1 because PowerShell propagated the expected failed native retention gate. A wrapper fix and rerun have separate provenance.

## Environment and scope

Source `7965204e0ca7fda5aaa232fe4dbaae4c229a4f82`. GitHub-hosted Windows Server 2025 (`win25-vs2026`, image `20260922.246.2`), AMD EPYC 7763 host CPU, **2 allocated cores / 4 logical processors**, Temurin OpenJDK 27+35. `UseAVX=2`, `UseFMA=true`, `MaxVectorSize=32`; Vector module enabled, heap 128 MiB–2 GiB, logging WARNING, file logging disabled. All measurements use **2→8→8→8→1**, 128 samples, EXACT sigmoid.

Native DLL SHA-256: `b7e2800cb69f6c1effd36ed72f93f27678ca1d934d05d2fdea0f96b294c732d1`. Host snapshots: 2026-09-30T20:56:14.6623270Z through 2026-09-30T21:04:31.7917361Z. Complete input hashes, benchmark class hashes, configurations and per-fork summaries are in `ci-results.json`.

## JMH: public mini-batch epochs

Three forks per engine, five 1-second warmups and five 1-second measurement iterations per fork; batch 64, one benchmark thread. The ten-epoch invocation is normalized to microseconds per epoch. Setup/open/close are excluded. Error is JMH’s **99.9% confidence half-width**. Median/p95 describe **15 iteration averages**, not individual epoch latency; nearest-rank p95 is their maximum.

| Engine | Mean ± error (µs/epoch) | Iteration-average median | Iteration-average p95 | Speedup vs best reference mean |
|---|---:|---:|---:|---:|
| REFERENCE_SCALAR | 193.216 ± 4.866 | 191.591 | 203.658 | 0.963× |
| REFERENCE_MATRIX | 186.070 ± 26.443 | 179.179 | 274.750 | 1.000× |
| SMALL_SCALAR | 172.861 ± 43.471 | 158.400 | 273.643 | 1.076× |
| SMALL_128 | 126.601 ± 2.147 | 126.390 | 130.681 | 1.470× |
| SMALL_256 | 120.636 ± 1.720 | 120.278 | 125.856 | 1.542× |
| SMALL_FP32_SCALAR | 155.954 ± 2.585 | 155.248 | 162.912 | 1.193× |
| SMALL_FP32_128 | 143.395 ± 14.930 | 140.014 | 183.856 | 1.298× |
| SMALL_FP32_256 | 128.231 ± 13.671 | 121.322 | 162.956 | 1.451× |

**SMALL_256 FP64 has the lowest mean: 120.636 µs/epoch, 1.542× the fastest reference mean (REFERENCE_MATRIX).** FP32_256 averages 128.231 µs with a much wider confidence interval; these data do not establish an FP32 speed advantage. REFERENCE_MATRIX and SMALL_SCALAR show substantial fork/iteration variation. Native is absent from this JMH table.

## Application calls: mini-batch and chunk

Values below are **training median / p95; total median / p95**, in milliseconds. Total includes open, train and close; data/model preparation and the independent parity oracle are excluded. In-session state validation and host publication remain included. Mini-batch uses 5 epochs, 100 warmups then 30 measurements per fork (90 pooled measured rounds). Chunk uses 64 epochs, 20 warmups then 20 measurements per fork (60 measured rounds). Each has three distinct JVM processes, CPU parallelism 1, rotating backend order and fixed seed 1234. These are independently scoped comparisons, not pooled with JMH.

### Mini-batch, 5 epochs, batch 16

Best tested FP64 JVM total median: **CPU_LEGACY**.

| Engine | Training median / p95 (ms) | Total median / p95 (ms) |
|---|---:|---:|
| CPU | 0.7317 / 1.1076 | 0.7552 / 1.1988 |
| CPU_LEGACY | 0.5465 / 0.9147 | 0.5629 / 0.9475 |
| NATIVE_FP64 | 1.1468 / 1.8095 | 2.7951 / 4.3052 |
| SMALL_128_FP32 | 0.6371 / 1.0454 | 0.6515 / 1.0889 |
| SMALL_128_FP64 | 0.6555 / 1.0833 | 0.6681 / 1.1164 |
| SMALL_256_FP32 | 0.6212 / 1.0067 | 0.6428 / 1.0385 |
| SMALL_256_FP64 | 0.6314 / 1.0360 | 0.6443 / 1.0620 |
| SMALL_SCALAR_FP32 | 0.7349 / 1.2412 | 0.7490 / 1.2642 |
| SMALL_SCALAR_FP64 | 0.7266 / 1.1933 | 0.7428 / 1.2264 |

### Mini-batch, 5 epochs, batch 64

Best tested FP64 JVM total median: **CPU_LEGACY**.

| Engine | Training median / p95 (ms) | Total median / p95 (ms) |
|---|---:|---:|
| CPU | 0.6411 / 0.9718 | 0.6509 / 0.9843 |
| CPU_LEGACY | 0.5323 / 0.8793 | 0.5388 / 0.8898 |
| NATIVE_FP64 | 1.0590 / 1.6964 | 2.4325 / 3.6031 |
| SMALL_128_FP32 | 0.6228 / 0.6528 | 0.6345 / 0.6664 |
| SMALL_128_FP64 | 0.6380 / 0.6795 | 0.6501 / 0.6905 |
| SMALL_256_FP32 | 0.6095 / 0.9134 | 0.6274 / 0.9326 |
| SMALL_256_FP64 | 0.6137 / 0.6313 | 0.6250 / 0.6454 |
| SMALL_SCALAR_FP32 | 0.7188 / 0.7327 | 0.7316 / 0.7561 |
| SMALL_SCALAR_FP64 | 0.7116 / 0.8395 | 0.7277 / 0.8642 |

### Chunk, 64 epochs, batch 16

Best tested FP64 JVM total median: **SMALL_256_FP64**.

| Engine | Training median / p95 (ms) | Total median / p95 (ms) |
|---|---:|---:|
| CPU | 10.7389 / 11.3060 | 10.7826 / 11.3411 |
| NATIVE_FP64 | 9.6418 / 12.8973 | 12.1895 / 15.3939 |
| SMALL_128_FP32 | 4.6764 / 5.2879 | 4.7351 / 5.3482 |
| SMALL_128_FP64 | 4.9851 / 7.1480 | 5.0494 / 7.3081 |
| SMALL_256_FP32 | 4.6839 / 6.1267 | 4.7477 / 6.2573 |
| SMALL_256_FP64 | 4.6143 / 5.0906 | 4.6722 / 5.1317 |
| SMALL_SCALAR_FP32 | 5.9554 / 9.9770 | 6.0316 / 10.0527 |
| SMALL_SCALAR_FP64 | 5.8980 / 6.3148 | 5.9675 / 6.4014 |

### Chunk, 64 epochs, batch 64

Best tested FP64 JVM total median: **SMALL_256_FP64**.

| Engine | Training median / p95 (ms) | Total median / p95 (ms) |
|---|---:|---:|
| CPU | 10.5844 / 11.1282 | 10.6218 / 11.1668 |
| NATIVE_FP64 | 9.1419 / 10.5563 | 11.2233 / 13.1144 |
| SMALL_128_FP32 | 4.5518 / 5.0419 | 4.5671 / 5.1058 |
| SMALL_128_FP64 | 4.8342 / 5.0245 | 4.8495 / 5.0409 |
| SMALL_256_FP32 | 4.3247 / 4.8591 | 4.3537 / 4.8848 |
| SMALL_256_FP64 | 4.4565 / 4.6406 | 4.4770 / 4.6541 |
| SMALL_SCALAR_FP32 | 5.7522 / 6.6186 | 5.7732 / 6.7323 |
| SMALL_SCALAR_FP64 | 5.7179 / 5.8932 | 5.7410 / 5.9140 |

CPU_LEGACY wins both short mini-batch scopes. SMALL_256_FP64 wins the tested FP64 chunk scopes; CPU_LEGACY was not included in chunk. At batch 64, FP32_256 has a 2.8% lower chunk total median than FP64_256, but its p95 is 4.8848 ms versus 4.6541 ms (5.0% worse). At batch 16, FP32_256 has both a worse median and p95. There is no blanket FP32 win.

## Cohorts

All rows use 64 epochs/model, batch 64, FP64, requested parallelism 4; 10 warmups and 20 measurements per each of three JVM forks. Timings cover the entire cohort. Actual workers differ by strategy, as shown.

### 1 model(s)

Best tested FP64 JVM total median: **SMALL_SEQUENTIAL**.

| Engine | Actual workers | Training median / p95 (ms) | Total median / p95 (ms) |
|---|---:|---:|---:|
| NATIVE | 1 | 9.3743 / 11.0538 | 12.0311 / 14.6396 |
| REFERENCE_MATRIX | 1 | 7.2392 / 8.1733 | 7.2411 / 8.1758 |
| REFERENCE_PARALLEL | 1 | 7.8255 / 8.9875 | 7.8328 / 8.9900 |
| SMALL_CPU | 1 | 23.2252 / 31.5750 | 23.5502 / 31.9617 |
| SMALL_SEQUENTIAL | 1 | 4.3830 / 5.4941 | 4.5551 / 5.6710 |

### 4 model(s)

Best tested FP64 JVM total median: **REFERENCE_PARALLEL**.

| Engine | Actual workers | Training median / p95 (ms) | Total median / p95 (ms) |
|---|---:|---:|---:|
| NATIVE | 1 | 11.7536 / 15.0247 | 14.4329 / 18.6495 |
| REFERENCE_MATRIX | 1 | 28.0744 / 32.8648 | 28.0765 / 32.8677 |
| REFERENCE_PARALLEL | 4 | 12.2131 / 20.0327 | 12.2157 / 20.0348 |
| SMALL_CPU | 1 | 26.7399 / 31.6314 | 27.0901 / 32.3673 |
| SMALL_SEQUENTIAL | 1 | 16.9872 / 18.7711 | 17.3697 / 19.2502 |

### 32 model(s)

Best tested FP64 JVM total median: **REFERENCE_PARALLEL**.

| Engine | Actual workers | Training median / p95 (ms) | Total median / p95 (ms) |
|---|---:|---:|---:|
| NATIVE | 1 | 92.6739 / 104.3558 | 95.6024 / 107.7225 |
| REFERENCE_MATRIX | 1 | 223.0065 / 232.2614 | 223.0108 / 232.2657 |
| REFERENCE_PARALLEL | 4 | 88.0559 / 100.2964 | 88.0586 / 100.2996 |
| SMALL_CPU | 4 | 121.1257 / 138.3588 | 121.8377 / 138.8808 |
| SMALL_SEQUENTIAL | 1 | 136.1782 / 145.9141 | 136.7715 / 146.3158 |

SMALL_CPU cohort SIMD regresses against the best tested strategy at every cohort size: use the measured SMALL_SEQUENTIAL baseline for one model and REFERENCE_PARALLEL for 4/32 models. Native’s 4-model training median is slightly below the parallel JVM baseline, but including its lifecycle makes total median 18.2% slower. At 32 models, compare native’s 95.60 ms with the parallel baseline’s 88.06 ms, not the slower 223.01 ms sequential matrix result.

## Native retention verdict

**FAIL in all seven required scopes.** Policy requires native total median ≤0.8× and total p95 ≤1.0× the fastest tested FP64 JVM strategy, with at least three verified JVM processes. Those data requirements are satisfied. A ratio greater than 1 means native is slower.

| Scope | Best tested FP64 JVM | Native/baseline total median | Native/baseline total p95 |
|---|---|---:|---:|
| cohort 1 models | SMALL_SEQUENTIAL | 2.641 | 2.581 |
| cohort 32 models | REFERENCE_PARALLEL | 1.086 | 1.074 |
| cohort 4 models | REFERENCE_PARALLEL | 1.182 | 0.931 |
| mini 5 epochs, batch 16 | CPU_LEGACY | 4.966 | 4.544 |
| mini 5 epochs, batch 64 | CPU_LEGACY | 4.515 | 4.049 |
| chunk 64 epochs, batch 16 | SMALL_256_FP64 | 2.609 | 3.000 |
| chunk 64 epochs, batch 64 | SMALL_256_FP64 | 2.507 | 2.818 |

## Validation and limits

Independently recomputed median/p95 for all **3480 measured application rounds**, verified all seven FP64 baseline selections and the three-process provenance, and checked finite full-state numerical error ≤1 scaled tolerance in every CLI/cohort record. FP64 maximum scaled error: 6.82061e-08; FP32: 0.00181801. This validates short-run numerical agreement, not convergence quality. The artifact contains no 32-seed quality study and no GPU performance results.

- Hosted virtual machine exposes 2 cores/4 logical processors; EPYC 7763 is host branding, not 64 allocated cores. No sustained frequency, thermal, scheduler or hypervisor isolation evidence.
- Host snapshots show 4% and 5% guest CPU load before/after; listed process CPU values are cumulative seconds, not overlapping utilization during measurement.
- JMH p95 is nearest rank over only 15 iteration averages, therefore their maximum, not a percentile of individual epoch latencies. JMH errors are 99.9% confidence half-widths, not standard deviations.
- JMH scores exclude invocation setup and teardown; every invocation uses fresh model/session. Setup can still create GC pressure. The ten-epoch invocation is normalized with OperationsPerInvocation(10). Public training includes RMSE, and scalar/vector reference configurations can change that inference cost.
- CLI mini/chunk and cohort totals include session open/train/close while preparation and numerical validation are excluded. CLI and JMH differ in hyperparameters, measurement scope and lifecycle; their numbers are not directly comparable.
- CLI backend order rotates by repetition, but the JMH engine/fork execution remains ordered. Fixed dataset/model seeds test repeatability, not distributional robustness.
- Chunk omits CPU_LEGACY; its baseline is best among the tested FP64 JVM engines. Native gates exclude FP32 engines and choose the lowest total median separately for each scope.
- Cohort native uses one CPU worker; REFERENCE_PARALLEL uses four for 4/32 models, SMALL_CPU uses one worker for 1/4 models and four for 32. These compare deployed strategies, not equal-thread kernel efficiency.
- All CI scopes use topology 2,8,8,8,1, 128 samples, EXACT sigmoid. No GPU, FAST sigmoid, other shapes, or 32-seed convergence/quality results are included.

Cleanup: no artifacts removed; compact reports retain unique CI provenance and regressions. At handoff, the evidence directory occupied 23 MiB, the CI benchmark inputs 3.8 MiB. Linux had 808 GiB free (16% used); Windows C: had approximately 92 GiB free (91% used), so no large new output was generated.
