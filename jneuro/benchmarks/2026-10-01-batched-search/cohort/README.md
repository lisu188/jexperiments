# Tensor cohort configuration sweep

This exploratory CPU sweep checks the actual model dimension in `TensorFlowCohortGraph`: independent weights, momentum and sample orders share batched TensorFlow operations. It does not measure complete architecture-search throughput. The parent report covers that separately.

The final compiled implementation uses TensorFlow Java 1.2.0 / TensorFlow 2.21.0, CPU FP64, EXACT sigmoid and online sample batches of one. Each case contains 176 training and 44 validation Spiral samples per model. One JVM ran two repeats per configuration, with two warm-up epochs followed by five measured epochs and one final vector training/validation score. The measured call includes permutation validation, order flattening, native training and best-checkpoint selection; it excludes initial model creation and cohort construction.

The table reports the median aggregate model-epochs/second of the two observations. `M` is the number of independent models in the tensor. Shallow is `2→6→1`; deep is `2→8→8→8→1`; padded mixes three-hidden-layer architectures with widths 1–8 in one cohort. Each lane has a separate seed and a distinct valid cyclic sample permutation.

| Family | M | 1 TF thread | 2 TF threads | 4 TF threads | Native open, 1 thread (ms) |
|---|---:|---:|---:|---:|---:|
| shallow | 5 | 295.4 | 270.1 | 204.5 | 187.88 |
| shallow | 32 | 1,333.1 | 1,100.4 | 877.2 | 82.87 |
| shallow | 128 | 2,764.1 | 4,470.6 | 3,098.5 | 67.74 |
| shallow | 512 | 6,222.7 | 4,830.2 | 8,542.4 | 54.36 |
| deep | 5 | 205.5 | 335.1 | 336.3 | 51.95 |
| deep | 32 | 741.4 | 480.2 | 446.8 | 81.43 |
| deep | 128 | 821.1 | 1,176.3 | 1,115.4 | 99.15 |
| deep | 512 | 1,340.0 | 1,209.8 | 2,056.3 | 98.77 |
| padded | 5 | 300.7 | 321.5 | 312.5 | 49.19 |
| padded | 32 | 717.5 | 702.9 | 823.0 | 51.82 |
| padded | 128 | 991.0 | 997.1 | 1,403.1 | 74.38 |
| padded | 512 | 2,202.9 | 1,544.0 | 1,740.5 | 73.52 |

Private per-session TensorFlow pools make the thread counts effective without changing the existing inference pool. More threads do not consistently help: small actual buckets can lose throughput to scheduling, while some wider cases benefit. The production default remains one thread. Selecting the best 128- or 512-model row is not evidence of the same speedup for a heterogeneous end-to-end search.

Native graph/session opening ranged from 29.54 to 277.10 ms across these observations. Initial host model creation is separately recorded as `initialMs` and includes first-use TensorFlow initialization in the first family. The graph retains datasets, current optimizer state and best checkpoints between measured calls. Neither parameter export nor graph regrouping occurs in this microbenchmark.

All 72 cases completed without failed lanes; every returned training RMSE was finite. For each family/capacity, the first lane’s validation RMSE was identical across the six thread-count/repeat observations. Complete parameter, bias and momentum comparisons against independent sessions, masked scoring, ragged tails, deep padding and numerical-failure rollback are covered by `TensorFlowSearchCohortTest`; this timing harness does not replace those assertions.

## Evidence and limitations

- [Exact harness](CohortBenchmark.kt), [all 72 observations](results.jsonl), [summary including two-observation ranges](summary.json), [launch arguments](launch-command.json), and [source/class hashes and environment](provenance.json), and [native-cache reproducibility audit](cache-audit.json).
- There is one JVM, two repeats and a fixed case order. JVM/native warmup, caching and workstation variation can affect comparisons. This is configuration exploration, not a statistically powered performance claim.
- The original launcher set the RAM temporary directory but omitted the JavaCPP cache override. It reused `/home/andrz/.javacpp/cache` rather than the isolated RAM cache. The same omission in the first focused cohort test had created this approximately 815 MiB extracted cache earlier in the task. After retained-jar reproducibility and inactivity checks, the two audited extraction directories were removed at 2026-10-01 11:22:42 UTC: 854,293,360 regular-file bytes plus 49 symlink bytes. Observed Linux free space increased by 854,134,784 bytes to 838,390,968,320 bytes; Windows allocation reclaimed was not measured. The parent lock, required RAM native cache and source jars remain available. The retained timings include no library loading; actual cache provenance is retained instead of relabeling the run.
- GPU execution was not measured. A tagged hardware acceptance test must run with a compatible GPU-enabled TensorFlow runtime.

## Reproduction

Use the recorded source revision/hashes, the repository’s Kotlin 2.4.20 compiler, JDK 27, and already resolved TensorFlow CPU runtime dependencies. Build the production module first using the repository workflow. The driver compiles only this harness, creates no dependency downloads, and requires explicit existing production classes, compiler/runtime classpaths, and native cache. Keep scratch outputs on RAM-backed storage while the host volume is under its disk reserve.

```bash
python3 reproduce.py \
  --java /path/to/jdk-27/bin/java \
  --compiler-classpath "$KOTLIN_COMPILER_CLASSPATH" \
  --runtime-classpath "$JNEURO_RUNTIME_CLASSPATH" \
  --main-classes /path/to/compiled/production/classes \
  --native-cache /run/shm/existing-tensorflow-native-cache \
  --scratch /run/shm/jneuro-cohort-reproduction
```

The exact original absolute arguments are in `launch-command.json`; the reproduction driver explicitly supplies the cache override that was missing there. It changes cache placement, not the numerical graph or measured warm-up protocol.
