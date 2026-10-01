# TensorFlow population search measurements

This protocol compares complete searches with the strongest preceding implementation: the retained
TensorFlow epoch-loop sessions from PR #64, running five independent workers. The new path runs models
along a TensorFlow tensor dimension. Both paths use FP64, exact sigmoid, sample batch size 1, learning
rate 0.6, momentum 0.2 and beta 1.0. The immutable Spiral dataset has 176 training and 44 validation
samples, with dataset/split seed 42 and model seeds 1, 42, 123, 999 and 2026.

## Measured results

Measurements used TensorFlow Java 1.2.0 / TensorFlow 2.21.0 and OpenJDK 27+35 on Linux/WSL, with default
oneDNN settings and one TensorFlow intra-op thread per batch. The baseline is
`11ed5d8` (PR #64); candidate source and class hashes are in [provenance.json](provenance.json).
The candidate was measured before commit on base `3f6ff25`; hashes identify its numerical/scheduler
sources independently of subsequent documentation or UI edits. [results.json](results.json) contains
exact values, CPU model, raw-file hashes and all observed ranges.

A later replay correction publishes TensorFlow's FP32-rounded initial parameters when the selected
checkpoint is at epoch zero. It changes host model/replay code after these measurements; the benchmark
does not invoke replay and its numerical graph and search scheduler remain unchanged. The recorded
source hashes describe the measured version and have intentionally been retained.

Elapsed times include the complete search. Each cell shows median milliseconds and observed range;
four observations are too few for a confidence interval.

| Equal-work full-budget case | Legacy independent sessions | Tensor-batched search | Median elapsed ratio |
|---|---:|---:|---:|
| Five seeds of 2→6→1 | 2,278.53 (1,930.47–2,307.80) | 710.15 (449.90–1,246.95) | 3.21× |
| Six architectures × five seeds | 13,366.27 (8,136.29–14,458.59) | 2,605.59 (2,278.70–3,962.83) | 5.13× |

For the mixed population, first eligible-result latency was 2,702.16ms median for the legacy route
and 2,605.44ms for batched search. Its ranges overlap: 2,285.26–3,638.44ms versus
2,278.50–3,962.60ms. One candidate observation had a slower first result despite faster total search.
This is not a universal latency improvement. The six architectures formed two same-depth numerical
batches, with at most 20 active models in one call; the configured 30-model limit is not a claim that
all 30 shared one operation.

Mixed-architecture admission became visible at a median 0.35 ms, compared with 2,406.60 ms in the legacy
route. This is the first published multi-architecture admission, **before native graph opening**, rather
than time to train multiple architectures.

All **140 full-budget best checkpoints**, containing **4,220 parameters**, retained exactly matching
completed/best epoch counts. Maximum parameter difference was 7.11e-15 and maximum best/final score
difference was 1.11e-16. The 32 focused integration tests also passed, including numerical full-state
and momentum parity, streaming scheduling and native Studio replay. These measurements do not replace
the module's full coverage and GUI checks.

## Pruning and validation quality

These runs use the new executor for both policies and have different training work. Both policies
admitted 12 architectures / 60 seed trials for each search seed.

| Policy | Search seed | Committed model-epochs | Elapsed seconds | First eligible seconds | Best median validation RMSE |
|---|---:|---:|---:|---:|---:|
| Full budget |42|9,000|15.056|8.971|0.463122755|
| Full budget |123|9,000|17.349|9.991|0.463122755|
| Successive halving |42|3,875|8.544|7.220|0.463514857|
| Successive halving |123|3,875|4.060|3.498|0.463514857|

Pruning completed three architectures and pruned 45 of 60 trials in each run. It used less training and
finished sooner here, while its best validation result was slightly worse. **Neither policy reached
the 0.05 target.** These two search seeds and short caps do not establish equal search quality or faster
time to a target. Raw observations, including per-trial parameters, remain in the six JSONL files.

## Workload and method

The [harness](PopulationBenchmark.kt) has two full-budget cases:

- Five seeds of 2→6→1, each trained for 50 epochs.
- Six architectures, each with five seeds and 50 epochs: 2→2→1, 2→4→1, 2→6→1,
  2→8→1, 2→4→4→1 and 2→8→4→1.

The second case uses a frozen manifest. Every model receives the same sample sequence, epoch budget
and scoring boundaries in both implementations. Timings include initialization, native graph/session
opening, training, scoring, cohort regrouping, final snapshot exports and cleanup. They exclude JVM
startup and a preceding three-epoch warmup. Each process measures each case twice. Two fresh JVMs per
implementation run in legacy/candidate/candidate/legacy order, giving four observations per case.

First-result latency means the first progress publication with an eligible whole seed group. Separate
first-diversity latency records the first published admission/progress containing at least two different
architectures among active trials. Initial publication precedes native graph opening, so this is an
admission/UI-response measurement, not time to train two architectures. Neither metric is a time-to-target
measurement. First eligible results can take longer in individual runs when a logical round waits for all
depth buckets, even if total elapsed time improves.

The separate quality experiment compares full budget with successive halving on the **new** executor,
using a 150-epoch cap, 60 admitted trials, 30 model slots and search seeds 42 and 123. Its policies perform
unequal amounts of training and may choose different later architectures. Its elapsed time, actual
committed epochs, pruning counts and validation results must be interpreted together; this comparison
does not measure an arithmetic speedup or establish search-quality equivalence.

## Reproduction

[reproduce.py](reproduce.py) compiles the same harness against the preserved baseline, then uses that
binary with each separate class tree. Constructor extensions and new metrics are accessed reflectively
so the original PR #64 classes remain unchanged. The runner performs no downloads and verifies class-tree
hashes before and after measurement. Supply Linux JDK 27, compatible Kotlin 2.4.20 compiler/runtime jars,
an existing TensorFlow Java1.2 native cache, and complete baseline/candidate class trees.

```sh
python3 reproduce.py \
  --java /path/to/jdk-27/bin/java \
  --compiler-classpath "$KOTLIN_COMPILER_CP" \
  --runtime-classpath "$JNEURO_RUNTIME_CP" \
  --baseline-classes /path/to/preserved-pr64-classes \
  --candidate-classes /path/to/batched-search-classes \
  --native-cache /path/to/existing-javacpp-cache \
  --work-dir /run/shm/jneuro-population-measurement
```

The work directory must be new. Outputs are bounded JSONL observations, compact logs, harness classes
and provenance hashes. `--skip-quality` runs only the equal-work comparison. `--analyze-only` recomputes
the report from existing observations. Every full-budget best checkpoint, best/final score, best epoch
and completed epoch count is compared. FP64 tolerances are 1e-9 for parameters and 1e-10 for scores.
Momentum and complete optimizer-state parity are covered separately by numerical tests.

Local measurements need a quiet window: do not overlap them with compilation, tests or another
benchmark. A shared workstation, four observations and short budgets do not establish confidence
intervals, GPU performance or quality at the Studio's much larger epoch cap. Native model batching
does not imply one GPU kernel launch; TensorFlow schedules the graph's constituent operations.
