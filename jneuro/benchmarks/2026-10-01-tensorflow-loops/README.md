# TensorFlow epoch loop measurements

These local Linux/WSL measurements compare the per-mini-batch Java `Session.run()` implementation at
`8d9c0438aa002c1c376eb5beff98b7c1c66ed0a3` with the functional TensorFlow loop in this change.
The candidate source SHA-256, exact settings and complete timing ranges are recorded in
[results.json](results.json). The compiled baseline and candidate classes were kept separate.

## Workload and method

- TensorFlow Java 1.2.0 / TensorFlow 2.21.0, Linux CPU, OpenJDK 27+35; FP64, EXACT sigmoid,
  learning rate 0.6, momentum 0.2, beta 1.0, seed 42. oneDNN retains its default enabled setting.
- The Studio workload contains all 220 Spiral samples, with shapes 2→6→1 and 2→8→8→8→1.
  Batch sizes 1, 16 and 32 are separate cases. No batch-size change is included in a reported speedup.
- Two fresh JVM processes per version, in baseline/candidate/candidate/baseline order. Each process
  runs three fresh models per case, warming each retained session for five epochs and measuring the
  next 20. The tables use the median of six observations, each observation being a 20-epoch mean.
- `epoch` calls the actual `NeuroTrainingSession.trainEpoch()` 20 times, including publication and
  RMSE each epoch, matching the Studio's REFERENCE training call pattern. `chunk` requests 20 epochs
  through `trainChunk`, including its final publication and RMSE. Session construction is timed separately.
- Search advancement uses five simultaneous retained sessions, seeds 1/42/123/999/2026,
  shape 2→6→1 and the actual 176/44 training/validation split with split seed 42. Workers synchronize
  after warmup, then call `advanceTrainingForSearch` for 20 epochs each. Throughput is 100 total epochs
  divided by the slowest worker's measured duration. This includes epoch publication and final training
  RMSE, and excludes initialization, validation scoring, ranking, checkpoint capture and UI work.
- Other agents paused compilation and tests during both measurement windows. This workstation was
  not isolated from unrelated activity. Two forks and short runs do not establish confidence intervals,
  production latency bounds, GPU performance or whole-search elapsed-time improvement.

## Retained training results

Times are milliseconds per epoch; smaller is better.

| Shape | API | Batch | Baseline | TensorFlow loop | Speedup |
|---|---|---:|---:|---:|---:|
| 2→6→1 | epoch | 1 | 30.037 | 5.881 | 5.11× |
| 2→6→1 | epoch | 16 | 2.282 | 1.184 | 1.93× |
| 2→6→1 | epoch | 32 | 1.198 | 0.961 | 1.25× |
| 2→6→1 | chunk | 1 | 30.028 | 4.681 | 6.42× |
| 2→6→1 | chunk | 16 | 1.726 | 0.344 | 5.02× |
| 2→6→1 | chunk | 32 | 0.878 | 0.255 | 3.44× |
| 2→8→8→8→1 | epoch | 1 | 29.149 | 8.945 | 3.26× |
| 2→8→8→8→1 | epoch | 16 | 2.933 | 1.186 | 2.47× |
| 2→8→8→8→1 | epoch | 32 | 1.584 | 0.849 | 1.86× |
| 2→8→8→8→1 | chunk | 1 | 26.250 | 7.254 | 3.62× |
| 2→8→8→8→1 | chunk | 16 | 2.314 | 0.569 | 4.07× |
| 2→8→8→8→1 | chunk | 32 | 1.155 | 0.352 | 3.28× |

Five-model search-training throughput increased from **117.89 to 264.46 aggregate epochs/second**,
or **2.24×**. The first process pair alone suggested 3.20×; the reported value includes the second,
reversed-order pair. The raw observations and ranges are retained so this variation remains visible.

Opening a session became more expensive: the median across all measured openings increased from
13.43 to 21.95 ms. The first opening in each fresh JVM was 35/64 ms for the baseline and 117/121 ms
for the loop. Retained-session results exclude that startup cost; callers that continually recreate
models or sessions need their own end-to-end measurement.

Snapshot copying and frame generation were measured separately. The harness times 1,000 detached
snapshot captures, 20 fresh `Studio.frame()` calls after one completed training epoch, and 1,000 cached
frame calls. Exact observations are in the JSONL files. These rendering paths were unchanged in the
measured candidate; no rendering speedup is claimed.

## Correctness evidence

All **102 final checkpoints** agreed between versions, including **16,044 weights, biases and momentum
values compared bit-for-bit**, RMSE, epoch/sample counters and the complete next shuffle permutation.
Each checkpoint follows the five warmup plus 20 measured epochs. Full transient checkpoint files stay
in the task's RAM evidence directory; [results.json](results.json) records the comparison totals.

The real `TensorFlowMathTest` and `NeuroTest` assertions also passed all 23 focused tests. This local
check invoked each `@Test` on a fresh instance with JUnit 6.1.2 assertions; it was not a Gradle/JUnit
engine run or a coverage report. The repository's separate 90% coverage and GUI checks remain required.
New tests cover ragged epoch boundaries with changing batch modes in FP32/FP64 and EXACT/FAST,
empty-dataset snapshots, and failure after an earlier finite batch.

## Reproduction

The exact harness is [EpochBenchmark.kt](EpochBenchmark.kt). [reproduce.py](reproduce.py) compiles it
against a preserved baseline, starts independent JVMs, retains every raw observation/checkpoint and
checks complete state equivalence. It performs no dependency downloads and never changes the supplied
class trees. Supply compatible Kotlin 2.4.20 compiler dependencies, the application runtime dependencies,
Linux JDK 27, and separate precompiled baseline/candidate classes. A candidate containing only the
changed `TensorFlowMath` class family may overlay the complete baseline.

```sh
python3 jneuro/benchmarks/2026-10-01-tensorflow-loops/reproduce.py \
  --java /path/to/jdk-27/bin/java \
  --compiler-classpath "$KOTLIN_COMPILER_CP" \
  --runtime-classpath "$JNEURO_RUNTIME_CP" \
  --baseline-classes /path/to/preserved-main-classes \
  --candidate-classes /path/to/candidate-classes \
  --native-cache /path/to/existing-compatible-javacpp-cache \
  --work-dir /run/shm/jneuro-epoch-reproduction \
  --forks 2
```

The work directory must be new. Without `--native-cache`, JavaCPP extracts approximately 815 MiB of
CPU native libraries under that directory; reusing a compatible existing cache avoids that additional
copy. All compiler/process temporary files and measurement outputs go under the selected work directory.
The harness is an intentionally bounded local qualification tool, not a replacement for JMH or CI.

Files `baseline.jsonl` and `optimized.jsonl` contain the first process pair;
`baseline-fork2.jsonl` and `optimized-fork2.jsonl` contain the reversed pair. The corresponding
complete-state comparisons are summarized in `results.json`.
