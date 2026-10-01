# Architecture-search measurement protocol

This record measures the execution changes in JNeuro's architecture search on Spiral. The completed local fixed-work matrix measures compute revision `529e7a9`. The [local results](LOCAL.md) and [CI results](CI.md) retain their own source, hardware and qualification scope; the separate long-budget quality experiment does not share the fixed-work timing claims.

## Fixed-work comparison

The fixed manifest contains eight hidden-layer configurations: `[8]`, `[8,8]`, `[8,8,8]`, `[8,8,8,8]`, `[4,8]`, `[16,4]`, `[4,8,16]`, and `[16,8,4,8]`. Every model has two inputs and one output. Every configuration evaluates seeds `1`, `42`, `123`, `999`, and `2026` to the full 2,000-epoch budget. There is no pruning or convergence-based early stop.

The 220-point Spiral dataset is split reproducibly into 176 training and 44 validation samples using split seed 42. Training uses FP64, EXACT sigmoid, and online updates (`batchSize=1`). Scores are checked at epoch zero and every 25 epochs, including the final boundary. A complete timed search therefore contains 40 completed seed trials, 80,000 committed epochs and 14,080,000 sample updates.

The local schedule runs these groups sequentially, with no overlapping benchmark JVMs:

| Group | Training engine/backend | Worker counts | Comparison |
|---|---|---|---|
| CPU SMALL | SMALL / CPU | 1, 4, 8, 16, 32 | Reference execution versus optimized execution |
| CPU reference engine | REFERENCE / CPU | 1, 4, 8, 16, 32 | Reference execution versus optimized execution |
| GPU SMALL | SMALL / CUDA | 4 CPU scoring workers | Reference execution versus the optimized CUDA queue |

Each group runs in three fresh JVMs. Each JVM executes one warmup round and three measured rounds per case. Case order rotates between measured rounds and between JVMs. The CPU reference engine remains an independent baseline: a gain over the SMALL reference scheduler alone does not establish a gain over the fastest applicable CPU baseline.

The CI benchmark now runs all three forks sequentially in one job on the same host, retaining them in the `jneuro-search-cpu` artifact. The earlier three-job matrix ran on three different CPU models (AMD EPYC 7763, AMD EPYC 9V74, and Intel Xeon Platinum 6973P-C); those historical reports remain separate host samples and cannot be pooled to qualify a three-fork comparison.

The fixed manifest establishes equal work for timing comparisons. It does not establish that the manifest contains the globally smallest reliable Spiral network. The separate quality protocol uses the adaptive planner, preserves complete candidate seed budgets, and requires validation RMSE at most `0.01` in at least four of five seeds. Parallel completion order can change the adaptive planner's subsequent candidate choices; compare fixed manifests for equivalent-work speed claims.

## Timing and resource boundaries

`totalNanos` measures one complete search call, including model construction, scheduling, session setup, training, scoring, best-snapshot publication, draining outstanding work and resource cleanup. JVM startup and JSON serialization after the search are outside this interval. The dataset fixture is created before the timed search. Warmup durations are retained but excluded from measured distributions.

Allocation counts cover the JVM's total allocated bytes across all threads during the search call, when the JVM exposes that counter. They are not retained-heap size or CPU-worker-only allocations. Unsupported counters remain absent rather than being inferred.

Per-trial phase times are diagnostic and can overlap across models. CPU advancement may compute a boundary RMSE inside its training call, while queued CUDA performs that scoring on a CPU executor after advancement. Consequently, summed phase durations and cross-backend training-versus-scoring breakdowns cannot replace the end-to-end comparison.

The reference SMALL scheduler may execute a seed through a single-model session or a cohort as admission slots become available. The optimized CPU path uses sessions; optimized CUDA uses the persistent queue. Reports retain actual route, kernel, device identity, precision, sigmoid and SIMD width. A route change is accepted only within the implementation's known route families and with unchanged workload and numerical checks.

## Qualification rules

The [summarizer](../../tools/summarize_search_benchmark.py) requires complete paired evidence before qualifying a comparison:

- At least three distinct JVM process identities and at least nine measured rounds per execution mode. Both modes must complete every configured warmup and measured round in each admitted process.
- Identical source revision, runtime, JVM arguments, CPU/OS metadata, heap limit, dataset fingerprint, manifest, seed list and other protocol settings. Host or protocol mixtures remain separate comparisons.
- Every expected candidate and seed must be present exactly once, with its full epoch and sample-update counts. Partial, failed, cancelled, budget-truncated and unpaired forks are excluded. Repeated paths, copied reports and duplicated records cannot increase the sample count.
- Every measured reference and optimized row must match the recorded best parameters, best/final RMSE and best epoch. The FP64 tolerance is `abs(actual-reference) <= 1e-10 + 1e-8*abs(reference)`. Non-finite values, changed device configuration and unsupported execution routes cannot qualify. Optimizer-state and shuffle continuation equivalence are covered separately by automated tests.
- The optimized median must be at most 90% of the reference median, with no p95 regression. A 1.10× speedup is only a 9.09% time reduction and does not meet this 10% time-reduction threshold.

Median is the conventional sample median; p95 is the nearest-rank percentile of complete search-call durations. With nine measured rounds, p95 equals the maximum. These are whole-search latencies, not percentiles of individual training epochs. Qualification applies to the recorded host and protocol; it is not a universal CPU/GPU crossover claim.

The quality run's independent Spiral test uses 218 interleaved points, after winner selection. Its reported score describes the representative winning seed. It does not establish five-seed reliability on the independent test set.

For a separate CPU quality run, manually dispatch the `CI` workflow with `search_quality=true`. The reusable `JNeuro Spiral search quality` job runs SMALL/CPU/OPTIMIZED on Ubuntu with Java 27 and four workers, one million epochs per seed, a 60-trial budget, and search seeds `42,123`. Its 3,500-second wall limit covers the entire invocation; interrupted trials or an unstarted second search remain incomplete evidence. Source revision, host metadata and flushed JSONL results are retained in the `jneuro-search-quality-cpu` artifact for 14 days. This CI host can run independently of local CPU/GPU measurements, but its quality results are not a paired throughput comparison with different local hardware. Normal PR and push checks remain enabled; only explicit experiment dispatches skip the standard build jobs.

## Reproduce

Use Java 27 and the existing Gradle wrapper. This CPU example executes one fresh JVM containing both modes; repeat it with distinct output paths and order offsets `0`, `1`, and `2`:

```text
./gradlew :jneuro:architectureSearchBenchmark -PneuroBenchmarkRevision=529e7a9 --args="--mode fixed --engine SMALL --epochs 2000 --workers 1,4,8,16,32 --warmups 1 --repeats 3 --order-offset 0 --output search-cpu-small-0.jsonl"
```

Use `--engine REFERENCE` for the general CPU engine baseline. Use `--backends CUDA --workers 4` with SMALL for the GPU group; this requires actual CUDA hardware and fails if unavailable. Run groups sequentially and retain process exit records alongside the JSONL reports. The bounded `--manifest general` option exercises arbitrary widths through the CPU reference engine.

Summarize only the intended compatible reports:

```text
python3 jneuro/tools/summarize_search_benchmark.py search-cpu-small-0.jsonl search-cpu-small-1.jsonl search-cpu-small-2.jsonl --output search-cpu-small-summary.json
```

The full local command schedule, incremental raw reports and process metadata are retained under `/tmp/jneuro-search-evidence` during validation. Compact final evidence and its source hashes should be published here after validation. Large runtime files and temporary build products are not required to read this protocol.

## Compact final evidence format

The final report should retain every measured duration without repeating the full parameter arrays in every row. A suitable JSON layout is:

```json
{
  "schemaVersion": 1,
  "status": "validated-or-incomplete",
  "computeRevision": "529e7a9",
  "analysis": {"sourceRevision": "analysis-revision", "scriptSha256": "sha256"},
  "protocols": {"protocol-id": {"datasetFingerprint": "sha256", "manifest": [], "seeds": [], "runtime": {}, "hardware": {}}},
  "runs": [{"id": "fork-id", "protocol": "protocol-id", "sourceFile": "report.jsonl", "sha256": "sha256", "started": "timestamp", "pid": 1, "exitCode": 0}],
  "rounds": [{"run": "fork-id", "case": "engine/backend/execution/workers/search-seed", "round": 0, "totalNanos": 1, "allocatedBytes": 1, "completeTrials": 40, "committedEpochs": 80000, "sampleUpdates": 14080000}],
  "comparisons": [],
  "quality": {"status": "pending", "uniqueCandidateSeedScores": [], "independentTest": null},
  "exclusions": [],
  "limitations": []
}
```

The example is a schema sketch, not measurement data. `protocols` should contain the complete environment records with dynamic fork identity removed. `rounds` retains warmup rows using negative round indices, but summary distributions use only measured rows. It may include summed allocation counts and diagnostic phase totals if their scopes remain explicit.

Each comparison should retain both median/p95 distributions, fork and round counts, speedup, median time reduction, work checks, numerical tolerances, maximum scaled error, best-epoch agreement, actual execution-route counts and the qualification verdict. Failed or incomplete observations belong in `exclusions`, with their original reason and process identity. Raw input hashes bind these summaries to the retained JSONL evidence; the raw reports remain necessary to independently recheck parameter comparisons.

Record quality scores once per unique architecture/seed and arithmetic engine when repeated runs agree. Repeated timing rounds are not independent quality seeds. Keep within-engine parity separate from cross-engine arithmetic comparisons: the general CPU engine's AUTO vector path uses horizontal dot reductions and some separate multiply/add updates, while SMALL preserves ordered FMA accumulation. Equal FP64 precision and EXACT sigmoid therefore do not guarantee identical long-horizon optimization trajectories.

After all nine JVM jobs finish, this Bash command validates and summarizes the three groups independently in one output:

```bash
python3 jneuro/tools/summarize_search_benchmark.py \
  /tmp/jneuro-search-evidence/cpu-small-{0,1,2}.jsonl \
  /tmp/jneuro-search-evidence/cpu-reference-{0,1,2}.jsonl \
  /tmp/jneuro-search-evidence/gpu-{0,1,2}.jsonl \
  --output /tmp/jneuro-search-evidence/performance-summary.json
```

The summarizer deliberately groups by engine, backend, worker count and full protocol. It does not promote a cross-engine ranking into an identical-arithmetic qualification or choose the strongest baseline automatically. That comparison needs an explicitly labeled final analysis of the separately validated groups and their quality differences.

## Completed CI evidence

The completed CI matrix used source `974b22ca8e93a726038364eac163f7f184521abc`, separately from the local compute revision above. Its three jobs ran on different CPUs: AMD EPYC 7763, AMD EPYC 9V74 and Intel Xeon 6973P-C, each exposed as a four-logical-processor hosted VM. The [CI notes](CI.md) retain separate timing tables and interpretation. The [compact evidence](ci-results.json) includes all 360 search-call durations, 12,000 trial records, deduplicated full parameter snapshots, source/input hashes and exact reconstruction checks; the [qualification summary](ci-qualification-summary.json) retains all 45 comparison verdicts.

All measured within-engine reference-versus-optimized parameter snapshots, best/final scores and best epochs match exactly, and every expected seed completes its full budget. Timing qualification remains **inconclusive**: each hardware/protocol has one actual JVM fork and three measured rounds. Pooling the three different CPU hosts would falsely satisfy the required fork count. No CI comparison is marked qualified.

## Verify or regenerate compact evidence

The retained [compact_evidence.py](compact_evidence.py) is a standalone artifact script with no additional dependencies. It uses the repository summarizer by default; `--tool PATH` overrides that location. Verify the published CI artifact and reconstruct one original parsed round from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import json, sys
from pathlib import Path
folder = Path("jneuro/benchmarks/2026-10-01-search")
sys.path.insert(0, str(folder))
import compact_evidence
report = json.loads((folder / "ci-results.json").read_text())
print(compact_evidence.verify(report))
first = compact_evidence.reconstruct_round(report, report["rounds"][0])
print(first["case"], first["round"], first["completeTrials"])
PY
```

Verification checks reconstructed round contents against their retained canonical SHA-256 hashes. The raw-file byte hashes and process/host metadata remain separate provenance checks. The artifact records the SHA-256 of both the compactor and summarizer used to produce it.

After all nine local jobs finish, build the final compact local record using their retained successful process sidecars:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 jneuro/benchmarks/2026-10-01-search/compact_evidence.py \
  /tmp/jneuro-search-evidence/cpu-small-{0,1,2}.jsonl \
  /tmp/jneuro-search-evidence/cpu-reference-{0,1,2}.jsonl \
  /tmp/jneuro-search-evidence/gpu-{0,1,2}.jsonl \
  --raw-root /tmp/jneuro-search-evidence \
  --output /tmp/jneuro-search-evidence/local-results.json \
  --summary-output /tmp/jneuro-search-evidence/local-qualification-summary.json \
  --status validated-fixed-work --require-process-exit
```

`validated-fixed-work` describes report completeness, not a blanket speedup qualification. Individual comparison verdicts still enforce compatible forks, numerical parity, median gain and p95. Use `preliminary` for an incomplete measurement series, and `inconclusive-mixed-hardware` for the CI matrix retained here. The generator verifies every reconstructed round before and after serializing the compact file; it never deletes the raw inputs.

## Completed local throughput

Nine sequential JVMs completed the fixed-work matrix with successful exits. All eleven same-engine/backend/worker groups have three compatible forks, nine measured calls per mode, complete 40-trial budgets and exact recorded best-parameter/score/epoch parity. Ten groups meet the 10% median-reduction/no-p95-regression gate. The exception is one-worker SMALL, whose median regresses by 8.1%.

The strongest observed prior CPU baseline and fastest optimized CPU configuration are both REFERENCE/CPU with 32 workers: **6.053 → 4.024 seconds** per complete search, **1.504x** faster. The CUDA queue improves its GPU baseline **53.335 → 12.462 seconds**, **4.280x**, while remaining **2.56x slower** than optimized CPU at the same four-worker setting. These figures describe the fixed 2,000-epoch manifest, not time to reach the quality target.

[LOCAL.md](LOCAL.md) contains every configuration's median/p95, the one-worker regression, third-fork variability, allocation tradeoffs and cross-arithmetic caveats. [Compact local evidence](local-results.json) reconstructs all original parsed round/trial data; the [qualification summary](local-qualification-summary.json) retains all verdicts. The later `7a0a1ab` Java/API/progress refinements and final-head smoke checks remain separate from the measured `529e7a9` matrix.
