# Local architecture-search results

All nine fixed-work JVMs completed successfully on Windows 11 with an Intel Core i5-14400F (16 logical processors), NVIDIA GeForce RTX 4060 Ti, and OpenJDK 27+35-2325. The measured compute revision is **`529e7a9`**. Every engine/backend case has three compatible JVM forks and nine measured calls per execution mode, plus one warmup call per fork and mode. Logs were disabled; JVM heap settings were `-Xms128m -Xmx2g`.

The fixed workload is the [eight-shape, five-seed protocol](README.md#fixed-work-comparison): 2,000 epochs per seed, FP64, EXACT sigmoid, batch size one, 176 training samples and 44 validation samples. Every timed call completes 40 trials, 80,000 epochs and 14,080,000 sample updates. All nine process exit codes are zero; no incomplete or failed measurements were excluded.

## Whole-search latency

Times are seconds. Medians and p95 values use nine complete measured search calls per mode; nearest-rank p95 is the maximum with this sample count. A qualified row passes the predeclared same-engine/backend/worker-count gate: complete work, matching recorded numerical results, at least 10% median-time reduction and no p95 regression. All outliers remain included.

| Engine / backend | Workers | Reference median | Optimized median | Speedup | Reference p95 | Optimized p95 | Qualified |
|---|---:|---:|---:|---:|---:|---:|---|
| REFERENCE / CPU | 1 | 22.058 | 13.548 | 1.628x | 24.532 | 16.470 | Yes |
| REFERENCE / CPU | 4 | 8.636 | 4.870 | 1.773x | 9.677 | 6.009 | Yes |
| REFERENCE / CPU | 8 | 6.907 | 4.668 | 1.480x | 8.114 | 5.836 | Yes |
| REFERENCE / CPU | 16 | 6.336 | 4.293 | 1.476x | 7.539 | 4.859 | Yes |
| REFERENCE / CPU | 32 | 6.053 | 4.024 | 1.504x | 6.990 | 4.570 | Yes |
| SMALL / CPU | 1 | 15.382 | 16.625 | 0.925x | 74.239 | 68.586 | No |
| SMALL / CPU | 4 | 9.141 | 5.437 | 1.681x | 34.657 | 6.331 | Yes |
| SMALL / CPU | 8 | 9.479 | 4.630 | 2.047x | 42.724 | 14.229 | Yes |
| SMALL / CPU | 16 | 9.620 | 4.270 | 2.253x | 43.946 | 9.664 | Yes |
| SMALL / CPU | 32 | 11.521 | 4.358 | 2.644x | 36.368 | 7.672 | Yes |
| SMALL / CUDA | 4 | 53.335 | 12.462 | 4.280x | 63.633 | 12.937 | Yes |

The strongest observed prior CPU baseline and fastest optimized CPU configuration are both the general REFERENCE engine at 32 workers: **6.053 → 4.024 seconds**, a **1.504x** speedup and 33.5% reduction. This comparison retains the same engine, worker count and arithmetic. At 8 and 16 workers, the two optimized CPU engines differ by less than 1% in median time; their tail distributions differ substantially, so these tiny median differences do not establish a meaningful engine advantage.

The CUDA queue improves its own GPU baseline from **53.335 → 12.462 seconds**, or **4.280x**. It still takes **2.56x** as long as optimized REFERENCE/CPU at the matching four-worker setting (4.870 seconds), and **3.10x** as long as the fastest measured CPU configuration, which uses 32 workers. CPU/GPU rankings compare equal work; they are not claims of identical long-horizon numerical trajectories.

**One-worker SMALL regresses:** its median rises from 15.382 to 16.625 seconds, an 8.1% increase. It does not qualify. Ten of the eleven same-configuration comparisons meet the stated gate; the results do not support an unconditional claim that every optimized path is faster.

## Numerical audit and quality scope

Every measured reference-versus-optimized best-parameter snapshot, best/final RMSE and best epoch matches exactly within its engine/backend group: maximum scaled error **0** in all eleven comparisons. This audit covers 198 measured search calls and 7,920 measured trial observations. Repeated timing observations are not independent quality seeds. The 66 warmup calls are retained separately; eight distinct warmup-only outcomes are preserved. Automated tests separately check optimizer-state and shuffle continuation.

Changing the arithmetic implementation can change the long trajectory. Across the 40 unique model/seed pairs, CPU SMALL versus CPU REFERENCE has a maximum best-RMSE difference of 0.066295 and two different best epochs. CPU SMALL versus CUDA SMALL has a maximum best-RMSE difference of 0.021351 and one different best epoch. The general CPU AUTO path uses horizontal dot reductions and some separate multiply/add operations; SMALL uses ordered FMA accumulation. Backend exponential implementations also differ. These facts explain why the execution-mode parity audit must be kept separate from cross-engine/backend quality comparisons; they do not isolate the cause of every observed score difference. [Paired cross-arithmetic observations](local-cross-arithmetic.json) retain the actual scores and epochs.

No individual seed or candidate meets validation RMSE 0.01 in these 2,000-epoch benchmark cases. This fixed-work experiment establishes throughput and execution parity; it does not identify the smallest reliable Spiral network. The later quality runs preserve one million epochs per seed, 60 trials and search seeds 42/123, subject to the wall deadline. Local CPU (eight workers) and GPU (four CPU scoring workers) quality jobs share the host when run concurrently, so their elapsed times must not be presented as a paired speed comparison. Incomplete seed trials or candidate groups cannot qualify as reliable winners.

## Allocation and timing variability

Allocation counters measure total JVM allocation during each search call, across all threads; they are not retained heap or native device memory. The general CPU engine decreases from roughly 13.1 MiB to 8.6 MiB per search at the median. SMALL at 32 workers drops from 148.3 MiB to 20.6 MiB. The faster CUDA queue instead increases median allocation from 168.4 MiB to 226.2 MiB, about 34.3%; throughput gains do not imply lower allocation on every route.

The third SMALL JVM contains substantial late slowdowns. A one-worker reference call takes 74.239 seconds and allocates 20.95 MiB, versus prior-fork medians of 15.061 seconds and 14.04 MiB. A late optimized call takes 68.586 seconds and allocates 17.96 MiB, versus 14.946 seconds and 14.13 MiB previously. Most extra duration is recorded in training; setup and close remain small. This modest allocation increase does not show the large allocation growth expected from a hot-path Vector-object allocation explosion. It does not rule out other JIT, GC, scheduling or frequency effects.

The existing Studio remained active during these measurements. A [late point sample](host-load-third-fork.txt) reports 16% CPU load, 3% GPU utilization and 36°C GPU temperature; it does not characterize the earlier slowdown interval or prove background contention, thermal throttling or any other cause. No profiling was introduced into the timed runs. The [offline phase/allocation diagnostic](third-fork-diagnostic.json) preserves the affected rounds and caveats. Passing the minimum timing gate is not evidence of stable tail latency on this interactive host.

## Revisions and retained evidence

Revision `7a0a1ab` subsequently added Java overloads and race-safe progress snapshots throttled to 10 Hz. Those refinements do not change training arithmetic, but they are outside these nine measured JVMs. The final-head checks and bounded 2,000-epoch smoke run are separate validation; one warmup and one measured smoke call do not replace the three-fork performance matrix. The later quality run also records its own revision and should remain a separate result.

[local-results.json](local-results.json) retains all 264 search-call records, all 10,560 trial records, 128 deduplicated full numerical outcomes, effective devices/routes, process exits and arguments, and input/analyzer/compactor hashes. Every parsed original round reconstructs exactly and matches its retained canonical SHA-256. The [qualification summary](local-qualification-summary.json) preserves all eleven verdicts. See the [verification command](README.md#verify-or-regenerate-compact-evidence) and [standalone compactor](compact_evidence.py). After verifying the committed compact artifact against all nine inactive raw reports, the duplicate raw JSONL files were removed from `/tmp/jneuro-search-evidence`: 37,697,394 logical bytes. Process metadata, diagnostic logs, hashes and every reconstructible parsed record remain retained. This frees Linux filesystem space; it does not assert Windows VHDX compaction.
