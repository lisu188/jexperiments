# Search implementation validation

The retained JNeuro CI reports cover source `cc5804507d2cd6b0360c90713a7e8ed783991fa4` in [Actions run 36815073298](https://github.com/lisu188/jexperiments/actions/runs/36815073298). The downloaded archives passed ZIP CRC verification and were inspected directly without extraction or a new runtime run. Source/run association comes from the Actions download metadata; the reports themselves do not embed a Git revision.

| Check | Verified result |
|---|---|
| Unit tests | 304 passed, zero failures or skips; all 48 class summaries sum to the aggregate |
| Unit line coverage | 7,986 covered / 8,240 total lines: 96.9175% |
| Native-control GUI tests | 26 passed, zero failures, errors or skips |
| Documented GUI paths | 131/131 covered: 100%; every inventory mapping matches a successful method from this run |
| Separate GUI line coverage | 1,312 covered / 1,389 total lines: 94.4564% |

All ten training-logging tests pass, including the added regressions for sampled INFO progress with per-chunk DEBUG details, committed progress on cancellation/failure, and public bulk-call start/completion logs. All three progress-snapshot regressions pass, covering a shrinking collection, detached submission ordering, and concurrent updates/completion. The GUI suite timestamp is `2026-10-01T04:27:41.364Z`, on host `runnervmtr4k5`.

The optimized Spiral GUI fixture uses two seeds, seven epochs, checkpoints `0,3,6,7`, and target RMSE **0.9**. It verifies execution selection, complete trial budgets, replay of the 176-point training partition with 44 held out, reset, and cancellation. It does not establish the separate **0.01 in four of five seeds** quality target. CUDA routing fixtures in CPU-only CI are also distinct from real GPU acceptance.

## Earlier local and hardware evidence

At source `7a0a1ab61c848500316367cb12720662b9f96ff0`, Windows validation recorded 301 unit tests: **300 passed, one skipped**, with zero failures/errors. Real `gpuCheck` acceptance recorded **18 passed**, zero failed/skipped. Its unit line coverage was 7,982/8,237, or 96.9042%. Those results retain their original revision; they are not relabelled as hardware validation of `cc58045`.

The later production change between these revisions is confined to training logging. Current CI validates those logging regressions. The [post-fix smoke](SMOKE.md) separately records numerical continuation at `7a0a1ab`, including its retained divergent warmup and route-metadata caveat. Neither those smoke runs nor the short GUI fixture qualify throughput distributions or million-epoch Spiral reliability; the measured timing matrix remains scoped to `529e7a9` in [LOCAL.md](LOCAL.md).

## Artifact integrity

The run publishes `jneuro-kotlin-validation` and `jneuro-gui-validation`, with 14-day retention. Downloaded archive SHA-256 values:

```text
validation-cc58045.zip
6631353833d093aca5dbac42b8d39f0d81c42e0c06b3935d2d79e789def7698b

gui-cc58045.zip
bd4a4eb1def4fa85990efc1c6c6e26b59b21d5b027d7f5fda461b2ae4f257493
```

The [compact CI audit](validation-ci.json) retains the archive byte sizes, all 48 unit-class counts and member hashes, aggregate HTML and JaCoCo hashes, current GUI XML/path-coverage hashes, source/run metadata, and the [earlier local validation metadata](validation-local.json) hash. No bulk report copies were extracted during this audit.

## Final local check after quality jobs closed

At checkout `d89d0e552ccb9ff4881c7d245731f47e6155341a`, `:jneuro:check :jneuro:jacocoTestReport` completed successfully after both quality JVMs exited. All 303 executable Windows unit tests passed, with one optional native-BLAS skip. Line coverage was 7,987/8,240 (96.9296%). The [post-quality audit](validation-post-quality.json) retains the exact command, source, timings, JUnit hashes and coverage counters. This validates the final logging changes locally without rebuilding files during a measurement. Hardware CUDA acceptance remains separately scoped to the unchanged training kernels at `7a0a1ab`.
