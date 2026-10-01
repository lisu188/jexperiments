# Final Spiral quality evidence

**No architecture qualified as reliable under the completed-work protocol.** The local GPU run produced two promising partial threshold crossings for hidden layers `[8,8,4]` (137 parameters), but neither trial completed its million-epoch budget, and the candidate had only two observed crossings among five seeds. This is not a winner, a reliable architecture, or evidence of a global minimum.

All three reports use source `7a0a1ab61c848500316367cb12720662b9f96ff0`, FP64/EXACT, online batch size 1, and the fixed Spiral split described in [QUALITY_PROTOCOL.md](QUALITY_PROTOCOL.md). Reliability requires all five distinct training seeds to complete 1,000,000 epochs, with at least four saved best-checkpoint validation RMSE values ≤0.01. Each search retains its 60-trial budget; no epochs or seeds were reduced to obtain a result.

## Completion, kept separate by execution environment

The local CPU and GPU runs executed concurrently on the same interactive Windows host. CI CPU ran separately on a four-logical-processor AMD EPYC Linux host. These quality-run elapsed times and completion counts are not controlled throughput comparisons. No trials or seeds are pooled between reports.

| Source | CPU workers | Full million-epoch trials | Cancelled partial trials | Unused trial-budget slots | Complete five-seed candidates | Reliable candidates |
|---|---:|---:|---:|---:|---:|---:|
| Local CPU | 8 | 42 | 8 | 10 | 8 | 0 |
| Local GPU | 4 scoring workers | 5 | 55 | 0 | 1 | 0 |
| CI CPU | 4 | 46 | 4 | 10 | 8 | 0 |

Every reported search used search seed 42 and ended with `CANCELLED` at its invocation deadline. Search seed 123 never started in any of the three invocations. Local processes exited successfully before the final deadline; CI's bounded quality job also succeeded. Successful process termination does not mean the full search protocol completed.

There were **zero failed trials** and no input, search-record, or trial validation issues. Each completed trial has exactly one million epochs and the expected sample count. Every recorded candidate has the five distinct training seeds `1,42,123,999,2026`. Work counters reconcile independently:

| Source | Committed model epochs | Sample updates |
|---|---:|---:|
| Local CPU | 48,070,786 | 8,460,458,336 |
| Local GPU | 10,575,579 | 1,861,301,904 |
| CI CPU | 48,873,363 | 8,601,711,888 |

## Completed candidate verdicts

Every completed candidate group below had **0/5** best-checkpoint scores at or below 0.01 and therefore failed the reliability target. The values are separate five-seed medians from each report; a dash denotes no completed five-seed group in that report. The repeated CPU values are retained as separate observations, not combined into a larger seed sample.

| Hidden layers | Parameters | Local CPU median best RMSE | CI CPU median best RMSE | Local GPU median best RMSE |
|---|---:|---:|---:|---:|
| `[8,8,8]` | 177 | 0.19727788092662593 | 0.19727788092662593 | 0.1706262317603639 |
| `[8,8,16]` | 257 | 0.15025784547691617 | 0.15025784547691617 | — |
| `[4,8,8,8]` | 205 | 0.29766569430109663 | 0.29766569430109663 | — |
| `[8,4,8]` | 109 | 0.28353578735853324 | 0.28353578735853324 | — |
| `[8,4,8,16]` | 261 | 0.3320748905080888 | 0.3320748905080888 | — |
| `[8,4,4,8]` | 129 | 0.2986740329527868 | 0.2986740329527868 | — |
| `[8,8,4,8]` | 181 | 0.3642654816116755 | 0.3642654816116755 | — |
| `[8,4,8,8]` | 181 | 0.2643579507805717 | 0.2643579507805717 | — |

The local CPU partial groups `[8,8]` and `[8,8,16,4]` each completed only one seed. CI completed two and four seeds in those groups, respectively. They remain unqualified separately; these seeds cannot be combined across hosts to complete a candidate.

## Promising, unqualified GPU observations

Both observed threshold crossings belonged to `[8,8,4]`, with **137 parameters**. All five trials were cancelled, and **0/5 completed** the required million epochs:

| Training seed | Committed epochs | Best checkpoint epoch | Best validation RMSE | Last reported validation RMSE | Reached ≤0.01 |
|---|---:|---:|---:|---:|---|
| 1 | 100899 | 9550 | 0.2614638107339375 | 0.311578067170069 | No |
| 42 | 101889 | 101875 | **0.0029529356625192663** | 0.0029529356625192663 | Yes, partial |
| 123 | 102225 | 9750 | **0.009316342789992495** | 0.07548218009688455 | Yes, partial |
| 999 | 102625 | 6075 | 0.2388074885999945 | 0.3577185715260253 | No |
| 2026 | 103755 | 4250 | 0.28323300343495106 | 0.29867228980500776 | No |

The last reported score is the harness's most recent scoring-boundary value. Cancellation can leave additional committed epochs after that score, so it is not necessarily a score of the final committed model state. The saved best checkpoint and its epoch remain explicit.

Two partial observations do not satisfy either requirement: four successful seeds or five fully completed million-epoch trials. No independent test was run for this candidate. It is a concrete observation to preserve, not a reliability qualification. No other retained trial in these three reports reached the 0.01 threshold.

## Evidence and reproducibility

- [Local CPU raw JSONL](quality-cpu.jsonl), SHA-256 `08b40ad94b460367ef0f127665a79dabff0f4745f5f9210c843d16b47757bf3d`.
- [Local GPU raw JSONL](quality-cuda.jsonl), SHA-256 `92e32d7773b9b77501207af2edffde8001167b25f4c2cedc4e196e8f59076dc9`.
- [CI raw JSONL](quality-ci.jsonl) and [CI host/workflow analysis](QUALITY_CI.md).
- [Strict combined analysis JSON](quality-analysis.json) and [per-trial Markdown analysis](quality-analysis.md). The JSON keeps each source separate and records missing search-seed results explicitly.

All individual best/final scores, best epochs, committed counts, states, and failures are retained. No harness winner was selected in any report, and every independent-test field is null. There is therefore no independent-test reliability claim.

Reproduce the strict analysis from the repository root without training or native execution:

```sh
python3 jneuro/benchmarks/2026-10-01-search/analyze_quality.py \
  --input cpu-local=jneuro/benchmarks/2026-10-01-search/quality-cpu.jsonl \
  --input gpu-local=jneuro/benchmarks/2026-10-01-search/quality-cuda.jsonl \
  --input cpu-ci=jneuro/benchmarks/2026-10-01-search/quality-ci.jsonl \
  --search-seeds 42,123 \
  --output quality-recheck
```

The helper succeeds because these are consistent partial reports, while full-search completion and reliability qualification remain false. The evidence neither establishes a smallest reliable architecture nor excludes one elsewhere in the unchanged search domain.
