# CI Spiral quality result

This bounded CI run did not find a qualifying architecture and did not complete the full two-search quality protocol. Eight candidate groups completed all five million-epoch trials; none had any training seed reach best-checkpoint validation RMSE ≤0.01. The deadline interrupted search seed 42, and search seed 123 did not start. This does not establish that no qualifying architecture exists or identify a global minimum.

[Workflow run 36811931542](https://github.com/lisu188/jexperiments/actions/runs/36811931542) successfully ran the bounded experiment at source `7a0a1ab61c848500316367cb12720662b9f96ff0`. Workflow success means the job completed and retained evidence, not that the search completed or met its quality target.

## Completion and validation

The run used SMALL/CPU/OPTIMIZED with four workers, FP64/EXACT, online batch size 1, and scoring every 25 epochs. The separate CI host exposed four logical processors from an AMD EPYC 7763 and ran Linux with OpenJDK 27. Dataset fingerprint and the 176/44 training/validation split exactly match [QUALITY_PROTOCOL.md](QUALITY_PROTOCOL.md).

- Search seed 42 stopped with `CANCELLED` after 3,500.017923632 seconds against a 3,500-second deadline.
- 46 trials completed exactly 1,000,000 epochs; four additional trials were cancelled after partial work. Ten of the 60 trial-budget slots remained unused.
- Eight architectures completed all five specified seeds; two architectures remained partial. There were zero failed trials and no malformed or inconsistent records detected by the strict helper.
- Total committed work was 48,873,363 model epochs and 8,601,711,888 sample updates.
- Search seed 123 has no result or progress record. The final budget-expired record confirms the invocation exhausted its budget before that search started.
- No candidate qualified, no harness winner was selected, and no independent-test score was produced.

The smallest individual best validation RMSE observed was **0.0567555991529361**, from `[8,8,16]`, training seed `999`, at epoch `7350`; that trial completed its million epochs with final RMSE `0.15606590754672028`. Even this best observation exceeds the 0.01 target.

| Hidden layers | Parameters | Full-budget trials | Qualifying successes | Median best RMSE across five completed seeds |
|---|---:|---:|---:|---:|
| `[8,8,8]` | 177 | 5/5 | 0/5 | 0.19727788092662593 |
| `[8,8,16]` | 257 | 5/5 | 0/5 | 0.15025784547691617 |
| `[4,8,8,8]` | 205 | 5/5 | 0/5 | 0.29766569430109663 |
| `[8,4,8]` | 109 | 5/5 | 0/5 | 0.28353578735853324 |
| `[8,4,8,16]` | 261 | 5/5 | 0/5 | 0.3320748905080888 |
| `[8,4,4,8]` | 129 | 5/5 | 0/5 | 0.2986740329527868 |
| `[8,8,4,8]` | 181 | 5/5 | 0/5 | 0.3642654816116755 |
| `[8,4,8,8]` | 181 | 5/5 | 0/5 | 0.2643579507805717 |
| `[8,8]` | 105 | 2/5 | Unqualified partial group | — |
| `[8,8,16,4]` | 313 | 4/5 | Unqualified partial group | — |

The cancelled trials were `[8,8]` seeds `123`, `999`, and `2026` at epochs `969930`, `695925`, and `382188`, respectively, and `[8,8,16,4]` seed `2026` at epoch `825320`. Their observations remain in the reports but cannot qualify either partial candidate group. All fifty recorded trials retain their individual best/final scores, best epochs, counters, and states in the analysis JSON.

## Retained evidence

These CI observations remain separate from concurrent local CPU/GPU quality runs. No hosts, trials, or seeds are pooled, and this record makes no cross-host throughput comparison.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| [Raw report](quality-ci.jsonl) | 235670 | `f0dd55d9ad1166a5234f8f7ae92d4e267979b09cd9d49945d46e07617ed9d628` |
| [Strict analysis](quality-ci-analysis.json) | 55890 | `0bacff993fab53ba0602a1be0bbc6afb28d96910ac501b694c90906b598b7ba9` |
| [Host description](quality-ci-host.txt) | 3351 | `c38c4cda71923c7891561cda3eb66ebceed6601447e75f9c36adab91d76a4ff6` |
| [Workflow provenance](quality-ci-workflow.json) | 2136 | `91b5cf6a4b61a6aea255165d383d0ac601f60f6080283329b319b682eb75894c` |

The analyzer used for this report has SHA-256 `f7b39e56fe525f901cd452fa91ac23d676c30fc76b7021f50f390adac18909fd`; its sixteen focused regression tests cover the acceptance rules. The report was analyzed with both expected search seeds, so the unstarted second search remains explicitly missing:

```sh
python3 jneuro/benchmarks/2026-10-01-search/analyze_quality.py \
  --input cpu-ci=jneuro/benchmarks/2026-10-01-search/quality-ci.jsonl \
  --search-seeds 42,123 \
  --output quality-ci-recheck
```

The helper returned success for a consistent **partial** result, with no input or search-record validation issues. The workflow source revision agrees with the raw report. Only the quality job ran in this workflow invocation; skipped build/benchmark jobs are not counted as validation here.
