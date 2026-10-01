# Post-fix correctness smoke

The measured fixed-work outcomes at source `7a0a1ab61c848500316367cb12720662b9f96ff0` exactly match the earlier `529e7a9` outcomes in [local-results.json](local-results.json), comparing the same engine, backend, execution mode, worker count, architecture, and training seed. All three final-source processes exited successfully.

A subsequent logging-only refinement at `cc58045` is validated separately; these runs specifically used `7a0a1ab`. This is correctness continuation evidence only: each engine/backend used one JVM fork, one warmup round, and one measured round for each execution mode. It cannot qualify timing, speedups, or latency distributions. It also does not qualify the separate million-epoch Spiral reliability protocol.

## Verified work and outcomes

Every case completed the same eight-architecture manifest and five training seeds: 40 trials, 2,000 epochs per trial, 80,000 committed epochs, and 14,080,000 sample updates. Every trial completed successfully, with zero failed trials or partial candidates. The dataset fingerprint, 176/44 training/validation split, FP64/EXACT arithmetic, online batch size 1, and scoring interval 25 match the prior fixed-work protocol.

| Engine/backend | Workers | Execution modes checked | Measured trials | Exact prior same-case outcomes | Prior full route/device metadata match |
|---|---:|---|---:|---:|---:|
| REFERENCE/CPU | 8 | REFERENCE, OPTIMIZED | 80 | 80/80 | 80/80 |
| SMALL/CPU | 8 | REFERENCE, OPTIMIZED | 80 | 80/80 | 80/80 |
| SMALL/CUDA | 4 | REFERENCE, OPTIMIZED | 80 | 80/80 | 79/80 |

For all 240 measured trials, state, epoch/sample counters, best checkpoint epoch, best/final validation RMSE, and every saved checkpoint parameter match all nine earlier measured same-case observations exactly. There was no tolerance widening. The retained parameter arrays are checkpoint weights and biases; the benchmark report does not contain final momentum buffers, so this comparison makes no additional momentum-state claim.

The optimized CUDA route used `CUDA_QUEUE` for all 40 measured trials and reached 40 resident models with four CPU scoring workers. CUDA device metadata identifies the NVIDIA GeForce RTX 4060 Ti and driver 13040. The CUDA REFERENCE trial with hidden layers `[4,8,16]`, seed `2026`, used `COHORT`; all nine earlier same-case measured records used `SESSION` for this trial. Its numerical outcome still matches exactly, but it is explicitly excluded from the 239 strictly matching route/device-metadata precedents.

## Retained warmup limitation

The first REFERENCE/CPU/REFERENCE warmup differs from the earlier measured outcomes on 8 of 40 trials. Maximum absolute differences are `0.04177631505221385` in best validation RMSE, `0.057288758933484785` in final validation RMSE, and `9.613073073340736` in a saved checkpoint parameter; one trial selected a different best epoch. The cause is not established by this artifact. The subsequent measured reference and optimized rounds match exactly. The divergent warmup is retained in full and is excluded from measured continuation claims.

## Compact evidence and provenance

[smoke-results.json](smoke-results.json) retains all twelve parsed rounds, 480 trial observations, and 128 distinct numerical outcomes without copying the raw JSONL files. Its canonical reconstruction check covers every complete round, including the divergent warmup. `smokeValidation.roundComparisons` contains per-round comparison counts, differences, and unmatched route metadata.

The artifact includes each raw JSONL filename, byte size and SHA-256; environment and device metadata; successful process-exit sidecars and their hashes; compactor/analysis-script hashes; and the prior artifact hash `f1264865be41149ab7b256fb8baf76883cef21a741d3daefa85db3cb4aefc926`. The three process-exit records and report environments agree on final source `7a0a1ab61c848500316367cb12720662b9f96ff0`. Source revisions are launcher/report declarations; this harness does not emit independent compiled-class hashes.

Recheck canonical reconstruction and measured numerical comparisons from the repository root with Python's standard library:

```sh
python3 - <<'PY'
import json
import sys
from pathlib import Path
folder = Path('jneuro/benchmarks/2026-10-01-search')
sys.path.insert(0, str(folder))
from compact_evidence import reconstruct_round, verify
old = json.loads((folder / 'local-results.json').read_text())
new = json.loads((folder / 'smoke-results.json').read_text())
print(verify(old), verify(new))
baseline = {}
for saved in old['rounds']:
    if saved['round'] < 0:
        continue
    row = reconstruct_round(old, saved)
    for candidate in row['candidates']:
        for trial in candidate['trials']:
            key = (row['case'], tuple(candidate['hidden']), trial['seed'])
            baseline.setdefault(key, []).append(trial)
fields = ('state', 'epochs', 'sampleUpdates', 'bestEpoch',
          'bestRmse', 'finalRmse', 'parametersAtBest')
checked = 0
for saved in new['rounds']:
    if saved['round'] < 0:
        continue
    row = reconstruct_round(new, saved)
    assert row['workComplete'] and row['completeTrials'] == 40
    for candidate in row['candidates']:
        for trial in candidate['trials']:
            key = (row['case'], tuple(candidate['hidden']), trial['seed'])
            assert len(baseline[key]) == 9
            assert all(all(trial[k] == prior[k] for k in fields)
                       for prior in baseline[key])
            checked += 1
assert checked == 240
print('All 240 measured same-case outcomes match exactly.')
PY
```

No runtime, native execution, recompilation, or network access is required for these checks. The timing qualification flags remain false because the smoke lacks independent repeated forks.

After verifying the committed compact artifact against all three inactive raw reports, their duplicate JSONL files were removed: 1,732,501 logical bytes. Full parsed evidence, including divergent warmup parameters, process exits and original byte hashes, remains retained.
