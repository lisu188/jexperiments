# Spiral quality protocol and provenance

This protocol qualifies observed architecture candidates; it does not establish a global minimum over all eligible architectures. Actual results belong in the retained JSONL files and their generated analysis reports. Progress observations alone are not completed results.

## Fixed quality requirements

- Adaptive search over the existing SMALL family: two inputs, one to four hidden layers, each hidden width 4, 8, or 16, and one output; at most 881 parameters.
- Search seeds 42 and 123 are separate searches. Each has a 60-trial budget: twelve architectures with five training seeds each, `1,42,123,999,2026`.
- Every trial retains its full 1,000,000-epoch budget, online batch size 1, FP64 arithmetic, EXACT sigmoid, and a scoring interval of 25 epochs. There is no early pruning or reduction of epochs or seeds.
- A candidate is reliable only when all five distinct training seeds have completed their full budgets successfully, and at least four have a saved best-checkpoint validation RMSE at or below 0.01. This uses the best validation-selected checkpoint, which may differ from the final epoch.
- A failed, cancelled, incomplete, duplicated-seed, non-finite, or otherwise invalid trial prevents that candidate from qualifying. A threshold crossing during partial training is an observation, not a qualifying success.
- Deadline cancellation preserves partial evidence. A fully completed, reliable candidate can be reported within a partial search, with that scope stated explicitly. Unfinished candidates remain unqualified. An unstarted or progress-only search seed is not a completed replication.

A trial cancelled before opening its training session, or failed while opening it, legitimately has no device metadata or checkpoint. The helper accepts this absence only for `CANCELLED`/`FAILED` trials with zero epochs, best epoch, and sample updates; absent device/checkpoint/score metadata; and no checkpoint history. Such a trial remains unqualified, but does not invalidate a different, fully completed reliable candidate in the same partial search. A failed open must retain its failure message. A wrong non-null backend, partial device metadata, or missing metadata on a trained/completed trial is still rejected.

The harness uses the Spiral dataset generated with seed 42 and a 20% validation split using seed 42. Qualification requires exactly 176 training samples, 44 validation samples, and dataset fingerprint `f8fc0ceb97466e339301faa42b05b45df6aee3c878a98986e65771cdac1d4dc9`. A different dataset or split cannot qualify under this protocol. Every best-checkpoint epoch must be a multiple of 25 between zero and the committed epoch count. Best and final validation scores must be finite and nonnegative, and the best score must not exceed the final score. Eligibility, ranking, and training continue to use the existing search algorithm; completion order can affect later adaptive proposals, so different execution routes need not visit identical architecture sets.

## Separate execution environments

Local CPU quality uses eight workers while local CUDA quality uses four CPU scoring workers. These two quality runs execute concurrently on the same interactive machine. Their elapsed times are not throughput benchmarks or valid CPU-versus-GPU performance comparisons. GPU resident-model count is separate from the scoring-worker count.

CI CPU quality runs on a separate host with four workers and its own deadline. Its results remain separate from local evidence. No trials, training seeds, search cases, hosts, processes, or source revisions are pooled to manufacture a completed candidate or full search.

JSONL environment records retain the source revision, JVM and OS details, CPU description, process ID, JVM arguments, dataset fingerprint, and configured execution selections. Trial records retain actual backend, route, engine, precision, sigmoid, kernel provenance, and device identity. The analysis adds the input SHA-256 and byte count at the time it reads the file. These hashes describe that observed file version; a running harness may append more records later.

The current harness environment does not emit the selected search-seed list or maximum trial count. The helper therefore identifies expected search seeds 42/123 and the 60-trial budget as analysis-contract inputs, not as fields recovered from runtime provenance. Explicitly change `--search-seeds` only when describing a deliberately narrower observed invocation; a missing second search must remain missing in the full two-search protocol.

Evidence under a `quality-superseded` path is excluded. That exploratory run was cancelled before the progress-snapshot race fix and cannot support qualification.

## Reproduce the analysis

The helper uses only Python's standard library and never launches training, native code, or network requests. Run these commands from the repository root, replacing the three input placeholders with the retained JSONL paths:

```sh
python3 jneuro/benchmarks/2026-10-01-search/analyze_quality.py \
  --input cpu-local=PATH_TO_CPU_JSONL \
  --input gpu-local=PATH_TO_GPU_JSONL \
  --input cpu-ci=PATH_TO_CI_JSONL \
  --search-seeds 42,123 \
  --output quality-analysis
```

This writes `quality-analysis.json` and `quality-analysis.md`. Input labels must be unique. Omit an input option only when intentionally analyzing fewer sources; do not treat an omitted source as a successful run.

```sh
python3 -m unittest discover \
  -s jneuro/benchmarks/2026-10-01-search \
  -p test_analyze_quality.py
```

The sixteen focused tests cover completed and partial candidates/searches, legitimate unopened cancellation/open-failure alongside completed reliable candidates, rejection of forged unopened states, duplicate seeds, short budgets, failures, wrong arithmetic or target, incorrect dataset fingerprints or sample counts, off-boundary best epochs, inconsistent score ordering, progress-only evidence, malformed or partial JSONL, superseded paths, and the independent-test scope. Tests use bounded temporary fixtures and no additional dependencies.

## Report interpretation

The JSON report has `schemaVersion`, an explicit `protocol`, and separate `inputs`. Each input contains its environment, content hash, validation issues, search records, latest progress, failure records, and expected cases that have not produced a result.

Each search record reports:

- `fullWorkComplete`: all sixty trials and twelve distinct eligible architectures completed correctly, with consistent epoch and sample-update counts and a normal completion or trial-budget termination.
- `reliableCompletedCandidates`: candidates meeting the full five-seed requirement, including candidates found within an otherwise partial search.
- `smallestObservedReliableCandidate`: the smallest qualifying candidate in this search's completed evidence, never a global-minimum claim.
- Per-candidate best/final validation RMSE, best epoch, state, committed epochs, sample updates, failures, and qualification checks for every recorded training seed.
- `independentTest`: only the harness-selected winner's representative seed evaluated on the independent 218-point Spiral grid. This is neither five-seed independent-test reliability nor an independent architecture-selection procedure.

`allExpectedSearchCasesComplete` requires completed results for every explicitly expected case and search seed. A `partial_run` may contain a reliable completed candidate; `progress_only_unqualified` contains no candidate-level evidence sufficient for qualification. Non-finite scores and invalid counters never count as success. Parameter arrays are retained in raw evidence but omitted from the compact analysis.

An unfinished final JSONL line is ignored and flagged, allowing inspection of a file while the harness is still flushing. Invalid earlier lines invalidate qualification. Exit status 1 denotes malformed or inconsistent evidence; valid negative and incomplete quality results return 0 and retain explicit completion statuses. No result in this analysis establishes a global minimum.
