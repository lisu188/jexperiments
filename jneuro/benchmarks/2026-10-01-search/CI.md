# CI fixed-search evidence

Compute source: `974b22ca8e93a726038364eac163f7f184521abc`; Java 27+35; Linux 6.17.0-1022-azure; four logical processors and 2 GiB maximum heap per hosted VM. All nine reports have complete manifests and budgets, 40 warmup/measured search calls and 10 final summaries.

**Qualification is inconclusive.** The three matrix jobs ran on three different CPU models: EPYC 7763, EPYC 9V74, and Xeon 6973P-C. Each hardware/protocol has only one JVM fork and three measured rounds. The reports must not be pooled to claim three compatible forks. All 45 comparisons correctly return `qualifies=false`; no raw rounds were rejected.

All measured within-engine reference-versus-optimized best parameters, best/final scores and best epochs match exactly (maximum scaled error 0). The 90 warmup searches are excluded from timing distributions; six warmup-only outcome variants are retained explicitly. There are 270 measured search calls and 9,000 measured trial observations, but only the declared candidate/seed identities count as quality samples. No target success occurred in these 2,000-epoch cases.

## Observed single-fork timings

Times are whole-search medians in seconds. Each row has three observations per mode; p95 is therefore the maximum observed search duration. These figures are descriptive, not qualified gains. `Reference` and `Optimized` name execution modes; the engine column names the arithmetic implementation.

| CPU | Manifest | Engine | Workers | Reference median | Optimized median | Ratio | Reference p95 | Optimized p95 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| EPYC 7763 | general | REFERENCE | 1 | 6.150 | 4.051 | 1.518x | 6.173 | 4.051 |
| EPYC 7763 | general | REFERENCE | 4 | 2.650 | 1.799 | 1.473x | 2.690 | 1.817 |
| EPYC 7763 | general | REFERENCE | 8 | 2.634 | 1.783 | 1.477x | 2.656 | 1.786 |
| EPYC 7763 | general | REFERENCE | 16 | 2.566 | 1.755 | 1.462x | 2.566 | 1.757 |
| EPYC 7763 | general | REFERENCE | 32 | 2.550 | 1.735 | 1.470x | 2.560 | 1.741 |
| EPYC 7763 | small | REFERENCE | 1 | 16.610 | 10.873 | 1.528x | 16.628 | 10.910 |
| EPYC 7763 | small | REFERENCE | 4 | 6.748 | 4.564 | 1.478x | 6.755 | 4.588 |
| EPYC 7763 | small | REFERENCE | 8 | 6.661 | 4.509 | 1.477x | 6.805 | 4.511 |
| EPYC 7763 | small | REFERENCE | 16 | 6.770 | 4.499 | 1.505x | 6.784 | 4.538 |
| EPYC 7763 | small | REFERENCE | 32 | 6.656 | 4.466 | 1.490x | 6.691 | 4.493 |
| EPYC 7763 | small | SMALL | 1 | 10.432 | 10.207 | 1.022x | 10.433 | 10.209 |
| EPYC 7763 | small | SMALL | 4 | 5.909 | 4.471 | 1.322x | 7.791 | 4.506 |
| EPYC 7763 | small | SMALL | 8 | 9.235 | 4.425 | 2.087x | 9.250 | 4.468 |
| EPYC 7763 | small | SMALL | 16 | 8.158 | 4.407 | 1.851x | 8.739 | 4.435 |
| EPYC 7763 | small | SMALL | 32 | 10.046 | 4.442 | 2.262x | 10.470 | 4.443 |
| EPYC 9V74 | general | REFERENCE | 1 | 6.195 | 4.080 | 1.518x | 6.197 | 4.109 |
| EPYC 9V74 | general | REFERENCE | 4 | 2.759 | 1.857 | 1.486x | 2.764 | 1.860 |
| EPYC 9V74 | general | REFERENCE | 8 | 2.734 | 1.813 | 1.508x | 2.784 | 1.857 |
| EPYC 9V74 | general | REFERENCE | 16 | 2.702 | 1.800 | 1.501x | 2.708 | 1.810 |
| EPYC 9V74 | general | REFERENCE | 32 | 2.681 | 1.787 | 1.500x | 2.696 | 1.798 |
| EPYC 9V74 | small | REFERENCE | 1 | 15.501 | 10.145 | 1.528x | 15.506 | 10.236 |
| EPYC 9V74 | small | REFERENCE | 4 | 6.480 | 4.319 | 1.500x | 6.505 | 4.355 |
| EPYC 9V74 | small | REFERENCE | 8 | 6.459 | 4.231 | 1.527x | 6.477 | 4.303 |
| EPYC 9V74 | small | REFERENCE | 16 | 6.382 | 4.250 | 1.502x | 6.440 | 4.258 |
| EPYC 9V74 | small | REFERENCE | 32 | 6.403 | 4.249 | 1.507x | 6.412 | 4.251 |
| EPYC 9V74 | small | SMALL | 1 | 10.752 | 10.659 | 1.009x | 10.772 | 10.671 |
| EPYC 9V74 | small | SMALL | 4 | 6.042 | 4.719 | 1.280x | 7.987 | 4.778 |
| EPYC 9V74 | small | SMALL | 8 | 9.253 | 4.670 | 1.981x | 9.763 | 4.699 |
| EPYC 9V74 | small | SMALL | 16 | 8.616 | 4.636 | 1.859x | 9.437 | 4.735 |
| EPYC 9V74 | small | SMALL | 32 | 10.639 | 4.649 | 2.288x | 10.877 | 4.721 |
| Xeon 6973P-C | general | REFERENCE | 1 | 4.007 | 2.668 | 1.502x | 4.201 | 2.833 |
| Xeon 6973P-C | general | REFERENCE | 4 | 1.886 | 1.342 | 1.406x | 1.934 | 1.481 |
| Xeon 6973P-C | general | REFERENCE | 8 | 1.923 | 1.259 | 1.528x | 1.955 | 1.276 |
| Xeon 6973P-C | general | REFERENCE | 16 | 1.913 | 1.412 | 1.355x | 1.917 | 1.435 |
| Xeon 6973P-C | general | REFERENCE | 32 | 1.897 | 1.260 | 1.506x | 1.908 | 1.292 |
| Xeon 6973P-C | small | REFERENCE | 1 | 11.705 | 8.270 | 1.415x | 11.943 | 8.306 |
| Xeon 6973P-C | small | REFERENCE | 4 | 5.194 | 3.490 | 1.488x | 5.284 | 3.550 |
| Xeon 6973P-C | small | REFERENCE | 8 | 5.203 | 3.476 | 1.497x | 5.308 | 3.688 |
| Xeon 6973P-C | small | REFERENCE | 16 | 5.132 | 3.439 | 1.492x | 5.249 | 3.583 |
| Xeon 6973P-C | small | REFERENCE | 32 | 5.307 | 3.482 | 1.524x | 5.356 | 3.743 |
| Xeon 6973P-C | small | SMALL | 1 | 7.068 | 6.806 | 1.039x | 7.609 | 6.822 |
| Xeon 6973P-C | small | SMALL | 4 | 4.621 | 3.262 | 1.417x | 4.837 | 3.449 |
| Xeon 6973P-C | small | SMALL | 8 | 5.540 | 3.111 | 1.781x | 5.563 | 3.231 |
| Xeon 6973P-C | small | SMALL | 16 | 5.557 | 3.088 | 1.800x | 5.808 | 3.228 |
| Xeon 6973P-C | small | SMALL | 32 | 6.504 | 3.075 | 2.115x | 6.619 | 3.151 |

## Interpretation and retained evidence

No per-host comparison shows a p95 regression. SMALL with one worker improves only 0.9–3.7% in observed median time and fails the 10% reduction rule even before the fork-count requirement. The general-width manifest improves in all three hosts without restricting the search to SMALL shapes. Worker counts above four oversubscribe these hosted VMs.

Across engines, equal work is not identical arithmetic. The general engine AUTO path can use horizontally reduced dot products and separate multiply/add updates; SMALL uses ordered FMA accumulation. Cross-engine and cross-host outcome differences must remain separate from the exact within-engine execution parity checks.

[ci-results.json](ci-results.json) retains deduplicated protocols, all 360 search-call durations and counters, every per-trial duration/phase duration, effective route/device metadata, 306 uniquely identified outcomes and all 45 comparison verdicts. Each unique outcome stores its complete best-parameter array once. Every trial observation links to its outcome, so all 12,000 trial records and all parsed round records can be reconstructed independently from this compact file. Reconstructed records matched the original parsed records and canonical SHA-256 hashes exactly; deliberate parameter and timing corruption was rejected. [ci-qualification-summary.json](ci-qualification-summary.json) is the unmodified strict summarizer output. Analyzer and compactor hashes, the nine original JSONL byte hashes and all three host-report hashes are retained. After exact reconstruction and original byte-hash verification against the committed compact artifact, the nine inactive raw JSONL copies were removed from task scratch storage (39,771,833 bytes). Host reports and provenance remain; the complete parsed round data is retained in `ci-results.json`.

No process-exit sidecars were included in these CI artifacts. Workflow-level success must be verified separately before asserting execution acceptance. The compact outcome arrays preserve the numerical evidence needed to recheck the recorded best-snapshot and score comparisons without the original raw reports. Final optimizer-state and shuffle-continuation parity remain separate automated-test claims.

