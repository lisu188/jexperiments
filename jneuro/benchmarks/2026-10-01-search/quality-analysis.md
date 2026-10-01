# Spiral quality evidence

Target: validation RMSE ≤0.01 in at least 4/5 specified seeds; all five trials must finish 1,000,000 epochs. Each search has a 60-trial budget. No global minimum is established.

Host/process results remain separate. Concurrent local CPU/GPU runs support quality observations, not throughput comparisons. Progress and partial threshold crossings never establish reliability.

| Source | Case | Status | Full-budget trials | Reliable completed candidates |
|---|---|---|---:|---|
| cpu-local | SMALL/CPU/OPTIMIZED/workers=8/searchSeed=42 | partial_search | 42/60 | None |
| gpu-local | SMALL/CUDA/OPTIMIZED/workers=4/searchSeed=42 | partial_search | 5/60 | None |
| cpu-ci | SMALL/CPU/OPTIMIZED/workers=4/searchSeed=42 | partial_search | 46/60 | None |

## cpu-local

Source: `jneuro/benchmarks/2026-10-01-search/quality-cpu.jsonl`

### SMALL/CPU/OPTIMIZED/workers=8/searchSeed=42

Reliability scope: within_completed_candidates_of_partial_search.

Candidate [8, 8, 8] / 177 parameters: fully completed=True; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | COMPLETED | 1000000 | 5075 | 0.22982123673136914 | 0.2592767160020913 |  |
| 42 | COMPLETED | 1000000 | 12975 | 0.19727788092662593 | 0.45947890328890784 |  |
| 123 | COMPLETED | 1000000 | 1000000 | 0.15607120666672347 | 0.15607120666672347 |  |
| 999 | COMPLETED | 1000000 | 4725 | 0.21081912635008343 | 0.21318123739116784 |  |
| 2026 | COMPLETED | 1000000 | 18500 | 0.07661367889283695 | 0.49009395198538214 |  |

Candidate [8, 8, 16] / 257 parameters: fully completed=True; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | COMPLETED | 1000000 | 5975 | 0.25021869023395626 | 0.513584810221858 |  |
| 42 | COMPLETED | 1000000 | 3425 | 0.15025784547691617 | 0.21307961946115198 |  |
| 123 | COMPLETED | 1000000 | 3750 | 0.14907647440285465 | 0.19150640834294103 |  |
| 999 | COMPLETED | 1000000 | 7350 | 0.0567555991529361 | 0.15606590754672028 |  |
| 2026 | COMPLETED | 1000000 | 5675 | 0.15707815813159964 | 0.20211609832699057 |  |

Candidate [8, 4, 8] / 109 parameters: fully completed=True; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | COMPLETED | 1000000 | 4150 | 0.24397831117043617 | 0.4710761928407419 |  |
| 42 | COMPLETED | 1000000 | 194300 | 0.40399219870847713 | 0.5241558119623153 |  |
| 123 | COMPLETED | 1000000 | 4725 | 0.3863720741065691 | 0.5002552956864837 |  |
| 999 | COMPLETED | 1000000 | 5575 | 0.17410866575270756 | 0.18717181745196393 |  |
| 2026 | COMPLETED | 1000000 | 5725 | 0.28353578735853324 | 0.47849122113069736 |  |

Candidate [4, 8, 8, 8] / 205 parameters: fully completed=True; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | COMPLETED | 1000000 | 21575 | 0.38598770112113756 | 0.5064808658405723 |  |
| 42 | COMPLETED | 1000000 | 12475 | 0.23240020456503474 | 0.4807819176016708 |  |
| 123 | COMPLETED | 1000000 | 16850 | 0.3611488980903106 | 0.500159071063721 |  |
| 999 | COMPLETED | 1000000 | 26675 | 0.29766569430109663 | 0.4883447220156318 |  |
| 2026 | COMPLETED | 1000000 | 15200 | 0.2813663277255416 | 0.4952011031329603 |  |

Candidate [8, 4, 8, 16] / 261 parameters: fully completed=True; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | COMPLETED | 1000000 | 6750 | 0.20682219497387108 | 0.3265902314401614 |  |
| 42 | COMPLETED | 1000000 | 6150 | 0.2533487374708068 | 0.5057651695793889 |  |
| 123 | COMPLETED | 1000000 | 112950 | 0.3320748905080888 | 0.44280734885219364 |  |
| 999 | COMPLETED | 1000000 | 25375 | 0.3950449789590552 | 0.46510599962775534 |  |
| 2026 | COMPLETED | 1000000 | 22900 | 0.36065497327690155 | 0.4687634112099268 |  |

Candidate [8, 4, 4, 8] / 129 parameters: fully completed=True; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | COMPLETED | 1000000 | 18000 | 0.2986740329527868 | 0.48138291323524324 |  |
| 42 | COMPLETED | 1000000 | 11550 | 0.33343737571397997 | 0.48246384451884994 |  |
| 123 | COMPLETED | 1000000 | 80425 | 0.3764305513319639 | 0.39198892393797086 |  |
| 999 | COMPLETED | 1000000 | 10525 | 0.2940622161147262 | 0.49211098786916685 |  |
| 2026 | COMPLETED | 1000000 | 9350 | 0.26338358179060495 | 0.49600066482853045 |  |

Candidate [8, 8, 4, 8] / 181 parameters: fully completed=True; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | COMPLETED | 1000000 | 94800 | 0.3540490052365065 | 0.4863116835758065 |  |
| 42 | COMPLETED | 1000000 | 3675 | 0.46936403779241537 | 0.5292584068084838 |  |
| 123 | COMPLETED | 1000000 | 10700 | 0.4673580993143123 | 0.4941581641431109 |  |
| 999 | COMPLETED | 1000000 | 35450 | 0.3642654816116755 | 0.445652823774186 |  |
| 2026 | COMPLETED | 1000000 | 645675 | 0.3182177653097129 | 0.4919556740805861 |  |

Candidate [8, 4, 8, 8] / 181 parameters: fully completed=True; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | COMPLETED | 1000000 | 17775 | 0.3350020697260805 | 0.48100355709211595 |  |
| 42 | COMPLETED | 1000000 | 5700 | 0.2325343034610923 | 0.47074553494708826 |  |
| 123 | COMPLETED | 1000000 | 13525 | 0.14575340406186227 | 0.15088058951137906 |  |
| 999 | COMPLETED | 1000000 | 22750 | 0.3081276876714257 | 0.4647361100585846 |  |
| 2026 | COMPLETED | 1000000 | 16900 | 0.2643579507805717 | 0.50075837051937 |  |

Candidate [8, 8, 16, 4] / 313 parameters: fully completed=False; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | COMPLETED | 1000000 | 29325 | 0.3272858618316648 | 0.4727412770913822 |  |
| 42 | CANCELLED | 942035 | 10025 | 0.21011254233536514 | 0.2120819423767234 |  |
| 123 | CANCELLED | 932778 | 27875 | 0.2101787956694461 | 0.46990356583703896 |  |
| 999 | CANCELLED | 925157 | 19050 | 0.41285474185688203 | 0.5076621538050893 |  |
| 2026 | CANCELLED | 824520 | 24725 | 0.2869181363852901 | 0.47204733144214484 |  |

Candidate [8, 8] / 105 parameters: fully completed=False; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | COMPLETED | 1000000 | 1000000 | 0.22704534110236363 | 0.22704534110236363 |  |
| 42 | CANCELLED | 924986 | 3775 | 0.14231530454829902 | 0.15040663188386993 |  |
| 123 | CANCELLED | 903233 | 8175 | 0.17716627155478992 | 0.2610015571999805 |  |
| 999 | CANCELLED | 535796 | 535775 | 0.1197296000238389 | 0.1197296000238389 |  |
| 2026 | CANCELLED | 82281 | 47550 | 0.3410441230294465 | 0.35714187261843383 |  |

Unreported case: `SMALL/CPU/OPTIMIZED/workers=8/searchSeed=123` — not_observed_no_result.

## gpu-local

Source: `jneuro/benchmarks/2026-10-01-search/quality-cuda.jsonl`

### SMALL/CUDA/OPTIMIZED/workers=4/searchSeed=42

Reliability scope: within_completed_candidates_of_partial_search.

Candidate [8, 8, 8] / 177 parameters: fully completed=True; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | COMPLETED | 1000000 | 7300 | 0.24764969781302543 | 0.26319467011841513 |  |
| 42 | COMPLETED | 1000000 | 8850 | 0.15035330527326146 | 0.2049369195903733 |  |
| 123 | COMPLETED | 1000000 | 5625 | 0.10274083005568825 | 0.1501491304329105 |  |
| 999 | COMPLETED | 1000000 | 5050 | 0.2469414665033835 | 0.5274952953765825 |  |
| 2026 | COMPLETED | 1000000 | 7525 | 0.1706262317603639 | 0.3096236766572467 |  |

Candidate [8, 4, 8, 8] / 181 parameters: fully completed=False; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | CANCELLED | 101525 | 26950 | 0.37449037525165 | 0.4711056898560449 |  |
| 42 | CANCELLED | 99752 | 5900 | 0.33559553033206846 | 0.4521097295999621 |  |
| 123 | CANCELLED | 101900 | 22225 | 0.3836291090854403 | 0.5215395060982723 |  |
| 999 | CANCELLED | 100125 | 23450 | 0.2200418302866184 | 0.45076664778119985 |  |
| 2026 | CANCELLED | 101800 | 19375 | 0.3513111818143368 | 0.43802184997810495 |  |

Candidate [8, 16, 8] / 313 parameters: fully completed=False; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | CANCELLED | 100398 | 11200 | 0.1955980275655333 | 0.4511793835494752 |  |
| 42 | CANCELLED | 101814 | 5125 | 0.18245315179673607 | 0.41195032414468147 |  |
| 123 | CANCELLED | 101350 | 5875 | 0.1477762218252802 | 0.1611514711793137 |  |
| 999 | CANCELLED | 101327 | 4400 | 0.20248821465940045 | 0.2256711507604993 |  |
| 2026 | CANCELLED | 100539 | 6125 | 0.14374543947338447 | 0.1505308373668828 |  |

Candidate [16, 8, 8] / 265 parameters: fully completed=False; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | CANCELLED | 100129 | 33600 | 0.1498300217728793 | 0.14985420427463417 |  |
| 42 | CANCELLED | 100694 | 100675 | 0.019795556255832354 | 0.019795556255832354 |  |
| 123 | CANCELLED | 100725 | 5100 | 0.154538640944474 | 0.19514202761812896 |  |
| 999 | CANCELLED | 101089 | 2700 | 0.06047332583402862 | 0.06594553860462832 |  |
| 2026 | CANCELLED | 100641 | 100625 | 0.15385244443245527 | 0.15385244443245527 |  |

Candidate [8, 8, 16] / 257 parameters: fully completed=False; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | CANCELLED | 100095 | 8625 | 0.1811012286984217 | 0.5031119167071469 |  |
| 42 | CANCELLED | 101817 | 2925 | 0.13276177245420503 | 0.15416505271456196 |  |
| 123 | CANCELLED | 101853 | 101850 | 0.14559100406441747 | 0.14559100406441747 |  |
| 999 | CANCELLED | 100820 | 6550 | 0.30341683532672753 | 0.3375961749104748 |  |
| 2026 | CANCELLED | 102397 | 4950 | 0.14976571615438602 | 0.1506519066810889 |  |

Candidate [4, 8, 8, 8] / 205 parameters: fully completed=False; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | CANCELLED | 100230 | 15675 | 0.36862951758226437 | 0.5309237794962135 |  |
| 42 | CANCELLED | 99655 | 8850 | 0.34551305093737 | 0.47833137492229977 |  |
| 123 | CANCELLED | 101022 | 12525 | 0.37978223381816123 | 0.48245475884811184 |  |
| 999 | CANCELLED | 101342 | 42175 | 0.3775400545275943 | 0.49440511124990016 |  |
| 2026 | CANCELLED | 100114 | 14750 | 0.3285883213170464 | 0.45818040310893365 |  |

Candidate [8, 8, 4, 8] / 181 parameters: fully completed=False; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | CANCELLED | 100088 | 98175 | 0.41201223142692983 | 0.5405012586986538 |  |
| 42 | CANCELLED | 100907 | 29375 | 0.335455364594443 | 0.4829993963573503 |  |
| 123 | CANCELLED | 100025 | 34625 | 0.31166402178711516 | 0.46690294391231313 |  |
| 999 | CANCELLED | 101200 | 26950 | 0.33722218045247904 | 0.4568089602794678 |  |
| 2026 | CANCELLED | 100625 | 69000 | 0.4609763558817934 | 0.46232916099324123 |  |

Candidate [8, 8] / 105 parameters: fully completed=False; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | CANCELLED | 104780 | 6325 | 0.12422506095090458 | 0.13428778438650324 |  |
| 42 | CANCELLED | 103250 | 4125 | 0.18510758139406477 | 0.6422276929253309 |  |
| 123 | CANCELLED | 103650 | 8450 | 0.21168027630508937 | 0.2605521861647489 |  |
| 999 | CANCELLED | 104462 | 29625 | 0.26408503110356774 | 0.26453030990391024 |  |
| 2026 | CANCELLED | 104450 | 8950 | 0.14448038383820244 | 0.16566513000388042 |  |

Candidate [4, 8, 8] / 133 parameters: fully completed=False; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | CANCELLED | 102886 | 5975 | 0.3995949480630964 | 0.45064125582034165 |  |
| 42 | CANCELLED | 101123 | 5700 | 0.2834293172784538 | 0.5592564169320207 |  |
| 123 | CANCELLED | 102648 | 6950 | 0.14699705621353037 | 0.47669094750751223 |  |
| 999 | CANCELLED | 101825 | 9100 | 0.3554591307152626 | 0.460114723366087 |  |
| 2026 | CANCELLED | 102819 | 10300 | 0.2782856108293662 | 0.5020143121062328 |  |

Candidate [8, 4, 8] / 109 parameters: fully completed=False; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | CANCELLED | 102250 | 3975 | 0.2533614750780032 | 0.4968608066493283 |  |
| 42 | CANCELLED | 100373 | 28900 | 0.2234845357393132 | 0.45397433319888914 |  |
| 123 | CANCELLED | 102058 | 4900 | 0.22470199627000675 | 0.48759614007209545 |  |
| 999 | CANCELLED | 101281 | 5750 | 0.17294168381951133 | 0.2110234076211112 |  |
| 2026 | CANCELLED | 100861 | 14225 | 0.27429828496023634 | 0.466917511551111 |  |

Candidate [8, 8, 4] / 137 parameters: fully completed=False; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | CANCELLED | 100899 | 9550 | 0.2614638107339375 | 0.311578067170069 |  |
| 42 | CANCELLED | 101889 | 101875 | 0.0029529356625192663 | 0.0029529356625192663 |  |
| 123 | CANCELLED | 102225 | 9750 | 0.009316342789992495 | 0.07548218009688455 |  |
| 999 | CANCELLED | 102625 | 6075 | 0.2388074885999945 | 0.3577185715260253 |  |
| 2026 | CANCELLED | 103755 | 4250 | 0.28323300343495106 | 0.29867228980500776 |  |

Candidate [8, 8, 8, 8] / 249 parameters: fully completed=False; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | CANCELLED | 99725 | 40300 | 0.3202155659758674 | 0.506094178183137 |  |
| 42 | CANCELLED | 101137 | 7975 | 0.3314030651726935 | 0.4768510154690509 |  |
| 123 | CANCELLED | 100875 | 11000 | 0.3461611476671266 | 0.4901423196042691 |  |
| 999 | CANCELLED | 97712 | 11600 | 0.30992895023885164 | 0.4940582647517905 |  |
| 2026 | CANCELLED | 100023 | 28500 | 0.174128755457232 | 0.5163661344708196 |  |

Unreported case: `SMALL/CUDA/OPTIMIZED/workers=4/searchSeed=123` — not_observed_no_result.

## cpu-ci

Source: `jneuro/benchmarks/2026-10-01-search/quality-ci.jsonl`

### SMALL/CPU/OPTIMIZED/workers=4/searchSeed=42

Reliability scope: within_completed_candidates_of_partial_search.

Candidate [8, 8, 8] / 177 parameters: fully completed=True; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | COMPLETED | 1000000 | 5075 | 0.22982123673136914 | 0.2592767160020913 |  |
| 42 | COMPLETED | 1000000 | 12975 | 0.19727788092662593 | 0.45947890328890784 |  |
| 123 | COMPLETED | 1000000 | 1000000 | 0.15607120666672347 | 0.15607120666672347 |  |
| 999 | COMPLETED | 1000000 | 4725 | 0.21081912635008343 | 0.21318123739116784 |  |
| 2026 | COMPLETED | 1000000 | 18500 | 0.07661367889283695 | 0.49009395198538214 |  |

Candidate [8, 8, 16] / 257 parameters: fully completed=True; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | COMPLETED | 1000000 | 5975 | 0.25021869023395626 | 0.513584810221858 |  |
| 42 | COMPLETED | 1000000 | 3425 | 0.15025784547691617 | 0.21307961946115198 |  |
| 123 | COMPLETED | 1000000 | 3750 | 0.14907647440285465 | 0.19150640834294103 |  |
| 999 | COMPLETED | 1000000 | 7350 | 0.0567555991529361 | 0.15606590754672028 |  |
| 2026 | COMPLETED | 1000000 | 5675 | 0.15707815813159964 | 0.20211609832699057 |  |

Candidate [4, 8, 8, 8] / 205 parameters: fully completed=True; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | COMPLETED | 1000000 | 21575 | 0.38598770112113756 | 0.5064808658405723 |  |
| 42 | COMPLETED | 1000000 | 12475 | 0.23240020456503474 | 0.4807819176016708 |  |
| 123 | COMPLETED | 1000000 | 16850 | 0.3611488980903106 | 0.500159071063721 |  |
| 999 | COMPLETED | 1000000 | 26675 | 0.29766569430109663 | 0.4883447220156318 |  |
| 2026 | COMPLETED | 1000000 | 15200 | 0.2813663277255416 | 0.4952011031329603 |  |

Candidate [8, 4, 8] / 109 parameters: fully completed=True; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | COMPLETED | 1000000 | 4150 | 0.24397831117043617 | 0.4710761928407419 |  |
| 42 | COMPLETED | 1000000 | 194300 | 0.40399219870847713 | 0.5241558119623153 |  |
| 123 | COMPLETED | 1000000 | 4725 | 0.3863720741065691 | 0.5002552956864837 |  |
| 999 | COMPLETED | 1000000 | 5575 | 0.17410866575270756 | 0.18717181745196393 |  |
| 2026 | COMPLETED | 1000000 | 5725 | 0.28353578735853324 | 0.47849122113069736 |  |

Candidate [8, 4, 8, 16] / 261 parameters: fully completed=True; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | COMPLETED | 1000000 | 6750 | 0.20682219497387108 | 0.3265902314401614 |  |
| 42 | COMPLETED | 1000000 | 6150 | 0.2533487374708068 | 0.5057651695793889 |  |
| 123 | COMPLETED | 1000000 | 112950 | 0.3320748905080888 | 0.44280734885219364 |  |
| 999 | COMPLETED | 1000000 | 25375 | 0.3950449789590552 | 0.46510599962775534 |  |
| 2026 | COMPLETED | 1000000 | 22900 | 0.36065497327690155 | 0.4687634112099268 |  |

Candidate [8, 4, 4, 8] / 129 parameters: fully completed=True; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | COMPLETED | 1000000 | 18000 | 0.2986740329527868 | 0.48138291323524324 |  |
| 42 | COMPLETED | 1000000 | 11550 | 0.33343737571397997 | 0.48246384451884994 |  |
| 123 | COMPLETED | 1000000 | 80425 | 0.3764305513319639 | 0.39198892393797086 |  |
| 999 | COMPLETED | 1000000 | 10525 | 0.2940622161147262 | 0.49211098786916685 |  |
| 2026 | COMPLETED | 1000000 | 9350 | 0.26338358179060495 | 0.49600066482853045 |  |

Candidate [8, 8, 4, 8] / 181 parameters: fully completed=True; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | COMPLETED | 1000000 | 94800 | 0.3540490052365065 | 0.4863116835758065 |  |
| 42 | COMPLETED | 1000000 | 3675 | 0.46936403779241537 | 0.5292584068084838 |  |
| 123 | COMPLETED | 1000000 | 10700 | 0.4673580993143123 | 0.4941581641431109 |  |
| 999 | COMPLETED | 1000000 | 35450 | 0.3642654816116755 | 0.445652823774186 |  |
| 2026 | COMPLETED | 1000000 | 645675 | 0.3182177653097129 | 0.4919556740805861 |  |

Candidate [8, 4, 8, 8] / 181 parameters: fully completed=True; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | COMPLETED | 1000000 | 17775 | 0.3350020697260805 | 0.48100355709211595 |  |
| 42 | COMPLETED | 1000000 | 5700 | 0.2325343034610923 | 0.47074553494708826 |  |
| 123 | COMPLETED | 1000000 | 13525 | 0.14575340406186227 | 0.15088058951137906 |  |
| 999 | COMPLETED | 1000000 | 22750 | 0.3081276876714257 | 0.4647361100585846 |  |
| 2026 | COMPLETED | 1000000 | 16900 | 0.2643579507805717 | 0.50075837051937 |  |

Candidate [8, 8] / 105 parameters: fully completed=False; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | COMPLETED | 1000000 | 1000000 | 0.22704534110236363 | 0.22704534110236363 |  |
| 42 | COMPLETED | 1000000 | 3775 | 0.14231530454829902 | 0.15040550282705412 |  |
| 123 | CANCELLED | 969930 | 8175 | 0.17716627155478992 | 0.2610057301308944 |  |
| 999 | CANCELLED | 695925 | 695925 | 0.11817746478298558 | 0.11817746478298558 |  |
| 2026 | CANCELLED | 382188 | 47550 | 0.3410441230294465 | 0.3735976790208687 |  |

Candidate [8, 8, 16, 4] / 313 parameters: fully completed=False; reliable=False; successes=0/5.

| Seed | State | Epochs | Best epoch | Best validation RMSE | Final validation RMSE | Failure |
|---:|---|---:|---:|---:|---:|---|
| 1 | COMPLETED | 1000000 | 29325 | 0.3272858618316648 | 0.4727412770913822 |  |
| 42 | COMPLETED | 1000000 | 10025 | 0.21011254233536514 | 0.21215070802185246 |  |
| 123 | COMPLETED | 1000000 | 27875 | 0.2101787956694461 | 0.45884167220785205 |  |
| 999 | COMPLETED | 1000000 | 19050 | 0.41285474185688203 | 0.47477080109431985 |  |
| 2026 | CANCELLED | 825320 | 24725 | 0.2869181363852901 | 0.47572338324054336 |  |

Unreported case: `SMALL/CPU/OPTIMIZED/workers=4/searchSeed=123` — not_observed_no_result.
