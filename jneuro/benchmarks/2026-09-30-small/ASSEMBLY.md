# Extended C2 instruction evidence

All eight instruction-only JVM cases exited successfully. The 142 captured C2 compilations have complete byte ranges. The comparison selects the latest normal compilation for each method, keeping OSR and superseded compilations separate.

| Precision | Requested/observed width | Layout | Forward, backward, gradients, batch update, online update | FAST polynomial |
| --- | --- | --- | --- | --- |
| FP64 | 128 | Single | Packed XMM FMA in all five methods | Packed `vmulpd` / `vaddpd` |
| FP64 | 256 | Single | Packed YMM FMA in all five methods | Packed `vmulpd` / `vaddpd` |
| FP32 | 128 | Single | Packed XMM FMA in all five methods | Packed `vmulps` / `vaddps` |
| FP32 | 256 | Single | Packed YMM FMA in all five methods | Packed `vmulps` / `vaddps` |
| FP64 | 128 | Cohort | Packed XMM FMA in all five methods | Packed `vmulpd` / `vaddpd` |
| FP64 | 256 | Cohort | Packed YMM FMA in all five methods | Packed `vmulpd` / `vaddpd` |
| FP32 | 128 | Cohort | Packed XMM FMA in all five methods | Packed `vmulps` / `vaddps` |
| FP32 | 256 | Cohort | Packed YMM FMA in all five methods | Packed `vmulps` / `vaddps` |

“Batch update” above refers to the array-update `update` method. No standalone C2 `updateBatch` wrapper was captured. Its vector update callees are present; the missing thin wrapper is consistent with inlining, but this capture is not an inlining trace.

A genuine limitation remains in the FP64 128-bit cohort path: `update` and `updateOnline` contain inline TLAB allocation and static helper calls, alongside their packed FMA. Each has 14 full-vector stack stores on paths leading to non-deoptimization calls. The retained local `fp64-128-mask-path.txt` lists exact compiled versions and addresses. Vector/mask helper boxing is the likely cause; exact callee names were discarded by the bounded capture filter, so that attribution is an inference. A direct comparison confirms that each FP64-128 method has four inline TLAB top stores, seven static helper calls, two new-array and two new-instance allocator slow paths, and zero native masked-store instructions. Default FP64-256 has no TLAB top stores, static helper calls or allocator slow paths and instead emits `vmaskmovpd` stores. FP32 at both widths similarly emits `vmaskmovps` without these allocation paths. See `cohort-mask-comparison.json`. The TLAB offset pattern is specific to this captured VM, not a portable allocation detector. This finding does not mean the arithmetic became entirely scalar.

The decoder distinguishes full-vector stack stores from scalar `vmovsd` / `vmovss` stores. Many full-vector stores lead directly, without an intervening conditional branch, to annotated `UncommonTrapBlob` exits and preserve deoptimization state. Remaining stores are either associated with another call or left unclassified. Static stack-store counts are not measured hot-loop spill counts; do not sum instructions across multiple compiled versions.

Scalar output dots and FAST range reduction/exponent scaling remain scalar by design. Packed polynomial arithmetic proves only that part of FAST activation uses SIMD. The diagnostic exercises the 2,8,8,8,1 shape, both online and mini-batch work, and nine-model cohorts with inactive and padded tail lanes; it does not independently certify every family shape or numerical parity. Those are covered by the separate parity tests.

These runs provide instruction evidence only. They contain no valid throughput or speedup measurement. `assembly.json` has compact per-case/method findings. The retained local `decoded-all-compilations.json` preserves every decoded compilation and `provenance.json` records source/class hashes, commands, exit statuses and the capture cap.
