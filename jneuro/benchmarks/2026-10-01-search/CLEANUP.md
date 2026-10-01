# Task artifact cleanup

Removed **85,400,511 logical bytes** from 32 explicitly checked files. The [machine-readable record](cleanup.json) retains every removed path, size, available hash, replacement and verification record.

- Linux task scratch: **85,343,280 logical bytes**, **85,405,696 allocated bytes** removed beneath `/tmp/jneuro-search-evidence`. These were duplicate fixed-work/CI/smoke reports, superseded derived summaries, and byte-identical copies of committed final quality reports.
- Windows checkout: **57,231 logical bytes** of reproducible task-generated Python bytecode removed from the worktree's `jneuro/tools/__pycache__` and benchmark `__pycache__` paths. Source was retained.
- No Windows VHDX compaction or corresponding host-allocation reclaim is claimed. Linux deletions do not establish that Windows disk allocation shrank.

Throughput raw files were removed only after verifying that committed compact artifacts reconstruct every parsed round and parameter outcome against canonical hashes. Final raw quality reports are committed unchanged. Each cleanup used an exact path allowlist with ownership, regular-file, canonical-containment, unchanged-file and inactivity checks. Successful process sidecars and completed CI metadata establish that those writers exited.

Retained outputs include the compact numerical evidence and strict analyzers in this directory, current unit/coverage build reports (about 20 MiB), task logs and process provenance, unique GUI screenshots and report archives, and unresolved warmup/timing diagnostics. These support review and recovery. The existing Studio (PID 54812) and its primary checkout, installed runtimes, dependencies, source and Git worktrees were preserved. No unrelated cleanup or process termination was performed.

At this cleanup snapshot Linux had **806.50 GiB free** and Windows C: had **62.52 GiB free**. C: remained above the 90% usage guard, so no large local installations or bulk artifact generation were added; repository-wide and GUI checks used existing CI toolchains.
