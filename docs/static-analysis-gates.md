# Shared static-analysis outcomes

The reusable workflow reports its correctness/security results in the
`static-analysis-outcomes` artifact and job summary. The JSON `status` reflects
actual analyzer step outcomes, including failures hidden by advisory execution.
Missing outputs, failed setup/detection, and unexpectedly skipped jobs cannot
produce a passing enforced result.

Callers retain advisory mode until their own baseline is verified. After the
baseline passes, pin the reviewed reusable-workflow commit and pass
`enforce_correctness: true` alongside existing inputs. The named correctness
job then fails on workflow syntax/security, Python parse/high-confidence
errors, high-confidence Bandit findings, or XML/JSON/Excel integrity errors.
Style/type/portability analyzers and CodeQL keep separate job/report status;
the correctness job does not claim those checks passed.

Run `python3 -m unittest discover -s .github/tests -v` to execute the actual
embedded outcome reporter against passing, failing, skipped, missing and
cancelled job results. The dedicated contracts workflow runs these tests.
Changing advisory callers requires their current-head analyzer baseline;
existing queued runs and missing account capacity cannot establish it.
