# Global Codex Instructions

## Git workflow
- Never work directly on `main` or `master`.
- Before making code changes, create a new branch from the current default branch.
- Use a clear branch name that reflects the task.
- Keep all work isolated to that branch.
- Rebase the branch onto the latest default branch before pushing or opening a pull request.

## Required delivery flow
- After finishing the task, review the diff.
- Run relevant tests, checks, or validation steps when available.
- Commit changes with a clear, descriptive commit message.
- Push the branch to the remote.
- Create a pull request.

## Documentation maintenance
- Each experiment folder should keep its `BLOG.md` in sync with the experiment.
- Whenever an experiment, post, or behavior described by an experiment changes, update that folder's `BLOG.md` in the same change.
- If a change intentionally does not affect the related `BLOG.md`, call that out in the PR notes.
- Every experiment folder must have a `BLOG.md`; the Spring Boot blog app renders those Markdown files as the source of truth for posts.
- Experiment posts should be technical deep dives for experienced Java developers, not short summaries or changelog entries.
- Inspect the current experiment source before writing or updating a post, and describe the actual execution path, important implementation details, runtime behavior, caveats, and follow-up experiments.
- Do not optimize for a strict character or word target; prefer enough technical depth to explain the experiment accurately.
- Include many short fenced `java` snippets copied or lightly annotated from the current source, and comment on why each snippet matters; keep snippets compact enough to render cleanly in the blog app.
- Document unsafe, destructive, hardware-bound, network-bound, or environment-specific entrypoints explicitly so readers know what is safe to run.
- Keep Markdown rendering in mind: use stable headings, fenced code blocks, and plain Markdown that will render cleanly through the blogsite templates.
- After post updates, run `./gradlew --gradle-user-home /tmp/jexperiments-gradle-home --no-daemon :blogsite:build` and `./gradlew --gradle-user-home /tmp/jexperiments-gradle-home --no-daemon clean build` when available.

## Test coverage
- Every Gradle module in this repository must maintain at least 90% line coverage from automated tests.
- Coverage is enforced per module, never only as a repository-wide aggregate.
- New or changed production code must include tests that keep its module at or above the 90% line-coverage threshold.
- Do not lower, bypass, or broadly exclude production code from the threshold to make CI pass. Exclusions are allowed only for generated code, third-party/vendored code, and environment-bound entrypoints that cannot execute meaningfully in automated unit tests; keep exclusions narrow and documented in the build.
- Run the module's coverage verification task for focused changes and the repository-wide coverage verification as part of the full build.
- The standalone Android project under `tesseractviewer` follows the same 90% requirement for JVM-testable production logic.

## GUI test coverage
- Every GUI module must cover at least 90% of its documented user-visible UI paths with automated GUI tests, independently of the per-module 90% line-coverage requirement.
- Maintain a reviewed `GUI_PATHS.md` inventory with stable IDs, meaningful actions/state transitions and links to asserting test methods. Include navigation, valid and invalid input, empty/failure/completed states, training controls, cancellation, replay, resizing/scrolling, export and window shutdown.
- Enforce the path ratio in CI from the current run's successful GUI test results. Failed, skipped, aborted, missing, stale or non-GUI results must not count. Verify the verifier with negative tests. Do not remove paths, weaken assertions, retry away failures, or change the denominator to make a failing build pass.
- Exercise real windows and user input (for Swing, AWT Robot with Xvfb and a window manager on Linux CI). Invoke native input outside the EDT and read or change Swing fixtures on the EDT. Use bounded state-based waits, not fixed sleep durations as the synchronization strategy.
- GUI changes must update regression tests and the inventory in the same PR. Screenshots alone and direct model calls are not GUI-path coverage.
- Publish GUI reports, path-coverage results and diagnostic screenshots. Keep UI line/branch reports separate from scenario coverage; 90% line coverage does not mean 90% of UI paths.
- For JNeuro run `:jneuro:guiCheck` in a graphical session (Linux CI: `xvfb-run`), in addition to `:jneuro:check`. A missing display must fail, not silently skip the GUI suite.

## Safety rules
- Do not merge pull requests.
- Do not delete branches unless explicitly asked.
- Do not bypass failing checks unless explicitly asked.
- Do not make unrelated cleanup changes outside the task scope.

## Working style
- Prefer minimal, targeted changes over broad rewrites.
- Follow existing project patterns and conventions.
- Avoid introducing new dependencies unless necessary.
- Call out uncertainty instead of guessing.
- If the repository is missing a remote or PR creation is not possible, state that explicitly.
