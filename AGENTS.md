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
