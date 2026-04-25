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
