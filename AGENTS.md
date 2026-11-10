# Agent instructions

These apply to any coding agent working in this repo.

- Repo map, conventions and commands: `CLAUDE.md`. Everything in it applies to you too.
- Active work: `out/burden_feasibility_20261102/HANDOFF.md`. Read it before touching that folder. It holds the goal,
  how the user runs experiments, the findings so far and the next steps.
- Working with the user:
  - Commit or push only when asked. Normally you give the command and the user runs it.
  - Name files in `git add`. The tree holds unrelated uncommitted work.
  - Don't ssh or connect to remote machines without asking.
  - Methods must be generic (lesion-agnostic), read every factor through one shared route, and work on real data
    whose subject count is fixed.
  - Define terms at first use, and keep answers short and plain.
