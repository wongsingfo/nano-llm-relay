---
name: task-board
description: Coordinate resumable multi-agent work through tasks.csv plus tasks/{id}.md and direct-write tasks/{id}.result.md. Use for multi-step, long-running, claimable work with durable handoff memory; skip task creation for simple work.
---

# Task Board

## Overview

Use this skill for multi-step or resumable work that benefits from explicit task claiming. Coordinate `tasks.csv` through scripts/task_tracker.py, and write durable notes directly to `tasks/{id}.result.md`. Keep task creation minimal.

## Default Loop

1. Run `init` if `tasks.csv` or `tasks/` is missing.
2. Prefer existing work: use `claim` for a pending leaf task, and use `list` or `show` when you need to inspect the board first.
3. Do the work. Keep the CSV `result` short and put findings, blockers, and handoff notes in `tasks/{id}.result.md`.
4. Use `update` for status, short result, and task description changes. If a task goes stale, the user may reset `ongoing` back to `pending`.

## Rules

- `tasks.csv` must be UTF-8 with header `id,task_name,status,parent_id,created_at,updated_at,result`.
- `tasks/{id}.md` (optionally) stores task description and acceptance criteria. `tasks/{id}.result.md` stores durable memory.
- Do not edit `tasks.csv` manually. Use the script for `init`, `list`, `claim`, `create`, and `update`.
- Write `tasks/{id}.result.md` directly. Do not route result details through the script.
- ID shape: top level `T001`, child `T001-1`. Do not create deeper child like `T001-1-1`.
- Statuses: `pending`, `ongoing`, `completed`, `failed`.
- Claimable means `pending` with no children.
- Prefer claiming unfinished tasks over creating duplicates.
- Create child tasks only when needed to expose the next actionable leaf.
- Read `tasks/{id}.result.md` before resuming. Append useful memory before yielding, failing, or completing.
- Do not start a task already in `ongoing` unless the user reset it or you are intentionally taking it over.

## Commands

```bash
python /path/to/task-board/scripts/progress_tracker.py init --root "$PWD"
python /path/to/task-board/scripts/progress_tracker.py claim --root "$PWD"
python /path/to/task-board/scripts/progress_tracker.py create --root "$PWD" --parent-id T001 --task-name "Map protocol matrix" --description "List request, auth, and streaming differences."
python /path/to/task-board/scripts/progress_tracker.py update --root "$PWD" --id T001-1 --status completed --result "Mapped protocol matrix"
```
