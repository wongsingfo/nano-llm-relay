#!/usr/bin/env python3
"""Coordinate tasks.csv plus task markdown files for shared task execution."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

HEADERS = [
    "id",
    "task_name",
    "status",
    "parent_id",
    "created_at",
    "updated_at",
    "result",
]
VALID_STATUSES = {"pending", "ongoing", "completed", "failed"}
TASKS_CSV_NAME = "tasks.csv"
LOCK_FILE_NAME = ".tasks.lock"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
TOP_LEVEL_ID_RE = re.compile(r"^T(\d{3})$")
TASK_ID_RE = re.compile(r"^T\d{3}(?:-\d+)*$")
DESCRIPTION_HEADING = "## Description"
DETAILS_HEADING = "## Details"


def now_string() -> str:
    return datetime.now().strftime(TIMESTAMP_FORMAT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coordinate tasks.csv and task markdown files for resumable multi-agent work."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create tasks.csv and tasks/ if missing.")
    add_root_argument(init_parser)

    list_parser = subparsers.add_parser("list", help="List tasks from tasks.csv.")
    add_root_argument(list_parser)
    list_parser.add_argument(
        "--status",
        action="append",
        choices=sorted(VALID_STATUSES),
        help="Filter by task status. Repeat for multiple statuses.",
    )
    list_parser.add_argument("--parent-id", help="Filter by parent task ID.")
    list_parser.add_argument(
        "--claimable",
        action="store_true",
        help="Show only pending tasks that have no children and can be claimed.",
    )

    show_parser = subparsers.add_parser(
        "show",
        help="Show one task, including task description and result memory.",
    )
    add_root_argument(show_parser)
    show_parser.add_argument("--id", dest="task_id", required=True, help="Existing task ID.")

    claim_parser = subparsers.add_parser(
        "claim",
        help="Atomically claim one pending leaf task by setting status to ongoing.",
    )
    add_root_argument(claim_parser)
    claim_parser.add_argument("--id", dest="task_id", help="Claim a specific task ID.")
    claim_parser.add_argument("--parent-id", help="Only auto-claim children of this parent ID.")
    claim_parser.add_argument(
        "--result",
        help="Optional short result summary to store at claim time.",
    )

    create_parser = subparsers.add_parser(
        "create",
        help="Create a task row and the corresponding task markdown file.",
    )
    add_root_argument(create_parser)
    create_parser.add_argument("--id", dest="task_id", help="Explicit task ID. Auto-generate when omitted.")
    create_parser.add_argument("--parent-id", help="Parent task ID. Leave empty for a top-level task.")
    create_parser.add_argument("--task-name", required=True, help="Short task summary for tasks.csv.")
    create_parser.add_argument("--status", default="pending", choices=sorted(VALID_STATUSES))
    create_parser.add_argument("--result", default="", help="Short result summary for tasks.csv.")
    add_description_arguments(create_parser)

    update_parser = subparsers.add_parser(
        "update",
        help="Update tasks.csv and tasks/{id}.md through the script.",
    )
    add_root_argument(update_parser)
    update_parser.add_argument("--id", dest="task_id", required=True, help="Existing task ID.")
    update_parser.add_argument("--task-name", help="Replace the task_name field.")
    update_parser.add_argument("--status", choices=sorted(VALID_STATUSES), help="Replace the status field.")
    update_parser.add_argument("--result", help="Replace the short result summary.")
    update_parser.add_argument(
        "--append-description",
        action="store_true",
        help="Append to tasks/{id}.md instead of replacing the description body.",
    )
    add_description_arguments(update_parser)

    return parser.parse_args()


def add_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        default=".",
        help="Workspace root that should contain tasks.csv and tasks/. Defaults to the current directory.",
    )


def add_description_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--description",
        dest="description",
        help="Detailed task description for tasks/{id}.md.",
    )
    parser.add_argument(
        "--description-file",
        dest="description_file",
        help="Read the task description from a UTF-8 file.",
    )

def main() -> int:
    args = parse_args()
    try:
        if args.command == "init":
            return init_tracker(args)
        if args.command == "list":
            return list_tasks(args)
        if args.command == "show":
            return show_task(args)
        if args.command == "claim":
            return claim_task(args)
        if args.command == "create":
            return create_task(args)
        if args.command == "update":
            return update_task(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 1


def init_tracker(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    workspace = prepare_workspace(root)
    with workspace_lock(workspace["lock_path"]):
        created = False
        if not workspace["csv_path"].exists():
            write_rows(workspace["csv_path"], [])
            created = True
        else:
            read_rows(workspace["csv_path"])
    emit_json(
        {
            "root": str(root),
            "tasks_csv": str(workspace["csv_path"]),
            "tasks_dir": str(workspace["tasks_dir"]),
            "created": created,
        }
    )
    return 0


def list_tasks(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    workspace = prepare_workspace(root)
    with workspace_lock(workspace["lock_path"]):
        rows = load_rows(workspace["csv_path"])
        filtered = filter_rows(
            rows,
            statuses=args.status,
            parent_id=normalize_optional_text(args.parent_id),
            claimable_only=args.claimable,
        )
        tasks = [
            build_task_view(row, rows, workspace["tasks_dir"], include_contents=False)
            for row in sort_rows(filtered)
        ]
    emit_json({"tasks": tasks})
    return 0


def show_task(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    workspace = prepare_workspace(root)
    with workspace_lock(workspace["lock_path"]):
        rows = load_rows(workspace["csv_path"])
        row = ensure_task_exists(rows, required_text(args.task_id, "task ID"))
        task = build_task_view(row, rows, workspace["tasks_dir"], include_contents=True)
    emit_json(task)
    return 0


def claim_task(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    workspace = prepare_workspace(root)
    with workspace_lock(workspace["lock_path"]):
        rows = load_rows(workspace["csv_path"])
        row = select_claimable_task(
            rows,
            task_id=normalize_optional_text(args.task_id),
            parent_id=normalize_optional_text(args.parent_id),
        )
        if row is None:
            emit_json({"claimed": False, "task": None})
            return 0

        timestamp = now_string()
        row["status"] = "ongoing"
        row["updated_at"] = timestamp
        if args.result is not None:
            row["result"] = args.result

        description = read_task_description(task_path(workspace["tasks_dir"], row["id"]))
        sync_task_file(
            task_path(workspace["tasks_dir"], row["id"]),
            row,
            description=description,
        )

        write_rows(workspace["csv_path"], rows)
        task = build_task_view(row, rows, workspace["tasks_dir"], include_contents=True)
    emit_json({"claimed": True, "task": task})
    return 0


def create_task(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    workspace = prepare_workspace(root)
    with workspace_lock(workspace["lock_path"]):
        rows = load_rows(workspace["csv_path"])
        description = resolve_text(args.description, args.description_file, "description")

        parent_id = normalize_optional_text(args.parent_id)
        explicit_id = normalize_optional_text(args.task_id)

        if explicit_id:
            validate_task_id(explicit_id)
            if parent_id:
                validate_parent_child_relation(explicit_id, parent_id)
            else:
                parent_id = infer_parent_id(explicit_id)
            if parent_id:
                ensure_task_exists(rows, parent_id)
        elif parent_id:
            ensure_task_exists(rows, parent_id)

        task_id = explicit_id or generate_next_task_id(rows, parent_id)
        if find_row(rows, task_id):
            raise ValueError(f"Task ID already exists: {task_id}")

        timestamp = now_string()
        row = {
            "id": task_id,
            "task_name": required_text(args.task_name, "task name"),
            "status": args.status,
            "parent_id": parent_id or "",
            "created_at": timestamp,
            "updated_at": timestamp,
            "result": args.result or "",
        }
        rows.append(row)
        write_rows(workspace["csv_path"], rows)

        sync_task_file(
            task_path(workspace["tasks_dir"], task_id),
            row,
            description=description,
        )

        task = build_task_view(row, rows, workspace["tasks_dir"], include_contents=True)
    emit_json(task)
    return 0


def update_task(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    workspace = prepare_workspace(root)
    with workspace_lock(workspace["lock_path"]):
        rows = load_rows(workspace["csv_path"])
        row = ensure_task_exists(rows, required_text(args.task_id, "task ID"))
        description = resolve_text(args.description, args.description_file, "description")

        if args.append_description and description is None:
            raise ValueError("--append-description requires --description or --description-file.")

        changed = False
        if args.task_name is not None:
            row["task_name"] = required_text(args.task_name, "task name")
            changed = True
        if args.status is not None:
            row["status"] = args.status
            changed = True
        if args.result is not None:
            row["result"] = args.result
            changed = True
        if description is not None:
            changed = True

        if not changed:
            raise ValueError("Nothing to update. Provide task fields or description text.")

        row["updated_at"] = now_string()
        sync_task_file(
            task_path(workspace["tasks_dir"], row["id"]),
            row,
            description=description,
            append_description=args.append_description,
        )

        write_rows(workspace["csv_path"], rows)
        task = build_task_view(row, rows, workspace["tasks_dir"], include_contents=True)
    emit_json(task)
    return 0


def prepare_workspace(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    tasks_dir = root / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "tasks_dir": tasks_dir,
        "csv_path": root / TASKS_CSV_NAME,
        "lock_path": root / LOCK_FILE_NAME,
    }


@contextmanager
def workspace_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise ValueError(f"{csv_path} not found. Run the init command first.")
    return read_rows(csv_path)


def filter_rows(
    rows: list[dict[str, str]],
    *,
    statuses: list[str] | None,
    parent_id: str | None,
    claimable_only: bool,
) -> list[dict[str, str]]:
    filtered = list(rows)
    if statuses:
        status_set = set(statuses)
        filtered = [row for row in filtered if row["status"] in status_set]
    if parent_id is not None:
        filtered = [row for row in filtered if row["parent_id"] == parent_id]
    if claimable_only:
        filtered = [row for row in filtered if is_claimable(row, rows)]
    return filtered


def sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: (row["created_at"], row["id"]))


def select_claimable_task(
    rows: list[dict[str, str]],
    *,
    task_id: str | None,
    parent_id: str | None,
) -> dict[str, str] | None:
    if parent_id is not None:
        validate_task_id(parent_id)

    if task_id is not None:
        row = ensure_task_exists(rows, task_id)
        if parent_id is not None and row["parent_id"] != parent_id:
            raise ValueError(f"Task {task_id} does not belong to parent {parent_id}.")
        if not is_claimable(row, rows):
            raise ValueError(
                f"Task {task_id} is not claimable. It must be pending and have no children."
            )
        return row

    candidates = [row for row in rows if is_claimable(row, rows)]
    if parent_id is not None:
        candidates = [row for row in candidates if row["parent_id"] == parent_id]
    if not candidates:
        return None
    return sort_rows(candidates)[0]


def is_claimable(row: dict[str, str], rows: list[dict[str, str]]) -> bool:
    return row["status"] == "pending" and not has_children(rows, row["id"])


def has_children(rows: list[dict[str, str]], task_id: str) -> bool:
    return any(row["parent_id"] == task_id for row in rows)


def build_task_view(
    row: dict[str, str],
    rows: list[dict[str, str]],
    tasks_dir: Path,
    *,
    include_contents: bool,
) -> dict[str, object]:
    task_file = task_path(tasks_dir, row["id"])
    memory_file = result_path(tasks_dir, row["id"])
    view: dict[str, object] = {
        key: row[key] for key in HEADERS
    }
    view.update(
        {
            "has_children": has_children(rows, row["id"]),
            "claimable": is_claimable(row, rows),
            "task_path": str(task_file),
            "result_path": str(memory_file),
            "task_exists": task_file.exists(),
            "result_exists": memory_file.exists(),
        }
    )
    if include_contents:
        view["description"] = read_task_description(task_file)
        view["memory"] = read_result_details(memory_file)
    return view


def task_path(tasks_dir: Path, task_id: str) -> Path:
    return tasks_dir / f"{task_id}.md"


def result_path(tasks_dir: Path, task_id: str) -> Path:
    return tasks_dir / f"{task_id}.result.md"


def sync_task_file(
    file_path: Path,
    row: dict[str, str],
    *,
    description: str | None,
    append_description: bool = False,
) -> None:
    existing = read_task_description(file_path)
    if description is None:
        body = existing or default_description()
    elif append_description and existing:
        body = f"{existing.rstrip()}\n\n{description.strip()}"
    else:
        body = description.strip()
    file_path.write_text(render_task_markdown(row, body), encoding="utf-8")


def read_task_description(file_path: Path) -> str | None:
    return read_markdown_section(file_path, DESCRIPTION_HEADING)


def read_result_details(file_path: Path) -> str | None:
    return read_markdown_section(file_path, DETAILS_HEADING)


def read_markdown_section(file_path: Path, heading: str) -> str | None:
    if not file_path.exists():
        return None
    content = file_path.read_text(encoding="utf-8").strip()
    if not content:
        return None
    marker = f"{heading}\n\n"
    index = content.find(marker)
    if index == -1:
        return content
    section = content[index + len(marker) :].strip()
    return section or None


def render_task_markdown(row: dict[str, str], description: str) -> str:
    parent = row["parent_id"] or "(top-level)"
    return (
        f"# {row['id']} {row['task_name']}\n\n"
        f"- Status: {row['status']}\n"
        f"- Parent ID: {parent}\n"
        f"- Created At: {row['created_at']}\n"
        f"- Updated At: {row['updated_at']}\n\n"
        f"{DESCRIPTION_HEADING}\n\n"
        f"{description.strip()}\n"
    )


def default_description() -> str:
    return "Fill in the detailed task description, constraints, and acceptance criteria."


def normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def required_text(value: str | None, label: str) -> str:
    if value is None:
        raise ValueError(f"Missing {label}.")
    text = value.strip()
    if not text:
        raise ValueError(f"{label.capitalize()} cannot be empty.")
    return text


def resolve_text(inline_value: str | None, file_path: str | None, label: str) -> str | None:
    inline = normalize_optional_text(inline_value)
    source_file = normalize_optional_text(file_path)
    if inline is not None and source_file is not None:
        raise ValueError(f"Provide {label} via either inline text or a file, not both.")
    if source_file is not None:
        return Path(source_file).read_text(encoding="utf-8").strip() or None
    return inline


def validate_task_id(task_id: str) -> None:
    if not TASK_ID_RE.match(task_id):
        raise ValueError(
            "Invalid task ID. Use T001 for top-level tasks and T001-1 / T001-1-1 for descendants."
        )


def infer_parent_id(task_id: str) -> str:
    parts = task_id.split("-")
    if len(parts) == 1:
        return ""
    return "-".join(parts[:-1])


def validate_parent_child_relation(task_id: str, parent_id: str) -> None:
    validate_task_id(parent_id)
    inferred_parent = infer_parent_id(task_id)
    if inferred_parent != parent_id:
        raise ValueError(f"Task ID {task_id} does not belong under parent {parent_id}.")


def generate_next_task_id(rows: list[dict[str, str]], parent_id: str | None) -> str:
    if not parent_id:
        highest = 0
        for row in rows:
            match = TOP_LEVEL_ID_RE.match(row["id"])
            if match:
                highest = max(highest, int(match.group(1)))
        return f"T{highest + 1:03d}"

    ensure_task_exists(rows, parent_id)
    parent_depth = parent_id.count("-")
    highest = 0
    for row in rows:
        candidate = row["id"]
        if not candidate.startswith(f"{parent_id}-"):
            continue
        if candidate.count("-") != parent_depth + 1:
            continue
        suffix = candidate.split("-")[-1]
        highest = max(highest, int(suffix))
    return f"{parent_id}-{highest + 1}"


def find_row(rows: list[dict[str, str]], task_id: str) -> dict[str, str] | None:
    for row in rows:
        if row["id"] == task_id:
            return row
    return None


def ensure_task_exists(rows: list[dict[str, str]], task_id: str) -> dict[str, str]:
    validate_task_id(task_id)
    row = find_row(rows, task_id)
    if row is None:
        raise ValueError(f"Unknown task ID: {task_id}")
    return row


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != HEADERS:
            raise ValueError(f"{csv_path} must use header: {','.join(HEADERS)}")
        return [{key: row.get(key, "") for key in HEADERS} for row in reader]


def write_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=csv_path.parent,
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in HEADERS})
        temp_path = Path(handle.name)
    temp_path.replace(csv_path)


def emit_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
