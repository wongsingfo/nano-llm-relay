from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "yolo-generic.sh"
BWRAP_AVAILABLE = shutil.which("bwrap") is not None


def make_executable(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def parse_kv_output(stdout: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in stdout.strip().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = value
    return result


def run_yolo_generic(command: list[str], env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT_PATH), *command],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.skipif(not BWRAP_AVAILABLE, reason="bwrap is required for script integration tests")
def test_yolo_generic_preserves_allowlisted_env_and_writable_workdir(tmp_path: Path):
    home_dir = tmp_path / "home"
    workdir = tmp_path / "work"
    bin_dir = tmp_path / "bin"
    cache_dir = home_dir / "cache"
    config_dir = home_dir / "config"
    home_dir.mkdir()
    workdir.mkdir()
    bin_dir.mkdir()
    cache_dir.mkdir()
    config_dir.mkdir()

    make_executable(
        bin_dir / "toolcmd",
        """#!/usr/bin/env bash
set -euo pipefail
touch "$PWD/.sandbox-write"
printf 'argc=%s\n' "$#"
printf 'arg1=%s\n' "${1-}"
printf 'arg2=%s\n' "${2-}"
printf 'cwd=%s\n' "$PWD"
printf 'openai=%s\n' "${OPENAI_API_KEY-unset}"
printf 'anthropic=%s\n' "${ANTHROPIC_API_KEY-unset}"
printf 'codex_home=%s\n' "${CODEX_HOME-unset}"
printf 'xdg_config=%s\n' "${XDG_CONFIG_HOME-unset}"
printf 'xdg_cache=%s\n' "${XDG_CACHE_HOME-unset}"
printf 'unrelated=%s\n' "${UNRELATED_VAR-unset}"
printf 'write_ok=%s\n' "$(test -f "$PWD/.sandbox-write" && echo yes || echo no)"
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home_dir),
            "PATH": f"{bin_dir}:{env['PATH']}",
            "OPENAI_API_KEY": "openai-test-key",
            "ANTHROPIC_API_KEY": "anthropic-test-key",
            "CODEX_HOME": str(home_dir / ".codex-home"),
            "XDG_CONFIG_HOME": str(config_dir),
            "XDG_CACHE_HOME": str(cache_dir),
            "UNRELATED_VAR": "drop-me",
            "TERM": "xterm-256color",
        }
    )

    result = run_yolo_generic(["toolcmd", "alpha", "two words"], env=env, cwd=workdir)

    assert result.returncode == 0, result.stderr
    data = parse_kv_output(result.stdout)
    assert data == {
        "argc": "2",
        "arg1": "alpha",
        "arg2": "two words",
        "cwd": str(workdir),
        "openai": "openai-test-key",
        "anthropic": "anthropic-test-key",
        "codex_home": str(home_dir / ".codex-home"),
        "xdg_config": str(config_dir),
        "xdg_cache": str(cache_dir),
        "unrelated": "unset",
        "write_ok": "yes",
    }


@pytest.mark.skipif(not BWRAP_AVAILABLE, reason="bwrap is required for script integration tests")
def test_yolo_generic_mounts_claude_state(tmp_path: Path):
    home_dir = tmp_path / "home"
    workdir = tmp_path / "work"
    bin_dir = tmp_path / "bin"
    home_dir.mkdir()
    workdir.mkdir()
    bin_dir.mkdir()
    (home_dir / ".cache").mkdir()
    (home_dir / ".claude").mkdir()
    (home_dir / ".claude.json").write_text("{}", encoding="utf-8")

    make_executable(
        bin_dir / "claude",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'claude_dir=%s\n' "$(test -d "$HOME/.claude" && echo yes || echo no)"
printf 'claude_json=%s\n' "$(test -f "$HOME/.claude.json" && echo yes || echo no)"
printf 'cache_dir=%s\n' "$(test -d "$HOME/.cache" && echo yes || echo no)"
""",
    )

    env = os.environ.copy()
    env.update({"HOME": str(home_dir), "PATH": f"{bin_dir}:{env['PATH']}"})

    result = run_yolo_generic(["claude", "--help"], env=env, cwd=workdir)

    assert result.returncode == 0, result.stderr
    data = parse_kv_output(result.stdout)
    assert data == {"claude_dir": "yes", "claude_json": "yes", "cache_dir": "yes"}


@pytest.mark.skipif(not BWRAP_AVAILABLE, reason="bwrap is required for script integration tests")
def test_yolo_generic_mounts_codex_state(tmp_path: Path):
    home_dir = tmp_path / "home"
    workdir = tmp_path / "work"
    bin_dir = tmp_path / "bin"
    cache_dir = home_dir / "cache-root"
    config_dir = home_dir / "config-root"
    home_dir.mkdir()
    workdir.mkdir()
    bin_dir.mkdir()
    cache_dir.mkdir()
    (home_dir / ".codex").mkdir()
    (config_dir / "codex").mkdir(parents=True)

    make_executable(
        bin_dir / "codex",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'codex_dir=%s\n' "$(test -d "$HOME/.codex" && echo yes || echo no)"
printf 'codex_config=%s\n' "$(test -d "$XDG_CONFIG_HOME/codex" && echo yes || echo no)"
printf 'cache_dir=%s\n' "$(test -d "$XDG_CACHE_HOME" && echo yes || echo no)"
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home_dir),
            "PATH": f"{bin_dir}:{env['PATH']}",
            "XDG_CONFIG_HOME": str(config_dir),
            "XDG_CACHE_HOME": str(cache_dir),
        }
    )

    result = run_yolo_generic(["codex", "--help"], env=env, cwd=workdir)

    assert result.returncode == 0, result.stderr
    data = parse_kv_output(result.stdout)
    assert data == {"codex_dir": "yes", "codex_config": "yes", "cache_dir": "yes"}


@pytest.mark.skipif(not BWRAP_AVAILABLE, reason="bwrap is required for script integration tests")
def test_yolo_generic_rejects_missing_or_shell_string_command(tmp_path: Path):
    home_dir = tmp_path / "home"
    workdir = tmp_path / "work"
    home_dir.mkdir()
    workdir.mkdir()

    env = os.environ.copy()
    env.update({"HOME": str(home_dir)})

    missing = run_yolo_generic(["does-not-exist"], env=env, cwd=workdir)
    assert missing.returncode != 0
    assert "command not found: does-not-exist" in missing.stderr

    shell_string = run_yolo_generic(["claude hello"], env=env, cwd=workdir)
    assert shell_string.returncode != 0
    assert "command not found: claude hello" in shell_string.stderr


@pytest.mark.skipif(not BWRAP_AVAILABLE, reason="bwrap is required for script integration tests")
def test_yolo_generic_exposes_resolver_config(tmp_path: Path):
    home_dir = tmp_path / "home"
    workdir = tmp_path / "work"
    home_dir.mkdir()
    workdir.mkdir()

    env = os.environ.copy()
    env.update({"HOME": str(home_dir)})

    result = run_yolo_generic(["cat", "/etc/resolv.conf"], env=env, cwd=workdir)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
