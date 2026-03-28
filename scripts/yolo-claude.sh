#!/usr/bin/env bash

set -euo pipefail

has_arg() {
    local needle=$1
    shift

    local arg
    for arg in "$@"; do
        if [[ "$arg" == "$needle" || "$arg" == "$needle="* ]]; then
            return 0
        fi
    done

    return 1
}

is_known_subcommand() {
    case "$1" in
        agents|auth|auto-mode|doctor|install|mcp|plugin|plugins|setup-token|update|upgrade)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

append_env_if_set() {
    local -n target=$1
    local name=$2

    if [[ -n "${!name-}" ]]; then
        target+=(--setenv "$name" "${!name}")
    fi
}

main() {
    if ! command -v claude >/dev/null 2>&1; then
        echo "claude is not installed or not on PATH" >&2
        exit 1
    fi

    if ! command -v bwrap >/dev/null 2>&1; then
        echo "bwrap is not installed or not on PATH" >&2
        exit 1
    fi

    local claude_path claude_root workdir
    claude_path=$(command -v claude)
    claude_root=$(dirname "$(dirname "$claude_path")")
    workdir=$(pwd)

    : "${HOME:?HOME must be set}"
    : "${YOLO_CLAUDE_BASE_URL:=http://127.0.0.1:4000}"
    : "${YOLO_CLAUDE_MODEL:=gpt-5.4}"
    : "${YOLO_CLAUDE_API_KEY:=test-key}"

    local anthropic_base_url anthropic_model anthropic_api_key
    anthropic_base_url=$YOLO_CLAUDE_BASE_URL
    anthropic_model=$YOLO_CLAUDE_MODEL
    anthropic_api_key=$YOLO_CLAUDE_API_KEY

    mkdir -p "$HOME/.claude" "$HOME/.cache"
    if [[ ! -e "$HOME/.claude.json" ]]; then
        touch "$HOME/.claude.json"
    fi

    local -a bwrap_args=(
        --ro-bind /usr /usr
        --ro-bind /lib /lib
        --ro-bind /lib64 /lib64
        --ro-bind /bin /bin
        --ro-bind /etc /etc
        --ro-bind "$claude_root" "$claude_root"
        --bind "$HOME/.claude" "$HOME/.claude"
        --bind "$HOME/.claude.json" "$HOME/.claude.json"
        --bind "$HOME/.cache" "$HOME/.cache"
        --bind "$workdir" "$workdir"
        --proc /proc
        --dev /dev
        --tmpfs /tmp
        --unshare-all
        --share-net
        --die-with-parent
        --chdir "$workdir"
        --setenv HOME "$HOME"
        --setenv PATH "${PATH:-/usr/bin:/bin}"
        --setenv ANTHROPIC_API_KEY "$anthropic_api_key"
        --setenv ANTHROPIC_BASE_URL "$anthropic_base_url"
        --setenv ANTHROPIC_MODEL "$anthropic_model"
    )

    append_env_if_set bwrap_args TERM
    append_env_if_set bwrap_args COLORTERM
    append_env_if_set bwrap_args LANG
    append_env_if_set bwrap_args LC_ALL
    append_env_if_set bwrap_args NO_COLOR

    local -a claude_args=("$@")
    if [[ ${#claude_args[@]} -gt 0 ]] \
        && ! has_arg "-p" "${claude_args[@]}" \
        && ! has_arg "--print" "${claude_args[@]}" \
        && ! has_arg "-c" "${claude_args[@]}" \
        && ! has_arg "--continue" "${claude_args[@]}" \
        && ! has_arg "-r" "${claude_args[@]}" \
        && ! has_arg "--resume" "${claude_args[@]}" \
        && ! has_arg "--help" "${claude_args[@]}" \
        && ! has_arg "-h" "${claude_args[@]}" \
        && ! has_arg "--version" "${claude_args[@]}" \
        && ! has_arg "-v" "${claude_args[@]}" \
        && ! is_known_subcommand "${claude_args[0]}"; then
        claude_args=(--bare --tools "" -p --no-session-persistence "${claude_args[@]}")
    fi

    if ! has_arg "--dangerously-skip-permissions" "${claude_args[@]}"; then
        claude_args+=(--dangerously-skip-permissions)
    fi
    if ! has_arg "--model" "${claude_args[@]}"; then
        claude_args+=(--model "$anthropic_model")
    fi

    echo "Claude root: $claude_root"
    echo "Using relay: $anthropic_base_url"
    echo "Using model: $anthropic_model"

    exec bwrap "${bwrap_args[@]}" "$claude_path" "${claude_args[@]}"
}

main "$@"
