from __future__ import annotations

from collections.abc import AsyncIterator

import httpx


def encode_sse(data: str, event: str | None = None) -> bytes:
    lines: list[str] = []
    if event:
        lines.append(f"event: {event}")
    payload_lines = data.splitlines() or [""]
    for line in payload_lines:
        lines.append(f"data: {line}")
    return ("\n".join(lines) + "\n\n").encode("utf-8")


async def iter_sse_json_lines(response: httpx.Response) -> AsyncIterator[str]:
    """Yield raw JSON strings from SSE data fields, skipping [DONE] sentinels.

    Used to bridge SSE passthrough into WebSocket JSON frames.
    """
    async for event in iter_sse_events(response):
        data = event.get("data")
        if data is None or data == "[DONE]":
            continue
        yield data


async def iter_sse_events(response: httpx.Response) -> AsyncIterator[dict[str, str | None]]:
    event_name: str | None = None
    data_lines: list[str] = []

    async for line in response.aiter_lines():
        if line == "":
            if event_name is not None or data_lines:
                yield {"event": event_name, "data": "\n".join(data_lines)}
            event_name = None
            data_lines = []
            continue

        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip() or None
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if event_name is not None or data_lines:
        yield {"event": event_name, "data": "\n".join(data_lines)}
