from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import httpx

from .models import (
    MessageBlock,
    NormalizedMessage,
    NormalizedRequest,
    NormalizedResponse,
    ProtocolName,
    StreamEvent,
    ToolDefinition,
    UsageStats,
)
from .sse import encode_sse, iter_sse_events

OPENAI_CHAT = "openai_chat"
OPENAI_RESPONSES = "openai_responses"
ANTHROPIC_MESSAGES = "anthropic_messages"
RESPONSES_PASSTHROUGH_TOOLS_KEY = "_responses_passthrough_tools"


class ProtocolError(ValueError):
    """Raised when a request or response cannot be converted safely."""


def normalize_request(protocol: ProtocolName, body: dict[str, Any]) -> NormalizedRequest:
    if protocol == OPENAI_CHAT:
        return _normalize_openai_chat_request(body)
    if protocol == OPENAI_RESPONSES:
        return _normalize_openai_responses_request(body)
    if protocol == ANTHROPIC_MESSAGES:
        return _normalize_anthropic_request(body)
    raise ProtocolError(f"Unsupported inbound protocol `{protocol}`.")


def serialize_request(
    target_protocol: ProtocolName,
    target_model: str,
    request: NormalizedRequest,
) -> dict[str, Any]:
    if target_protocol == OPENAI_CHAT:
        return _serialize_openai_chat_request(target_model, request)
    if target_protocol == OPENAI_RESPONSES:
        return _serialize_openai_responses_request(target_model, request)
    if target_protocol == ANTHROPIC_MESSAGES:
        return _serialize_anthropic_request(target_model, request)
    raise ProtocolError(f"Unsupported outbound protocol `{target_protocol}`.")


def parse_response(protocol: ProtocolName, payload: dict[str, Any]) -> NormalizedResponse:
    if protocol == OPENAI_CHAT:
        return _parse_openai_chat_response(payload)
    if protocol == OPENAI_RESPONSES:
        return _parse_openai_responses_response(payload)
    if protocol == ANTHROPIC_MESSAGES:
        return _parse_anthropic_response(payload)
    raise ProtocolError(f"Unsupported response protocol `{protocol}`.")


def serialize_response(
    inbound_protocol: ProtocolName,
    request: NormalizedRequest,
    response: NormalizedResponse,
) -> dict[str, Any]:
    if inbound_protocol == OPENAI_CHAT:
        return _serialize_openai_chat_response(response)
    if inbound_protocol == OPENAI_RESPONSES:
        return _serialize_openai_responses_response(request, response)
    if inbound_protocol == ANTHROPIC_MESSAGES:
        return _serialize_anthropic_response(response)
    raise ProtocolError(f"Unsupported inbound protocol `{inbound_protocol}`.")


def build_stream_encoder(inbound_protocol: ProtocolName) -> BaseStreamEncoder:
    if inbound_protocol == OPENAI_CHAT:
        return OpenAIChatStreamEncoder()
    if inbound_protocol == OPENAI_RESPONSES:
        return OpenAIResponsesStreamEncoder()
    if inbound_protocol == ANTHROPIC_MESSAGES:
        return AnthropicStreamEncoder()
    raise ProtocolError(f"Unsupported inbound protocol `{inbound_protocol}`.")


async def iter_normalized_stream(
    protocol: ProtocolName,
    response: httpx.Response,
) -> AsyncIterator[StreamEvent]:
    if protocol == OPENAI_CHAT:
        async for event in _iter_openai_chat_stream(response):
            yield event
        return
    if protocol == OPENAI_RESPONSES:
        async for event in _iter_openai_responses_stream(response):
            yield event
        return
    if protocol == ANTHROPIC_MESSAGES:
        async for event in _iter_anthropic_stream(response):
            yield event
        return
    raise ProtocolError(f"Unsupported streaming protocol `{protocol}`.")


def default_endpoint_path(protocol: ProtocolName) -> str:
    if protocol == OPENAI_CHAT:
        return "/v1/chat/completions"
    if protocol == OPENAI_RESPONSES:
        return "/v1/responses"
    if protocol == ANTHROPIC_MESSAGES:
        return "/v1/messages"
    raise ProtocolError(f"Unsupported protocol `{protocol}`.")


def join_endpoint(base_url: str, endpoint: str) -> str:
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint
    # Split off query params from base_url so path joining works correctly.
    query = ""
    if "?" in base_url:
        base_url, query = base_url.split("?", 1)
        query = f"?{query}"
    base = base_url.rstrip("/")
    normalized = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    if base.endswith("/v1") and normalized.startswith("/v1/"):
        normalized = normalized[3:]
    return f"{base}{normalized}{query}"


def _normalize_openai_chat_request(body: dict[str, Any]) -> NormalizedRequest:
    model = _require_string(body.get("model"), "OpenAI chat request missing `model`.")
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ProtocolError("OpenAI chat request must include a non-empty `messages` list.")

    messages: list[NormalizedMessage] = []
    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            raise ProtocolError("Each OpenAI chat message must be an object.")
        role = _normalize_role(raw_message.get("role"))
        blocks = _parse_content_blocks(raw_message.get("content"))

        if role == "assistant":
            for tool_call in raw_message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    raise ProtocolError("OpenAI chat `tool_calls` entries must be objects.")
                function = tool_call.get("function") or {}
                blocks.append(
                    MessageBlock(
                        type="tool_use",
                        tool_name=_require_string(
                            function.get("name"),
                            "OpenAI chat tool call missing `function.name`.",
                        ),
                        tool_call_id=_optional_string(tool_call.get("id")) or _new_id("call"),
                        tool_input=_parse_json_or_raw(function.get("arguments")),
                    )
                )

            function_call = raw_message.get("function_call")
            if isinstance(function_call, dict):
                blocks.append(
                    MessageBlock(
                        type="tool_use",
                        tool_name=_require_string(
                            function_call.get("name"),
                            "OpenAI legacy function call missing `name`.",
                        ),
                        tool_call_id=_new_id("call"),
                        tool_input=_parse_json_or_raw(function_call.get("arguments")),
                    )
                )

        if role in {"tool", "function"}:
            blocks = [
                MessageBlock(
                    type="tool_result",
                    text=_extract_text(raw_message.get("content")),
                    tool_call_id=_optional_string(raw_message.get("tool_call_id"))
                    or _optional_string(raw_message.get("name"))
                    or _new_id("call"),
                )
            ]
            role = "tool"

        messages.append(NormalizedMessage(role=role, blocks=blocks))

    return NormalizedRequest(
        inbound_protocol=OPENAI_CHAT,
        model=model,
        stream=bool(body.get("stream")),
        messages=messages,
        tools=_normalize_openai_chat_tools(body),
        tool_choice=body.get("tool_choice") or _normalize_legacy_function_choice(body.get("function_call")),
        max_tokens=_optional_int(body.get("max_tokens")),
        temperature=_optional_float(body.get("temperature")),
        top_p=_optional_float(body.get("top_p")),
        stop_sequences=_normalize_stop_sequences(body.get("stop")),
        metadata={},
        extra=_pick_keys(
            body,
            {
                "frequency_penalty",
                "n",
                "parallel_tool_calls",
                "presence_penalty",
                "response_format",
                "seed",
                "stream_options",
                "user",
            },
        ),
    )


def _normalize_openai_responses_request(body: dict[str, Any]) -> NormalizedRequest:
    model = _require_string(body.get("model"), "OpenAI responses request missing `model`.")
    messages: list[NormalizedMessage] = []

    instructions = body.get("instructions")
    if instructions is not None:
        messages.append(
            NormalizedMessage(role="developer", blocks=_parse_content_blocks(instructions))
        )

    raw_input = body.get("input")
    if raw_input is None:
        raw_items: list[Any] = []
    elif isinstance(raw_input, list):
        raw_items = raw_input
    else:
        raw_items = [raw_input]

    for item in raw_items:
        if isinstance(item, str):
            messages.append(
                NormalizedMessage(
                    role="user",
                    blocks=[MessageBlock(type="text", text=item)],
                )
            )
            continue

        if not isinstance(item, dict):
            raise ProtocolError("Responses API input entries must be strings or objects.")

        item_type = item.get("type")
        if item_type == "message":
            role = _normalize_role(item.get("role"))
            messages.append(
                NormalizedMessage(role=role, blocks=_parse_content_blocks(item.get("content")))
            )
            continue

        if item_type == "function_call":
            messages.append(
                NormalizedMessage(
                    role="assistant",
                    blocks=[
                        MessageBlock(
                            type="tool_use",
                            tool_name=_require_string(
                                item.get("name"),
                                "Responses function call missing `name`.",
                            ),
                            tool_call_id=_require_string(
                                item.get("call_id"),
                                "Responses function call missing `call_id`.",
                            ),
                            tool_input=_parse_json_or_raw(item.get("arguments")),
                        )
                    ],
                )
            )
            continue

        if item_type == "function_call_output":
            messages.append(
                NormalizedMessage(
                    role="tool",
                    blocks=[
                        MessageBlock(
                            type="tool_result",
                            tool_call_id=_require_string(
                                item.get("call_id"),
                                "Responses function_call_output missing `call_id`.",
                            ),
                            text=_extract_text(item.get("output")),
                            is_error=bool(item.get("is_error")),
                        )
                    ],
                )
            )
            continue

        if item_type == "reasoning":
            continue

        if item_type is None and "role" in item:
            role = _normalize_role(item.get("role"))
            messages.append(
                NormalizedMessage(role=role, blocks=_parse_content_blocks(item.get("content")))
            )
            continue

        raise ProtocolError(f"Unsupported responses input item type `{item_type}`.")

    tools, passthrough_tools = _normalize_openai_responses_tools(body.get("tools"))
    extra = _pick_keys(
        body,
        {
            "background",
            "conversation",
            "include",
            "parallel_tool_calls",
            "reasoning",
            "store",
            "text",
            "truncation",
            "user",
        },
    )
    if passthrough_tools:
        extra[RESPONSES_PASSTHROUGH_TOOLS_KEY] = passthrough_tools

    return NormalizedRequest(
        inbound_protocol=OPENAI_RESPONSES,
        model=model,
        stream=bool(body.get("stream")),
        messages=messages,
        tools=tools,
        tool_choice=body.get("tool_choice"),
        max_tokens=_optional_int(body.get("max_output_tokens")),
        temperature=_optional_float(body.get("temperature")),
        top_p=_optional_float(body.get("top_p")),
        stop_sequences=[],
        metadata=_normalize_mapping(body.get("metadata")),
        previous_response_id=_optional_string(body.get("previous_response_id")),
        extra=extra,
    )


def _normalize_anthropic_request(body: dict[str, Any]) -> NormalizedRequest:
    model = _require_string(body.get("model"), "Anthropic request missing `model`.")
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ProtocolError("Anthropic request must include a non-empty `messages` list.")

    messages: list[NormalizedMessage] = []
    if body.get("system") is not None:
        messages.append(
            NormalizedMessage(role="system", blocks=_parse_content_blocks(body.get("system")))
        )

    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            raise ProtocolError("Anthropic messages entries must be objects.")
        role = _normalize_role(raw_message.get("role"))
        if role not in {"user", "assistant"}:
            raise ProtocolError("Anthropic message roles must be `user` or `assistant`.")
        messages.append(
            NormalizedMessage(role=role, blocks=_parse_content_blocks(raw_message.get("content")))
        )

    return NormalizedRequest(
        inbound_protocol=ANTHROPIC_MESSAGES,
        model=model,
        stream=bool(body.get("stream")),
        messages=messages,
        tools=_normalize_anthropic_tools(body.get("tools")),
        tool_choice=body.get("tool_choice"),
        max_tokens=_optional_int(body.get("max_tokens")),
        temperature=_optional_float(body.get("temperature")),
        top_p=_optional_float(body.get("top_p")),
        stop_sequences=_normalize_stop_sequences(body.get("stop_sequences")),
        metadata=_normalize_mapping(body.get("metadata")),
        extra=_pick_keys(
            body,
            {
                "container",
                "service_tier",
                "thinking",
            },
        ),
    )


def _serialize_openai_chat_request(target_model: str, request: NormalizedRequest) -> dict[str, Any]:
    _ensure_no_passthrough_responses_tools(request, OPENAI_CHAT)
    payload: dict[str, Any] = {
        "model": target_model,
        "messages": _messages_to_openai_chat(request.messages),
        "stream": request.stream,
    }
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.stop_sequences:
        payload["stop"] = request.stop_sequences
    if request.tools and request.tool_choice != "none":
        payload["tools"] = [_tool_to_openai_chat(tool) for tool in request.tools]
        if request.tool_choice is not None:
            payload["tool_choice"] = _tool_choice_to_openai_chat(request.tool_choice)
    payload.update(_pick_keys(request.extra, {"response_format", "seed", "user"}))
    if request.stream:
        stream_options = dict(request.extra.get("stream_options") or {})
        stream_options.setdefault("include_usage", True)
        payload["stream_options"] = stream_options
    return payload


def _serialize_openai_responses_request(
    target_model: str,
    request: NormalizedRequest,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": target_model,
        "input": _messages_to_openai_responses(request.messages),
        "stream": request.stream,
    }
    instructions = _collect_system_text(request.messages)
    if instructions:
        payload["instructions"] = instructions
    if request.max_tokens is not None:
        payload["max_output_tokens"] = request.max_tokens
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.previous_response_id:
        payload["previous_response_id"] = request.previous_response_id
    response_tools: list[dict[str, Any]] = []
    if request.tools and request.tool_choice != "none":
        response_tools.extend(_tool_to_openai_responses(tool) for tool in request.tools)
    if request.tool_choice != "none":
        response_tools.extend(_responses_passthrough_tools(request))
    if response_tools:
        payload["tools"] = response_tools
    if request.tool_choice is not None:
        payload["tool_choice"] = _tool_choice_to_openai_responses(request.tool_choice)
    payload.update(
        _pick_keys(
            request.extra,
            {
                "background",
                "conversation",
                "include",
                "parallel_tool_calls",
                "reasoning",
                "store",
                "text",
                "truncation",
                "user",
            },
        )
    )
    if request.metadata and request.inbound_protocol == OPENAI_RESPONSES:
        payload["metadata"] = request.metadata
    return payload


def _serialize_anthropic_request(target_model: str, request: NormalizedRequest) -> dict[str, Any]:
    _ensure_no_passthrough_responses_tools(request, ANTHROPIC_MESSAGES)
    payload: dict[str, Any] = {
        "model": target_model,
        "messages": _messages_to_anthropic(request.messages),
        "stream": request.stream,
        "max_tokens": request.max_tokens or 1024,
    }
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.stop_sequences:
        payload["stop_sequences"] = request.stop_sequences

    system = _collect_system_text(request.messages)
    if system:
        payload["system"] = system

    if request.tool_choice == "none":
        pass
    elif request.tools:
        payload["tools"] = [_tool_to_anthropic(tool) for tool in request.tools]
        if request.tool_choice is not None:
            payload["tool_choice"] = _tool_choice_to_anthropic(request.tool_choice)

    payload.update(_pick_keys(request.extra, {"service_tier", "thinking"}))
    if request.metadata and request.inbound_protocol == ANTHROPIC_MESSAGES:
        payload["metadata"] = request.metadata
    return payload


def _parse_openai_chat_response(payload: dict[str, Any]) -> NormalizedResponse:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProtocolError("OpenAI chat response is missing `choices`.")
    choice = choices[0] or {}
    message = choice.get("message") or {}
    blocks = _parse_content_blocks(message.get("content"))
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        blocks.append(
            MessageBlock(
                type="tool_use",
                tool_name=_require_string(
                    function.get("name"),
                    "OpenAI chat response tool call missing `function.name`.",
                ),
                tool_call_id=_optional_string(tool_call.get("id")) or _new_id("call"),
                tool_input=_parse_json_or_raw(function.get("arguments")),
            )
        )
    return NormalizedResponse(
        response_id=_optional_string(payload.get("id")) or _new_id("chatcmpl"),
        model=_optional_string(payload.get("model")) or "",
        created=_optional_int(payload.get("created")) or _now_ts(),
        blocks=blocks,
        stop_reason=_map_openai_finish_reason(choice.get("finish_reason"), blocks),
        usage=_usage_from_openai_chat(payload.get("usage")),
    )


def _parse_openai_responses_response(payload: dict[str, Any]) -> NormalizedResponse:
    if payload.get("error"):
        raise ProtocolError(f"Responses API returned an error: {payload['error']}")
    blocks: list[MessageBlock] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            blocks.extend(_parse_content_blocks(item.get("content")))
        elif item_type == "function_call":
            blocks.append(
                MessageBlock(
                    type="tool_use",
                    tool_name=_require_string(
                        item.get("name"),
                        "Responses output function call missing `name`.",
                    ),
                    tool_call_id=_require_string(
                        item.get("call_id"),
                        "Responses output function call missing `call_id`.",
                    ),
                    tool_input=_parse_json_or_raw(item.get("arguments")),
                )
            )
    stop_reason = "tool_calls" if any(block.type == "tool_use" for block in blocks) else "stop"
    status = _optional_string(payload.get("status"))
    if status in {"incomplete", "max_output_tokens"}:
        stop_reason = "length"
    return NormalizedResponse(
        response_id=_optional_string(payload.get("id")) or _new_id("resp"),
        model=_optional_string(payload.get("model")) or "",
        created=_optional_int(payload.get("created_at")) or _now_ts(),
        blocks=blocks,
        stop_reason=stop_reason,
        usage=_usage_from_openai_responses(payload.get("usage")),
    )


def _parse_anthropic_response(payload: dict[str, Any]) -> NormalizedResponse:
    blocks = _parse_content_blocks(payload.get("content"))
    return NormalizedResponse(
        response_id=_optional_string(payload.get("id")) or _new_id("msg"),
        model=_optional_string(payload.get("model")) or "",
        created=_now_ts(),
        blocks=blocks,
        stop_reason=_map_anthropic_stop_reason(payload.get("stop_reason"), blocks),
        usage=_usage_from_anthropic(payload.get("usage")),
    )


def _serialize_openai_chat_response(response: NormalizedResponse) -> dict[str, Any]:
    text_content = _join_text_blocks(response.blocks) or None
    tool_calls = [
        {
            "id": block.tool_call_id or _new_id("call"),
            "type": "function",
            "function": {
                "name": block.tool_name or "tool",
                "arguments": _stringify_tool_input(block.tool_input),
            },
        }
        for block in response.blocks
        if block.type == "tool_use"
    ]
    finish_reason = _map_stop_reason_to_openai(response.stop_reason, response.blocks)
    return {
        "id": response.response_id,
        "object": "chat.completion",
        "created": response.created,
        "model": response.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text_content,
                    **({"tool_calls": tool_calls} if tool_calls else {}),
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": response.usage.to_openai_chat_dict(),
    }


def _serialize_openai_responses_response(
    request: NormalizedRequest,
    response: NormalizedResponse,
) -> dict[str, Any]:
    output = _blocks_to_openai_responses_output(response.blocks)
    data: dict[str, Any] = {
        "id": response.response_id,
        "object": "response",
        "created_at": response.created,
        "status": "completed",
        "model": response.model,
        "error": None,
        "incomplete_details": None,
        "instructions": _collect_system_text(request.messages) or None,
        "metadata": request.metadata,
        "output": output,
        "parallel_tool_calls": sum(1 for block in response.blocks if block.type == "tool_use") > 1,
        "temperature": request.temperature,
        "tool_choice": _tool_choice_to_openai_responses(request.tool_choice)
        if request.tool_choice is not None
        else None,
        "tools": [_tool_to_openai_responses(tool) for tool in request.tools],
        "top_p": request.top_p,
        "max_output_tokens": request.max_tokens,
        "previous_response_id": request.previous_response_id,
        "reasoning": request.extra.get("reasoning"),
        "text": request.extra.get("text"),
        "truncation": request.extra.get("truncation", "disabled"),
        "usage": response.usage.to_openai_responses_dict(),
        "user": request.extra.get("user"),
        "store": bool(request.extra.get("store", False)),
        "background": bool(request.extra.get("background", False)),
    }
    return data


def _serialize_anthropic_response(response: NormalizedResponse) -> dict[str, Any]:
    return {
        "id": response.response_id,
        "type": "message",
        "role": "assistant",
        "model": response.model,
        "content": _blocks_to_anthropic_content(response.blocks),
        "stop_reason": _map_stop_reason_to_anthropic(response.stop_reason, response.blocks),
        "stop_sequence": None,
        "usage": response.usage.to_anthropic_dict(),
    }


async def _iter_openai_chat_stream(response: httpx.Response) -> AsyncIterator[StreamEvent]:
    response_id: str | None = None
    model: str | None = None
    created: int | None = None
    usage: UsageStats | None = None
    finish_reason: str | None = None
    started = False
    seen_tools: set[str] = set()

    async for event in iter_sse_events(response):
        data = event.get("data") or ""
        if data == "[DONE]":
            break
        payload = _load_json_event(data)
        response_id = _optional_string(payload.get("id")) or response_id or _new_id("chatcmpl")
        model = _optional_string(payload.get("model")) or model or ""
        created = _optional_int(payload.get("created")) or created or _now_ts()

        if not started:
            started = True
            yield StreamEvent(
                type="response_started",
                response_id=response_id,
                model=model,
                created=created,
            )

        if payload.get("usage"):
            usage = _usage_from_openai_chat(payload.get("usage"))

        for choice in payload.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str) and content:
                yield StreamEvent(
                    type="text_delta",
                    response_id=response_id,
                    model=model,
                    created=created,
                    item_key="message",
                    text=content,
                )

            for tool_delta in delta.get("tool_calls") or []:
                index = tool_delta.get("index", 0)
                item_key = f"tool:{index}"
                tool_id = _optional_string(tool_delta.get("id")) or _new_id("call")
                function = tool_delta.get("function") or {}
                tool_name = _optional_string(function.get("name"))
                if item_key not in seen_tools:
                    seen_tools.add(item_key)
                    yield StreamEvent(
                        type="tool_call_started",
                        response_id=response_id,
                        model=model,
                        created=created,
                        item_key=item_key,
                        tool_call_id=tool_id,
                        tool_name=tool_name,
                    )
                arguments = function.get("arguments")
                if isinstance(arguments, str) and arguments:
                    yield StreamEvent(
                        type="tool_call_delta",
                        response_id=response_id,
                        model=model,
                        created=created,
                        item_key=item_key,
                        tool_call_id=tool_id,
                        tool_name=tool_name,
                        arguments=arguments,
                    )

            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    if started:
        yield StreamEvent(
            type="response_finished",
            response_id=response_id,
            model=model,
            created=created,
            stop_reason=_map_openai_finish_reason(finish_reason, []),
            usage=usage,
        )


async def _iter_openai_responses_stream(response: httpx.Response) -> AsyncIterator[StreamEvent]:
    response_id: str | None = None
    model: str | None = None
    created: int | None = None
    started = False
    tool_ids: dict[str, tuple[str, str | None]] = {}
    last_tool_key: str | None = None

    async for event in iter_sse_events(response):
        data = event.get("data") or ""
        payload = _load_json_event(data)
        event_type = _require_string(payload.get("type"), "Responses stream event missing `type`.")

        if event_type == "response.created":
            raw_response = payload.get("response") or {}
            response_id = _optional_string(raw_response.get("id")) or response_id or _new_id("resp")
            model = _optional_string(raw_response.get("model")) or model or ""
            created = _optional_int(raw_response.get("created_at")) or created or _now_ts()
            started = True
            yield StreamEvent(
                type="response_started",
                response_id=response_id,
                model=model,
                created=created,
            )
            continue

        if not started:
            started = True
            response_id = response_id or _new_id("resp")
            model = model or ""
            created = created or _now_ts()
            yield StreamEvent(
                type="response_started",
                response_id=response_id,
                model=model,
                created=created,
            )

        if event_type == "response.output_item.added":
            item = payload.get("item") or {}
            output_index = payload.get("output_index", 0)
            item_key = f"output:{output_index}"
            if item.get("type") == "function_call":
                tool_call_id = _require_string(
                    item.get("call_id"),
                    "Responses streaming function call missing `call_id`.",
                )
                tool_name = _optional_string(item.get("name"))
                tool_ids[item_key] = (tool_call_id, tool_name)
                last_tool_key = item_key
                yield StreamEvent(
                    type="tool_call_started",
                    response_id=response_id,
                    model=model,
                    created=created,
                    item_key=item_key,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                )
            continue

        if event_type == "response.output_text.delta":
            output_index = payload.get("output_index", 0)
            yield StreamEvent(
                type="text_delta",
                response_id=response_id,
                model=model,
                created=created,
                item_key=f"output:{output_index}",
                text=_require_string(payload.get("delta"), "Responses text delta missing `delta`."),
            )
            continue

        if event_type == "response.function_call_arguments.delta":
            output_index = payload.get("output_index")
            item_key = last_tool_key if output_index is None else f"output:{output_index}"
            if item_key is None:
                raise ProtocolError("Responses tool argument delta arrived before tool call start.")
            tool_call_id, tool_name = tool_ids.get(item_key, (_new_id("call"), None))
            yield StreamEvent(
                type="tool_call_delta",
                response_id=response_id,
                model=model,
                created=created,
                item_key=item_key,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=_require_string(
                    payload.get("delta"),
                    "Responses function_call_arguments.delta missing `delta`.",
                ),
            )
            continue

        if event_type == "response.output_item.done":
            item = payload.get("item") or {}
            output_index = payload.get("output_index")
            item_key = (
                f"output:{output_index}"
                if output_index is not None
                else next(
                    (
                        key
                        for key, value in tool_ids.items()
                        if value[0] == item.get("call_id")
                    ),
                    None,
                )
            )
            if item_key and item.get("type") == "function_call":
                tool_call_id, tool_name = tool_ids.get(item_key, (_new_id("call"), None))
                yield StreamEvent(
                    type="tool_call_finished",
                    response_id=response_id,
                    model=model,
                    created=created,
                    item_key=item_key,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                )
            continue

        if event_type in {"response.completed", "response.incomplete", "response.failed"}:
            raw_response = payload.get("response") or {}
            usage = _usage_from_openai_responses(raw_response.get("usage"))
            if event_type == "response.failed":
                yield StreamEvent(
                    type="error",
                    response_id=response_id,
                    model=model,
                    created=created,
                    message=json.dumps(raw_response.get("error") or {"message": "Upstream response failed."}),
                )
                stop_reason = "error"
            else:
                stop_reason = _responses_completion_stop_reason(raw_response)
            yield StreamEvent(
                type="response_finished",
                response_id=response_id,
                model=model,
                created=created,
                stop_reason=stop_reason,
                usage=usage,
            )
            return


async def _iter_anthropic_stream(response: httpx.Response) -> AsyncIterator[StreamEvent]:
    response_id: str | None = None
    model: str | None = None
    created = _now_ts()
    started = False
    block_types: dict[str, str] = {}
    usage = UsageStats()
    stop_reason = "stop"

    async for event in iter_sse_events(response):
        data = event.get("data") or ""
        payload = _load_json_event(data)
        event_type = _require_string(payload.get("type"), "Anthropic stream event missing `type`.")

        if event_type == "message_start":
            message = payload.get("message") or {}
            response_id = _optional_string(message.get("id")) or response_id or _new_id("msg")
            model = _optional_string(message.get("model")) or model or ""
            usage = _usage_from_anthropic(message.get("usage"))
            started = True
            yield StreamEvent(
                type="response_started",
                response_id=response_id,
                model=model,
                created=created,
            )
            continue

        if not started:
            started = True
            response_id = response_id or _new_id("msg")
            model = model or ""
            yield StreamEvent(
                type="response_started",
                response_id=response_id,
                model=model,
                created=created,
            )

        if event_type == "content_block_start":
            index = payload.get("index", 0)
            item_key = f"block:{index}"
            content_block = payload.get("content_block") or {}
            block_type = _require_string(
                content_block.get("type"),
                "Anthropic content block missing `type`.",
            )
            block_types[item_key] = block_type
            if block_type == "tool_use":
                yield StreamEvent(
                    type="tool_call_started",
                    response_id=response_id,
                    model=model,
                    created=created,
                    item_key=item_key,
                    tool_call_id=_require_string(
                        content_block.get("id"),
                        "Anthropic tool_use block missing `id`.",
                    ),
                    tool_name=_require_string(
                        content_block.get("name"),
                        "Anthropic tool_use block missing `name`.",
                    ),
                )
            elif block_type == "text":
                initial_text = content_block.get("text")
                if isinstance(initial_text, str) and initial_text:
                    yield StreamEvent(
                        type="text_delta",
                        response_id=response_id,
                        model=model,
                        created=created,
                        item_key=item_key,
                        text=initial_text,
                    )
            continue

        if event_type == "content_block_delta":
            index = payload.get("index", 0)
            item_key = f"block:{index}"
            delta = payload.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                yield StreamEvent(
                    type="text_delta",
                    response_id=response_id,
                    model=model,
                    created=created,
                    item_key=item_key,
                    text=_require_string(delta.get("text"), "Anthropic text delta missing `text`."),
                )
            elif delta_type == "input_json_delta":
                yield StreamEvent(
                    type="tool_call_delta",
                    response_id=response_id,
                    model=model,
                    created=created,
                    item_key=item_key,
                    arguments=_require_string(
                        delta.get("partial_json"),
                        "Anthropic input_json_delta missing `partial_json`.",
                    ),
                )
            continue

        if event_type == "content_block_stop":
            index = payload.get("index", 0)
            item_key = f"block:{index}"
            if block_types.get(item_key) == "tool_use":
                yield StreamEvent(
                    type="tool_call_finished",
                    response_id=response_id,
                    model=model,
                    created=created,
                    item_key=item_key,
                )
            continue

        if event_type == "message_delta":
            delta = payload.get("delta") or {}
            stop_reason = _map_anthropic_stop_reason(delta.get("stop_reason"), [])
            usage = _usage_from_anthropic(payload.get("usage")) or usage
            continue

        if event_type == "message_stop":
            yield StreamEvent(
                type="response_finished",
                response_id=response_id,
                model=model,
                created=created,
                stop_reason=stop_reason,
                usage=usage,
            )
            return

        if event_type == "error":
            yield StreamEvent(
                type="error",
                response_id=response_id,
                model=model,
                created=created,
                message=json.dumps(payload.get("error") or {"message": "Anthropic stream error."}),
            )
            yield StreamEvent(
                type="response_finished",
                response_id=response_id,
                model=model,
                created=created,
                stop_reason="error",
                usage=usage,
            )
            return


class BaseStreamEncoder:
    def encode(self, event: StreamEvent) -> list[bytes]:
        raise NotImplementedError


class OpenAIChatStreamEncoder(BaseStreamEncoder):
    def __init__(self) -> None:
        self.response_id = _new_id("chatcmpl")
        self.model = ""
        self.created = _now_ts()
        self.started = False
        self.tool_indices: dict[str, int] = {}

    def encode(self, event: StreamEvent) -> list[bytes]:
        if event.response_id:
            self.response_id = event.response_id
        if event.model is not None:
            self.model = event.model
        if event.created is not None:
            self.created = event.created

        if event.type == "error":
            body = {"error": {"message": event.message or "Streaming proxy error.", "type": "proxy_error"}}
            return [encode_sse(json.dumps(body, ensure_ascii=False)), encode_sse("[DONE]")]

        chunks: list[bytes] = []
        if event.type == "response_started" and not self.started:
            self.started = True
            chunks.append(
                encode_sse(json.dumps(self._chunk({"role": "assistant"}, None), ensure_ascii=False))
            )
            return chunks

        if event.type == "text_delta":
            if not self.started:
                chunks.extend(self.encode(StreamEvent(type="response_started")))
            chunks.append(
                encode_sse(
                    json.dumps(self._chunk({"content": event.text or ""}, None), ensure_ascii=False)
                )
            )
            return chunks

        if event.type == "tool_call_started":
            if not self.started:
                chunks.extend(self.encode(StreamEvent(type="response_started")))
            item_key = event.item_key or _new_id("tool")
            index = self.tool_indices.setdefault(item_key, len(self.tool_indices))
            tool_id = event.tool_call_id or _new_id("call")
            payload = {
                "tool_calls": [
                    {
                        "index": index,
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": event.tool_name or "tool",
                            "arguments": "",
                        },
                    }
                ]
            }
            chunks.append(encode_sse(json.dumps(self._chunk(payload, None), ensure_ascii=False)))
            return chunks

        if event.type == "tool_call_delta":
            item_key = event.item_key or _new_id("tool")
            index = self.tool_indices.setdefault(item_key, len(self.tool_indices))
            payload = {
                "tool_calls": [
                    {
                        "index": index,
                        "function": {
                            "arguments": event.arguments or "",
                        },
                    }
                ]
            }
            chunks.append(encode_sse(json.dumps(self._chunk(payload, None), ensure_ascii=False)))
            return chunks

        if event.type == "response_finished":
            if not self.started:
                chunks.extend(self.encode(StreamEvent(type="response_started")))
            finish_reason = _map_stop_reason_to_openai(event.stop_reason, [])
            chunks.append(encode_sse(json.dumps(self._chunk({}, finish_reason), ensure_ascii=False)))
            if event.usage is not None:
                usage_payload = {
                    "id": self.response_id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "choices": [],
                    "usage": event.usage.to_openai_chat_dict(),
                }
                chunks.append(encode_sse(json.dumps(usage_payload, ensure_ascii=False)))
            chunks.append(encode_sse("[DONE]"))
            return chunks

        return []

    def _chunk(self, delta: dict[str, Any], finish_reason: str | None) -> dict[str, Any]:
        return {
            "id": self.response_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }


class AnthropicStreamEncoder(BaseStreamEncoder):
    def __init__(self) -> None:
        self.response_id = _new_id("msg")
        self.model = ""
        self.created = _now_ts()
        self.started = False
        self.item_indices: dict[str, int] = {}
        self.open_item_key: str | None = None
        self.open_type: str | None = None

    def encode(self, event: StreamEvent) -> list[bytes]:
        if event.response_id:
            self.response_id = event.response_id
        if event.model is not None:
            self.model = event.model
        if event.created is not None:
            self.created = event.created

        if event.type == "error":
            payload = {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": event.message or "Streaming proxy error.",
                },
            }
            return [encode_sse(json.dumps(payload, ensure_ascii=False), event="error")]

        chunks: list[bytes] = []
        if event.type == "response_started" and not self.started:
            self.started = True
            payload = {
                "type": "message_start",
                "message": {
                    "id": self.response_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }
            chunks.append(encode_sse(json.dumps(payload, ensure_ascii=False), event="message_start"))
            return chunks

        if event.type == "text_delta":
            chunks.extend(self._ensure_started())
            item_key = event.item_key or "text"
            chunks.extend(self._ensure_text_block(item_key))
            payload = {
                "type": "content_block_delta",
                "index": self.item_indices[item_key],
                "delta": {"type": "text_delta", "text": event.text or ""},
            }
            chunks.append(
                encode_sse(json.dumps(payload, ensure_ascii=False), event="content_block_delta")
            )
            return chunks

        if event.type == "tool_call_started":
            chunks.extend(self._ensure_started())
            item_key = event.item_key or _new_id("tool")
            chunks.extend(self._close_open_block())
            if item_key not in self.item_indices:
                self.item_indices[item_key] = len(self.item_indices)
            self.open_item_key = item_key
            self.open_type = "tool_use"
            payload = {
                "type": "content_block_start",
                "index": self.item_indices[item_key],
                "content_block": {
                    "type": "tool_use",
                    "id": event.tool_call_id or _new_id("call"),
                    "name": event.tool_name or "tool",
                    "input": {},
                },
            }
            chunks.append(
                encode_sse(json.dumps(payload, ensure_ascii=False), event="content_block_start")
            )
            return chunks

        if event.type == "tool_call_delta":
            chunks.extend(self._ensure_started())
            item_key = event.item_key or _new_id("tool")
            if item_key not in self.item_indices:
                chunks.extend(
                    self.encode(
                        StreamEvent(
                            type="tool_call_started",
                            item_key=item_key,
                            tool_call_id=event.tool_call_id,
                            tool_name=event.tool_name,
                        )
                    )
                )
            elif self.open_item_key != item_key or self.open_type != "tool_use":
                chunks.extend(self._close_open_block())
                self.open_item_key = item_key
                self.open_type = "tool_use"
            payload = {
                "type": "content_block_delta",
                "index": self.item_indices[item_key],
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": event.arguments or "",
                },
            }
            chunks.append(
                encode_sse(json.dumps(payload, ensure_ascii=False), event="content_block_delta")
            )
            return chunks

        if event.type == "tool_call_finished":
            if self.open_item_key == event.item_key and self.open_type == "tool_use":
                chunks.extend(self._close_open_block())
            return chunks

        if event.type == "response_finished":
            chunks.extend(self._ensure_started())
            chunks.extend(self._close_open_block())
            payload = {
                "type": "message_delta",
                "delta": {
                    "stop_reason": _map_stop_reason_to_anthropic(event.stop_reason, []),
                    "stop_sequence": None,
                },
                "usage": (event.usage or UsageStats()).to_anthropic_dict(),
            }
            chunks.append(encode_sse(json.dumps(payload, ensure_ascii=False), event="message_delta"))
            chunks.append(
                encode_sse(json.dumps({"type": "message_stop"}, ensure_ascii=False), event="message_stop")
            )
            return chunks

        return []

    def _ensure_started(self) -> list[bytes]:
        if self.started:
            return []
        return self.encode(StreamEvent(type="response_started"))

    def _ensure_text_block(self, item_key: str) -> list[bytes]:
        if item_key in self.item_indices and self.open_item_key == item_key and self.open_type == "text":
            return []
        chunks = self._close_open_block()
        if item_key not in self.item_indices:
            self.item_indices[item_key] = len(self.item_indices)
        self.open_item_key = item_key
        self.open_type = "text"
        payload = {
            "type": "content_block_start",
            "index": self.item_indices[item_key],
            "content_block": {"type": "text", "text": ""},
        }
        chunks.append(encode_sse(json.dumps(payload, ensure_ascii=False), event="content_block_start"))
        return chunks

    def _close_open_block(self) -> list[bytes]:
        if self.open_item_key is None:
            return []
        payload = {
            "type": "content_block_stop",
            "index": self.item_indices[self.open_item_key],
        }
        self.open_item_key = None
        self.open_type = None
        return [encode_sse(json.dumps(payload, ensure_ascii=False), event="content_block_stop")]


class OpenAIResponsesStreamEncoder(BaseStreamEncoder):
    def __init__(self) -> None:
        self.response_id = _new_id("resp")
        self.model = ""
        self.created = _now_ts()
        self.started = False
        self.next_output_index = 0
        self.message_item: dict[str, Any] | None = None
        self.tool_items: dict[str, dict[str, Any]] = {}
        self.output_order: list[str] = []

    def encode(self, event: StreamEvent) -> list[bytes]:
        if event.response_id:
            self.response_id = event.response_id
        if event.model is not None:
            self.model = event.model
        if event.created is not None:
            self.created = event.created

        if event.type == "error":
            payload = {
                "type": "response.failed",
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "created_at": self.created,
                    "model": self.model,
                    "status": "failed",
                    "error": {"message": event.message or "Streaming proxy error."},
                    "output": [],
                },
            }
            return [encode_sse(json.dumps(payload, ensure_ascii=False))]

        chunks: list[bytes] = []
        if event.type == "response_started" and not self.started:
            self.started = True
            payload = {
                "type": "response.created",
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "created_at": self.created,
                    "model": self.model,
                    "status": "in_progress",
                },
            }
            chunks.append(encode_sse(json.dumps(payload, ensure_ascii=False)))
            return chunks

        if event.type == "text_delta":
            chunks.extend(self._ensure_started())
            message, created = self._ensure_message_item()
            if created:
                chunks.append(
                    encode_sse(
                        json.dumps(
                            {
                                "type": "response.output_item.added",
                                "output_index": message["output_index"],
                                "item": {
                                    "id": message["id"],
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "in_progress",
                                    "content": [],
                                },
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            message["text"] += event.text or ""
            chunks.append(
                encode_sse(
                    json.dumps(
                        {
                            "type": "response.output_text.delta",
                            "output_index": message["output_index"],
                            "item_id": message["id"],
                            "delta": event.text or "",
                        },
                        ensure_ascii=False,
                    )
                )
            )
            return chunks

        if event.type == "tool_call_started":
            chunks.extend(self._ensure_started())
            item_key = event.item_key or _new_id("tool")
            if item_key not in self.tool_items:
                tool_item = {
                    "id": event.tool_call_id or _new_id("call"),
                    "call_id": event.tool_call_id or _new_id("call"),
                    "name": event.tool_name or "tool",
                    "arguments": "",
                    "output_index": self.next_output_index,
                    "done": False,
                }
                self.next_output_index += 1
                self.tool_items[item_key] = tool_item
                self.output_order.append(item_key)
                chunks.append(
                    encode_sse(
                        json.dumps(
                            {
                                "type": "response.output_item.added",
                                "output_index": tool_item["output_index"],
                                "item": {
                                    "id": tool_item["id"],
                                    "type": "function_call",
                                    "status": "in_progress",
                                    "call_id": tool_item["call_id"],
                                    "name": tool_item["name"],
                                    "arguments": "",
                                },
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            return chunks

        if event.type == "tool_call_delta":
            item_key = event.item_key or _new_id("tool")
            if item_key not in self.tool_items:
                chunks.extend(
                    self.encode(
                        StreamEvent(
                            type="tool_call_started",
                            item_key=item_key,
                            tool_call_id=event.tool_call_id,
                            tool_name=event.tool_name,
                        )
                    )
                )
            tool_item = self.tool_items[item_key]
            tool_item["arguments"] += event.arguments or ""
            chunks.append(
                encode_sse(
                    json.dumps(
                        {
                            "type": "response.function_call_arguments.delta",
                            "output_index": tool_item["output_index"],
                            "delta": event.arguments or "",
                        },
                        ensure_ascii=False,
                    )
                )
            )
            return chunks

        if event.type == "tool_call_finished":
            item_key = event.item_key or ""
            tool_item = self.tool_items.get(item_key)
            if tool_item is None or tool_item["done"]:
                return chunks
            tool_item["done"] = True
            chunks.append(
                encode_sse(
                    json.dumps(
                        {
                            "type": "response.output_item.done",
                            "output_index": tool_item["output_index"],
                            "item": {
                                "id": tool_item["id"],
                                "type": "function_call",
                                "status": "completed",
                                "call_id": tool_item["call_id"],
                                "name": tool_item["name"],
                                "arguments": tool_item["arguments"],
                            },
                        },
                        ensure_ascii=False,
                    )
                )
            )
            return chunks

        if event.type == "response_finished":
            chunks.extend(self._ensure_started())
            if self.message_item is not None and not self.message_item["done"]:
                message = self.message_item
                message["done"] = True
                chunks.append(
                    encode_sse(
                        json.dumps(
                            {
                                "type": "response.output_item.done",
                                "output_index": message["output_index"],
                                "item": {
                                    "id": message["id"],
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "completed",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": message["text"],
                                            "annotations": [],
                                        }
                                    ],
                                },
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            for item_key in self.output_order:
                if item_key == "message" or item_key not in self.tool_items:
                    continue
                chunks.extend(self.encode(StreamEvent(type="tool_call_finished", item_key=item_key)))
            output = self._build_output()
            response_payload = {
                "type": "response.completed",
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "created_at": self.created,
                    "model": self.model,
                    "status": "completed",
                    "error": None,
                    "incomplete_details": None,
                    "instructions": None,
                    "metadata": {},
                    "output": output,
                    "parallel_tool_calls": len(self.tool_items) > 1,
                    "temperature": None,
                    "tool_choice": "auto",
                    "tools": [],
                    "top_p": None,
                    "max_output_tokens": None,
                    "previous_response_id": None,
                    "reasoning": None,
                    "text": None,
                    "truncation": "disabled",
                    "usage": (event.usage or UsageStats()).to_openai_responses_dict(),
                    "user": None,
                    "store": False,
                    "background": False,
                },
            }
            chunks.append(encode_sse(json.dumps(response_payload, ensure_ascii=False)))
            return chunks

        return []

    def _ensure_started(self) -> list[bytes]:
        if self.started:
            return []
        return self.encode(StreamEvent(type="response_started"))

    def _ensure_message_item(self) -> tuple[dict[str, Any], bool]:
        if self.message_item is not None:
            return self.message_item, False
        message = {
            "id": _new_id("msg"),
            "text": "",
            "output_index": self.next_output_index,
            "done": False,
        }
        self.next_output_index += 1
        self.message_item = message
        self.output_order.append("message")
        return message, True

    def _build_output(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for item_key in self.output_order:
            if item_key == "message" and self.message_item is not None:
                output.append(
                    {
                        "id": self.message_item["id"],
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": self.message_item["text"],
                                "annotations": [],
                            }
                        ],
                    }
                )
                continue
            tool_item = self.tool_items.get(item_key)
            if tool_item is None:
                continue
            output.append(
                {
                    "id": tool_item["id"],
                    "type": "function_call",
                    "call_id": tool_item["call_id"],
                    "name": tool_item["name"],
                    "arguments": tool_item["arguments"],
                    "status": "completed",
                }
            )
        return output


def _messages_to_openai_chat(messages: list[NormalizedMessage]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for message in messages:
        text_blocks = [block for block in message.blocks if block.type == "text"]
        tool_use_blocks = [block for block in message.blocks if block.type == "tool_use"]
        tool_result_blocks = [block for block in message.blocks if block.type == "tool_result"]

        if message.role in {"system", "developer"}:
            payload.append({"role": message.role, "content": _join_text_blocks(text_blocks)})
            continue

        if message.role == "assistant":
            if text_blocks or tool_use_blocks:
                item: dict[str, Any] = {
                    "role": "assistant",
                    "content": _join_text_blocks(text_blocks) if text_blocks else None,
                }
                if tool_use_blocks:
                    item["tool_calls"] = [
                        {
                            "id": block.tool_call_id or _new_id("call"),
                            "type": "function",
                            "function": {
                                "name": block.tool_name or "tool",
                                "arguments": _stringify_tool_input(block.tool_input),
                            },
                        }
                        for block in tool_use_blocks
                    ]
                payload.append(item)
            for block in tool_result_blocks:
                payload.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.tool_call_id or _new_id("call"),
                        "content": block.text or "",
                    }
                )
            continue

        if text_blocks:
            payload.append({"role": "user", "content": _join_text_blocks(text_blocks)})
        for block in tool_result_blocks:
            payload.append(
                {
                    "role": "tool",
                    "tool_call_id": block.tool_call_id or _new_id("call"),
                    "content": block.text or "",
                }
            )
    return payload


def _messages_to_openai_responses(messages: list[NormalizedMessage]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        if message.role in {"system", "developer"}:
            continue

        text_blocks = [block for block in message.blocks if block.type == "text"]
        tool_use_blocks = [block for block in message.blocks if block.type == "tool_use"]
        tool_result_blocks = [block for block in message.blocks if block.type == "tool_result"]

        if text_blocks and message.role in {"user", "assistant"}:
            content_type = "output_text" if message.role == "assistant" else "input_text"
            items.append(
                {
                    "type": "message",
                    "role": message.role,
                    "content": [
                        {"type": content_type, "text": block.text or ""}
                        for block in text_blocks
                    ],
                }
            )

        for block in tool_use_blocks:
            items.append(
                {
                    "type": "function_call",
                    "call_id": block.tool_call_id or _new_id("call"),
                    "name": block.tool_name or "tool",
                    "arguments": _stringify_tool_input(block.tool_input),
                }
            )

        for block in tool_result_blocks:
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": block.tool_call_id or _new_id("call"),
                    "output": block.text or "",
                }
            )
    return items


def _messages_to_anthropic(messages: list[NormalizedMessage]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for message in messages:
        if message.role in {"system", "developer"}:
            continue

        assistant_content = [
            _block_to_anthropic(block)
            for block in message.blocks
            if block.type in {"text", "tool_use"} and message.role == "assistant"
        ]
        user_content = [
            _block_to_anthropic(block)
            for block in message.blocks
            if block.type in {"text", "tool_result"} and message.role != "assistant"
        ]
        if message.role == "assistant" and assistant_content:
            payload.append({"role": "assistant", "content": assistant_content})
        elif message.role != "assistant" and user_content:
            payload.append({"role": "user", "content": user_content})
        if message.role == "assistant":
            trailing_tool_results = [
                _block_to_anthropic(block)
                for block in message.blocks
                if block.type == "tool_result"
            ]
            if trailing_tool_results:
                payload.append({"role": "user", "content": trailing_tool_results})
    return payload


def _blocks_to_openai_responses_output(blocks: list[MessageBlock]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    text_buffer: list[str] = []

    def flush_text() -> None:
        if not text_buffer:
            return
        output.append(
            {
                "id": _new_id("msg"),
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "".join(text_buffer),
                        "annotations": [],
                    }
                ],
            }
        )
        text_buffer.clear()

    for block in blocks:
        if block.type == "text":
            text_buffer.append(block.text or "")
            continue
        if block.type != "tool_use":
            continue
        flush_text()
        output.append(
            {
                "id": block.tool_call_id or _new_id("call"),
                "type": "function_call",
                "call_id": block.tool_call_id or _new_id("call"),
                "name": block.tool_name or "tool",
                "arguments": _stringify_tool_input(block.tool_input),
                "status": "completed",
            }
        )

    flush_text()
    return output


def _blocks_to_anthropic_content(blocks: list[MessageBlock]) -> list[dict[str, Any]]:
    return [_block_to_anthropic(block) for block in blocks if block.type in {"text", "tool_use"}]


def _block_to_anthropic(block: MessageBlock) -> dict[str, Any]:
    if block.type == "text":
        return {"type": "text", "text": block.text or ""}
    if block.type == "tool_result":
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_call_id or _new_id("call"),
            "content": block.text or "",
            "is_error": block.is_error,
        }
    tool_input = block.tool_input
    if isinstance(tool_input, str):
        tool_input = _parse_json_or_raw(tool_input)
    if not isinstance(tool_input, dict):
        raise ProtocolError("Anthropic tool_use blocks require JSON object arguments.")
    return {
        "type": "tool_use",
        "id": block.tool_call_id or _new_id("call"),
        "name": block.tool_name or "tool",
        "input": tool_input,
    }


def _normalize_openai_chat_tools(body: dict[str, Any]) -> list[ToolDefinition]:
    tools: list[ToolDefinition] = []
    for raw_tool in body.get("tools") or []:
        if not isinstance(raw_tool, dict) or raw_tool.get("type") != "function":
            raise ProtocolError("Only OpenAI function tools are supported.")
        function = raw_tool.get("function") or {}
        tools.append(
            ToolDefinition(
                name=_require_string(function.get("name"), "OpenAI tool missing `function.name`."),
                description=_optional_string(function.get("description")),
                input_schema=_normalize_mapping(function.get("parameters")),
            )
        )
    for raw_function in body.get("functions") or []:
        if not isinstance(raw_function, dict):
            raise ProtocolError("OpenAI `functions` entries must be objects.")
        tools.append(
            ToolDefinition(
                name=_require_string(raw_function.get("name"), "OpenAI function missing `name`."),
                description=_optional_string(raw_function.get("description")),
                input_schema=_normalize_mapping(raw_function.get("parameters")),
            )
        )
    return tools


def _normalize_openai_responses_tools(raw_tools: Any) -> tuple[list[ToolDefinition], list[dict[str, Any]]]:
    tools: list[ToolDefinition] = []
    passthrough_tools: list[dict[str, Any]] = []
    for raw_tool in raw_tools or []:
        if not isinstance(raw_tool, dict):
            raise ProtocolError("Responses tools must be objects.")
        if raw_tool.get("type") == "function":
            tools.append(
                ToolDefinition(
                    name=_require_string(raw_tool.get("name"), "Responses tool missing `name`."),
                    description=_optional_string(raw_tool.get("description")),
                    input_schema=_normalize_mapping(
                        raw_tool.get("parameters") or raw_tool.get("input_schema")
                    ),
                )
            )
            continue
        passthrough_tools.append(dict(raw_tool))
    return tools, passthrough_tools


def _normalize_anthropic_tools(raw_tools: Any) -> list[ToolDefinition]:
    tools: list[ToolDefinition] = []
    for raw_tool in raw_tools or []:
        if not isinstance(raw_tool, dict):
            raise ProtocolError("Anthropic tool entries must be objects.")
        tools.append(
            ToolDefinition(
                name=_require_string(raw_tool.get("name"), "Anthropic tool missing `name`."),
                description=_optional_string(raw_tool.get("description")),
                input_schema=_normalize_mapping(
                    raw_tool.get("input_schema") or raw_tool.get("parameters")
                ),
            )
        )
    return tools


def _tool_to_openai_chat(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _tool_to_openai_responses(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema,
    }


def _tool_to_anthropic(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def _tool_choice_to_openai_chat(choice: Any) -> Any:
    if isinstance(choice, dict) and choice.get("type") == "tool":
        return {"type": "function", "function": {"name": choice.get("name")}}
    if choice == "required":
        return "required"
    return choice


def _tool_choice_to_openai_responses(choice: Any) -> Any:
    if isinstance(choice, dict) and choice.get("type") == "tool":
        return {"type": "function", "name": choice.get("name")}
    if isinstance(choice, dict) and choice.get("type") == "function":
        return {"type": "function", "name": choice.get("function", {}).get("name") or choice.get("name")}
    return choice


def _tool_choice_to_anthropic(choice: Any) -> Any:
    if choice in {None, "auto"}:
        return {"type": "auto"}
    if choice == "required":
        return {"type": "any"}
    if isinstance(choice, dict) and choice.get("type") == "function":
        function = choice.get("function") or {}
        return {"type": "tool", "name": function.get("name")}
    if isinstance(choice, dict) and choice.get("type") == "tool":
        return choice
    return choice


def _responses_passthrough_tools(request: NormalizedRequest) -> list[dict[str, Any]]:
    raw_tools = request.extra.get(RESPONSES_PASSTHROUGH_TOOLS_KEY) or []
    return [dict(tool) for tool in raw_tools if isinstance(tool, dict)]


def _ensure_no_passthrough_responses_tools(
    request: NormalizedRequest,
    target_protocol: ProtocolName,
) -> None:
    raw_tools = _responses_passthrough_tools(request)
    if not raw_tools:
        return
    raw_types = sorted({str(tool.get("type") or "unknown") for tool in raw_tools})
    raise ProtocolError(
        "Responses tools of type "
        f"{', '.join(f'`{tool_type}`' for tool_type in raw_types)} "
        f"require an `{OPENAI_RESPONSES}` target, not `{target_protocol}`."
    )


def _normalize_role(value: Any) -> str:
    role = _require_string(value, "Message is missing `role`.")
    if role not in {"system", "developer", "user", "assistant", "tool", "function"}:
        raise ProtocolError(f"Unsupported role `{role}`.")
    return role


def _parse_content_blocks(content: Any) -> list[MessageBlock]:
    if content is None:
        return []
    if isinstance(content, str):
        return [MessageBlock(type="text", text=content)]
    if isinstance(content, dict):
        if "role" in content:
            return _parse_content_blocks(content.get("content"))
        return _parse_content_blocks([content])
    if not isinstance(content, list):
        raise ProtocolError("Message content must be a string, object, or list of objects.")

    blocks: list[MessageBlock] = []
    for item in content:
        if isinstance(item, str):
            blocks.append(MessageBlock(type="text", text=item))
            continue
        if not isinstance(item, dict):
            raise ProtocolError("Content blocks must be strings or objects.")
        item_type = item.get("type")
        if item_type in {None, "text", "input_text", "output_text"}:
            blocks.append(
                MessageBlock(
                    type="text",
                    text=_require_string(item.get("text"), "Text block missing `text`."),
                )
            )
            continue
        if item_type == "tool_use":
            blocks.append(
                MessageBlock(
                    type="tool_use",
                    tool_name=_require_string(item.get("name"), "tool_use block missing `name`."),
                    tool_call_id=_require_string(item.get("id"), "tool_use block missing `id`."),
                    tool_input=item.get("input") or {},
                )
            )
            continue
        if item_type == "tool_result":
            blocks.append(
                MessageBlock(
                    type="tool_result",
                    tool_call_id=_require_string(
                        item.get("tool_use_id"),
                        "tool_result block missing `tool_use_id`.",
                    ),
                    text=_extract_text(item.get("content")),
                    is_error=bool(item.get("is_error")),
                )
            )
            continue
        if item_type == "function_call":
            blocks.append(
                MessageBlock(
                    type="tool_use",
                    tool_name=_require_string(item.get("name"), "function_call missing `name`."),
                    tool_call_id=_require_string(item.get("call_id"), "function_call missing `call_id`."),
                    tool_input=_parse_json_or_raw(item.get("arguments")),
                )
            )
            continue
        if item_type == "function_call_output":
            blocks.append(
                MessageBlock(
                    type="tool_result",
                    tool_call_id=_require_string(
                        item.get("call_id"),
                        "function_call_output missing `call_id`.",
                    ),
                    text=_extract_text(item.get("output")),
                    is_error=bool(item.get("is_error")),
                )
            )
            continue
        if item_type in {"thinking", "redacted_thinking"}:
            continue
        raise ProtocolError(f"Unsupported content block type `{item_type}`.")
    return blocks


def _collect_system_text(messages: list[NormalizedMessage]) -> str:
    parts = [
        _join_text_blocks(message.blocks)
        for message in messages
        if message.role in {"system", "developer"}
    ]
    return "\n\n".join(part for part in parts if part)


def _join_text_blocks(blocks: list[MessageBlock]) -> str:
    return "".join(block.text or "" for block in blocks if block.type == "text")


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {None, "text", "input_text", "output_text"}:
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if isinstance(value, dict):
        if "text" in value and isinstance(value["text"], str):
            return value["text"]
        if "content" in value:
            return _extract_text(value["content"])
    return str(value)


def _normalize_stop_sequences(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        items = [str(item) for item in value if item is not None]
        return items
    raise ProtocolError("Stop sequences must be a string or a list of strings.")


def _normalize_legacy_function_choice(value: Any) -> Any:
    if value is None or value == "auto":
        return None
    if value == "none":
        return "none"
    if isinstance(value, dict):
        return {"type": "function", "function": {"name": value.get("name")}}
    return value


def _usage_from_openai_chat(raw: Any) -> UsageStats:
    if not isinstance(raw, dict):
        return UsageStats()
    return UsageStats(
        input_tokens=_optional_int(raw.get("prompt_tokens")),
        output_tokens=_optional_int(raw.get("completion_tokens")),
        total_tokens=_optional_int(raw.get("total_tokens")),
    )


def _usage_from_openai_responses(raw: Any) -> UsageStats:
    if not isinstance(raw, dict):
        return UsageStats()
    return UsageStats(
        input_tokens=_optional_int(raw.get("input_tokens")),
        output_tokens=_optional_int(raw.get("output_tokens")),
        total_tokens=_optional_int(raw.get("total_tokens")),
    )


def _usage_from_anthropic(raw: Any) -> UsageStats:
    if not isinstance(raw, dict):
        return UsageStats()
    input_tokens = _optional_int(raw.get("input_tokens"))
    output_tokens = _optional_int(raw.get("output_tokens"))
    total_tokens = None
    if input_tokens is not None or output_tokens is not None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return UsageStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _responses_completion_stop_reason(response: dict[str, Any]) -> str:
    output = response.get("output") or []
    if any(isinstance(item, dict) and item.get("type") == "function_call" for item in output):
        return "tool_calls"
    status = response.get("status")
    if status in {"incomplete", "max_output_tokens"}:
        return "length"
    return "stop"


def _map_openai_finish_reason(reason: Any, blocks: list[MessageBlock]) -> str:
    if reason == "tool_calls":
        return "tool_calls"
    if reason == "length":
        return "length"
    if reason in {None, "", "stop"}:
        return "tool_calls" if any(block.type == "tool_use" for block in blocks) else "stop"
    return str(reason)


def _map_anthropic_stop_reason(reason: Any, blocks: list[MessageBlock]) -> str:
    if reason == "tool_use":
        return "tool_calls"
    if reason == "max_tokens":
        return "length"
    if reason in {None, "end_turn"}:
        return "tool_calls" if any(block.type == "tool_use" for block in blocks) else "stop"
    return str(reason)


def _map_stop_reason_to_openai(reason: Any, blocks: list[MessageBlock]) -> str:
    if reason == "tool_calls":
        return "tool_calls"
    if reason == "length":
        return "length"
    if any(block.type == "tool_use" for block in blocks):
        return "tool_calls"
    return "stop"


def _map_stop_reason_to_anthropic(reason: Any, blocks: list[MessageBlock]) -> str:
    if reason == "tool_calls" or any(block.type == "tool_use" for block in blocks):
        return "tool_use"
    if reason == "length":
        return "max_tokens"
    return "end_turn"


def _stringify_tool_input(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _parse_json_or_raw(value: Any) -> Any:
    if value is None:
        return {}
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _pick_keys(data: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key in keys and value is not None}


def _normalize_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ProtocolError("Expected a mapping object.")
    return dict(value)


def _load_json_event(data: str) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Invalid JSON stream event: {data}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("Stream event payload must be a JSON object.")
    return payload


def _require_string(value: Any, message: str) -> str:
    if value is None or (isinstance(value, str) and value == ""):
        raise ProtocolError(message)
    return str(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value)
    return value or None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:24]}"


def _now_ts() -> int:
    return int(time.time())
