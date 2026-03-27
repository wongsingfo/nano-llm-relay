from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ProtocolName = Literal["openai_chat", "openai_responses", "anthropic_messages"]
MessageRole = Literal["system", "user", "assistant", "tool"]
BlockType = Literal["text", "tool_use", "tool_result"]
StreamEventType = Literal[
    "response_started",
    "text_delta",
    "tool_call_started",
    "tool_call_delta",
    "tool_call_finished",
    "response_finished",
    "error",
]


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str | None = None
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MessageBlock:
    type: BlockType
    text: str | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_input: Any = None
    is_error: bool = False


@dataclass(slots=True)
class NormalizedMessage:
    role: MessageRole
    blocks: list[MessageBlock] = field(default_factory=list)


@dataclass(slots=True)
class UsageStats:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def to_openai_chat_dict(self) -> dict[str, int]:
        prompt_tokens = self.input_tokens or 0
        completion_tokens = self.output_tokens or 0
        total_tokens = self.total_tokens or (prompt_tokens + completion_tokens)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def to_openai_responses_dict(self) -> dict[str, int]:
        input_tokens = self.input_tokens or 0
        output_tokens = self.output_tokens or 0
        total_tokens = self.total_tokens or (input_tokens + output_tokens)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    def to_anthropic_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens or 0,
            "output_tokens": self.output_tokens or 0,
        }


@dataclass(slots=True)
class NormalizedRequest:
    inbound_protocol: ProtocolName
    model: str
    stream: bool
    messages: list[NormalizedMessage]
    tools: list[ToolDefinition] = field(default_factory=list)
    tool_choice: Any = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    previous_response_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NormalizedResponse:
    response_id: str
    model: str
    created: int
    blocks: list[MessageBlock]
    stop_reason: str
    usage: UsageStats = field(default_factory=UsageStats)


@dataclass(slots=True)
class StreamEvent:
    type: StreamEventType
    response_id: str | None = None
    model: str | None = None
    created: int | None = None
    text: str | None = None
    item_key: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments: str | None = None
    stop_reason: str | None = None
    usage: UsageStats | None = None
    message: str | None = None
