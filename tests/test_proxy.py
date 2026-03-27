from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import yaml

from nano_llm_api.app import create_app

pytestmark = pytest.mark.anyio


def make_config(tmp_path: Path) -> Path:
    config = {
        "server": {
            "host": "127.0.0.1",
            "port": 8080,
            "log_level": "INFO",
            "log_file": str(tmp_path / "logs" / "test.log"),
            "timeout_seconds": 60,
        },
        "providers": {
            "anthropic": {
                "base_url": "https://anthropic.test",
                "api_key": "anthropic-test-key",
                "headers": {"anthropic-version": "2023-06-01"},
            },
            "openai": {
                "base_url": "https://openai.test",
                "api_key": "openai-test-key",
            },
        },
        "models": {
            "claude": {
                "provider": "anthropic",
                "protocol": "anthropic_messages",
                "target_model": "claude-3-7-sonnet-latest",
            },
            "gpt-chat": {
                "provider": "openai",
                "protocol": "openai_chat",
                "target_model": "gpt-4o-mini",
            },
            "gpt-responses": {
                "provider": "openai",
                "protocol": "openai_responses",
                "target_model": "gpt-4.1-mini",
            },
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


async def test_openai_chat_inbound_to_anthropic_non_stream(tmp_path: Path):
    config_path = make_config(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://anthropic.test/v1/messages"
        assert request.headers["x-api-key"] == "anthropic-test-key"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["model"] == "claude-3-7-sonnet-latest"
        assert payload["system"] == "You are terse."
        assert payload["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        ]
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-7-sonnet-latest",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 4},
            },
        )

    app = create_app(config_path=str(config_path), transport=httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "claude",
                    "messages": [
                        {"role": "system", "content": "You are terse."},
                        {"role": "user", "content": "hello"},
                    ],
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "hi"
    assert body["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
    }


async def test_anthropic_inbound_to_openai_chat_stream(tmp_path: Path):
    config_path = make_config(tmp_path)
    sse_payload = (
        'data: {"id":"chatcmpl_1","object":"chat.completion.chunk","created":1,'
        '"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl_1","object":"chat.completion.chunk","created":1,'
        '"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl_1","object":"chat.completion.chunk","created":1,'
        '"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    ).encode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://openai.test/v1/chat/completions"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["messages"] == [
            {"role": "user", "content": "Say hello"},
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_payload,
        )

    app = create_app(config_path=str(config_path), transport=httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            async with client.stream(
                "POST",
                "/v1/messages",
                headers={"anthropic-version": "2023-06-01"},
                json={
                    "model": "gpt-chat",
                    "stream": True,
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "Say hello"}],
                },
            ) as response:
                body = (await response.aread()).decode("utf-8")

    assert response.status_code == 200
    assert "event: message_start" in body
    assert "event: content_block_delta" in body
    assert "Hello" in body
    assert "event: message_stop" in body


async def test_openai_responses_inbound_to_anthropic_tool_call(tmp_path: Path):
    config_path = make_config(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://anthropic.test/v1/messages"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["tools"] == [
            {
                "name": "get_weather",
                "description": "Weather lookup",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ]
        return httpx.Response(
            200,
            json={
                "id": "msg_tool_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-7-sonnet-latest",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "get_weather",
                        "input": {"location": "Tokyo"},
                    }
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 12, "output_tokens": 6},
            },
        )

    app = create_app(config_path=str(config_path), transport=httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/responses",
                json={
                    "model": "claude",
                    "input": [{"role": "user", "content": "What's the weather?"}],
                    "tools": [
                        {
                            "type": "function",
                            "name": "get_weather",
                            "description": "Weather lookup",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                        }
                    ],
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response"
    assert body["output"][0]["type"] == "function_call"
    assert body["output"][0]["call_id"] == "toolu_123"
    assert body["output"][0]["name"] == "get_weather"
    assert json.loads(body["output"][0]["arguments"]) == {"location": "Tokyo"}


async def test_list_models(tmp_path: Path):
    config_path = make_config(tmp_path)

    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("No upstream request expected for model listing.")

    app = create_app(config_path=str(config_path), transport=httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.get("/v1/models")

    assert response.status_code == 200
    data = response.json()["data"]
    assert {item["id"] for item in data} == {"claude", "gpt-chat", "gpt-responses"}
