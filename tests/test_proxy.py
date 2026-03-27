from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest
import yaml

from nano_llm_api.app import create_app
from nano_llm_api.config import ConfigStore, load_config
from nano_llm_api.service import ProxyService

pytestmark = pytest.mark.anyio


def make_config(tmp_path: Path, *, force_stream_openai_chat: bool = False) -> Path:
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
                **({"force_stream": True} if force_stream_openai_chat else {}),
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


def test_load_config_parses_provider_proxy(tmp_path: Path):
    config = {
        "providers": {
            "openai": {
                "base_url": "https://openai.test",
                "api_key": "openai-test-key",
                "proxy": "http://127.0.0.1:7890",
            }
        },
        "models": {
            "gpt-chat": {
                "provider": "openai",
                "protocol": "openai_chat",
                "target_model": "gpt-4o-mini",
            }
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded = load_config(path)

    assert loaded.providers["openai"].proxy == "http://127.0.0.1:7890"


async def test_provider_proxy_client_cache_reload_and_transport_bypass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = {
        "server": {
            "host": "127.0.0.1",
            "port": 8080,
            "log_level": "INFO",
            "log_file": str(tmp_path / "logs" / "test.log"),
            "timeout_seconds": 60,
        },
        "providers": {
            "openai": {
                "base_url": "https://openai.test",
                "api_key": "openai-test-key",
                "proxy": "http://127.0.0.1:7890",
            },
            "anthropic": {
                "base_url": "https://anthropic.test",
                "api_key": "anthropic-test-key",
                "proxy": "http://127.0.0.1:7891",
                "headers": {"anthropic-version": "2023-06-01"},
            },
        },
        "models": {
            "gpt-chat": {
                "provider": "openai",
                "protocol": "openai_chat",
                "target_model": "gpt-4o-mini",
            },
            "claude": {
                "provider": "anthropic",
                "protocol": "anthropic_messages",
                "target_model": "claude-3-7-sonnet-latest",
            },
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    created_clients: list[object] = []

    class RecordingAsyncClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            created_clients.append(self)

        async def aclose(self):
            self.closed = True

    monkeypatch.setattr("nano_llm_api.service.httpx.AsyncClient", RecordingAsyncClient)

    service = ProxyService(ConfigStore(path))
    initial_config = service.config_store.get_config()
    openai_client = await service._get_provider_client(initial_config.providers["openai"])
    anthropic_client = await service._get_provider_client(initial_config.providers["anthropic"])

    assert openai_client.kwargs["proxy"] == "http://127.0.0.1:7890"
    assert anthropic_client.kwargs["proxy"] == "http://127.0.0.1:7891"
    assert openai_client.kwargs["trust_env"] is False
    assert anthropic_client.kwargs["trust_env"] is False
    assert await service._get_provider_client(initial_config.providers["openai"]) is openai_client

    config["providers"]["openai"]["proxy"] = "http://127.0.0.1:7892"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    updated_mtime_ns = path.stat().st_mtime_ns + 1
    os.utime(path, ns=(updated_mtime_ns, updated_mtime_ns))

    reloaded_config = service.config_store.get_config()
    reloaded_openai_client = await service._get_provider_client(reloaded_config.providers["openai"])

    assert reloaded_openai_client is not openai_client
    assert reloaded_openai_client.kwargs["proxy"] == "http://127.0.0.1:7892"
    assert not openai_client.closed

    await service.close()
    assert all(client.closed for client in created_clients)

    bypass_service = ProxyService(
        ConfigStore(path),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True})),
    )
    bypass_config = bypass_service.config_store.get_config()
    bypass_client = await bypass_service._get_provider_client(bypass_config.providers["openai"])

    assert bypass_client.kwargs["transport"] is not None
    assert "proxy" not in bypass_client.kwargs
    assert bypass_client.kwargs["trust_env"] is False

    await bypass_service.close()


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


async def test_anthropic_inbound_to_openai_chat_non_stream_with_forced_upstream_stream(
    tmp_path: Path,
):
    config_path = make_config(tmp_path, force_stream_openai_chat=True)
    sse_payload = (
        'data: {"id":"chatcmpl_1","object":"chat.completion.chunk","created":1,'
        '"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl_1","object":"chat.completion.chunk","created":1,'
        '"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl_1","object":"chat.completion.chunk","created":1,'
        '"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n'
        "data: [DONE]\n\n"
    ).encode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://openai.test/v1/chat/completions"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["stream"] is True
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
            response = await client.post(
                "/v1/messages",
                headers={"anthropic-version": "2023-06-01"},
                json={
                    "model": "gpt-chat",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "Say hello"}],
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert body["content"] == [{"type": "text", "text": "Hello"}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"] == {"input_tokens": 5, "output_tokens": 2}


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


async def test_list_provider_models(tmp_path: Path):
    config_path = make_config(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "https://openai.test/v1/models"
        assert request.headers["authorization"] == "Bearer openai-test-key"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "gpt-4o-mini",
                        "object": "model",
                        "created": 1,
                        "owned_by": "openai",
                    }
                ],
            },
        )

    app = create_app(config_path=str(config_path), transport=httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.get("/v1/provider-models")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "provider_model_list"
    providers = {item["provider"]: item for item in body["data"]}
    assert providers["openai"] == {
        "provider": "openai",
        "status": "ok",
        "discovery_protocol": "openai_models",
        "models": [
            {
                "id": "gpt-4o-mini",
                "object": "model",
                "created": 1,
                "owned_by": "openai",
            }
        ],
        "error": None,
    }
    assert providers["anthropic"]["status"] == "error"
    assert providers["anthropic"]["discovery_protocol"] is None
    assert providers["anthropic"]["models"] == []
    assert providers["anthropic"]["error"]["type"] == "unsupported_provider"


async def test_list_provider_models_handles_upstream_failure(tmp_path: Path):
    config_path = make_config(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://openai.test/v1/models"
        raise httpx.ConnectError("boom", request=request)

    app = create_app(config_path=str(config_path), transport=httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.get("/v1/provider-models")

    assert response.status_code == 200
    providers = {item["provider"]: item for item in response.json()["data"]}
    assert providers["openai"]["status"] == "error"
    assert providers["openai"]["error"]["type"] == "request_failed"
    assert "boom" in providers["openai"]["error"]["message"]
    assert providers["anthropic"]["error"]["type"] == "unsupported_provider"


async def test_list_provider_models_handles_invalid_json(tmp_path: Path):
    config_path = make_config(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://openai.test/v1/models"
        return httpx.Response(
            200,
            text="not-json",
            headers={"content-type": "application/json"},
        )

    app = create_app(config_path=str(config_path), transport=httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.get("/v1/provider-models")

    assert response.status_code == 200
    providers = {item["provider"]: item for item in response.json()["data"]}
    assert providers["openai"]["status"] == "error"
    assert providers["openai"]["error"]["type"] == "invalid_response"


async def test_list_provider_models_uses_custom_auth_header(tmp_path: Path):
    config = {
        "server": {
            "host": "127.0.0.1",
            "port": 8080,
            "log_level": "INFO",
            "log_file": str(tmp_path / "logs" / "test.log"),
            "timeout_seconds": 60,
        },
        "providers": {
            "custom-openai": {
                "base_url": "https://custom-openai.test",
                "api_key": "custom-test-key",
                "auth_header": "x-api-key",
                "auth_prefix": "",
                "headers": {"x-tenant": "local"},
            }
        },
        "models": {
            "custom-gpt": {
                "provider": "custom-openai",
                "protocol": "openai_chat",
                "target_model": "gpt-custom",
            }
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "custom-test-key"
        assert request.headers["x-tenant"] == "local"
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "gpt-custom"}]},
        )

    app = create_app(config_path=str(config_path), transport=httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.get("/v1/provider-models")

    assert response.status_code == 200
    body = response.json()["data"]
    assert body == [
        {
            "provider": "custom-openai",
            "status": "ok",
            "discovery_protocol": "openai_models",
            "models": [{"id": "gpt-custom"}],
            "error": None,
        }
    ]
