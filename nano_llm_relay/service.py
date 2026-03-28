from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

from fastapi import WebSocket

from .config import ConfigStore, ProviderConfig
from .models import MessageBlock, NormalizedResponse, ProtocolName, StreamEvent, UsageStats
from .protocols import (
    ANTHROPIC_MESSAGES,
    OPENAI_CHAT,
    OPENAI_RESPONSES,
    ProtocolError,
    build_stream_encoder,
    default_endpoint_path,
    iter_normalized_stream,
    join_endpoint,
    normalize_request,
    parse_response,
    serialize_request,
    serialize_response,
)
from .sse import iter_sse_json_lines

_DEBUG_LOG_STRING_LIMIT = 100
_DEBUG_LOG_STRING_SUFFIX = " ..."


class ProxyError(Exception):
    def __init__(self, status_code: int, message: str, details: Any | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.details = details


@dataclass(slots=True)
class _ProviderClientState:
    proxy: str | None
    client: httpx.AsyncClient


class ProxyService:
    def __init__(
        self,
        config_store: ConfigStore,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.config_store = config_store
        self.logger = logging.getLogger("nano_llm_relay")
        self.transport = transport
        self._client_lock = asyncio.Lock()
        self._provider_clients: dict[str, _ProviderClientState] = {}
        self._retired_clients: list[httpx.AsyncClient] = []

    async def close(self) -> None:
        async with self._client_lock:
            active_clients = [state.client for state in self._provider_clients.values()]
            retired_clients = list(self._retired_clients)
            self._provider_clients.clear()
            self._retired_clients.clear()

        for client in [*active_clients, *retired_clients]:
            await client.aclose()

    def health(self) -> dict[str, Any]:
        config = self.config_store.get_config()
        return {
            "status": "ok",
            "models": len(config.models),
            "config_path": str(self.config_store.path),
        }

    def list_models(self) -> dict[str, Any]:
        config = self.config_store.get_config()
        data = []
        for name, route in sorted(config.models.items()):
            data.append(
                {
                    "id": name,
                    "object": "model",
                    "created": 0,
                    "owned_by": f"nano-llm-relay/{route.provider}",
                }
            )
        return {"object": "list", "data": data}

    async def list_provider_models(self) -> dict[str, Any]:
        config = self.config_store.get_config()
        provider_names = sorted(config.providers)
        self.logger.info("provider_model_discovery providers=%s", ",".join(provider_names))
        results = await asyncio.gather(
            *[
                self._discover_provider_models(
                    provider_name=name,
                    provider=config.providers[name],
                    discovery_protocol=self._provider_discovery_protocol(name, config.models),
                    timeout_seconds=config.providers[name].timeout_seconds
                    or config.server.timeout_seconds,
                )
                for name in provider_names
            ]
        )
        return {"object": "provider_model_list", "data": results}

    async def handle_request(
        self,
        inbound_protocol: ProtocolName,
        body: dict[str, Any],
        inbound_headers: Mapping[str, str],
    ):
        started_at = time.perf_counter()
        self._log_inbound_request(inbound_protocol, body)
        try:
            normalized = normalize_request(inbound_protocol, body)
        except ProtocolError as exc:
            raise ProxyError(400, str(exc)) from exc

        config = self.config_store.get_config()
        route = config.models.get(normalized.model)
        if route is None:
            raise ProxyError(404, f"Unknown model alias `{normalized.model}`.")

        if (
            normalized.previous_response_id
            and not normalized.messages
            and route.protocol != OPENAI_RESPONSES
        ):
            raise ProxyError(
                400,
                "Stateless conversion of `previous_response_id` is only supported for openai_responses targets.",
            )

        provider = config.providers[route.provider]
        client = await self._get_provider_client(provider)
        try:
            payload = serialize_request(route.protocol, route.target_model, normalized)
        except ProtocolError as exc:
            raise ProxyError(400, str(exc)) from exc
        payload = self._merge_route_extra_body(route.protocol, payload, route.extra_body)

        url = join_endpoint(provider.base_url, route.endpoint or default_endpoint_path(route.protocol))
        headers = self._build_outbound_headers(
            provider=provider,
            protocol=route.protocol,
            inbound_headers=inbound_headers,
            stream=normalized.stream,
        )
        timeout = route.timeout_seconds or provider.timeout_seconds or config.server.timeout_seconds

        self.logger.info(
            "proxy_request inbound=%s outbound=%s model=%s target_model=%s stream=%s",
            inbound_protocol,
            route.protocol,
            normalized.model,
            route.target_model,
            normalized.stream,
        )

        if normalized.stream:
            return await self._handle_stream(
                client=client,
                inbound_protocol=inbound_protocol,
                outbound_protocol=route.protocol,
                url=url,
                headers=headers,
                payload=payload,
                timeout=timeout,
                started_at=started_at,
            )

        return await self._handle_json(
            client=client,
            inbound_protocol=inbound_protocol,
            outbound_protocol=route.protocol,
            url=url,
            headers=headers,
            payload=payload,
            timeout=timeout,
            normalized_request=normalized,
            started_at=started_at,
        )

    async def handle_websocket_request(
        self,
        ws: WebSocket,
        body: dict[str, Any],
        session_history: list[dict[str, Any]],
    ) -> None:
        """Handle a single response.create message over WebSocket.

        Proxies through to upstream via HTTP, then forwards SSE events
        as individual WebSocket JSON text frames.

        ``session_history`` is a mutable list of input items accumulated
        across turns on the same WS connection.  This method appends the
        current turn's input items and the assistant's output items so
        that subsequent turns carry the full conversation context (the
        upstream provider does not store responses).
        """
        started_at = time.perf_counter()
        inbound_protocol = OPENAI_RESPONSES

        # The body from response.create is the same shape as an HTTP POST body
        # (model, input, tools, stream, etc.) — strip WS-specific fields.
        body.pop("type", None)
        generate = body.pop("generate", True)
        body.pop("previous_response_id", None)
        body.setdefault("stream", True)

        # Warm-up request: generate=false means "cache prompt, don't generate".
        # Return synthetic response events without hitting upstream.
        if not generate:
            resp_id = f"resp_{uuid4().hex}"
            model = body.get("model", "unknown")
            created = int(time.time())
            self.logger.info("ws_warmup model=%s", model)
            stub_response = {
                "id": resp_id,
                "object": "response",
                "created_at": created,
                "status": "completed",
                "model": model,
                "output": [],
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }
            await ws.send_json({"type": "response.created", "response": stub_response})
            await ws.send_json({"type": "response.completed", "response": stub_response})
            return

        # Prepend session history so the upstream sees the full conversation.
        current_input = body.get("input") or []
        if isinstance(current_input, str):
            current_input = [{"type": "message", "role": "user", "content": current_input}]
        body["input"] = session_history + current_input

        # Save passthrough fields before normalization strips them.
        original_body = dict(body)

        self._log_inbound_request(inbound_protocol, body)
        try:
            normalized = normalize_request(inbound_protocol, body)
        except ProtocolError as exc:
            await ws.send_json({"type": "error", "error": {"message": str(exc)}})
            return

        config = self.config_store.get_config()
        route = config.models.get(normalized.model)
        if route is None:
            await ws.send_json({"type": "error", "error": {"message": f"Unknown model alias `{normalized.model}`."}})
            return

        provider = config.providers[route.provider]
        client = await self._get_provider_client(provider)
        try:
            payload = serialize_request(route.protocol, route.target_model, normalized)
        except ProtocolError as exc:
            await ws.send_json({"type": "error", "error": {"message": str(exc)}})
            return
        payload = self._merge_route_extra_body(route.protocol, payload, route.extra_body)

        # Force streaming for WebSocket
        payload["stream"] = True

        # Re-inject passthrough fields from the original body that the
        # normalize→serialize cycle doesn't handle (e.g. prompt_cache_key,
        # client_metadata, include).
        if inbound_protocol == route.protocol:
            for key, value in original_body.items():
                if key not in payload:
                    payload[key] = value

        url = join_endpoint(provider.base_url, route.endpoint or default_endpoint_path(route.protocol))
        headers = self._build_outbound_headers(
            provider=provider,
            protocol=route.protocol,
            inbound_headers={},
            stream=True,
        )
        timeout = route.timeout_seconds or provider.timeout_seconds or config.server.timeout_seconds

        self.logger.info(
            "ws_proxy_request outbound=%s model=%s target_model=%s",
            route.protocol,
            normalized.model,
            route.target_model,
        )

        request_context = client.stream("POST", url, headers=headers, json=payload, timeout=timeout)
        try:
            upstream_response = await request_context.__aenter__()
        except httpx.HTTPError as exc:
            await ws.send_json({"type": "error", "error": {"message": f"Upstream request failed: {exc}"}})
            return

        completed_response: dict[str, Any] | None = None
        try:
            if upstream_response.status_code >= 400:
                details = await self._safe_error_body(upstream_response)
                self.logger.warning(
                    "ws_upstream_error status=%s details=%s",
                    upstream_response.status_code,
                    details,
                )
                await ws.send_json({"type": "error", "error": {"message": "Upstream provider error.", "details": details}})
                return

            if inbound_protocol == route.protocol:
                # Passthrough: strip SSE framing, forward raw JSON as WS frames.
                # Also capture the response.completed event for session history.
                async for json_line in iter_sse_json_lines(upstream_response):
                    await ws.send_text(json_line)
                    if completed_response is None:
                        try:
                            evt = json.loads(json_line)
                            if isinstance(evt, dict) and evt.get("type") == "response.completed":
                                completed_response = evt.get("response", {})
                        except json.JSONDecodeError:
                            pass
            else:
                encoder = build_stream_encoder(inbound_protocol)
                async for event in iter_normalized_stream(route.protocol, upstream_response):
                    for chunk in encoder.encode(event):
                        text = chunk.decode("utf-8", errors="replace")
                        for line in text.splitlines():
                            if line.startswith("data:"):
                                data = line[5:].lstrip()
                                if data and data != "[DONE]":
                                    await ws.send_text(data)
                                    if completed_response is None:
                                        try:
                                            evt = json.loads(data)
                                            if isinstance(evt, dict) and evt.get("type") == "response.completed":
                                                completed_response = evt.get("response", {})
                                        except json.JSONDecodeError:
                                            pass
        except Exception as exc:
            self.logger.exception("ws_proxy_stream_error")
            try:
                await ws.send_json({"type": "error", "error": {"message": str(exc)}})
            except Exception:
                pass
        finally:
            await request_context.__aexit__(None, None, None)
            self.logger.info(
                "ws_proxy_complete model=%s duration_ms=%.2f",
                normalized.model,
                (time.perf_counter() - started_at) * 1000,
            )

        # Update session history with the current turn's input and the
        # assistant's output so the next turn has full context.
        if completed_response:
            session_history.extend(current_input)
            for output_item in completed_response.get("output", []):
                session_history.append(output_item)

    async def _handle_json(
        self,
        client: httpx.AsyncClient,
        inbound_protocol: ProtocolName,
        outbound_protocol: ProtocolName,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
        normalized_request,
        started_at: float,
    ) -> JSONResponse:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=timeout)
        except httpx.HTTPError as exc:
            raise ProxyError(502, f"Upstream request failed: {exc}") from exc

        if response.status_code >= 400:
            raise ProxyError(
                response.status_code,
                "Upstream provider returned an error.",
                await self._safe_error_body(response),
            )

        try:
            raw_response = response.json()
        except json.JSONDecodeError as exc:
            raise ProxyError(502, "Upstream provider returned non-JSON response.") from exc

        try:
            normalized_response = parse_response(outbound_protocol, raw_response)
            final_payload = serialize_response(inbound_protocol, normalized_request, normalized_response)
        except ProtocolError as exc:
            raise ProxyError(502, str(exc)) from exc

        self.logger.info(
            "proxy_response inbound=%s outbound=%s status=%s duration_ms=%.2f",
            inbound_protocol,
            outbound_protocol,
            response.status_code,
            (time.perf_counter() - started_at) * 1000,
        )
        return JSONResponse(final_payload)

    async def _handle_stream(
        self,
        client: httpx.AsyncClient,
        inbound_protocol: ProtocolName,
        outbound_protocol: ProtocolName,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
        started_at: float,
    ) -> StreamingResponse:
        request_context = client.stream(
            "POST",
            url,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        try:
            upstream_response = await request_context.__aenter__()
        except httpx.HTTPError as exc:
            raise ProxyError(502, f"Upstream streaming request failed: {exc}") from exc

        if upstream_response.status_code >= 400:
            details = await self._safe_error_body(upstream_response)
            await request_context.__aexit__(None, None, None)
            raise ProxyError(
                upstream_response.status_code,
                "Upstream provider returned an error.",
                details,
            )

        if inbound_protocol == outbound_protocol:
            return StreamingResponse(
                self._stream_passthrough_body(
                    request_context=request_context,
                    upstream_response=upstream_response,
                    inbound_protocol=inbound_protocol,
                    outbound_protocol=outbound_protocol,
                    started_at=started_at,
                ),
                media_type="text/event-stream",
            )

        encoder = build_stream_encoder(inbound_protocol)

        async def stream_body():
            try:
                async for event in iter_normalized_stream(outbound_protocol, upstream_response):
                    for chunk in encoder.encode(event):
                        yield chunk
            except Exception as exc:
                self.logger.exception("stream_proxy_error protocol=%s", inbound_protocol)
                for chunk in encoder.encode(StreamEvent(type="error", message=str(exc))):
                    yield chunk
            finally:
                await request_context.__aexit__(None, None, None)
                self.logger.info(
                    "proxy_stream_complete inbound=%s outbound=%s duration_ms=%.2f",
                    inbound_protocol,
                    outbound_protocol,
                    (time.perf_counter() - started_at) * 1000,
                )

        return StreamingResponse(stream_body(), media_type="text/event-stream")

    async def _stream_passthrough_body(
        self,
        request_context,
        upstream_response: httpx.Response,
        inbound_protocol: ProtocolName,
        outbound_protocol: ProtocolName,
        started_at: float,
    ):
        try:
            async for chunk in upstream_response.aiter_bytes():
                if chunk:
                    yield chunk
        finally:
            await request_context.__aexit__(None, None, None)
            self.logger.info(
                "proxy_stream_complete inbound=%s outbound=%s duration_ms=%.2f",
                inbound_protocol,
                outbound_protocol,
                (time.perf_counter() - started_at) * 1000,
            )

    def _build_outbound_headers(
        self,
        provider: ProviderConfig,
        protocol: ProtocolName,
        inbound_headers: Mapping[str, str],
        stream: bool,
    ) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        headers["accept"] = "text/event-stream" if stream else "application/json"
        headers.update(provider.headers)

        api_key = provider.resolved_api_key()
        if api_key:
            header_name = provider.auth_header or (
                "x-api-key" if protocol == ANTHROPIC_MESSAGES else "Authorization"
            )
            prefix = provider.auth_prefix
            if prefix is None and header_name.lower() == "authorization":
                prefix = "Bearer"
            headers[header_name] = f"{prefix} {api_key}".strip() if prefix else api_key

        if protocol == ANTHROPIC_MESSAGES:
            anthropic_version = inbound_headers.get("anthropic-version") or "2023-06-01"
            headers.setdefault("anthropic-version", anthropic_version)
            anthropic_beta = inbound_headers.get("anthropic-beta")
            if anthropic_beta:
                headers.setdefault("anthropic-beta", anthropic_beta)

        openai_beta = inbound_headers.get("openai-beta")
        if openai_beta and protocol in {OPENAI_CHAT, OPENAI_RESPONSES}:
            headers.setdefault("OpenAI-Beta", openai_beta)

        return headers

    def _log_inbound_request(self, inbound_protocol: ProtocolName, body: dict[str, Any]) -> None:
        if not self.logger.isEnabledFor(logging.DEBUG):
            return
        sanitized_body = _truncate_debug_log_value(body)
        try:
            serialized_body = json.dumps(sanitized_body, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            serialized_body = repr(sanitized_body)
        self.logger.debug(
            "proxy_inbound_request inbound=%s body=%s",
            inbound_protocol,
            serialized_body,
        )

    def _merge_route_extra_body(
        self,
        protocol: ProtocolName,
        payload: dict[str, Any],
        extra_body: Mapping[str, Any],
    ) -> dict[str, Any]:
        merged_payload = dict(payload)
        remaining_extra = dict(extra_body)

        if protocol == OPENAI_RESPONSES:
            self._merge_responses_reasoning_defaults(merged_payload, remaining_extra)

        merged_payload.update(remaining_extra)
        return merged_payload

    def _merge_responses_reasoning_defaults(
        self,
        payload: dict[str, Any],
        extra_body: dict[str, Any],
    ) -> None:
        reasoning_defaults = extra_body.pop("reasoning", None)
        legacy_effort = extra_body.pop("model_reasoning_effort", None)

        if reasoning_defaults is None and legacy_effort is None:
            return

        if reasoning_defaults is not None and not isinstance(reasoning_defaults, Mapping):
            if "reasoning" not in payload:
                payload["reasoning"] = reasoning_defaults
            return

        merged_reasoning = dict(payload.get("reasoning") or {}) if isinstance(payload.get("reasoning"), Mapping) else {}
        if isinstance(reasoning_defaults, Mapping):
            for key, value in reasoning_defaults.items():
                merged_reasoning.setdefault(str(key), value)
        if legacy_effort is not None:
            merged_reasoning.setdefault("effort", legacy_effort)
        if merged_reasoning:
            payload["reasoning"] = merged_reasoning

    async def _safe_error_body(self, response: httpx.Response) -> Any:
        try:
            raw = response.content
        except httpx.ResponseNotRead:
            try:
                raw = await response.aread()
            except Exception:
                return ""

        if not raw:
            return ""
        try:
            return json.loads(raw)
        except Exception:
            return raw.decode("utf-8", errors="replace")

    async def _discover_provider_models(
        self,
        provider_name: str,
        provider: ProviderConfig,
        discovery_protocol: str | None,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        if discovery_protocol != "openai_models":
            self.logger.warning(
                "provider_model_discovery_unsupported provider=%s",
                provider_name,
            )
            return self._provider_discovery_error(
                provider_name=provider_name,
                discovery_protocol=discovery_protocol,
                error_type="unsupported_provider",
                message="Provider model discovery currently supports only providers with OpenAI-compatible routes.",
            )

        url = join_endpoint(provider.base_url, "/v1/models")
        headers = self._build_provider_discovery_headers(provider)
        client = await self._get_provider_client(provider)
        try:
            response = await client.get(url, headers=headers, timeout=timeout_seconds)
        except httpx.HTTPError as exc:
            self.logger.warning(
                "provider_model_discovery_failed provider=%s error=%s",
                provider_name,
                exc,
            )
            return self._provider_discovery_error(
                provider_name=provider_name,
                discovery_protocol=discovery_protocol,
                error_type="request_failed",
                message=f"Provider model discovery request failed: {exc}",
            )

        if response.status_code >= 400:
            details = await self._safe_error_body(response)
            self.logger.warning(
                "provider_model_discovery_upstream_error provider=%s status=%s",
                provider_name,
                response.status_code,
            )
            return self._provider_discovery_error(
                provider_name=provider_name,
                discovery_protocol=discovery_protocol,
                error_type="upstream_error",
                message="Provider returned an error while listing models.",
                details={"status_code": response.status_code, "body": details},
            )

        try:
            payload = response.json()
        except json.JSONDecodeError:
            self.logger.warning(
                "provider_model_discovery_invalid_json provider=%s",
                provider_name,
            )
            return self._provider_discovery_error(
                provider_name=provider_name,
                discovery_protocol=discovery_protocol,
                error_type="invalid_response",
                message="Provider returned non-JSON model listing response.",
            )

        try:
            models = self._normalize_openai_models(payload)
        except ValueError as exc:
            self.logger.warning(
                "provider_model_discovery_invalid_shape provider=%s error=%s",
                provider_name,
                exc,
            )
            return self._provider_discovery_error(
                provider_name=provider_name,
                discovery_protocol=discovery_protocol,
                error_type="invalid_response",
                message=str(exc),
            )

        return {
            "provider": provider_name,
            "status": "ok",
            "discovery_protocol": discovery_protocol,
            "models": models,
            "error": None,
        }

    async def _get_provider_client(self, provider: ProviderConfig) -> httpx.AsyncClient:
        proxy = None if self.transport is not None else provider.proxy

        async with self._client_lock:
            state = self._provider_clients.get(provider.name)
            if state is not None and state.proxy == proxy:
                return state.client

            client = self._create_provider_client(proxy)
            if state is not None:
                # Keep stale clients alive until shutdown so in-flight requests can finish.
                self._retired_clients.append(state.client)
            self._provider_clients[provider.name] = _ProviderClientState(proxy=proxy, client=client)
            return client

    def _create_provider_client(self, proxy: str | None) -> httpx.AsyncClient:
        client_kwargs: dict[str, Any] = {
            "timeout": None,
            "trust_env": False,
        }
        if self.transport is not None:
            client_kwargs["transport"] = self.transport
        elif proxy:
            client_kwargs["proxy"] = proxy
        return httpx.AsyncClient(**client_kwargs)

    async def _collect_stream_response(
        self,
        outbound_protocol: ProtocolName,
        upstream_response: httpx.Response,
    ) -> NormalizedResponse:
        response_id = f"resp_{uuid4().hex}"
        model = ""
        created = int(time.time())
        stop_reason = "stop"
        usage = UsageStats()
        item_order: list[str] = []
        items: dict[str, dict[str, Any]] = {}

        async for event in iter_normalized_stream(outbound_protocol, upstream_response):
            if event.response_id:
                response_id = event.response_id
            if event.model is not None:
                model = event.model
            if event.created is not None:
                created = event.created

            if event.type == "error":
                raise ProxyError(
                    502,
                    "Upstream provider stream returned an error.",
                    self._parse_error_details(event.message),
                )

            if event.type == "response_finished":
                stop_reason = event.stop_reason or stop_reason
                if event.usage is not None:
                    usage = event.usage
                continue

            if event.type == "text_delta":
                item_key = event.item_key or "message"
                item = items.get(item_key)
                if item is None:
                    item = {"type": "text", "text_parts": []}
                    items[item_key] = item
                    item_order.append(item_key)
                item["text_parts"].append(event.text or "")
                continue

            if event.type in {"tool_call_started", "tool_call_delta", "tool_call_finished"}:
                item_key = event.item_key or event.tool_call_id or f"tool:{len(item_order)}"
                item = items.get(item_key)
                if item is None:
                    item = {
                        "type": "tool_use",
                        "tool_call_id": event.tool_call_id,
                        "tool_name": event.tool_name,
                        "arguments_parts": [],
                    }
                    items[item_key] = item
                    item_order.append(item_key)
                if event.tool_call_id:
                    item["tool_call_id"] = event.tool_call_id
                if event.tool_name:
                    item["tool_name"] = event.tool_name
                if event.arguments:
                    item["arguments_parts"].append(event.arguments)

        blocks: list[MessageBlock] = []
        for item_key in item_order:
            item = items[item_key]
            if item["type"] == "text":
                text = "".join(item["text_parts"])
                if text:
                    blocks.append(MessageBlock(type="text", text=text))
                continue
            arguments = "".join(item["arguments_parts"])
            blocks.append(
                MessageBlock(
                    type="tool_use",
                    tool_call_id=item.get("tool_call_id") or f"call_{uuid4().hex}",
                    tool_name=item.get("tool_name") or "tool",
                    tool_input=self._parse_tool_arguments(arguments),
                )
            )

        return NormalizedResponse(
            response_id=response_id,
            model=model,
            created=created,
            blocks=blocks,
            stop_reason=stop_reason,
            usage=usage,
        )

    def _parse_error_details(self, message: str | None) -> Any:
        if message is None:
            return None
        try:
            return json.loads(message)
        except Exception:
            return message

    def _provider_discovery_protocol(
        self,
        provider_name: str,
        models: Mapping[str, Any],
    ) -> str | None:
        route_protocols = {
            route.protocol
            for route in models.values()
            if route.provider == provider_name
        }
        if route_protocols & {OPENAI_CHAT, OPENAI_RESPONSES}:
            return "openai_models"
        return None

    def _build_provider_discovery_headers(self, provider: ProviderConfig) -> dict[str, str]:
        headers = {"accept": "application/json"}
        headers.update(provider.headers)

        api_key = provider.resolved_api_key()
        if api_key:
            header_name = provider.auth_header or "Authorization"
            prefix = provider.auth_prefix
            if prefix is None and header_name.lower() == "authorization":
                prefix = "Bearer"
            headers[header_name] = f"{prefix} {api_key}".strip() if prefix else api_key

        return headers

    def _normalize_openai_models(self, payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            raise ValueError("Provider model listing response must be a JSON object.")

        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("Provider model listing response is missing `data` list.")

        models: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("Provider model listing contains a non-object entry.")
            model_id = item.get("id")
            if model_id is None or str(model_id).strip() == "":
                raise ValueError("Provider model listing entry is missing `id`.")

            normalized: dict[str, Any] = {"id": str(model_id)}
            if item.get("object") is not None:
                normalized["object"] = str(item["object"])
            if item.get("owned_by") is not None:
                normalized["owned_by"] = str(item["owned_by"])
            created = item.get("created")
            if created is not None:
                try:
                    normalized["created"] = int(created)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Provider model listing entry has invalid `created`.") from exc
            models.append(normalized)
        return models

    def _provider_discovery_error(
        self,
        provider_name: str,
        discovery_protocol: str | None,
        error_type: str,
        message: str,
        details: Any | None = None,
    ) -> dict[str, Any]:
        error = {"type": error_type, "message": message}
        if details is not None:
            error["details"] = details
        return {
            "provider": provider_name,
            "status": "error",
            "discovery_protocol": discovery_protocol,
            "models": [],
            "error": error,
        }

    def _parse_tool_arguments(self, arguments: str) -> Any:
        stripped = arguments.strip()
        if not stripped:
            return {}
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return arguments


def _truncate_debug_log_value(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) <= _DEBUG_LOG_STRING_LIMIT:
            return value
        return f"{value[:_DEBUG_LOG_STRING_LIMIT]}{_DEBUG_LOG_STRING_SUFFIX}"
    if isinstance(value, Mapping):
        return {key: _truncate_debug_log_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_truncate_debug_log_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_truncate_debug_log_value(item) for item in value)
    return value
