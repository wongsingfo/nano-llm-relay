from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from typing import Any

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

from .config import ConfigStore, ProviderConfig
from .models import ProtocolName, StreamEvent
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


class ProxyError(Exception):
    def __init__(self, status_code: int, message: str, details: Any | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.details = details


class ProxyService:
    def __init__(
        self,
        config_store: ConfigStore,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.config_store = config_store
        self.logger = logging.getLogger("nano_llm_api")
        self.client = httpx.AsyncClient(transport=transport, timeout=None)

    async def close(self) -> None:
        await self.client.aclose()

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
                    "owned_by": f"nano-llm-api/{route.provider}",
                }
            )
        return {"object": "list", "data": data}

    async def handle_request(
        self,
        inbound_protocol: ProtocolName,
        body: dict[str, Any],
        inbound_headers: Mapping[str, str],
    ):
        started_at = time.perf_counter()
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
        try:
            payload = serialize_request(route.protocol, route.target_model, normalized)
        except ProtocolError as exc:
            raise ProxyError(400, str(exc)) from exc
        payload.update(route.extra_body)

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
                inbound_protocol=inbound_protocol,
                outbound_protocol=route.protocol,
                url=url,
                headers=headers,
                payload=payload,
                timeout=timeout,
                started_at=started_at,
            )

        return await self._handle_json(
            inbound_protocol=inbound_protocol,
            outbound_protocol=route.protocol,
            url=url,
            headers=headers,
            payload=payload,
            timeout=timeout,
            normalized_request=normalized,
            started_at=started_at,
        )

    async def _handle_json(
        self,
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
            response = await self.client.post(url, headers=headers, json=payload, timeout=timeout)
        except httpx.HTTPError as exc:
            raise ProxyError(502, f"Upstream request failed: {exc}") from exc

        if response.status_code >= 400:
            raise ProxyError(
                response.status_code,
                "Upstream provider returned an error.",
                self._safe_error_body(response),
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
        inbound_protocol: ProtocolName,
        outbound_protocol: ProtocolName,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
        started_at: float,
    ) -> StreamingResponse:
        request_context = self.client.stream(
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
            details = self._safe_error_body(upstream_response)
            await request_context.__aexit__(None, None, None)
            raise ProxyError(
                upstream_response.status_code,
                "Upstream provider returned an error.",
                details,
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

    def _safe_error_body(self, response: httpx.Response) -> Any:
        try:
            return response.json()
        except Exception:
            return response.text
