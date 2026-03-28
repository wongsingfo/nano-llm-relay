# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install with dev dependencies
pip install -e '.[dev]'
# or with uv:
uv sync

# Run the service
NANO_LLM_RELAY_CONFIG=config.yaml python -m nano_llm_relay

# Run all tests (no external calls — uses httpx.MockTransport)
pytest

# Run a single test
pytest tests/test_proxy.py::test_name
```

## Architecture

nano-llm-relay is a local proxy that bridges OpenAI (`/v1/chat/completions`, `/v1/responses`) and Anthropic (`/v1/messages`) API dialects. All inbound requests are normalized to a common representation, then re-serialized for the target upstream protocol.

**Request flow:**

```
HTTP Request
  → FastAPI route (app.py)
  → normalize_request()         # inbound protocol → NormalizedRequest
  → serialize_request()         # NormalizedRequest → target protocol
  → upstream HTTP (service.py)  # httpx.AsyncClient per provider
  → parse_response()            # upstream response → NormalizedResponse
  → serialize_response()        # NormalizedResponse → inbound protocol
  → HTTP Response
```

**Key modules:**

- `protocols.py` — All protocol translation logic. Three protocols: `openai_chat`, `openai_responses`, `anthropic_messages`. Stateful stream encoders (`OpenAIChatStreamEncoder`, `AnthropicStreamEncoder`, `OpenAIResponsesStreamEncoder`) buffer partial SSE events and emit complete ones.
- `service.py` — `ProxyService` owns per-provider `httpx.AsyncClient` instances, handles routing via config model aliases, and drives the normalize→serialize→upstream→parse→serialize cycle. Clients are recreated on proxy config changes and retired gracefully.
- `config.py` — `ConfigStore` hot-reloads YAML on mtime change; thread-safe via `Lock()`. Model routes map local aliases → `(provider, protocol, target_model)`.
- `models.py` — Normalized dataclasses: `NormalizedRequest`, `NormalizedResponse`, `NormalizedMessage`, `MessageBlock` (text/tool_use/tool_result), `StreamEvent`.
- `app.py` — FastAPI routes and lifespan (starts/stops `ProxyService`).
- `sse.py` — `encode_sse()` / `iter_sse_events()` helpers.

## Notable Conventions

- **Protocol normalization is the core invariant.** All message content reduces to typed `MessageBlock` objects. Never add shortcuts that bypass normalization.
- **Responses API passthrough tools**: Non-portable tool types (e.g. web search) are stashed in `NormalizedRequest.extra[RESPONSES_PASSTHROUGH_TOOLS_KEY]` and re-injected only when the target is the Responses protocol.
- **"developer" role** is preserved as a first-class role through normalization. It is only collapsed to "system" at serialization time for protocols that don't support it (Anthropic). For OpenAI Chat and Responses targets, "developer" is emitted as-is.
- **`extra_body`** in model route config is merged into the serialized upstream request body — used for provider-specific params.
- **Tests use `MockTransport`** — pass it via `ProxyService(mock_transport=...)` to intercept all upstream calls without network access.
