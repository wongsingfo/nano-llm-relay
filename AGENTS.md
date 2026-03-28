## Final goal

Build an extremely small local proxy service that can dynamically translate between multiple inbound and outbound LLM protocols. Clients that expect either the OpenAI format or the Anthropic format, including SSE streaming behavior, must be able to call any configured backend model through this relay layer. This includes local tools such as Codex and Claude Code CLI.

This project intentionally targets a subset of `./litellm` and may use it as a reference. `litellm/litellm/llms`. Existing tools such as LiteLLM are powerful, but they also carry many dependencies to support enterprise features like load balancing, database integrations, and complex authorization flows. That makes startup slower and memory usage higher. For personal local usage, the priorities are lightweight runtime behavior and minimal configuration.

To stay lightweight and low-memory:

1. Language: Python + FastAPI. Go or Rust may use less memory, but Python is still the fastest way to build on top of the current AI ecosystem, especially for streaming, SDK integration, and prompt-processing logic. FastAPI's async model is enough to keep latency low under concurrency.
2. Package management: `uv` for all Python-related tasks — dependency resolution, virtualenv, running scripts and tools. No pip/pip-tools.
3. SDK choice: official SDKs such as `openai` or `anthropic` are acceptable when they improve development speed. If startup time and memory footprint matter more, prefer direct `httpx` requests.
4. Configuration: YAML via PyYAML. It is the easiest format to read and edit by hand for provider and model mappings.
5. No database: file logging is enough.

## Commands

```bash
# Install dependencies (uv manages all Python tooling)
uv sync

# Run the service
NANO_LLM_RELAY_CONFIG=config.yaml uv run python -m nano_llm_relay

# Run all tests (no external calls — uses httpx.MockTransport)
uv run pytest

# Run a single test
uv run pytest tests/test_proxy.py::test_name
```

## Architecture

nano-llm-relay is a local proxy that bridges OpenAI (`/v1/chat/completions`, `/v1/responses`) and Anthropic (`/v1/messages`) API dialects. All inbound requests are normalized to a common representation, then re-serialized for the target upstream protocol.

**Request flow (HTTP):**

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

**Request flow (WebSocket, `/v1/responses`):**

```
WS connection
  → FastAPI websocket handler (app.py)
  → loop: receive response.create JSON
    → handle warm-up (generate=false) locally
    → prepend session history to input
    → normalize → serialize → upstream HTTP SSE
    → strip SSE framing → forward as WS JSON frames
    → capture response.completed → update session history
```

**Key modules:**

- `protocols.py` — All protocol translation logic. Three protocols: `openai_chat`, `openai_responses`, `anthropic_messages`. Stateful stream encoders (`OpenAIChatStreamEncoder`, `AnthropicStreamEncoder`, `OpenAIResponsesStreamEncoder`) buffer partial SSE events and emit complete ones.
- `service.py` — `ProxyService` owns per-provider `httpx.AsyncClient` instances, handles routing via config model aliases, and drives the normalize→serialize→upstream→parse→serialize cycle. Also provides `handle_websocket_request()` for WS sessions with warm-up handling, session history management, and SSE→WS frame conversion. Clients are recreated on proxy config changes and retired gracefully.
- `config.py` — `ConfigStore` hot-reloads YAML on mtime change; thread-safe via `Lock()`. Model routes map local aliases → `(provider, protocol, target_model)`.
- `models.py` — Normalized dataclasses: `NormalizedRequest`, `NormalizedResponse`, `NormalizedMessage`, `MessageBlock` (text/tool_use/tool_result), `StreamEvent`.
- `app.py` — FastAPI routes (including `@websocket /v1/responses`) and lifespan (starts/stops `ProxyService`).
- `sse.py` — `encode_sse()` / `iter_sse_events()` / `iter_sse_json_lines()` helpers.

## Notable Conventions

- **Protocol normalization is the core invariant.** All message content reduces to typed `MessageBlock` objects. Never add shortcuts that bypass normalization.
- **Responses API passthrough tools**: Non-portable tool types (e.g. web search) are stashed in `NormalizedRequest.extra[RESPONSES_PASSTHROUGH_TOOLS_KEY]` and re-injected only when the target is the Responses protocol. When the target is a different protocol, passthrough tools are silently dropped.
- **"developer" role** is preserved as a first-class role through normalization. It is only collapsed to "system" at serialization time for protocols that don't support it (Anthropic). For OpenAI Chat and Responses targets, "developer" is emitted as-is.
- **`extra_body`** in model route config is merged into the serialized upstream request body — used for provider-specific params.
- **Tests use `MockTransport`** — pass it via `ProxyService(mock_transport=...)` to intercept all upstream calls without network access.
