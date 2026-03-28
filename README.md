# nano-llm-relay

`nano-llm-relay` is a minimal local LLM proxy that bridges the two most common API dialects, OpenAI and Anthropic, with as few dependencies as possible. The goal is to give local tools a single endpoint that can forward requests to different backend models without pulling in the heavier operational surface area of a full gateway.

## Status

This project is still under active development. APIs, configuration, and protocol support may change, and some behavior is not yet finalized.

## Current support

- Inbound protocols:
  - `POST /v1/chat/completions`
  - `POST /v1/responses`
  - `POST /v1/messages`
- Outbound protocols:
  - `openai_chat`
  - `openai_responses`
  - `anthropic_messages`
- Additional endpoints:
  - `GET /`
  - `GET /healthz`
  - `GET /v1/models`
  - `GET /v1/provider-models`
- Capabilities:
  - Text request and response translation
  - Function/tool call translation
  - SSE streaming passthrough with event-shape conversion
  - YAML config hot reload
  - File-based logging

## Non-goals

- Databases, auth backends, load balancing, or admin UI
- Full compatibility for multimodal blocks, MCP, or custom tool types
- Server-side conversation state management; `previous_response_id` is only forwarded as-is to `openai_responses` backends

## Installation

```bash
uv sync
```

`uv` creates and manages the project virtual environment from `uv.lock`. For a runtime-only environment, use `uv sync --no-dev`.

## Configuration

Start from [`config.example.yaml`](config.example.yaml), copy it to `config.yaml`, then adjust provider settings, model routes, and credential sources.

```yaml
providers:
  anthropic:
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY
    proxy: http://127.0.0.1:7890
    headers:
      anthropic-version: "2023-06-01"

models:
  claude-sonnet:
    provider: anthropic
    protocol: anthropic_messages
    target_model: claude-3-7-sonnet-latest
```

Important fields:

- `server.host` / `server.port`: bind address for the local proxy
- `server.log_file`: file path for proxy logs
- `server.timeout_seconds`: default upstream timeout
- `providers.<name>.base_url`: upstream API base URL
- `providers.<name>.api_key` or `api_key_env`: upstream API key source
- `providers.<name>.proxy`: optional outbound proxy URL for requests to that provider
- `providers.<name>.auth_header` / `auth_prefix`: custom auth header settings for non-standard OpenAI-compatible backends
- `providers.<name>.headers`: extra static headers sent to the upstream provider
- `models.<alias>.protocol`: target upstream protocol
- `models.<alias>.target_model`: model name sent to the upstream provider
- `models.<alias>.endpoint`: optional endpoint override for that route
- `models.<alias>.extra_body`: extra fixed fields merged into the upstream request body

The config is reloaded automatically when the YAML file changes on disk.

Provider proxy settings apply to both normal JSON requests and SSE streaming requests. When `transport` is injected for tests, the configured proxy is ignored so mock transports continue to work.

`GET /v1/models` returns local alias routes configured in `models`.

`GET /v1/provider-models` performs live upstream discovery for each configured provider. In v1 it supports providers with at least one `openai_chat` or `openai_responses` route and queries their OpenAI-compatible `GET /v1/models` endpoint. Providers that are not yet supported for discovery are returned with a provider-scoped error entry instead of failing the whole response.

## Running

```bash
NANO_LLM_RELAY_CONFIG=config.yaml uv run nano-llm-relay
```

Or directly:

```bash
NANO_LLM_RELAY_CONFIG=config.yaml uv run python -m nano_llm_relay
```

## Examples

OpenAI Chat client calling an Anthropic backend:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "claude-sonnet",
    "messages": [{"role": "user", "content": "Say hello"}]
  }'
```

Anthropic client calling an OpenAI Chat backend:

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model": "gpt-4o-chat",
    "max_tokens": 128,
    "messages": [{"role": "user", "content": "Say hello"}]
  }'
```

List live supported models per provider:

```bash
curl http://127.0.0.1:8080/v1/provider-models
```

## Testing

```bash
uv run pytest
```

The test suite uses `httpx.MockTransport`, so it does not make external network calls. If you change dependencies, refresh the lockfile with `uv lock`.
