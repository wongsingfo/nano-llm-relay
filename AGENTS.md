## Final goal

Build an extremely small local proxy service that can dynamically translate between multiple inbound and outbound LLM protocols. Clients that expect either the OpenAI format or the Anthropic format, including SSE streaming behavior, must be able to call any configured backend model through this relay layer. This includes local tools such as Codex and Claude Code CLI.

This project intentionally targets a subset of `./litellm` and may use it as a reference. `litellm/litellm/llms`. Existing tools such as LiteLLM are powerful, but they also carry many dependencies to support enterprise features like load balancing, database integrations, and complex authorization flows. That makes startup slower and memory usage higher. For personal local usage, the priorities are lightweight runtime behavior and minimal configuration.

To stay lightweight and low-memory:

1. Language: Python + FastAPI. Go or Rust may use less memory, but Python is still the fastest way to build on top of the current AI ecosystem, especially for streaming, SDK integration, and prompt-processing logic. FastAPI's async model is enough to keep latency low under concurrency.
2. SDK choice: official SDKs such as `openai` or `anthropic` are acceptable when they improve development speed. If startup time and memory footprint matter more, prefer direct `httpx` requests.
3. Configuration: YAML via PyYAML. It is the easiest format to read and edit by hand for provider and model mappings.
4. No database: file logging is enough.
