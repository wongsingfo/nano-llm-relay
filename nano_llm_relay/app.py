from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import ConfigStore
from .logutil import setup_logging
from .protocols import ANTHROPIC_MESSAGES, OPENAI_CHAT, OPENAI_RESPONSES
from .service import ProxyError, ProxyService


def create_app(
    config_path: str | None = None,
    transport=None,
) -> FastAPI:
    resolved_config_path = config_path or os.environ.get("NANO_LLM_RELAY_CONFIG", "config.yaml")
    config_store = ConfigStore(resolved_config_path)
    config = config_store.get_config()
    logger = setup_logging(config.server)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = ProxyService(config_store=config_store, transport=transport)
        app.state.proxy_service = service
        yield
        await service.close()

    app = FastAPI(title="nano-llm-relay", lifespan=lifespan)

    @app.exception_handler(ProxyError)
    async def handle_proxy_error(_: Request, exc: ProxyError) -> JSONResponse:
        payload = {"error": {"message": exc.message}}
        if exc.details is not None:
            payload["error"]["details"] = exc.details
        return JSONResponse(payload, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception path=%s", request.url.path, exc_info=exc)
        return JSONResponse(
            {"error": {"message": "Internal server error."}},
            status_code=500,
        )

    @app.get("/")
    async def root():
        return {"name": "nano-llm-relay", "status": "ok"}

    @app.get("/healthz")
    async def healthz(request: Request):
        service: ProxyService = request.app.state.proxy_service
        return service.health()

    @app.get("/v1/models")
    async def list_models(request: Request):
        service: ProxyService = request.app.state.proxy_service
        return service.list_models()

    @app.get("/v1/provider-models")
    async def list_provider_models(request: Request):
        service: ProxyService = request.app.state.proxy_service
        return await service.list_provider_models()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _proxy_request(request, OPENAI_CHAT)

    @app.post("/v1/responses")
    async def responses(request: Request):
        return await _proxy_request(request, OPENAI_RESPONSES)

    @app.post("/v1/messages")
    async def messages(request: Request):
        return await _proxy_request(request, ANTHROPIC_MESSAGES)

    async def _proxy_request(request: Request, protocol: str):
        try:
            body = await request.json()
        except Exception as exc:
            raise ProxyError(400, "Request body must be valid JSON.") from exc
        if not isinstance(body, dict):
            raise ProxyError(400, "Request body must be a JSON object.")
        service: ProxyService = request.app.state.proxy_service
        return await service.handle_request(protocol, body, request.headers)

    return app


def main() -> None:
    config_path = os.environ.get("NANO_LLM_RELAY_CONFIG", "config.yaml")
    config = ConfigStore(config_path).get_config()
    uvicorn.run(
        create_app(config_path=config_path),
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level.lower(),
        log_config=None,
    )
