from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import yaml

from .models import ProtocolName


@dataclass(slots=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    log_file: str = "logs/nano-llm-api.log"
    timeout_seconds: float = 600.0


@dataclass(slots=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str | None = None
    api_key_env: str | None = None
    auth_header: str | None = None
    auth_prefix: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float | None = None

    def resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


@dataclass(slots=True)
class ModelRoute:
    name: str
    provider: str
    protocol: ProtocolName
    target_model: str
    max_tokens: int | None = None
    timeout_seconds: float | None = None
    endpoint: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AppConfig:
    server: ServerConfig
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelRoute]


class ConfigStore:
    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path).resolve()
        self._lock = Lock()
        self._mtime_ns: int | None = None
        self._config: AppConfig | None = None

    def get_config(self) -> AppConfig:
        with self._lock:
            if not self.path.exists():
                raise ValueError(f"Config file not found: {self.path}")

            mtime_ns = self.path.stat().st_mtime_ns
            if self._config is None or self._mtime_ns != mtime_ns:
                self._config = load_config(self.path)
                self._mtime_ns = mtime_ns
            return self._config


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a YAML mapping.")

    server_raw = raw.get("server") or {}
    server = ServerConfig(
        host=str(server_raw.get("host", "127.0.0.1")),
        port=int(server_raw.get("port", 8000)),
        log_level=str(server_raw.get("log_level", "INFO")).upper(),
        log_file=str(server_raw.get("log_file", "logs/nano-llm-api.log")),
        timeout_seconds=float(server_raw.get("timeout_seconds", 600.0)),
    )

    providers_raw = raw.get("providers") or {}
    if not isinstance(providers_raw, dict) or not providers_raw:
        raise ValueError("Config must define at least one provider under `providers`.")
    providers: dict[str, ProviderConfig] = {}
    for name, value in providers_raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"Provider `{name}` must be a mapping.")
        base_url = value.get("base_url")
        if not base_url:
            raise ValueError(f"Provider `{name}` is missing `base_url`.")
        headers = value.get("headers") or {}
        if not isinstance(headers, dict):
            raise ValueError(f"Provider `{name}` field `headers` must be a mapping.")
        providers[name] = ProviderConfig(
            name=name,
            base_url=str(base_url),
            api_key=_optional_str(value.get("api_key")),
            api_key_env=_optional_str(value.get("api_key_env")),
            auth_header=_optional_str(value.get("auth_header")),
            auth_prefix=_optional_str(value.get("auth_prefix")),
            headers={str(key): str(item) for key, item in headers.items()},
            timeout_seconds=_optional_float(value.get("timeout_seconds")),
        )

    models_raw = raw.get("models") or {}
    if not isinstance(models_raw, dict) or not models_raw:
        raise ValueError("Config must define at least one model under `models`.")
    models: dict[str, ModelRoute] = {}
    for name, value in models_raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"Model `{name}` must be a mapping.")
        provider_name = _required_str(value.get("provider"), f"Model `{name}` missing `provider`.")
        if provider_name not in providers:
            raise ValueError(f"Model `{name}` references unknown provider `{provider_name}`.")
        protocol = _required_str(value.get("protocol"), f"Model `{name}` missing `protocol`.")
        if protocol not in {"openai_chat", "openai_responses", "anthropic_messages"}:
            raise ValueError(f"Model `{name}` has unsupported protocol `{protocol}`.")
        target_model = _required_str(
            value.get("target_model"),
            f"Model `{name}` missing `target_model`.",
        )
        extra_body = value.get("extra_body") or {}
        if not isinstance(extra_body, dict):
            raise ValueError(f"Model `{name}` field `extra_body` must be a mapping.")
        models[name] = ModelRoute(
            name=name,
            provider=provider_name,
            protocol=protocol,
            target_model=target_model,
            max_tokens=_optional_int(value.get("max_tokens")),
            timeout_seconds=_optional_float(value.get("timeout_seconds")),
            endpoint=_optional_str(value.get("endpoint")),
            extra_body=dict(extra_body),
        )

    return AppConfig(server=server, providers=providers, models=models)


def _required_str(value: Any, message: str) -> str:
    if value is None or str(value).strip() == "":
        raise ValueError(message)
    return str(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
