from __future__ import annotations

import logging
from pathlib import Path

from .config import ServerConfig


def setup_logging(config: ServerConfig) -> logging.Logger:
    level = config.log_level.upper()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    handlers: list[logging.Handler] = []

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    if config.log_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    for logger_name in ("nano_llm_api", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.propagate = False
        logger.handlers.clear()
        for handler in handlers:
            logger.addHandler(handler)

    return logging.getLogger("nano_llm_api")
