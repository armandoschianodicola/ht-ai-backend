import logging
import os
from typing import Optional, Dict, Any

from pythonjsonlogger.json import JsonFormatter


class ContextLoggerAdapter(logging.LoggerAdapter):
    """LoggerAdapter that properly injects extra fields into log records."""

    def process(self, msg, kwargs):
        if "extra" in kwargs:
            kwargs["extra"].update(self.extra)
        else:
            kwargs["extra"] = self.extra
        return msg, kwargs


def get_logger(name: str = "rag_backend", extra: Optional[Dict[str, Any]] = None) -> logging.LoggerAdapter:
    """
    Returns a configured JSON logger compatible with Alloy/Loki.
    Supports stdout logging and optional file logging.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logging.LoggerAdapter(logger, extra or {})

    # Log level from environment
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(log_level)

    # ✅ Use the custom formatter
    formatter = JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s")

    # Stdout handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Optional file logging
    log_file = os.getenv("LOG_FILE")
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return ContextLoggerAdapter(logger, extra or {})
