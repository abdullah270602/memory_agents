from __future__ import annotations
import logging
import os
import sys
import contextvars


request_id_ctx = contextvars.ContextVar("request_id", default="-")


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.request_id = request_id_ctx.get()
        except Exception:
            record.request_id = "-"
        return True


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()
    if root.handlers:
        # Avoid duplicate handlers on reload
        for h in list(root.handlers):
            root.removeHandler(h)
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s %(levelname)s %(name)s rid=%(request_id)s %(message)s"
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%dT%H:%M:%S"))
    handler.addFilter(ContextFilter())
    root.addHandler(handler)

    # Silence overly chatty libs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)

