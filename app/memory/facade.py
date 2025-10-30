from __future__ import annotations
from typing import Iterable, List, Optional, Any
import logging
import httpx


logger = logging.getLogger(__name__)


class StoreFacade:
    """Resilient store wrapper with automatic fallback.

    Tries the primary store first; if a network/storage error occurs, it switches
    to the fallback store (typically in-memory) for the rest of the process.
    """

    def __init__(self, primary: Any, fallback: Any):
        self.primary = primary
        self.fallback = fallback
        self._use_fallback = False

    @property
    def active_backend(self) -> str:
        if self._use_fallback:
            return "fallback"
        return "primary"

    def _call(self, fn_name: str, *args, **kwargs):
        if not self._use_fallback:
            try:
                fn = getattr(self.primary, fn_name)
                return fn(*args, **kwargs)
            except (httpx.RequestError, httpx.ConnectError, Exception) as e:  # broad fallback
                logger.warning("store_primary_error op=%s err=%s -> switching to fallback", fn_name, e)
                self._use_fallback = True
        fn = getattr(self.fallback, fn_name)
        return fn(*args, **kwargs)

    # Methods mirrored from store interface
    def save_memory(self, *args, **kwargs):
        return self._call("save_memory", *args, **kwargs)

    def edit_memory(self, *args, **kwargs):
        return self._call("edit_memory", *args, **kwargs)

    def delete_memory(self, *args, **kwargs):
        return self._call("delete_memory", *args, **kwargs)

    def get_short_term(self, *args, **kwargs):
        return self._call("get_short_term", *args, **kwargs)

    def get_long_term_personal_keys(self, *args, **kwargs):
        return self._call("get_long_term_personal_keys", *args, **kwargs)

    def get_long_term_shared_keys(self, *args, **kwargs):
        return self._call("get_long_term_shared_keys", *args, **kwargs)

    def get_by_keys(self, *args, **kwargs):
        return self._call("get_by_keys", *args, **kwargs)

