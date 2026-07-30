"""
Structured error boundaries for thread entry points.

Provides a single decorator that catches exceptions at thread boundaries,
logs the full traceback, updates application state, and optionally
triggers a controlled failure (UI notification, thread restart).

Usage:
    @error_boundary(name="LiveScanner")
    def run_scanner(...):
        ...
"""
import functools
import logging
import threading
import time
import traceback
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class ErrorBoundary:
    """Holds configuration for how a thread should behave on crash."""

    def __init__(self, name: str = "Thread", on_error: Optional[Callable] = None,
                 auto_restart: bool = False, max_restarts: int = 5):
        self.name = name
        self.on_error = on_error
        self.auto_restart = auto_restart
        self.max_restarts = max_restarts
        self._restarts = 0

    def wrap(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                logger.error("[%s] Unhandled exception:\n%s",
                             self.name, traceback.format_exc())
                if self.on_error:
                    try:
                        self.on_error()
                    except Exception:
                        logger.error("[%s] on_error callback failed", self.name)
                if self.auto_restart:
                    self._restarts += 1
                    if self._restarts > self.max_restarts:
                        logger.critical("[%s] restart limit (%d) reached, giving up",
                                        self.name, self.max_restarts)
                        return None
                    delay = min(60.0, 2.0 ** self._restarts)
                    logger.info("[%s] restart %d/%d in %.0fs",
                                self.name, self._restarts, self.max_restarts, delay)
                    # CRITICAL: restart through `wrapper`, not `func`, so the new
                    # thread is ALSO protected. The original respawned the bare
                    # function -> second crash was unhandled + thread leaked.
                    def _delayed():
                        time.sleep(delay)
                        wrapper(*args, **kwargs)
                    t = threading.Thread(target=_delayed, name=f"{self.name}-restart",
                                         daemon=True)
                    t.start()
                    return None
                return None
        return wrapper


def error_boundary(name: str = "Thread", on_error: Optional[Callable] = None,
                   auto_restart: bool = False, max_restarts: int = 5) -> Callable:
    """Decorator for thread entry points.

    Parameters
    ----------
    name : str
        Human-readable name for log messages.
    on_error : callable, optional
        e.g. ``lambda: setattr(config, 'live_trading_running', False)``
    auto_restart : bool
        If True, spawns a new thread running the same function when it crashes.
    max_restarts : int
        Maximum number of restart attempts.
    """
    boundary = ErrorBoundary(name=name, on_error=on_error, auto_restart=auto_restart, max_restarts=max_restarts)
    return boundary.wrap
