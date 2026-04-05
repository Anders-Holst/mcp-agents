"""
Shared event dispatcher used by all modules.

Provides a thread-safe publish/subscribe system. Any module can create
an EventDispatcher, and consumers subscribe with a callback + optional
event type filter.

Usage:
    dispatcher = EventDispatcher()
    unsub = dispatcher.subscribe(my_callback, event_types={MyEventType.FOO})
    dispatcher.dispatch(some_event)
    unsub()  # unsubscribe
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

EventCallback = Callable[[Any], None]


@dataclass
class _Subscription:
    callback: EventCallback
    event_types: Optional[set]


class EventDispatcher:
    """Thread-safe event dispatcher with optional type filtering."""

    def __init__(self, owner: str = ""):
        self._subs: list[_Subscription] = []
        self._lock = threading.Lock()
        self._owner = owner  # for log messages

    def subscribe(self, callback: EventCallback,
                  event_types: Optional[set] = None) -> Callable[[], None]:
        """Subscribe to events. Returns an unsubscribe function."""
        sub = _Subscription(callback=callback, event_types=event_types)
        with self._lock:
            self._subs.append(sub)

        def _unsub():
            with self._lock:
                try:
                    self._subs.remove(sub)
                except ValueError:
                    pass
        return _unsub

    def unsubscribe(self, callback: EventCallback) -> bool:
        """Remove all subscriptions for a given callback. Returns True if any were removed."""
        with self._lock:
            before = len(self._subs)
            self._subs = [s for s in self._subs if s.callback is not callback]
            return len(self._subs) < before

    def dispatch(self, event):
        """Send an event to all matching subscribers."""
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            if sub.event_types is None or event.type in sub.event_types:
                try:
                    sub.callback(event)
                except Exception:
                    owner = f" ({self._owner})" if self._owner else ""
                    logger.exception(f"Exception in event callback{owner} for {event.type.name}")
