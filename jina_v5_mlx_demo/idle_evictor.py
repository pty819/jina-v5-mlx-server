import threading
import time


class IdleEvictor:
    """Background thread that clears a holder callback after a configurable idle period.

    Usage:
        evictor = IdleEvictor(evict=lambda: setattr(service, '_model', None), idle_seconds=1800)
        evictor.touch()   # call on every inference
        evictor.start()   # starts daemon thread
    """

    def __init__(self, evict, *, idle_seconds=1800):
        self._evict = evict
        self._idle_seconds = idle_seconds
        self._last_used = 0.0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stopping = False

    def touch(self):
        """Record that the resource was just used."""
        with self._lock:
            self._last_used = time.monotonic()

    def start(self):
        if self._thread is not None:
            return
        self._stopping = False
        self.touch()
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        self._thread = t

    def stop(self):
        self._stopping = True
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self):
        check_interval = min(60, self._idle_seconds / 2)
        while not self._stopping:
            time.sleep(check_interval)
            if self._stopping:
                return
            with self._lock:
                idle = time.monotonic() - self._last_used
            if idle >= self._idle_seconds:
                self._evict()
