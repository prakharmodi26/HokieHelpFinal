from __future__ import annotations
import logging
from collections import deque

_buffer: deque[str] = deque(maxlen=5000)


class RingBufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            _buffer.append(self.format(record))
        except Exception:
            pass


def install(level: int = logging.INFO) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    h = RingBufferHandler()
    h.setFormatter(fmt)
    h.setLevel(level)
    root = logging.getLogger()
    for existing in root.handlers:
        if isinstance(existing, RingBufferHandler):
            return
    root.addHandler(h)
    if root.level > level or root.level == logging.NOTSET:
        root.setLevel(level)


def get_logs(lines: int = 200) -> str:
    items = list(_buffer)
    if lines and lines > 0:
        items = items[-lines:]
    return "\n".join(items)
