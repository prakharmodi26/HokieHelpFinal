"""Single-runner advisory lock for the crawler.

The frontier is a single-writer SQLite DB, and the engine is built for exactly
ONE crawler process at a time (lease_pending is a SELECT-then-UPDATE, and
requeue_leased() at startup reclaims *all* leased rows — a second concurrent
process would double-lease rows and steal the first runner's in-flight work).
This advisory POSIX file lock turns that foot-gun into a clean refusal: the
second process exits instead of corrupting the run.

POSIX-only (uses fcntl.flock) — the crawler runs on Linux containers and macOS
dev, both of which have fcntl. The lock is advisory: it only blocks other
processes that also acquire it, which every crawler entrypoint does.

The lock file lives alongside the frontier DB on the emptyDir volume, so a
pod restart (which preserves emptyDir) can detect a stale lock from a crashed
prior process within the same pod. On pod death, emptyDir is wiped and the
lock goes with it — a fresh pod starts clean.
"""
from __future__ import annotations

import contextlib
import errno
import fcntl
import logging
import os

logger = logging.getLogger(__name__)


class RunnerLockHeld(RuntimeError):
    """Raised when another crawler process already holds the run lock."""


@contextlib.contextmanager
def single_runner_lock(lock_path: str):
    """Hold an exclusive, non-blocking advisory lock at *lock_path* for the body.

    Yields on success; raises RunnerLockHeld immediately if another process
    already holds the lock (flock conflicts across independent open file
    descriptions, so even a second acquire within the same process is refused).
    The lock — and the underlying fd — is released on exit, and the OS releases
    it automatically if the process dies, so a crashed run never wedges the next.
    """
    parent = os.path.dirname(lock_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            raise RunnerLockHeld(
                f"Another crawler already holds the run lock at {lock_path}"
            ) from exc
        raise

    # Record our pid for debuggability (best-effort).
    with contextlib.suppress(OSError):
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())

    logger.info("Acquired single-runner lock %s (pid %d)", lock_path, os.getpid())
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
