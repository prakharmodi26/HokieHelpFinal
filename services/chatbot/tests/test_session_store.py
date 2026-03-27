# services/chatbot/tests/test_session_store.py
import time
import pytest
from chatbot.session_store import SessionStore


def test_new_session_is_allowed():
    store = SessionStore(max_requests=3, window_seconds=60)
    assert store.is_allowed("session-abc") is True


def test_within_limit_is_allowed():
    store = SessionStore(max_requests=3, window_seconds=60)
    store.is_allowed("s1")
    store.is_allowed("s1")
    assert store.is_allowed("s1") is True  # 3rd request, still ok


def test_exceeds_limit_is_rejected():
    store = SessionStore(max_requests=3, window_seconds=60)
    store.is_allowed("s1")
    store.is_allowed("s1")
    store.is_allowed("s1")
    assert store.is_allowed("s1") is False  # 4th request, over limit


def test_different_sessions_are_independent():
    store = SessionStore(max_requests=1, window_seconds=60)
    store.is_allowed("s1")
    assert store.is_allowed("s1") is False
    assert store.is_allowed("s2") is True  # different session unaffected


def test_window_expiry_resets_count():
    store = SessionStore(max_requests=2, window_seconds=1)
    store.is_allowed("s1")
    store.is_allowed("s1")
    assert store.is_allowed("s1") is False  # over limit
    time.sleep(1.05)
    assert store.is_allowed("s1") is True   # window expired, reset


def test_remaining_requests():
    store = SessionStore(max_requests=5, window_seconds=60)
    store.is_allowed("s1")
    store.is_allowed("s1")
    remaining = store.remaining("s1")
    assert remaining == 3


def test_get_or_create_session_returns_new_uuid():
    store = SessionStore(max_requests=10, window_seconds=60)
    sid = store.get_or_create_session(None)
    assert len(sid) == 36  # UUID4 format


def test_get_or_create_session_returns_existing():
    """Known sessions (previously minted) are returned as-is."""
    store = SessionStore(max_requests=10, window_seconds=60)
    # Mint a session first
    minted = store.get_or_create_session(None)
    # Now present it back — should be recognized
    sid = store.get_or_create_session(minted)
    assert sid == minted


def test_get_or_create_session_rejects_forged():
    """Arbitrary session IDs not minted by the store get a new UUID."""
    store = SessionStore(max_requests=10, window_seconds=60)
    forged = "forged-session-id"
    sid = store.get_or_create_session(forged)
    assert sid != forged
    assert len(sid) == 36  # new UUID minted


def test_get_or_create_session_accepts_after_is_allowed():
    """A session that was created via is_allowed() is recognized."""
    store = SessionStore(max_requests=10, window_seconds=60)
    store.is_allowed("real-session")
    sid = store.get_or_create_session("real-session")
    assert sid == "real-session"
