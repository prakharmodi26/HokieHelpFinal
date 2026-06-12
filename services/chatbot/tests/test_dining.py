"""Tests for the VT dining runtime tool — HTTP is faked via injected getters."""
import json
from pathlib import Path

import pytest

from chatbot.tools.dining import (
    DINING_TOOL_SCHEMA,
    clear_cache,
    format_dining_context,
    get_dining,
    maybe_answer_dining,
    parse_hours,
    parse_menu,
)

FIXTURES = Path(__file__).parent / "fixtures"
HOURS_JSON = (FIXTURES / "hours_unitsopenonday.json").read_text()
MENU_HTML = (FIXTURES / "foodpro_menu.html").read_text()


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_cache()
    yield
    clear_cache()


class FakeHttp:
    """Counting fake HTTP getter: routes hours API vs FoodPro menu URLs."""

    def __init__(self, hours_body: str = HOURS_JSON, menu_body: str = MENU_HTML):
        self.hours_body = hours_body
        self.menu_body = menu_body
        self.calls: list[str] = []

    def __call__(self, url: str) -> str:
        self.calls.append(url)
        if "UnitsOpenOnDay" in url:
            return self.hours_body
        if "MenuAtLocation.aspx" in url:
            return self.menu_body
        raise AssertionError(f"unexpected URL fetched: {url}")

    @property
    def hours_calls(self) -> list[str]:
        return [u for u in self.calls if "UnitsOpenOnDay" in u]

    @property
    def menu_calls(self) -> list[str]:
        return [u for u in self.calls if "MenuAtLocation.aspx" in u]


# ── parse_hours ──────────────────────────────────────────────────────────────

def test_parse_hours_extracts_units_hours_and_location_num():
    units = parse_hours(json.loads(HOURS_JSON))

    assert len(units) == 3
    d2 = units[0]
    assert d2["name"] == "Dietrick - D2"
    assert d2["location_num"] == "15"
    assert [h["label"] for h in d2["hours"]] == ["Breakfast", "Lunch", "Dinner"]
    assert d2["hours"][0]["open"] == "07:00:01"
    assert d2["hours"][0]["close"] == "10:30:00"


def test_parse_hours_finds_location_num_when_foodpro_url_not_first():
    """Xpress Lane's urls[] lists the About page before the FoodPro menu URL."""
    units = parse_hours(json.loads(HOURS_JSON))
    xpress = next(u for u in units if "Xpress" in u["name"])
    assert xpress["location_num"] == "07"  # zero-padded string preserved


def test_parse_hours_absent_unit_means_closed():
    """A unit missing from the day's roster is simply not in the parsed list."""
    units = parse_hours(json.loads(HOURS_JSON))
    assert not any("West End" in u["name"] for u in units)


# ── parse_menu ───────────────────────────────────────────────────────────────

def test_parse_menu_extracts_items_with_portion_and_dietary():
    items = parse_menu(MENU_HTML)

    assert len(items) == 4
    by_name = {i["name"]: i for i in items}
    assert by_name["Bacon"]["portion"] == "1 Each"
    assert by_name["Bacon"]["dietary"] == []
    assert by_name["Cinnamon Apples"]["dietary"] == ["vegan"]
    assert by_name["Garden Salad"]["dietary"] == ["vegetarian"]
    assert by_name["Tofu Stir Fry"]["portion"] == "6 oz"


def test_parse_menu_attributes_meal_from_tab_panes():
    items = parse_menu(MENU_HTML)
    by_name = {i["name"]: i for i in items}
    assert by_name["Bacon"]["meal"] == "Breakfast"
    assert by_name["Garden Salad"]["meal"] == "Lunch"


# ── get_dining: join, filters, defaults ──────────────────────────────────────

def test_get_dining_joins_menu_via_location_num():
    fake = FakeHttp()
    result = get_dining(date="2026-06-11", location="D2", http_get=fake)

    assert result["date"] == "2026-06-11"
    assert len(result["units"]) == 1
    unit = result["units"][0]
    assert unit["name"] == "Dietrick - D2"
    assert unit["is_open"] is True
    # menu fetched from the FoodPro URL carrying locationNum=15
    assert len(fake.menu_calls) == 1
    assert "locationNum=15" in fake.menu_calls[0]
    assert any(i["name"] == "Bacon" for i in unit["menu"])


def test_get_dining_date_defaults_to_eastern_today():
    fake = FakeHttp()
    # 2026-06-12 01:30 UTC == 2026-06-11 21:30 EDT — Eastern date must win
    utc_just_past_midnight = 1781227800.0
    result = get_dining(http_get=fake, now=lambda: utc_just_past_midnight)

    assert result["date"] == "2026-06-11"
    assert "2026-06-11" in fake.hours_calls[0]


def test_get_dining_location_substring_filter_case_insensitive():
    fake = FakeHttp()
    result = get_dining(date="2026-06-11", location="turner", http_get=fake)

    assert [u["name"] for u in result["units"]] == ["Turner Place at Lavery Hall"]


def test_get_dining_no_location_match_returns_empty_units():
    fake = FakeHttp()
    result = get_dining(date="2026-06-11", location="West End", http_get=fake)

    assert result["units"] == []
    assert "error" not in result


def test_get_dining_meal_filter_restricts_hours_and_menu():
    fake = FakeHttp()
    result = get_dining(date="2026-06-11", location="D2", meal="Lunch", http_get=fake)

    unit = result["units"][0]
    assert [h["label"] for h in unit["hours"]] == ["Lunch"]
    menu_names = {i["name"] for i in unit["menu"]}
    assert "Garden Salad" in menu_names
    assert "Bacon" not in menu_names  # Breakfast item filtered out


def test_get_dining_meal_filter_marks_unit_closed_for_unserved_meal():
    fake = FakeHttp()
    result = get_dining(date="2026-06-11", location="Xpress", meal="Dinner", http_get=fake)

    unit = result["units"][0]
    assert unit["name"] == "Dietrick - Xpress Lane Market"
    assert unit["is_open"] is False
    assert unit["hours"] == []
    assert unit["menu"] == []
    assert fake.menu_calls == []


def test_get_dining_meal_filter_keeps_unattributed_menu_items():
    """Menu items without meal attribution stay included under a meal filter."""
    no_tabs_html = MENU_HTML.replace('href="#meal_1_content"', 'href="#x"').replace(
        'href="#meal_2_content"', 'href="#y"'
    )
    fake = FakeHttp(menu_body=no_tabs_html)
    result = get_dining(date="2026-06-11", location="D2", meal="Dinner", http_get=fake)

    unit = result["units"][0]
    assert len(unit["menu"]) == 4  # nothing attributable, so nothing dropped


# ── TTL cache ────────────────────────────────────────────────────────────────

def test_cache_second_call_within_ttl_does_not_refetch():
    fake = FakeHttp()
    clock = {"t": 1000.0}
    now = lambda: clock["t"]

    get_dining(date="2026-06-11", location="D2", http_get=fake, now=now)
    assert len(fake.hours_calls) == 1
    assert len(fake.menu_calls) == 1

    clock["t"] += 100  # still inside default 3600s TTL
    get_dining(date="2026-06-11", location="D2", http_get=fake, now=now)
    assert len(fake.hours_calls) == 1
    assert len(fake.menu_calls) == 1


def test_cache_expired_ttl_refetches():
    fake = FakeHttp()
    clock = {"t": 1000.0}
    now = lambda: clock["t"]

    get_dining(date="2026-06-11", location="D2", http_get=fake, now=now, ttl=60)
    clock["t"] += 61
    get_dining(date="2026-06-11", location="D2", http_get=fake, now=now, ttl=60)

    assert len(fake.hours_calls) == 2
    assert len(fake.menu_calls) == 2


def test_cache_keyed_by_date():
    fake = FakeHttp()
    get_dining(date="2026-06-11", location="D2", http_get=fake)
    get_dining(date="2026-06-12", location="D2", http_get=fake)

    assert len(fake.hours_calls) == 2
    assert len(fake.menu_calls) == 2


# ── graceful degradation ─────────────────────────────────────────────────────

def test_get_dining_hours_http_error_degrades_gracefully():
    def boom(url: str) -> str:
        raise ConnectionError("network down")

    result = get_dining(date="2026-06-11", http_get=boom)

    assert result["date"] == "2026-06-11"
    assert result["units"] == []
    assert "error" in result


def test_get_dining_menu_http_error_keeps_unit_with_hours():
    def hours_only(url: str) -> str:
        if "UnitsOpenOnDay" in url:
            return HOURS_JSON
        raise ConnectionError("foodpro down")

    result = get_dining(date="2026-06-11", location="D2", http_get=hours_only)

    assert "error" not in result
    unit = result["units"][0]
    assert unit["is_open"] is True
    assert len(unit["hours"]) == 3
    assert unit["menu"] == []


# ── maybe_answer_dining (pre-router seam) ────────────────────────────────────

def test_maybe_answer_dining_fires_on_dining_query():
    fake = FakeHttp()
    result = maybe_answer_dining("What's on the menu at D2 today?", http_get=fake)

    assert result is not None
    assert [u["name"] for u in result["units"]] == ["Dietrick - D2"]


def test_maybe_answer_dining_extracts_meal():
    fake = FakeHttp()
    result = maybe_answer_dining("What is Turner serving for lunch?", http_get=fake)

    assert result is not None
    unit = result["units"][0]
    assert "Turner" in unit["name"]
    assert all(h["label"] == "Lunch" for h in unit["hours"])


def test_maybe_answer_dining_extracts_relative_and_explicit_dates():
    fake = FakeHttp()
    now = lambda: 1781227800.0  # 2026-06-11 Eastern

    maybe_answer_dining("Is D2 open tomorrow?", http_get=fake, now=now)
    maybe_answer_dining("What is D2 serving on 2026-06-15?", http_get=fake, now=now)
    maybe_answer_dining("What is D2 serving June 16?", http_get=fake, now=now)

    assert "2026-06-12" in fake.hours_calls[0]
    assert "2026-06-15" in fake.hours_calls[1]
    assert "2026-06-16" in fake.hours_calls[2]


def test_maybe_answer_dining_fires_on_generic_dining_words():
    fake = FakeHttp()
    result = maybe_answer_dining("Which dining halls are open right now?", http_get=fake)

    assert result is not None
    assert len(result["units"]) == 3
    assert fake.menu_calls == []
    assert "specific" in result["note"]


def test_maybe_answer_dining_ignores_unrelated_query():
    def must_not_call(url: str) -> str:
        raise AssertionError("network must not be touched for non-dining queries")

    assert maybe_answer_dining("Who is the head of the CS department?", http_get=must_not_call) is None
    assert maybe_answer_dining("What are the MS admission deadlines?", http_get=must_not_call) is None


def test_maybe_answer_dining_ignores_dining_services_non_lookup_queries():
    def must_not_call(url: str) -> str:
        raise AssertionError("network must not be touched for non-live dining queries")

    assert maybe_answer_dining("How do I apply for dining services jobs?", http_get=must_not_call) is None
    assert maybe_answer_dining("Who manages VT dining services employment?", http_get=must_not_call) is None
    assert maybe_answer_dining("What are dining services policies?", http_get=must_not_call) is None
    assert maybe_answer_dining("Are dining services jobs available?", http_get=must_not_call) is None


def test_maybe_answer_dining_does_not_fire_on_substrings():
    def must_not_call(url: str) -> str:
        raise AssertionError("network must not be touched")

    # "great"/"theater" contain "eat"; word boundaries must prevent firing
    assert maybe_answer_dining("Is the theater program great?", http_get=must_not_call) is None


# ── format_dining_context ────────────────────────────────────────────────────

def test_format_dining_context_labels_live_data_with_timestamp():
    fake = FakeHttp()
    result = get_dining(date="2026-06-11", location="D2", http_get=fake, now=lambda: 1781227800.0)
    text = format_dining_context(result)

    assert "LIVE VT DINING DATA" in text
    assert "2026-06-11" in text
    assert "Dietrick - D2" in text
    assert "Breakfast" in text
    assert "Bacon" in text
    assert "vegan" in text


# ── DINING_TOOL_SCHEMA ───────────────────────────────────────────────────────

def test_dining_tool_schema_shape():
    assert DINING_TOOL_SCHEMA["type"] == "function"
    fn = DINING_TOOL_SCHEMA["function"]
    assert fn["name"] == "get_dining"
    assert fn["description"]
    params = fn["parameters"]
    assert params["type"] == "object"
    assert set(params["properties"]) == {"date", "location", "meal"}
    assert params.get("required", []) == []  # all three are optional


# ── live smoke test (skipped by default; run with -m live) ──────────────────

@pytest.mark.live
def test_live_smoke_hours_and_menu():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    result = get_dining(date=today)

    assert result["date"] == today
    assert "error" not in result, result.get("error")
    # roster may be small in summer/breaks, but the shape must hold
    for unit in result["units"]:
        assert unit["name"]
        assert isinstance(unit["hours"], list)
        assert isinstance(unit["menu"], list)


# ── app wiring: pre-router injection into the chat flow ─────────────────────

CANNED_DINING_RESULT = {
    "date": "2026-06-11",
    "fetched_at": "2026-06-11T12:00-04:00",
    "units": [
        {
            "name": "Dietrick - D2",
            "is_open": True,
            "hours": [{"label": "Lunch", "open": "10:30:01", "close": "14:00:00"}],
            "menu": [{"name": "Garden Salad", "portion": "1 Cup", "dietary": ["vegetarian"]}],
        }
    ],
}


@pytest.fixture
def app_client():
    from unittest.mock import MagicMock, patch

    from fastapi.testclient import TestClient

    from chatbot.session_store import SessionStore

    retriever = MagicMock()
    retriever.search.return_value = [
        {
            "score": 0.85,
            "chunk_id": "abc_0000",
            "text": "Sally Hamouda is a professor of CS.",
            "url": "https://website.cs.vt.edu/faculty/sally.html",
            "title": "Sally Hamouda | CS | VT",
        }
    ]
    llm = MagicMock()
    llm.chat.return_value = "answer"
    llm.rewrite_query.side_effect = lambda q, h: q
    store = SessionStore(max_requests=1000, window_seconds=3600)
    with patch("chatbot.app.retriever", retriever), \
         patch("chatbot.app.llm_client", llm), \
         patch("chatbot.app._session_store", store):
        from chatbot.app import app

        yield TestClient(app, cookies={"hokiehelp_session": "dining-test-session"}), llm


def test_chat_injects_live_dining_context_for_dining_query(app_client):
    from unittest.mock import patch

    client, llm = app_client
    with patch("chatbot.app.maybe_answer_dining", return_value=CANNED_DINING_RESULT):
        resp = client.post("/chat", json={"question": "Is D2 open for lunch?", "history": []})

    assert resp.status_code == 200
    chunks = llm.chat.call_args[0][1]
    assert chunks[0]["chunk_id"] == "live_dining"
    assert "LIVE VT DINING DATA" in chunks[0]["text"]
    assert "Garden Salad" in chunks[0]["text"]
    # live dining surfaces as the top source
    assert resp.json()["sources"][0]["url"].startswith("https://dining.vt.edu")


def test_chat_does_not_inject_for_non_dining_query(app_client):
    from unittest.mock import patch

    client, llm = app_client
    with patch("chatbot.app.maybe_answer_dining", return_value=None):
        resp = client.post("/chat", json={"question": "Who is Sally Hamouda?", "history": []})

    assert resp.status_code == 200
    chunks = llm.chat.call_args[0][1]
    assert all(c.get("chunk_id") != "live_dining" for c in chunks)


def test_chat_survives_dining_seam_crash(app_client):
    from unittest.mock import patch

    client, llm = app_client
    with patch("chatbot.app.maybe_answer_dining", side_effect=RuntimeError("boom")):
        resp = client.post("/chat", json={"question": "Is D2 open?", "history": []})

    assert resp.status_code == 200  # seam failure must never break the request


def test_chat_stream_injects_live_dining_context_and_sources(app_client):
    from unittest.mock import patch

    client, llm = app_client
    llm.chat_stream.return_value = iter(["D2", " is", " open"])
    with patch("chatbot.app.maybe_answer_dining", return_value=CANNED_DINING_RESULT):
        resp = client.post("/chat/stream", json={"question": "Is D2 open for lunch?", "history": []})

    assert resp.status_code == 200
    chunks = llm.chat_stream.call_args[0][1]
    assert chunks[0]["chunk_id"] == "live_dining"
    body = resp.text
    assert '"type": "token"' in body
    assert '"type": "sources"' in body
    assert "https://dining.vt.edu/hours.html" in body
