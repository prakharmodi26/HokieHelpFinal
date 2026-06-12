"""VT dining runtime tool — live open/closed status, hours, and menus.

Sources (live feeds; the source of truth is never crawled or cached long-term):
- Hours/status JSON: apps.students.vt.edu/hours/Api/NonRestricted/UnitsOpenOnDay/Date/<YYYY-MM-DD>
  Each unit carries name, hours[] (label/open_time/close_time/title) and urls[] —
  one of which is the FoodPro menu URL containing its locationNum (the join key).
  A unit absent from the day's response is closed that day.
- Menus HTML (server-rendered): foodpro.students.vt.edu/menus/MenuAtLocation.aspx
  ?locationNum=<N>&naFlag=1[&dtdate=MM/DD/YYYY] — parsed via CSS classes
  recipe_title / recipe_description / portion_size / legend_icon.

dining.vt.edu/menus.html and hours.html are JS shells over these feeds — never scrape them.
"""
from __future__ import annotations

import json
import logging
import calendar
import re
import time
from datetime import datetime, timedelta
from typing import Callable
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HOURS_API_URL = (
    "https://apps.students.vt.edu/hours/Api/NonRestricted/UnitsOpenOnDay/Date/{date}"
)
MENU_URL = (
    "https://foodpro.students.vt.edu/menus/MenuAtLocation.aspx"
    "?locationNum={location_num}&naFlag=1&dtdate={dtdate}"
)

EASTERN = ZoneInfo("America/New_York")
DEFAULT_TTL_SECONDS = 3600.0  # 1h; plan allows 1-6h
HTTP_TIMEOUT_SECONDS = 15.0
# Menus are ~2MB FoodPro pages. Fetch them only after a location filter narrows
# the query; broad dining queries stay responsive and return hours-only.

HttpGet = Callable[[str], str]

_LOCATION_NUM_RE = re.compile(r"MenuAtLocation\.aspx[^\"']*?locationNum=(\d+)", re.I)
# In-process TTL caches: {date: (fetched_at, units)} / {(date, locationNum): (fetched_at, items)}
_hours_cache: dict[str, tuple[float, list[dict]]] = {}
_menu_cache: dict[tuple[str, str], tuple[float, list[dict]]] = {}


def clear_cache() -> None:
    """Drop all cached hours/menu data (test seam)."""
    _hours_cache.clear()
    _menu_cache.clear()


def _default_http_get(url: str) -> str:
    resp = httpx.get(url, timeout=HTTP_TIMEOUT_SECONDS, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


# ── parsing ──────────────────────────────────────────────────────────────────

def parse_hours(payload: object) -> list[dict]:
    """Parse the UnitsOpenOnDay JSON into units with hours and the FoodPro join key.

    Returns [{"name", "location_num": str|None, "hours": [{"label","open","close"}]}].
    Units absent from the payload are closed that day (the roster is date-scoped).
    """
    if isinstance(payload, dict):  # tolerate a wrapping object
        payload = next((v for v in payload.values() if isinstance(v, list)), [])
    units: list[dict] = []
    for raw in payload or []:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        location_num = None
        for entry in raw.get("urls") or []:
            url = entry.get("url", "") if isinstance(entry, dict) else str(entry)
            match = _LOCATION_NUM_RE.search(url)
            if match:
                location_num = match.group(1)
                break
        hours = [
            {
                "label": h.get("label") or h.get("title") or "",
                "open": h.get("open_time", ""),
                "close": h.get("close_time", ""),
            }
            for h in raw.get("hours") or []
            if isinstance(h, dict)
        ]
        units.append({"name": raw["name"], "location_num": location_num, "hours": hours})
    return units


def parse_menu(html: str) -> list[dict]:
    """Parse a FoodPro MenuAtLocation page into menu items.

    Returns [{"name", "portion": str|None, "dietary": [...], "meal": str|None}].
    Meal attribution comes from the bootstrap tab pane each recipe sits in,
    joined to the nav-tab labels (Breakfast/Lunch/Dinner); None if unattributable.
    """
    soup = BeautifulSoup(html, "html.parser")

    meal_by_pane: dict[str, str] = {}
    for anchor in soup.select('a[href^="#meal_"]'):
        pane_id = (anchor.get("href") or "").lstrip("#")
        label = anchor.get_text(strip=True)
        if pane_id and label:
            meal_by_pane[pane_id] = label

    items: list[dict] = []
    for container in soup.select(".recipe_container"):
        title_el = container.select_one(".recipe_title")
        if title_el is None:
            continue
        name = title_el.get_text(strip=True)
        if not name:
            continue
        portion_el = container.select_one(".portion_size")
        portion = portion_el.get_text(strip=True) if portion_el else None
        dietary = []
        for icon in container.select(".legend_icon"):
            dietary.extend(c for c in (icon.get("class") or []) if c != "legend_icon")
        meal = None
        pane = container.find_parent(class_="tab-pane")
        if pane is not None:
            meal = meal_by_pane.get(pane.get("id") or "")
        items.append({"name": name, "portion": portion, "dietary": dietary, "meal": meal})
    return items


# ── cached fetchers ──────────────────────────────────────────────────────────

def _get_hours_cached(
    date: str, http_get: HttpGet, now: Callable[[], float], ttl: float
) -> list[dict]:
    t = now()
    hit = _hours_cache.get(date)
    if hit is not None and t - hit[0] < ttl:
        return hit[1]
    units = parse_hours(json.loads(http_get(HOURS_API_URL.format(date=date))))
    _hours_cache[date] = (t, units)
    return units


def _get_menu_cached(
    date: str, location_num: str, http_get: HttpGet, now: Callable[[], float], ttl: float
) -> list[dict]:
    t = now()
    key = (date, location_num)
    hit = _menu_cache.get(key)
    if hit is not None and t - hit[0] < ttl:
        return hit[1]
    year, month, day = date.split("-")  # date is YYYY-MM-DD; FoodPro expects MM/DD/YYYY
    dtdate = f"{month}/{day}/{year}"
    url = MENU_URL.format(location_num=location_num, dtdate=dtdate)
    items = parse_menu(http_get(url))
    _menu_cache[key] = (t, items)
    return items


# ── tool entrypoint ──────────────────────────────────────────────────────────

def get_dining(
    date: str | None = None,
    location: str | None = None,
    meal: str | None = None,
    *,
    http_get: HttpGet | None = None,
    now: Callable[[], float] | None = None,
    ttl: float = DEFAULT_TTL_SECONDS,
) -> dict:
    """Tool entrypoint for the chatbot. Returns structured open/closed + hours + menu.

    {"date","units":[{"name","is_open","hours":[{"label","open","close"}],
      "menu":[{"name","portion","dietary":[...]}]}]}

    date defaults to today in America/New_York (VT is Eastern). location is a
    case-insensitive substring filter on unit names. meal filters hours/menu by
    label; menu items without meal attribution stay included only when the unit
    has matching meal hours. Never raises — network failures degrade to
    {"date", "units": [], "error": "..."}.
    """
    http_get = http_get or _default_http_get
    now = now or time.time
    if date is None:
        date = datetime.fromtimestamp(now(), tz=EASTERN).date().isoformat()
    fetched_at = datetime.fromtimestamp(now(), tz=EASTERN).isoformat(timespec="minutes")

    try:
        units = _get_hours_cached(date, http_get, now, ttl)
    except Exception as exc:
        logger.warning("DINING hours fetch failed for %s: %s", date, exc)
        return {
            "date": date,
            "units": [],
            "error": f"Could not reach the VT dining hours service: {exc}",
            "fetched_at": fetched_at,
        }

    if location:
        needle = location.lower()
        units = [u for u in units if needle in u["name"].lower()]

    fetch_menus = bool(location)
    meal_lower = meal.lower() if meal else None

    result_units: list[dict] = []
    for unit in units:
        hours = unit["hours"]
        if meal_lower:
            hours = [h for h in hours if meal_lower in (h.get("label") or "").lower()]
        is_open = bool(hours) if meal_lower else True

        menu: list[dict] = []
        if is_open and fetch_menus and unit.get("location_num"):
            try:
                menu = _get_menu_cached(date, unit["location_num"], http_get, now, ttl)
            except Exception as exc:
                logger.warning(
                    "DINING menu fetch failed for %s (locationNum=%s): %s",
                    unit["name"], unit["location_num"], exc,
                )
        if meal_lower:
            menu = [
                i for i in menu
                if i.get("meal") is None or meal_lower in i["meal"].lower()
            ]

        result_units.append({
            "name": unit["name"],
            "is_open": is_open,
            "hours": hours,
            "menu": [
                {"name": i["name"], "portion": i["portion"], "dietary": i["dietary"]}
                for i in menu
            ],
        })

    result: dict = {"date": date, "units": result_units, "fetched_at": fetched_at}
    if result_units and not fetch_menus:
        result["note"] = (
            "Menus omitted (many locations open) — ask about a specific "
            "dining location to see its menu."
        )
    return result


# ── function-calling tool schema (Ollama-compatible) ────────────────────────

DINING_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_dining",
        "description": (
            "Get live Virginia Tech dining information: which dining locations are "
            "open on a date, their hours (breakfast/lunch/dinner), and their menus "
            "with portion sizes and vegan/vegetarian flags. Locations absent from "
            "the result are closed that day."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD. Defaults to today (US Eastern).",
                },
                "location": {
                    "type": "string",
                    "description": (
                        "Filter to dining locations whose name contains this "
                        "substring (case-insensitive), e.g. 'D2', 'Turner', 'Owens'."
                    ),
                },
                "meal": {
                    "type": "string",
                    "description": "Filter hours and menu to one meal: Breakfast, Lunch, or Dinner.",
                },
            },
            "required": [],
        },
    },
}


# ── deterministic pre-router seam ────────────────────────────────────────────

_HALL_NAMES = {
    r"\bd2\b": "D2",
    r"\bdietrick\b": "Dietrick",
    r"\bturner\b": "Turner",
    r"\bwest\s+end\b": "West End",
    r"\bowens\b": "Owens",
    r"\bperry\s+place\b": "Perry Place",
    r"\bsquires\b": "Squires",
    r"\bhokie\s+grill\b": "Hokie Grill",
    r"\bdxpress\b": "DXpress",
    r"\bdeet'?s\b": "Deet's",
}
_DINING_CONTEXT_RE = re.compile(
    r"\b(dining(\s+halls?| centers?| locations?| services?)?|menus?|"
    r"food\s+courts?|meal\s+plans?|breakfast|brunch|lunch|dinner)\b",
    re.I,
)
_LIVE_LOOKUP_RE = re.compile(
    r"\b(open|closed|hours?|menus?|serv(e|ing|ed)|available|today|tonight|"
    r"now|breakfast|brunch|lunch|dinner|eat)\b",
    re.I,
)
_NON_LOOKUP_RE = re.compile(
    r"\b(job|jobs|employment|career|careers|hiring|apply|application|"
    r"nutritionist|dietitian|policy|policies|contact|director|staff)\b",
    re.I,
)
_MEAL_RE = re.compile(r"\b(breakfast|brunch|lunch|dinner)\b", re.I)
_ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_MONTH_NAMES = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTH_NAMES.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
_MONTH_RE = re.compile(
    r"\b("
    + "|".join(sorted(map(re.escape, _MONTH_NAMES), key=len, reverse=True))
    + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(20\d{2}))?\b",
    re.I,
)


def _today(now: Callable[[], float]) -> datetime.date:
    return datetime.fromtimestamp(now(), tz=EASTERN).date()


def _date_from_query(query: str, now: Callable[[], float]) -> str | None:
    """Extract simple date references from a dining query."""
    iso = _ISO_DATE_RE.search(query)
    if iso:
        return iso.group(1)

    q = query.lower()
    today = _today(now)
    if re.search(r"\btoday\b", q):
        return today.isoformat()
    if re.search(r"\btomorrow\b", q):
        return (today + timedelta(days=1)).isoformat()

    month_match = _MONTH_RE.search(query)
    if not month_match:
        return None
    month = _MONTH_NAMES[month_match.group(1).rstrip(".").lower()]
    day = int(month_match.group(2))
    year = int(month_match.group(3)) if month_match.group(3) else today.year
    try:
        parsed = today.replace(year=year, month=month, day=day)
    except ValueError:
        return None
    return parsed.isoformat()


def maybe_answer_dining(query: str, **kwargs) -> dict | None:
    """Deterministic pre-router: detect dining intent and answer with live data.

    Returns the structured get_dining() result when the query asks for live
    dining availability/menu data, else None. Extra kwargs (http_get/now/ttl)
    pass through to get_dining for tests.
    """
    location = None
    for pattern, canonical in _HALL_NAMES.items():
        if re.search(pattern, query, re.I):
            location = canonical
            break

    live_lookup = bool(_LIVE_LOOKUP_RE.search(query))
    dining_context = bool(_DINING_CONTEXT_RE.search(query))

    if _NON_LOOKUP_RE.search(query):
        return None
    if location is None and not (dining_context and live_lookup):
        return None
    if location is not None and not (live_lookup or dining_context):
        return None

    meal_match = _MEAL_RE.search(query)
    meal = meal_match.group(1).capitalize() if meal_match else None
    now = kwargs.get("now") or time.time
    date = _date_from_query(query, now)

    return get_dining(date=date, location=location, meal=meal, **kwargs)


# ── prompt-context formatter ─────────────────────────────────────────────────

_MAX_MENU_ITEMS_IN_CONTEXT = 50  # per unit; keeps the prompt bounded


def format_dining_context(result: dict) -> str:
    """Render a get_dining() result as labeled text for LLM grounding."""
    lines = [
        f"LIVE VT DINING DATA for {result['date']} "
        f"(fetched {result.get('fetched_at', 'just now')} US Eastern). "
        "Dining locations not listed below are CLOSED on this date."
    ]
    if result.get("error"):
        lines.append(f"ERROR: {result['error']}")
    if result.get("note"):
        lines.append(result["note"])
    if not result.get("units") and not result.get("error"):
        lines.append("No dining locations matching the request are open on this date.")

    for unit in result.get("units", []):
        if unit.get("is_open", True):
            lines.append(f"\n{unit['name']} — OPEN on {result['date']}.")
        else:
            lines.append(f"\n{unit['name']} — not open for the requested meal on {result['date']}.")
        for h in unit["hours"]:
            lines.append(f"  {h['label']}: {h['open']} to {h['close']}")
        menu = unit.get("menu") or []
        if menu:
            lines.append("  Menu:")
            for item in menu[:_MAX_MENU_ITEMS_IN_CONTEXT]:
                flags = f" [{', '.join(item['dietary'])}]" if item["dietary"] else ""
                portion = f" ({item['portion']})" if item.get("portion") else ""
                lines.append(f"    - {item['name']}{portion}{flags}")
            if len(menu) > _MAX_MENU_ITEMS_IN_CONTEXT:
                lines.append(f"    ...and {len(menu) - _MAX_MENU_ITEMS_IN_CONTEXT} more items")
    return "\n".join(lines)
