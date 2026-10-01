"""Conservative lexical checks on existing catalog metadata, not an ontology.

These checks reserve recall for explicitly named programs and reject explicit
wrong-source identities. They never infer a program from incidental body text,
treat an unknown field as a match, or verify claims. Sparse/unidentified PDFs
continue through general retrieval.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import unquote, urlsplit

_ACADEMIC = re.compile(r"\b(major|minor|degree|checksheet|graduation|graduate|bachelor'?s?|master'?s?|doctoral|undergraduate|phd)\b", re.I)
_UNDERGRAD = re.compile(r"\b(undergraduate|bachelor'?s?|bs|ba|bsc)\b", re.I)
_GRAD = re.compile(r"\b(graduate|master'?s?|doctoral|phd|ph\.?d\.?)\b", re.I)
_STOP = {"major", "minor", "option", "degree", "bs", "ba", "ms", "ma", "phd"}
_IDENTITY_STOP = _STOP | {
    "virginia", "tech", "vt", "undergraduate", "graduate", "bachelor", "bachelors",
    "master", "masters", "doctoral", "requirements", "requirement", "required",
    "credits", "credit", "hours", "hour", "total", "complete", "completion", "degree",
    "checksheet", "checksheets", "catalog", "edition", "curriculum", "courses", "course",
    "college", "department", "school", "program", "programs", "plan", "study",
    "and", "or", "of", "the", "for", "in", "to", "an", "what", "which", "how",
    "many", "do", "need", "does", "it", "are", "is", "at", "with", "about",
    "compare", "versus", "vs", "between",
}
_SOURCE_STOP = _IDENTITY_STOP | {
    "pdf", "html", "htm", "content", "dam", "website", "students", "registrar", "updates",
    "documents", "document", "supporting", "support", "docs", "background", "resolution",
    "first", "second", "reading", "final", "revised", "draft", "proposal", "proposals",
    "majorhandbook", "handbook", "catalogs", "catalogue", "www", "edu", "coe", "cos", "cusp",
}
_SOURCE_PREFIXES = {"coe", "cos", "cals", "cnre", "clahs", "cob", "caus", "minor", "minors", "majors", "checksheets"}
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?:\s*[-/–]\s*(\d{4}|\d{2}))?(?!\d)")
_TWO_DIGIT_YEAR = re.compile(r"(?<!\d)(\d{2})\s*[-_/]\s*(\d{2})(?!\d)")
_GOVERNANCE_SOURCE = re.compile(r"/governance/|\b(governance|resolution|supporting\s+docs?|supporting\s+documents?|background\s+doc|first\s+reading|second\s+reading|cusp|catalog\s+changes)\b", re.I)
_GOVERNANCE_QUERY = re.compile(r"\b(governance|resolution|proposal|supporting\s+docs?|supporting\s+documents?|catalog\s+changes|first\s+reading|second\s+reading|cusp)\b", re.I)
_PROGRAM_SPECIFIC = re.compile(r"\b(major|minor|checksheet|degree|bachelor'?s?|bs|ba|master'?s?|phd|credits?|credit\s+hours?|curriculum)\b", re.I)


@dataclass
class _SourceIdentity:
    phrases: list[tuple[str, ...]] = field(default_factory=list)
    codes: set[str] = field(default_factory=set)
    kind: str | None = None
    level: str | None = None
    years: set[int] = field(default_factory=set)
    governance: bool = False
    specialized: bool = False


def program_scope(query: str) -> dict | None:
    """An optional catalog lane; never a global filter on all evidence."""
    if not _ACADEMIC.search(query):
        return None
    undergraduate = bool(_UNDERGRAD.search(query))
    # 'graduate' is often a verb in bachelor's questions, not a degree level.
    graduate = bool(re.search(r"\b(master'?s?|doctoral|phd|ph\.?d\.?)\b", query, re.I))
    if not undergraduate:
        graduate = graduate or bool(_GRAD.search(query))
    elif re.search(r"\b(?:and|versus|vs|with)\s+graduate\b|\bgraduate\b.*\b(?:and|versus|vs|with)\s+undergraduate\b", query, re.I):
        graduate = True
    level = "undergraduate" if undergraduate and not graduate else "graduate" if graduate and not undergraduate else None
    return {"level": level}


def _years(text: str) -> set[int]:
    years = set()
    for match in _YEAR.finditer(text):
        first = int(match[1])
        years.add(first)
        if match[2]:
            second = int(match[2])
            if second < 100:
                second += first // 100 * 100
            years.add(second)
    for match in _TWO_DIGIT_YEAR.finditer(text):
        first = int(match[1])
        second = int(match[2])
        if 80 <= first <= 99:
            base = 1900
        else:
            base = 2000
        years.update({base + first, base + second})
    return years


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", unquote(str(text)).casefold())


def _identity_tokens(text: str, *, source: bool = False) -> list[str]:
    stop = _SOURCE_STOP if source else _IDENTITY_STOP
    return [word for word in _tokens(text)
            if len(word) > 1 and not word.isdigit() and word not in stop and not re.fullmatch(r"\d{2,4}", word)]


def _strip_aliases(name: str) -> str:
    # Remove only uppercase acronym aliases, not substantive option qualifiers.
    return re.sub(r"\([A-Z&]{2,8}\)", "", name)


def _trim_program_identity(candidate: str) -> str:
    """Drop trailing course-topic words after a clear program identity."""
    # Keep option/specialization qualifiers when they are the requested identity.
    specialized = list(re.finditer(r"\b(option|specialization|concentration|track)\b", candidate, re.I))
    if specialized:
        return candidate[:specialized[-1].end()]
    program = re.search(r"\b(major|minor)\b", candidate, re.I)
    if program:
        return candidate[:program.end()]
    return candidate


def _program_focus(query: str) -> str:
    """Keep a named degree subject separate from requested course topics.

    E.g. 'undergraduate Computer Science major required mathematics courses'
    names CS, not Mathematics. Unclear grammar falls back without inventing a
    subject. Comparisons retain all their named entities.
    """
    if re.search(r"\b(compare|comparison|versus|vs|difference|both)\b", query, re.I):
        return query
    stop = r"\b(checksheets?|required|requirements?|total|credits?|hours?|courses?|curriculum|electives?)\b"
    level = re.search(r"\b(?:undergraduate|graduate)\s+(.+)", query, re.I)
    if level:
        candidate = _trim_program_identity(re.split(stop, level[1], maxsplit=1, flags=re.I)[0])
        if _identity_tokens(candidate):
            return candidate
    for match in reversed(list(re.finditer(r"\b(?:for|in)\s+(?:the\s+)?(.+)", query, re.I))):
        candidate = _trim_program_identity(re.split(stop, match[1], maxsplit=1, flags=re.I)[0])
        if _identity_tokens(candidate):
            return candidate
    candidate = _trim_program_identity(re.split(stop, query, maxsplit=1, flags=re.I)[0])
    if re.search(r"\b(major|minor|degree|bachelor'?s?|master'?s?|bs|ba)\b", candidate, re.I):
        return candidate
    return query


def _query_terms(query: str) -> tuple[list[str], set[str]]:
    words = _identity_tokens(_program_focus(query))
    codes = set(words)
    codes.update(word[:4] for word in words if len(word) >= 6)
    for size in (2, 3):
        for i in range(len(words) - size + 1):
            codes.add("".join(word[0] for word in words[i:i + size]))
    return words, codes


def program_filter_names(query: str) -> list[str]:
    """Literal catalog program names safe enough for an exact metadata lane.

    This is a recall optimization, not an ontology: only unambiguous single
    major/minor identities are returned. Comparisons and specializations keep
    using the broader page_type/level lane.
    """
    focus = _program_focus(query)
    if re.search(r"\b(compare|comparison|versus|vs|difference|both|option|specialization|concentration|track)\b", focus, re.I):
        return []
    query_kind = _query_kind(query)
    if query_kind not in {"major", "minor"}:
        return []
    words = _identity_tokens(focus)
    if not words or any(word in {"available", "list"} for word in words):
        return []
    # Drop an acronym alias when the following words already spell it out
    # (e.g. 'CS Computer Science major' -> 'Computer Science Major').
    if len(words) >= 3 and 2 <= len(words[0]) <= 5:
        for size in range(min(5, len(words) - 1), 1, -1):
            if "".join(word[0] for word in words[1:1 + size]) == words[0]:
                words = words[1:]
                break
    if not words or all(len(word) <= 5 for word in words):
        return []
    name = " ".join(word.capitalize() for word in words)
    return [f"{name} {query_kind.capitalize()}"]


def _phrase_in_query(phrase: tuple[str, ...], query_words: list[str]) -> bool:
    if not phrase:
        return False
    haystack = " " + " ".join(query_words) + " "
    return " " + " ".join(phrase) + " " in haystack


def _kind(text: str) -> str | None:
    tokens = set(_tokens(text))
    if "minor" in tokens or "minors" in tokens:
        return "minor"
    if "major" in tokens or "majors" in tokens:
        return "major"
    if "option" in tokens or "options" in tokens:
        return "option"
    return None


def _query_kind(query: str) -> str | None:
    if len(set(_tokens(query)) & {"major", "minor", "option"}) > 1:
        return None  # comparison: do not arbitrarily pick one program type
    kind = _kind(query)
    if kind:
        return kind
    if re.search(r"\b(bachelor'?s?|bs|ba|bsc|degree|checksheet)\b", query, re.I):
        return "major"
    return None


def _level(text: str) -> str | None:
    if re.search(r"\b(master'?s?|doctoral|phd|ph\.?d\.?|graduate)\b|/graduate/", text, re.I):
        return "graduate"
    if re.search(r"\b(undergraduate|bachelor'?s?)\b|/undergraduate/", text, re.I):
        return "undergraduate"
    return None


def _append_phrase(identity: _SourceIdentity, words: list[str]) -> None:
    if words and tuple(words) not in identity.phrases:
        identity.phrases.append(tuple(words))


def _extract_title_phrase(text: str) -> tuple[list[str], str | None]:
    identity_marker = r"\b(?:major|minor|option|specialization|concentration|program|checksheet|checksheets|supporting\s+docs?|supporting\s+documents?)\b"
    # Site suffixes such as 'Public Health Program' identify a program page,
    # whereas 'Academic Policies' to the left of the pipe is merely its topic.
    parts = re.split(r"\s*[|—]\s*", text)
    target = next((part for part in parts if re.search(identity_marker, part, re.I)), text)
    kind = _kind(target)
    if not re.search(identity_marker, target, re.I):
        return [], None
    # A specialization is part of identity, not a suffix to discard. Otherwise
    # every 'Mathematics Major with ... Option' collapses to plain Mathematics.
    if re.search(r"\b(option|specialization|concentration|track)\b", target, re.I):
        cleaned = _strip_aliases(target)
    else:
        cleaned = re.split(identity_marker, target, maxsplit=1, flags=re.I)[0]
    return _identity_tokens(cleaned, source=True), kind


def _source_identity(chunk: dict) -> _SourceIdentity:
    identity = _SourceIdentity()
    identity.specialized = any(
        re.search(r"\b(option|specialization|concentration|track)\b", unquote(value).replace("_", " "), re.I)
        for value in (chunk.get("program_name"), chunk.get("title"), chunk.get("url"))
        if isinstance(value, str)
    )
    program_name = chunk.get("program_name")
    if isinstance(program_name, str) and program_name.strip():
        _append_phrase(identity, _identity_tokens(_strip_aliases(program_name), source=True))
        identity.kind = identity.kind or _kind(program_name)
    program_level = chunk.get("program_level")
    if isinstance(program_level, str) and program_level in {"undergraduate", "graduate"}:
        identity.level = program_level
    edition = chunk.get("catalog_edition")
    if isinstance(edition, str):
        identity.years.update(_years(edition))

    title = chunk.get("title")
    if isinstance(title, str):
        words, kind = _extract_title_phrase(title)
        if not identity.phrases:
            _append_phrase(identity, words)
        identity.kind = identity.kind or kind
        identity.level = identity.level or _level(title)
        identity.years.update(_years(title))
        identity.governance = identity.governance or bool(_GOVERNANCE_SOURCE.search(title))

    # Only the document-level heading can establish identity. A later heading
    # about another program's electives/minor is not the identity of this page.
    headings = chunk.get("headings_path")
    if not isinstance(headings, (list, tuple)):
        headings = []
    for heading in headings[:1] if not identity.phrases else []:
        if isinstance(heading, str):
            words, kind = _extract_title_phrase(heading)
            _append_phrase(identity, words)
            identity.kind = identity.kind or kind
            identity.level = identity.level or _level(heading)
            identity.years.update(_years(heading))
            identity.governance = identity.governance or bool(_GOVERNANCE_SOURCE.search(heading))

    url = chunk.get("url")
    if isinstance(url, str):
        decoded = unquote(url)
        identity.governance = identity.governance or bool(_GOVERNANCE_SOURCE.search(decoded))
        identity.level = identity.level or _level(decoded)
        identity.years.update(_years(decoded))
        try:
            path = urlsplit(decoded).path
        except ValueError:
            path = ""  # Invalid source URL is discarded by evidence validation.
        filename = path.rsplit("/", 1)[-1]
        stem = re.sub(r"\.[A-Za-z0-9]+$", "", filename)
        identity.kind = identity.kind or _kind(path)
        words, kind = _extract_title_phrase(stem.replace("_", " ").replace("-", " "))
        _append_phrase(identity, words)
        identity.kind = identity.kind or kind
        parts = [part.casefold() for part in re.split(r"[^A-Za-z0-9]+", stem) if part]
        if parts and (parts[0] in _SOURCE_PREFIXES or "checksheets" in decoded.casefold()):
            for part in parts:
                if part in _SOURCE_PREFIXES or part in _SOURCE_STOP or re.fullmatch(r"\d{2,4}", part):
                    continue
                if len(part) <= 5:
                    identity.codes.add(part)
    return identity


def source_identity_matches(query: str, chunk: dict) -> bool:
    """An explicit match used to anchor filtering of the whole candidate pool."""
    identity = _source_identity(chunk)
    words, codes = _query_terms(query)
    return (any(_phrase_in_query(phrase, words) for phrase in identity.phrases)
            or bool(not identity.specialized and identity.codes.intersection(codes)))


def source_conflicts(query: str, chunk: dict, *, identity_anchored: bool = True) -> bool:
    """Whether authoritative source identity conflicts with an academic query.

    Only metadata, title, URL, and headings participate. Passage body mentions do
    not make a document about that entity. Unknown/sparse identity is allowed.
    """
    scope = program_scope(query)
    if scope is None:
        return False
    identity = _source_identity(chunk)
    query_words, query_codes = _query_terms(query)

    if scope["level"] and identity.level and identity.level != scope["level"]:
        return True

    query_kind = _query_kind(query)
    if query_kind and identity.kind and identity.kind != query_kind:
        # Treat option pages as major-specific rather than as wrong-degree when
        # the question only asks for the major; entity matching below handles the
        # stricter option wording for priority reservation.
        if not (query_kind == "major" and identity.kind == "option"):
            return True

    requested_years = _years(query)
    # Multiple separate year expressions can be a comparison, not one cohort.
    # Do not require every compared edition to occur on each individual source.
    if requested_years and identity.years and len(list(_YEAR.finditer(query))) <= 1:
        if len(requested_years) > 1:
            if not requested_years.issubset(identity.years):
                return True
        elif requested_years.isdisjoint(identity.years):
            return True

    if identity.governance and _PROGRAM_SPECIFIC.search(query) and not _GOVERNANCE_QUERY.search(query):
        return True

    if not query_words or not identity_anchored:
        return False
    has_entity = bool(identity.phrases or identity.codes)
    if not has_entity:
        return False
    if any(_phrase_in_query(phrase, query_words) for phrase in identity.phrases):
        return False
    if not identity.specialized and identity.codes and identity.codes.intersection(query_codes):
        return False
    return True


def program_matches(query: str, chunk: dict) -> bool:
    scope = program_scope(query)
    name = chunk.get("program_name")
    if scope is None or chunk.get("page_type") != "program" or not isinstance(name, str) or not name.strip():
        return False
    if scope["level"] and chunk.get("program_level") != scope["level"]:
        return False
    requested_years = _years(query)
    if requested_years and not requested_years.issubset(_years(str(chunk.get("catalog_edition") or ""))):
        return False
    focus = _program_focus(query)
    query_words = set(re.findall(r"[a-z0-9]+", focus.casefold()))
    name = _strip_aliases(name)
    name_words = set(re.findall(r"[a-z0-9]+", name.casefold()))
    qualifiers = {"option", "specialization", "concentration", "track"}
    if query_words & qualifiers and not name_words & qualifiers:
        return False  # Generic base-major page is not a specialization match.
    if "minor" in name_words and "minor" not in query_words:
        return False
    if "minor" in query_words and "major" not in query_words and "minor" not in name_words:
        return False
    identity = [word for word in re.findall(r"[a-z0-9]+", name.casefold())
                if word not in _STOP and len(word) > 1]
    query_tokens = [word for word in re.findall(r"[a-z0-9]+", focus.casefold())
                    if word not in _STOP and len(word) > 1]
    # Ordered phrase matching avoids treating 'Computer Engineering science
    # electives' as an explicit mention of the Computer Science program.
    return bool(identity) and " " + " ".join(identity) + " " in " " + " ".join(query_tokens) + " "
