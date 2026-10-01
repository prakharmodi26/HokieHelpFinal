"""Lossless-enough, ordered CourseLeaf requirements blocks for markdown cards.

Tables are data, not layout to discard. Keep codes, alternatives, section rows,
credits and footnote references; don't guess which elective is mandatory.
"""
from __future__ import annotations

import re
from bs4 import BeautifulSoup, Tag


def text(tag: Tag) -> str:
    copy = BeautifulSoup(str(tag), "html.parser")
    for sup in copy.find_all("sup"):
        sup.replace_with(" [note " + sup.get_text(" ", strip=True) + "] ")
    return re.sub(r"\s+", " ", copy.get_text(" ", strip=True)).strip()


def table_markdown(table: Tag) -> str:
    grid: list[list[str]] = []
    spanning: dict[int, tuple[str, int]] = {}
    header_rows: set[int] = set()
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:
            continue
        cells = tr.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        row: dict[int, str] = {}
        for column, (value, remaining) in list(spanning.items()):
            row[column] = value
            if remaining <= 1:
                del spanning[column]
            else:
                spanning[column] = (value, remaining - 1)
        column = 0
        for cell in cells:
            while column in row:
                column += 1
            value = text(cell).replace("|", "\\|")
            try:
                width = max(1, min(30, int(cell.get("colspan", 1))))
                height = max(1, min(100, int(cell.get("rowspan", 1))))
            except (ValueError, TypeError):
                width = height = 1
            for offset in range(width):
                row[column + offset] = value if offset == 0 else ""
                if height > 1:
                    spanning[column + offset] = (row[column + offset], height - 1)
            column += width
        if any(c.name == "th" for c in cells):
            header_rows.add(len(grid))
        grid.append([row.get(i, "") for i in range(max(row) + 1)])
    if not grid:
        return ""
    width = max(map(len, grid))
    rows = [r + [""] * (width - len(r)) for r in grid]
    if 0 not in header_rows:
        rows.insert(0, [f"Column {i + 1}" for i in range(width)])
    rendered = ["| " + " | ".join(r) + " |" for r in rows]
    rendered.insert(1, "| " + " | ".join(["---"] * width) + " |")
    caption = table.find("caption", recursive=False)
    return ((text(caption) + "\n\n") if caption else "") + "\n".join(rendered)


def requirements_blocks(container: Tag) -> list[str]:
    """Walk structural blocks once, in DOM order, not descendant text twice."""
    blocks: list[str] = []

    def walk(node: Tag, depth: int = 0):
        for child in node.children:
            if not isinstance(child, Tag) or child.name in {"script", "style", "nav"}:
                continue
            if child.name == "table":
                if block := table_markdown(child):
                    blocks.append(block)
            elif child.name in {"h2", "h3", "h4", "h5"}:
                if value := text(child):
                    blocks.append("#" * min(6, int(child.name[1]) + 1) + " " + value)
            elif child.name == "li":
                own = BeautifulSoup(str(child), "html.parser")
                for nested in own.find_all(["ul", "ol", "table"]):
                    nested.decompose()
                if value := text(own):
                    blocks.append("  " * depth + "- " + value)
                for nested in child.find_all(["ul", "ol", "table"], recursive=False):
                    if nested.name == "table":
                        blocks.append(table_markdown(nested))
                    else:
                        walk(nested, depth + 1)
            elif child.name in {"p", "dl"}:
                if value := text(child):
                    blocks.append(value)
            else:
                walk(child, depth)
    walk(container)
    return blocks
