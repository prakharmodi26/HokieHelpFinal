import yaml
import pytest
from unittest.mock import MagicMock

from crawler.courseleaf import (
    COURSE_INDEX_URL,
    PROGRAM_SEED_URLS,
    course_card_markdown,
    course_key,
    course_object_key,
    course_record_url,
    course_subject_url,
    harvest_courseleaf_catalog,
    parse_courseblocks,
    parse_programs,
    parse_subject_urls,
    program_card_markdown,
    program_object_key,
)


COURSE_HTML = """
<html>
  <body>
    <div id="content">
      <div class="courseblock">
        <p class="courseblocktitle">
          <strong>CS 3114: Data Structures and Algorithms (3 credits)</strong>
        </p>
        <p class="courseblockdesc">
          Advanced data structures and algorithm analysis with implementation.
        </p>
        <p class="courseblockextra noindent">
          Prerequisite(s): CS 2114 or CS 2505.
        </p>
        <p class="courseblockextra noindent">
          Corequisite(s): MATH 2534.
        </p>
        <p class="courseblockextra noindent">
          When Offered: Fall and Spring.
        </p>
        <p class="courseblockextra noindent">
          Cross-Listed: CMDA 3114.
        </p>
      </div>
      <div class="courseblock">
        <p class="courseblocktitle">
          <strong>CS 5804: Introduction to Artificial Intelligence (3 credits)</strong>
        </p>
        <p class="courseblockdesc">
          Search, knowledge representation, learning, and reasoning.
        </p>
        <p class="courseblockextra noindent">
          Prerequisite: Graduate standing.
        </p>
      </div>
    </div>
  </body>
</html>
"""


def _frontmatter(markdown: str) -> dict:
    _, raw, _body = markdown.split("---", 2)
    return yaml.safe_load(raw)


def test_parse_courseblocks_extracts_structured_fields():
    records = parse_courseblocks(
        COURSE_HTML,
        source_url="https://catalog.vt.edu/course-descriptions/cs/",
    )

    assert len(records) == 2
    cs3114 = records[0]
    assert cs3114.subject == "CS"
    assert cs3114.number == "3114"
    assert cs3114.level == "ug"
    assert cs3114.title == "Data Structures and Algorithms"
    assert cs3114.credits == "3"
    assert cs3114.prerequisites == "CS 2114 or CS 2505."
    assert cs3114.corequisites == "MATH 2534."
    assert cs3114.when_offered == "Fall and Spring."
    assert cs3114.cross_listings == ["CMDA 3114"]
    assert cs3114.description == (
        "Advanced data structures and algorithm analysis with implementation."
    )
    assert cs3114.key == "CS-3114"

    assert records[1].level == "grad"
    assert records[1].prerequisites == "Graduate standing."


def test_parse_courseblocks_handles_range_credits_and_letter_suffix():
    html = """
    <div class="courseblock">
      <p class="courseblocktitle">
        <strong>AAEC 1004H - First-Year Experience (1-2 credits)</strong>
      </p>
      <p class="courseblockdesc">Orientation to the major.</p>
    </div>
    """
    [record] = parse_courseblocks(html)
    assert record.subject == "AAEC"
    assert record.number == "1004H"
    assert record.level == "ug"
    assert record.title == "First-Year Experience"
    assert record.credits == "1-2"


def test_course_card_markdown_has_payload_frontmatter_and_semantic_body():
    record = parse_courseblocks(
        COURSE_HTML, "https://catalog.vt.edu/course-descriptions/cs/"
    )[0]
    markdown = course_card_markdown(record)
    data = _frontmatter(markdown)

    assert data["page_type"] == "course"
    assert len(data["doc_id"]) == 16
    assert data["key"] == "CS-3114"
    assert data["subject"] == "CS"
    assert data["course_number"] == "3114"
    assert data["course_level"] == "ug"
    assert data["credits"] == "3"
    assert data["prerequisites"] == "CS 2114 or CS 2505."
    assert data["corequisites"] == "MATH 2534."
    assert data["when_offered"] == "Fall and Spring."
    assert data["cross_listings"] == ["CMDA 3114"]
    assert data["url"] == "https://catalog.vt.edu/course-descriptions/cs#CS-3114"
    assert data["source_host"] == "catalog.vt.edu"
    assert data["subdomain"] == "catalog"
    assert len(data["content_hash"]) == 64
    assert "# CS 3114: Data Structures and Algorithms" in markdown
    assert "**Prerequisites:** CS 2114 or CS 2505." in markdown


def test_course_helpers_are_stable():
    record = parse_courseblocks(COURSE_HTML)[0]
    assert course_key("cs", "3114") == "CS-3114"
    assert course_subject_url("CS") == "https://catalog.vt.edu/course-descriptions/cs/"
    assert course_record_url(record) == "https://catalog.vt.edu/course-descriptions/cs#CS-3114"
    assert course_object_key(record) == (
        "catalog.vt.edu/course-descriptions/cs/CS-3114.md"
    )


def test_parse_subject_urls_from_course_index():
    html = """
    <a href="/course-descriptions/cs/">Computer Science (CS)</a>
    <a href="/course-descriptions/aaec/">Agricultural Economics (AAEC)</a>
    <a href="/course-descriptions/">Course Descriptions</a>
    """

    assert parse_subject_urls(html) == [
        "https://catalog.vt.edu/course-descriptions/cs/",
        "https://catalog.vt.edu/course-descriptions/aaec/",
    ]


PROGRAM_LISTING_HTML = """
<html>
  <body>
    <main id="content">
      <h2>College of Engineering</h2>
      <ul class="program-list">
        <li>
          <a href="/undergraduate/college-engineering/computer-science/computer-science-bs/">
            Computer Science Major, B.S.
          </a>
        </li>
        <li>
          <a href="/graduate/college-engineering/computer-science/computer-science-ms/">
            Computer Science, M.S.
          </a>
        </li>
      </ul>
    </main>
  </body>
</html>
"""


def test_parse_program_listing_cards():
    records = parse_programs(PROGRAM_LISTING_HTML, "https://catalog.vt.edu/program-explorer/")

    assert len(records) == 2
    assert records[0].title == "Computer Science Major, B.S."
    assert records[0].url == (
        "https://catalog.vt.edu/undergraduate/college-engineering/"
        "computer-science/computer-science-bs/"
    )
    assert records[0].degree == "B.S."
    assert records[0].college == "College of Engineering"
    assert records[0].level == "undergraduate"
    assert records[1].level == "graduate"


PROGRAM_DETAIL_HTML = """
<html>
  <head><title>Computer Science Major, B.S.</title></head>
  <body>
    <main id="content">
      <h1>Computer Science Major, B.S.</h1>
      <p>
        Computer science students learn software development, systems, and
        computational theory.
      </p>
      <p>Graduates work across computing, data, and research roles.</p>
      <h2>Degree Requirements</h2>
      <ul>
        <li>Minimum 120 credit hours required for graduation.</li>
        <li>Major requirements include CS, math, and technical electives.</li>
      </ul>
    </main>
  </body>
</html>
"""


def test_parse_program_detail_extracts_description_and_requirements():
    [record] = parse_programs(
        PROGRAM_DETAIL_HTML,
        "https://catalog.vt.edu/undergraduate/college-engineering/"
        "computer-science/computer-science-bs/",
    )

    assert record.title == "Computer Science Major, B.S."
    assert record.degree == "B.S."
    assert record.level == "undergraduate"
    assert "software development" in record.description
    assert "### Degree Requirements" in record.requirements
    assert "- Minimum 120 credit hours required for graduation." in record.requirements


def test_program_card_markdown_has_program_page_type_and_fields():
    [record] = parse_programs(
        PROGRAM_DETAIL_HTML,
        "https://catalog.vt.edu/undergraduate/college-engineering/"
        "computer-science/computer-science-bs/",
    )
    markdown = program_card_markdown(record)
    data = _frontmatter(markdown)

    assert data["page_type"] == "program"
    assert len(data["doc_id"]) == 16
    assert data["title"] == "Computer Science Major, B.S."
    assert data["source_host"] == "catalog.vt.edu"
    assert data["subdomain"] == "catalog"
    assert data["program_name"] == "Computer Science Major, B.S."
    assert data["degree_type"] == "B.S."
    assert data["program_level"] == "undergraduate"
    # Tables/requirements stay in the body: large YAML fields exceed the
    # deployed ingestor's bounded frontmatter parser.
    assert "requirements" not in data
    assert "### Degree Requirements" in markdown
    assert len(data["content_hash"]) == 64
    assert "# Computer Science Major, B.S." in markdown
    assert "## Requirements" in markdown
    assert program_object_key(record).endswith(
        "undergraduate-college-engineering-computer-science-computer-science-bs.md"
    )


@pytest.mark.asyncio
async def test_harvest_courseleaf_catalog_writes_course_and_program_cards():
    storage = MagicMock()
    program_url = (
        "https://catalog.vt.edu/undergraduate/college-engineering/"
        "computer-science/computer-science-bs/"
    )
    html_by_url = {
        COURSE_INDEX_URL: '<a href="/course-descriptions/cs/">CS</a>',
        "https://catalog.vt.edu/course-descriptions/cs/": COURSE_HTML,
        PROGRAM_SEED_URLS[0]: PROGRAM_LISTING_HTML,
        PROGRAM_SEED_URLS[1]: "",
        PROGRAM_SEED_URLS[2]: "",
        program_url: PROGRAM_DETAIL_HTML,
        "https://catalog.vt.edu/graduate/college-engineering/computer-science/computer-science-ms/": "",
    }

    async def fetch_html(urls):
        return {url: html_by_url[url] for url in urls if url in html_by_url}

    stats = await harvest_courseleaf_catalog(
        storage, "crawled-pages-cleaned", fetch_html=fetch_html,
    )

    assert stats["courses_harvested"] == 2
    assert stats["programs_harvested"] == 1
    calls = storage.upload_document.call_args_list
    keys = [call.args[0] for call in calls]
    assert "catalog.vt.edu/course-descriptions/cs/CS-3114.md" in keys
    assert any(key.startswith("catalog.vt.edu/programs/") for key in keys)
    assert all(call.kwargs["bucket"] == "crawled-pages-cleaned" for call in calls)
