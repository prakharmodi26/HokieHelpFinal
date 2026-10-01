import pytest

from chatbot.program_metadata import program_matches, program_scope, source_conflicts


def program(name="Computer Science Major", level="undergraduate", edition="2026-2027"):
    return {"page_type": "program", "program_name": name,
            "program_level": level, "catalog_edition": edition}


@pytest.mark.parametrize("query, level", [
    ("undergraduate Computer Science major credits", "undergraduate"),
    ("graduate BS in Computer Science degree credits", "undergraduate"),
    ("graduate Computer Science degree requirements", "graduate"),
    ("Computer Science master's degree", "graduate"),
    ("Computer Science major", None),
    ("Compare undergraduate and master's Computer Science degrees", None),
    ("Compare undergraduate Computer Science and graduate Computer Science degrees", None),
    ("Compare graduate and undergraduate Computer Science degrees", None),
])
def test_level_scope(query, level):
    assert program_scope(query) == {"level": level}


@pytest.mark.parametrize("query", ["Parking permits", "Who teaches CS 1114?", "When is the deadline?", "Hi"])
def test_non_program_questions_have_no_lane(query):
    assert program_scope(query) is None


def test_explicit_full_program_identity_not_acronym():
    assert program_matches("undergraduate CS (Computer Science) BS degree checksheet credits", program())
    assert not program_matches("How many credits to graduate BS in CS?", program())
    assert not program_matches("undergraduate Computer Engineering major credits", program())
    assert not program_matches("undergraduate Computer Engineering major science electives", program())
    assert not program_matches("Computer and Science major credits", program())
    assert not program_matches("undergraduate Computer Science major credits", program("Dairy Science Major"))


def test_major_minor_and_option_distinctions():
    query = "Virginia Tech undergraduate mathematics major degree required credit hours"
    assert program_matches(query, program("Mathematics Major"))
    assert not program_matches(query, program("Mathematics Minor"))
    assert not program_matches(query, program("Mathematics Major with Applied Computational Mathematics Option"))
    assert program_matches("Mathematics major with Applied Computational Mathematics option credits", program("Mathematics Major with Applied Computational Mathematics Option"))
    assert not program_matches("Computer Science minor", program())
    assert program_matches("Computer Science minor", program("Computer Science (CS) Minor"))


def test_missing_or_wrong_metadata_not_a_priority_match():
    query = "undergraduate Computer Science major credits"
    assert not program_matches(query, {})
    assert not program_matches(query, {**program(), "page_type": "general"})
    assert not program_matches(query, {**program(), "program_name": ""})
    assert not program_matches(query, {**program(), "program_level": None})
    assert not program_matches(query, program(level="graduate"))


@pytest.mark.parametrize("edition", ["2023-2024", "2023–24", "2023/24"])
def test_explicit_edition_is_preserved(edition):
    query = "undergraduate Computer Science major credits for 2023–24"
    assert program_matches(query, program(edition=edition))
    assert not program_matches(query, program())
    assert not program_matches(query, program(edition=""))
    assert program_matches("undergraduate Computer Science major in 2023", program(edition=edition))


def test_comparison_can_match_both_named_programs():
    query = "Compare undergraduate Computer Science and mathematics major credit requirements"
    assert program_matches(query, program())
    assert program_matches(query, program("Mathematics Major"))
    assert not program_matches(query, program("Statistics Major"))


def test_source_conflicts_use_authoritative_identity_not_body_mentions():
    query = "Virginia Tech undergraduate Computer Science major required mathematics courses"
    assert source_conflicts(query, {**program("Statistics Major"), "text": "Computer Science is mentioned here"})
    assert source_conflicts(query, {"title": "Secure Computing Supporting Docs", "url": "https://registrar.vt.edu/updates/governance/secure_computing_supporting_docs.pdf"})
    assert source_conflicts(query, {"title": "CUSP 2025-2026C First Reading Final Revised", "url": "https://registrar.vt.edu/Updates/governance/2025-2026/CUSP.pdf"})
    assert source_conflicts(query, {"title": "Master of Engineering | students.cs | Virginia Tech", "url": "https://students.cs.vt.edu/Graduate/Degrees/MENG.html"})
    assert source_conflicts(query, {"title": "Requirements", "headings_path": ["Dairy Science Major"]})
    assert not source_conflicts(query, {"title": "Unlabeled Requirements PDF", "url": "https://vt.edu/files/requirements.pdf", "text": "Statistics Major"})


def test_source_conflicts_handle_degree_kind_year_and_codes_conservatively():
    query = "Virginia Tech undergraduate Computer Science major checksheet credits for 2023-24"
    assert not source_conflicts(query, {"title": "Computer Science Major Checksheet 2023 24", "url": "https://students.cs.vt.edu/checksheets/CS-Major-2023-24.pdf"})
    assert source_conflicts(query, {"title": "Computer Science Major Checksheet 2022", "url": "https://students.cs.vt.edu/checksheets/CS-Major-2022.pdf"})
    assert source_conflicts(query, {"title": "Computer Science Major Checksheet 2022 2023", "url": "https://students.cs.vt.edu/checksheets/CS-Major-2022-2023.pdf"})
    assert source_conflicts(query, {"title": "minor_math_23_24.pdf", "url": "https://registrar.vt.edu/Updates/minors/23-24/minor_math_23_24.pdf"})
    assert source_conflicts(query, {"title": "Cos Stat Smth 23 24", "url": "https://registrar.vt.edu/Updates/cos/23-24/cos_stat_smth_23_24.pdf"})
    assert not source_conflicts("Virginia Tech undergraduate Statistics major requirements", {"title": "Cos Stat Smth 23 24", "url": "https://registrar.vt.edu/Updates/cos/23-24/cos_stat_smth_23_24.pdf"})


def test_source_conflicts_allow_comparison_entities_and_unknown_sparse_sources():
    query = "Compare undergraduate Computer Science and Mathematics major credit requirements"
    assert not source_conflicts(query, program("Computer Science Major"))
    assert not source_conflicts(query, program("Mathematics Major"))
    assert source_conflicts(query, program("Statistics Major"))
    assert not source_conflicts(query, {"title": "Undergraduate Policies | Virginia Tech Academic Catalog", "url": "https://catalog.vt.edu/undergraduate/undergraduate-policies/"})
    assert not source_conflicts("Parking permits", {"title": "Statistics Major", "url": "https://catalog.vt.edu/statistics/"})
    assert not source_conflicts("What did the CUSP governance proposal say about mathematics?", {"title": "CUSP 2025-2026C First Reading Final Revised", "url": "https://registrar.vt.edu/Updates/governance/2025-2026/CUSP.pdf"})
    assert not source_conflicts("Virginia Tech undergraduate first-year admissions requirements", {"title": "Changes to 2026-2027 Catalog", "url": "https://registrar.vt.edu/Updates/governance/catalog-changes.pdf"})
