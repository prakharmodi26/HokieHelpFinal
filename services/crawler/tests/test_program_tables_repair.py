"""Table/source-card regressions against the actual deployed crawler API."""
from bs4 import BeautifulSoup
from crawler.courseleaf import parse_program_detail, program_card_markdown, detect_catalog_edition, program_requirement_containers
from crawler.program_content import requirements_blocks, table_markdown

URL = 'https://catalog.vt.edu/undergraduate/college-engineering/computer-science/computer-science-bs/'


def html(body):
    return '<h1 id="site-title">2026-2027 Academic Catalog</h1><h1 class="page-title">Computer Science Major</h1><main id="contentarea"><div id="requirementstextcontainer">' + body + '</div></main>'


def test_required_courses_alternatives_credits_and_footnotes():
    source = html('''<h2>Program Curriculum</h2><table><tr><th>Course</th><th>Title</th><th>Credits</th></tr>
      <tr><td>MATH 1225</td><td>Calculus</td><td>4</td></tr>
      <tr><td>MATH 1226</td><td>Calculus II</td><td>4</td></tr>
      <tr><td>MATH 2114</td><td>Linear Algebra</td><td>3<sup>1</sup></td></tr>
      <tr><td>CS 1114</td><td>Introduction to Software Design</td><td>3</td></tr>
      <tr><td colspan="2">Choose one statistics elective:</td><td>3</td></tr>
      <tr><td>STAT 4705</td><td>or STAT 4714</td><td></td></tr>
      <tr><td colspan="2">Total Credits</td><td>123</td></tr></table>
      <p><sup>1</sup>Consult an advisor about alternatives.</p><h3>Electives</h3><p>Choose one approved course, not every option.</p>''')
    record = parse_program_detail(source, URL)
    assert record is not None
    card = program_card_markdown(record, detect_catalog_edition(source))
    for marker in ('MATH 1225', 'MATH 1226', 'MATH 2114', 'CS 1114', 'STAT 4705', 'or STAT 4714', '123'):
        assert marker in card.split('---', 2)[2]
    assert '3 [note 1]' in card and 'Consult an advisor' in card
    assert '\n| Course | Title | Credits |\n' in card
    assert '#### Electives' in card and 'not every option' in card
    assert 'program_level: undergraduate' in card and 'catalog_edition: 2026-2027' in card


def test_nested_lists_do_not_repeat_parent_or_child():
    soup = BeautifulSoup('<div><h2>Requirements</h2><ul><li>Select one:<ul><li>A</li><li>B</li></ul></li></ul></div>', 'html.parser')
    assert requirements_blocks(soup.div) == ['### Requirements', '- Select one:', '  - A', '  - B']


def test_rowspan_colspan_and_pipe_escape():
    table = BeautifulSoup('<table><tr><td rowspan="2">Track A | B</td><td>X</td></tr><tr><td>or Y</td></tr><tr><td colspan="2">Choose one</td></tr></table>', 'html.parser').table
    rendered = table_markdown(table)
    assert rendered.count('Track A \\| B') == 2
    assert '| Track A \\| B | or Y |' in rendered
    assert '| Choose one |  |' in rendered


def test_graduate_and_no_table_content_retained():
    source = html('<h2>Graduation Requirements</h2><p>Complete 30 credits including research.</p><ul><li>Complete a thesis.</li></ul>')
    record = parse_program_detail(source, 'https://catalog.vt.edu/graduate/college-engineering/computer-science/ms/')
    assert record.level == 'graduate'
    assert 'Complete a thesis.' in program_card_markdown(record)
    assert 'Complete 30 credits' in program_card_markdown(record)


def test_shells_and_missing_title_rejected():
    assert parse_program_detail('', URL) is None
    assert parse_program_detail('<html><title>Loading</title></html>', URL) is None


def test_real_cs_source_card_fidelity():
    from pathlib import Path
    fixture = Path(__file__).parent / 'fixtures/cs-program-2026-2027.html'
    source = fixture.read_text()
    soup = BeautifulSoup(source, 'html.parser')
    pane_ids = [pane.get('id') for pane in program_requirement_containers(soup)]
    assert 'programcurriculumtextcontainer' in pane_ids
    assert 'graduationrequirementstextcontainer' in pane_ids
    assert 'roadmaptextcontainer' not in pane_ids
    record = parse_program_detail(source, URL)
    assert record.title == 'Computer Science Major'
    card = program_card_markdown(record, detect_catalog_edition(source))
    body = card.split('---', 2)[2]
    assert 'catalog_edition: 2026-2027' in card
    assert body.count('MATH 1225') >= 1
    for marker in ('MATH 1226', 'MATH 2114', 'CS 1114', 'Total Credits', '123', 'Natural Science Electives', 'Statistics Elective'):
        assert marker in body
    assert '### Program Curriculum' in body


def test_repair_script_validates_real_cs_card_with_table_frontmatter():
    from pathlib import Path
    from scripts.repair_program_cards import validated_card
    fixture = Path(__file__).parent / 'fixtures/cs-program-2026-2027.html'
    card, detail = validated_card(
        fixture.read_text(),
        URL,
        'old repaired body placeholder ' * 40,
    )
    assert detail['title'] == 'Computer Science Major'
    assert detail['edition'] == '2026-2027'
    assert detail['table_count'] >= 1
    assert detail['source_course_codes'] >= 100
    assert 'MATH 1225' in card and '| ---' in card
