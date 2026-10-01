"""Conservative identity rejection must not become a universal program filter."""
from chatbot.program_metadata import source_conflicts, program_matches
from chatbot.retriever import Retriever


def chunk(name, i=1, **extra):
    return {'chunk_id': str(i), 'text': 'Example curriculum',
            'url': f'https://catalog.vt.edu/undergraduate/example/{i}/',
            'title': name, 'program_name': name, 'program_level': 'undergraduate',
            'page_type': 'program', 'rerank_score': .9, **extra}


def retriever():
    r = object.__new__(Retriever)
    r._top_k = 12
    r._min_score = .3
    r._program_candidates_enabled = True
    return r


def test_no_identity_anchor_does_not_filter_university_wide_program_discovery():
    pool = [chunk('Computer Science Major'), chunk('Statistics Major', 2)]
    result = retriever()._select_ranked('What undergraduate degree programs are available?', pool)
    assert len(result) == 2


def test_explicit_program_anchor_filters_wrong_program_not_unknown_general_policy():
    pool = [chunk('Statistics Major', 2), chunk('Computer Science Major'),
            {'chunk_id': '3', 'text': 'General policy', 'title': 'Undergraduate Policies',
             'url': 'https://vt.edu/policy', 'rerank_score': .8}]
    result = retriever()._select_ranked('Computer Science undergraduate major credits', pool)
    assert [c['chunk_id'] for c in result] == ['1', '3']


def test_major_minor_comparison_keeps_both_types():
    query = 'Compare Computer Science major and minor requirements'
    assert not source_conflicts(query, chunk('Computer Science Major'))
    assert not source_conflicts(query, chunk('Computer Science Minor'))


def test_multiple_editions_comparison_is_not_an_impossible_year_intersection():
    query = 'Compare Computer Science major requirements in 2023-24 and 2026-27'
    assert not source_conflicts(query, chunk('Computer Science Major', catalog_edition='2023-2024'))
    assert not source_conflicts(query, chunk('Computer Science Major', catalog_edition='2026-2027'))


def test_lower_heading_is_not_document_identity():
    source = chunk('Computer Science Major', headings_path=['Computer Science Major', 'Mathematics Minor'])
    assert not source_conflicts('undergraduate Computer Science major requirements', source)


def test_course_topic_is_not_a_second_requested_degree():
    query = 'Virginia Tech undergraduate Computer Science major checksheet required mathematics courses'
    assert not program_matches(query, chunk('Mathematics Major'))
    assert program_matches(query, chunk('Computer Science Major'))
    assert source_conflicts(query, {'title': 'cos_math_adm_22_23.pdf', 'url': 'https://registrar.vt.edu/Updates/cos/22-23/cos_math_adm_22_23.pdf'})
    assert source_conflicts(query, {'title': 'Mathematics | Virginia Tech', 'url': 'https://vt.edu/academics/minors/mathematics.html'})
    assert source_conflicts('undergraduate Mathematics major credits', {'title': 'Cals Agsc Cld 23 24', 'url': 'https://registrar.vt.edu/Updates/cals/23-24/cals_agsc_cld_23_24.pdf'})


def test_specialization_identity_is_not_reduced_to_base_major():
    query = 'Mathematics major with Applied Computational Mathematics option credits'
    assert not program_matches(query, chunk('Mathematics Major'))
    assert program_matches(query, chunk('Mathematics Major with Applied Computational Mathematics Option'))
    assert source_conflicts(query, chunk('Mathematics Major with Mathematics Education Option'))
    assert source_conflicts(query, {'title': 'Mathematics Major with Mathematics Education Option', 'url': 'https://vt.edu/checksheets/math_education_option.pdf'})


def test_generic_degree_advice_is_not_a_program_identity():
    query = 'undergraduate Mathematics major credits'
    for title in ('Accel U G Degree', 'HokieGPS - Graduation Plan for Success: Managing Your Degree'):
        assert not source_conflicts(query, {'title': title, 'url': 'https://vt.edu/advising/general.pdf'})


def test_malformed_optional_identity_fields_do_not_crash_retrieval():
    assert not source_conflicts('undergraduate Computer Science major credits', {
        'program_level': [], 'headings_path': {}, 'url': 'http://['})


def test_governance_and_wrong_level_excluded_even_without_program_anchor():
    query = 'undergraduate Chronoengineering degree requirements'
    assert source_conflicts(query, {'title': 'CUSP First Reading', 'url': 'https://vt.edu/governance/proposal.pdf'}, identity_anchored=False)
    assert source_conflicts(query, chunk('Something Else', program_level='graduate'), identity_anchored=False)
