# CS curriculum review reference

Reference: browser-rendered **2026–2027** Computer Science Major catalog fixture in the separate ingestion repair worktree, `services/crawler/tests/fixtures/cs-program-2026-2027.html`.

Official page: https://catalog.vt.edu/undergraduate/college-engineering/computer-science/computer-science-bs/

This is a review of the captured edition, **not** a determination of a student's cohort or a fresh currency check.

## Required mathematics / alternatives shown in the curriculum table

- MATH 1225 and MATH 1226 (Pathways quantitative requirements).
- MATH 2114.
- MATH 2204 **or CMDA 2005**.
- MATH 2534 **or MATH 3034**.
- MATH 3134 **or MATH 3124** (additional course requirements).
- A three-credit **statistics elective**, not a blanket requirement to take a single specific STAT course.

The degree table states **Total Credits: 123**. Do not reconstruct the degree total by summing all listed alternatives, electives, or double-countable Pathways rows.

## Important distinctions for generated-answer review

- The roadmap lists MATH 3134; the curriculum table also allows MATH 3124. A recommended semester plan must not erase that alternative.
- Technical-elective tables contain other MATH courses, including MATH 3414, 4414 and 4175/4176. Those are **not** blanket required mathematics courses for every CS major.
- The statistics-elective table lists options (STAT 4705, 4105, 4714, 4604, 3704, CMDA 2006), with footnotes and differing course-credit values. Do not present every option as required or infer that all options carry three credits.
- The real HTML uses `programcurriculumtextcontainer`, `graduationrequirementstextcontainer` and `roadmaptextcontainer`; looking only for `requirementstextcontainer` misses it.

The `math-catalog-2026` deployed API probe checks for the required/alternative identifiers, catalog citation and edition. Those checks cannot establish correct OR relationships or elective distinctions; the complete answer and cited passages still need review.
