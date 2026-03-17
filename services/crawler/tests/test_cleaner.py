import pytest
from crawler.cleaner import clean_markdown, build_department_info_doc

FRONTMATTER = """\
---
url: 'https://website.cs.vt.edu/people/admin/jane.html'
title: 'Jane Doe  | Computer Science | Virginia Tech'
crawl_depth: 2
crawl_timestamp: '2026-03-17T00:06:52.348399+00:00'
---"""

BOILERPLATE_HEADER = """\
  * [Skip to main content](https://website.cs.vt.edu/page.html#vt_main)
  * [Skip to search](https://website.cs.vt.edu/page.html#vt_search_box)


[![](https://www.assets.cms.vt.edu/images/whiteVTonTransparent.svg)Virginia Tech\u00ae home](https://www.vt.edu)
![](https://www.assets.cms.vt.edu/images/accessibility_icon_white.svg) Universal Access Toggle
Universal Access
Close Universal Access dialog Universal Access Options
  * [![](https://www.assets.cms.vt.edu/images/vt-accessibility_report-barrier.svg)Report a barrier](https://www.vt.edu/accessibility/barrier.html)
  * [![](https://www.assets.cms.vt.edu/images/vt-accessibility_accessibility-portal.svg)Accessibility portal](https://www.vt.edu/accessibility.html)
  * ![](https://www.assets.cms.vt.edu/images/toggle_slider_off-01.svg) ![](https://www.assets.cms.vt.edu/images/toggle_slider_on-01.svg) Pause all background videos
  * ![](https://www.assets.cms.vt.edu/images/toggle_slider_off-01.svg) ![](https://www.assets.cms.vt.edu/images/toggle_slider_on-01.svg) Underline all links


  * [Apply](https://www.vt.edu/apply.html)
  * [Visit](https://www.vt.edu/visit.html)
  * [Give](https://give.vt.edu)
  * Shop
    * [Hokie Sports Shop
Apparel, clothing, gear and merchandise](https://shop.hokiesports.com)
    * [Hokie Shop
University Bookstore, merchandise and gifts](https://www.bkstr.com/virginiatechstore/home)
    * [Hokie Gear
Everything you need to know about Hokie gear](https://hokiegear.com)
    * [Hokie License Plates
Part of every Virginia Tech plate purchase funds scholarships](https://www.vt.edu/plates)


Resources for
  * [Future Students](https://www.vt.edu/admissions.html)
  * [Current Students](https://www.vt.edu/resources/current-students.html)
  * [Parents and Families](https://www.vt.edu/resources/parents-and-families.html)
  * [Faculty and Staff](https://www.vt.edu/resources/faculty-and-staff.html)
  * [Alumni](https://alumni.vt.edu)
  * [Industry and Partners](https://www.vt.edu/link.html)
"""

NAV_MENU = """\
[ Computer Science ](https://website.cs.vt.edu/)
Menu
  1. [ College of Engineering / ](https://eng.vt.edu/)
  2. [Department of Computer Science](https://website.cs.vt.edu/)


  * [Home](https://website.cs.vt.edu/index.html)
  * [About](https://website.cs.vt.edu/About.html) About Submenu Toggle
    * [Accreditation](https://website.cs.vt.edu/About/accreditation.html)
    * [News](https://website.cs.vt.edu/About/News.html)
  * [People](https://website.cs.vt.edu/people.html) People Submenu Toggle
    * [Faculty](https://website.cs.vt.edu/people/faculty.html)
    * [Advisory board](https://website.cs.vt.edu/people/advisory-board.html)
  * [Research](https://website.cs.vt.edu/research.html) Research Submenu Toggle
    * [Research areas](https://website.cs.vt.edu/research/research-areas.html)
  * [Engage](https://website.cs.vt.edu/engage.html) Engage Submenu Toggle
    * [Alumni](https://website.cs.vt.edu/engage/Alumni.html)
"""

SEARCH_WIDGET = """\
Search
Search query
[\u00d7](javascript:void(0) "Clear search box")
search
Search this site
Search all vt.edu sites
People search
Frequent Searches:
  * [Hokie Spa](https://hokiespa.vt.edu)
  * [Canvas](https://canvas.vt.edu)
  * [Jobs](https://jobs.vt.edu/)
  * [Academic Calendar](https://www.registrar.vt.edu/dates-deadlines/academic-calendar.html)
  * [Tuition](https://www.vt.edu/admissions/undergraduate/cost.html)


webnewsvideopeople
Web results for
Sort by relevance Sort by date



News results for
Sort by relevance Sort by date
Filter search
Categories
Academics
Campus Experience
Culture
Impact
Research
Story type
Feature
Notice
Story
Video
Apply filters Clear filters



Video results for
Sort by relevance Sort by date



People results for



## People Results


## VT News Results


## Search Help
The search feature within the content management system themes has options.
## Search Tips
**quantum physics**
Finds all documents that contain both words.
## More search options
  * Search the [University Photo Library](https://www.photolibrary.unirel.vt.edu)
  * Search the [University Libraries](https://virginiatech.on.worldcat.org/advancedsearch)
"""

DEPT_FOOTER = """\
### Follow Computer Science
[Facebook](https://facebook.com/VT.ComputerScience) [![](https://www.assets.cms.vt.edu/images/social-media/x-logo-white.svg)X](https://x.com/vt_cs) [Instagram](https://www.instagram.com/vt_cs/) [Linked In](https://www.linkedin.com/company/vt-cs/mycompany/?viewAsMember=true) [YouTube](https://www.youtube.com/@VTEngineering/featured)
**Blacksburg campus**
Torgersen Hall RM 3160
620 Drillfield Dr.
Blacksburg, VA 24061
(540) 231-6931 (undergraduate)
(540) 231-0746 (graduate)
**Institute for Advanced Computing**
3625 Potomac Ave.
Alexandria, VA 22305
(703) 538-8370 (M.S. and Ph.D.)
(703) 538-3768 (M.Eng)
**Virginia Tech Research Center
**900 N. Glebe Rd.
Arlington, VA 22203
(571) 858-3000
[Intranet](https://admin.cs.vt.edu/) (Internal)
"""

VT_FOOTER = """\
![Map of Virginia with pins](https://www.assets.cms.vt.edu/images/vt-campuses.svg)
![Virginia Tech logo](https://www.assets.cms.vt.edu/images/logo-white-black.svg)
[Get Directions ](https://www.vt.edu/maps/directions.html)
[See All Locations ](https://www.vt.edu/maps.html#locations)
[Contact Virginia Tech ](https://www.vt.edu/contacts.html)
  * [University Status](https://www.vt.edu/status.html)
  * [Principles of Community](https://www.vt.edu/principles-of-community.html)
  * [Privacy Statement](https://www.vt.edu/privacy.html)
  * [Acceptable Use](https://www.vt.edu/acceptable-use.html)
  * [We Remember](https://www.weremember.vt.edu/)
  * [University Libraries](https://lib.vt.edu)
  * [Accessibility](https://www.vt.edu/accessibility.html)
  * [Equal Opportunity](https://www.vt.edu/equal-opportunity.html)
  * [Jobs at Virginia Tech](https://jobs.vt.edu)


\u00a9 2026 Virginia Polytechnic Institute and State University. All rights reserved.
  * [Instagram](https://instagram.com/virginia.tech/ "instagram")
  * [Facebook](https://facebook.com/virginiatech/ "facebook")
  * [X](https://x.com/virginia_tech/ "x")


![](https://bat.bing.com/action/0?ti=343199719&Ver=2&mid=abc)
"""

EXPLORE_SIDEBAR = """\
Explore
  * [ Administration ](https://website.cs.vt.edu/people/administration.html)
    * [ Chris Arnold ](https://website.cs.vt.edu/people/administration/chris-arnold.html)
    * [ Jen Bradley ](https://website.cs.vt.edu/people/administration/jennifer-bradley.html)
  * [ Faculty ](https://website.cs.vt.edu/people/faculty.html)
  * [ Advisory board ](https://website.cs.vt.edu/people/advisory-board.html)
"""


def test_clean_full_boilerplate_page():
    """Strips header, nav, search, explore, footer from a full raw page."""
    raw = (
        f"{FRONTMATTER}\n\n{BOILERPLATE_HEADER}\n{NAV_MENU}\n"
        f"{SEARCH_WIDGET}\n{EXPLORE_SIDEBAR}\n"
        f"#  Jane Doe \nAssistant Professor\nShe studies AI.\n\n"
        f"{DEPT_FOOTER}\n{VT_FOOTER}"
    )
    result = clean_markdown(raw)

    # Boilerplate gone
    assert "Skip to main content" not in result
    assert "Universal Access" not in result
    assert "Submenu Toggle" not in result
    assert "Search query" not in result
    assert "Frequent Searches" not in result
    assert "quantum physics" not in result
    assert "Follow Computer Science" not in result
    assert "Virginia Polytechnic Institute" not in result
    assert "bat.bing.com" not in result
    assert "Chris Arnold" not in result
    assert "Hokie Sports Shop" not in result
    assert "Resources for" not in result

    # Content preserved
    assert "Jane Doe" in result
    assert "Assistant Professor" in result
    assert "She studies AI." in result

    # Frontmatter preserved
    assert "url: 'https://website.cs.vt.edu/people/admin/jane.html'" in result


def test_clean_already_clean_page_is_noop():
    """Pages where fit_markdown already worked pass through unchanged."""
    clean_page = f"{FRONTMATTER}\n\n# About\nThe department has been an incubator.\n"
    result = clean_markdown(clean_page)
    assert result.strip() == clean_page.strip()


def test_clean_removes_cms_artifacts():
    page = (
        f"{FRONTMATTER}\n\n# Faculty\n"
        f"  * Bio Item Bio Item\n"
        f"[ Jane Doe , bio ](https://example.com)\n"
        f"Professor\n"
        f"  * General Item General Item\n"
        f"  * Article Item Article Item\n"
        f"  * Redirect Item Redirect Item\n"
        f"[ John , redirect ](https://example.com)\n"
        f"Associate Professor\n"
    )
    result = clean_markdown(page)
    assert "Bio Item Bio Item" not in result
    assert "General Item General Item" not in result
    assert "Article Item Article Item" not in result
    assert "Redirect Item Redirect Item" not in result
    assert "Jane Doe" in result
    assert "Professor" in result
    assert "John" in result


def test_clean_removes_tracking_pixels():
    page = f"{FRONTMATTER}\n\n# Title\nContent.\n\n![](https://bat.bing.com/action/0?ti=343199719&Ver=2&mid=abc)\n"
    result = clean_markdown(page)
    assert "bat.bing.com" not in result
    assert "Content." in result


def test_clean_removes_breadcrumb():
    page = (
        f"{FRONTMATTER}\n\n"
        f"  1. [ College of Engineering  /  ](https://eng.vt.edu/)\n"
        f"  2. [Department of Computer Science / ](https://website.cs.vt.edu/)\n"
        f"  3. [About / ](https://website.cs.vt.edu/About.html)\n\n"
        f"# About\nContent here.\n"
    )
    result = clean_markdown(page)
    assert "College of Engineering" not in result
    assert "Content here." in result


def test_clean_removes_explore_sidebar():
    page = f"{FRONTMATTER}\n\n{EXPLORE_SIDEBAR}\n# Person Name\nBio here.\n"
    result = clean_markdown(page)
    assert "Chris Arnold" not in result
    assert "Jen Bradley" not in result
    assert "Bio here." in result


def test_clean_removes_svg_data_images():
    page = f"{FRONTMATTER}\n\n# Title\n![](data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg')\nContent.\n"
    result = clean_markdown(page)
    assert "data:image/svg+xml" not in result
    assert "Content." in result


def test_clean_preserves_content_images():
    page = f"{FRONTMATTER}\n\n# Title\n![Photo](https://website.cs.vt.edu/content/image.jpg)\nContent.\n"
    result = clean_markdown(page)
    assert "![Photo](https://website.cs.vt.edu/content/image.jpg)" in result


def test_build_department_info_doc():
    """Department info document contains campus addresses and contact info."""
    doc = build_department_info_doc()
    assert "Torgersen Hall" in doc
    assert "Blacksburg" in doc
    assert "(540) 231-6931" in doc
    assert "Alexandria" in doc
    assert "Arlington" in doc
    assert "facebook.com/VT.ComputerScience" in doc
