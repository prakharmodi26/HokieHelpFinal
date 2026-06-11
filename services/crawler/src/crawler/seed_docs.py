"""Hand-curated seed documents uploaded alongside crawled content."""

from __future__ import annotations


def build_department_info_doc() -> str:
    """Generate a reference document with CS department institutional info.

    This is stored once in the cleaned bucket so RAG can find campus addresses,
    phone numbers, and social links even though they're stripped from individual pages.
    """
    return """\
---
url: '_department-info'
title: 'CS Department Contact Information and Locations'
crawl_depth: 0
crawl_timestamp: 'generated'
---

# Computer Science Department - Contact Information and Locations

## Social Media
- Facebook: https://facebook.com/VT.ComputerScience
- X (Twitter): https://x.com/vt_cs
- Instagram: https://www.instagram.com/vt_cs/
- LinkedIn: https://www.linkedin.com/company/vt-cs/
- YouTube: https://www.youtube.com/@VTEngineering/featured

## Blacksburg Campus
Torgersen Hall RM 3160
620 Drillfield Dr.
Blacksburg, VA 24061
- Undergraduate: (540) 231-6931
- Graduate: (540) 231-0746

## Institute for Advanced Computing
3625 Potomac Ave.
Alexandria, VA 22305
- M.S. and Ph.D.: (703) 538-8370
- M.Eng: (703) 538-3768

## Virginia Tech Research Center
900 N. Glebe Rd.
Arlington, VA 22203
- Phone: (571) 858-3000

## Part of College of Engineering
Virginia Tech College of Engineering: https://eng.vt.edu/
"""
