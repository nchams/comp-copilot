"""Normalize free-text job titles (e.g. from H-1B LCA disclosures) into a
consistent (role_family, level) pair.

Real comp data is messy: "Sr. Software Engineer II", "Staff ML Scientist",
"Principal Product Designer". Benchmarking is impossible until titles are mapped
onto a single leveling framework -- this module does that mapping with simple,
auditable keyword rules (no black box, which matters for a high-stakes,
human-reviewed comp workflow).
"""

import re

# Ordered: more senior signals first so "senior staff" -> staff, etc.
_LEVEL_KEYWORDS = [
    (6, ["principal", "distinguished", "fellow", "vp ", "vice president", "director"]),
    (5, ["staff", "lead ", " lead", "architect", "l5"]),
    (4, ["senior", "sr.", "sr ", "iii", " iv", "l4"]),
    (2, ["junior", "jr.", "jr ", "associate", "entry", " i ", " ii", "l2"]),
    (1, ["intern", "apprentice", "new grad", "graduate", "l1"]),
]

_ROLE_KEYWORDS = [
    ("Engineering Manager", ["engineering manager", "eng manager", "manager, engineering", "software engineering manager"]),
    ("Data Scientist", ["data scientist", "machine learning scientist", "ml scientist", "research scientist", "applied scientist"]),
    ("Data Engineer", ["data engineer", "analytics engineer", "ml engineer", "machine learning engineer"]),
    ("Product Manager", ["product manager", "program manager", "product owner", "tpm"]),
    ("Product Designer", ["designer", "ux ", "ui ", "user experience", "product design"]),
    ("Software Engineer", ["software engineer", "software developer", "swe", "developer", "programmer", "engineer"]),
]


def normalize_title(title: str):
    """Return (role_family, level_int). Defaults to ('Software Engineer', 3)."""
    if not isinstance(title, str) or not title.strip():
        return "Software Engineer", 3
    t = f" {title.lower().strip()} "

    role = None
    for fam, kws in _ROLE_KEYWORDS:
        if any(kw in t for kw in kws):
            role = fam
            break
    if role is None:
        role = "Software Engineer"

    level = None
    for lvl, kws in _LEVEL_KEYWORDS:
        if any(kw in t for kw in kws):
            level = lvl
            break
    # No explicit seniority marker -> assume mid-level (L3), the most common case.
    if level is None:
        level = 3

    # Managers don't sit below L4 in this ladder; floor them.
    if role == "Engineering Manager" and level < 4:
        level = 4
    return role, level


if __name__ == "__main__":
    for ex in [
        "Senior Software Engineer",
        "Staff Data Scientist",
        "Jr. Product Designer",
        "Principal ML Engineer",
        "Product Manager II",
        "Engineering Manager",
        "Software Developer",
    ]:
        print(f"{ex:35s} -> {normalize_title(ex)}")
