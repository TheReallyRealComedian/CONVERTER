"""MCP-DOCWRITE Phase 1 — the heading-addressed section replacer.

Pure unit tests for ``services.markdown_sections.replace_section`` — the heart
of the sprint. Covers boundary detection (subsections stay in, same-level
siblings end the section), the fenced-code trap in both directions, the
conservative ambiguity gate (>1 match → error, never guess), and exact
trailing-newline round-tripping.
"""
import pytest

from services.markdown_sections import (
    replace_section,
    SectionNotFound,
    SectionAmbiguous,
)


# --- Single-match replace, everything else untouched ---

def test_single_match_replaces_heading_and_body_only():
    md = "# A\nbody a\n# B\nbody b\n# C\nbody c\n"
    out = replace_section(md, "B", "# B\nNEW body")
    assert out == "# A\nbody a\n# B\nNEW body\n# C\nbody c\n"


def test_heading_param_with_or_without_hashes_is_equivalent():
    """``heading`` is normalised via ``lstrip('#').strip()`` — both forms hit."""
    md = "# A\na\n# B\nb\n"
    expected = "# A\na\n# B\nX"
    assert replace_section(md, "B", "# B\nX") == expected
    assert replace_section(md, "## B", "# B\nX") == expected
    assert replace_section(md, "  B  ", "# B\nX") == expected


# --- Subsection boundaries: deeper headings belong, siblings end the section ---

def test_section_swallows_its_subsections():
    md = "# Main\nintro\n## Sub1\ns1\n## Sub2\ns2\n# Next\nn\n"
    out = replace_section(md, "Main", "# Main\nrewritten")
    # ## Sub1 / ## Sub2 are inside Main and get replaced; # Next is untouched.
    assert out == "# Main\nrewritten\n# Next\nn\n"


def test_subsection_replace_stops_at_same_level_sibling():
    md = "# Main\nintro\n## Sub1\ns1\n## Sub2\ns2\n# Next\nn\n"
    out = replace_section(md, "Sub1", "## Sub1\nX")
    # Only Sub1's own body is replaced; Sub2 and Next are left alone.
    assert out == "# Main\nintro\n## Sub1\nX\n## Sub2\ns2\n# Next\nn\n"


# --- Position: first / middle / last section ---

@pytest.mark.parametrize("heading,new,expected", [
    ("One", "# One\nX", "# One\nX\n# Two\n2\n# Three\n3"),
    ("Two", "# Two\nX", "# One\n1\n# Two\nX\n# Three\n3"),
    ("Three", "# Three\nX", "# One\n1\n# Two\n2\n# Three\nX"),
])
def test_first_middle_last_section(heading, new, expected):
    md = "# One\n1\n# Two\n2\n# Three\n3"
    assert replace_section(md, heading, new) == expected


def test_section_at_doc_end_runs_to_eof():
    md = "# A\na\n# B\nb\n"
    out = replace_section(md, "B", "# B\nNEW")
    # B owns everything to EOF, incl. the trailing newline line.
    assert out == "# A\na\n# B\nNEW"


# --- Fenced-code trap, both directions ---

def test_hash_line_inside_fence_is_not_a_section_boundary():
    """A ``# comment`` in a code block must not end the enclosing section."""
    md = "\n".join([
        "# Code",
        "```python",
        "# this is a comment, not a heading",
        "x = 1",
        "```",
        "more",
        "# After",
        "a",
    ])
    out = replace_section(md, "Code", "# Code\nREPLACED")
    # Section Code spans the whole fenced block up to # After.
    assert out == "\n".join(["# Code", "REPLACED", "# After", "a"])


def test_hash_line_inside_fence_is_not_an_addressable_target():
    md = "\n".join([
        "# Code",
        "```python",
        "# this is a comment, not a heading",
        "x = 1",
        "```",
        "# After",
        "a",
    ])
    with pytest.raises(SectionNotFound):
        replace_section(md, "this is a comment, not a heading", "X")


def test_tilde_fence_also_shields_hash_lines():
    md = "\n".join(["# T", "~~~", "# fake", "~~~", "# Real", "r"])
    # # fake inside the ~~~ fence is neither boundary nor target.
    out = replace_section(md, "T", "# T\nX")
    assert out == "\n".join(["# T", "X", "# Real", "r"])
    with pytest.raises(SectionNotFound):
        replace_section(md, "fake", "X")


# --- Ambiguity gate: never guess between matches ---

def test_two_headings_same_text_is_ambiguous():
    md = "# Dup\na\n# Dup\nb\n"
    with pytest.raises(SectionAmbiguous):
        replace_section(md, "Dup", "X")


def test_same_text_different_levels_is_ambiguous():
    """Text match is level-agnostic: ``# Intro`` and ``### Intro`` collide."""
    md = "# Intro\na\n### Intro\nb\n"
    with pytest.raises(SectionAmbiguous):
        replace_section(md, "Intro", "X")


# --- Not found ---

def test_missing_heading_raises_not_found():
    md = "# A\na\n# B\nb\n"
    with pytest.raises(SectionNotFound):
        replace_section(md, "Nope", "X")


# --- new_section shapes ---

def test_new_section_without_heading_is_spliced_cleanly():
    md = "# A\nold a\n# B\nb\n"
    out = replace_section(md, "A", "just body, no heading")
    # Heading dropped on purpose; no line-gluing into the next section.
    assert out == "just body, no heading\n# B\nb\n"


def test_closed_atx_trailing_hashes_match():
    md = "## Setup ##\nbody\n## Next ##\nn\n"
    out = replace_section(md, "Setup", "## Setup ##\nbody2")
    assert out == "## Setup ##\nbody2\n## Next ##\nn\n"


# --- Trailing-newline round-trip ---

def test_trailing_newline_preserved_for_non_final_section():
    md = "# A\na\n# B\nb\n"  # ends with \n
    out = replace_section(md, "A", "# A\nX")
    assert out == "# A\nX\n# B\nb\n"  # trailing \n kept


def test_no_trailing_newline_stays_absent():
    md = "# A\na"  # no trailing \n
    out = replace_section(md, "A", "# A\nX")
    assert out == "# A\nX"  # still no trailing \n
