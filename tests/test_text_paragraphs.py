"""TXT-BINDESTRICH (KLEINKRAM, 2026-08-22) — ``services/text_paragraphs.py``.

The pure grouper is tested on synthetic text here (no ``unstructured`` on the
dev machine). The last test is the container-only sentinel: it pins the
upstream defect the module exists for — the default grouper of the pinned
``unstructured`` tears hyphenated words — so a dependency bump that fixes it
upstream fails loudly and the custom grouper can be retired on evidence.
"""
import pytest

from services.text_paragraphs import group_paragraphs, is_bullet_line


def _paragraphs(text):
    return group_paragraphs(text).split('\n\n')


# --------------------------------------------------------------------------
# The defect: hyphens and en dashes inside short lines and bullet items
# --------------------------------------------------------------------------

def test_the_sync_freeze_probe_line_stays_whole():
    # The live probe from SYNC-FREEZE P1 — one short line, trailing newline
    # (blank-line branch: 1 of 2 "lines" is empty).
    assert _paragraphs('SYNC-FREEZE multi-process RQ check\n') == \
        ['SYNC-FREEZE multi-process RQ check']


def test_short_lines_keep_hyphens_and_en_dashes_but_stay_separate():
    text = ('Hallo Heike,\n\n'
            'Kurz-Update:\nSYNC-FREEZE lief\nmulti-process ok\nRQ-Check gruen\n'
            '2024-2026 – alles gut\n\n'
            'Viele Gruesse\nOliver\n')
    assert _paragraphs(text) == [
        'Hallo Heike,',
        'Kurz-Update:', 'SYNC-FREEZE lief', 'multi-process ok', 'RQ-Check gruen',
        '2024-2026 – alles gut',
        'Viele Gruesse', 'Oliver',
    ]


def test_bullet_items_keep_hyphenated_words_and_join_continuation_lines():
    text = ('Liste:\n\n'
            '- erster Punkt mit Binde-Strich\n'
            '- zweiter Punkt, multi-process\n'
            '  Fortsetzung des zweiten Punkts mit E-Mail\n'
            '- dritter – mit Gedankenstrich\n\n'
            'Ende.\n')
    assert _paragraphs(text) == [
        'Liste:',
        '- erster Punkt mit Binde-Strich',
        '- zweiter Punkt, multi-process Fortsetzung des zweiten Punkts mit E-Mail',
        '- dritter – mit Gedankenstrich',
        'Ende.',
    ]


# --------------------------------------------------------------------------
# The upstream heuristics that are kept
# --------------------------------------------------------------------------

def test_hard_wrapped_prose_is_joined_into_one_paragraph():
    text = ('Der multi-process RQ check lief gestern Abend auf der Mintbox\n'
            'sauber durch, alle 960 Keep-Alive-Anfragen wurden beantwortet\n'
            'und der Thread-Pool hat die Sonden bei 3-9 ms gehalten.\n\n'
            'Zweiter Absatz.\n')
    assert _paragraphs(text) == [
        'Der multi-process RQ check lief gestern Abend auf der Mintbox sauber '
        'durch, alle 960 Keep-Alive-Anfragen wurden beantwortet und der '
        'Thread-Pool hat die Sonden bei 3-9 ms gehalten.',
        'Zweiter Absatz.',
    ]


def test_text_with_almost_no_blank_lines_keeps_one_paragraph_per_line():
    # Below the 10 % blank-line ratio → every line is its own paragraph, even
    # the long ones (upstream new_line_grouper) — and hyphens are untouched.
    lines = [f'Zeile {i} mit Binde-Strich und genug Woertern fuer eine lange Zeile'
             for i in range(12)]
    text = '\n'.join(lines)  # no trailing newline, no blank line at all
    assert _paragraphs(text) == lines


def test_crlf_bodies_are_handled_like_lf():
    text = 'Kurz-Update:\r\nSYNC-FREEZE lief\r\n\r\nViele Gruesse\r\nOliver\r\n'
    assert _paragraphs(text) == ['Kurz-Update:', 'SYNC-FREEZE lief',
                                 'Viele Gruesse', 'Oliver']


def test_empty_or_blank_text_yields_empty_string():
    assert group_paragraphs('') == ''
    assert group_paragraphs('  \n\n \n') == ''
    assert group_paragraphs(None) == ''


# --------------------------------------------------------------------------
# Bullet recognition: at the start of a line, followed by whitespace — only
# --------------------------------------------------------------------------

@pytest.mark.parametrize('line', ['- item', '  - item', '– item', '• item',
                                  '* item', '·', '-'])
def test_bullet_line_markers(line):
    assert is_bullet_line(line)


@pytest.mark.parametrize('line', ['-Strich', '*kursiv* ist wichtig', '2024-2026',
                                  'Binde-Strich', '-- ', '1. item', 'e item'])
def test_non_bullet_lines(line):
    assert not is_bullet_line(line)


def test_a_marker_glued_to_a_word_does_not_open_an_item():
    # Upstream treated "-Strich"/"*kursiv*" as bullet paragraphs; here they
    # are text, and a short-line block stays a short-line block.
    assert _paragraphs('*kursiv* ist wichtig\n-Strich bleibt\n\n') == \
        ['*kursiv* ist wichtig', '-Strich bleibt']


# --------------------------------------------------------------------------
# Container-only sentinel: the upstream defect still exists (else retire us)
# --------------------------------------------------------------------------

def test_pinned_unstructured_still_tears_hyphens_without_this_grouper():
    partition_text = pytest.importorskip(
        'unstructured.partition.text', reason='unstructured only in the image'
    ).partition_text
    text = 'SYNC-FREEZE multi-process RQ check\n\nZweiter Absatz.\n'
    torn = [el.text for el in partition_text(text=text)]
    assert torn[:3] == ['SYNC', 'FREEZE multi', 'process RQ check'], (
        'the pinned unstructured no longer splits at hyphens — '
        'services/text_paragraphs.py may be retired (re-measure first)')
    whole = [el.text for el in partition_text(text=text,
                                              paragraph_grouper=group_paragraphs)]
    assert whole == ['SYNC-FREEZE multi-process RQ check', 'Zweiter Absatz.']
