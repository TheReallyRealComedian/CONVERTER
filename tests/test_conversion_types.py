"""ARCH-LIBRARY-KLEIN — the type vocabulary is held together by ONE sentinel.

``conversion_type`` is a string with seven allowed values
(``app_pkg.library.ALLOWED_CONVERSION_TYPES``) that is maintained at six
places in four files: the allow-list itself, the badge label chain in
``templates/library.html``, the same chain in ``templates/library_detail.html``,
the type filter ``<select>`` in ``library.html``, the ``.type-<typ>`` tone
rules in ``static/css/style.css`` and the ingest default. None of these
places checks the others — ``type-{{ conv.conversion_type }}`` is a contract
without a referee: a type added to the allow-list rendered with the raw
string as its label (``audio_narration`` in the list) or with no tone at all
(``document_conversion``, no CSS rule) and had no way into the filter.

This file is the referee. For EVERY allowed type it creates a row and renders
the list and the detail page, and asserts, one assertion per concern so a
failure names the type and the place:

* the list badge text is a word, not the type string;
* the detail badge text is the same word;
* ``style.css`` carries a tone rule whose selector list names ``.type-<typ>``;
* the type has a filter path: an ``<option>`` of its own or one that covers
  it (``FILTER_PATH`` below — the "Dokument" option spans both document
  types), and the list filtered by that option shows the row.

``FILTER_PATH`` is a dict on purpose: an eighth type in the allow-list makes
``test_filter_path_dict_covers_the_allow_list`` red until someone decides its
filter path, instead of silently inheriting none.
"""
import html
import re
from pathlib import Path

import pytest

from app_pkg.library import ALLOWED_CONVERSION_TYPES
from models import Conversion, db

STYLE_CSS = Path(__file__).resolve().parent.parent / 'static' / 'css' / 'style.css'

# type → the <option value=…> of the type filter that lists it. A type whose
# own value is an option maps to itself; a type covered by another option
# maps to that option (web list only — /api/conversions?type= stays exact).
FILTER_PATH = {
    'document_to_markdown': 'document_to_markdown',
    'document_conversion': 'document_to_markdown',   # "Dokument" spans both
    'audio_transcription': 'audio_transcription',
    'audio_narration': 'audio_narration',
    'dialogue_formatting': 'dialogue_formatting',
    'markdown_input': 'markdown_input',
    'ai_newsletter': 'ai_newsletter',
}

TYPES = sorted(ALLOWED_CONVERSION_TYPES)

_BADGE_RE = re.compile(r'<span class="type-badge type-([A-Za-z0-9_]+)">\s*(.*?)\s*</span>', re.S)
_OPTION_RE = re.compile(r'<option value="([^"]*)"[^>]*>\s*(.*?)\s*</option>', re.S)
_RULE_RE = re.compile(r'([^{}]+)\{([^{}]*)\}', re.S)


def _make_row(app, user_id, typ):
    with app.app_context():
        c = Conversion(user_id=user_id, conversion_type=typ,
                       title=f'Zeile vom Typ {typ}', content=f'Inhalt vom Typ {typ}.',
                       lifecycle_status='later', metadata_json='{}')
        db.session.add(c)
        db.session.commit()
        return c.id


def _badges(html_bytes):
    """{type: label} for every type badge in a rendered page."""
    return {typ: html.unescape(label) for typ, label in _BADGE_RE.findall(html_bytes.decode('utf-8'))}


def _list_badges(client):
    # The neutral shelf (Bibliothek tab) shows rows placed 'later' + unqueued.
    resp = client.get('/library?view=bibliothek&per_page=50')
    assert resp.status_code == 200
    return _badges(resp.data)


def _detail_badges(client, cid):
    resp = client.get(f'/library/{cid}')
    assert resp.status_code == 200
    return _badges(resp.data)


def _type_filter_options(client):
    resp = client.get('/library?view=bibliothek')
    assert resp.status_code == 200
    page = resp.data.decode('utf-8')
    select = re.search(r'<select name="type"[^>]*>(.*?)</select>', page, re.S)
    assert select is not None, 'the type filter <select> is on the Bibliothek tab'
    return {value: html.unescape(label) for value, label in _OPTION_RE.findall(select.group(1))}


def _tone_rules():
    """[(selector list, declarations)] of style.css, braces-flat (the tone
    rules are top-level one-liners; nested @media blocks only matter for
    their inner rules, which the regex also yields)."""
    return _RULE_RE.findall(STYLE_CSS.read_text(encoding='utf-8'))


# --- the dict mirrors the allow-list ----------------------------------------

def test_filter_path_dict_covers_the_allow_list():
    assert set(FILTER_PATH) == set(ALLOWED_CONVERSION_TYPES), (
        'a type was added to or removed from ALLOWED_CONVERSION_TYPES — '
        'decide its filter path in FILTER_PATH (own option or a covering one)')
    options = set(FILTER_PATH.values())
    for typ in ALLOWED_CONVERSION_TYPES:
        assert FILTER_PATH[typ] in ALLOWED_CONVERSION_TYPES
    assert options <= set(ALLOWED_CONVERSION_TYPES)


# --- label: list and detail -------------------------------------------------

@pytest.mark.parametrize('typ', TYPES)
def test_list_badge_label_is_a_word(app, authenticated_client, test_user, typ):
    _make_row(app, test_user['id'], typ)
    badges = _list_badges(authenticated_client)
    assert typ in badges, f'the list renders a type badge for {typ}'
    assert badges[typ] and badges[typ] != typ, (
        f'list badge of {typ} shows the raw type string — add a label branch in library.html')


@pytest.mark.parametrize('typ', TYPES)
def test_detail_badge_label_is_a_word(app, authenticated_client, test_user, typ):
    cid = _make_row(app, test_user['id'], typ)
    badges = _detail_badges(authenticated_client, cid)
    assert typ in badges, f'the detail page renders a type badge for {typ}'
    assert badges[typ] and badges[typ] != typ, (
        f'detail badge of {typ} shows the raw type string — add a label branch in library_detail.html')


@pytest.mark.parametrize('typ', TYPES)
def test_detail_label_equals_list_label(app, authenticated_client, test_user, typ):
    cid = _make_row(app, test_user['id'], typ)
    assert _detail_badges(authenticated_client, cid)[typ] == _list_badges(authenticated_client)[typ]


# --- tone: a CSS rule per type -----------------------------------------------

@pytest.mark.parametrize('typ', TYPES)
def test_css_has_a_tone_rule(typ):
    needle = re.compile(r'(^|[\s,])\.type-' + re.escape(typ) + r'(?=[\s,{:]|$)')
    toned = [sel for sel, decl in _tone_rules()
             if needle.search(sel) and 'background' in decl]
    assert toned, (
        f'style.css has no .type-{typ} rule with a background — the badge falls '
        f'back to the toneless .type-badge base')


# --- filter path -------------------------------------------------------------

@pytest.mark.parametrize('typ', TYPES)
def test_type_has_a_filter_path(app, authenticated_client, test_user, typ):
    option = FILTER_PATH[typ]
    options = _type_filter_options(authenticated_client)
    assert option in options, (
        f'{typ} has no filter path: the type filter has no <option value="{option}"> '
        f'(options: {sorted(v for v in options if v)})')
    assert options[option] and options[option] != option, 'the option label is a word'
    cid = _make_row(app, test_user['id'], typ)
    resp = authenticated_client.get(f'/library?type={option}')
    assert resp.status_code == 200
    assert f'data-id="{cid}"'.encode() in resp.data, (
        f'/library?type={option} does not list the {typ} row')


def test_type_filter_has_exactly_the_mapped_options(authenticated_client):
    options = {v for v in _type_filter_options(authenticated_client) if v}
    assert options == set(FILTER_PATH.values())
