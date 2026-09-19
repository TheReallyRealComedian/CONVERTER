"""RICH-MEDIA Phase 1.3 — media in a document's TEXT: finding it, budgeting it
at the write paths (413), keeping it out of the list preview.

First half pure (``services.doc_media``), second half the routes: every path
that accepts Markdown from outside — ingest, session create, editor PUT,
docwrite full + section — answers an over-budget ``content`` with 413 and a
German sentence, and has written nothing. The audio upload keeps its 500 MB.
"""
import os
import time

import pytest

from models import Conversion, db
from services.doc_media import (
    DATA_URI_TOO_LARGE,
    DOCUMENT_MEDIA_TOO_LARGE,
    MAX_DATA_URI_BYTES,
    MAX_DOCUMENT_MEDIA_BYTES,
    check_media_limits,
    find_svg_spans,
    strip_media_for_preview,
)

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')
_PREFIX = 'data:image/png;base64,'


def _fixture(name):
    with open(os.path.join(FIXTURES, name), encoding='utf-8') as f:
        return f.read()


def _data_uri(total_bytes):
    """A data URI of exactly ``total_bytes`` bytes (ASCII, so bytes == chars)."""
    return _PREFIX + 'A' * (total_bytes - len(_PREFIX))


def _spans(text):
    islands, dangling = find_svg_spans(text)
    return [text[s:e] for s, e in islands], [text[s:e] for s, e in dangling]


# --- find_svg_spans -----------------------------------------------------------


def test_spans_outermost_islands_self_closed_and_dangling():
    islands, dangling = _spans(
        'a <svg><svg></svg></svg> b <svg/> c <SVG x="1"><rect/></SVG> d <svg y="2">')
    assert islands == ['<svg><svg></svg></svg>', '<svg/>', '<SVG x="1"><rect/></SVG>']
    assert dangling == ['<svg y="2">']


def test_spans_island_inside_an_unclosed_svg_is_still_found():
    islands, dangling = _spans('<svg a="1"> x <svg><rect/></svg> y')
    assert islands == ['<svg><rect/></svg>']
    assert dangling == ['<svg a="1">']


def test_spans_are_tag_aware():
    # An <svg inside another tag's attribute value is not an edge — neither in
    # alt text nor in the common hand-written unencoded SVG data URI.
    islands, dangling = _spans(
        '<img alt="<svg></svg>" src="data:image/svg+xml;utf8,<svg xmlns=\'x\'></svg>">'
        ' <svg title="a>b"><rect/></svg>')
    assert islands == ['<svg title="a>b"><rect/></svg>']
    assert dangling == []


def test_spans_skip_comments_and_survive_a_lone_angle_bracket():
    islands, _ = _spans('<!-- <svg></svg> --> wenn a <b dann "zitat <svg><rect/></svg>')
    assert islands == ['<svg><rect/></svg>']
    assert _spans('<svgfoo></svgfoo> kein <svg') == ([], [])


# --- check_media_limits -------------------------------------------------------


def test_limits_small_and_media_free_texts_pass():
    assert check_media_limits('# Titel\n\nText.') is None
    assert check_media_limits(_fixture('rich_media_probe_240.md')) is None
    # The budget is for MEDIA. Plain text of any length is not its business.
    assert check_media_limits('Wort ' * 3_000_000) is None


def test_limits_data_uri_at_the_limit_passes_one_byte_over_is_refused():
    assert check_media_limits(f'![x]({_data_uri(MAX_DATA_URI_BYTES)})') is None
    assert check_media_limits(
        f'![x]({_data_uri(MAX_DATA_URI_BYTES + 1)})') == DATA_URI_TOO_LARGE
    assert check_media_limits(
        f'<img alt="x" src="{_data_uri(MAX_DATA_URI_BYTES + 1)}">') == DATA_URI_TOO_LARGE


def test_limits_document_total():
    one = f'![x]({_data_uri(MAX_DATA_URI_BYTES)})\n\n'  # 2 MB each
    assert check_media_limits(one * 5) is None           # 10 MB — at the limit
    assert check_media_limits(one * 5 + '![y](data:image/png;base64,AA)') \
        == DOCUMENT_MEDIA_TOO_LARGE


def test_limits_inline_svg_counts_and_overlap_is_billed_once():
    rect = '<rect x="1" width="5" height="5"/>'
    big_svg = '<svg viewBox="0 0 1 1">' + rect * 320_000 + '</svg>'
    assert len(big_svg) > MAX_DOCUMENT_MEDIA_BYTES
    assert check_media_limits(big_svg) == DOCUMENT_MEDIA_TOO_LARGE

    # 6 MB of SVG source that CONTAINS a 2-MB data URI is 6 MB of media, not 8
    # — together with 4 MB of further URIs that is exactly the budget.
    uri = _data_uri(MAX_DATA_URI_BYTES)
    head = f'<svg viewBox="0 0 1 1"><desc>{uri}</desc>'
    svg = head + 'x' * (6 * 1024 * 1024 - len(head) - len('</svg>')) + '</svg>'
    assert len(svg) == 6 * 1024 * 1024
    text = f'{svg}\n\n![a]({uri})\n\n![b]({uri})'
    assert check_media_limits(text) is None
    assert check_media_limits(text + '\n\n![c](data:image/png;base64,AA)') \
        == DOCUMENT_MEDIA_TOO_LARGE


def test_limits_count_utf8_bytes_and_ignore_non_strings():
    # 1.2 M two-byte characters inside an SVG: 1.2 M chars, 2.4 MB — over 2 MB
    # as a whole text, so it is measured; and ten of them are over 10 MB.
    svg = '<svg viewBox="0 0 1 1"><text>' + 'ä' * 1_200_000 + '</text></svg>'
    assert check_media_limits(svg * 4) is None
    assert check_media_limits(svg * 5) == DOCUMENT_MEDIA_TOO_LARGE
    for junk in (None, 123, ['data:image/png;base64,AA'], {'a': 1}):
        assert check_media_limits(junk) is None


# --- strip_media_for_preview ---------------------------------------------------


def test_preview_of_the_probe_reads_as_prose():
    preview = strip_media_for_preview(_fixture('rich_media_probe_240.md'))
    for source in ('<svg ', '</svg>', '<rect', 'data:', '%3Csvg', 'flowchart',
                   '```', '<img alt'):
        assert source not in preview, source
    assert preview.startswith('# Render-Probe: SVG in Markdown\n\nDrei Varianten')
    # Prose that NAMES a tag is not media: both headings stay whole.
    assert '## Variante 1 — inline `<svg>`\n\n## Variante 2 — `<img>` mit data-URI' in preview
    assert 'Variante 3' in preview          # the Markdown image keeps its alt text
    assert '## Variante 4 — Mermaid (Codeblock)\n\nEnde der Probe.' in preview
    assert '\n\n\n' not in preview


def test_preview_of_a_document_that_opens_with_a_figure():
    text = ('<svg viewBox="0 0 10 10">\n\n<rect x="1" width="5" height="5"/>\n</svg>\n\n'
            '# Mesomerie\n\nDer erste Satz.')
    assert strip_media_for_preview(text)[:300] == '# Mesomerie\n\nDer erste Satz.'


def test_preview_strips_tilde_and_unclosed_mermaid_fences():
    assert strip_media_for_preview(
        'Davor.\n\n~~~~mermaid\nflowchart LR\n  A --> B\n~~~~\n\nDanach.') == 'Davor.\n\nDanach.'
    assert strip_media_for_preview(
        'Davor.\n\n```mermaid\nflowchart LR\n  A --> B\n') == 'Davor.\n\n'
    # Any other fence stays.
    code = 'Davor.\n\n```python\nmermaid = 1\n```\n'
    assert strip_media_for_preview(code) is code


def test_preview_of_a_media_free_text_is_the_same_object():
    for text in ('', '# Titel\n\nText mit <uuid> und `code`.\n\n\n\nViel Luft.',
                 'Das Wort mermaid und data: als Prosa.'):
        assert strip_media_for_preview(text) is text
    assert strip_media_for_preview(None) is None


# --- runtime on pathological input (Pflicht-Nachtrag nach Phase 1) -----------

# ``strip_media_for_preview`` runs for EVERY ROW of every library list and of
# the MCP's list_conversions — one degenerate document would stall every list
# it appears in. The first version: 4 000 lines of ``<svg `` → 946 ms, 8 000 →
# 3.7 s (×4 per doubling); measured after the fix: ≤ 15 ms for every form at
# 16 000 repeats. ``find_svg_spans`` is the same scan the renderer uses.
_REPEATS = 16000
_SCAN_BOUND_S = 1.0

_SCAN_FORMS = {
    'A_svg_never_closed': '<svg \n' * _REPEATS,
    'B_openers_outnumber_closers': '<svg><svg></svg>\n' * _REPEATS,
    'D_img_without_gt': '<img \n' * _REPEATS,
    'E_md_image_without_bracket': '![ \n' * _REPEATS,
    'E2_md_image_one_line': '![' * _REPEATS,
    'F_md_data_image_one_token': '![x](data:image/png;base64,' * _REPEATS,
    # Every "<b " walks to the one far quote, opens it, never closes it: the
    # miss that the last-">" shortcut cannot see — the budget's case.
    'M_every_tag_start_misses': '<b ' * _REPEATS + '" > <svg><rect/></svg>',
    'Q_unclosed_quoted_data_uris': '"data:image/png;base64,AAAA\n' * _REPEATS,
    'T_unclosed_mermaid_fences': '```mermaid\nflowchart LR\n' * (_REPEATS // 4),
    'K_control_plain_text': 'Zeile mit Text.\n' * _REPEATS,
}


@pytest.mark.parametrize('form', sorted(_SCAN_FORMS))
def test_scan_and_preview_time_is_bounded_on_pathological_input(form):
    text = _SCAN_FORMS[form]
    for fn in (find_svg_spans, strip_media_for_preview):
        started = time.perf_counter()
        fn(text)
        elapsed = time.perf_counter() - started
        assert elapsed < _SCAN_BOUND_S, f'{form} / {fn.__name__}: {elapsed:.2f}s'


def test_scanner_stops_cutting_when_the_miss_budget_is_spent_not_before():
    """The boundary itself: every ``<b `` walks to the one far quote and
    misses. One miss UNDER the budget, the figure behind them is still found;
    AT the budget the scan has stopped — fewer figures cut, by design."""
    from services.doc_media import _MAX_TAG_END_MISSES
    figure = '<svg><rect/></svg>'

    def text(misses):
        return '<b ' * misses + '" > ' + figure

    under = text(_MAX_TAG_END_MISSES - 1)
    assert [under[s:e] for s, e in find_svg_spans(under)[0]] == [figure]
    assert find_svg_spans(text(_MAX_TAG_END_MISSES))[0] == []
    # No ">" anywhere after the point: the exact stop, no budget involved.
    assert find_svg_spans('<svg ' * 50) == ([], [])


# --- the write paths: 413, nothing written -------------------------------------

CARD_TOKEN = 'rich-media-test-card-token-1d4c'
INGEST_TOKEN = 'rich-media-test-ingest-token-8a2f'
_OVER = f'# Dokument\n\n![x]({_data_uri(MAX_DATA_URI_BYTES + 1)})\n'


def _bearer(token):
    return {'Authorization': f'Bearer {token}'}


def _make_conversion(app, user_id, content='# Alt\n\nUrsprünglicher Text.', title='Alt'):
    with app.app_context():
        conv = Conversion(user_id=user_id, conversion_type='markdown_input',
                          title=title, content=content)
        db.session.add(conv)
        db.session.commit()
        return conv.id


def _row(app, cid):
    with app.app_context():
        conv = db.session.get(Conversion, cid)
        return conv.title, conv.content, conv.content_version, conv.is_favorite


def _count(app):
    with app.app_context():
        return Conversion.query.count()


def test_ingest_over_budget_is_413_and_creates_nothing(app, client, test_user, monkeypatch):
    monkeypatch.setenv('INGEST_TOKEN', INGEST_TOKEN)
    resp = client.post('/api/ingest/conversion', headers=_bearer(INGEST_TOKEN), json={
        'conversion_type': 'markdown_input', 'title': 'Zu groß', 'content': _OVER,
        'source_id': 'rich-media-over-budget'})
    assert resp.status_code == 413
    assert resp.get_json() == {'error': DATA_URI_TOO_LARGE}
    assert _count(app) == 0


def test_session_create_over_budget_is_413_and_creates_nothing(app, authenticated_client):
    resp = authenticated_client.post('/api/conversions', json={
        'conversion_type': 'markdown_input', 'title': 'Zu groß', 'content': _OVER})
    assert resp.status_code == 413
    assert resp.get_json() == {'error': DATA_URI_TOO_LARGE}
    assert _count(app) == 0


def test_editor_put_over_budget_is_413_and_applies_no_field(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    before = _row(app, cid)
    resp = authenticated_client.put(f'/api/conversions/{cid}', json={
        'title': 'Neuer Titel', 'is_favorite': True, 'content': _OVER})
    assert resp.status_code == 413
    assert resp.get_json() == {'error': DATA_URI_TOO_LARGE}
    assert _row(app, cid) == before  # not the title, not the star, not the text


def test_docwrite_full_over_budget_is_413_and_keeps_the_document(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    cid = _make_conversion(app, test_user['id'])
    before = _row(app, cid)
    resp = client.patch(f'/api/conversions/{cid}/content', headers=_bearer(CARD_TOKEN),
                        json={'content': _OVER})
    assert resp.status_code == 413
    assert resp.get_json() == {'error': DATA_URI_TOO_LARGE}
    assert _row(app, cid) == before


def test_docwrite_section_budget_is_measured_on_the_spliced_document(app, client, test_user, monkeypatch):
    """A section of 1 MB is fine by itself — but the budget is per document:
    spliced into one that already carries 9.5 MB of figures it tips it over."""
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    figure = f'![f]({_data_uri(1_900_000)})\n\n'
    original = '# Dokument\n\n## Bestand\n\n' + figure * 5 + '## Neu\n\nPlatz.\n'
    cid = _make_conversion(app, test_user['id'], content=original)
    before = _row(app, cid)

    resp = client.patch(f'/api/conversions/{cid}/section', headers=_bearer(CARD_TOKEN), json={
        'heading': 'Neu', 'content': f'## Neu\n\n![g]({_data_uri(1_000_000)})\n'})
    assert resp.status_code == 413
    assert resp.get_json() == {'error': DOCUMENT_MEDIA_TOO_LARGE}
    assert _row(app, cid) == before

    # The same section into the same document with room left goes through.
    ok = client.patch(f'/api/conversions/{cid}/section', headers=_bearer(CARD_TOKEN), json={
        'heading': 'Neu', 'content': f'## Neu\n\n![g]({_data_uri(400_000)})\n'})
    assert ok.status_code == 200
    assert _row(app, cid)[2] == before[2] + 1


@pytest.mark.parametrize('route', ['ingest', 'create', 'put', 'docwrite'])
def test_documents_within_budget_still_write(app, client, test_user, monkeypatch, route):
    monkeypatch.setenv('INGEST_TOKEN', INGEST_TOKEN)
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    content = _fixture('rich_media_probe_240.md')
    if route == 'ingest':
        resp = client.post('/api/ingest/conversion', headers=_bearer(INGEST_TOKEN), json={
            'conversion_type': 'markdown_input', 'title': 'Probe', 'content': content})
        assert resp.status_code == 201
    elif route == 'docwrite':
        cid = _make_conversion(app, test_user['id'])
        resp = client.patch(f'/api/conversions/{cid}/content', headers=_bearer(CARD_TOKEN),
                            json={'content': content})
        assert resp.status_code == 200
    else:
        login = client.post('/login', data={'username': test_user['username'],
                                            'password': test_user['password']})
        assert login.status_code == 302
        if route == 'create':
            resp = client.post('/api/conversions', json={
                'conversion_type': 'markdown_input', 'title': 'Probe', 'content': content})
            assert resp.status_code == 201
        else:
            cid = _make_conversion(app, test_user['id'])
            resp = client.put(f'/api/conversions/{cid}', json={'content': content})
            assert resp.status_code == 200
    assert resp.get_json()['content'] == content


def test_audio_upload_keeps_its_500_mb(app):
    # The media budget lives in the content routes, not in the request limit.
    assert app.config['MAX_CONTENT_LENGTH'] == 500 * 1024 * 1024


# --- the list preview ------------------------------------------------------------


def test_list_preview_is_media_free_and_content_length_stays_raw(app, authenticated_client, test_user):
    content = _fixture('rich_media_probe_240.md')
    _make_conversion(app, test_user['id'], content=content, title='Probe')
    plain = '# Nur Text\n\nOhne Figuren.'
    _make_conversion(app, test_user['id'], content=plain, title='Text')

    items = {i['title']: i for i in authenticated_client.get('/api/conversions').get_json()['items']}
    probe = items['Probe']
    assert probe['content_length'] == len(content)  # raw, every embedded byte
    assert len(probe['content_preview']) <= 300
    assert '<svg ' not in probe['content_preview']  # the heading's `<svg>` is prose
    assert '<rect' not in probe['content_preview'] and 'data:' not in probe['content_preview']
    assert probe['content_preview'].startswith('# Render-Probe: SVG in Markdown')
    assert items['Text']['content_preview'] == plain
