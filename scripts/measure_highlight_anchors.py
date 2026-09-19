#!/usr/bin/env python3
"""Where do a document's highlights sit in the REAL reader? (RICH-MEDIA 3.1)

A highlight is a text-quote selector (``exact``/``prefix``/``suffix``) that the
reader re-finds in ``readerRawText`` on every open. A change to the renderer or
to the reader's DOM can move that coordinate system without any test noticing —
the anchors are data, the text nodes are a by-product. This measures it, on the
real page with the real JS, for one existing document:

* run it BEFORE a deploy and AFTER it, diff the two JSON files — identical
  ``start``/``end`` per highlight and an identical ``raw_sha256`` is the claim
  "no anchor of this document moved";
* it never touches the owner's session or password: the document and its
  highlights are COPIED (read from the owner's rows, written under the
  throwaway user's ``user_id``) through the ORM, opened as the throwaway user,
  and the copy is removed again, strictly by ``user_id`` + the id it wrote.
  The script REFUSES to run as user id 1 or the INGEST_USER.

Per highlight it records what the page's own ``locateHighlightOffset`` answers
(``start``/``end`` raw offsets, or ``null`` = not found) and how many
``span.highlight`` elements carry it; on a stand with Mermaid rendering
(``static/js/reader_figures.js``) it scrolls every fence into view and waits
for all of them first, so "after" really is after.

How to run (Mintbox; user recipe as in scripts/smoke_reader_media.py):

    docker cp scripts/measure_highlight_anchors.py markdown-converter-web:/tmp/measure_anchors.py
    docker exec -e SMOKE_USER=zz_media -e SMOKE_PASSWORD='<random>' -e SOURCE_ID=133 \\
        -e ANCHOR_OUT=/tmp/anchors_133_before.json markdown-converter-web python /tmp/measure_anchors.py
    docker cp markdown-converter-web:/tmp/anchors_133_before.json .

Env: BASE_URL (default http://localhost:5000), SMOKE_USER, SMOKE_PASSWORD,
SOURCE_ID, ANCHOR_OUT (default /tmp/anchors_<id>.json), SMOKE_APP_ROOT. The
JSON carries offsets, lengths and hashes — no document text. Exit 1 if the
copy could not be measured or not removed.
"""
import hashlib
import json
import os
import sys

from playwright.sync_api import sync_playwright

BASE = os.environ.get('BASE_URL', 'http://localhost:5000')
USER = os.environ.get('SMOKE_USER') or sys.exit('SMOKE_USER missing')
PASSWORD = os.environ.get('SMOKE_PASSWORD') or sys.exit('SMOKE_PASSWORD missing')
SOURCE_ID = int(os.environ.get('SOURCE_ID') or sys.exit('SOURCE_ID missing'))
OUT = os.environ.get('ANCHOR_OUT', f'/tmp/anchors_{SOURCE_ID}.json')

sys.path.insert(0, os.environ.get('SMOKE_APP_ROOT', os.getcwd()))
from app import app  # noqa: E402
from models import Conversion, Highlight, User, db  # noqa: E402


def copy_document():
    """Copy SOURCE_ID + its highlights under the throwaway user. Returns
    (copy_id, user_id, {copy_highlight_id: source_highlight_id})."""
    with app.app_context():
        user = User.query.filter_by(username=USER).first()
        if user is None:
            sys.exit(f'user {USER!r} does not exist — create it with flask create-user')
        if user.id == 1 or USER == os.environ.get('INGEST_USER'):
            sys.exit(f'refusing to run as user id {user.id} ({USER!r}): that is Oli\'s account')
        source = db.session.get(Conversion, SOURCE_ID)
        if source is None:
            sys.exit(f'conversion {SOURCE_ID} does not exist')
        copy = Conversion(user_id=user.id, conversion_type=source.conversion_type,
                          title=f'ANCHOR-MEASURE copy of #{SOURCE_ID}', content=source.content)
        db.session.add(copy)
        db.session.flush()
        id_map = {}
        for hl in source.highlights.order_by(Highlight.id):
            twin = Highlight(conversion_id=copy.id, exact=hl.exact,
                             prefix=hl.prefix or '', suffix=hl.suffix or '')
            db.session.add(twin)
            db.session.flush()
            id_map[twin.id] = hl.id
        db.session.commit()
        return copy.id, user.id, id_map


def remove_copy(copy_id, user_id):
    with app.app_context():
        copy = Conversion.query.filter_by(id=copy_id, user_id=user_id).first()
        if copy is not None:
            db.session.delete(copy)  # highlights cascade ORM-side
            db.session.commit()
        return (Conversion.query.filter_by(id=copy_id).count(),
                Highlight.query.filter_by(conversion_id=copy_id).count())


MEASURE = """async () => {
  const reader = highlightReaderEl();
  const raw = readerRawText(reader);
  const list = await (await fetch(`/api/conversions/${window.PageData.conversionId}/highlights`)).json();
  const spans = {};
  reader.querySelectorAll('span.highlight[data-highlight-id]').forEach(s => {
    spans[s.dataset.highlightId] = (spans[s.dataset.highlightId] || 0) + 1;
  });
  const hosts = [...reader.querySelectorAll('.reader-mermaid')];
  return {
    raw,
    highlights: list.map(h => {
      const loc = locateHighlightOffset(reader, h);
      return { id: h.id, start: loc ? loc.start : null, end: loc ? loc.end : null,
               exact_len: h.exact.length, spans: spans[String(h.id)] || 0 };
    }),
    fences: reader.querySelectorAll('pre > code.language-mermaid').length,
    mermaid_states: hosts.map(h => h.dataset.mermaidState),
    marked_figures: hosts.filter(h => h.classList.contains('reader-mermaid--marked')).length,
  };
}"""

copy_id, user_id, id_map = copy_document()
print(f'copied #{SOURCE_ID} -> #{copy_id} under user_id={user_id} with {len(id_map)} highlights')
result = None
try:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_context(viewport={'width': 1200, 'height': 900}).new_page()
        page.goto(f'{BASE}/login')
        page.fill('input[name=username]', USER)
        page.fill('input[name=password]', PASSWORD)
        page.click('button[type=submit]')
        page.wait_for_url(lambda url: '/login' not in url)
        page.goto(f'{BASE}/library/{copy_id}', wait_until='domcontentloaded')
        page.wait_for_selector('.reader-view')
        # Stand with Mermaid rendering: bring every fence into view, wait for all.
        if page.evaluate("() => !!window.ReaderFigures"):
            page.evaluate("""async () => {
                for (const pre of document.querySelectorAll('.reader-view pre[data-mermaid-bound]')) {
                    (pre.previousElementSibling || pre).scrollIntoView({block: 'center'});
                    await new Promise(r => setTimeout(r, 400));
                }
                window.scrollTo(0, 0);
            }""")
            page.wait_for_function(
                """() => [...document.querySelectorAll('.reader-view .reader-mermaid')]
                          .every(h => ['rendered', 'failed'].includes(h.dataset.mermaidState))""",
                timeout=60000)
        page.wait_for_timeout(2500)  # highlights load async after DOMContentLoaded
        result = page.evaluate(MEASURE)
        browser.close()
finally:
    left = remove_copy(copy_id, user_id)
    print(f'removed the copy: conversions left={left[0]}, highlights left={left[1]}')

if result is None or left != (0, 0):
    sys.exit(1)

raw = result.pop('raw')
report = {
    'source_id': SOURCE_ID,
    'raw_chars': len(raw),
    'raw_sha256': hashlib.sha256(raw.encode('utf-8')).hexdigest(),
    # A highlight's identity across runs is its SOURCE id, not the copy's.
    'highlights': sorted(({**h, 'id': id_map[h['id']]} for h in result['highlights']),
                         key=lambda h: h['id']),
    **{k: result[k] for k in ('fences', 'mermaid_states', 'marked_figures')},
}
with open(OUT, 'w', encoding='utf-8') as fh:
    json.dump(report, fh, indent=1, sort_keys=True)
found = sum(1 for h in report['highlights'] if h['start'] is not None)
print(f'raw chars={report["raw_chars"]} sha256={report["raw_sha256"][:16]}… '
      f'highlights={len(report["highlights"])} located={found} '
      f'with spans={sum(1 for h in report["highlights"] if h["spans"])} '
      f'fences={report["fences"]} mermaid={report["mermaid_states"]} marked={report["marked_figures"]}')
print(f'written: {OUT}')
