#!/usr/bin/env python3
"""Browser smoke for ARCH-LIBRARY-KLEIN — the type vocabulary in list and detail.

The pytest suite renders the templates but never computes a style — this is
the one REAL-browser check that every allowed ``conversion_type`` shows up in
``/library`` (list) and ``/library/<id>`` (detail) with a label, a tone and a
filter path, and that the list preview of a document that starts with a
figure reads as prose. It runs with Playwright INSIDE the app image (the
playwright base image ships Chromium) as a throwaway user with its OWN rows
written through the ORM — one Conversion per type in
``app_pkg.library.ALLOWED_CONVERSION_TYPES`` plus one ``markdown_input``
whose content BEGINS with an ``<svg>`` block — and MEASURES, in BOTH themes
(``localStorage.globalTheme`` = ``light`` / ``dark``, the mechanism of
``templates/base.html``):

* per row in the list (``/library?view=archiv``): the badge text, its class
  list, the computed ``background-color`` and ``color``; the computed
  background of a bare ``.type-badge`` span injected for reference ("Basis");
* the ``<option>`` values + labels of the type filter (``/library?view=bibliothek``);
* the preview text of the figure document and of a plain document (first
  60 characters, and whether ``<svg`` occurs);
* per row on the detail page: the badge text and its computed background.

Two modes, chosen with ``--expect``:

``--expect head`` (default, Phase 1): the findings are the Master's
derivations FROM THE CODE; the smoke prints "Ist bestätigt" / "Ist widerlegt"
per finding and exits 0 either way — it is the evidence, not the gate:
  F1 ``document_conversion`` list badge has the SAME background as the bare
     ``.type-badge`` (no tone — ``.type-document_conversion`` has no CSS rule);
  F2 ``audio_narration`` list badge shows the raw string ``audio_narration``;
  F3 ``audio_narration`` DETAIL badge shows the raw string (Master: "prüfen");
  F4 the type filter has five non-empty options and none for ``audio_narration``;
  F5 the figure document's list preview contains ``<svg``.
  Measured 2026-10-08 on ``fa1eda8aac07`` (two runs, identical, both
  themes): F1, F2, F4, F5 confirmed; F3 REFUTED — the detail template
  already labels ``audio_narration`` "Vertonung" (its slate tone shows on
  list and detail; only the list label and the filter option are missing).
  Beyond the derivation: ``document_conversion`` is toneless on the DETAIL
  page too (same class, same missing rule); the "..." ellipsis of a preview
  follows the RAW length (203 = 200 + "..." on the figure document).

``--expect fixed`` (Phase 2 gate, Phase 3 / acceptance): the same
measurements with the reversed expectation, PASS/FAIL, exit 1 on any FAIL:
  every type's badge text differs from the raw type string (list AND detail);
  every type's badge background differs from the bare base (every badge has
  a tone); ``document_conversion`` carries the SAME background as
  ``document_to_markdown`` (both are "Dokument"); ``audio_narration`` reads
  "Vertonung" in list and detail; the filter has six non-empty options, one
  of them ``audio_narration``; the figure preview has no ``<svg`` and is not
  empty; the plain preview starts with its first prose characters.

Test rows are written through the ORM under the throwaway user's own
``user_id`` — never through ``POST /api/conversions``. The script REFUSES to
run for user id 1 or the INGEST_USER. Everything it wrote is removed at the
end, strictly by ``user_id`` + the ids it wrote; the user itself is created
and removed by the operator.

How to run — (a) WITHOUT a deploy, on a throwaway instance from the deployed
image with the Mac working tree streamed in (own SQLite under /tmp/w, no
volume, no network; ~1 min). ``zz_first`` is a placeholder so the smoke user
is not user id 1 — the id-1 guard below is the same on every instance:

    cd ~/CODE/CONVERTER && git ls-files -co --exclude-standard | grep -v -E '^(corpus|docs)/' > /tmp/lt_files \
    && COPYFILE_DISABLE=1 tar --no-xattrs -cf - -T /tmp/lt_files | ssh mintbox 'docker run -i --rm --network none --entrypoint sh converter-app:latest -c "
        mkdir /tmp/w && tar -xf - -C /tmp/w && cd /tmp/w \
        && export DATABASE_URL=sqlite:////tmp/w/smoke.db SECRET_KEY=smoke-only REDIS_URL=redis://127.0.0.1:9/0 \
        && flask --app app create-user zz_first --password placeholder-1234 >/dev/null \
        && flask --app app create-user zz_types --password smoke-pass-1234 >/dev/null \
        && (gunicorn --bind 127.0.0.1:5000 --workers 1 --worker-class uvicorn.workers.UvicornWorker app:asgi_app >/tmp/w/gunicorn.log 2>&1 &) \
        && sleep 5 \
        && SMOKE_USER=zz_types SMOKE_PASSWORD=smoke-pass-1234 BASE_URL=http://127.0.0.1:5000 python scripts/smoke_library_types.py --expect head"'

(b) against the DEPLOYED web container (Phase 3 / acceptance, ``--expect fixed``):

    docker exec markdown-converter-web flask --app app create-user zz_types --password '<random>'
    docker exec -i markdown-converter-web sh -c 'cat > /tmp/smoke_library_types.py' < scripts/smoke_library_types.py
    docker exec -e SMOKE_USER=zz_types -e SMOKE_PASSWORD='<random>' markdown-converter-web python /tmp/smoke_library_types.py --expect fixed
    # clean up STRICTLY by user_id (api_token carries Oli's iOS tokens): the
    # script removed its rows; delete the user's remaining Conversion/ApiToken
    # rows and the User row via the ORM by that user_id, then:
    #   docker exec markdown-converter-web sh -c 'rm -f /tmp/smoke_library_types.py; rm -rf /tmp/pulse-*'

Env: BASE_URL (default http://localhost:5000), SMOKE_USER, SMOKE_PASSWORD,
SMOKE_APP_ROOT (default: cwd — where ``app.py`` lives). No screenshots are
written. Every measured value is printed so a verdict is diagnosable from
the output alone.
"""
import argparse
import json
import os
import sys

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
parser.add_argument('--expect', choices=('head', 'fixed'), default='head',
                    help="'head' = confirm/refute the derivations (Phase 1, exit 0); "
                         "'fixed' = the gate after the fixes (PASS/FAIL, exit 1 on FAIL)")
ARGS = parser.parse_args()

BASE = os.environ.get('BASE_URL', 'http://localhost:5000')
USER = os.environ.get('SMOKE_USER') or sys.exit('SMOKE_USER missing')
PASSWORD = os.environ.get('SMOKE_PASSWORD') or sys.exit('SMOKE_PASSWORD missing')

THEMES = ('light', 'dark')
NARRATION_LABEL = 'Vertonung'          # the word the detail page already uses
FIGURE_KEY = 'figure'                  # the extra markdown_input starting with <svg>
PLAIN_PROSE = 'Smoke-Dokument ohne Figur: die ersten Zeichen sind Prosa'

# A closed figure longer than the 200-character raw cut, so a raw preview is
# SVG markup from the first to the last character.
FIGURE_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 80" width="320" height="80" role="img" aria-label="Smoke-Figur">'
              '<rect x="4" y="4" width="312" height="72" fill="#eef" stroke="#336" stroke-width="2"/>'
              '<rect x="20" y="20" width="60" height="40" fill="#99c"/>'
              '<rect x="100" y="20" width="60" height="40" fill="#9c9"/>'
              '<rect x="180" y="20" width="60" height="40" fill="#c99"/>'
              '<text x="260" y="46" font-size="14" fill="#333">Figur</text>'
              '</svg>')
FIGURE_PROSE = 'Figur-Dokument des Smokes: dieser Satz folgt auf die Figur und ist die Vorschau, die ein Leser sehen soll.'

failures = []
verdicts = []


def check(ok, what):
    print(('  PASS ' if ok else '  FAIL ') + what)
    if not ok:
        failures.append(what)


def verdict(ok, finding):
    """Phase-1 form: the Master derived a finding from the code; the browser
    confirms or refutes it. Informational — never fails the run."""
    print(('  Ist bestätigt  ' if ok else '  Ist widerlegt  ') + finding)
    verdicts.append((ok, finding))


# --- test data through the ORM (same container, same DB) -------------------
sys.path.insert(0, os.environ.get('SMOKE_APP_ROOT', os.getcwd()))
from app import app  # noqa: E402
from app_pkg.library import ALLOWED_CONVERSION_TYPES  # noqa: E402
from models import Conversion, User, db  # noqa: E402

TYPES = sorted(ALLOWED_CONVERSION_TYPES)

# Terminal job states so the detail page's pollers (narration / transcription
# / document job) find nothing to reconcile and the page just renders.
METADATA = {
    'audio_narration': {'narration_status': 'failed', 'error': 'smoke: kein Render'},
    'audio_transcription': {'transcription_status': 'ready'},
    'document_conversion': {'doc_status': 'ready'},
}


def resolve_user():
    with app.app_context():
        user = User.query.filter_by(username=USER).first()
        if user is None:
            sys.exit(f'user {USER!r} does not exist — create it with flask create-user')
        if user.id == 1 or USER == os.environ.get('INGEST_USER'):
            sys.exit(f'refusing to run as user id {user.id} ({USER!r}): that is the '
                     'INGEST/first user = Oli\'s account')
        return user.id


def create_rows(user_id):
    """One archived Conversion per allowed type (content = plain prose) plus
    the figure document (markdown_input, content starts with the <svg>)."""
    ids = {}
    with app.app_context():
        for t in TYPES:
            row = Conversion(user_id=user_id, conversion_type=t,
                             title=f'ARCH-LIBRARY-KLEIN Smoke {t}',
                             content=f'{PLAIN_PROSE} ({t}). ' + 'Weiterer Text. ' * 20,
                             lifecycle_status='archive',
                             metadata_json=json.dumps(METADATA.get(t, {})))
            db.session.add(row)
            db.session.flush()
            ids[t] = row.id
        fig = Conversion(user_id=user_id, conversion_type='markdown_input',
                         title='ARCH-LIBRARY-KLEIN Smoke Figur-Dokument',
                         content=FIGURE_SVG + '\n\n' + FIGURE_PROSE + '\n',
                         lifecycle_status='archive', metadata_json='{}')
        db.session.add(fig)
        db.session.flush()
        ids[FIGURE_KEY] = fig.id
        db.session.commit()
    return ids


def remove_rows(ids, user_id):
    with app.app_context():
        rows = Conversion.query.filter(Conversion.user_id == user_id,
                                       Conversion.id.in_(list(ids.values()))).all()
        for r in rows:
            db.session.delete(r)
        db.session.commit()
        left = Conversion.query.filter(Conversion.user_id == user_id,
                                       Conversion.id.in_(list(ids.values()))).count()
        return len(rows), left


# --- page helpers ----------------------------------------------------------
LIST_STATE = """(ids) => {
  const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
  const probe = document.createElement('span');
  probe.className = 'type-badge';
  document.body.appendChild(probe);
  const pcs = getComputedStyle(probe);
  const base = {bg: pcs.backgroundColor, color: pcs.color};
  probe.remove();
  const rows = {};
  for (const [key, id] of Object.entries(ids)) {
    const card = document.querySelector(`.c-card[data-id="${id}"]`);
    if (!card) { rows[key] = null; continue; }
    const badge = card.querySelector('.type-badge');
    const cs = getComputedStyle(badge);
    const preview = norm((card.querySelector('p.line-clamp-3') || {}).textContent);
    rows[key] = {
      label: norm(badge.textContent),
      classes: badge.className,
      bg: cs.backgroundColor,
      color: cs.color,
      preview60: preview.slice(0, 60),
      previewLen: preview.length,
      previewHasSvg: preview.includes('<svg'),
    };
  }
  return {theme: document.documentElement.getAttribute('data-global-theme') || 'light',
          base, rows, cardsOnPage: document.querySelectorAll('.c-card[data-id]').length};
}"""

FILTER_STATE = """() => {
  const sel = document.querySelector('select[name=type]');
  if (!sel) return null;
  return Array.from(sel.options).map(o => ({value: o.value, label: o.textContent.trim()}));
}"""

DETAIL_STATE = """() => {
  const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
  const badge = document.querySelector('.type-badge');
  if (!badge) return null;
  const cs = getComputedStyle(badge);
  return {label: norm(badge.textContent), classes: badge.className,
          bg: cs.backgroundColor, color: cs.color,
          theme: document.documentElement.getAttribute('data-global-theme') || 'light'};
}"""


def login(page):
    page.goto(f'{BASE}/login')
    page.fill('#username', USER)
    page.fill('#password', PASSWORD)
    page.click('button[type=submit]')
    page.wait_for_url(lambda url: '/login' not in url)


def measure_theme(browser, theme, ids):
    # The real mechanism: base.html's inline script reads localStorage before
    # the first paint. An init script sets the key before any page script.
    ctx = browser.new_context(viewport={'width': 1200, 'height': 900})
    ctx.add_init_script(f"localStorage.setItem('globalTheme', '{theme}');")
    page = ctx.new_page()
    login(page)

    page.goto(f'{BASE}/library?view=archiv&per_page=50')
    page.wait_for_selector('.c-card[data-id]')
    lst = page.evaluate(LIST_STATE, ids)

    page.goto(f'{BASE}/library?view=bibliothek')
    page.wait_for_selector('select[name=type]')
    options = page.evaluate(FILTER_STATE)

    detail = {}
    for key, cid in ids.items():
        page.goto(f'{BASE}/library/{cid}')
        page.wait_for_selector('.type-badge')
        detail[key] = page.evaluate(DETAIL_STATE)
    ctx.close()
    return {'list': lst, 'options': options, 'detail': detail}


def print_table(theme, m):
    lst, options, detail = m['list'], m['options'], m['detail']
    print(f'--- theme={theme} (html[data-global-theme]={lst["theme"]!r}; cards on page: {lst["cardsOnPage"]}) ---')
    print(f'  Basis .type-badge: bg={lst["base"]["bg"]} color={lst["base"]["color"]}')
    print(f'  {"row":22} {"list label":16} {"list bg":26} {"detail label":16} {"detail bg":26}')
    for key in TYPES + [FIGURE_KEY]:
        r, d = lst['rows'].get(key), detail.get(key)
        rl = r['label'] if r else '<missing>'
        rb = r['bg'] if r else '-'
        dl = d['label'] if d else '<missing>'
        db_ = d['bg'] if d else '-'
        print(f'  {key:22} {rl:16} {rb:26} {dl:16} {db_:26}')
    non_empty = [o for o in options if o['value']] if options else []
    print(f'  filter options ({len(non_empty)} non-empty): ' +
          ', '.join(f'{o["value"]}="{o["label"]}"' for o in options or []))
    fig, plain = lst['rows'].get(FIGURE_KEY), lst['rows'].get('markdown_input')
    print(f'  preview figure  (len {fig["previewLen"] if fig else "-"}, <svg: {fig["previewHasSvg"] if fig else "-"}): '
          f'{fig["preview60"]!r}' if fig else '  preview figure: <missing>')
    print(f'  preview plain   (len {plain["previewLen"] if plain else "-"}, <svg: {plain["previewHasSvg"] if plain else "-"}): '
          f'{plain["preview60"]!r}' if plain else '  preview plain: <missing>')


def judge_head(theme, m):
    lst, options, detail = m['list'], m['options'], m['detail']
    base_bg = lst['base']['bg']
    dc, an = lst['rows']['document_conversion'], lst['rows']['audio_narration']
    non_empty = [o['value'] for o in options if o['value']] if options else []
    fig = lst['rows'][FIGURE_KEY]
    print(f'--- Befunde, hergeleitet vs. gesehen, theme={theme} ---')
    verdict(dc['bg'] == base_bg,
            f'F1 document_conversion list badge == Basis-Hintergrund (kein Ton): '
            f'gesehen {dc["bg"]} vs Basis {base_bg}')
    verdict(an['label'] == 'audio_narration',
            f'F2 audio_narration list badge zeigt den rohen Typ-String: gesehen {an["label"]!r}')
    verdict(detail['audio_narration']['label'] == 'audio_narration',
            f'F3 audio_narration DETAIL badge zeigt den rohen Typ-String: gesehen {detail["audio_narration"]["label"]!r}')
    verdict(len(non_empty) == 5 and 'audio_narration' not in non_empty,
            f'F4 Typ-Filter hat fünf Optionen ohne audio_narration: gesehen {len(non_empty)} {non_empty}')
    verdict(fig['previewHasSvg'],
            f'F5 Vorschau des Figur-Dokuments enthält "<svg": gesehen {fig["preview60"]!r}')


def judge_fixed(theme, m):
    lst, options, detail = m['list'], m['options'], m['detail']
    base_bg = lst['base']['bg']
    print(f'--- gate, theme={theme} ---')
    check(lst['theme'] == theme, f'the page runs in the {theme} theme (data-global-theme={lst["theme"]!r})')
    for t in TYPES:
        r, d = lst['rows'][t], detail[t]
        check(r is not None and d is not None, f'{t}: row on the list and detail page')
        check(r['label'] != t and r['label'] != '', f'{t}: list label is a word, not the type string ({r["label"]!r})')
        check(d['label'] == r['label'], f'{t}: detail label == list label ({d["label"]!r})')
        check(r['bg'] != base_bg, f'{t}: list badge has a tone (bg {r["bg"]} ≠ Basis {base_bg})')
        check(d['bg'] == r['bg'], f'{t}: detail badge bg == list badge bg ({d["bg"]})')
    dc, dm = lst['rows']['document_conversion'], lst['rows']['document_to_markdown']
    check(dc['bg'] == dm['bg'] and dc['label'] == dm['label'] == 'Dokument',
          f'document_conversion carries the Dokument tone of document_to_markdown ({dc["bg"]} == {dm["bg"]}, "{dc["label"]}")')
    check(lst['rows']['audio_narration']['label'] == NARRATION_LABEL,
          f'audio_narration list label == {NARRATION_LABEL!r} ({lst["rows"]["audio_narration"]["label"]!r})')
    check(detail['audio_narration']['label'] == NARRATION_LABEL,
          f'audio_narration detail label == {NARRATION_LABEL!r} ({detail["audio_narration"]["label"]!r})')
    non_empty = {o['value']: o['label'] for o in options if o['value']} if options else {}
    check(len(non_empty) == 6 and non_empty.get('audio_narration') == NARRATION_LABEL,
          f'type filter: six non-empty options, audio_narration="{NARRATION_LABEL}" ({non_empty})')
    fig, plain = lst['rows'][FIGURE_KEY], lst['rows']['markdown_input']
    check(not fig['previewHasSvg'] and fig['previewLen'] > 0,
          f'figure preview reads as prose, no "<svg" ({fig["preview60"]!r})')
    check(fig['preview60'].startswith(FIGURE_PROSE[:60]),
          f'figure preview starts with the prose after the figure')
    check(plain['preview60'].startswith(PLAIN_PROSE[:60]),
          f'plain preview starts with its first prose characters ({plain["preview60"]!r})')


user_id = resolve_user()
ids = create_rows(user_id)
print(f'mode=--expect {ARGS.expect} user_id={user_id} rows={json.dumps(ids)}')

try:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        measured = {theme: measure_theme(browser, theme, ids) for theme in THEMES}
        browser.close()
finally:
    removed, left = remove_rows(ids, user_id)
    print(f'cleanup: removed {removed} rows of user_id={user_id} (left: {left})')

for theme in THEMES:
    print_table(theme, measured[theme])
print()
for theme in THEMES:
    if ARGS.expect == 'head':
        judge_head(theme, measured[theme])
    else:
        judge_fixed(theme, measured[theme])

print()
if ARGS.expect == 'head':
    confirmed = sum(1 for ok, _ in verdicts if ok)
    print(f'{confirmed} von {len(verdicts)} Herleitungen bestätigt, {len(verdicts) - confirmed} widerlegt (kein Gate — exit 0)')
    sys.exit(0)
if failures:
    print(f'{len(failures)} FAILED:')
    for f in failures:
        print('  -', f)
    sys.exit(1)
print('ALL PASSED')
