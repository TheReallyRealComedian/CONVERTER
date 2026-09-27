"""DOC-LOCAL P1 — the mineru engine behind page_fn; since SEC-SOCKET the
worker side of the launcher.

The worker holds no docker socket: it writes the run's input into the
exchange and asks the launcher over HTTP. These tests drive the REAL
launcher (``fake_launcher`` in conftest: stdlib server on a loopback port)
with only the launcher's own process boundary faked — the docker CLI. So a
test here exercises worker → HTTP → validation → argv from the launcher's
env → (fake) daemon → content_list → page assembly, end to end. The PDFs
are REAL two-page PyMuPDF documents with text layers, so the sub-PDF
cutting and the text-layer fallback run for real — what the failure path
serves is actual page text, not a placeholder.

The pure vector sentinel lives in ``tests/test_mineru_invocation.py``
(``build_run_argv``); here the same pairs are checked on the argv the
daemon would receive for a worker request — the chain, not the builder.
"""
import json
import socket
from pathlib import Path

import pytest

import fitz

from services import mineru_launcher, pdf_local
from services.mineru_invocation import MINERU_MAX_PAGES
from services.pdf_local import (
    LocalPdfEngine,
    content_list_to_pages,
    is_scanned_page,
    mineru_run_timeout_for,
    run_local_pdf,
)


def _two_page_pdf(tmp_path, texts=('Seite eins Text.', 'Seite zwei Text.')):
    pdf = fitz.open()
    for text in texts:
        page = pdf.new_page()
        page.insert_text((72, 100), text)
    path = tmp_path / 'doc.pdf'
    pdf.save(str(path))
    pdf.close()
    return str(path)


def _vol_source(cmd, container_suffix):
    """The host side of the ``-v`` arg whose container side matches."""
    for arg in cmd:
        if isinstance(arg, str) and arg.endswith(container_suffix):
            return arg[: -len(container_suffix)]
    return None


def _mineru_calls(state):
    return state['runs']()


# -- assembly (pure, no container) ------------------------------------------

def test_entry_markdown_semantics():
    md = pdf_local._entry_markdown
    assert md({'type': 'text', 'text': 'Titel', 'text_level': 1}) == '# Titel'
    assert md({'type': 'text', 'text': 'Absatz.'}) == 'Absatz.'
    # Locked decision 5: page furniture is KEPT as plain paragraphs
    # (mineru's own .md drops it — measured 2026-08-16 on 03_gold).
    assert md({'type': 'header', 'text': 'AGOF'}) == 'AGOF'
    assert md({'type': 'footer', 'text': '© AGOF e.V.'}) == '© AGOF e.V.'
    assert md({'type': 'page_number', 'text': 'Seite 11'}) == 'Seite 11'
    assert md({'type': 'header', 'text': '  '}) == ''  # empty stays out
    # Equation text already carries its $$ delimiters — verbatim.
    assert md({'type': 'equation', 'text': '$$\nE=mc^2\n$$'}) == '$$\nE=mc^2\n$$'


def test_entry_markdown_table_keeps_html_body():
    entry = {
        'type': 'table',
        'table_caption': ['TABLE I. Netze.'],
        'table_body': '<table><tr><td rowspan="2">a</td><td>b</td></tr></table>',
        'table_footnote': ['* undirected'],
    }
    out = pdf_local._entry_markdown(entry)
    assert out.splitlines()[0] == 'TABLE I. Netze.'
    assert 'rowspan="2"' in out  # merged cells travel untouched (04-Messung)
    assert out.rstrip().endswith('* undirected')


def test_entry_markdown_image_with_description():
    entry = {
        'type': 'image',
        'img_path': 'images/abc.jpg',
        'image_caption': ['FIG. 1. Netzstruktur.'],
        'image_footnote': [],
        'content': '```mermaid\ngraph TD\n```',
        'sub_type': 'flowchart',
    }
    out = pdf_local._entry_markdown(entry)
    assert 'FIG. 1. Netzstruktur.' in out
    assert '![](images/abc.jpg)' in out
    assert '<details>\n<summary>flowchart</summary>' in out
    # Without a description there is no details block:
    bare = pdf_local._entry_markdown({'type': 'image', 'img_path': 'x.jpg'})
    assert bare == '![](x.jpg)'


def test_content_list_grouping_with_offset_and_blank_pages():
    entries = [
        {'type': 'text', 'text': 'Auf Seite N.', 'page_idx': 0},
        {'type': 'text', 'text': 'Auf Seite N+2.', 'page_idx': 2},
        {'type': 'text', 'text': 'ausserhalb', 'page_idx': 99},
        {'type': 'text', 'text': 'kaputt', 'page_idx': None},
    ]
    pages = content_list_to_pages(entries, 3, 6)
    # Every page of the range answers — a silent page is '', not a KeyError.
    assert sorted(pages) == [3, 4, 5]
    assert pages[3] == 'Auf Seite N.'
    assert pages[4] == ''
    assert pages[5] == 'Auf Seite N+2.'


# -- the memoized run + contract ---------------------------------------------

def test_full_local_run_serves_pages_from_one_container_run(fake_launcher, tmp_path):
    path = _two_page_pdf(tmp_path)
    fake_launcher['content_list'] = [
        {'type': 'text', 'text': 'Erste mineru-Seite.', 'page_idx': 0},
        {'type': 'footer', 'text': 'Seite 2 Fusszeile', 'page_idx': 1},
    ]
    payload = run_local_pdf(path, 2)
    assert payload['markdown'] == 'Erste mineru-Seite.\n\nSeite 2 Fusszeile'
    assert payload['provenance_unit'] == 'page'
    assert payload['provenance'] == ['modell', 'modell']
    assert payload['degradations'] == []
    # model_calls counts PAID cloud calls — a local VLM is not one.
    assert payload['usage'] == {'model_calls': 0, 'cost_eur': 0.0}
    assert len(_mineru_calls(fake_launcher)) == 1  # one run, both pages served
    # SEC-SOCKET: copy in, run, copy out, remove the job's volumes.
    assert fake_launcher['kinds']() == ['copy_in', 'run', 'copy_out', 'volume_rm']
    assert not fake_launcher['volumes'].exists() or not any(
        fake_launcher['volumes'].iterdir())


def test_worker_request_reaches_the_measured_vector(fake_launcher, tmp_path):
    """The chain, end to end: what the worker sends (data) becomes the
    measured bake-off vector at the daemon (mineru 3.4.4, vlm-engine) — the
    pure builder has its own sentinel (test_mineru_invocation)."""
    path = _two_page_pdf(tmp_path)
    fake_launcher['content_list'] = [
        {'type': 'text', 'text': 'x', 'page_idx': 0}]
    run_local_pdf(path, 2)
    cmd = _mineru_calls(fake_launcher)[0]['cmd']
    assert cmd[:2] == ['docker', 'run']
    adjacent = set(zip(cmd, cmd[1:]))
    for pair in (('--gpus', 'all'), ('--shm-size', '16g'),
                 ('-e', 'HF_HOME=/models'),
                 ('-e', 'MINERU_MODEL_SOURCE=huggingface'),
                 ('-p', '/in/doc.pdf'), ('-o', '/out'),
                 ('-b', 'vlm-engine')):
        assert pair in adjacent
    assert 'mineru:latest' in cmd
    # SEC-SOCKET: the run's sources are the job's volumes (input read-only),
    # never a path below the exchange the worker could turn into a symlink.
    job = cmd[cmd.index('--name') + 1]
    assert _vol_source(cmd, ':/in:ro') == f'{job}_in'
    assert _vol_source(cmd, ':/out') == f'{job}_out'
    copy_in = fake_launcher['calls'][0]['cmd']
    assert f"{fake_launcher['exchange']}:/x:ro" in copy_in
    assert f'if=/x/{job}/in/doc.pdf' in copy_in
    # Whole-document start copies the original byte-identically (measured
    # invocation ran on the full file, never a fitz re-save).
    assert fake_launcher['input_pdfs'][0] == Path(path).read_bytes()


def test_models_dir_env_adds_cache_mount(fake_launcher, tmp_path, monkeypatch):
    monkeypatch.setenv('MINERU_MODELS_DIR', '/srv/hf-cache')
    path = _two_page_pdf(tmp_path)
    fake_launcher['content_list'] = [{'type': 'text', 'text': 'x', 'page_idx': 0}]
    run_local_pdf(path, 2)
    assert '/srv/hf-cache:/models' in _mineru_calls(fake_launcher)[0]['cmd']


def test_container_failure_falls_back_to_text_layer(fake_launcher, tmp_path):
    """Sprint 1.3: no further engine below lokal — pages come from the REAL
    PyMuPDF text layer with ONE named backend_fallback entry, no per-page
    container retry (a failed run is memoized as failed)."""
    path = _two_page_pdf(tmp_path, ('Textebene eins.', 'Textebene zwei.'))
    fake_launcher['rc'] = 1
    fake_launcher['stderr'] = 'CUDA out of memory'
    payload = run_local_pdf(path, 2)
    assert payload['provenance'] == ['deterministisch', 'deterministisch']
    assert 'Textebene eins.' in payload['markdown']
    assert 'Textebene zwei.' in payload['markdown']
    codes = [d['code'] for d in payload['degradations']]
    assert codes == ['backend_fallback']
    entry = payload['degradations'][0]
    assert entry['pages'] == [1, 2]
    assert 'CUDA out of memory' in entry['message']  # raw tool output cited
    assert len(_mineru_calls(fake_launcher)) == 1  # exactly one attempt


def test_timeout_falls_back_with_named_deadline(fake_launcher, tmp_path):
    """SEC-SOCKET: the launcher owns the deadline — it removes the container
    (``docker rm -f <job>``) before it answers ``timed_out``; the worker
    keeps naming the deadline exactly as before."""
    path = _two_page_pdf(tmp_path)
    fake_launcher['raise_timeout'] = True
    payload = run_local_pdf(path, 2)
    assert payload['provenance'] == ['deterministisch', 'deterministisch']
    message = payload['degradations'][0]['message']
    assert message == ('Lokale Engine fehlgeschlagen. Textebene übernommen. '
                       f'(Zeitlimit {mineru_run_timeout_for(2)} s überschritten.)')
    run = _mineru_calls(fake_launcher)[0]
    job = run['cmd'][run['cmd'].index('--name') + 1]
    kills = [c['cmd'] for c in fake_launcher['calls'] if c['kind'] == 'kill']
    assert kills == [['docker', 'rm', '-f', job]]
    # Killed after the run, no copy-out of a dead run, volumes removed.
    assert fake_launcher['kinds']() == ['copy_in', 'run', 'kill', 'volume_rm']


def test_missing_content_list_falls_back(fake_launcher, tmp_path):
    path = _two_page_pdf(tmp_path)
    fake_launcher['write_output'] = False
    payload = run_local_pdf(path, 2)
    assert payload['provenance'] == ['deterministisch', 'deterministisch']
    assert [d['code'] for d in payload['degradations']] == ['backend_fallback']


def test_midflight_start_cuts_subpdf_from_that_page(fake_launcher, tmp_path):
    """Locked decision 3: a switch at page N runs mineru over N..end — the
    input the container sees is the 1-page cut, page_idx maps back."""
    path = _two_page_pdf(tmp_path, ('Cloud hatte Seite eins.', 'Rest ab zwei.'))
    fake_launcher['content_list'] = [
        {'type': 'text', 'text': 'mineru sieht nur Seite zwei.', 'page_idx': 0}]
    engine = LocalPdfEngine(path, 2)
    try:
        result = engine.page(1)
        assert result == {'markdown': 'mineru sieht nur Seite zwei.',
                          'origin': 'modell', 'cost_eur': 0.0}
        cut = fitz.open(stream=fake_launcher['input_pdfs'][0], filetype='pdf')
        assert cut.page_count == 1
        assert 'Rest ab zwei.' in cut[0].get_text('text')
        cut.close()
        # A request BELOW the memoized start would need a second 61 s run —
        # caller bug, loud:
        with pytest.raises(ValueError, match='memoisierten'):
            engine.page(0)
    finally:
        engine.close()


def test_run_timeout_scales_with_cut_range(fake_launcher, tmp_path):
    path = _two_page_pdf(tmp_path)
    fake_launcher['content_list'] = [{'type': 'text', 'text': 'x', 'page_idx': 0}]
    engine = LocalPdfEngine(path, 2)
    try:
        engine.page(1)  # range = 1 page
    finally:
        engine.close()
    assert (_mineru_calls(fake_launcher)[0]['timeout']
            == mineru_run_timeout_for(1))
    assert mineru_run_timeout_for(280) == 300 + 10 * 280  # carries 12_grosses


def test_host_view_env_travels_into_volume_args(fake_launcher, tmp_path,
                                                monkeypatch):
    """The P2 Falle: -v sources are DAEMON paths. Since SEC-SOCKET the host
    view is the LAUNCHER's env, and it appears as exactly one mount: the
    exchange ROOT of the copy helpers. The worker writes into its own view
    and sends only the job name."""
    monkeypatch.setenv('DOC_LOCAL_EXCHANGE_HOST_DIR', '/host/anders')
    path = _two_page_pdf(tmp_path)
    payload = run_local_pdf(path, 2)  # host view existiert hier nicht
    copy_in = fake_launcher['calls'][0]['cmd']
    assert '/host/anders:/x:ro' in copy_in
    # The input could not be copied → 422, no mineru run, named fallback.
    assert fake_launcher['kinds']() == ['copy_in', 'volume_rm']
    message = payload['degradations'][0]['message']
    assert 'mineru-Launcher antwortete 422: Eingabe nicht übernehmbar' in message
    assert payload['provenance'] == ['deterministisch', 'deterministisch']


def test_exchange_job_dir_is_cleaned_up(fake_launcher, tmp_path):
    path = _two_page_pdf(tmp_path)
    fake_launcher['content_list'] = [{'type': 'text', 'text': 'x', 'page_idx': 0}]
    run_local_pdf(path, 2)
    fake_launcher['rc'] = 1
    run_local_pdf(path, 2)
    assert list(fake_launcher['exchange'].iterdir()) == []  # success AND failure


# --- SEC-SOCKET: every launcher failure takes the text-layer fallback,
# with its own reason — no new behaviour, only a different source ----------

def _closed_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _fallback_message(payload):
    codes = [d['code'] for d in payload['degradations']]
    assert codes == ['backend_fallback']
    assert payload['provenance'] == ['deterministisch', 'deterministisch']
    return payload['degradations'][0]['message']


def test_launcher_unreachable_falls_back_with_reason(tmp_path, monkeypatch):
    monkeypatch.setenv('DOC_LOCAL_EXCHANGE_DIR', str(tmp_path))
    monkeypatch.setenv('MINERU_LAUNCHER_URL', f'http://127.0.0.1:{_closed_port()}')
    payload = run_local_pdf(_two_page_pdf(tmp_path, ('Eins.', 'Zwei.')), 2)
    message = _fallback_message(payload)
    assert 'mineru-Launcher nicht erreichbar' in message
    assert 'Eins.' in payload['markdown']  # the real text layer served
    assert [p.name for p in tmp_path.iterdir()] == ['doc.pdf']  # job dir gone


def test_launcher_busy_409_falls_back(fake_launcher, tmp_path):
    """One run at a time: while the launcher's lock is held, a second
    request gets 409 and starts nothing — the worker falls back."""
    assert mineru_launcher._RUN_LOCK.acquire(blocking=False)
    try:
        payload = run_local_pdf(_two_page_pdf(tmp_path), 2)
    finally:
        mineru_launcher._RUN_LOCK.release()
    message = _fallback_message(payload)
    assert 'mineru-Launcher antwortete 409: Es läuft bereits ein mineru-Lauf.' in message
    assert fake_launcher['calls'] == []  # nothing started, not even a chown


def test_launcher_not_configured_is_named(fake_launcher, tmp_path, monkeypatch):
    monkeypatch.delenv('DOC_LOCAL_EXCHANGE_HOST_DIR')
    payload = run_local_pdf(_two_page_pdf(tmp_path), 2)
    message = _fallback_message(payload)
    assert 'mineru-Launcher antwortete 503' in message
    assert 'DOC_LOCAL_EXCHANGE_HOST_DIR' in message
    assert fake_launcher['calls'] == []


def test_page_range_above_the_launcher_limit_is_refused_and_named(fake_launcher,
                                                                 tmp_path):
    """The launcher accepts at most MINERU_MAX_PAGES per run (the invariant
    chain deadline < worker timeout < RQ envelope holds inside it) — a
    larger range is a 400 the worker names, never a started container."""
    pdf = fitz.open()
    for _ in range(MINERU_MAX_PAGES + 1):
        pdf.new_page()
    path = tmp_path / 'lang.pdf'
    pdf.save(str(path))
    pdf.close()
    payload = run_local_pdf(str(path), MINERU_MAX_PAGES + 1)
    entry = payload['degradations'][0]
    assert entry['code'] == 'backend_fallback'
    assert (f'mineru-Launcher antwortete 400: page_count muss eine ganze Zahl '
            f'von 1 bis {MINERU_MAX_PAGES} sein.') in entry['message']
    assert fake_launcher['calls'] == []


def test_worker_sends_data_never_arguments(fake_launcher, tmp_path, monkeypatch):
    """What travels to the launcher is exactly job, pdf_name, page_count."""
    seen = []
    real_validate = mineru_launcher.validate_run_request

    def spy(payload):
        seen.append(payload)
        return real_validate(payload)

    monkeypatch.setattr(mineru_launcher, 'validate_run_request', spy)
    fake_launcher['content_list'] = [{'type': 'text', 'text': 'x', 'page_idx': 0}]
    run_local_pdf(_two_page_pdf(tmp_path), 2)
    assert len(seen) == 1
    assert sorted(seen[0]) == ['job', 'page_count', 'pdf_name']
    assert seen[0]['pdf_name'] == 'doc.pdf'
    assert seen[0]['page_count'] == 2
    assert seen[0]['job'].startswith('mineru_')


# --- DOC-WEB 2.3: the surviving page classifier — scan pages are NAMED on
# the text-layer fallback instead of silently served empty -------------------

def _scan_plus_text_pdf(tmp_path):
    """Page 1: a full-page image, no text (a scan). Page 2: text layer."""
    pdf = fitz.open()
    scan = pdf.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 16, 16), False)
    pix.clear_with(200)
    scan.insert_image(scan.rect, stream=pix.tobytes('png'))
    text_page = pdf.new_page()
    text_page.insert_text((72, 100), 'Seite zwei Text.')
    path = tmp_path / 'scan.pdf'
    pdf.save(str(path))
    pdf.close()
    return str(path)


def test_is_scanned_page_distinguishes_scan_from_text(tmp_path):
    doc = fitz.open(_scan_plus_text_pdf(tmp_path))
    try:
        assert is_scanned_page(doc[0]) is True
        assert is_scanned_page(doc[1]) is False
    finally:
        doc.close()


def test_fallback_names_scan_pages_with_empty_text_layer(fake_launcher, tmp_path):
    """Engine fails → text-layer fallback; the scan page yields '' and the
    payload SAYS so (one ``scan_text_layer_empty`` entry, pages 1-based),
    the text page is not listed."""
    fake_launcher['rc'] = 1
    fake_launcher['stderr'] = 'GPU busy'
    payload = run_local_pdf(_scan_plus_text_pdf(tmp_path), 2)
    codes = [d['code'] for d in payload['degradations']]
    assert codes == ['backend_fallback', 'scan_text_layer_empty']
    scan_entry = payload['degradations'][1]
    assert scan_entry['pages'] == [1]
    assert scan_entry['message'].startswith('Seite 1 ist ein Scan')
    assert payload['markdown'].strip() == 'Seite zwei Text.'


def test_successful_run_never_emits_scan_entry(fake_launcher, tmp_path):
    fake_launcher['content_list'] = [
        {'type': 'text', 'text': 'OCR der Scan-Seite', 'page_idx': 0},
        {'type': 'text', 'text': 'Seite zwei', 'page_idx': 1}]
    payload = run_local_pdf(_scan_plus_text_pdf(tmp_path), 2)
    assert payload['degradations'] == []
    assert payload['provenance'] == ['modell', 'modell']
