"""DOC-LOCAL P1 — the mineru sibling-container backend behind page_fn.

Mocked at the subprocess boundary (the docker CLI is the module's only
process seam). The PDFs are REAL two-page PyMuPDF documents with text
layers, so the sub-PDF cutting and the text-layer fallback run for real —
what the failure path serves is actual page text, not a placeholder.

The invocation-vector test is a SENTINEL (locked decision 2 /
``reference_measured_winner_version_gap``): the measurement only holds for
the verbatim bake-off call — a deviating vector devalues it and must fail
loudly here, like the pandoc vector sentinel in DOC-ENGINE.
"""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import fitz

from services import pdf_local
from services.pdf_local import (
    LocalPdfEngine,
    content_list_to_pages,
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


@pytest.fixture
def fake_docker(monkeypatch, tmp_path):
    """Replace subprocess.run: records calls, writes a content_list into the
    run's /out host dir like a real mineru run would (nested output tree)."""
    state = {'calls': [], 'content_list': [], 'rc': 0,
             'raise_timeout': False, 'write_output': True, 'stderr': '',
             'input_pdfs': []}

    def fake_run(cmd, capture_output=True, text=None, timeout=None, **kwargs):
        state['calls'].append({'cmd': list(cmd), 'timeout': timeout})
        if 'busybox' in cmd:
            return SimpleNamespace(returncode=0, stdout='', stderr='')
        # Snapshot the input NOW — the engine removes its job dir afterwards.
        in_host = _vol_source(cmd, ':/in:ro')
        if in_host and Path(in_host, 'doc.pdf').exists():
            state['input_pdfs'].append(Path(in_host, 'doc.pdf').read_bytes())
        if state['raise_timeout']:
            raise subprocess.TimeoutExpired(cmd, timeout)
        out_host = _vol_source(cmd, ':/out')
        if state['rc'] == 0 and state['write_output'] and out_host:
            nested = Path(out_host) / 'doc' / 'vlm'
            nested.mkdir(parents=True, exist_ok=True)
            (nested / 'doc_content_list.json').write_text(
                json.dumps(state['content_list']), encoding='utf-8')
        return SimpleNamespace(returncode=state['rc'], stdout='',
                               stderr=state['stderr'])

    monkeypatch.setattr(pdf_local.subprocess, 'run', fake_run)
    # Isolated exchange dir per test, both views identical (bare metal).
    exchange = tmp_path / 'exchange'
    exchange.mkdir()
    monkeypatch.setenv('DOC_LOCAL_EXCHANGE_DIR', str(exchange))
    monkeypatch.delenv('DOC_LOCAL_EXCHANGE_HOST_DIR', raising=False)
    monkeypatch.delenv('MINERU_MODELS_DIR', raising=False)
    state['exchange'] = exchange
    return state


def _mineru_calls(state):
    return [c for c in state['calls'] if 'busybox' not in c['cmd']]


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

def test_full_local_run_serves_pages_from_one_container_run(fake_docker, tmp_path):
    path = _two_page_pdf(tmp_path)
    fake_docker['content_list'] = [
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
    assert len(_mineru_calls(fake_docker)) == 1  # one run, both pages served


def test_invocation_vector_is_the_measured_one(fake_docker, tmp_path):
    """SENTINEL: verbatim bake-off invocation (mineru 3.4.4, vlm-engine)."""
    path = _two_page_pdf(tmp_path)
    fake_docker['content_list'] = [
        {'type': 'text', 'text': 'x', 'page_idx': 0}]
    run_local_pdf(path, 2)
    cmd = _mineru_calls(fake_docker)[0]['cmd']
    assert cmd[:2] == ['docker', 'run']
    adjacent = set(zip(cmd, cmd[1:]))
    for pair in (('--gpus', 'all'), ('--shm-size', '16g'),
                 ('-e', 'HF_HOME=/models'),
                 ('-e', 'MINERU_MODEL_SOURCE=huggingface'),
                 ('-p', '/in/doc.pdf'), ('-o', '/out'),
                 ('-b', 'vlm-engine')):
        assert pair in adjacent
    assert 'mineru:latest' in cmd
    assert _vol_source(cmd, ':/in:ro')  # source dir read-only
    # Whole-document start copies the original byte-identically (measured
    # invocation ran on the full file, never a fitz re-save).
    assert fake_docker['input_pdfs'][0] == Path(path).read_bytes()


def test_models_dir_env_adds_cache_mount(fake_docker, tmp_path, monkeypatch):
    monkeypatch.setenv('MINERU_MODELS_DIR', '/srv/hf-cache')
    path = _two_page_pdf(tmp_path)
    fake_docker['content_list'] = [{'type': 'text', 'text': 'x', 'page_idx': 0}]
    run_local_pdf(path, 2)
    assert '/srv/hf-cache:/models' in _mineru_calls(fake_docker)[0]['cmd']


def test_container_failure_falls_back_to_text_layer(fake_docker, tmp_path):
    """Sprint 1.3: no further engine below lokal — pages come from the REAL
    PyMuPDF text layer with ONE named backend_fallback entry, no per-page
    container retry (a failed run is memoized as failed)."""
    path = _two_page_pdf(tmp_path, ('Textebene eins.', 'Textebene zwei.'))
    fake_docker['rc'] = 1
    fake_docker['stderr'] = 'CUDA out of memory'
    payload = run_local_pdf(path, 2)
    assert payload['provenance'] == ['deterministisch', 'deterministisch']
    assert 'Textebene eins.' in payload['markdown']
    assert 'Textebene zwei.' in payload['markdown']
    codes = [d['code'] for d in payload['degradations']]
    assert codes == ['backend_fallback']
    entry = payload['degradations'][0]
    assert entry['pages'] == [1, 2]
    assert 'CUDA out of memory' in entry['message']  # raw tool output cited
    assert len(_mineru_calls(fake_docker)) == 1  # exactly one attempt


def test_timeout_falls_back_with_named_deadline(fake_docker, tmp_path):
    path = _two_page_pdf(tmp_path)
    fake_docker['raise_timeout'] = True
    payload = run_local_pdf(path, 2)
    assert payload['provenance'] == ['deterministisch', 'deterministisch']
    assert 'Zeitlimit' in payload['degradations'][0]['message']


def test_missing_content_list_falls_back(fake_docker, tmp_path):
    path = _two_page_pdf(tmp_path)
    fake_docker['write_output'] = False
    payload = run_local_pdf(path, 2)
    assert payload['provenance'] == ['deterministisch', 'deterministisch']
    assert [d['code'] for d in payload['degradations']] == ['backend_fallback']


def test_midflight_start_cuts_subpdf_from_that_page(fake_docker, tmp_path):
    """Locked decision 3: a switch at page N runs mineru over N..end — the
    input the container sees is the 1-page cut, page_idx maps back."""
    path = _two_page_pdf(tmp_path, ('Cloud hatte Seite eins.', 'Rest ab zwei.'))
    fake_docker['content_list'] = [
        {'type': 'text', 'text': 'mineru sieht nur Seite zwei.', 'page_idx': 0}]
    engine = LocalPdfEngine(path, 2)
    try:
        result = engine.page(1)
        assert result == {'markdown': 'mineru sieht nur Seite zwei.',
                          'origin': 'modell', 'cost_eur': 0.0}
        cut = fitz.open(stream=fake_docker['input_pdfs'][0], filetype='pdf')
        assert cut.page_count == 1
        assert 'Rest ab zwei.' in cut[0].get_text('text')
        cut.close()
        # A request BELOW the memoized start would need a second 61 s run —
        # caller bug, loud:
        with pytest.raises(ValueError, match='memoisierten'):
            engine.page(0)
    finally:
        engine.close()


def test_run_timeout_scales_with_cut_range(fake_docker, tmp_path):
    path = _two_page_pdf(tmp_path)
    fake_docker['content_list'] = [{'type': 'text', 'text': 'x', 'page_idx': 0}]
    engine = LocalPdfEngine(path, 2)
    try:
        engine.page(1)  # range = 1 page
    finally:
        engine.close()
    assert (_mineru_calls(fake_docker)[0]['timeout']
            == mineru_run_timeout_for(1))
    assert mineru_run_timeout_for(280) == 300 + 10 * 280  # carries 12_grosses


def test_host_view_env_travels_into_volume_args(fake_docker, tmp_path,
                                                monkeypatch):
    """The P2 Falle: -v sources are DAEMON paths. When the worker's view and
    the host's view differ, the docker args must carry the HOST view."""
    monkeypatch.setenv('DOC_LOCAL_EXCHANGE_HOST_DIR', '/host/anders')
    fake_docker['write_output'] = False  # host view existiert hier nicht
    path = _two_page_pdf(tmp_path)
    payload = run_local_pdf(path, 2)  # kein lesbarer Output → Fallback
    cmd = _mineru_calls(fake_docker)[0]['cmd']
    assert _vol_source(cmd, ':/in:ro').startswith('/host/anders/')
    assert _vol_source(cmd, ':/out').startswith('/host/anders/')
    assert payload['provenance'] == ['deterministisch', 'deterministisch']


def test_exchange_job_dir_is_cleaned_up(fake_docker, tmp_path):
    path = _two_page_pdf(tmp_path)
    fake_docker['content_list'] = [{'type': 'text', 'text': 'x', 'page_idx': 0}]
    run_local_pdf(path, 2)
    fake_docker['rc'] = 1
    run_local_pdf(path, 2)
    assert list(fake_docker['exchange'].iterdir()) == []  # success AND failure
