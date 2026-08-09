"""DOC-ENGINE P1 — the office/web backends behind the DOC-API result shape.

Two layers, mocked at the same boundary the production code crosses:

* Adapter layer (``services/office_backends``): pandoc is exercised through a
  mocked ``subprocess.run`` (the SENTINEL pins the exact measured argument
  vector — the bake-off's numbers only hold for that call), markitdown and
  trafilatura through ``sys.modules`` stubs (neither is installed on the dev
  box, mirroring the unstructured stub in conftest).
* Task routing (``tasks.convert_document_task``): each extension reaches its
  backend, the payload is the shared document-level deterministic shape, and
  an empty trafilatura extraction degrades to the unstructured path with a
  named ``backend_fallback`` entry instead of failing.

The real binaries/libraries run only in the container live-smoke (P1.3), not
here — same suite limit as the torch import chain (CLAUDE.md).
"""
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest

from services import document_conversions as doc_lib
from services import office_backends as ob
from tasks import convert_document_task


# --- pandoc adapter (subprocess mocked) --------------------------------------

@pytest.fixture
def fake_pandoc(monkeypatch):
    """Mock shutil.which + subprocess.run inside office_backends; records
    the argument vector and serves a configurable CompletedProcess."""
    calls = []
    state = {'returncode': 0, 'stdout': '# Doc\n\nText [^1]\n\n[^1]: Fussnote',
             'stderr': ''}

    def fake_run(args, capture_output=None, text=None, timeout=None):
        calls.append({'args': args, 'timeout': timeout})
        return subprocess.CompletedProcess(
            args, state['returncode'], stdout=state['stdout'],
            stderr=state['stderr'])

    monkeypatch.setattr(ob.shutil, 'which', lambda name: '/usr/bin/pandoc')
    monkeypatch.setattr(ob.subprocess, 'run', fake_run)
    return {'calls': calls, 'state': state}


def test_pandoc_sentinel_pins_the_measured_invocation(fake_pandoc):
    """The bake-off measured EXACTLY ``-f docx -t gfm --wrap=none`` — any
    drift in the vector voids the measurement this sprint builds on."""
    markdown, warnings = ob.convert_docx_pandoc('/tmp/x.docx')
    call = fake_pandoc['calls'][0]
    assert call['args'] == [
        '/usr/bin/pandoc', '-f', 'docx', '-t', 'gfm', '--wrap=none',
        '/tmp/x.docx']
    assert call['timeout'] == ob.PANDOC_TIMEOUT_SECONDS  # per-call deadline
    assert markdown.startswith('# Doc')
    assert warnings == []


def test_pandoc_stderr_becomes_warning(fake_pandoc):
    fake_pandoc['state']['stderr'] = '[WARNING] Could not convert image x\n'
    markdown, warnings = ob.convert_docx_pandoc('/tmp/x.docx')
    assert warnings == ['[WARNING] Could not convert image x']


def test_pandoc_nonzero_rc_raises(fake_pandoc):
    fake_pandoc['state']['returncode'] = 2
    fake_pandoc['state']['stderr'] = 'kaputt'
    with pytest.raises(RuntimeError, match='rc=2'):
        ob.convert_docx_pandoc('/tmp/x.docx')


def test_pandoc_empty_output_raises(fake_pandoc):
    fake_pandoc['state']['stdout'] = '   \n'
    with pytest.raises(RuntimeError, match='leeres Markdown'):
        ob.convert_docx_pandoc('/tmp/x.docx')


def test_pandoc_missing_binary_raises(monkeypatch):
    monkeypatch.setattr(ob.shutil, 'which', lambda name: None)
    with pytest.raises(RuntimeError, match='nicht im PATH'):
        ob.convert_docx_pandoc('/tmp/x.docx')


# --- markitdown adapter (sys.modules stub) -----------------------------------

@pytest.fixture
def fake_markitdown(monkeypatch):
    state = {'text_content': '# Folie 1\n\n### Notes:\nSprechernotiz'}
    calls = []

    class FakeMarkItDown:
        def convert(self, path):
            calls.append(path)
            return SimpleNamespace(text_content=state['text_content'])

    module = types.ModuleType('markitdown')
    module.MarkItDown = FakeMarkItDown
    monkeypatch.setitem(sys.modules, 'markitdown', module)
    return {'state': state, 'calls': calls}


def test_markitdown_returns_text_content(fake_markitdown):
    markdown, warnings = ob.convert_pptx_markitdown('/tmp/deck.pptx')
    assert fake_markitdown['calls'] == ['/tmp/deck.pptx']
    assert 'Sprechernotiz' in markdown
    assert warnings == []


def test_markitdown_empty_raises(fake_markitdown):
    fake_markitdown['state']['text_content'] = ''
    with pytest.raises(RuntimeError, match='leeres Markdown'):
        ob.convert_pptx_markitdown('/tmp/deck.pptx')


# --- trafilatura adapter (sys.modules stub) ----------------------------------

@pytest.fixture
def fake_trafilatura(monkeypatch):
    state = {'body': 'Der Artikeltext.', 'meta': SimpleNamespace(
        title='Meta-Titel', author='C. Stöcker', date='2006-12-19')}

    module = types.ModuleType('trafilatura')
    module.extract = lambda html, **kwargs: state['body']
    module.extract_metadata = lambda html: state['meta']
    monkeypatch.setitem(sys.modules, 'trafilatura', module)
    return state


def _write_html(tmp_path, html, encoding='utf-8'):
    p = tmp_path / 'seite.html'
    p.write_bytes(html.encode(encoding))
    return str(p)


def test_html_title_tag_wins_and_byline_forms(fake_trafilatura, tmp_path):
    """The raw <title> tag beats trafilatura's title field (measured on the
    corpus exemplar: the tag carries kicker+headline, the field the bare
    sitename); author/date form one italic line."""
    path = _write_html(
        tmp_path,
        '<html><head><title>Dachzeile: Der  Titel &amp; mehr</title></head>'
        '<body><p>x</p></body></html>')
    markdown, warnings = ob.convert_html_trafilatura(path)
    lines = markdown.split('\n\n')
    assert lines[0] == '# Dachzeile: Der Titel & mehr'  # entities + collapse
    assert lines[1] == '*C. Stöcker · 2006-12-19*'
    assert lines[2] == 'Der Artikeltext.'
    assert warnings == []


def test_html_meta_title_is_only_the_fallback(fake_trafilatura, tmp_path):
    path = _write_html(tmp_path, '<html><body><p>ohne head</p></body></html>')
    markdown, _ = ob.convert_html_trafilatura(path)
    assert markdown.split('\n\n')[0] == '# Meta-Titel'


def test_html_without_any_metadata_is_bare_body(fake_trafilatura, tmp_path):
    fake_trafilatura['meta'] = SimpleNamespace(title=None, author=None, date=None)
    path = _write_html(tmp_path, '<html><body><p>nackt</p></body></html>')
    markdown, _ = ob.convert_html_trafilatura(path)
    assert markdown == 'Der Artikeltext.'


def test_html_empty_extraction_returns_none(fake_trafilatura, tmp_path):
    fake_trafilatura['body'] = None
    path = _write_html(tmp_path, '<html><body></body></html>')
    markdown, warnings = ob.convert_html_trafilatura(path)
    assert markdown is None
    assert warnings == []


def test_html_cp1252_decodes(fake_trafilatura, tmp_path):
    """The measured decode chain: utf-8 → cp1252 (the corpus exemplar's
    encoding) → latin-1."""
    seen = {}
    mod = sys.modules['trafilatura']
    original = mod.extract
    mod.extract = lambda html, **kw: seen.setdefault('html', html) and original(html)
    path = _write_html(tmp_path,
                       '<html><head><title>Gänsefüßchen „richtig“</title>'
                       '</head><body>x</body></html>', encoding='cp1252')
    markdown, _ = ob.convert_html_trafilatura(path)
    assert 'Gänsefüßchen „richtig“' in seen['html']
    assert markdown.startswith('# Gänsefüßchen „richtig“')


def test_html_title_helper_first_tag_only():
    html = ('<html><head><title>Kopf</title></head><body>'
            '<svg><title>SVG-Beschriftung</title></svg></body></html>')
    assert ob._html_title(html) == 'Kopf'


# --- task routing: extension → backend → shared payload shape -----------------

@pytest.fixture
def doc_convert_dir(tmp_path, monkeypatch):
    d = tmp_path / 'doc_conversions'
    monkeypatch.setattr(doc_lib, 'DOC_CONVERT_DIR', str(d))
    return d


def _plant_source(cid, ext, data=b'x'):
    doc_lib.ensure_doc_convert_dir()
    with open(doc_lib.doc_source_path(cid, ext), 'wb') as f:
        f.write(data)


def test_task_routes_docx_to_pandoc(monkeypatch, doc_convert_dir):
    monkeypatch.setattr('services.office_backends.convert_docx_pandoc',
                        lambda path: ('# Via pandoc', ['eine Warnung']))
    _plant_source(801, 'docx')
    convert_document_task(801, 'docx', 'cloud', 1.0, None)
    payload = doc_lib.read_result_file(801)
    assert payload['markdown'] == '# Via pandoc'
    assert payload['provenance_unit'] == 'document'
    assert payload['provenance'] == ['deterministisch']
    assert payload['degradations'] == [
        {'code': 'serializer', 'message': 'eine Warnung', 'pages': None}]
    assert payload['usage'] == {'model_calls': 0, 'cost_eur': 0.0}


def test_task_routes_pptx_to_markitdown(monkeypatch, doc_convert_dir):
    monkeypatch.setattr('services.office_backends.convert_pptx_markitdown',
                        lambda path: ('# Via markitdown', []))
    _plant_source(802, 'pptx')
    convert_document_task(802, 'pptx', 'cloud', 1.0, None)
    payload = doc_lib.read_result_file(802)
    assert payload['markdown'] == '# Via markitdown'
    assert payload['provenance'] == ['deterministisch']
    assert payload['degradations'] == []


@pytest.mark.parametrize('ext', ['html', 'htm'])
def test_task_routes_html_family_to_trafilatura(monkeypatch, doc_convert_dir,
                                                ext):
    monkeypatch.setattr('services.office_backends.convert_html_trafilatura',
                        lambda path: ('# Via trafilatura', []))
    cid = 803 if ext == 'html' else 804
    _plant_source(cid, ext)
    convert_document_task(cid, ext, 'cloud', 1.0, None)
    payload = doc_lib.read_result_file(cid)
    assert payload['markdown'] == '# Via trafilatura'
    assert payload['provenance'] == ['deterministisch']


def test_task_html_empty_extraction_falls_back_named(monkeypatch,
                                                     doc_convert_dir):
    """trafilatura finds nothing → the unstructured path serves the result
    and the switch is a NAMED degradation on a ready result, not a failure."""
    monkeypatch.setattr('services.office_backends.convert_html_trafilatura',
                        lambda path: (None, []))
    monkeypatch.setattr(
        sys.modules['unstructured.partition.auto'], 'partition',
        lambda filename=None, strategy=None: [SimpleNamespace(
            category='NarrativeText', text='Roher Seitentext.',
            metadata=SimpleNamespace(category_depth=None, page_number=None,
                                     text_as_html=None))])
    _plant_source(805, 'html')
    convert_document_task(805, 'html', 'cloud', 1.0, None)
    payload = doc_lib.read_result_file(805)
    assert payload['markdown'] == 'Roher Seitentext.'
    assert payload['provenance'] == ['deterministisch']
    assert [d['code'] for d in payload['degradations']] == ['backend_fallback']
    assert 'Element-Extraktion' in payload['degradations'][0]['message']


def test_task_eml_stays_on_unstructured(monkeypatch, doc_convert_dir):
    """EML deliberately keeps the legacy path (decision doc: functional,
    without competition) — partition IS the backend here."""
    calls = []

    def fake_partition(filename=None, strategy=None):
        calls.append(strategy)
        return [SimpleNamespace(
            category='NarrativeText', text='Mailtext.',
            metadata=SimpleNamespace(category_depth=None, page_number=None,
                                     text_as_html=None))]

    monkeypatch.setattr(sys.modules['unstructured.partition.auto'],
                        'partition', fake_partition)
    _plant_source(806, 'eml')
    convert_document_task(806, 'eml', 'cloud', 1.0, None)
    payload = doc_lib.read_result_file(806)
    assert calls == ['fast']
    assert payload['markdown'] == 'Mailtext.'
    assert payload['provenance'] == ['deterministisch']


def test_xlsx_stays_a_400_with_a_clear_message(app, client, test_user,
                                               monkeypatch):
    """Sprint 1.2: XLSX is deliberately unbuilt — the submit answers 400 with
    the accepted-list message, no row, no job."""
    import io
    monkeypatch.setenv('DOC_CONVERT_TOKEN', 'tok-x')
    resp = client.post(
        '/api/document-conversions',
        data={'file': (io.BytesIO(b'PK\x03\x04'), 'tabelle.xlsx')},
        headers={'Authorization': 'Bearer tok-x'},
        content_type='multipart/form-data')
    assert resp.status_code == 400
    assert 'wird nicht unterstützt' in resp.get_json()['error']
