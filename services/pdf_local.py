"""Local PDF backend (DOC-LOCAL P1): mineru VLM in a sibling container.

Replaces the raw PyMuPDF text layer as the REAL local engine behind the
``page_fn`` contract of ``services/document_pipeline``: pages come from ONE
memoized mineru run, carry provenance ``modell`` (mineru is a VLM — locked
decision 4: ``mode=lokal`` now means *local model, no money*, no longer
*provably deterministic*) and cost 0.00 €.

**The invocation is replicated verbatim** from the measured bake-off adapter
(``corpus/bakeoff/harness/adapters.py::run_mineru_vlm``, mineru 3.4.4,
gold-f1 0.9551 on 01.gold) — locked decision 2, lesson
``reference_measured_winner_version_gap``:

    docker run --rm --gpus all --shm-size 16g
      -v <in>:/in:ro -v <out>:/out [-v <models>:/models]
      -e HF_HOME=/models -e MINERU_MODEL_SOURCE=huggingface
      mineru:latest mineru -p /in/<datei>.pdf -o /out -b vlm-engine

The backend name is ``vlm-engine`` (the 2.x docs still say
``vlm-vllm-engine`` — verified live against ``--help`` in the bake-off).
The container runs as root (``--user`` dies on missing passwd entries,
live-hit); root-owned ``/out`` files are made removable with a busybox
``chown -R`` afterwards — chown, NOT chmod: ``a+rX`` left the temp-dir
cleanup dying on EPERM (both live-hit in the bake-off).

**Measured 2026-08-16** (this sprint's P1 runs on the Mintbox, three
documents): the model weights are BAKED INTO the image — ``/root/mineru.json``
pins ``models-dir`` to the snapshot inside ``/root/.cache/huggingface``
(``MinerU2.5-Pro-2605-1.2B`` @ ``bff20d4``, 4.6 GB), which is why no host
cache ever held MinerU weights and the 61 s start needs no download. The
``/models`` mount + env stay replicated anyway: they are the measured call,
and they are the safety net the day the image resolves anything remotely.

**One run per needed page range, not per page** (locked decision 3): at
~61 s fixed model start + ~2.5 s per page (fitted 2..280 pages; re-measured
today: 2 pages 64/66 s, 12 pages 95 s), a call per page would be absurd.
``LocalPdfEngine.page(index)`` stays page-wise outward; the FIRST call cuts a
sub-PDF from that page to the end (a mid-flight budget switch at page N never
re-renders pages the cloud already produced), runs mineru once, and serves
every later page from the memo. A start at page 0 copies the original file
instead of re-saving through fitz — byte-identical input to the measured
invocation.

**Per-page Markdown comes from ``<name>_content_list.json``** (measured on
01_gold/03_gold/04 today, not assumed): every element carries ``page_idx``
(0-based within the run's input), ``type`` and its payload. Tables arrive as
``table_body`` — raw ``<table>`` HTML WITH real rowspan/colspan (04's merged
cells intact) — plus caption/footnote string lists, so page attribution
needs no text-cutting of mineru's own ``.md``. Deliberate deviation from
that ``.md``: mineru DROPS ``header``/``footer``/``page_number`` elements
there; the content_list carries them, and this assembly KEEPS them (locked
decision 5 / Bewertungsregel 4 — repeated page furniture is measured
content, "alles rein").

**Failure path** (sprint 1.3): the budget cap degrades cloud→lokal; if the
local engine itself fails (GPU busy, container error, timeout, unusable
output), there is no further engine — pages fall back to the PyMuPDF text
layer (provenance ``deterministisch``, empty on scans) and the switch is
named as ONE ``backend_fallback`` degradation (DOC-ENGINE P1 pattern: a hard
fail would be a capability regression). One failed run is memoized as
failed — no 61 s retry per page.

Pure module in the ``pdf_cloud`` mold: no Flask, no SDK singleton; fitz and
subprocess work live inside functions (worker-side, in-task import
convention). The docker CLI + socket reach the HOST daemon (P2 wiring), so
``-v`` sources must be HOST paths: ``DOC_LOCAL_EXCHANGE_HOST_DIR`` names the
exchange directory as the daemon sees it whenever the worker's own view
(``DOC_LOCAL_EXCHANGE_DIR``) differs — on bare metal both default to the
same temp directory.
"""
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from services.document_conversions import (
    DEGRADATION_BACKEND_FALLBACK,
    PROVENANCE_DETERMINISTIC,
    PROVENANCE_MODEL,
    build_result_payload,
    UNIT_PAGE,
    degradation,
)
from services.document_pipeline import PAGE_JOIN

logger = logging.getLogger(__name__)

# Image env-overridable (DOC-FIX lesson: a hardcoded name cost two months of
# silent failure). The measurement holds for mineru 3.4.4 — the Mintbox image
# id 6cc9e57ff5bd has NO registry digest (locally loaded), so P2 pins by tag.
MINERU_IMAGE = os.environ.get('MINERU_IMAGE') or 'mineru:latest'
MINERU_BACKEND = 'vlm-engine'

# Container run deadline from the measured cost curve (~61 s start + ~2.5 s
# per page), with ~4x margin: 280 pages measured 766 s, capped here at
# 300 + 10×280 = 3100 s. The per-call deadline doctrine
# (reference_worker_sdk_per_call_deadline): the RQ envelope alone would not
# interrupt a wedged docker-CLI child — subprocess.run(timeout=) does.
MINERU_TIMEOUT_BASE_SECONDS = 300
MINERU_TIMEOUT_PER_PAGE_SECONDS = 10


def mineru_run_timeout_for(page_count):
    """Deadline in seconds for ONE mineru container run over ``page_count`` pages."""
    n = page_count if isinstance(page_count, int) and page_count > 0 else 1
    return MINERU_TIMEOUT_BASE_SECONDS + MINERU_TIMEOUT_PER_PAGE_SECONDS * n


def _exchange_dirs():
    """(our_view, host_view) of the directory sibling containers mount from.

    The docker daemon lives on the HOST, so every ``-v`` source must be a
    host path. In the worker container the exchange dir is a bind mount whose
    host-side path differs from the container-side one — P2 sets both envs.
    Bare metal (tests, dev box): one temp dir, both views identical.
    """
    ours = os.environ.get('DOC_LOCAL_EXCHANGE_DIR') or tempfile.gettempdir()
    host = os.environ.get('DOC_LOCAL_EXCHANGE_HOST_DIR') or ours
    return ours, host


def _entry_markdown(entry):
    """One content_list element → its Markdown block(s), or '' to skip.

    Measured field semantics (2026-08-16 runs): ``text`` elements carry
    optional ``text_level`` (1 = heading; absent = paragraph); ``equation``
    text already includes its ``$$`` delimiters; ``table`` carries
    caption/footnote lists + raw HTML ``table_body``; ``image`` carries
    ``img_path`` (dead outside the run — kept as figure marker, Regel 3:
    any image syntax with any target counts), caption/footnote lists and an
    optional model description in ``content`` (mineru's own .md wraps it in
    a details block — replicated). header/footer/page_number render as plain
    paragraphs on purpose (locked decision 5).
    """
    etype = entry.get('type')
    if etype == 'table':
        parts = [c.strip() for c in entry.get('table_caption') or [] if c.strip()]
        body = (entry.get('table_body') or '').strip()
        if body:
            parts.append(body)
        parts += [f.strip() for f in entry.get('table_footnote') or [] if f.strip()]
        return '\n\n'.join(parts)
    if etype == 'image':
        parts = [c.strip() for c in entry.get('image_caption') or [] if c.strip()]
        parts.append(f"![]({entry.get('img_path') or ''})")
        content = (entry.get('content') or '').strip()
        if content:
            summary = entry.get('sub_type') or 'abbildung'
            parts.append(f'<details>\n<summary>{summary}</summary>\n\n'
                         f'{content}\n\n</details>')
        parts += [f.strip() for f in entry.get('image_footnote') or [] if f.strip()]
        return '\n\n'.join(parts)
    text = (entry.get('text') or '').strip()
    if not text:
        return ''
    level = entry.get('text_level')
    if etype == 'text' and isinstance(level, int) and level >= 1:
        return f"{'#' * min(level, 6)} {text}"
    return text


def content_list_to_pages(entries, start_index, page_count):
    """Group content_list elements by page and render each page's Markdown.

    ``page_idx`` is 0-based WITHIN the run's input (the sub-PDF), so absolute
    page = ``start_index + page_idx``. Returns {absolute_index: markdown} with
    an entry for EVERY page in [start_index, page_count) — a page mineru saw
    but emitted nothing for is an empty string, not a missing key.
    """
    blocks = {index: [] for index in range(start_index, page_count)}
    for entry in entries:
        rel = entry.get('page_idx')
        if not isinstance(rel, int):
            continue
        absolute = start_index + rel
        if absolute not in blocks:
            continue
        rendered = _entry_markdown(entry)
        if rendered:
            blocks[absolute].append(rendered)
    return {index: '\n\n'.join(parts) for index, parts in blocks.items()}


def _run_mineru_container(in_dir_host, out_dir_host, pdf_name, timeout_seconds):
    """The measured sibling-container invocation, verbatim (docstring above)."""
    models_host = os.environ.get('MINERU_MODELS_DIR')
    cmd = ['docker', 'run', '--rm', '--gpus', 'all', '--shm-size', '16g',
           '-v', f'{in_dir_host}:/in:ro', '-v', f'{out_dir_host}:/out']
    if models_host:
        cmd += ['-v', f'{models_host}:/models']
    cmd += ['-e', 'HF_HOME=/models', '-e', 'MINERU_MODEL_SOURCE=huggingface',
            MINERU_IMAGE,
            'mineru', '-p', f'/in/{pdf_name}', '-o', '/out', '-b', MINERU_BACKEND]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout_seconds)
    # chown, not chmod: root-owned /out subdirs killed the exchange-dir
    # cleanup with EPERM under a+rX (bake-off, live-hit). Never fatal — a
    # failed chown surfaces later as a cleanup warning, not a failed page.
    subprocess.run(['docker', 'run', '--rm', '-v', f'{out_dir_host}:/out',
                    'busybox', 'chown', '-R',
                    f'{os.getuid()}:{os.getgid()}', '/out'],
                   capture_output=True, timeout=120)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or '')[-800:]
        raise RuntimeError(
            f'mineru-Container rc={proc.returncode}. mineru meldete: {tail}')
    return proc


def _load_content_list(out_dir):
    """Find + parse ``<name>_content_list.json`` in the run's output tree.

    The v2 sibling (``*_content_list_v2.json``) does NOT match this suffix
    glob — the flat v1 list with per-element ``page_idx`` is the measured,
    master-verified format this module builds on.
    """
    matches = sorted(Path(out_dir).rglob('*_content_list.json'))
    if not matches:
        raise RuntimeError('mineru-Output ohne content_list.json.')
    data = json.loads(matches[0].read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise RuntimeError('content_list.json ist keine Liste.')
    return data


class LocalPdfEngine:
    """Serves the ``page_fn`` contract from ONE memoized mineru run.

    ``page(index)`` is the contract callable: the first call triggers the
    single container run over pages [index, page_count) and every subsequent
    call is served from the memo — the paged pipeline iterates monotonically,
    so the first requested page IS the start of the needed range. A request
    below the memoized start would need a second 61 s run and signals a
    caller bug → ValueError, loud.

    On run failure every page from the start falls back to the PyMuPDF text
    layer; ``degradations`` then carries exactly one named
    ``backend_fallback`` entry the caller attaches to its payload.

    Owns a lazy fitz handle (sub-PDF cutting + text-layer fallback) —
    ``close()`` releases it; ``run_local_pdf`` wraps this in try/finally.
    """

    def __init__(self, source_path, page_count):
        self.source_path = source_path
        self.page_count = page_count
        self.degradations = []
        self._doc = None
        self._start = None
        self._pages = None   # {absolute_index: markdown} on success
        self._failed = False

    # -- lifecycle -----------------------------------------------------------

    def _fitz_doc(self):
        if self._doc is None:
            import fitz
            self._doc = fitz.open(self.source_path)
        return self._doc

    def close(self):
        if self._doc is not None:
            self._doc.close()
            self._doc = None

    # -- the memoized run ----------------------------------------------------

    def _prepare_input(self, start, in_dir):
        """Write the run's input PDF: whole-file copy at start 0 (byte-equal
        to the measured invocation), fitz-cut sub-PDF from ``start`` else."""
        dest = os.path.join(in_dir, 'doc.pdf')
        if start == 0:
            shutil.copyfile(self.source_path, dest)
        else:
            import fitz
            sub = fitz.open()
            sub.insert_pdf(self._fitz_doc(), from_page=start,
                           to_page=self.page_count - 1)
            sub.save(dest)
            sub.close()
        return dest

    def _ensure_run(self, start):
        if self._pages is not None or self._failed:
            return
        self._start = start
        exchange_ours, exchange_host = _exchange_dirs()
        job_name = f'mineru_{uuid.uuid4().hex[:12]}'
        job_dir = os.path.join(exchange_ours, job_name)
        n_pages = self.page_count - start
        try:
            in_dir = os.path.join(job_dir, 'in')
            out_dir = os.path.join(job_dir, 'out')
            os.makedirs(in_dir)
            os.makedirs(out_dir)
            self._prepare_input(start, in_dir)
            _run_mineru_container(
                os.path.join(exchange_host, job_name, 'in'),
                os.path.join(exchange_host, job_name, 'out'),
                'doc.pdf',
                mineru_run_timeout_for(n_pages))
            entries = _load_content_list(out_dir)
            self._pages = content_list_to_pages(entries, start, self.page_count)
            logger.info(
                'mineru-Lauf ok: Seiten %d–%d, %d Elemente',
                start + 1, self.page_count, len(entries))
        except Exception as e:
            self._failed = True
            reason = str(e)
            if isinstance(e, subprocess.TimeoutExpired):
                reason = (f'Zeitlimit {mineru_run_timeout_for(n_pages)} s '
                          f'überschritten.')
            logger.error('mineru-Lauf fehlgeschlagen (Seiten %d–%d): %s',
                         start + 1, self.page_count, reason)
            self.degradations.append(degradation(
                DEGRADATION_BACKEND_FALLBACK,
                f'Lokale Engine fehlgeschlagen. Textebene übernommen. '
                f'({reason[:300]})',
                pages=list(range(start + 1, self.page_count + 1)),
            ))
        finally:
            try:
                shutil.rmtree(job_dir, ignore_errors=False)
            except OSError as cleanup_error:
                logger.warning('Exchange-Verzeichnis nicht aufräumbar: %s',
                               cleanup_error)

    # -- the contract --------------------------------------------------------

    def page(self, index):
        """``page_fn(page_index_0based) -> {markdown, origin, cost_eur}``."""
        self._ensure_run(index if self._start is None else self._start)
        if self._pages is not None:
            if index < self._start:
                raise ValueError(
                    f'Seite {index} liegt vor dem memoisierten Lauf '
                    f'(Start {self._start}).')
            return {'markdown': self._pages.get(index, ''),
                    'origin': PROVENANCE_MODEL,
                    'cost_eur': 0.0}
        return {'markdown': self._fitz_doc()[index].get_text('text').strip(),
                'origin': PROVENANCE_DETERMINISTIC,
                'cost_eur': 0.0}


def run_local_pdf(source_path, page_count):
    """Full local conversion: every page through the mineru engine.

    The pure-``lokal`` counterpart of ``run_cloud_pdf`` — no budget mechanic
    (nothing here costs money), so no ``run_paged_conversion``: pages walk the
    engine directly and ``usage.model_calls`` stays 0 on purpose — that
    counter means PAID cloud calls in the contract, and the honest signal for
    "a model wrote this" is the per-page ``modell`` provenance, not a call
    count.
    """
    engine = LocalPdfEngine(source_path, page_count)
    try:
        results = [engine.page(index) for index in range(page_count)]
    finally:
        engine.close()
    return build_result_payload(
        PAGE_JOIN.join(r['markdown'] for r in results),
        provenance_unit=UNIT_PAGE,
        provenance=[r['origin'] for r in results],
        degradations=engine.degradations,
        usage={'model_calls': 0, 'cost_eur': 0.0},
    )
