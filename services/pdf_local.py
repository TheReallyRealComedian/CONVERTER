"""Local PDF backend (DOC-LOCAL P1): mineru VLM in a sibling container.

Replaces the raw PyMuPDF text layer as the REAL local engine behind the
``page_fn`` contract of ``services/document_pipeline``: pages come from ONE
memoized mineru run, carry provenance ``modell`` (mineru is a VLM — locked
decision 4: ``mode=lokal`` now means *local model, no money*, no longer
*provably deterministic*) and cost 0.00 €.

**The container run happens in the launcher, not here** (SEC-SOCKET): the
worker holds no docker socket. It writes the run's input into a job
directory of the exchange (``<job>/in/doc.pdf``, an empty ``<job>/out``)
and asks ``services.mineru_launcher`` — the one holder of the host socket —
for a run, sending data only (job name, PDF name, page count;
``_request_mineru_run``). The launcher copies the input into the run's own
volume, runs the measured vector, copies the output back into ``<job>/out``
and removes the container at the deadline (before SEC-SOCKET a timed-out
run kept the GPU on the host). The vector and the deadline arithmetic live
in ``services.mineru_invocation``. Any launcher failure — unreachable, busy
(409), refused, a copy step, 5xx — takes the same text-layer fallback as a
failed run.

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

Pure module in the ``pdf_cloud`` mold: no Flask, no SDK singleton, no
subprocess; fitz lives inside functions (worker-side, in-task import
convention). The worker only knows its OWN view of the exchange directory
(``DOC_LOCAL_EXCHANGE_DIR``); the host view of the exchange root is the
launcher's business (``DOC_LOCAL_EXCHANGE_HOST_DIR``, set there only).
"""
import json
import logging
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from services.document_conversions import (
    DEGRADATION_BACKEND_FALLBACK,
    DEGRADATION_SCAN_TEXT_LAYER_EMPTY,
    PROVENANCE_DETERMINISTIC,
    PROVENANCE_MODEL,
    build_result_payload,
    UNIT_PAGE,
    degradation,
)
from services.document_pipeline import PAGE_JOIN
from services.mineru_invocation import (
    LAUNCHER_PORT,
    launcher_reply_timeout_for,
    mineru_run_timeout_for,  # re-exported: the deadline has ONE source
    new_job_name,
)

logger = logging.getLogger(__name__)

# The launcher's compose service name; MINERU_LAUNCHER_URL overrides it.
DEFAULT_LAUNCHER_URL = f'http://mineru-launcher:{LAUNCHER_PORT}'


# Scan detection thresholds — the ONE surviving use of the retired
# ``services/pdf_extraction`` page classifier (DOC-WEB 2.3, locked decision
# 5): not a router anymore (both engines read every page type), only the
# ability to SAY on the text-layer fallback that a page is a scan and "empty"
# is expected there. Values verbatim from the retired ``_classify_page``
# (image coverage > 0.7 of the page area AND text density < 0.5 chars per
# 1000 pt²); the ``mixed`` class fell with the router.
SCAN_IMAGE_COVERAGE_MIN = 0.7
SCAN_TEXT_DENSITY_MAX = 0.5


def is_scanned_page(page):
    """True when a fitz page is (almost) all image and (almost) no text —
    i.e. a scan without a usable text layer. Never raises: an unreadable
    image rect counts as no image (conservative: not-a-scan)."""
    text = page.get_text('text').strip()
    page_area = page.rect.width * page.rect.height
    if page_area <= 0:
        return False
    total_image_area = 0.0
    for img in page.get_images(full=True):
        try:
            for rect in page.get_image_rects(img[0]):
                total_image_area += rect.width * rect.height
        except Exception:
            pass
    image_coverage = total_image_area / page_area
    text_density = len(text) / (page_area / 1000)
    return (image_coverage > SCAN_IMAGE_COVERAGE_MIN
            and text_density < SCAN_TEXT_DENSITY_MAX)


def _exchange_dir():
    """The worker's own view of the exchange directory (a host bind mount in
    compose). The launcher resolves the same job directory in the host view;
    bare metal (tests, dev box) falls back to the temp dir."""
    return os.environ.get('DOC_LOCAL_EXCHANGE_DIR') or tempfile.gettempdir()


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


class MineruTimeout(RuntimeError):
    """The launcher reports the run tore its deadline — and removed the
    container before answering."""

    def __init__(self, deadline_seconds):
        super().__init__(f'Zeitlimit {deadline_seconds} s überschritten.')
        self.deadline_seconds = deadline_seconds


def _launcher_url():
    return (os.environ.get('MINERU_LAUNCHER_URL') or DEFAULT_LAUNCHER_URL).rstrip('/')


def _launcher_error_detail(http_error):
    try:
        detail = json.loads(http_error.read().decode('utf-8')).get('error')
    except Exception:  # an unreadable error body still names the status
        detail = None
    return str(detail or http_error.reason)[:300]


def _request_mineru_run(job_name, pdf_name, page_count):
    """Ask the launcher for ONE run over ``<exchange>/<job_name>/{in,out}``.

    Data, never arguments — the launcher builds the command line from its
    own environment. The HTTP timeout is the deadline plus the launcher's
    reply margin (copy steps, kill, volume cleanup), so the launcher always
    answers first; the socket timeout covers the whole wait, because the
    launcher sends nothing until the run is over. Every failure raises —
    the caller turns it into the text-layer fallback with the reason named.
    """
    body = json.dumps({'job': job_name, 'pdf_name': pdf_name,
                       'page_count': page_count}).encode('utf-8')
    request = urllib.request.Request(
        f'{_launcher_url()}/run', data=body, method='POST',
        headers={'Content-Type': 'application/json'})
    timeout = launcher_reply_timeout_for(page_count)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as e:  # before URLError: it is one
        raise RuntimeError(
            f'mineru-Launcher antwortete {e.code}: {_launcher_error_detail(e)}') from None
    except TimeoutError:
        raise RuntimeError(
            f'mineru-Launcher antwortete nicht innerhalb von {timeout} s.') from None
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(
            f'mineru-Launcher nicht erreichbar: {getattr(e, "reason", e)}') from None
    try:
        result = json.loads(raw.decode('utf-8'))
    except ValueError:
        raise RuntimeError('Antwort des mineru-Launchers ist kein JSON.') from None
    if result.get('timed_out'):
        raise MineruTimeout(result.get('deadline_seconds')
                            or mineru_run_timeout_for(page_count))
    returncode = result.get('returncode')
    if returncode != 0:
        tail = (result.get('stderr_tail') or result.get('stdout_tail') or '')[-800:]
        raise RuntimeError(f'mineru-Container rc={returncode}. mineru meldete: {tail}')
    return result


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
    ``backend_fallback`` entry the caller attaches to its payload — plus,
    if fallback pages are scans with an empty text layer, ONE
    ``scan_text_layer_empty`` entry naming those pages (DOC-WEB 2.3: the
    answer says "empty is expected here" instead of silently serving
    nothing).

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
        self._scan_entry = None  # the one scan_text_layer_empty entry, lazy

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
        job_name = new_job_name()
        job_dir = os.path.join(_exchange_dir(), job_name)
        n_pages = self.page_count - start
        try:
            in_dir = os.path.join(job_dir, 'in')
            out_dir = os.path.join(job_dir, 'out')
            os.makedirs(in_dir)
            os.makedirs(out_dir)
            self._prepare_input(start, in_dir)
            _request_mineru_run(job_name, 'doc.pdf', n_pages)
            entries = _load_content_list(out_dir)
            self._pages = content_list_to_pages(entries, start, self.page_count)
            logger.info(
                'mineru-Lauf ok: Seiten %d–%d, %d Elemente',
                start + 1, self.page_count, len(entries))
        except Exception as e:
            # MineruTimeout's text is the deadline sentence; everything else
            # (launcher unreachable, 409, 5xx, rc != 0, no output) cites
            # itself.
            self._failed = True
            reason = str(e)
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
        page = self._fitz_doc()[index]
        text = page.get_text('text').strip()
        if not text and is_scanned_page(page):
            self._note_scan_page(index)
        return {'markdown': text,
                'origin': PROVENANCE_DETERMINISTIC,
                'cost_eur': 0.0}

    def _note_scan_page(self, index):
        """Accumulate scan pages into ONE degradation entry (pages 1-based).
        The entry is appended on the first hit and MUTATED afterwards — the
        caller reads ``self.degradations`` after the page loop, so the list
        is complete by then."""
        if self._scan_entry is None:
            self._scan_entry = degradation(
                DEGRADATION_SCAN_TEXT_LAYER_EMPTY, '', pages=[])
            self.degradations.append(self._scan_entry)
        self._scan_entry['pages'].append(index + 1)
        pages = self._scan_entry['pages']
        label = (f'Seite {pages[0]} ist ein Scan' if len(pages) == 1
                 else f'Seiten {", ".join(str(p) for p in pages)} sind Scans')
        self._scan_entry['message'] = (
            f'{label}, die Textebene ist dort leer. Ohne lokale Engine '
            f'bleibt der Inhalt leer.')


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
