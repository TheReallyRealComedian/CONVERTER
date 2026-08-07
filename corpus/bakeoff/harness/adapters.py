# corpus/bakeoff/harness/adapters.py
"""Kandidaten-Adapter: nimm diese Datei, gib Markdown und Kennzahlen zurueck.

Ein Adapter ist eine Funktion ``(input_path, ctx) -> AdapterResult``. Schwere
Imports leben IN der Funktion — das Harness selbst laeuft stdlib-only, jeder
Adapter deklariert, in welchem Env er lauffaehig ist. Neue Kandidaten sind
ein Dict-Eintrag in ``ADAPTERS`` plus eine Funktion; mehr nicht.

Der Eigenbau wird als Kandidat AUFGERUFEN, nicht veraendert: der Import
laeuft ueber Fake-Parent-Packages (``services``/``app_pkg`` als leere
Namespace-Module mit ``__path__`` auf die echten Verzeichnisse), damit weder
``services/__init__.py`` (zieht Deepgram & Co.) noch ``app_pkg/__init__.py``
(zieht Flask) ausgefuehrt werden — nur ``services.pdf_extraction.*`` und
``app_pkg.config`` (reine Konstanten). Kosten-Instrumentierung haengt am
``genai``-Client-Objekt der Service-INSTANZ (Wrapper um
``models.generate_content``), nicht am Paket.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class AdapterResult:
    markdown: str
    model_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    warnings: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class Ctx:
    candidate: str
    class_id: str
    ledger: object  # budget.Ledger oder None


def _load_env_key(name: str) -> str:
    """Liest einen Key aus der Umgebung oder aus .env am Repo-Root."""
    if os.environ.get(name):
        return os.environ[name]
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{name}=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


# ---------------------------------------------------------------------------
# Kandidat: textlayer — deterministische Referenz-Baseline (PyMuPDF get_text)
# ---------------------------------------------------------------------------

def run_textlayer(input_path: str, ctx: Ctx) -> AdapterResult:
    """Nullpunkt-Kandidat: rohe Textebene, Seiten mit --- getrennt.

    Absichtlich strukturlos — misst, was ein PDF „gratis" hergibt. Auf Scans
    scheitert er per Konstruktion (leere Textebene) und belegt damit, dass
    das Harness Fehlschlaege sauber verbucht.
    """
    import fitz  # PyMuPDF

    warnings = []
    doc = fitz.open(input_path)
    pages = []
    empty = 0
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if len(text) < 10:
            empty += 1
        pages.append(text)
    doc.close()
    if empty:
        warnings.append(f"{empty}/{len(pages)} Seiten ohne/fast ohne Textebene")
    md = "\n\n---\n\n".join(p for p in pages if p)
    if not md.strip():
        raise RuntimeError(
            f"Keine Textebene extrahierbar ({len(pages)} Seiten leer) — "
            "Kandidat auf diesem Dokument nicht faehig"
        )
    return AdapterResult(markdown=md, warnings=warnings,
                         meta={"pages": len(pages), "empty_pages": empty})


# ---------------------------------------------------------------------------
# Kandidat: eigenbau — CONVERTERs PDFExtractionService (der Anlass des Sprints)
# ---------------------------------------------------------------------------

def _bootstrap_converter_packages():
    """Macht services.pdf_extraction + app_pkg.config importierbar, ohne die
    App zu laden (Fake-Parent-Packages mit __path__ auf die echten Ordner)."""
    import types
    for name, sub in (("services", "services"), ("app_pkg", "app_pkg")):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(REPO_ROOT / sub)]
            sys.modules[name] = pkg


def run_eigenbau(input_path: str, ctx: Ctx) -> AdapterResult:
    _bootstrap_converter_packages()
    import importlib
    svc_mod = importlib.import_module("services.pdf_extraction.service")

    api_key = _load_env_key("GEMINI_API_KEY")
    service = svc_mod.PDFExtractionService(api_key or None)

    counters = {"calls": 0, "tin": 0, "tout": 0, "cost": 0.0}
    warnings = []

    if service.gemini_client is not None:
        models_obj = service.gemini_client.models
        orig = models_obj.generate_content

        def counted_generate_content(*args, **kwargs):
            if ctx.ledger:
                ctx.ledger.precheck()
            resp = orig(*args, **kwargs)
            um = getattr(resp, "usage_metadata", None)
            tin = getattr(um, "prompt_token_count", 0) or 0
            total = getattr(um, "total_token_count", 0) or 0
            tout = max(total - tin, getattr(um, "candidates_token_count", 0) or 0)
            counters["calls"] += 1
            counters["tin"] += tin
            counters["tout"] += tout
            if ctx.ledger:
                model = kwargs.get("model") or service.VISION_MODEL
                counters["cost"] += ctx.ledger.record(
                    ctx.candidate, ctx.class_id, model, tin, tout)
            return resp

        models_obj.generate_content = counted_generate_content
    else:
        warnings.append("Kein GEMINI_API_KEY — Eigenbau laeuft rein lokal (Scan-Seiten leer)")

    md = service.extract_markdown(input_path)
    if not md.strip():
        raise RuntimeError("Eigenbau lieferte leeres Markdown")
    return AdapterResult(
        markdown=md,
        model_calls=counters["calls"],
        tokens_in=counters["tin"],
        tokens_out=counters["tout"],
        cost_usd=round(counters["cost"], 6),
        warnings=warnings,
        meta={"vision_model": service.VISION_MODEL,
              "gemini_enabled": service.gemini_client is not None},
    )


# ---------------------------------------------------------------------------
# Kandidat: gemini-nativ — gemini-3.6-flash mit nativem PDF-Input
# ---------------------------------------------------------------------------

GEMINI_MODEL = os.environ.get("PDF_VISION_MODEL") or "gemini-3.6-flash"
# Env-overridable: tabellendichte Seiten (04: ~3,3k Output-Tokens/Seite)
# sprengen bei 10 Seiten/Chunk die 32k max_output_tokens — der Abbruch ist
# dann ein Harness-Deckel, kein Kandidaten-Verhalten (gemessen am stillen
# 192-Zeilen-Loch von 04, Chunk 1 = exakt 32764 Tokens).
GEMINI_CHUNK_PAGES = int(os.environ.get("GEMINI_CHUNK_PAGES", "10"))
# 04 zeigte: HTML-Tabellenform kostet ~15 Tokens/Zelle — 5 dichte Ranking-
# Seiten ≈ 25-50k Output-Tokens. Deckel deshalb ebenfalls env-overridable.
GEMINI_MAX_OUTPUT = int(os.environ.get("GEMINI_MAX_OUTPUT", "32768"))

# media_resolution-Kalibrierung 2026-08-07 (3 Stufen x {01.gold nativ,
# 07.gold Scan}, Scores in results/gemini-cal-*/): MEDIUM gewinnt auf BEIDEN
# Regimen — 01: f1 0,981/CER 0,050 (low 0,966/0,079, high 0,968/0,068);
# 07: f1 0,979 + Zellen 0,779 + Regel1 3/3 (low: Regel1 0/3 und 67 statt 43
# Fill-Linien; high: Zellen 0,571). Die Register-Erwartung „LOW reicht fuer
# Native" hielt der Messung nicht stand.
GEMINI_LEVEL_BY_FORMAT = {
    "pdf": "medium",
    "pdf-scan": "medium",
    "pdf-mixed": "medium",
}

_GEMINI_PROMPT = """Convert this PDF document to faithful Markdown. Rules:

1. FIDELITY: Reproduce the exact wording of the source. Do NOT correct
   typos, spacing quirks, numbering errors or inconsistencies — they are
   part of the document. Do NOT invent or fill in anything.
2. TABLES: Use GFM pipe tables. If a table has merged cells, use a raw HTML
   <table> with colspan/rowspan instead. Every row must keep its column count.
3. TEXT: Headings via # by visual hierarchy; preserve bold/italic; footnotes
   as [^n] with definitions at the end.
4. FORMS: Keep blank fields blank. Render empty checkboxes as ☐, dotted or
   underscored fill-in lines as _____ (five underscores). Never fill them.
5. READING ORDER: Follow the visual reading order (columns top-to-bottom,
   left column before right column).
6. PAGES: Separate consecutive pages with a line containing only ---.
7. OUTPUT: Only the Markdown content. No commentary, no code fences."""


def _strip_fence(text: str) -> str:
    """Wie service._strip_wrapper_fence: nur eine Ganz-Antwort-Fence entfernen."""
    import re
    opening = re.match(r"```[^\n]*\n", text)
    if not opening:
        return text
    body = text[opening.end():]
    if body.count("```") == 1 and body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


def _gemini_thinking_configs(types_mod):
    """Fallback-Kette zur Thinking-Deckelung: Output-Tokens kosten 7,50/M
    INKLUSIVE Thinking — fuer reine Transkription ist Denkbudget Verschwendung.
    Welche Config das Modell akzeptiert, entscheidet der erste Call."""
    chain = []
    try:
        chain.append(("thinking_level=low",
                      types_mod.ThinkingConfig(thinking_level="low")))
    except Exception:
        pass
    try:
        chain.append(("thinking_budget=0",
                      types_mod.ThinkingConfig(thinking_budget=0)))
    except Exception:
        pass
    chain.append(("default", None))
    return chain


def _run_gemini_nativ(input_path: str, ctx: Ctx, level: str) -> AdapterResult:
    import time as _time
    import fitz
    from google import genai
    from google.genai import types

    api_key = _load_env_key("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY fehlt — Cloud-Kandidat nicht lauffaehig")
    client = genai.Client(api_key=api_key)

    resolution = {
        "low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
        "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
        "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
    }[level]

    doc = fitz.open(input_path)
    n_pages = len(doc)
    chunks = []
    if n_pages <= GEMINI_CHUNK_PAGES:
        chunks.append(Path(input_path).read_bytes())
    else:
        for start in range(0, n_pages, GEMINI_CHUNK_PAGES):
            sub = fitz.open()
            sub.insert_pdf(doc, from_page=start,
                           to_page=min(start + GEMINI_CHUNK_PAGES, n_pages) - 1)
            chunks.append(sub.tobytes())
            sub.close()
    doc.close()

    counters = {"calls": 0, "tin": 0, "tout": 0, "cost": 0.0}
    warnings = []
    thinking_chain = _gemini_thinking_configs(types)
    thinking_used = None
    parts_out = []

    for ci, pdf_bytes in enumerate(chunks):
        if ctx.ledger:
            ctx.ledger.precheck()
        response = None
        for tc_name, tc in list(thinking_chain):
            config_kwargs = dict(
                temperature=0.1,
                max_output_tokens=GEMINI_MAX_OUTPUT,
                media_resolution=resolution,
                http_options=types.HttpOptions(timeout=600_000),
            )
            if tc is not None:
                config_kwargs["thinking_config"] = tc
            try:
                for attempt in range(3):
                    try:
                        response = client.models.generate_content(
                            model=GEMINI_MODEL,
                            contents=[
                                types.Part.from_bytes(data=pdf_bytes,
                                                      mime_type="application/pdf"),
                                _GEMINI_PROMPT,
                            ],
                            config=types.GenerateContentConfig(**config_kwargs),
                        )
                        break
                    except Exception as e:
                        msg = str(e).lower()
                        if attempt < 2 and ("429" in msg or "rate" in msg or "resource" in msg):
                            _time.sleep(2.0 * (2 ** attempt))
                            continue
                        raise
            except Exception as e:
                # Nur Config-Ablehnungen (400/INVALID_ARGUMENT zu thinking)
                # eskalieren zur naechsten Stufe der Kette:
                msg = str(e).lower()
                if tc is not None and ("thinking" in msg or "invalid" in msg or "400" in msg):
                    warnings.append(f"ThinkingConfig {tc_name} abgelehnt: {str(e)[:120]}")
                    thinking_chain = [x for x in thinking_chain if x[0] != tc_name]
                    continue
                raise
            thinking_used = tc_name
            break
        if response is None:
            raise RuntimeError("Kein Gemini-Call erfolgreich (Thinking-Kette erschoepft)")

        um = getattr(response, "usage_metadata", None)
        tin = getattr(um, "prompt_token_count", 0) or 0
        total = getattr(um, "total_token_count", 0) or 0
        tout = max(total - tin, getattr(um, "candidates_token_count", 0) or 0)
        counters["calls"] += 1
        counters["tin"] += tin
        counters["tout"] += tout
        if ctx.ledger:
            counters["cost"] += ctx.ledger.record(
                ctx.candidate, ctx.class_id, GEMINI_MODEL, tin, tout,
                note=f"chunk {ci + 1}/{len(chunks)} level={level}")

        fr = ""
        try:
            fr = str(response.candidates[0].finish_reason)
        except Exception:
            pass
        if "MAX_TOKENS" in fr:
            warnings.append(f"Chunk {ci + 1}: Output an max_output_tokens abgeschnitten")
        text = response.text or ""
        if not text.strip():
            warnings.append(f"Chunk {ci + 1}: leere Antwort")
        parts_out.append(_strip_fence(text.strip()))

    md = "\n\n---\n\n".join(p for p in parts_out if p)
    if not md.strip():
        raise RuntimeError("Gemini lieferte insgesamt leeres Markdown")
    return AdapterResult(
        markdown=md, model_calls=counters["calls"],
        tokens_in=counters["tin"], tokens_out=counters["tout"],
        cost_usd=round(counters["cost"], 6), warnings=warnings,
        meta={"model": GEMINI_MODEL, "media_resolution": level,
              "chunks": len(chunks), "chunk_pages": GEMINI_CHUNK_PAGES,
              "thinking": thinking_used},
    )


def run_gemini_nativ(input_path: str, ctx: Ctx) -> AdapterResult:
    from manifest import CLASSES
    fmt = CLASSES[ctx.class_id]["format"]
    return _run_gemini_nativ(input_path, ctx, GEMINI_LEVEL_BY_FORMAT[fmt])


def _make_gemini_fixed(level):
    def run(input_path: str, ctx: Ctx) -> AdapterResult:
        return _run_gemini_nativ(input_path, ctx, level)
    return run


# ---------------------------------------------------------------------------
# Kandidat: unstructured — der CONVERTER-Office-Pfad (partition fast +
# DOC-FIX-Serializer). Pin vs. aktuell entscheidet das ENV, nicht der Code.
# ---------------------------------------------------------------------------

def run_unstructured(input_path: str, ctx: Ctx) -> AdapterResult:
    _bootstrap_converter_packages()
    import importlib
    import importlib.metadata
    from unstructured.partition.auto import partition
    um = importlib.import_module("services.unstructured_markdown")

    elements = partition(filename=input_path, strategy="fast")
    md, warns = um.elements_to_markdown(elements, source_ext=Path(input_path).suffix)
    if not md.strip():
        raise RuntimeError("Serializer lieferte leeres Markdown")
    return AdapterResult(
        markdown=md, warnings=list(warns),
        # 0.18.x exponiert __version__ als Submodul, nicht als String —
        # importlib.metadata ist die versionsfeste Quelle.
        meta={"unstructured_version": importlib.metadata.version("unstructured"),
              "strategy": "fast", "serializer": "services.unstructured_markdown"},
    )


# ---------------------------------------------------------------------------
# Kandidat: docling — ⚠️ >= v2.109.0, OCR-Engine EXPLIZIT, nie `auto`
# (Befund-Register R-05: der Default reicht die Sprache nicht durch).
# ---------------------------------------------------------------------------

def run_docling(input_path: str, ctx: Ctx) -> AdapterResult:
    import docling
    version = getattr(docling, "__version__", "0")
    parts = [int(x) for x in str(version).split(".")[:3] if x.isdigit()]
    if parts < [2, 109, 0]:
        raise RuntimeError(f"docling {version} < 2.109.0 — Sprint-Mindestversion (Deutsch-OCR-Fix)")

    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

    warnings = []
    opts = PdfPipelineOptions()
    opts.do_ocr = True
    opts.do_table_structure = True
    try:
        opts.table_structure_options.mode = TableFormerMode.ACCURATE
    except Exception as e:
        warnings.append(f"TableFormer ACCURATE nicht setzbar: {e}")

    ocr_engine = None
    try:
        from docling.datamodel.pipeline_options import RapidOcrOptions
        opts.ocr_options = RapidOcrOptions(lang=["de"])
        ocr_engine = "rapidocr/de"
    except Exception:
        from docling.datamodel.pipeline_options import TesseractCliOcrOptions
        opts.ocr_options = TesseractCliOcrOptions(lang=["deu"])
        ocr_engine = "tesseract-cli/deu"

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    result = converter.convert(input_path)
    md = result.document.export_to_markdown()
    if not md.strip():
        raise RuntimeError("docling lieferte leeres Markdown")
    status = getattr(result, "status", None)
    return AdapterResult(
        markdown=md, warnings=warnings,
        meta={"docling_version": version, "ocr_engine": ocr_engine,
              "tableformer": "ACCURATE", "status": str(status)},
    )


# ---------------------------------------------------------------------------
# Kandidat: pandoc — DOCX-Referenz (traegt Fussnoten). Subprozess, kein venv.
# ---------------------------------------------------------------------------

def run_pandoc(input_path: str, ctx: Ctx) -> AdapterResult:
    import shutil
    import subprocess
    exe = shutil.which("pandoc")
    if not exe:
        raise RuntimeError("pandoc nicht im PATH")
    proc = subprocess.run(
        [exe, "-f", "docx", "-t", "gfm", "--wrap=none", input_path],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"pandoc rc={proc.returncode}: {proc.stderr[:400]}")
    version = subprocess.run([exe, "--version"], capture_output=True,
                             text=True).stdout.splitlines()[0]
    md = proc.stdout
    if not md.strip():
        raise RuntimeError("pandoc lieferte leeres Markdown")
    return AdapterResult(markdown=md,
                         warnings=[proc.stderr[:300]] if proc.stderr.strip() else [],
                         meta={"pandoc": version, "to": "gfm"})


# ---------------------------------------------------------------------------
# Kandidat: markitdown — PPTX (Sprint-Tabelle; XLSX existiert im Korpus nicht)
# ---------------------------------------------------------------------------

def run_markitdown(input_path: str, ctx: Ctx) -> AdapterResult:
    from markitdown import MarkItDown
    import markitdown as mid_pkg
    result = MarkItDown().convert(input_path)
    md = getattr(result, "text_content", "") or ""
    if not md.strip():
        raise RuntimeError("markitdown lieferte leeres Markdown")
    return AdapterResult(
        markdown=md,
        meta={"markitdown_version": getattr(mid_pkg, "__version__", "?")},
    )


# ---------------------------------------------------------------------------
# Kandidat: trafilatura — HTML-Artikel (Boilerplate-Entfernung ist der Zweck)
# ---------------------------------------------------------------------------

def run_trafilatura(input_path: str, ctx: Ctx) -> AdapterResult:
    import trafilatura
    raw = Path(input_path).read_bytes()
    html = None
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            html = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    md = trafilatura.extract(
        html, output_format="markdown", include_tables=True,
        include_links=True, include_formatting=True, include_comments=False,
    )
    if not md or not md.strip():
        raise RuntimeError("trafilatura extrahierte nichts")
    return AdapterResult(
        markdown=md,
        meta={"trafilatura_version": trafilatura.__version__, "encoding": enc},
    )


# ---------------------------------------------------------------------------
# Kandidat: tesseract — die CPU-OCR-Referenz (BASELINE-OCR ist der Nullpunkt).
# Rendert 300 dpi und OCRt IMMER, auch wenn eine Textebene existiert —
# fuer Klasse 14 ist genau das der Rettungspfad.
# ---------------------------------------------------------------------------

def run_tesseract(input_path: str, ctx: Ctx) -> AdapterResult:
    import shutil
    import subprocess
    import tempfile
    import fitz
    exe = shutil.which("tesseract")
    if not exe:
        raise RuntimeError("tesseract nicht im PATH")
    version = subprocess.run([exe, "--version"], capture_output=True,
                             text=True).stdout.splitlines()[0]
    doc = fitz.open(input_path)
    pages, empty = [], 0
    warnings = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            png = f"{tmp}/p{i}.png"
            pix.save(png)
            proc = subprocess.run([exe, png, "stdout", "-l", "deu"],
                                  capture_output=True, text=True, timeout=300)
            text = proc.stdout.strip()
            if proc.returncode != 0:
                warnings.append(f"Seite {i + 1}: rc={proc.returncode} {proc.stderr[:120]}")
            if len(text) < 10:
                empty += 1
            pages.append(text)
    doc.close()
    if empty:
        warnings.append(f"{empty}/{len(pages)} Seiten (fast) ohne OCR-Text")
    md = "\n\n---\n\n".join(p for p in pages if p)
    if not md.strip():
        raise RuntimeError("Tesseract lieferte keinerlei Text")
    return AdapterResult(markdown=md, warnings=warnings,
                         meta={"tesseract": version, "lang": "deu", "dpi": 300,
                               "pages": len(pages)})


# ---------------------------------------------------------------------------
# P3: GPU-Kandidaten — docker-run-Wrapper, laufen NUR auf der Mintbox
# (A2000 12 GB). Eigene Container, CONVERTERs Stack unberuehrt. Jeder Lauf
# wird mit einem nvidia-smi-Sampler umschlossen → vram_peak_mb in meta.
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.expanduser("~/bakeoff-models")


def _require_gpu_host():
    import shutil
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("GPU-Kandidat: nvidia-smi fehlt — nur auf der Mintbox lauffaehig")


class _VramSampler:
    def __enter__(self):
        import subprocess
        import tempfile
        self.log = tempfile.NamedTemporaryFile(mode="w+", suffix=".vram", delete=False)
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-l", "2"],
            stdout=self.log, stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        self.log.flush()

    def peak_mb(self):
        try:
            vals = [int(x) for x in Path(self.log.name).read_text().split()
                    if x.strip().isdigit()]
            return max(vals) if vals else None
        finally:
            Path(self.log.name).unlink(missing_ok=True)


def _docker_convert(image: str, inner_cmd: list, input_path: str,
                    extra_args: list = None, timeout: int = 5400) -> tuple:
    """docker run mit /in (ro) + /out (tmp) + Modell-Cache; liefert (md, meta)."""
    import subprocess
    import tempfile
    src = Path(input_path)
    with tempfile.TemporaryDirectory() as out_dir:
        # Container laufen als root (--user scheitert an fehlenden
        # passwd-Eintraegen: mineru wirft `getpwuid(): uid not found` —
        # live getroffen). Der EPERM-Gegenpart (root-eigene /out-Dateien,
        # die der Host-Glob nicht lesen darf — ebenfalls live getroffen)
        # wird nach dem Lauf image-agnostisch per busybox-chmod geloest.
        cmd = ["docker", "run", "--rm", "--gpus", "all", "--shm-size", "16g",
               "-v", f"{src.parent}:/in:ro", "-v", f"{out_dir}:/out",
               "-v", f"{MODELS_DIR}:/models",
               "-e", "HF_HOME=/models",
               "-e", "MINERU_MODEL_SOURCE=huggingface",
               ] + (extra_args or []) + [image] + inner_cmd
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        # chown (nicht chmod): a+rX liess das rmtree des TemporaryDirectory
        # an root-eigenen Unterordnern sterben (EPERM beim Cleanup).
        subprocess.run(["docker", "run", "--rm", "-v", f"{out_dir}:/out",
                        "busybox", "chown", "-R",
                        f"{os.getuid()}:{os.getgid()}", "/out"],
                       capture_output=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Container rc={proc.returncode}: {proc.stderr[-800:] or proc.stdout[-800:]}")
        md_files = sorted(Path(out_dir).rglob("*.md"),
                          key=lambda p: p.stat().st_size, reverse=True)
        if not md_files:
            raise RuntimeError(f"Kein .md im Container-Output (stdout: {proc.stdout[-300:]})")
        md = md_files[0].read_text(encoding="utf-8", errors="replace")
        return md, {"container_cmd": " ".join(inner_cmd),
                    "md_file": md_files[0].name,
                    "stderr_tail": proc.stderr[-300:]}


def run_mineru_vlm(input_path: str, ctx: Ctx) -> AdapterResult:
    _require_gpu_host()
    src = Path(input_path)
    with _VramSampler() as vram:
        md, meta = _docker_convert(
            "mineru:latest",
            # mineru 3.4.4: Backend heisst `vlm-engine` (die 2.x-Doku sagte
            # noch vlm-vllm-engine — live gegen --help verifiziert).
            ["mineru", "-p", f"/in/{src.name}", "-o", "/out",
             "-b", "vlm-engine"],
            input_path)
    meta["vram_peak_mb"] = vram.peak_mb()
    if not md.strip():
        raise RuntimeError("MinerU lieferte leeres Markdown")
    return AdapterResult(markdown=md, meta=meta)


SURYA_SERVER_NAME = "bakeoff-surya-vllm"
SURYA_MODEL = "datalab-to/surya-ocr-2"


def _ensure_openai_server(name: str, model: str, port: int,
                          gpu_util: str, max_len: str,
                          extra_server_args: list = None,
                          timeout_s: int = 1800) -> None:
    """Startet einen vLLM-OpenAI-Server als Geschwister-Container, falls er
    nicht laeuft. Gemeinsame Mechanik fuer surya (marker v2 spawnt sonst
    selbst docker-in-docker — live gescheitert) und dots.ocr."""
    import subprocess
    import time as _time
    import urllib.request

    def alive():
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/v1/models", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    if alive():
        return
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    subprocess.run([
        "docker", "run", "-d", "--name", name, "--gpus", "all",
        "--shm-size", "16g", "-p", f"{port}:8000",
        "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "-v", f"{MODELS_DIR}:/root/.cache/huggingface",
        "vllm/vllm-openai:latest",
        model, "--trust-remote-code",
        "--gpu-memory-utilization", gpu_util, "--max-model-len", max_len,
    ] + (extra_server_args or []), check=True, capture_output=True, text=True)
    t0 = _time.monotonic()
    while _time.monotonic() - t0 < timeout_s:
        if alive():
            return
        _time.sleep(5)
    logs = subprocess.run(["docker", "logs", "--tail", "30", name],
                          capture_output=True, text=True).stderr[-800:]
    raise RuntimeError(f"vLLM-Server {name} ({model}) kam nicht hoch: {logs}")


def run_marker2(input_path: str, ctx: Ctx) -> AdapterResult:
    _require_gpu_host()
    # marker v2s surya-Backend wuerde selbst einen vllm-Container spawnen
    # (docker-in-docker, im Container unmoeglich). Stattdessen: Geschwister-
    # Server mit dem erwarteten Checkpoint. Gewichte nur 1,3 GB — der erste
    # Start starb NICHT an ihnen, sondern an vLLMs Default-Profiling
    # (max_num_seqs 256 x 18k Kontext -> KV -5,18 GiB). Klein dimensioniert
    # bleibt >6 GB fuer markers eigene Torch-Modelle.
    # ⚠️ --limit-mm-per-prompt video:0 ist load-bearing: vLLM profiliert
    # sonst den Multimodal-Encoder-Cache mit 114k-Token-VIDEO-Budget und
    # der KV-Cache faellt auf -5 GiB, egal wie klein Batch/Kontext sind
    # (live: mit dem Limit 16k-Budget, KV +4,22 GiB, Server in 300s oben).
    _ensure_openai_server(SURYA_SERVER_NAME, SURYA_MODEL, 8001,
                          gpu_util="0.6", max_len="8192",
                          extra_server_args=["--max-num-seqs", "16",
                                             "--enforce-eager",
                                             "--limit-mm-per-prompt",
                                             '{"image":4,"video":0}'])
    src = Path(input_path)
    with _VramSampler() as vram:
        md, meta = _docker_convert(
            "bakeoff-marker:latest",
            ["marker_single", f"/in/{src.name}", "--output_dir", "/out",
             "--output_format", "markdown"],
            input_path,
            extra_args=["--network", "host",
                        "-e", "SURYA_INFERENCE_BACKEND=vllm",
                        "-e", "SURYA_INFERENCE_URL=http://127.0.0.1:8001/v1"])
    meta["vram_peak_mb"] = vram.peak_mb()
    meta["surya_server"] = SURYA_MODEL
    if not md.strip():
        raise RuntimeError("marker lieferte leeres Markdown")
    return AdapterResult(markdown=md, meta=meta)


DOTS_SERVER_NAME = "bakeoff-dots-vllm"
DOTS_MODEL = "rednote-hilab/dots.ocr"


def run_vlm_dots(input_path: str, ctx: Ctx) -> AdapterResult:
    _require_gpu_host()
    # 3B-Gewichte ~6 GB; klein dimensionierte Batch/Kontext-Budgets, damit
    # KV + Profiling in die 12 GB passen (Lehre aus dem surya-Start).
    # Host-Port 8003: 8000 ist auf der Mintbox anderweitig belegt.
    _ensure_openai_server(DOTS_SERVER_NAME, DOTS_MODEL, 8003,
                          gpu_util="0.72", max_len="24576",
                          extra_server_args=["--max-num-seqs", "2",
                                             "--enforce-eager",
                                             "--limit-mm-per-prompt",
                                             '{"image":2,"video":0}',
                                             # Upstream-Parser fragt nach
                                             # dem Literal-Namen `model`:
                                             "--served-model-name", "model"])
    src = Path(input_path)
    with _VramSampler() as vram:
        md, meta = _docker_convert(
            "bakeoff-dotsclient:latest",
            ["python3", "/opt/dots.ocr/dots_ocr/parser.py", f"/in/{src.name}",
             "--output", "/out", "--ip", "127.0.0.1", "--port", "8003",
             "--prompt", "prompt_layout_all_en"],
            input_path,
            extra_args=["--network", "host"])
    meta["vram_peak_mb"] = vram.peak_mb()
    meta["server_model"] = DOTS_MODEL
    if not md.strip():
        raise RuntimeError("dots.ocr lieferte leeres Markdown")
    return AdapterResult(markdown=md, meta=meta)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTERS = {
    "textlayer": {
        "run": run_textlayer,
        "formats": {"pdf", "pdf-scan", "pdf-mixed"},
        "env": "eigenbau",  # braucht nur PyMuPDF, laeuft im selben venv
        "beschreibung": "PyMuPDF-Textebene roh — deterministischer Nullpunkt",
    },
    "eigenbau": {
        "run": run_eigenbau,
        "formats": {"pdf", "pdf-scan", "pdf-mixed"},
        "env": "eigenbau",
        "beschreibung": "CONVERTERs PDFExtractionService (Ensemble + Gemini-Fallback)",
    },
    "gemini-nativ": {
        "run": run_gemini_nativ,
        "formats": {"pdf", "pdf-scan", "pdf-mixed"},
        "env": "eigenbau",  # braucht nur fitz + google-genai
        "beschreibung": "gemini-3.6-flash, natives PDF, Stufe nach Format (Kalibrierung)",
    },
    # Fixierte Stufen NUR fuer die media_resolution-Kalibrierung auf den
    # Gold-Inputs (Sprint: alle drei Stufen auf mind. einem Dokument):
    "gemini-cal-low": {
        "run": _make_gemini_fixed("low"),
        "formats": {"pdf", "pdf-scan", "pdf-mixed"},
        "env": "eigenbau",
        "beschreibung": "Kalibrierung: media_resolution=LOW",
    },
    "gemini-cal-medium": {
        "run": _make_gemini_fixed("medium"),
        "formats": {"pdf", "pdf-scan", "pdf-mixed"},
        "env": "eigenbau",
        "beschreibung": "Kalibrierung: media_resolution=MEDIUM",
    },
    "gemini-cal-high": {
        "run": _make_gemini_fixed("high"),
        "formats": {"pdf", "pdf-scan", "pdf-mixed"},
        "env": "eigenbau",
        "beschreibung": "Kalibrierung: media_resolution=HIGH",
    },
    "unstructured-pin": {
        "run": run_unstructured,
        "formats": {"docx", "pptx", "html", "eml"},
        "env": "unstructured-pin",  # ==0.18.32 (Prod-Pin)
        "beschreibung": "CONVERTER-Office-Pfad mit Prod-Pin 0.18.32",
    },
    "unstructured-neu": {
        "run": run_unstructured,
        "formats": {"docx", "pptx", "html", "eml"},
        "env": "unstructured-neu",  # ==0.24.1
        "beschreibung": "CONVERTER-Office-Pfad mit aktuellem 0.24.1 (Drift-Messung)",
    },
    "docling": {
        "run": run_docling,
        "formats": {"pdf", "pdf-scan", "pdf-mixed", "docx", "pptx"},
        "env": "docling",
        "beschreibung": "docling >=2.109.0, TableFormer ACCURATE, OCR explizit de/deu",
    },
    "pandoc": {
        "run": run_pandoc,
        "formats": {"docx"},
        "env": "eigenbau",  # nur Subprozess; irgendein Python reicht
        "beschreibung": "pandoc -t gfm (DOCX-Referenz, traegt Fussnoten)",
    },
    "markitdown": {
        "run": run_markitdown,
        "formats": {"pptx"},
        "env": "markitdown",
        "beschreibung": "markitdown (PPTX; XLSX-Klasse existiert im Korpus nicht)",
    },
    "trafilatura": {
        "run": run_trafilatura,
        "formats": {"html"},
        "env": "trafilatura",
        "beschreibung": "trafilatura markdown-Output (Artikel-Extraktion)",
    },
    "tesseract": {
        "run": run_tesseract,
        "formats": {"pdf", "pdf-scan", "pdf-mixed"},
        "env": "eigenbau",  # fitz zum Rendern + tesseract-Binary
        "beschreibung": "Tesseract 5 deu @300dpi — CPU-OCR-Referenz (immer OCR, nie Textebene)",
    },
    # --- P3: GPU-Feld (nur Mintbox) ---
    "mineru-vlm": {
        "run": run_mineru_vlm,
        "formats": {"pdf", "pdf-scan", "pdf-mixed"},
        "env": "mintbox-host",
        "beschreibung": "MinerU 3.x, VLM-Backend via vllm-engine im offiziellen Container",
    },
    "marker2": {
        "run": run_marker2,
        "formats": {"pdf", "pdf-scan", "pdf-mixed"},
        "env": "mintbox-host",
        "beschreibung": "marker v2 (datalab) im CUDA-Container",
    },
    "vlm-dots": {
        "run": run_vlm_dots,
        "formats": {"pdf", "pdf-scan", "pdf-mixed"},
        "env": "mintbox-host",
        "beschreibung": "dots.ocr (3B) via vLLM-Server + Upstream-Parser-Client",
    },
}
