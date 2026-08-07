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
    # P2: gemini-nativ, docling, unstructured 0.18.32/0.24.1, pandoc,
    #     markitdown, trafilatura, tesseract — je ein Eintrag + Funktion.
}
