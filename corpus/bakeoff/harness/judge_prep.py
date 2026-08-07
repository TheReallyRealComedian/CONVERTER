# corpus/bakeoff/harness/judge_prep.py
"""Judge-Inputs (Metrik c): Original-Seiten als PNG je Klasse.

Deterministische Seiten-Stichprobe — DIESELBE fuer alle Kandidaten (Rubrik):
<=6 Seiten → alle; sonst {1, 2, Mitte, Mitte+1, letzte}. DOCX/PPTX gehen
vorher durch soffice → PDF (nur fuers Rendern; das ist KEIN Kandidat).
HTML/EML bekommen keine Renders — dort ist der deterministische
Referenztext (results/_references/) die Vergleichsbasis des Judges.

Output: results/_judge/<klasse>/p<seite>.png + _sample.json (welche Seiten).
Unversioniert (abgeleitet). Aufruf im eigenbau-Env:
    python harness/judge_prep.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import CLASSES, RESULTS, input_path  # noqa: E402

JUDGE = RESULTS / "_judge"
DPI = 150


def sample_pages(n: int) -> list:
    if n <= 6:
        return list(range(n))
    mid = n // 2
    return sorted({0, 1, mid, mid + 1, n - 1})


def render_pdf(pdf_path: Path, out_dir: Path) -> list:
    import fitz
    doc = fitz.open(str(pdf_path))
    pages = sample_pages(len(doc))
    for p in pages:
        pix = doc[p].get_pixmap(dpi=DPI)
        pix.save(str(out_dir / f"p{p + 1}.png"))
    n = len(doc)
    doc.close()
    return [p + 1 for p in pages], n


def to_pdf_via_soffice(src: Path, tmp: Path) -> Path:
    subprocess.run(
        ["soffice", "--headless", "--convert-to", "pdf",
         "--outdir", str(tmp), str(src)],
        capture_output=True, timeout=600, check=True,
    )
    produced = tmp / (src.stem + ".pdf")
    if not produced.exists():
        cands = list(tmp.glob("*.pdf"))
        if not cands:
            raise RuntimeError(f"soffice erzeugte kein PDF fuer {src.name}")
        produced = cands[0]
    return produced


def main():
    only = sys.argv[1:] or None
    meta = {}
    for cid, cls in CLASSES.items():
        if only and cid not in only:
            continue
        fmt = cls["format"]
        out_dir = JUDGE / cid
        out_dir.mkdir(parents=True, exist_ok=True)
        src = input_path(cid)
        try:
            if fmt.startswith("pdf"):
                pages, n = render_pdf(src, out_dir)
            elif fmt in ("docx", "pptx"):
                with tempfile.TemporaryDirectory() as tmp:
                    pdf = to_pdf_via_soffice(src, Path(tmp))
                    pages, n = render_pdf(pdf, out_dir)
            else:
                meta[cid] = {"renders": None,
                             "hinweis": "kein Render — Referenztext ist die Basis"}
                continue
            meta[cid] = {"renders": pages, "seiten_gesamt": n, "dpi": DPI,
                         "via_soffice": fmt in ("docx", "pptx")}
            print(f"OK: {cid} — Seiten {pages} von {n}")
        except Exception as e:
            meta[cid] = {"fehler": str(e)[:300]}
            print(f"FEHLER: {cid} — {e}")
    (JUDGE / "_sample.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
