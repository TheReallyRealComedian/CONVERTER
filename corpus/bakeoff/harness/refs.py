# corpus/bakeoff/harness/refs.py
"""Baut abgeleitete Gold-Inputs und deterministische Referenztexte.

Einmal pro Korpus-Stand laufen lassen (im eigenbau-Env, braucht PyMuPDF):
    python harness/refs.py

1. derived/01_gold-seiten.pdf — die zwei transkribierten Seiten des Papers
   (Seite „the WWW as a network has boomed…" + Seite „TABLE I."), inhaltlich
   verifiziert statt blind per Index.
2. derived/07_gold-seite2.pdf — Seite 2 des Formular-Scans.
3. results/_references/<klasse>.txt — deterministischer Textbestand je Klasse
   fuer die Struktur-Metrik (b): PDFs via PyMuPDF-Textebene, DOCX/PPTX via
   stdlib-Zipfile+XML (w:t / a:t — inkl. SmartArt-diagrams und Notes),
   HTML via stdlib-Parser, EML via email-Modul. Scans (05/06) haben keine.

Die Referenz ist bewusst ROH (keine Reparatur): fuer 14 ist sie die kaputte
OCR-Ebene — hoher Recall dagegen + fehlende Umlaute heisst „kommentarlos
durchgereicht", genau das prueft die Klasse.
"""

import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import CLASSES, CORPUS, DERIVED, REFERENCES, input_path  # noqa: E402


def build_gold_pdf_01():
    import fitz
    src = input_path("01")
    doc = fitz.open(str(src))
    page_text = None
    idx_text = idx_table = None
    for i in range(len(doc)):
        page_text = doc[i].get_text("text")
        if idx_text is None and "as a network has boomed" in page_text:
            idx_text = i
        if idx_table is None and "TABLE I." in page_text:
            idx_table = i
    if idx_text is None or idx_table is None:
        raise RuntimeError(f"Gold-Seiten in 01 nicht gefunden (text={idx_text}, table={idx_table})")
    doc.select([idx_text, idx_table])
    out = DERIVED / CLASSES["01"]["gold_input"]
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    doc.close()
    print(f"OK: {out.name} (Seitenindizes {idx_text}, {idx_table})")


def build_gold_pdf_03():
    """Zwei aufeinanderfolgende Seiten, auf denen EINE Sub-Tabelle ueberlaeuft.

    Inhaltlich verifiziert statt per Index: gesucht ist das erste Seitenpaar,
    dessen Sub-Tabellen-Kopf (Kopf-Ebene x~93, unterhalb des Haupt-Kopfes)
    auf BEIDEN Seiten identisch ist — genau dann laeuft dieselbe Liste ueber
    die Grenze und der Kopf wird wiederholt (bei 2010-II: Seiten 11+12, OMS).
    """
    import fitz
    src = input_path("03")
    doc = fitz.open(str(src))

    def sub_head(i):
        heads = [(b[1], " ".join(b[4].split()))
                 for b in doc[i].get_text("blocks")
                 if 92 < b[0] < 95 and b[1] < 700 and b[4].strip()]
        heads.sort()
        # heads[0] ist der Haupt-Kopf ("Uebersicht der Angebote…"), heads[1]
        # der Sub-Kopf. Fehlt einer, ist die Seite kein Kandidat.
        return heads[1][1] if len(heads) > 1 else None

    pair = None
    for i in range(len(doc) - 1):
        a, b = sub_head(i), sub_head(i + 1)
        if a and a == b:
            pair = (i, i + 1)
            break
    if pair is None:
        raise RuntimeError("03: kein Seitenpaar mit wiederholtem Sub-Kopf gefunden")
    doc.select(list(pair))
    out = DERIVED / CLASSES["03"]["gold_input"]
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    doc.close()
    print(f"OK: {out.name} (Seitenindizes {pair[0]}, {pair[1]})")


def build_gold_pdf_07():
    import fitz
    src = input_path("07")
    doc = fitz.open(str(src))
    if len(doc) < 2:
        raise RuntimeError(f"07-Scan hat {len(doc)} Seite(n), erwartet 2")
    doc.select([1])
    out = DERIVED / CLASSES["07"]["gold_input"]
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    doc.close()
    print(f"OK: {out.name}")


# --- Referenz-Extraktoren -------------------------------------------------

def _pdf_text(path: Path) -> tuple:
    import fitz
    doc = fitz.open(str(path))
    parts, empty = [], 0
    for page in doc:
        t = page.get_text("text").strip()
        if len(t) < 10:
            empty += 1
        parts.append(t)
    n = len(doc)
    doc.close()
    return "\n\n".join(p for p in parts if p), f"{n} Seiten, {empty} ohne Textebene"


# ⚠️ Tag-Grenze ist Pflicht: `<w:t[^>]*>` matcht auch `<w:tab/>` und
# `<w:txbxContent>` und schluckt dann rohes XML bis zum naechsten `</w:t>`
# (gemessen an 08: VML-Styles „mso/position/margin" blaehten die Referenz um
# Faktor ~6). `(?:\s[^>]*|/)?` erlaubt nur Attribute oder Self-Closing.
_XML_TEXT = {
    "docx": (re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.S),
             ["word/document.xml", "word/footnotes.xml", "word/endnotes.xml"]),
    "pptx": (re.compile(r"<a:t(?:\s[^>]*)?>(.*?)</a:t>", re.S), None),
}


def _unescape(s: str) -> str:
    return (s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
             .replace("&quot;", '"').replace("&apos;", "'"))


def _docx_text(path: Path) -> tuple:
    rx, members = _XML_TEXT["docx"]
    parts = []
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        for m in members:
            if m in names:
                xml = z.read(m).decode("utf-8", "replace")
                # Absatzgrenzen als Zeilenumbrueche erhalten:
                xml = xml.replace("</w:p>", "</w:p>\n")
                parts.append("\n".join(
                    "".join(rx.findall(para)) for para in xml.split("\n")))
    text = _unescape("\n".join(parts))
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text, f"w:t aus {len(parts)} XML-Teilen"


def _slide_no(name: str) -> int:
    m = re.search(r"(\d+)\.xml$", name)
    return int(m.group(1)) if m else 0


def _pptx_text(path: Path) -> tuple:
    rx, _ = _XML_TEXT["pptx"]
    slides, notes, diagrams = [], [], []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if re.match(r"ppt/slides/slide\d+\.xml$", name):
                slides.append(name)
            elif re.match(r"ppt/notesSlides/notesSlide\d+\.xml$", name):
                notes.append(name)
            elif re.match(r"ppt/diagrams/data\d*\.xml$", name):
                diagrams.append(name)
        parts = []
        for group in (sorted(slides, key=_slide_no),
                      sorted(diagrams, key=_slide_no),
                      sorted(notes, key=_slide_no)):
            for name in group:
                xml = z.read(name).decode("utf-8", "replace")
                xml = xml.replace("</a:p>", "</a:p>\n")
                parts.append("\n".join(
                    "".join(rx.findall(para)) for para in xml.split("\n")))
    text = _unescape("\n".join(parts))
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text, (f"{len(slides)} Folien, {len(diagrams)} SmartArt-XML, "
                  f"{len(notes)} Notes — alle a:t")


def _html_text(path: Path) -> tuple:
    from html.parser import HTMLParser

    class Grab(HTMLParser):
        def __init__(self):
            super().__init__()
            self.skip = 0
            self.out = []

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style"):
                self.skip += 1

        def handle_endtag(self, tag):
            if tag in ("script", "style") and self.skip:
                self.skip -= 1

        def handle_data(self, data):
            if not self.skip and data.strip():
                self.out.append(data.strip())

    raw = path.read_bytes()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            html = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    g = Grab()
    g.feed(html)
    return "\n".join(g.out), f"HTML-Textknoten, Kodierung {enc}"


def _eml_text(path: Path) -> tuple:
    import email
    from email import policy
    msg = email.message_from_bytes(path.read_bytes(), policy=policy.default)
    body = msg.get_body(preferencelist=("plain", "html"))
    if body is None:
        return "", "kein Body gefunden"
    content = body.get_content()
    if body.get_content_type() == "text/html":
        content = re.sub(r"<[^>]+>", " ", content)
    return content.strip(), f"Body {body.get_content_type()}"


def build_references():
    REFERENCES.mkdir(parents=True, exist_ok=True)
    meta = {}
    for cid, cls in CLASSES.items():
        if cls.get("no_reference"):
            meta[cid] = {"note": "kein deterministischer Referenztext (Scan)"}
            continue
        src = CORPUS / cls["dir"] / cls.get("reference_file", cls["file"])
        fmt = cls["format"]
        if fmt.startswith("pdf"):
            text, note = _pdf_text(src)
        elif fmt == "docx":
            text, note = _docx_text(src)
        elif fmt == "pptx":
            text, note = _pptx_text(src)
        elif fmt == "html":
            text, note = _html_text(src)
        elif fmt == "eml":
            text, note = _eml_text(src)
        else:
            raise ValueError(f"Unbekanntes Format {fmt}")
        (REFERENCES / f"{cid}.txt").write_text(text, encoding="utf-8")
        meta[cid] = {"chars": len(text), "source": src.name, "note": note}
        if cls.get("reference_note"):
            meta[cid]["lesehinweis"] = cls["reference_note"]
        print(f"OK: Referenz {cid} — {len(text)} Zeichen ({note})")
    (REFERENCES / "_meta.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    build_gold_pdf_01()
    build_gold_pdf_03()
    build_gold_pdf_07()
    build_references()
