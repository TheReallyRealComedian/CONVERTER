# corpus/bakeoff/harness/normalize.py
"""Gemeinsame Text-Normalisierung fuer alle Metriken.

Grundsatz: normalisiert wird nur, was **Darstellung** ist, nie was Inhalt ist.
Jede Regel hier ist eine dokumentierte Gleichwertigkeits-Entscheidung:

* ``\\_`` → ``_``  — Gold ``07.md`` maskiert Unterstriche gegen Emphasis-Parsing;
  die Maskierung ist Rohtext-Kosmetik (_UNSICHERHEITEN.md sagt ausdruecklich:
  vor dem Diff normalisieren).
* Ausfuelllinien-Laeufe (Punkte, Auslassungspunkte, Unterstriche, ≥3 Zeichen)
  → ein kanonisches Token. Gold normalisiert die Laenge selbst auf ``_____``;
  ein Kandidat, der ``........`` oder 12 Unterstriche schreibt, meint dasselbe.
* Ankreuzkaestchen-Varianten (☐ □ ❑ ▢ [ ]) → ein kanonisches Token. Gold waehlt
  U+2610 als Konvention; die Wahl des Zeichens ist keine Faehigkeit.
* Strich-Varianten (- − – —) → ``-`` fuer Text-Diffs. Die Gegenpruefung von
  ``01.md`` haelt fest, dass ein Werkzeug mit ``30−40`` „naeher an der
  Textebene" ist als Gold — Strich-Codepoints duerfen nicht bestrafen.
  (Der Entfaellt-Gedankenstrich in 07 wird VOR der Vereinheitlichung als
  eigenes Token geschuetzt: eine einzelne ``—``-Zelle ist Inhalt.)
* Typografische Anfuehrungszeichen → gerade. Ligaturen/NFC wie in Gold.

NICHT normalisiert: Gross-/Kleinschreibung, Umlaute, Zahlen, Wortlaut —
genau dort liegt die Messung (inkl. der Quell-Eigenheiten, die ein Werkzeug
nicht „reparieren" darf).
"""

import re
import unicodedata

FILL = ""      # Private-Use: kanonisches Ausfuellfeld
CHECKBOX = ""  # Private-Use: kanonisches Ankreuzkaestchen
NA_DASH = ""   # Private-Use: Entfaellt-Strich (einzelne —-Zelle)

_FILL_RUN = re.compile(r"(?:[._]\s?){3,}|[…‥]{1,}\.*|\.{3,}")
_CHECKBOX = re.compile(r"[☐□❑▢]|\[\s?\]")
_DASHES = re.compile(r"[−–—‐‑‒]")
_QUOTES = {ord(c): '"' for c in "„“”«»"} | {ord(c): "'" for c in "‚‘’‹›"}


def canonicalize(text: str) -> str:
    """Markdown-erhaltende Kanonisierung (fuer Struktur-Checks am Rohtext)."""
    t = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    t = t.replace("\\_", "_")
    # Entfaellt-Strich schuetzen: — allein in einer Tabellenzelle oder Zeile
    t = re.sub(r"(?<=\|)\s*—\s*(?=\|)", f" {NA_DASH} ", t)
    t = re.sub(r"^\s*—\s*$", NA_DASH, t, flags=re.M)
    t = _FILL_RUN.sub(FILL, t)
    t = re.sub(f"{FILL}(?:\\s*{FILL})+", FILL, t)  # zusammengesetzte Laeufe
    t = _CHECKBOX.sub(CHECKBOX, t)
    t = _DASHES.sub("-", t)
    t = t.translate(_QUOTES)
    return t


_MD_SYNTAX = re.compile(
    r"^#{1,6}\s+|^\s*[-*+]\s+|^\s*\d+\.\s+|^\s*>\s?|[*_]{1,3}(?=\S)|(?<=\S)[*_]{1,3}",
    re.M,
)
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}.*$", re.M)
_HTML_TAG = re.compile(r"<[^>\n]{1,120}>")
_LINK = re.compile(r"!?\[([^\]]*)\]\(([^)]*)\)")
_FOOTNOTE_DEF = re.compile(r"^\[\^[^\]]+\]:\s?", re.M)
_FOOTNOTE_REF = re.compile(r"\[\^[^\]]+\]")


def to_plain(text: str) -> str:
    """Markdown/HTML → Fliesstext fuer Wort-/Zeichenvergleiche.

    Link-/Bildsyntax faellt auf den sichtbaren Text zusammen (URL ist
    Faehigkeits-Check in score_gold, nicht Textbestand). Fussnoten-Marker
    fallen weg (Syntax), Fussnoten-INHALT bleibt.
    """
    t = canonicalize(text)
    t = _TABLE_SEP.sub("", t)
    t = _LINK.sub(lambda m: m.group(1), t)
    t = _FOOTNOTE_DEF.sub("", t)
    t = _FOOTNOTE_REF.sub("", t)
    t = _HTML_TAG.sub(" ", t)
    t = _MD_SYNTAX.sub("", t)
    t = t.replace("|", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


_WORD = re.compile(rf"[\w{FILL}{CHECKBOX}{NA_DASH}]+", re.UNICODE)


def words(text: str) -> list:
    """Wortliste aus bereits plain-gemachtem Text."""
    return _WORD.findall(text)


def word_multiset_prf(ref_words: list, out_words: list) -> dict:
    """Bidirektionaler Multiset-Vergleich — dieselbe Methode, mit der die
    Gold-Fassungen verifiziert wurden (0 Woerter ohne Gegenstueck)."""
    from collections import Counter
    rc, oc = Counter(ref_words), Counter(out_words)
    overlap = sum((rc & oc).values())
    recall = overlap / max(sum(rc.values()), 1)
    precision = overlap / max(sum(oc.values()), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    missing = (rc - oc).most_common(15)
    invented = (oc - rc).most_common(15)
    return {
        "word_recall": round(recall, 4),
        "word_precision": round(precision, 4),
        "word_f1": round(f1, 4),
        "words_ref": sum(rc.values()),
        "words_out": sum(oc.values()),
        "top_missing": [f"{w}×{n}" for w, n in missing],
        "top_invented": [f"{w}×{n}" for w, n in invented],
    }


def cer(ref: str, out: str) -> float:
    """Zeichenfehler-Rate auf normalisiertem Fliesstext.

    Whitespace wird auf Einzel-Leerzeichen kollabiert (Layout ist keine
    Treue). Levenshtein via rapidfuzz, wenn vorhanden; sonst difflib-
    Approximation (deterministisch, nicht minimal — wird im Score vermerkt).
    """
    a = re.sub(r"\s+", " ", ref).strip()
    b = re.sub(r"\s+", " ", out).strip()
    if not a:
        return 0.0 if not b else 1.0
    try:
        from rapidfuzz.distance import Levenshtein
        return round(Levenshtein.distance(a, b) / len(a), 4)
    except ImportError:
        import difflib
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        dist = sum(max(i2 - i1, j2 - j1)
                   for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
        return round(dist / len(a), 4)
