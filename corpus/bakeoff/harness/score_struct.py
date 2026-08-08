# corpus/bakeoff/harness/score_struct.py
"""Metrik (b): strukturelle Kennzahlen ohne Gold, fuer alle 14 Klassen.

Jede Kennzahl in einem Satz begruendet (Sprint-Vorgabe):

* ``pipe_tables`` / ``html_tables`` / ``table_rows_max`` — kommt Tabellarisches
  als Tabelle heraus statt als Fliesstext, und ueberlebt eine seitenlange
  Tabelle als EIN Block (03: Fortsetzung ueber Seitengrenzen)?
* ``colspan_attrs`` / ``rowspan_attrs`` — verbundene Zellen (04, 08) sind nur
  in HTML-Tabellen ausdrueckbar; ihr Fehlen heisst Struktur eingeebnet.
* ``headings`` — erscheint die Dokumenthierarchie, und auf wie vielen Ebenen?
* ``word_recall`` / ``word_precision`` — ueberlebt der Textbestand gegen die
  deterministische Referenz (Verlust), und kommt nichts dazu (Halluzination)?
* ``order_lcs`` — Multiset sagt „Woerter da", LCS sagt „in der Reihenfolge
  der Quelle" (Lesereihenfolge, zweispaltige Klassen).
* ``length_ratio`` — faengt Truncation UND Aufblaehung (Loops) in einer Zahl.
* ``umlauts_out`` / ``umlaut_recall`` — der belegte Deutsch-Fehlermodus ist
  Umlautverlust bei hoher Konfidenz (BASELINE-OCR); fuer 14 ist die Kombination
  hoher Recall gegen die kaputte Ebene + 0 Umlaute der Durchreich-Beweis.
* ``max_line_repeat`` / ``max_ngram_repeat`` / ``loop_flag`` — der dokumentierte
  VLM-Fehlermodus auf Punktlinien (06/07) ist der Wiederholungs-Loop; gemessen
  statt uebersehen. ``length_ratio`` ist bewusst NICHT Teil des Flags
  (partiell gedeckte Referenzen wie 13 erzeugten Fehlalarme).
* ``footnote_defs`` — Fussnoten sind das Klassenmerkmal von 08 und in 12 real
  (pandoc traegt sie, docling verwirft sie — Sprint-Tabelle).
* ``fill_tokens`` / ``checkbox_tokens`` — ueberleben Formular-Markierungen (07)
  als solche, statt zu verschwinden oder zu wuchern?

Aufruf (stdlib, jedes Env):
    python harness/score_struct.py --candidate eigenbau [--class 01] [--role main]
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import normalize as nz  # noqa: E402
from manifest import CLASSES, REFERENCES, result_dir  # noqa: E402

UMLAUTS = "äöüÄÖÜß"


def _pipe_table_blocks(md: str) -> list:
    """Bloecke aufeinanderfolgender Pipe-Zeilen, die eine Separator-Zeile enthalten."""
    blocks, cur = [], []
    for line in md.splitlines() + [""]:
        if line.lstrip().startswith("|") or (line.count("|") >= 2 and not cur):
            cur.append(line)
        else:
            if cur:
                blocks.append(cur)
                cur = []
    return [b for b in blocks
            if len(b) >= 2 and any(re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", ln) for ln in b)]


def _max_consecutive_line_repeat(lines: list) -> int:
    best = run = 1
    prev = None
    for ln in lines:
        if ln and ln == prev:
            run += 1
            best = max(best, run)
        else:
            run = 1
        prev = ln
    return best


_LOOP_SYMBOLS = {nz.FILL, nz.CHECKBOX, nz.NA_DASH}


def _cells_iter(canon: str):
    """Alle Zellinhalte: HTML-<td>/<th> und Pipe-Zellen."""
    for m in re.finditer(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", canon, re.S | re.I):
        yield m.group(1)
    for line in canon.splitlines():
        if line.lstrip().startswith("|"):
            for cell in line.strip().strip("|").split("|"):
                yield cell


def _loop_signals(canon: str) -> dict:
    """Zell- und Zeilen-lokale Wiederholungssignale.

    Der vom Judge verifizierte mineru-Loop wiederholte in EINER Zelle ~30x
    „□ <OCR-Rauschvariante>" — exakte n-Gramm-/Zeilenvergleiche sehen das
    nicht (Varianten differieren), und die Zeile ist als Einheit wertlos,
    sobald ein Kandidat ganze HTML-Tabellen einzeilig schreibt. Deshalb:
    Symbole (Fill/Checkbox/Strich) pro ZELLE (Gold-Maximum: 3/Zelle;
    Schwelle 8) und Inhaltswoerter >=4 Zeichen pro Zeile (Schwelle 25 —
    Absatz-Langzeilen tragen legitim 15x „der", das zaehlt nicht).
    """
    from collections import Counter
    sym_cell = 0
    for cell in _cells_iter(canon):
        c = sum(cell.count(s) for s in _LOOP_SYMBOLS)
        sym_cell = max(sym_cell, c)
    wrd = 0
    for line in canon.splitlines():
        toks = nz.words(nz.to_plain(line)) if line.strip() else []
        if len(toks) < 25:
            continue
        for t, n in Counter(toks).most_common(6):
            if len(t) >= 4 and t not in _LOOP_SYMBOLS:
                wrd = max(wrd, n)
    return {"symbol_pro_zelle": sym_cell, "wort_pro_zeile": wrd}


def _max_ngram_repeat(tokens: list, n: int = 8) -> int:
    """Maximale ANZAHL direkt aufeinanderfolgender Wiederholungen eines n-Gramms."""
    if len(tokens) < 2 * n:
        return 1
    best = 1
    i = 0
    while i + 2 * n <= len(tokens):
        if tokens[i:i + n] == tokens[i + n:i + 2 * n]:
            reps = 2
            j = i + 2 * n
            while j + n <= len(tokens) and tokens[j:j + n] == tokens[i:i + n]:
                reps += 1
                j += n
            best = max(best, reps)
            i = j
        else:
            i += 1
    return best


def score_output(md: str, ref_text: str = None, order_limit: int = 30000) -> dict:
    canon = nz.canonicalize(md)
    plain = nz.to_plain(md)
    out_words = nz.words(plain)
    lines = [ln.strip() for ln in canon.splitlines()]

    pipe_blocks = _pipe_table_blocks(canon)
    html_tables = len(re.findall(r"<table\b", canon, re.I))
    row_counts = [sum(1 for ln in b if ln.lstrip().startswith("|")) for b in pipe_blocks]
    html_rows = len(re.findall(r"<tr\b", canon, re.I))

    s = {
        "pipe_tables": len(pipe_blocks),
        "pipe_rows": sum(row_counts),
        "table_rows_max": max(row_counts + [html_rows] or [0]),
        "html_tables": html_tables,
        "colspan_attrs": len(re.findall(r"colspan=", canon, re.I)),
        "rowspan_attrs": len(re.findall(r"rowspan=", canon, re.I)),
        "headings": {f"h{i}": len(re.findall(rf"^#{{{i}}}\s", canon, re.M))
                     for i in range(1, 5)},
        "footnote_defs": len(re.findall(r"^\[\^[^\]]+\]:", canon, re.M)),
        "fill_tokens": canon.count(nz.FILL),
        "checkbox_tokens": canon.count(nz.CHECKBOX),
        "chars_out": len(md),
        "words_out": len(out_words),
        "umlauts_out": sum(md.count(c) for c in UMLAUTS),
        "max_line_repeat": _max_consecutive_line_repeat(lines),
        "max_ngram_repeat": _max_ngram_repeat(out_words),
        "loop_signale": _loop_signals(canon),
    }

    if ref_text:
        ref_plain = nz.to_plain(ref_text)
        ref_words = nz.words(ref_plain)
        s.update(nz.word_multiset_prf(ref_words, out_words))
        s["length_ratio"] = round(len(plain) / max(len(ref_plain), 1), 3)
        umlauts_ref = sum(ref_text.count(c) for c in UMLAUTS)
        s["umlauts_ref"] = umlauts_ref
        s["umlaut_recall"] = round(s["umlauts_out"] / umlauts_ref, 3) if umlauts_ref else None
        if len(ref_words) <= order_limit and len(out_words) <= order_limit:
            import difflib
            sm = difflib.SequenceMatcher(None, ref_words, out_words, autojunk=False)
            lcs = sum(bl.size for bl in sm.get_matching_blocks())
            s["order_lcs"] = round(lcs / max(len(ref_words), 1), 4)
        else:
            s["order_lcs"] = None
            s.setdefault("notes", []).append(
                f"order_lcs uebersprungen (>{order_limit} Woerter)")

    # length_ratio bleibt eigenstaendiges Signal und geht NICHT in loop_flag:
    # bei partiell gedeckten Referenzen (13: Textebene nur auf nativen Seiten)
    # ist ein hohes Verhaeltnis korrekte Mehrleistung, kein Loop — gemessen
    # am tesseract-13-Fehlalarm (ratio 5,56 bei line_repeat=1, ngram=1).
    ls = s["loop_signale"]
    s["loop_flag"] = bool(s["max_line_repeat"] >= 8 or s["max_ngram_repeat"] >= 5
                          or ls["symbol_pro_zelle"] >= 8 or ls["wort_pro_zeile"] >= 25)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--class", dest="class_id", choices=sorted(CLASSES))
    ap.add_argument("--role", choices=["main", "gold"], default="main")
    args = ap.parse_args()

    class_ids = [args.class_id] if args.class_id else sorted(CLASSES)
    for cid in class_ids:
        rd = result_dir(args.candidate, cid, args.role)
        out_file = rd / "output.md"
        if not out_file.exists():
            continue
        md = out_file.read_text(encoding="utf-8")
        ref_file = REFERENCES / f"{cid}.txt"
        ref = ref_file.read_text(encoding="utf-8") if ref_file.exists() else None
        scores = score_output(md, ref)
        scores["_ref"] = ref_file.name if ref else None
        path = rd / "struct.json"
        path.write_text(json.dumps(scores, indent=1, ensure_ascii=False), encoding="utf-8")
        flag = " ⚠LOOP" if scores["loop_flag"] else ""
        rec = scores.get("word_recall")
        rec_s = f" recall={rec}" if rec is not None else ""
        print(f"OK: {args.candidate} × {cid}({args.role}) → struct.json"
              f" tables={scores['pipe_tables']}+{scores['html_tables']}html{rec_s}{flag}")


if __name__ == "__main__":
    main()
