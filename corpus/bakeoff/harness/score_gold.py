# corpus/bakeoff/harness/score_gold.py
"""Metrik (a): Vergleich gegen die Gold-Fassungen 01/07/08.

Die Bewertungsregeln aus corpus/gold/_UNSICHERHEITEN.md sind TEIL der Metrik:

* **Regel 1** (07): die drei Fettbalken zaehlen als richtig, wenn sie als
  Ueberschrift ODER als verbundene Kopfzeile (``<th colspan>``) erscheinen.
* **Regel 2** (01): Tief-/Hochstellungen zaehlen als richtig in JEDER
  Notation, die die Unterscheidung erhaelt (LaTeX, ``_``-Marker, ``<sub>``,
  ``~``-Pandoc, Unicode); Einebnen (``Pout``, ``lrand``) ist der Fehler.
* **Regel 3** (08): jede Bild-Syntax mit beliebigem Ziel zaehlt; gemessen
  wird nur, dass die Abbildung markiert ist und die Fussnote an ihr haengt.

Die Quell-Eigenheiten-Liste ist ein PRUEFKRITERIUM: wer ``Markteilnehmer``
zu ``Marktteilnehmer`` „korrigiert", hat einen Fehler gemacht.

Textvergleich = Wort-Multiset (bidirektional, wie die Gold-Verifikation
selbst) + CER auf normalisiertem Fliesstext. Zellvergleich = Multiset der
nicht-leeren Zellinhalte ueber alle Tabellen (robust gegen Split/Merge)
plus Formen (Zeilen × Spalten) beider Seiten.

01/07 laufen gegen die abgeleiteten Gold-Seiten-Inputs (role=gold);
08 laeuft gegen den Volltext-Output und wird anker-basiert geschnitten.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import normalize as nz  # noqa: E402
from manifest import CLASSES, gold_path, result_dir  # noqa: E402


# --- Zell-Extraktion --------------------------------------------------------

_SEP_ROW = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")


# Zellvergleich ist notations-agnostisch im Geist von Regel 2: ``$2.1$`` und
# ``2.1`` sind dieselbe Zelle, ``$\ell_{rand}$`` und ``ℓ_rand`` auch —
# gemessen am gemini-nativ-Lauf, der JEDE Zahlenzelle math-wrappte und damit
# von 0,86 auf 0,32 Zell-Recall fiel, ohne eine Ziffer zu aendern.
_CELL_LATEX = {
    "\\langle": "⟨", "\\rangle": "⟩", "\\ell": "ℓ", "\\gamma": "γ",
    "\\kappa": "κ", "\\times": "×", "\\sim": "∼", "\\ast": "∗", "\\pm": "±",
}


def _strip_cell(cell: str) -> str:
    c = re.sub(r"<br\s*/?>", " ", cell, flags=re.I)
    c = re.sub(r"<[^>]+>", " ", c)
    c = re.sub(r"[*`]{1,3}", "", c)
    c = c.replace("$", "")
    for k, v in _CELL_LATEX.items():
        c = c.replace(k, v)
    c = c.replace("{", "").replace("}", "")
    c = re.sub(r"\s+", " ", c)
    return c.strip()


def extract_cells(md_canon: str) -> dict:
    """Zellen aus Pipe- UND HTML-Tabellen; Formen je Tabelle."""
    cells, shapes = [], []
    # Pipe-Tabellen
    block = []
    for line in md_canon.splitlines() + [""]:
        if line.lstrip().startswith("|"):
            block.append(line)
        else:
            if len(block) >= 2:
                rows = [ln for ln in block if not _SEP_ROW.match(ln)]
                ncols = 0
                for ln in rows:
                    row = [c for c in ln.strip().strip("|").split("|")]
                    row = [_strip_cell(c) for c in row]
                    ncols = max(ncols, len(row))
                    cells.extend(c for c in row if c)
                if rows:
                    shapes.append((len(rows), ncols, "pipe"))
            block = []
    # HTML-Tabellen
    for tbl in re.findall(r"<table\b.*?</table>", md_canon, re.S | re.I):
        trs = re.findall(r"<tr\b.*?</tr>", tbl, re.S | re.I)
        ncols = 0
        for tr in trs:
            tds = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", tr, re.S | re.I)
            ncols = max(ncols, len(tds))
            cells.extend(c for c in (_strip_cell(x) for x in tds) if c)
        shapes.append((len(trs), ncols, "html"))
    return {"cells": cells, "shapes": shapes}


def _cell_prf(gold_cells: list, out_cells: list) -> dict:
    from collections import Counter
    gc, oc = Counter(gold_cells), Counter(out_cells)
    overlap = sum((gc & oc).values())
    recall = overlap / max(sum(gc.values()), 1)
    precision = overlap / max(sum(oc.values()), 1)
    return {
        "cell_recall": round(recall, 4),
        "cell_precision": round(precision, 4),
        "cells_gold": sum(gc.values()),
        "cells_out": sum(oc.values()),
        "top_missing_cells": [f"{c[:60]}×{n}" for c, n in (gc - oc).most_common(10)],
        "top_invented_cells": [f"{c[:60]}×{n}" for c, n in (oc - gc).most_common(10)],
    }


# --- Regel 2: Tief-/Hochstellungen (01) --------------------------------------

_SUB_MARK = r"(?:_\{?|<sub[^>]*>|~)"
_SUP_MARK = r"(?:\^\{?|<sup[^>]*>|~~?)"

R2_FAMILIES = [
    ("ell_rand", r"(?:\\ell|ℓ|l)", "rand"),
    ("ell_real", r"(?:\\ell|ℓ|l)", "real"),
    ("ell_pow", r"(?:\\ell|ℓ|l)", "pow"),
    ("P_out", r"P", "out"),
    ("P_in", r"P", "in"),
    ("gamma_in", r"(?:\\gamma|γ|gamma)", "in"),
    ("gamma_out", r"(?:\\gamma|γ|gamma)", "out"),
    ("C_rand", r"C", "rand"),
]


def check_r2(out_raw: str) -> dict:
    res = {}
    for name, base, sub in R2_FAMILIES:
        preserved = re.search(rf"{base}\s*{_SUB_MARK}\s*{sub}\b", out_raw)
        flattened = re.search(rf"{base}{sub}\b", out_raw)
        res[name] = ("erhalten" if preserved
                     else "eingeebnet" if flattened else "fehlt")
    # Hochstellung dom (γ_in^dom) und numerische Exponenten 10⁻⁴/10⁻⁵
    res["dom_sup"] = ("erhalten" if re.search(rf"{_SUP_MARK}\s*\{{?dom", out_raw)
                      or "^dom" in out_raw
                      else "eingeebnet" if re.search(r"(?:in|γ|gamma)dom", out_raw)
                      else "fehlt")
    # Numerische Exponenten: Unicode-Hochstellung nutzt U+2074/U+2075 (⁴/⁵),
    # nicht die ASCII-Ziffer — beide Formen zaehlen als erhalten.
    for exp, sup in (("4", "⁴"), ("5", "⁵")):
        if re.search(rf"10\s*(?:⁻\s*{sup}|[-−]\s*{sup}|{_SUP_MARK}\s*[-−]?\s*{exp}|<sup>\s*[-−]\s*{exp})",
                     out_raw):
            res[f"exp_minus_{exp}"] = "erhalten"
        elif re.search(rf"10\s?[-−]\s?{exp}\b", out_raw):
            res[f"exp_minus_{exp}"] = "eingeebnet"
        else:
            res[f"exp_minus_{exp}"] = "fehlt"
    counts = {"erhalten": 0, "eingeebnet": 0, "fehlt": 0}
    for v in res.values():
        counts[v] += 1
    return {"familien": res, "zusammenfassung": counts}


# --- Regel 1: Fettbalken (07) -------------------------------------------------

R1_BALKEN = [
    "Angaben zur letzten bisherigen",
    "Sonstige Angaben zu Familienangehörigen",
    "Angaben zur Vergabe einer Krankenversichertennummer",
]


def check_r1(out_canon: str) -> dict:
    res = {}
    for text in R1_BALKEN:
        pat = re.escape(text)
        if re.search(rf"^#{{1,6}}\s+.*{pat}", out_canon, re.M):
            res[text] = "ueberschrift"
        elif re.search(rf"<t[dh][^>]*colspan[^>]*>[^<]*{pat}", out_canon, re.I):
            res[text] = "colspan_kopf"
        elif re.search(pat, out_canon):
            res[text] = "vorhanden_unstrukturiert"
        else:
            res[text] = "fehlt"
    ok = sum(1 for v in res.values() if v in ("ueberschrift", "colspan_kopf"))
    return {"balken": res, "gleichwertig_erfuellt": f"{ok}/3"}


# --- Regel 3: Abbildung + Fussnote (08) ---------------------------------------

def check_r3(out_canon: str) -> dict:
    img_positions = [m.start() for m in re.finditer(r"!\[[^\]]*\]\(|<img\b", out_canon)]
    bild_markiert = bool(img_positions)
    fussnote_am_bild = False
    for pos in img_positions:
        window = out_canon[max(0, pos - 120):pos + 200]
        if re.search(r"\[\^[^\]]+\]|<sup>", window):
            fussnote_am_bild = True
            break
    return {
        "bild_markiert": bild_markiert,
        "fussnote_am_bild": fussnote_am_bild,
        "fussnote_definiert": bool(re.search(r"^\[\^[^\]]+\]:", out_canon, re.M))
                              or "GNU FDL" in out_canon,
        "link_erhalten": "Five-forces.gif" in out_canon,
    }


# --- Quell-Eigenheiten ---------------------------------------------------------

def _nr_sequence_verdict(out_canon: str) -> str:
    """TABLE-II-Eigenheit: Nr.-Spalte traegt 14, 14, 16 (die 15 fehlt).

    TABLE I enthaelt legitim eine 15 — deshalb kein globaler ``| 15 |``-Test,
    sondern das Fenster (14, 14, 16) in der Folge der Zeilenend-Zahlen.
    """
    seq = [int(m.group(1))
           for m in re.finditer(r"\|\s*(\d{1,2})\s*\|\s*$", out_canon, re.M)]
    # HTML-Tabellen (mineru, gemini): letzte Zelle jeder <tr> zaehlt.
    for tr in re.findall(r"<tr\b.*?</tr>", out_canon, re.S | re.I):
        tds = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", tr, re.S | re.I)
        if tds:
            last = re.sub(r"<[^>]+>", "", tds[-1]).strip()
            if re.fullmatch(r"\d{1,2}", last):
                seq.append(int(last))
    for i in range(len(seq) - 2):
        if seq[i:i + 3] == [14, 14, 16]:
            return "erhalten"
    # „Repariert" hiesse: die TABLE-II-Region traegt 14,15,16 — erkennbar an
    # ZWEI getrennten 14,15,16-Fenstern (eines gehoert ohnehin TABLE I).
    windows = sum(1 for i in range(len(seq) - 2) if seq[i:i + 3] == [14, 15, 16])
    return "repariert" if windows >= 2 else "nicht nachweisbar (Tabellen fehlen/veraendert)"


def check_eigenheiten_01(out_canon: str) -> dict:
    return {
        "nr_14_doppelt_15_fehlt": _nr_sequence_verdict(out_canon),
        "bereich_absteigend_636_618": "erhalten" if re.search(r"6\.36\s*-\s*6\.18", out_canon)
                                      else ("repariert" if re.search(r"6\.18\s*-\s*6\.36", out_canon)
                                            else "fehlt"),
        "punkt_460_902": "erhalten" if "460.902" in out_canon else "fehlt/geaendert",
        "komma_460_902": "erhalten" if re.search(r"460,\s?902", out_canon) else "fehlt/geaendert",
    }


def check_eigenheiten_08(out_raw: str) -> dict:
    def verdict(erhalten: bool, repariert: bool) -> str:
        return "erhalten" if erhalten else ("REPARIERT (Fehler)" if repariert else "fehlt")
    return {
        "markteilnehmer_ein_t": verdict("Markteilnehmer" in out_raw,
                                        "Marktteilnehmer" in out_raw),
        "auflage_ohne_leerzeichen": verdict("19.Auflage" in out_raw,
                                            bool(re.search(r"19\.\s+Auflage", out_raw))),
        "s189_ohne_leerzeichen": verdict("S.189." in out_raw,
                                         bool(re.search(r"S\.\s+189", out_raw))),
    }


# --- 08: Anker-Slicing ----------------------------------------------------------

def slice_08(out_canon: str) -> tuple:
    """Schneidet den Gold-Abschnitt (Wettbewerbsanalyse … SWOT-Tabelle) heraus.

    Start: Ueberschrift/Zeile „Wettbewerbsanalyse", disambiguiert dadurch,
    dass kurz danach „Branchenstrukturanalyse" folgt (gegen TOC-Treffer).
    Ende: Ende des Tabellenblocks mit der letzten SWOT-Zelle
    („…Suchoptionen"). Fussnoten-Definitionen des Slices werden aus dem
    Gesamtdokument nachgezogen (Konverter setzen sie ans Dokumentende).
    """
    notes = []
    starts = [m.start() for m in re.finditer(r"Wettbewerbsanalyse", out_canon)]
    start = None
    for pos in starts:
        if "Branchenstrukturanalyse" in out_canon[pos:pos + 800]:
            start = out_canon.rfind("\n", 0, pos) + 1
            break
    if start is None:
        if starts:
            start = out_canon.rfind("\n", 0, starts[0]) + 1
            notes.append("Start-Anker ohne Disambiguierung (Branchenstrukturanalyse fehlt danach)")
        else:
            return out_canon, ["Start-Anker fehlt — ganzer Output bewertet"]

    end_anchor = out_canon.rfind("Suchoptionen")
    if end_anchor == -1:
        sliced = out_canon[start:]
        notes.append("End-Anker fehlt — bis Dokumentende bewertet")
    else:
        rest = out_canon[end_anchor:]
        m = re.search(r"</table>", rest, re.I)
        if m:
            end = end_anchor + m.end()
        else:
            nl = rest.find("\n\n")
            end = end_anchor + (nl if nl != -1 else len(rest))
        sliced = out_canon[start:end]
    defs = [ln for ln in out_canon.splitlines()
            if re.match(r"^\[\^[^\]]+\]:", ln) and ln not in sliced]
    if defs:
        sliced += "\n" + "\n".join(defs)
    return sliced, notes


# --- Hauptvergleich --------------------------------------------------------------

def score_against_gold(class_id: str, out_raw: str) -> dict:
    gold_raw = gold_path(class_id).read_text(encoding="utf-8")
    gold_canon = nz.canonicalize(gold_raw)
    out_canon = nz.canonicalize(out_raw)
    notes = []

    if CLASSES[class_id].get("gold_slice"):
        out_canon, slice_notes = slice_08(out_canon)
        notes.extend(slice_notes)

    gold_plain, out_plain = nz.to_plain(gold_canon), nz.to_plain(out_canon)
    text = nz.word_multiset_prf(nz.words(gold_plain), nz.words(out_plain))
    text["cer"] = nz.cer(gold_plain, out_plain)
    try:
        import rapidfuzz  # noqa: F401
    except ImportError:
        notes.append("CER via difflib-Approximation (rapidfuzz nicht installiert)")

    g_cells = extract_cells(gold_canon)
    o_cells = extract_cells(out_canon)
    cells = _cell_prf(g_cells["cells"], o_cells["cells"])
    cells["shapes_gold"] = g_cells["shapes"]
    cells["shapes_out"] = o_cells["shapes"]

    scores = {"text": text, "cells": cells, "notes": notes}

    if class_id == "01":
        # Auf kanonisiertem Text: entmaskiert \_ und laesst LaTeX/HTML intakt.
        scores["regel2_stellungen"] = check_r2(out_canon)
        scores["eigenheiten"] = check_eigenheiten_01(out_canon)
    elif class_id == "07":
        scores["regel1_fettbalken"] = check_r1(out_canon)
        scores["formular"] = {
            "fill_tokens_gold": gold_canon.count(nz.FILL),
            "fill_tokens_out": out_canon.count(nz.FILL),
            "checkbox_gold": gold_canon.count(nz.CHECKBOX),
            "checkbox_out": out_canon.count(nz.CHECKBOX),
        }
    elif class_id == "08":
        scores["regel3_abbildung"] = check_r3(out_canon)
        scores["eigenheiten"] = check_eigenheiten_08(out_raw)
        scores["colspan_rowspan"] = {
            "colspan_out": len(re.findall(r"colspan=", out_canon, re.I)),
            "rowspan_out": len(re.findall(r"rowspan=", out_canon, re.I)),
            "colspan_gold": len(re.findall(r"colspan=", gold_canon, re.I)),
            "rowspan_gold": len(re.findall(r"rowspan=", gold_canon, re.I)),
        }
    return scores


GOLD_ROLE = {"01": "gold", "07": "gold", "08": "main"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--class", dest="class_id", choices=sorted(GOLD_ROLE))
    args = ap.parse_args()

    for cid in ([args.class_id] if args.class_id else sorted(GOLD_ROLE)):
        rd = result_dir(args.candidate, cid, GOLD_ROLE[cid])
        out_file = rd / "output.md"
        if not out_file.exists():
            continue
        scores = score_against_gold(cid, out_file.read_text(encoding="utf-8"))
        path = rd / "gold_scores.json"
        path.write_text(json.dumps(scores, indent=1, ensure_ascii=False), encoding="utf-8")
        t = scores["text"]
        print(f"OK: {args.candidate} × {cid} → gold_scores.json"
              f" f1={t['word_f1']} cer={t['cer']}"
              f" cells R/P={scores['cells']['cell_recall']}/{scores['cells']['cell_precision']}")


if __name__ == "__main__":
    main()
