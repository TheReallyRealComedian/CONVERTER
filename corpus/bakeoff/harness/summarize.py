# corpus/bakeoff/harness/summarize.py
"""Rollup: alle results/ → results/SUMMARY.md (Kandidat × Klasse).

Reine Ableitung aus den dateibasierten Ergebnissen — jederzeit neu erzeugbar,
Quelle für die P2-Ergebnistabelle. Struktur:
  1. Statusmatrix Kandidat × Klasse (OK/Fehler/fehlt)
  2. Gold-Tabelle (01/07/08): f1, CER, Zellen, Regeln, Eigenheiten
  3. Struktur-Kennzahlen je Klasse (Recall/Precision/Ordnung/Tabellen/Loops)
  4. Kosten & Laufzeit (LEDGER + metrics)

Aufruf: python harness/summarize.py  (stdlib)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import CLASSES, RESULTS  # noqa: E402

SKIP_DIRS = {"_references"}
CAL_PREFIX = "gemini-cal-"  # Kalibrierungslaeufe: eigener Abschnitt, nicht Feld


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def collect():
    rows = {}
    for cand_dir in sorted(RESULTS.iterdir()):
        if not cand_dir.is_dir() or cand_dir.name in SKIP_DIRS:
            continue
        for run_dir in sorted(cand_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            m = load(run_dir / "metrics.json")
            if not m:
                continue
            rows[(cand_dir.name, run_dir.name)] = {
                "metrics": m,
                "struct": load(run_dir / "struct.json"),
                "gold": load(run_dir / "gold_scores.json"),
            }
    return rows


def fmt(v, spec=".3f", na="—"):
    if v is None:
        return na
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def main():
    rows = collect()
    field = {k: v for k, v in rows.items() if not k[0].startswith(CAL_PREFIX)
             and k[0] != "goldself"}
    candidates = sorted({c for c, _ in field})
    class_runs = sorted({r for _, r in field})

    out = ["# Bake-off — Rollup (automatisch aus results/)", ""]

    # 1. Statusmatrix
    out += ["## Statusmatrix (main-Läufe)", ""]
    main_runs = [r for r in class_runs if not r.endswith(".gold")]
    out.append("| Kandidat | " + " | ".join(main_runs) + " |")
    out.append("|---|" + "---|" * len(main_runs))
    for c in candidates:
        cells = []
        for r in main_runs:
            e = field.get((c, r))
            if not e:
                cells.append("·")
            elif "error" in e["metrics"]:
                cells.append("✗ " + e["metrics"]["error"]["type"])
            else:
                cells.append("✓")
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    out.append("")

    # 2. Gold
    out += ["## Gegen Gold (Metrik a) — inkl. Kalibrierungs-Kandidaten", ""]
    out.append("| Kandidat | Dok | f1 | CER | ZellenR | ZellenP | Regeln | Eigenheiten |")
    out.append("|---|---|---|---|---|---|---|---|")
    for (c, r), e in sorted(rows.items()):
        g = e["gold"]
        if not g or c == "goldself":
            continue
        t, z = g["text"], g["cells"]
        regeln = []
        if "regel1_fettbalken" in g:
            regeln.append("R1 " + g["regel1_fettbalken"]["gleichwertig_erfuellt"])
        if "regel2_stellungen" in g:
            zs = g["regel2_stellungen"]["zusammenfassung"]
            regeln.append(f"R2 {zs['erhalten']}/11")
        if "regel3_abbildung" in g:
            r3 = g["regel3_abbildung"]
            regeln.append("R3 " + "/".join("✓" if r3[k] else "✗" for k in
                          ("bild_markiert", "fussnote_am_bild", "fussnote_definiert", "link_erhalten")))
        eig = ""
        if g.get("eigenheiten"):
            vals = list(g["eigenheiten"].values())
            bad = sum(1 for v in vals if "REPARIERT" in v or v == "repariert")
            eig = f"{sum(1 for v in vals if v == 'erhalten')}/{len(vals)} erhalten" + \
                  (f", {bad} repariert!" if bad else "")
        out.append(f"| {c} | {r} | {t['word_f1']:.4f} | {t['cer']:.4f} | "
                   f"{z['cell_recall']:.3f} | {z['cell_precision']:.3f} | "
                   f"{'; '.join(regeln)} | {eig} |")
    out.append("")

    # 3. Struktur je Klasse
    out += ["## Struktur (Metrik b, main-Läufe)", ""]
    for r in main_runs:
        title = CLASSES.get(r, {}).get("title", "")
        out.append(f"### {r} — {title}")
        out.append("| Kandidat | Status | recall | precision | orderLCS | len× | Tab(pipe+html) | h1-4 | Umlaute | Loop | s | USD |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c in candidates:
            e = field.get((c, r))
            if not e:
                continue
            m, s = e["metrics"], e["struct"]
            if "error" in m:
                out.append(f"| {c} | ✗ {m['error']['type']} | | | | | | | | | {m['runtime_s']} | |")
                continue
            if not s:
                out.append(f"| {c} | ✓ (ohne struct) | | | | | | | | | {m['runtime_s']} | {fmt(m.get('cost_usd'), '.3f')} |")
                continue
            h = s["headings"]
            out.append(
                f"| {c} | ✓ | {fmt(s.get('word_recall'))} | {fmt(s.get('word_precision'))} | "
                f"{fmt(s.get('order_lcs'))} | {fmt(s.get('length_ratio'), '.2f')} | "
                f"{s['pipe_tables']}+{s['html_tables']} | "
                f"{h['h1']}/{h['h2']}/{h['h3']}/{h['h4']} | {s['umlauts_out']} | "
                f"{'⚠' if s['loop_flag'] else '–'} | {m['runtime_s']} | {fmt(m.get('cost_usd'), '.3f')} |")
        out.append("")

    # 3b. Judge-Verdikte (Metrik c)
    judge_dir = RESULTS / "_judge"
    verdicts = sorted(judge_dir.glob("*/verdict.json")) if judge_dir.exists() else []
    verdicts += sorted(judge_dir.glob("*/verdict_gpu.json")) if judge_dir.exists() else []
    if verdicts:
        out += ["## Judge-Rankings (Metrik c)", ""]
        for v in verdicts:
            data = load(v)
            if not data:
                continue
            ranking = " > ".join(r["kandidat"] for r in
                                 sorted(data.get("ranking", []), key=lambda r: r.get("platz", 99)))
            tag = " (GPU-Feld)" if v.name == "verdict_gpu.json" else ""
            out.append(f"- **{v.parent.name}{tag}**: {ranking}")
        out.append("")

    # 4. Kosten
    led = load(RESULTS / "LEDGER.json")
    if led:
        total = sum(e["cost_usd"] for e in led["entries"])
        by_cand = {}
        for e in led["entries"]:
            by_cand[e["candidate"]] = by_cand.get(e["candidate"], 0) + e["cost_usd"]
        out += ["## Kosten (LEDGER)", "",
                f"Summe: **{total:.3f} USD** von {led['cap_usd']} USD Deckel", ""]
        for c, v in sorted(by_cand.items()):
            out.append(f"- {c}: {v:.3f} USD")
        out.append("")

    (RESULTS / "SUMMARY.md").write_text("\n".join(out), encoding="utf-8")
    print(f"OK: {RESULTS / 'SUMMARY.md'} ({len(rows)} Läufe, {len(candidates)} Feld-Kandidaten)")


if __name__ == "__main__":
    main()
