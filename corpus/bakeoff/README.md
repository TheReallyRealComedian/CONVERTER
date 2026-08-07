# Bake-off-Harness (DOC-BAKE)

Ein Kandidat rein, pro Korpus-Dokument raus: **Markdown**, **Kennzahlen**
(Laufzeit, Speicher-Spitze, Modell-Calls, Tokens, Kosten), **Fehler/
Degradationen**. Dateibasiert — ein abgebrochener Lauf verwirft nichts, ein
Kandidat kann nachträglich ergänzt werden.

**Ort**: `corpus/bakeoff/`, weil das Harness Korpus-Werkzeug ist wie
`pruefen.py` (liest `corpus/`, schreibt daneben) — kein App-Wartungsskript
wie `scripts/*` und **kein** Teil der Flask-App; nichts hier wird von ihr
importiert.

## Aufbau

```
bakeoff/
  harness/
    manifest.py      Klassen → Eingabedateien (+ Gold-Zuordnung)
    adapters.py      Kandidaten: (datei, ctx) → Markdown + Kennzahlen
    run_one.py       ein Kandidat × ein Dokument → results/…/output.md + metrics.json
    budget.py        Kostendeckel (20 €), im Harness durchgesetzt (precheck vor jedem Call)
    normalize.py     dokumentierte Gleichwertigkeits-Normalisierung (Fill-Linien, ☐, \_, Striche)
    refs.py          baut Gold-Seiten-PDFs (derived/) + Referenztexte (results/_references/)
    score_gold.py    Metrik (a): gegen Gold 01/07/08 — Bewertungsregeln 1–3 sind Teil der Metrik
    score_struct.py  Metrik (b): strukturelle Kennzahlen für alle 14 Klassen
    judge_rubric.md  Metrik (c): LLM-Judge-Rubrik für Klassen ohne Gold (P2/P3)
  envs/              venv-Builder pro Kandidaten-Umgebung (venvs selbst unversioniert)
  derived/           abgeleitete Gold-Inputs (PDF-Seiten) — unversioniert, via refs.py
  results/           <kandidat>/<klasse>[.gold]/{output.md, metrics.json, struct.json, gold_scores.json}
                     + LEDGER.json (Kosten) + _references/ (unversioniert)
```

Versioniert: Code, Rubrik, `metrics.json`/`struct.json`/`gold_scores.json`/
`LEDGER.json` (die Evidenz). Unversioniert: Roh-Outputs (`output.md`),
venvs, `derived/`, `_references/` — alles reproduzierbar.

## Benutzen

```bash
cd corpus/bakeoff
bash envs/build_eigenbau.sh                 # einmalig
envs/eigenbau/bin/python harness/refs.py    # einmalig pro Korpus-Stand

# Ein Lauf:
envs/eigenbau/bin/python harness/run_one.py --candidate eigenbau --class 01
envs/eigenbau/bin/python harness/run_one.py --candidate eigenbau --class 01 --role gold

# Scoring (stdlib; CER exakt, wenn rapidfuzz im Env):
envs/eigenbau/bin/python harness/score_struct.py --candidate eigenbau
envs/eigenbau/bin/python harness/score_gold.py --candidate eigenbau
```

Neuer Kandidat = ein Eintrag + eine Funktion in `adapters.py` (schwere
Imports in der Funktion), ggf. ein `envs/build_<name>.sh`. Mehr nicht.

## Die drei Messebenen

**(a) Gegen Gold** (01/07/08): Wort-Multiset bidirektional (dieselbe Methode
wie die Gold-Verifikation) + CER + Zell-Multiset. Die **Bewertungsregeln aus
`../gold/_UNSICHERHEITEN.md` sind implementiert, nicht Fußnote**: Fettbalken
als Überschrift ODER `<th colspan>` (R1), jede stellungs-erhaltende Notation
(R2), jede Bild-Syntax mit Fußnote dran (R3). Die Quell-Eigenheiten
(`Markteilnehmer`, `19.Auflage`, `S.189.`, Nr. 14/14/16, `6.36 - 6.18`) sind
Prüfkriterien: „repariert" ist ein Fehler. 01/07 laufen gegen abgeleitete
Nur-Gold-Seiten-PDFs; 08 wird anker-basiert aus dem Volltext geschnitten.

**(b) Strukturell, alle 14**: jede Kennzahl im Modul-Docstring von
`score_struct.py` in einem Satz begründet (Tabellen-als-Tabellen, Recall/
Precision/Ordnung gegen deterministische Referenz, Umlaut-Überleben,
Loop-Erkennung, Längenverhältnis, colspan/rowspan, Fußnoten, Formular-Token).

**(c) Urteil**: `judge_rubric.md` — strukturierte 5-Achsen-Bewertung für die
11 Klassen ohne Gold, mit Pflicht-Stichprobe von Hand und offengelegter
Judge-Modellfamilie.

## Kostendeckel

`budget.py`: 20 € hart, `precheck()` **vor** jedem Modell-Call, kumulativ
über `results/LEDGER.json`. Token-Zählung exakt aus `usage_metadata`;
Preise bis zur Verifikation (P2) konservative obere Schranken
(`verified: false` steht an jedem Eintrag).
