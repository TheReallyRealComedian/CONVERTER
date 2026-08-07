# Kalibrierung P1 (2026-08-07) — das Harness ist geprüft, nicht die Kandidaten

Zwei Kandidaten × drei Dokumente + Metrik-Selbsttest. Sprint-Vorgabe: einer,
der sicher funktioniert (Eigenbau auf nativem PDF), einer, der sicher
scheitert (Textebene auf dem degradierten Scan).

## Matrix

| Kandidat | Dok | Ergebnis | Laufzeit | RSS | Gemini | Kosten |
|---|---|---|---|---|---|---|
| textlayer | 01 nativ | OK, recall 1,0 | 0,2 s | 51 MB | – | 0 |
| textlayer | 03 nativ | OK, recall 1,0 | 0,1 s | 48 MB | – | 0 |
| textlayer | 06 Scan | **FEHLER, sauber verbucht** (metrics.json mit `error`, Exit 1) | 0,03 s | 47 MB | – | 0 |
| eigenbau | 01 nativ | OK, recall 0,9992 | 5,6 s | 246 MB | 0 Calls | 0 |
| eigenbau | 03 nativ | OK, recall 0,9002 | 21,6 s | 245 MB | 0 Calls | 0 |
| eigenbau | 06 Scan | OK via Gemini-Vision | 62,7 s | 273 MB | 2 Calls, echte `usage_metadata`-Tokens | 0,0404 USD im LEDGER |

Budget-Pfad belegt: `precheck()` lief vor jedem Call, LEDGER.json kumuliert
(0,04 von 22 USD Deckel). Preise noch `verified: false` — Verifikation ist
P2-Vorbedingung vor den breiten Läufen.

## Gold-Metrik end-to-end

* **Selbsttest Gold-gegen-Gold (alle drei Dateien): perfekt** — f1 = 1,0,
  CER = 0, Zellen R/P = 1,0/1,0, Regel 1 „3/3", Regel 2 „11/11 erhalten",
  Regel 3 alle vier Checks wahr, Formular-Token 43/43 + 16/16,
  colspan/rowspan 5+2 = Gold. (Zwei dabei gefundene Metrik-Bugs — Unicode-
  Exponenten ⁴/⁵, TABLE-I-15-False-Positive — gefixt; genau dafür ist der
  Selbsttest da.)
* **Negativ-Tests bestanden**: eingeebnete Notation (`ellrand`, `Pout`,
  `10-4`) → „eingeebnet"; reparierte Quell-Eigenheiten (`Marktteilnehmer`,
  `19. Auflage`, `S. 189.`) → „REPARIERT (Fehler)"; `<th colspan="5">` →
  „colspan_kopf" (gleichwertig per Regel 1).
* Reale Läufe gegen Gold 01 (Seiten 4+8 als abgeleitetes PDF): textlayer
  f1 0,9149 / CER 6,45 % / Zellen 0 — der Nullpunkt (Ligaturen, kaputte
  Akzente, Trennstriche, keine Tabellenstruktur).

## Früher Sachbefund (nur notiert, P2 misst)

Auf der Tabellenseite des Papers (dvips-Pixellinien, `get_drawings()` = 0)
findet **keiner der fünf Eigenbau-Detektoren** etwas: der Eigenbau-Output der
Gold-Seiten ist score-identisch zur rohen Textebene (f1 0,9149, Zellen 0,
0 Gemini-Calls). Auf 03 verliert der Eigenbau zudem ~10 % des Wortbestands
gegen die Textebene (recall 0,90) — die Tabellenextraktion frisst Fließtext.
Auf 06 (degradierter Blanko-Scan) liefert Gemini 6 Pipe-Tabellen, 60
Kästchen-Token, 95 Umlaute, **kein** Loop (max_ngram_repeat 4, Schwelle 5).
