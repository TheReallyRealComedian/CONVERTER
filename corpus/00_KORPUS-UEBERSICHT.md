# Benchmark-Korpus Converter — Übersicht

Zusammengestellt am 31.07.2026 aus der Nextcloud. Alle Dateien sind **Kopien**, die
Originale liegen unverändert an ihrem Platz.

## Wie ausgewählt wurde

Nicht über Dateinamen geraten, sondern gemessen:

- **3.645 PDFs** strukturell analysiert (PyMuPDF): Spaltenlayout über Textblock-Verteilung,
  Tabellenerkennung mit Zeilen-/Spaltenzahl, Tabellenfortsetzung über Seitengrenzen,
  Punkt-/Unterstrichläufe, Formelzeichen, Bildflächendeckung pro Seite, Sprache, Seitenanzahl
- **904 Scan-PDFs** zusätzlich auf Bildqualität vermessen: effektive DPI, Laplace-Varianz
  (Schärfe), Rauschen auf Papierflächen, Schieflage über Projektionsprofil, Histogramm-Bimodalität
- **1.738 Office-/Web-Dateien** (DOCX, PPTX, EML, HTML) über ihr XML ausgewertet: echte
  Fußnotenreferenzen, Änderungsverfolgung, Überschriftenebenen, verbundene Zellen, SmartArt-Teile,
  Speaker-Notes, mehrspaltige Textkörper, Boilerplate-Anteil
- **14 .msg-Dateien** auf deutsche Zitatketten, Signaturen und Anhänge geprüft
- Die engere Auswahl wurde **visuell gesichtet** (Seiten gerendert und angesehen), weil
  Metriken bei Tabellen und Scans regelmäßig lügen

Jede Zahl in den READMEs stammt aus dieser Messung.

## A — Vergleichskorpus (cloud-fähig)

| # | Ordner | Datei | Umfang | Status |
|---|---|---|---|---|
| 1 | `01_paper-zweispaltig` | Albert-Barabási, *Statistical Mechanics of Complex Networks* | 54 S., EN | ✅ |
| 2 | `02_guideline` | AGOF *internet facts 2010-II*, Berichtsband | 57 S., DE | ⚠️ Stellvertreter |
| 3 | `03_tabelle-seitengrenze` | AGOF Zuordnung Angebot→Vermarkter | 20 S., DE | ✅ |
| 4 | `04_verbundene-zellen` | AGOF Ranking Angebote/Monat | 12 S., DE | ✅ |
| 5 | `05_scan-sauber` | Dahlhaus, *E.T.A. Hoffmanns Beethoven-Kritik*, **gerastert** | 15 S., DE, 300 dpi bilevel | ✅ |
| 6 | `06_scan-degradiert` | — | — | ❌ **nicht vorhanden** |
| 7 | `07_formular-punktlinien` | AOK-PLUS Fragebogen Familienversicherung (blanko) | 2 S., DE | ⚠️ nativ statt Scan |
| 8 | `08_docx-fussnoten` | Leitfaden Businessplan 1.1 | DOCX, DE | ⚠️ ohne Änderungsverfolgung |
| 9 | `09_pptx-smartart` | zwei Decks (mehrspaltig+Notes / SmartArt) | 25 + 98 Folien | ⚠️ auf zwei Dateien verteilt |
| 10 | `10_html-artikel` | SPIEGEL ONLINE, *Korruptes Web 2.0* | 84 KB HTML, DE | ⚠️ ohne Cookie-Banner |
| 11 | `11_eml-zitatkette` | Polar-Care-Korrespondenz | 6.123 Zeichen, DE | ⚠️ ohne Anhang |
| 12 | `12_grosses-pdf` | Enquete-Kommission, BT-Drs. 14/9020 | 280 S., DE | ⚠️ 280 statt 150–250 |
| 13 | `13_mischdokument` | Bock, Telekom Deutschland (Konferenzvortrag) | 32 S., DE | ✅ |
| 14 | `14_ocr-ebene-kaputt` | dasselbe Dahlhaus-Original, **ungerastert** | 15 S., DE, 600 dpi | ✅ neu |

6 Klassen sauber getroffen, 7 mit benannter Abweichung, 1 Lücke (#06). Jede Abweichung steht
in der README des jeweiligen Ordners; die Sammelbegründung und die konkreten Vorschläge zum
Schließen stehen in **`LUECKEN.md`**.

### Nachtrag 2026-07-31 — warum aus #05 zwei Klassen wurden

Das Original trug eine **OCR-Textebene**, und zwar eine mit englischem Modell erzeugte
(`Asthetik` statt `Ästhetik`). Bei Textdichte 8,20 gegen den Schwellwert 0,5 hätte
CONVERTERs Klassifikator die Seiten als **NATIVE** eingestuft — der OCR-Pfad wäre nie
betreten worden, und die kaputte Ebene wäre kommentarlos durchgereicht worden.

Damit war die Lage: von den drei Dokumenten für den Scan-/OCR-Pfad (#05, #06, #07) hat
**keines** ihn ausgelöst — #06 fehlt, #05 und #07 gelten als nativ. Nur #13 betritt ihn
(19 von 32 Seiten ohne Textebene).

Deshalb liegt in **#05** jetzt eine textebenen-freie Fassung (300 dpi bilevel, aus dem
600-dpi-Original gerendert; Textebene 0 Zeichen, Bildabdeckung 1,00 → **SCANNED**), und das
Original bildet als **#14** eine eigene Klasse: *erkennt der Konverter, dass eine vorhandene
Textebene unbrauchbar ist?* Kein Kandidat des Bestands beantwortet das, und der Fall ist in
Firmenablagen die Regel.

Gegengemessen am selben Bild: `tesseract -l deu` → `Ästhetik` ✅, `-l eng` → `Asthetik` ✗.
Der Raster ist also gut genug; wer hier scheitert, scheitert am Sprachmodell.

## B — Betrieblicher Satz

**Liegt bewusst NICHT in diesem Repo**, sondern unverändert unter
`~/Nextcloud/00 Inbox/Benchmarkfiles/intern/` — vier Dateien (CMC-Dokument,
Gremienprotokoll, Foliensatz, Mailkette mit Anhang). Verlassen die Maschine nicht, ranken
nichts, validieren nur den lokalen Pfad. Details in der dortigen `README.md`.

Nicht bloß gitignored, sondern gar nicht erst hier: ein `git add -A` im falschen Moment
reicht sonst.

## Ablage und Versionierung

Der cloud-fähige Satz liegt unter `corpus/` im CONVERTER-Repo. **Versioniert werden nur die
Markdown-Dateien** — die Quelldateien sind ~60 MB (davon 51 MB ein PPTX) und haben in einer
History nichts zu suchen, die bei jedem Mintbox-Deploy gezogen wird. Begründung und Regel
stehen in `.gitignore`.

Die Originale in der Nextcloud sind unberührt; dieser Ordner ist eine Kopie.

## Sprachverteilung

11 von 14 Dateien sind deutsch. Das entspricht deinem Arbeitsmaterial, ist aber ein
bewusster Schnitt: Werkzeuge, die englischen wissenschaftlichen Satz gut können, fallen bei
deutschen Behördenvordrucken und Umlauten oft deutlich ab — der Korpus misst also genau die
Seite, auf der du arbeitest.

## OCR-Baseline

`tesseract-lang` ist installiert (163 Sprachen, `deu` vorhanden). Der Nullpunkt-Lauf über
die Scan-Klassen steht in **`BASELINE-OCR.md`** — inklusive des Befunds, dass der
Sprachschalter `eng`→`deu` nicht ein paar Konfidenzpunkte kostet, sondern **jeden Umlaut**,
und zwar bei fast unveränderter gemeldeter Konfidenz.

Daraus folgt für den Bake-off: Wortkonfidenz taugt nicht als Qualitätsmaß, es braucht ein
Zeichenfehler-Maß (CER) gegen die Gold-Fassungen in `gold/`.
