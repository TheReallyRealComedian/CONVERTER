# SPRINT DOC-BAKE — der Bake-off entscheidet die Engine-Frage

**Größe**: L (4 Phasen) · **Datum**: 2026-08-07 · **Vorhaben**: DOC-SVC

## Warum

CONVERTERs PDF-Pfad sind **1.418 Zeilen Eigenbau** aus fünf Tabellen-Detektoren mit Konsens-Clustering — und sie wurden **nie gegen irgendetwas gehalten**. Der Cowork-Report hat das als die zentrale ungemessene Behauptung des ganzen Vorhabens benannt; das Befund-Register hat drei seiner folgenreichsten Negativurteile widerlegt und damit auch die vermeintlich einfachen Antworten kassiert.

Jetzt ist alles da, was zum Messen fehlte: **14 Korpus-Klassen** mit begründeten Belegexemplaren, **drei Gold-Fassungen**, **Bewertungsregeln**, und eine **freie GPU**.

Dieser Sprint misst. Er entscheidet nicht — das Entscheidungs-Doc schreibt der Master aus den Zahlen.

## Gegroundeter Ist-Zustand (Master — nicht neu herleiten)

**Korpus**: `corpus/`, 14 Klassen, `python3 corpus/pruefen.py` meldet vollständig. Gold-Fassungen in `corpus/gold/` (`01.md` Paper zweispaltig + Tabelle · `07.md` Formular-Scan · `08.md` DOCX-Abschnitt). **Die Bewertungsregeln in `corpus/gold/_UNSICHERHEITEN.md` sind verbindlich** — drei Stellen kodieren eine Wahl, keine Wahrheit, und dort gilt Gleichwertigkeit statt Zeichengleichheit. Lies den Abschnitt „Bewertungsregeln für den Bake-off" vollständig, bevor du eine Metrik baust.

**Mintbox, am 2026-08-07 verifiziert**: RTX A2000 12 GB (15 MiB belegt), `nvidia-container-toolkit` **1.19.1**, `docker run --rm --gpus all` liefert die GPU — **Passthrough funktioniert bereits**, nichts einzurichten. 2,3 TB frei. CONVERTERs eigener Compose-Stack hat **keine** GPU-Konfiguration und bekommt auch keine: der Bake-off fährt eigene Container.

**Befund-Register** (`docs/doc_convert_register_2026-07-30.md`) — was daraus für die Kandidatenwahl folgt:
- **docling ist NICHT ausgeschlossen.** Die 50,3 % stammen aus dem Harness des Konkurrenten, mit Default-Konfiguration ohne Formel-Enrichment. Im relevanten Korridor steht es bei 64,0. Der Deutsch-OCR-Bug ist **seit 16.06.2026 behoben** — Mindestversion **v2.109.0**, und die OCR-Engine muss **explizit** gesetzt werden (`--ocr-engine rapidocr --ocr-lang de` oder `tesseract`/`deu`), **nie** `auto`.
- **MinerU ist lizenzrechtlich unkritisch** (Code Apache-basiert, Gewichte des Standardmodells Apache-2.0).
- **unstructured bleibt** — die „Herabstufung" war ein zwei Jahre alter Marketingtext über einen Codepfad, den CONVERTER nicht fährt. Aber: Pin ist **0.18.32**, aktuell **0.24.1**.
- **Gemini**: `gemini-2.0-flash` ist tot, Nachfolger `gemini-3.6-flash`. Nativer PDF-Input bis 50 MB / 1.000 Seiten. `media_resolution` gemessen: **LOW 266 · MEDIUM 532 · HIGH 1092** Bildtokens/Seite, der API-Default liegt auf HIGH.
- **AGPL ist kein Auswahlkriterium mehr** (camelot zieht seit v1.0.0 kein Ghostscript mehr nach; PyMuPDF bleibt zu prüfen, aber für einen LAN-Dienst ohne Distribution praktisch gegenstandslos).

**Zielfunktion, unverändert**: **1. Treue zum Original, 2. Formatbreite.** Geschwindigkeit und Betriebs-Einfachheit ausdrücklich nachrangig.

## Gesperrte Entscheidungen

- **Der Eigenbau ist ein Kandidat wie jeder andere.** Ohne ihn im Feld bleiben die 1.418 Zeilen unbewertbar, und genau das ist der Anlass des Sprints.
- **Kein betriebliches Material.** Der Bake-off läuft ausschließlich auf `corpus/`. Der `intern/`-Satz liegt bewusst außerhalb des Repos und wird **nicht** angefasst — schon gar nicht mit Cloud-Kandidaten.
- **Kostendeckel**: harte Obergrenze für Modell-Ausgaben, im Harness durchgesetzt, nicht per Disziplin. Vorschlag 20 €; wenn du bei der Kalibrierung merkst, dass das nicht reicht, **stoppen und berichten** statt aufdrehen.
- **Nicht TEDS allein als Tabellenmaß.** Es korreliert nur mit r=0,684 zum menschlichen Urteil (LLM-Judge: 0,93). Wo du automatisch misst, miss mehrdimensional und lass eine Stichprobe von Hand nachprüfen.

---

# Phase 1 — Das Harness

## 1.1 Was es tut

Ein Kandidat rein, pro Korpus-Dokument raus: **Markdown**, **Kennzahlen** (Laufzeit, Speicher-Spitze, Modell-Calls, Tokens, geschätzte Kosten), **Fehler/Degradationen**. Ergebnisse landen dateibasiert, damit ein abgebrochener Lauf nicht alles verwirft und ein Kandidat nachträglich ergänzt werden kann.

**Ein Kandidat ist ein kleines Adapter-Objekt** mit einer Methode „nimm diese Datei, gib Markdown und Kennzahlen zurück". Neue Kandidaten hinzuzufügen muss billig sein — davon hängt ab, ob Phase 2 und 3 überhaupt breit werden.

Wo das Harness lebt, ist deine Wahl (`corpus/bakeoff/` oder `scripts/` — begründe kurz). Es ist **kein** Teil der Flask-App und wird nicht importiert.

## 1.2 Die Metrik

Drei Ebenen, weil nur drei von vierzehn Klassen eine Gold-Fassung haben:

**(a) Gegen Gold** (`01.md`, `07.md`, `08.md`): der schärfste Vergleich. ⚠️ **Die Bewertungsregeln aus `_UNSICHERHEITEN.md` sind Teil der Metrik**, nicht eine Fußnote — verbundene Kopfzeile gilt als gleichwertig zur Überschrift, jede Notation die Tief-/Hochstellung *erhält* gilt als gleichwertig zu LaTeX, jede Bild-Syntax mit beliebigem Ziel zählt. Und die Liste der Quell-Eigenheiten, die ein Werkzeug **nicht** reparieren darf, ist ein **Prüfkriterium**: wer `Markteilnehmer` zu `Marktteilnehmer` „korrigiert", hat einen Fehler gemacht.

**(b) Strukturell, für alle 14**: messbare Eigenschaften ohne Gold — kommt eine Tabelle als Tabelle heraus (Pipe-Zeilen oder HTML), erscheinen Überschriften, überlebt der Textbestand gegen eine deterministische Referenz (Textebene, wo es eine gibt), bleibt die Lesereihenfolge plausibel. **Was du hier misst, definierst du — begründe jede Kennzahl in einem Satz.**

**(c) Urteil**: für die Klassen ohne Gold eine strukturierte Bewertung pro Dokument. Ein LLM-Judge ist dafür belegt besser als TEDS; **nimm eine Stichprobe selbst in Augenschein** und sag im Bericht, wo Judge und eigener Eindruck auseinanderliefen.

## 1.3 Kalibrierung

Fahre **zwei** Kandidaten gegen **drei** Dokumente, bevor du das Feld aufmachst: einen, der sicher funktioniert (der Eigenbau auf einem nativen PDF), und einen, der sicher scheitert (irgendetwas auf dem degradierten Scan). Das prüft das Harness, nicht die Kandidaten.

## Stop
Harness läuft, Kalibrierung belegt, `pytest tests/` unverändert grün (Baseline **861** — das Harness gehört nicht in die Suite, aber es darf sie auch nicht brechen). **Commit + Push** `feat(DOC-BAKE): Bake-off-Harness (P1)`. Dann warten.

---

# Phase 2 — Das CPU- und Cloud-Feld

Alles, was ohne GPU läuft. **Diese Phase muss für sich allein ein verwertbares Ergebnis liefern** — falls die GPU wieder belegt ist, bevor Phase 3 läuft.

Kandidaten, mindestens:

| Kandidat | Wofür | Randbedingung |
|---|---|---|
| **CONVERTERs Eigenbau** | PDF | der Anlass des Sprints |
| **Gemini nativ-PDF** | PDF, Scans | `gemini-3.6-flash`; **alle drei `media_resolution`-Stufen** auf mindestens einem Dokument, dann die beste weiterfahren |
| **docling CPU** | PDF, DOCX, PPTX, XLSX | ⚠️ **≥ v2.109.0**, OCR-Engine **explizit**, `--ocr-lang de` |
| **unstructured** | Office | beide Versionen: der Pin **0.18.32** gegen aktuell **0.24.1** — die Drift ist ein eigenes Backlog-Item, hier fällt die Messung nebenbei ab |
| **pandoc** | DOCX | trägt Fußnoten, die docling verwirft |
| **markitdown** | PPTX, XLSX | |
| **trafilatura** | HTML | |
| **Tesseract 5 `deu`** | Scans | die CPU-Referenz; `corpus/BASELINE-OCR.md` hat den Nullpunkt schon |

**Kosten mitzählen, nicht schätzen.** Gemini liefert `usage_metadata`; nimm die echten Zahlen. Wenn der Deckel greift, bricht der Lauf sauber ab und meldet, was fehlt.

## Stop
Ergebnistabelle über alle 14 Klassen für das CPU/Cloud-Feld, Kosten beziffert. **Commit + Push** `feat(DOC-BAKE): CPU- und Cloud-Feld gemessen (P2)`. Dann warten.

---

# Phase 3 — Das GPU-Feld

Eigene Container, CONVERTERs Stack bleibt unberührt. Kandidaten: **MinerU 3.x** (VLM-Backend), **marker v2**, und **ein lokales Dokument-VLM** über vLLM — `PaddleOCR-VL` oder `dots.ocr`, beide ≤3B und damit sicher in 12 GB.

⚠️ **Zwei Dinge sind hier zu messen, die kein Benchmark liefert:**
1. **Durchsatz auf der A2000.** Für diese Karte existiert keine einzige publizierte Messung eines Dokument-VLM-Stacks. Seiten/Minute bei realer Auslastung.
2. **Deutsche Qualität.** Für `dots.ocr`, `PaddleOCR-VL` und MinerU-VLM gibt es **keinen** publizierten Deutsch-Score. Der Korpus ist die einzige Quelle.

⚠️ **Wiederholungs-/Halluzinations-Loops sind quantifiziert zweistellig** möglich, und der dokumentierte Auslöser sind durchgehende Punkt-/Unterstrichlinien — also **genau `06_scan-degradiert` und `07_formular-punktlinien`**. Miss die Loop-Rate, statt sie zu übersehen: wiederholte n-Gramme, absurde Längen, Zeichensatz-Wechsel.

## Stop
GPU-Feld gemessen, Durchsatz und Loop-Raten beziffert. **Commit + Push** `feat(DOC-BAKE): GPU-Feld gemessen (P3)`. Dann warten.

---

# Phase 4 — Auswertung und Wrap

- **Ergebnis-Doc** `docs/doc_convert_bakeoff_<datum>.md`: Tabelle Kandidat × Klasse, Kosten, Durchsatz, und **pro Format eine Reihung mit dem Handel im Klartext**. Kein Entscheidungs-Doc — das schreibt der Master.
- **Die eine Frage explizit beantworten**: schlägt der 1.418-Zeilen-Eigenbau eine Standardpipeline, und wenn ja, wo genau? Ein Satz Antwort, darunter die Zahlen.
- **Was nicht messbar war**, mit Grund — genauso wichtig wie die Ergebnisse.
- **STATUS.md** + **BACKLOG.md** (Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md`).
- **Memory**, falls sich eine übertragbare Lehre zeigt. Nach dem Schreiben mit `ls` prüfen, dass Datei und Index-Zeile zusammenpassen.

## Nicht-Ziele

- **Keine Engine wird eingebaut.** Dieser Sprint misst; der Umbau ist ein eigener.
- **Keine API-Fläche**, kein Endpunkt, kein Token, kein Job-Modell.
- **Kein Anfassen von `services/pdf_extraction/`** — der Eigenbau wird als Kandidat *aufgerufen*, nicht verändert.
- **Kein** betriebliches Material, **kein** Zugriff auf `intern/`.
- **Kein** Deploy, **keine** Änderung an CONVERTERs Compose-Stack.
