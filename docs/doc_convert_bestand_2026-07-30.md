# Bestandsaufnahme — fünf Dokument-Konvertierungen in vier Repos

**Datum**: 2026-07-30 · **Methode**: 7 parallele Leser-Agenten über CONVERTER, Muncher, Agentsuite2, image_extracter, CLeak, search_nextcloud.py — jeder Befund gegen Quelltext belegt, 216 Tool-Calls · **Zweck**: Grundlage für den Cowork-Recherche-Auftrag ([doc_convert_research_brief_2026-07-30.md](doc_convert_research_brief_2026-07-30.md)) und die Verwertungs-Kette ([doc_convert_verwertung_2026-07-30.md](doc_convert_verwertung_2026-07-30.md)).

> **Lesehinweis**: Dies ist ein Ist-Zustands-Dokument, keine Roadmap. Jede Aussage hier ist am Code verifiziert. Wo etwas *nicht* entscheidbar war, steht das ausdrücklich — die Trennung zwischen „gemessen" und „ungemessen" ist der eigentliche Wert dieser Aufnahme.

---

## Der Ein-Satz-Befund

Fünf Implementierungen, und **jede kann genau eine Sache gut, die alle anderen schlecht können** — die Stärken sind komplementär verteilt und nirgends kombiniert.

| Implementierung | LOC | kann als Einzige gut | ist gleichzeitig |
|---|---:|---|---|
| **CONVERTER** doc→md | 1.510 | Zugangskontrolle (eigener Token je Billing-Fläche) | 0 Tests für 1.418 LOC Extraktion |
| **CONVERTER** md→out | 651 | Mathe-Rendering (vier Wege für dieselbe Zwischenform) | erkennt Mathe in der Eingabe nirgends |
| **Muncher** | 3.975 | Betriebsmechanik (SAQ, Watchdog, Heartbeat, Circuit Breaker, Dead-Letter) | kaputteste Testsuite, zerstörerischstes Post-Processing |
| **Agentsuite2** | 2.860 | Folien-Layout-Semantik (geometrisches Spalten-Clustering) | null Auth, CORS `*`, bis zu 100 VLM-Calls pro anonymem Request |
| **image_extracter** | 23.227 | echtes OCR (PaddleOCR 3.x + PPStructureV3) | Konsens *senkt* die Konfidenz, Halluzinationsfall geht ohne Review durch |

---

## 1. Was jede Implementierung tut

### CONVERTER — Dokument → Markdown

Eine harte Gabelung an der Dateiendung ([documents.py:58](../app_pkg/documents.py:58)) trennt zwei Welten, die nichts voneinander wissen.

**PDF-Pfad** (`services/pdf_extraction/`, 1.418 LOC): fünf Tabellen-Detektoren in drei Eskalationsstufen — PyMuPDF `find_tables`, pdfplumber, camelot `lattice`, camelot `stream`, img2table, dazu eine pdfminer-Heuristik. Detektionen werden per Bbox-IoU ≥ 0,3 geclustert, ein Cluster überlebt ab zwei unterschiedlichen Detektoren. Danach ein Extraktor-Wettbewerb mit Scoring (Spaltenzahl, Zeilenplausibilität, Zellfüllgrad). Gemini Vision nur als letzter Ausweg, mit Rückvalidierung gegen PyMuPDF. Plus seitenübergreifender Tabellen-Merge über vier Signale. Das ist ernsthafte Ingenieursarbeit.

**Alles andere** ([documents.py:65-66](../app_pkg/documents.py:65)):

```python
elements = partition(filename=temp_file_path, strategy="fast")
output_markdown = "\n\n".join([el.text for el in elements])
```

Zwei Zeilen. `unstructured` liefert Element-Kategorien (`Title`, `ListItem`, `Table`) und `metadata.text_as_html` für Tabellen — beides wird in derselben Zeile weggeworfen. `text_as_html` kommt im gesamten Repo **kein einziges Mal** vor.

Was der PDF-Pfad trotz seines Aufwands **gar nicht** behandelt: Überschriften auf tabellenlosen Seiten (`get_text('text')`, jede Zeile gestrippt), mehrspaltiges Layout (keine Spaltenerkennung, Sortierung stumpf nach y), Mathematik, Bilder (nur vermessen, nie extrahiert), OCR (tesseract ist im Image, `img2table` wird hart mit `use_ocr=False` gerufen), Dokument-Metadaten.

### Muncher — Bulk-Wortgetreue

3.975 LOC, FastAPI auf Port 8002, **keine Authentifizierung**. Zweck ist die wortgetreue Überführung betrieblicher Biopharma-Dokumente (PDF/PPTX) — Treue, nicht Lesbarkeit. Nutzt **docling** und **markitdown**, also die beiden modernsten Allzweck-Konverter, dazu einen dreistufigen VLM-Pass über Gemini.

Die Betriebsmechanik ist die reifste im ganzen Bestand: SAQ auf Postgres, Retries auf Datei- und Seitenebene, Watchdog mit Requeue-Deckel, Heartbeats, Circuit Breaker, Dead-Letter-Endpoints. **Das ist die einzige Komponente, die man unverändert übernehmen kann.**

Das Post-Processing steht in scharfem Gegensatz dazu:

- `_ENCODING_MAP` enthält `" 3 " → " – "`. `_fix_encoding('Phase 3 study with 3 sites')` liefert `'Phase – study with – sites'`. In dem einen Repo, dessen erklärtes Ziel Wortgetreue ist.
- Eine Zeile darunter steht `_DASH_PATTERN`, das genau den Fall beheben soll, den die Map-Ersetzung vorher schon überschrieben hat. Der dokumentierte Fix wird nie ausgeführt.
- `re.sub(r'equipment(\w)', r'equipment \1', …)` macht aus `'equipments'` → `'equipment s'`.
- `_GARBAGE_PATTERNS` ersetzt eine Unicode-Sequenz hart durch `'≈ 2,100 m²'` — den Inhalt genau eines Testdokuments.
- `deduplicate_content` löscht Blöcke ab 80 % Token-Overlap; zwei Absätze, die sich in einem Wort unterscheiden, verlieren einen.
- Der Cleanup behauptet im Docstring, idempotent zu sein, ist es nicht, und **läuft zweimal** — beim Emit und beim Ausliefern. Der Downloader bekommt systematisch anderen Inhalt als die Datenbank hält.

Der VLM-Trigger ist `confidence < 0.85`, und das Maximum der Confidence-Funktion **ist** 0,85. Das Kosten-Gate schließt per Konstruktion fast nie — was die im Repo dokumentierte Text-Genauigkeits-Regression von −12,2 erklärt.

### Agentsuite2 — zwei Pfade, die sich nicht kennen

Unternehmenskontext (Boehringer Ingelheim), LLM-Zugang über das interne Apollo-Gateway. **Zwei Konverter im selben Repo, die nichts voneinander wissen:**

**pptx-parser**: python-pptx-Shape-Tree mit EMU-Koordinaten, Connector-Graph, Gruppen, Speaker-Notes, core-properties. Geometrisches Spalten-Clustering bei 15 % Folienbreite, Full-Width ab 0,7, Fazit-Zone unterhalb 85 % Höhe, Heading-Level aus der Fontgröße. Für Folien der klar überlegene Ansatz im Bestand — **und das UI lehnt `.pptx` aktiv ab** mit dem Hinweis, man solle als PDF exportieren. Der gesamte Reichtum ist unerreichbar.

**cms/ingest**: `pypdf.extract_text()`, sonst nichts. Keine Format-Whitelist — alles außer PDF wird als utf-8 mit `errors='replace'` dekodiert. Ein hochgeladenes DOCX wird nicht abgelehnt, sondern als U+FFFD-Binärmüll gechunkt und eingebettet. Und: eine Extraktions-Exception **wird zum Dokumentinhalt** (`f'Error extracting text: {e}'` ist truthy, der Aufrufer prüft nur `if not content`), wird gechunkt, eingebettet und mit `{'status':'ok'}` quittiert.

### image_extracter — der einzige echte OCR-Stack

23.227 LOC, PaddleOCR 3.x mit PPStructureV3, GPU-Zweig vorhanden, Claude als Validator, 36 API-Endpoints, **keine Authentifizierung**. Zweck ist nicht lesbarer Text, sondern schema-konforme Geschäftsdaten aus Dashboard-Slides, plus eine Human-in-the-Loop-QA-Queue.

Konzeptionell der beste Umgang mit Modell-Output im Bestand — Herkunft je Wert wird persistiert, Abweichungen gehen in eine Review-Queue. Und an zwei Stellen verkehrt herum implementiert:

- **Übereinstimmung senkt die Konfidenz.** `calculate_overall_confidence` gewichtet niedrigere Werte höher: OCR 0,99 + LLM 0,85 ergibt 0,8928 — weniger als die 0,99, mit der der OCR-Wert allein durchgegangen wäre. Konsens wird bestraft.
- **Der Halluzinationsfall geht ohne Review durch.** Ein Wert, den *nur* der LLM hat, bekommt `needs_review=False`; jede bloße Abweichung zwischen OCR und LLM wandert dagegen in die Queue.

Die OCR-Sprache ist unerreichbar auf `'en'` festgenagelt, bei überwiegend deutschem Material.

---

## 2. Dieselbe Aufgabe, verschieden gelöst

Der Kern von Olis Ausgangsfrage. Elf Divergenzen, hier die entscheidenden:

**Tabellen aus PDF.** CONVERTER: 1.418 LOC Eigenbau aus fünf Detektoren. Muncher: `DocumentConverter()` ohne Optionen — docling bringt Layoutmodell, TableFormer und OCR mit, drei Zeilen. **Der teuerste Eigenbau steht neben der billigsten Bibliothekslösung, und keiner wurde je gegen den anderen gehalten.** Unentscheidbar ohne Messung — und genau das ist die Erkenntnis.

**Nicht-PDF-Formate.** CONVERTER hat über `unstructured` den mit Abstand besten Zugang und nutzt ihn am schlechtesten. Zwischen „DOCX-Tabelle als Textbrei" und „DOCX-Tabelle als HTML" liegt hier ein Einzeiler.

**Ausführungsmodell.** Muncher gewinnt eindeutig und ohne Messung. CONVERTER ist der schärfste Fall von „Infrastruktur vorhanden, nicht benutzt": synchron hinter `--workers 1`, obwohl Redis/RQ im selben Projekt für Narration läuft.

**Fehlermeldung.** Agentsuite2-Parser hat das richtige Muster — Teilerfolg ist ein 200 mit Fehlerliste, nicht ein 500. Die gemeinsame Lücke aller fünf: **kein Dienst gibt ein Degradationssignal zurück**, an dem ein Aufrufer erkennen könnte, dass er gerade OCR-Text oder Modell-Text statt Extraktion bekommt.

**Post-Processing.** Agentsuite2 gewinnt durch Unterlassung. Beide aufwendigen Post-Prozessoren zerstören nachweislich Nutzdaten — CONVERTER jeden echten Codeblock (globaler Fence-Sweep) und jedes Vorkommen eines Linktexts (globaler `str.replace`, auch in Tabellenzellen), Muncher jede freistehende Ziffer 3.

**Screenshot-Auflösung.** Vier Implementierungen, vier Werte: zoom 3,0 (≈216 dpi), zoom 4,0 (≈288 dpi), 200 dpi, 200 dpi, 300 dpi. **Keine einzige davon ist irgendwo begründet** — obwohl der Parameter direkt auf Bild-Tokens und OCR-Güte durchschlägt.

**Seitentrenner.** Drei unabhängige Implementierungen sind auf `'\n\n---\n\n'` gekommen. Der einzige Fall echter Konvergenz im Bestand — und damit die einzige Format-Entscheidung, die man ohne Diskussion übernehmen kann. Agentsuite2s Variante ist besser, weil sie die Seitennummer mitführt.

**Zugangskontrolle.** CONVERTER, ohne Vorbehalt — und die Begründung des `NARRATION_TOKEN` (eigener Token, weil der Pfad Geld pro Call kostet, unabhängig revozierbar) ist **wörtlich die Anforderung des künftigen Dienstes**. Drei von vier Diensten haben null Auth, zwei lösen unauthentifiziert bezahlte Modell-Calls aus, und in Munchers Fall stehen interne GMP- und Gremiendokumente dahinter.

---

## 3. Was in *allen* fehlt

- **Mathematik** wird in keiner der fünf Extraktionen behandelt — und ausgerechnet CONVERTER, das die Formeln verliert, betreibt auf der Ausgabeseite vier sorgfältig gebaute Renderwege für genau diese Formeln. **Die Ausgabe kann mehr als die Eingabe liefert.**
- **Kein Repo extrahiert je ein Bild.** Muncher ersetzt Abbildungen durch Textbeschreibungen, Agentsuite2 durch `[Image]`, CONVERTER lässt sie verschwinden — und wirft auf der Ausgabeseite `data:`-URI-Bilder zusätzlich aktiv weg (nh3 ohne `url_schemes`).
- **Dokument-Metadaten** werden nirgends wirksam gelesen.
- **Mehrspaltige Seiten** werden von CONVERTER und image_extracter zeilenweise verschränkt (stumpfe (y,x)-Sortierung).
- **Qualitätsverlust ist von außen unsichtbar** — überall nur im Log.
- **Langläufer laufen im Web-Prozess** (außer bei Muncher).
- **Kein Token- oder Kostenbudget je Job**, nirgends. Keine Seitenobergrenze, kein Abbruch.
- **Der Konvertierungspfad ist nirgends getestet.** CONVERTER: 0 Tests für 1.418 LOC (der Routentest mockt den Service komplett weg). Agentsuite2: kein Treffer. image_extracter: `testpaths` konfiguriert, `tests/` existiert nicht. Muncher: 5 failed.
- **Sprache ist überall Zufall statt Konfiguration**, auf vier verschiedene Weisen.
- **Konfigurationsfelder ohne Leser** in drei Repos — sie suggerieren eine Wahl, die es nicht gibt.

---

## 4. Anforderungs-Union

33 Anforderungen wurden abgeleitet; 24 als **muss**. Vollständig im Workflow-Ergebnis, hier die Struktur:

**Formate (muss)**: PDF nativ mit Blockstruktur · PDF gescannt mit OCR und Sprachwahl · Tabellen **gleichzeitig** als GFM-Pipe *und* als Array · DOCX mit Überschriften und Tabellen · PPTX mit Folienlogik · TXT/MD-Passthrough ohne Strukturverlust.

**Qualität (muss)**: Lesereihenfolge bei mehrspaltigem Layout · Formeln als LaTeX · Abbildungen mindestens als beschriebener Platzhalter · Seiten-Provenienz je Block · Sprachkonfiguration, die OCR *und* Prompt gemeinsam steuert.

**Betrieb (muss)**: asynchroner Job mit Polling · Fehler-Isolation nach unten · **Degradationssignal in der Antwort** · deterministischer Fallback ohne Modell · Kosten-Gate, das tatsächlich Fälle ausschließt · harte Deadline je Modell-Call · Token-Budget je Job mit Abbruch · Service-zu-Service-Auth mit eigenem Token für die Billing-Fläche · Größenprüfung **vor** der RAM-Allokation · maschinenlesbare JSON-Antwort · Idempotenz über Content-Hash · Eval-Harness mit Gold-Standard · gepinnte Deps und Testabdeckung.

**Soll**: XLSX/CSV · HTML/EML mit Boilerplate-Entfernung · verbundene Zellen und Multi-Page-Tabellen · Dokument-Metadaten · Batch/Ordner-Eingang · Aufbewahrungsfrist · die Ausgaberichtung (MD→PDF/EPUB) als Teil desselben Dienstes.

**Kann** (nur image_extracter fordert es): Template-basierte Feldextraktion · Human-in-the-Loop-QA-Queue.

---

## 5. Ungemessen — die Liste, die den Recherche-Auftrag trägt

- Der **zentrale Vergleich fehlt vollständig**: CONVERTERs Ensemble wurde nie gegen docling, gegen Gemini-direkt oder gegen irgendetwas gehalten.
- Munchers Score-Historie (55,6 → 86,9) ist die einzige Messung im Bestand und misst Muncher gegen Muncher-Dokumente.
- Alle Ensemble-Schwellwerte sind Setzungen ohne Kalibrierung: IoU 0,3 · Konsens ab 2 · Escape-Hatch bei ≥ 0,8 · die Score-Gewichte · der Verwerfungsschwellwert 0,4. Die Detektor-Konfidenzen sind Konstanten; nur camelot liefert einen gemessenen Wert.
- **Kein Repo zählt Tokens oder Kosten.** „Gemini ist der letzte Ausweg, also billig" hat keine Zahl — obwohl CONVERTER den Fallback nachweislich mehrfach pro Seite auslöst.
- **Kein Repo misst Durchsatz.**
- **Deutsche Sprachqualität ist nirgends gemessen**, obwohl der Bestand überwiegend deutsch ist.
- `strategy="fast"` wurde nie gegen `hi_res` gehalten.

---

## 6. Kuriositäten, die man kennen sollte

- CONVERTER hat **pandoc, LibreOffice und tesseract im Docker-Image** und ruft keines davon je auf. Gleichzeitig ist DOCX der schwächste Punkt der ganzen Union — die billigste verfügbare Lösung liegt ungenutzt im Container.
- CONVERTERs PDF-Service baut seinen genai-Client selbst und **umgeht damit die einzige Stelle im Repo, die einen Timeout auf Gemini setzt** (`TIMEOUT_GEMINI_SECONDS=300`) — synchron, hinter `--workers 1`. Das ist exakt der Fehlermodus, den NARR-TIMEOUT auf der Narration-Seite bereits einmal behoben hat.
- Munchers `scan_folder` ist als Task registriert und hat keinen Endpoint — das gemountete Eingabelaufwerk ist unerreichbar.
- Agentsuite2s `vlm_model` wird berechnet, gemeldet und beim Call ignoriert.
