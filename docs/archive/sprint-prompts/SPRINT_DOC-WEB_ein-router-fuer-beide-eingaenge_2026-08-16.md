# SPRINT DOC-WEB — ein Router für beide Eingänge, und der tote Eigenbau fällt

**Größe**: L (3 Phasen) · **Datum**: 2026-08-16 · **Vorhaben**: DOC-SVC

## Warum

Seit DOC-ENGINE und DOC-LOCAL gibt es **zwei Qualitäten für dieselbe Datei**. Die API routet auf die gemessenen Sieger; der Knopf im Browser ([app_pkg/documents.py](../../../app_pkg/documents.py)) hängt unverändert an `partition` + `elements_to_markdown` für Office und an `PDFExtractionService` für PDF. Wer die gute Konvertierung will, braucht einen Token.

Dieser Sprint führt beide Eingänge auf **denselben Router**. Dadurch wird der Eigenbau — fünf Detektoren, Ensemble, Multi-Page-Merge — **tot**, und dann fällt er.

**Er ersetzt das Backlog-Item DOC-ROUTE in seiner Juli-Form.** Begründung steht unten in den gegroundeten Befunden: das Seiten-Routing hat seinen Zweck verloren, und die Detektoren aus einem Dienst zu reißen, den dieser Sprint ohnehin ersetzt, wäre doppelte Arbeit. Er schließt außerdem **DOC-WEB-KONVERGENZ** und als Nebenwirkung **DOC-MEDIARES**.

## Gegroundeter Ist-Zustand (Master — nicht neu herleiten, alles am 2026-08-16 gemessen)

**`PDFExtractionService` hat genau EINEN Konsumenten**: [app_pkg/documents.py:63](../../../app_pkg/documents.py). Sonst nichts — der API-Dienst berührt das Paket nicht. Die fünf Detektoren, `ensemble.py` und `multi_page.py` existieren ausschließlich für diesen einen Aufruf.

**Was der Web-Pfad heute tut** ([app_pkg/documents.py](../../../app_pkg/documents.py)): synchron, Session-authed, Multipart rein, **Datei-Download raus** (`send_file`). PDF → `pdf_extraction_service.extract_markdown()`. Alles andere → `partition(strategy="fast")` + `elements_to_markdown`. ⚠️ **Die Warnungen gehen ausschließlich ins Log** (`app.logger.warning`) — der Nutzer erfährt nie, was degradiert ist. Das widerspricht der Doktrin des ganzen Vorhabens („Degradationen gehören in die Antwort, nicht ins Log") und ist Teil dieses Sprints.

**Der Router existiert schon**, als Endungs-Verzweigung in [tasks.py](../../../tasks.py)`::convert_document_task`: `docx`→pandoc · `pptx`→markitdown · `html`/`htm`→trafilatura (+`backend_fallback`) · `eml`/`txt`/`md`→unstructured · `pdf`→`_convert_pdf`. Die Backends in [services/office_backends.py](../../../services/office_backends.py) liefern bereits `(markdown, warnings)` — **genau die Form, die der synchrone Web-Pfad braucht**. Was der Web-Pfad **nicht** braucht: Job, Budget, Provenance-Payload.

**Das Seiten-Routing hat seinen Zweck verloren.** `_classify_page` ([services/pdf_extraction/service.py:250](../../../services/pdf_extraction/service.py)) unterscheidet `native`/`mixed`/`scanned` über Bildabdeckung und Textdichte, um zu entscheiden, ob eine Seite einen Vision-Call wert ist. Seit DOC-LOCAL sind **beide** Engines Vision-Engines, die jeden Seitentyp lesen (mineru: 15-Seiten-Scan ohne Textebene → 38.610 Zeichen). Übrig bleibt **eine** kleine Verwendung, die du mitnehmen sollst: beim Textebenen-Rückfall benennen können, dass eine Seite ein Scan ist und „leer" dort erwartbar ist — statt still nichts zu liefern.

**Der Merge lässt sich nicht heben, wie er ist.** Von den vier Signalen in `detect_continuation_signals` sind **drei geometrisch** (bbox am Seitenfuß/-kopf, Spalten-x-Koordinaten, Spaltenzahl) und einer ein Keyword. Der Cloud-Pfad liefert reines Markdown, **null Geometrie** — `is_continuation` fiele auf den Keyword-Zweig zurück und feuerte praktisch nie. ⚠️ **Er wird in diesem Sprint nicht gerettet, sondern gelöscht.** Die Fähigkeit „Tabelle über die Seitengrenze zusammenführen" bleibt als eigenes Item offen und ist heute ohnehin **nicht messbar**: das Gold für den harten Fall (Kopf wiederholt sich **nicht**) existiert nicht (DOC-KORPUS-SPAN). Eine strukturelle Neuformulierung ohne Geometrie ist skizziert — „letzter Block von Seite N ist eine Tabelle, erster Block von N+1 auch, gleiche Spaltenzahl" — aber **nicht Teil dieses Sprints**.

**Die Detektoren sind gemessen wertlos**: auf `01.gold` ist die Ausgabe bis zur vierten Nachkommastelle identisch mit roher Textextraktion, **Tabellenzellen 0,0** auf einer Seite mit 37 Tabellen-Datenzeilen; auf Klasse 02 verlieren sie durch Falschdetektion **die halbe Wortmenge** (Recall 0,489, zerhackter Fließtext: `| im per | s | önlichen | Gesprä |`).

**Testabdeckung des Pakets ist dünn**: nur `tests/test_pdf_postprocess.py` und `tests/test_documents.py` fassen es an. ⚠️ Das heißt beides — der Abriss bricht wenig, **und** die Suite fängt eine Regression im Web-Pfad nicht. Live-Smoke ist Pflicht.

**Zeitrahmen des synchronen Pfads**: gunicorn läuft mit `--timeout 1800` und **einem** Worker. mineru kostet 61 s + 2,5 s/Seite, ein Cloud-Lauf N sequentielle Gemini-Calls. Der heutige Pfad hat dieselbe Aussetzung (er ruft schon heute pro Seite Gemini) — **neu ist, dass du sie messen und begrenzen sollst**, statt sie hängen zu lassen.

## Gesperrte Entscheidungen

1. **Ein Router, zwei Aufrufer.** Die Endungs-Verzweigung wandert in eine geteilte, **pure** Funktion, die `(markdown, warnings)` liefert. `tasks.py` wickelt sie weiter in den Payload; der Web-Pfad nimmt sie direkt. Keine zweite Wahrheit über Formate.
2. **Der Web-PDF-Zweig fährt die echten Engines** (`run_cloud_pdf` / `run_local_pdf`). Der Modus kommt aus **Olis bestehender Einstellung** (`GET/PUT /api/document-conversions/settings`, `document_api`-Namespace) — es wird **keine** zweite Einstellung erfunden. Budget: `DOC_CONVERT_BUDGET_EUR`, derselbe Deckel wie die API.
3. **`services/pdf_extraction/` fällt komplett**, sobald sein einziger Konsument umgestellt ist — inklusive `detectors.py`, `ensemble.py`, `multi_page.py`, `service.py`, dem Shim `services/pdf_extraction_service.py` und dem Singleton `app.pdf_extraction_service`. ⚠️ **Caller-first**: vor jedem Delete `grep`en, **Tests sind auch Caller** (Lehre `reference_flow_retirement_shared_package`).
4. **Warnungen erreichen den Nutzer.** Was der API-Dienst als `degradations` ausliefert, darf der Web-Nutzer nicht nur im Log haben.
5. **Kein Merge, kein Seiten-Routing als Feature.** Beides fällt; die eine überlebende Verwendung des Klassifikators (Scan-Erkennung für die Rückfall-Meldung) wird als kleiner Helfer mitgenommen, nicht als Router.

---

# Phase 1 — Ein Router für alle Formate außer PDF

## 1.1 Die geteilte Funktion

Hebe die Endungs-Verzweigung aus `tasks.py` in ein pures Modul (Vorschlag: `services/document_router.py`, oder in `services/document_conversions.py` — begründe). Signatur in Richtung `convert_non_pdf(source_path, source_ext) -> (markdown, warnings)`.

`tasks.py` ruft sie und wickelt wie bisher in `_deterministic_document_payload`. `app_pkg/documents.py` ruft sie und schreibt das Markdown in den Download. **Eine** Stelle kennt die Formate.

## 1.2 Der Web-Pfad bekommt sie

DOCX, PPTX, HTML/HTM, EML, TXT, MD laufen im Browser ab jetzt über dieselben Backends wie die API. XLSX bleibt der bestehende 400.

## 1.3 Belege

Für jedes Format ein Lauf **durch die Web-UI** (nicht nur durch die Funktion), Ergebnis gegen den API-Output derselben Datei: **byte-identisch**. Das ist der Beweis, dass es eine Qualität gibt und nicht zwei. Wo eine Gold-Fassung existiert (`08.md`), zusätzlich dagegen.

## Stop
`pytest tests/` grün (Baseline **940**). **Commit + Push** `feat(DOC-WEB): geteilter Router fuer die Nicht-PDF-Formate (P1)`. Dann warten.

---

# Phase 2 — Der PDF-Zweig und der Abriss

## 2.1 Der Web-PDF-Zweig

Er ruft `run_cloud_pdf` / `run_local_pdf` nach dem Modus aus der Einstellung (Entscheidung 2). Die Funktionen liefern den vollen Payload; der Web-Pfad braucht daraus `markdown` und `degradations`.

⚠️ **Miss die synchrone Grenze, statt sie zu raten.** Ein Lauf blockiert den einzigen gunicorn-Worker. Stell fest, ab welcher Seitenzahl das unvertretbar wird, und setz eine **benannte** Grenze mit klarer deutscher Meldung („Dieses PDF hat N Seiten. Nutze dafür den Dienst unter …") statt eines Timeouts. Der Wert gehört in den Bericht.

## 2.2 Der Abriss

Wenn 2.1 steht, ist `services/pdf_extraction/` ohne Konsument. Löschen — Paket, Shim, Singleton in `app.py`, Import-Kette, tote Tests. ⚠️ **`services/pdf_extraction/` war bisher in jedem DOC-Sprint ein ausdrückliches Nicht-Ziel; hier ist es das Ziel.** Caller-first grep, dann löschen.

Prüf, ob dabei Dependencies frei werden (`camelot`, `img2table`, `pdfplumber` — die Detektoren waren ihre einzigen Nutzer?). Wenn ja, aus `requirements.txt` und dem Image raus, und **die Image-Größe vorher/nachher berichten**.

⚠️ **Was NICHT verloren gehen darf**, weil es teuer gelernt wurde und in `pdf_cloud.py` schon repliziert ist: die per-Call-Deadline, das explizite `media_resolution`, `_strip_wrapper_fence`. Vergleiche vor dem Löschen, ob `pdf_extraction/service.py` etwas kann, das die neuen Module **nicht** können — und wenn ja, **berichte es, statt es stillschweigend fallen zu lassen**.

## 2.3 Der überlebende Rest des Klassifikators

Nimm die Scan-Erkennung als kleinen Helfer mit (Entscheidung 5): fällt der lokale Pfad auf die Textebene zurück und ist die Seite ein Scan, soll die Degradation das **sagen** — „Seite N ist ein Scan, die Textebene ist dort leer" — statt still nichts zu liefern.

## Stop
Web-PDF fährt die echten Engines, `pdf_extraction/` ist weg, Suite grün. **Commit + Push** `feat(DOC-WEB): Web-PDF auf die echten Engines, Eigenbau abgerissen (P2)`. Dann warten.

---

# Phase 3 — Hinweise sichtbar machen, dann Wrap

## 3.1 Der Nutzer sieht, was degradiert ist

Heute endet der Web-Pfad in `send_file`, und Warnungen gehen ins Log. Bring sie zum Nutzer. ⚠️ **Prüf zuerst, was das im Frontend kostet** ([static/js/document_converter.js](../../../static/js/document_converter.js)): eine JSON-Antwort mit clientseitigem Download ist der saubere Weg, aber wenn das mehr als eine überschaubare Änderung ist, **berichte es und schlag den kleineren Schnitt vor**, statt die UI umzubauen. Microcopy nach Hausregel: Fehler max 2 Sätze, keine Emojis.

## 3.2 Wrap

- **Kontrakt-Doc** — der Vertrag der API ändert sich nicht, aber §10 sagt jetzt „ein Router, zwei Eingänge".
- **CLAUDE.md**: der `pdf_extraction`-Bullet und alle Verweise darauf sind nach dem Abriss **falsch** — nachziehen, nicht ergänzen. Der DOC-FIX-Bullet beschreibt Verhalten, das es nicht mehr gibt.
- **STATUS.md**, **BACKLOG.md** (Bullet-Guard): DOC-WEB-KONVERGENZ und DOC-ROUTE schließen, **DOC-MEDIARES** prüfen (wenn der Web-Pfad jetzt über `run_cloud_pdf` mit MEDIUM läuft, ist es erledigt — wenn nicht, sagen warum). Ein neues Item für die **Tabellen-Fortsetzung über Seitengrenzen** anlegen, mit der strukturellen Skizze und der Abhängigkeit von einem Gold-Belegexemplar.
- ⚠️ **Engine-Generation bumpen** (`DOC_CONVERT_ENGINE_GENERATION`, [services/document_conversions.py](../../../services/document_conversions.py)): wenn sich am Ergebnis irgendeiner Datei etwas ändert, muss der Dedup aufhören, alte Antworten zu servieren. Die Bump-Regel steht am Konstanten-Wert.
- **Memory**, falls übertragbar; nach dem Schreiben mit `ls` prüfen, dass Datei und Index-Zeile zusammenpassen.
- **Im Bericht benennen**: die synchrone Seitengrenze und wie du sie gemessen hast · welche Dependencies frei wurden und was das Image kostet · ob `pdf_extraction` etwas konnte, das die neuen Module nicht können · was die Hinweis-Anzeige im Frontend gekostet hat.

## Nicht-Ziele

- **Kein** Tabellen-Merge über Seitengrenzen (eigenes Item, heute nicht messbar).
- **Kein** Seiten-Routing als Feature.
- **Kein** Umbau der API-Antwortform, des Job-Modells oder des Kontrakts.
- **Kein** Umbau der Web-UI über die Hinweis-Anzeige hinaus.
- **Kein** neuer Einstellungs-Schalter.
- ⚠️ **Editiert wird nur auf dem Mac.** Die Mintbox ist Runtime — Deploy und Smoke ja, Arbeitsplatz nein, keine unversionierten Dateien zurücklassen.
- ⚠️ **Live-Smoke ist Pflicht**, nicht optional: die Suite rendert keine Templates und deckt `pdf_extraction` kaum ab.
