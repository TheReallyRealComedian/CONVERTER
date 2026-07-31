# SPRINT DOC-FIX — toter Gemini-Pfad, stiller Datenverlust, zerstörerisches Post-Processing

**Größe**: M (4 Phasen) · **Datum**: 2026-07-30 · **Vorhaben**: DOC-SVC (Dokument-Konvertierung als API-Dienst)

## Warum

CONVERTERs Dokument-Konvertierung hat fünf Fehler, die **unabhängig von jeder künftigen Engine-Entscheidung** falsch sind. Zwei davon sind stiller Datenverlust, einer ist seit dem 01.06.2026 in Prod tot. Sie werden jetzt behoben, damit der Pfad eine ehrliche Ausgangsbasis für den späteren Dienst ist — und damit der Bake-off den Eigenbau überhaupt fair messen kann.

**Dieser Sprint entscheidet nichts über die künftige Engine.** Er repariert, was heute kaputt ist.

## Gegroundeter Ist-Zustand (Master — nicht neu herleiten, alle Stellen verifiziert)

1. **[service.py:28](../../../services/pdf_extraction/service.py:28)** — `VISION_MODEL = "gemini-2.0-flash"`. Das Modell ist **abgeschaltet**; gegen die Prod-Instanz verifiziert: `404 NOT_FOUND — This model models/gemini-2.0-flash is no longer available.` Der 404 ist nicht retrybar ([:446](../../../services/pdf_extraction/service.py:446) prüft nur `429`/`rate`/`resource`) → `raise` → in `_extract_scanned_page` ([:341](../../../services/pdf_extraction/service.py:341)) gefangen → Fallback auf `page.get_text()` → bei gescannten Seiten **leer**. Einziges Signal: eine `WARNING`-Zeile.
2. **[service.py:37](../../../services/pdf_extraction/service.py:37)** — `genai.Client(api_key=...)` **ohne Timeout**. Umgeht `TIMEOUT_GEMINI_SECONDS` aus [app_pkg/config.py](../../../app_pkg/config.py), synchron hinter gunicorn `--workers 1 --timeout 1800`. Exakt der Fehlermodus, den NARR-TIMEOUT auf der Narration-Seite schon behoben hat (Memory `reference_worker_sdk_per_call_deadline`).
3. **[app_pkg/documents.py:65-66](../../../app_pkg/documents.py:65)** — `"\n\n".join(el.text)`. Der gesamte Nicht-PDF-Pfad wirft Element-Kategorie (`Title`, `ListItem`, `Table`) und `metadata.text_as_html` weg. `text_as_html` kommt im ganzen Repo **kein einziges Mal** vor. Für DOCX/PPTX/HTML/EML entsteht Fließtext in einer `.md`-Datei. Live belegt: die einzige Dokument-Konvertierung der Library (id 17) hat 21.790 Zeichen und **null** Pipe-Tabellenzeilen.
4. **[service.py:287](../../../services/pdf_extraction/service.py:287)** — `if table_bboxes:` vor `_add_non_table_text`. Schlägt auf einer Seite mit erkannter Tabelle **jede** Extraktion fehl, bleibt `table_bboxes` leer und der **gesamte Fließtext der Seite** wird nie eingesammelt. Die Seite verschwindet rückstandslos.
5. **[service.py:661-662](../../../services/pdf_extraction/service.py:661)** — `re.sub(r'```\w*\n','')` + `.replace('```','')` über das **ganze Dokument**. Entfernt jeden echten Codeblock. Der berechtigte Zweck (Gemini-Artefakte) ist bereits an der Quelle erledigt: [:461-466](../../../services/pdf_extraction/service.py:461) strippt die Fences der Modell-Antwort.
6. **[service.py:682-685](../../../services/pdf_extraction/service.py:682)** — `markdown.replace(link_text, markdown_link)` über das ganze Dokument. Jedes weitere Vorkommen desselben Worts wird zum Link, auch in Tabellenzellen. `link_map` ist zusätzlich auf den sichtbaren Linktext über **alle Seiten** gekeyt — zwei URLs unter „hier" kollidieren.

## Gesperrte Entscheidungen

- **Nachfolgemodell ist `gemini-3.6-flash`.** Ausdrücklich **nicht** `gemini-2.5-flash` — das wird am 16.10.2026 abgeschaltet und landet bei demselben Nachfolger.
- **Der Modellname wird env-overridable.** Heute hat ein hartkodierter Name zwei Monate stillen Ausfall gekostet; das darf sich nicht wiederholen.
- **Kein Umbau der Ensemble-Logik.** Detektoren, IoU-Clustering, Konsens, Scoring, Multi-Page-Merge bleiben unangetastet.

---

# Phase 1 — Den toten Pfad wiederbeleben

## 1.1 Modell

`VISION_MODEL` auf **`gemini-3.6-flash`**, env-overridable über `PDF_VISION_MODEL` (Muster: `NARRATION_TTS_MODEL` in [services/narration_render.py:46](../../../services/narration_render.py:46) — genau so, inklusive `or`-Default).

## 1.2 Timeout

Der selbstgebaute `genai.Client` bekommt eine Deadline aus `TIMEOUT_GEMINI_SECONDS`. Prüfe zuerst, **wie** das SDK das entgegennimmt (`http_options` am Client oder per-Call-`config`) und wähle den Weg, der auch bei einem hängenden Call greift — ein Client-Default, der nur die Verbindung und nicht die Antwort deckelt, ist wertlos. Im Zweifel per-Call, wie NARR-TIMEOUT es für Cloud-TTS gelöst hat.

## 1.3 `media_resolution`

Für Seiten **mit** Textebene soll `media_resolution` auf `low` gesetzt werden (halbiert die Bildtokens; der native Textlayer geht ohnehin ungekürzt und unberechnet mit). Für **gescannte** Seiten bleibt es beim Default.

⚠️ **Erst verifizieren, dann bauen**: `media_resolution` ist ein Gemini-3-Feature. Prüfe, ob das im Container installierte `google-genai` es im `GenerateContentConfig` kennt. Kennt es das nicht, ist die richtige Antwort ein SDK-Bump **oder** — falls der Bump Risiko trägt — dieser Punkt fällt aus Phase 1 heraus und wird als Backlog-Item berichtet. **Nicht** raten, **nicht** an der API vorbeikonstruieren.

## 1.4 Verifikation

Ein echter Call gegen die Prod-Credentials mit einer einseitigen Test-PDF: Modell antwortet, Timeout greift nachweislich (künstlich kleiner Wert → sauberer Fehler statt Hänger), und — falls 1.3 gebaut wurde — `usage_metadata` zeigt den erwarteten Unterschied zwischen `low` und Default.

## Stop
`pytest tests/` grün (Baseline **798**), Live-Call belegt. **Commit + Push** `fix(DOC-FIX): toter Gemini-Pfad wiederbelebt (P1)`. Dann warten.

---

# Phase 2 — Der Office-Pfad liefert echtes Markdown

Der Kern des Sprints und der höchste Ertrag pro Zeile im ganzen Vorhaben.

## 2.1 Was gebaut wird

`"\n\n".join(el.text)` wird durch einen **Serializer** ersetzt, der die `unstructured`-Elemente in Markdown übersetzt. Er gehört **nicht** in die Route — leg ihn als pures Modul an (Vorschlag `services/unstructured_markdown.py`), damit er testbar ist und der spätere Dienst ihn wiederverwenden kann.

Mindestens abzudecken:

- **`Table`** → `metadata.text_as_html` in eine GFM-Pipe-Tabelle. Trägt die Tabelle verbundene Zellen, die Pipes nicht ausdrücken können, ist **HTML im Markdown zu behalten** die ehrlichere Ausgabe als eine falsche Pipe-Tabelle. Fehlt `text_as_html`, Fallback auf `el.text` mit einem Degradations-Vermerk (s. 2.3).
- **`Title`** → ATX-Überschrift. Die Tiefe kommt aus `metadata.category_depth`, falls vorhanden; **verifiziere am echten Objekt**, ob und wie das Feld gefüllt ist, statt es anzunehmen. Ist keine verlässliche Tiefe da, ist eine flache Ebene korrekt — **erfinde keine Hierarchie**.
- **`ListItem`** → `- `. Verschachtelung nur, wenn `category_depth` sie belegt.
- **`NarrativeText`/`Text`/Rest** → Absatz.
- **`PageBreak`** → `\n\n---\n\n`. Trage die Seitennummer mit, wenn `metadata.page_number` sie liefert (drei unabhängige Implementierungen im Bestand sind auf denselben Trenner gekommen; nur eine führt die Nummer mit — das ist die bessere).
- **Code/Formeln**: nicht sonderbehandeln. Nicht raten.

## 2.2 Die Regel, an der sich alles ausrichtet

**Nie mehr Struktur behaupten, als die Quelle hergibt.** Ein Absatz, der als Absatz herauskommt, ist richtig. Ein Absatz, der als `##` herauskommt, weil er kurz ist, ist falsch. Bei Unsicherheit ist die konservative Ausgabe die richtige.

## 2.3 Degradations-Vermerk

Wenn ein Element nicht sauber übersetzt werden konnte (fehlendes `text_as_html`, unbekannte Kategorie), soll das **im Rückgabewert** erkennbar sein — nicht nur im Log. Die Funktion gibt Markdown **und** eine Warnungsliste zurück; die Route entscheidet, was sie damit tut (in dieser Phase: loggen reicht, die Antwortform ändern wir hier noch nicht). Das ist die Vorarbeit für das Degradationssignal des künftigen Dienstes.

## 2.4 Tests

Der Pfad hat heute **null** Tests. Für den Serializer sind sie Pflicht — mit **synthetischen Element-Listen**, nicht mit echten Dateien, damit sie schnell bleiben und die SDK-Boundary gemockt bleibt (Hauspattern). Mindestens: Tabelle mit `text_as_html` → Pipes · Tabelle ohne → Fallback + Warnung · Titel → Überschrift · Liste → Bullets · unbekannte Kategorie → Absatz + Warnung · Seitentrenner mit Nummer · leere Element-Liste → leerer String, kein Crash.

## Stop
`pytest tests/` grün, Testzahl vorher/nachher. **Live-Smoke Pflicht**: ein echtes DOCX **und** ein echtes PPTX durch den Container, Ausgabe zeigen. **Commit + Push** `feat(DOC-FIX): Office-Pfad liefert echtes Markdown (P2)`. Dann warten.

---

# Phase 3 — Stiller Datenverlust und zerstörerisches Post-Processing

## 3.1 Die verschwindende Seite

[service.py:287](../../../services/pdf_extraction/service.py:287): Fließtext wird **immer** eingesammelt, nicht nur wenn `table_bboxes` gefüllt ist. Schlägt die Tabellenextraktion fehl, ist die richtige Ausgabe die Seite **ohne** Tabelle plus eine Warnung — nicht die leere Seite. Test: Seite mit erkannter, aber nicht extrahierbarer Tabelle behält ihren Fließtext.

## 3.2 Der Fence-Sweep

[service.py:661-662](../../../services/pdf_extraction/service.py:661) fällt weg. Der Zweck ist an der Quelle erledigt ([:461-466](../../../services/pdf_extraction/service.py:461)). Falls dort eine Lücke bleibt, wird sie **dort** geschlossen, nicht durch einen globalen Sweep. Test: ein Dokument mit einem echten Codeblock behält ihn.

## 3.3 Der globale Link-Replace

[service.py:682-685](../../../services/pdf_extraction/service.py:682). Anforderung, Mechanik ist deine Wahl:

- Ein Linktext darf **nur an seiner eigenen Fundstelle** zum Link werden, nicht an jeder weiteren im Dokument.
- `link_map` darf nicht global über alle Seiten auf den sichtbaren Text gekeyt sein — zwei URLs unter „hier" kollidieren heute.
- Tabellenzellen dürfen nicht nachträglich verlinkt werden.

Wenn eine saubere Zuordnung mit vertretbarem Aufwand nicht möglich ist, ist **Linkeinbettung ganz abzuschalten** die bessere Antwort als die heutige — begründe die Wahl im Bericht. Tests für beide Kollisionsfälle (Wiederholung, gleicher Anchor-Text mit zwei URLs).

## Stop
`pytest tests/` grün. **Commit + Push** `fix(DOC-FIX): Seitenverlust und zerstörerisches Post-Processing (P3)`. Dann warten.

---

# Phase 4 — Wrap

- **CLAUDE.md**: ein Absatz zum Dokument-Konvertierungs-Pfad — dass der Office-Pfad jetzt strukturiertes Markdown liefert, dass der Modellname env-overridable ist (`PDF_VISION_MODEL`) **und warum** (zwei Monate stiller Ausfall durch einen hartkodierten Namen), und der Hinweis, dass `gemini-2.5-flash` am 16.10.2026 stirbt.
- **STATUS.md** + **BACKLOG.md** (Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md` muss leer sein).
- **Memory**: *Ein hartkodierter Modellname in einem Fallback-Pfad ist ein stiller Ausfall mit Verzögerung — der Fehler zeigt sich erst, wenn der Fallback gebraucht wird, und degradiert dann auf einen Pfad, der für diesen Fall gar nicht gedacht war.* Verlinken auf `[[reference_worker_sdk_per_call_deadline]]` und `[[feedback_verify_feature_reachable_under_user_config]]`. **Nach dem Schreiben mit `ls` prüfen**, dass die Datei liegt und MEMORY.md genauso viele Index-Zeilen hat wie Dateien existieren.
- **Im Bericht benennen**: ob 1.3 (`media_resolution`) gebaut wurde oder als Backlog-Item herausfiel, und was der Live-Smoke aus Phase 2 an DOCX/PPTX tatsächlich gezeigt hat.

## Nicht-Ziele

- **Keine** Engine-Entscheidung, **kein** docling, **kein** MinerU, **kein** Bake-off. Das ist ein eigener Sprint nach dem Korpus.
- **Kein** Umbau der Ensemble-Logik, der Detektoren oder des Multi-Page-Merges.
- **Keine** API-Fläche, **kein** Token, **kein** Job-Modell, **keine** Änderung der Antwortform von `/transform-document`. Der künftige Dienst ist ein eigener Sprint.
- **Kein** Umschalter Cloud-vs-lokal — der gehört zum Engine-Sprint, nicht hierher.
- **Kein** OCR, **keine** Formel-Erkennung, **keine** Bild-Extraktion.
