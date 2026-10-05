# SPRINT LERN-TEXT — Lerntexte zu Karten und Sammlungen

**Größe**: L (4 Phasen: Schema + API · Reader-Sprung + Detailseite · Review + Launcher + Deploy · Wrap + Brief) · **Datum**: 2026-10-05 · **Herkunft**: Olis Wunsch vom 2026-10-05, Entwurf mit dem Master abgestimmt (drei Entscheidungen von Oli: eine Textstelle je Karte · Done-Panel „Nacharbeiten" gleich mit · Knopf erst nach dem Aufdecken)

## Warum

Oli lässt sich zu Lernabschnitten (z. B. Kapitel aus „Chemie für Dummies") **Lerntexte** vom Agenten schreiben: zusammenhängende Einführungen, die mehr Kontext geben als die Karten allein. Zwei Nutzungen: **Einarbeitung** — den Text lesen, bevor der Abschnitt gelernt wird; **Nacharbeiten** — von einer Karte, die schwerfällt, zu der Textstelle gehen, die sie erklärt. Heute gibt es dafür keinen Ort: Texte liegen irgendwo in der Library, Karten kennen ihre Herkunft (`highlight_id`, `source_doc_title`), aber keinen Weg zurück zum Lesen, und „Vertiefen" markiert eine Karte als `wackelt`, ohne ihr ein Ziel zu geben.

Gebaut wird **die Verknüpfung**, in zwei Richtungen: Sammlung → Texte (Kapitelebene) und Karte → Textstelle (Kartenebene). Texte entstehen beim Agenten (wie die Karten), gelesen werden sie in der Library (wie jedes Dokument). CONVERTER speichert, zeigt, verknüpft — **kein LLM-Pfad in CONVERTER**.

## Gegroundeter Ist-Zustand (Master, gemessen 2026-10-05)

**Datenmodell** ([models.py](../../../models.py)):
- `Card`: `id`, `user_id`, `highlight_id` (Provenienz-FK, `ondelete='SET NULL'`, bare column), `source_snapshot`, `source_doc_title`, `type` (`atomic`|`generative`), `front`/`back`/`cloze_text`/`prompt`, `note`, `front_svg`/`back_svg`, `state` (`ok`|`wackelt`, `CARD_STATES` in [app_pkg/cards.py](../../../app_pkg/cards.py) Z. 58), `created_by`, `created_at`/`updated_at`, `tags` (M2M `card_tags`), `collections` (M2M `card_collections`). **`Card.to_dict()`** (Z. 379) ist die eine autoritative Serialisierung (CARD-SVG: dort wird sanitisiert — Web, MCP, iOS bekommen dasselbe); `_card_summary` ([cards.py](../../../app_pkg/cards.py) Z. 321) ist die schlanke Listenform ohne Figuren.
- `Collection`: `id`, `user_id`, `name`, `description`, `created_at`; `to_dict(card_count, due_count)` (Z. 530). **Keine Beziehung zu Dokumenten.**
- `Conversion`: Typen-Allow-List `ALLOWED_CONVERSION_TYPES` ([app_pkg/library.py](../../../app_pkg/library.py) Z. 24: `document_to_markdown`, `audio_transcription`, `dialogue_formatting`, `markdown_input`, `ai_newsletter`, `audio_narration`, `document_conversion`). Der Agent legt Dokumente heute per `create_conversion` an und ändert sie per `update_document`/`replace_section`.
- ⚠️ **SQLite ohne FK-Pragma** (Memory `reference_sqlite_no_fk_pragma_orm_delete`): `ON DELETE` ist DB-seitig inert. Jede Lösch-Folge (Dokument weg → Karten-Kontext nullen, Junction-Zeilen weg) läuft über den ORM — `relationship(cascade=)` oder explizit im Lösch-Handler.

**Endpunkte:**
- Karten ([app_pkg/cards.py](../../../app_pkg/cards.py)): `POST /api/cards` (Z. 378, `_authorize_card_write` Z. 197 — Token oder Session; ⚠️ der Token-Pfad schreibt immer auf `INGEST_USER`/ersten User = Olis Konto), `PATCH /api/cards/<id>` (Z. 441), `GET /api/cards` (Z. 535), `GET /api/cards/<id>` (Z. 559), `POST …/review` (Z. 730), `POST …/annotate` (Z. 772, setzt `state`/`note`), `DELETE` (Z. 803). `GET /api/review-state` (Z. 567) liefert **volle Karten** (`to_dict`).
- Sammlungen ([app_pkg/collections.py](../../../app_pkg/collections.py)): alle `@login_required` (Session oder per-User-Bearer — der converter-mcp meldet sich per Cookie-Session an): `GET /api/collections` (Z. 33, ⚠️ **blankes Array**, die iOS-App dekodiert `[LearnCollection]` — kein Wrapper, kein synthetischer Eintrag; additive Felder je Eintrag sind unschädlich), `POST` (Z. 65), `PATCH /<id>` (Z. 93), `DELETE /<id>` (Z. 130), `POST /<id>/cards` (Z. 142), `DELETE /<id>/cards/<card_id>` (Z. 162).
- Agent-Fläche am lebenden converter-mcp (Tool-Liste 2026-10-05 gelesen): `create_card`, `update_card`, `get_card`, `list_cards`, `list_collections`, `review_state`, `create_conversion`, `update_document`, `replace_section`, `list_conversions`, `get_transcript`, Tag-Tools, Highlight-Lese-Tools. **Kein Tool schreibt an Sammlungen** (sie entstehen by-name über `create_card`). **Kein Tool legt Highlights an.**

**Review** ([templates/review.html](../../../templates/review.html), [static/js/review.js](../../../static/js/review.js)):
- Done-Panel `#review-done` (Z. 71) mit `#review-done-text` (Z. 73) und `#review-more` (Z. 74, LEARN-MORE). Aufdecken `#review-reveal-btn` (Z. 104), Lösung `#review-answer-wrap` (Z. 115). Nach dem Aufdecken: die vier Bewertungen, dazu **„Vertiefen"** `#review-deepen-btn` (Z. 159, Titel „Karte als ‚wackelt' markieren — Einstieg in den Agent-Dialog") und **„Notiz"** `#review-note-toggle` (Z. 161).
- JS: `renderCard` (Z. 139), `decrementPoolCounts` (Z. 232), `finishSession` (Z. 288, zeigt `doneEl`), `rate(rating)` (Z. 344), `deepen()` (Z. 390, `POST …/annotate {state:'wackelt'}`), `load()` (Z. 513), `renderScopePills` (Z. 585). Der Kartentext läuft durch `renderCardMarkup` (DOM-Knoten, kein `innerHTML` — CARD-MD); die zwei Figuren-Container sind die einzige `innerHTML`-Stelle.
- Tasten 0–4 sind belegt (LEARN-SKIP, LEARN-RATE).

**Reader** ([templates/library_detail.html](../../../templates/library_detail.html) Z. 103 `<article class="reader-view p-6">`, [static/js/library_detail.js](../../../static/js/library_detail.js)): `scrollToHighlight(id)` (Z. 616: `querySelector` in `.reader-view` auf `data-highlight-id`, `scrollIntoView({behavior:'smooth', block:'center'})`), `loadHighlights()` (Z. 525); Lese-Fortschritt wird beim Öffnen wiederhergestellt („Weiterlesen", furthest-read). **Keine Anker an Überschriften**, kein Umgang mit `location.hash`. Der Renderer ([app_pkg/markdown_render.py](../../../app_pkg/markdown_render.py), markdown-it-py 3.0.0 + mdit-py-plugins 0.4.2, nur `dollarmath`) bleibt in diesem Sprint **unangetastet** (Begründung in Entscheidung 4).

**Überschriften als Adresse:** [services/markdown_sections.py](../../../services/markdown_sections.py) hat `_iter_headings(lines)` (Z. 48), `derive_title` (Z. 75), `replace_section(markdown_text, heading, new_section)` (Z. 114) — fenced-code-aware, level-aware, bei Mehrdeutigkeit konservativ (> 1 Treffer → der Aufrufer antwortet 409). **Das ist die Haus-Konvention für „eine Stelle im Dokument"**; die Kontext-Prüfung dieses Sprints nutzt dieselbe Überschriften-Erkennung und dieselbe Vergleichsregel, keine zweite.

**Daten:** Olis Sammlungen: BI-Pipeline, Plant & Process, TCE, Chemie-Basics (Stand LEARN-QUEUE). 206 Karten (Stand CARD-MD), davon viele agent-geschrieben ohne `highlight_id`.

**Baseline:** 1581 passed + 1 skipped (Mac und Pin). Prod-Image `4c4b45a93c15`, Rollback-Tag `pre-notion-meeting-link`; Mintbox-Clone auf `8b22606`. Web läuft unter `docker-init` (Browser-Smokes im Web-Container unkritisch).

## Gesperrte Entscheidungen

1. **Lerntexte sind Library-Dokumente.** Kein neuer `conversion_type`, keine eigene Tabelle für Texte, kein zweiter Reader. Ein Dokument ist ein Lerntext, weil eine Sammlung oder eine Karte auf es zeigt — **die Verknüpfung ist die Markierung.** (Ein neuer Typ bräche womöglich die iOS-Dekodierung und kostete Filter, Badges, Kindle-/PDF-Pfade; alles, was ein Text zum Gelesenwerden braucht, hat die Library schon.)
2. **Karte → genau eine Textstelle** (Oli): zwei Spalten an `Card` — `context_conversion_id` (FK `conversion.id`, nullable) und `context_heading` (Text, nullable). `context_heading` darf `NULL` sein bei gesetztem Dokument (= „der Text als Ganzes", für kurze Texte); umgekehrt nie. Geschrieben über `context` an `POST /api/cards` und `PATCH /api/cards/<id>`: `{"document_id": <int>, "heading": "<Text>" | null}` oder `null` zum Lösen. Prüfungen beim Schreiben, fail-closed: Dokument muss **demselben User** gehören (sonst 404 „Dokument nicht gefunden."), die Überschrift muss im Dokument **vorkommen** (sonst 400 „Überschrift nicht gefunden: ‚…'.") und **eindeutig** sein (sonst 409 „Überschrift kommt N-mal vor." — dieselbe Regel wie `replace_section`). Der Agent bekommt damit sofort Rückmeldung, ob sein Link trägt.
3. **Sammlung → Texte** als Junction `collection_documents(collection_id, conversion_id, position)`, owner-gleich. Schreibweg `PUT /api/collections/<id>/documents` mit `{"documents": [<id>, …]}` (**ersetzt** die Liste, Reihenfolge = Position; leere Liste löst alles; fremdes oder fehlendes Dokument → 404, nichts geschrieben), `@login_required` wie die Geschwister. Lesen: jeder Eintrag von `GET /api/collections` trägt additiv `documents: [{id, title, url}]` in Positionsreihenfolge — **das Array bleibt ein Array.**
4. **Der Sprung zur Textstelle geht über den Überschriften-Text, nicht über `id`-Attribute.** `Card.to_dict()` liefert `context: {document_id, document_title, heading, url} | null` mit `url = /library/<id>` bzw. `/library/<id>#h=<encodeURIComponent(heading)>`. Der Reader liest `location.hash` nach dem Rendern **und** nach `loadHighlights()`: `#h=…` → erste `h1–h6` in `.reader-view`, deren `textContent` (getrimmt, Whitespace normalisiert) der dekodierten Überschrift entspricht → `scrollIntoView({block:'start'})` + eine kurze visuelle Hervorhebung (CSS-Klasse, die nach ~2 s abklingt); **der Hash schlägt die Fortschritts-Wiederherstellung.** Kein Treffer → ruhiger Hinweis über dem Text: „Abschnitt nicht gefunden. Der Text wurde seit der Verknüpfung geändert." (kein Fehler, der Text bleibt lesbar). **Warum nicht `id`s am Renderer:** sie änderten die Bytes **jedes** gerenderten Dokuments (Byte-Gate, EPUB-XML-Namen, Highlight-Gates), brauchten eine Slug-Funktion an zwei Orten und dedupe-Regeln — für einen Sprung, den auch der Text leistet. Die `id`-Variante steht als benannte Alternative im Bericht, nicht im Code.
5. **Review:** nach dem Aufdecken (Oli) ein Link-Knopf **„Lerntext"** neben Vertiefen und Notiz — `<a href=context.url target="_blank" rel="noopener">`, `title` = `document_title` + „ › " + `heading`; nur gerendert, wenn `card.context` gesetzt ist; **vor dem Aufdecken nicht im DOM** (LEARN-SKIP-Doktrin: vor dem Aufdecken wäre er ein Spickzettel). Keine Taste (0–4 belegt). Kein Reader im Review.
6. **Done-Panel „Nacharbeiten"** (Oli): unter dem Done-Text eine Liste der Textstellen zu den Karten, die **in dieser Session** mit Nochmal (1) oder Schwer (2) bewertet **oder** per Vertiefen als `wackelt` markiert wurden **und** einen Kontext tragen — gruppiert nach Dokument (Titel), darunter je Überschrift ein Link (neuer Tab), eine Überschrift einmal auch bei mehreren Karten. Rein client-seitig aus dem Session-Zustand (kein neuer Server-Zustand, kein Request); jedes `load()` setzt die Liste zurück (dieselbe Semantik wie LEARN-SKIP). Leere Liste → Abschnitt nicht gerendert. Alle Texte als **Text-Knoten** (Dokument-Titel und Überschriften sind Eingabe).
7. **Launcher:** unter den Sammlungs-Pillen eine Zeile „Lesen: ‹Titel› · ‹Titel›" mit den Texten der **angehakten** Sammlungen (Links, neuer Tab, Positionsreihenfolge, dedupliziert, Text-Knoten); nur gerendert, wenn mindestens ein Text da ist. Daten aus `documents` an `/api/collections` — kein zweiter Request.
8. **Library-Detailseite:** zwei ruhige Zeilen unter dem Titel, serverseitig gerendert (Jinja, Autoescape): „Lerntext zu: ‹Sammlung›, ‹Sammlung›" (Sammlungen, die auf das Dokument zeigen) und „‹N› Karten verweisen auf diesen Text" (N > 0). Beide nur, wenn zutreffend.
9. **Löschen ist ORM-Sache** (Entscheidung aus dem Ist): Dokument gelöscht → `context_*` der Karten auf `NULL`, `collection_documents`-Zeilen weg, **im selben Commit**; Sammlung gelöscht → ihre Zeilen weg; Karte gelöscht → nichts weiter. Je Fall ein Test, der das Löschen **über den bestehenden Endpunkt** fährt (nicht nur `db.session.delete`).
10. **`to_dict` ohne N+1:** `review-state` liefert bis zu 200 Karten — `document_title` kommt über eine Beziehung mit `lazy='joined'` oder eine Vorab-Abfrage der Titel je Request, nicht über eine Query je Karte. Beleg: Zahl der SQL-Statements für 50 Karten mit Kontext im Test (SQLAlchemy-Event-Zähler) ist unabhängig von der Kartenzahl.
11. **Microcopy deutsch, Haus-Regeln** (Fehler ≤ 2 Sätze, Knopf ≤ 3 Wörter, keine Emojis): „Lerntext", „Nacharbeiten", „Lesen", „Dokument nicht gefunden.", „Überschrift nicht gefunden: ‚…'.", „Überschrift kommt N-mal vor.", „Abschnitt nicht gefunden. Der Text wurde seit der Verknüpfung geändert."
12. **Nicht in diesem Sprint:** iOS (die neuen Felder sind additiv, Knopf und Liste gibt es nur im Web — Folge-Item), ein LLM-Pfad in CONVERTER, `id`-Anker für Menschen, mehrere Stellen je Karte, eine Nacharbeiten-Seite über Sessions hinweg (wackelnde Karten mit Kontext wären ihr Inhalt — Folge-Item, falls Oli sie will), Änderungen an EPUB/PDF, ein Highlight-Schreib-Tool, der Skill-Text für claude.ai (liefert der Master nach dem Sprint), Änderungen an der Semantik von `wackelt`.

**Arbeitsweise:** inline, **kein Workflow, keine Subagenten** ohne Olis ausdrückliches Wort. Commit + Push je Phase, dann Stop + Bericht. Nichts Tragendes im Session-Scratch. Editiert wird nur auf dem Mac; die Mintbox ist Runtime. Tests zuerst, **gegen HEAD rot gezeigt** (Zahl der roten Tests im Bericht — der Master prüft sie unabhängig per `git archive`). Wegwerf-User in Smokes **strikt per `user_id`** aufräumen (`api_token` trägt Olis iOS-Tokens). Keine Karten-, Dokument- oder Sammlungs-Inhalte Olis in Ausgaben.

---

# Phase 1 — Schema + API (Backend)

## Tests zuerst (rot gegen HEAD)

Neue Datei `tests/test_lern_text.py`, Muster der Nachbarn (`test_job_id_reuse.py`, `test_settings_lost_update.py`):
- **Karten-Kontext:** `POST /api/cards` mit `context` → gespeichert, `to_dict().context` trägt `document_id`, `document_title`, `heading`, `url` (Form mit und ohne `#h=`, `encodeURIComponent`-Form für Umlaute/Leerzeichen); `context: null` löst; `PATCH` ändert und löst; fremdes Dokument → 404; fehlende Überschrift → 400 mit dem Satz; mehrdeutige Überschrift → 409 mit der Zahl; `heading` ohne `document_id` → 400; `heading` in einem Code-Fence zählt nicht (fenced-code-aware wie `replace_section`); Vergleichsregel identisch mit `replace_section` (ein Test, der beide mit derselben Eingabe füttert).
- **Sammlung → Texte:** `PUT …/documents` ersetzt, ordnet, löst; fremdes Dokument → 404 und **nichts** geschrieben (Zustand davor bleibt); `GET /api/collections` ist weiter eine Liste (`isinstance(list)`-Sentinel) und trägt `documents` in Positionsreihenfolge; Owner-404 der Sammlung.
- **Löschen über die Endpunkte** (Entscheidung 9): Dokument löschen → Karten-Kontext `NULL`, Junction leer; Sammlung löschen → Junction leer; Karte löschen → Dokument und Junction unberührt.
- **Kein N+1** (Entscheidung 10).
- **Migration:** Bestands-DB ohne die Spalten → nach `_run_pending_migrations` vorhanden, zweiter Lauf idempotent (Muster `test_lifecycle.py`; ⚠️ Pool-Hygiene: Tests mit zweiter Anfrage im Thread geben danach `db.engine.dispose()` zurück, Memory `reference_sqlite_pool_stale_schema_cache`).
- **`_card_summary`** bleibt ohne `context` (Listenform schlank) — oder trägt es bewusst; entscheiden und im Test festnageln (Empfehlung: ohne, wie bei den Figuren).

## Bau

- `models.py`: zwei Spalten + Beziehung an `Card` (ohne DB-seitige Kaskade zu bauen — sie wäre inert), Junction `collection_documents`, Beziehungen an `Collection` und `Conversion` so, dass die ORM-Löschfolgen aus Entscheidung 9 greifen; `Card.to_dict()` → `context`; `Collection.to_dict()` → `documents`.
- Migration in `_run_pending_migrations` (zwei `ALTER TABLE card ADD COLUMN`; die Junction entsteht per `create_all`).
- `app_pkg/cards.py`: `context` an POST und PATCH, Prüfungen aus Entscheidung 2 über **einen** Helfer (`_resolve_card_context(user, payload) -> (conversion, heading) | raise`), Überschriften-Erkennung aus `services/markdown_sections.py` (dort ggf. eine kleine öffentliche Funktion `find_heading(markdown_text, heading) -> count` neben `replace_section`, dieselbe Erkennung).
- `app_pkg/collections.py`: `PUT /<id>/documents`; `GET /api/collections` mit `documents`.
- Lösch-Handler: Dokument-Delete in `app_pkg/library.py` räumt Karten-Kontext und Junction im selben Commit.

## Stop
Bericht: rote Zahl gegen HEAD (vor dem Bau) · grüne Suite · Statement-Zähler aus Entscheidung 10 · Entscheidung zu `_card_summary` · Commit-Hash. Dann warten.

---

# Phase 2 — Reader-Sprung + Detailseite (kein Deploy)

- **Reader** ([static/js/library_detail.js](../../../static/js/library_detail.js)): `#h=…` nach Entscheidung 4 — Auflösung über `textContent` der `h1–h6` in `.reader-view`, Hervorhebung per CSS-Klasse in [static/css/style.css](../../../static/css/style.css) (TOC-Abschnitt ergänzen), Hinweis bei Nicht-Treffer als Text-Knoten; der Hash gewinnt gegen die Fortschritts-Wiederherstellung (belegen: ein Dokument mit gespeichertem Fortschritt weit unten, Sprung auf eine Überschrift oben → Viewport steht auf der Überschrift). Kein Eingriff in `markdown_render.py` (Diff-Beleg im Bericht: die Datei ist nicht angefasst, `gate_render_bytes.py` damit nicht nötig — ausdrücklich sagen).
- **Detailseite** (Entscheidung 8): zwei Zeilen, serverseitig; die Sammlungs-Namen und die Kartenzahl kommen aus der View ([app_pkg/library.py](../../../app_pkg/library.py)), eine Query je Zeile.
- **Smoke Teil A** — neue Datei `scripts/smoke_lern_text.py` (Muster [scripts/smoke_review_skip.py](../../../scripts/smoke_review_skip.py): Playwright im Web-Container, Wegwerf-User mit **eigenem** Dokument, eigener Sammlung und eigenen Karten per ORM, Docstring = Laufanleitung): Dokument mit drei Überschriften (eine mit Umlaut und Leerzeichen) → Aufruf `/library/<id>#h=<encoded>` → die Überschrift steht im Viewport (BoundingClientRect), Hervorhebungs-Klasse gesetzt und nach dem Abklingen weg; falsche Überschrift → Hinweis sichtbar, Text sichtbar; Fortschritt-vs-Hash-Fall. Zweimal laufen lassen, Kriterien erst danach festschreiben (Memory `reference_browser_smoke_in_app_container`).

## Stop
Bericht: Smoke-Ausgabe Teil A (zwei Läufe) · Beleg „Renderer unangetastet" · Suite · Commit-Hash. Dann warten.

---

# Phase 3 — Review + Launcher + Deploy

## Bau
- **Knopf „Lerntext"** (Entscheidung 5) in [templates/review.html](../../../templates/review.html) neben Z. 159–161; in `renderCard` **nicht** gerendert, erst beim Aufdecken eingehängt (und beim nächsten `renderCard` wieder entfernt — Stale-Falle wie bei der Back-Figur, CARD-SVG).
- **Done-Panel „Nacharbeiten"** (Entscheidung 6): Session-Zustand in `review.js` (Set der Karten-ids mit Rating 1/2 oder `deepen()`), Aufbau in `finishSession()` **und** im „Tagespensum erreicht"-Zweig (Z. 552 — beide Wege ins Done-Panel), Reset in `load()`. DOM per `createElement`/`textContent`.
- **Launcher** (Entscheidung 7) in `renderScopePills` oder direkt danach; reagiert auf Umschalten der Pillen.
- **Smoke Teil B** in `scripts/smoke_lern_text.py`: vor dem Aufdecken kein `Lerntext`-Link im DOM; nach dem Aufdecken Link mit richtiger `href` und `target`; Karte ohne Kontext → kein Link; Bewertung Nochmal → Done-Panel zeigt den Text mit der Überschrift, zweimal Nochmal auf zwei Karten derselben Überschrift → Überschrift einmal; Vertiefen allein → ebenfalls in der Liste; Bewertung Gut → nicht; Launcher-Zeile „Lesen" erscheint mit der angehakten Sammlung und verschwindet beim Abhaken; **0 Requests** beim Aufbau der Nacharbeiten-Liste (Route-Interception wie in `smoke_review_skip.py`). Zweimal laufen lassen.

## Deploy (Mintbox, in dieser Reihenfolge)
1. Prod-DB-Backup nach Rezept (CLAUDE.md *Running*): `~/converter.db.pre-lern-text-2026-10-05`, erwartete Ausgabe `-rw------- oliver oliver` ohne `+`; Kopie nur mit `mode=ro&immutable=1` prüfen.
2. `docker tag converter-app:latest converter-app:pre-lern-text`.
3. `cd ~/CODE/CONVERTER && git pull --ff-only && docker compose up -d --build` (aus dem Projektverzeichnis; die Migration läuft beim Web-Start unter dem `flock`).
4. Verifikation: `docker logs` des Web ohne Migrations-Fehler; Schema per `mode=ro`-Lesung im Container (`PRAGMA table_info(card)` zeigt die zwei Spalten, `collection_documents` existiert); Login 200; Smoke Teil A + B **gegen Prod** mit Wegwerf-User, Aufräumen per `user_id`; Olis Daten unberührt (Zeilenzahlen `card`, `collection`, `conversion` vorher/nachher gleich).
5. Container-Suite nach Rezept (stdin-Stream, `--network none`) — erwartete Zahl im Bericht.

## Stop
Bericht: Smoke Teil B (zwei Läufe, Mac-Container oder Prod) · Deploy-Belege (Backup-Zeile, Image-ID, Schema, Login, Smoke gegen Prod, Zeilenzahlen) · Suite Mac + Container · Commit-Hash. Dann warten.

---

# Phase 4 — Wrap + Brief

1. **Brief ans converter-mcp** `docs/converter_mcp_lern_text_brief.md` — ⚠️ **zuerst die Tool-Liste am lebenden Connector lesen** (Memory `feedback_verify_counterpart_surface_before_briefing`), dann: `create_card`/`update_card` bekommen `context` (Form, Fehler 400/404/409 mit Sätzen, `null` löst); neues Tool für `PUT /api/collections/<id>/documents` (Namensvorschlag `set_collection_documents`, Semantik „ersetzt"); `list_collections` trägt `documents`; `get_card`/`review_state` tragen `context` additiv. Dazu der Hinweis, wie ein Lerntext entsteht (`create_conversion` + `set_collection_documents` + `update_card` mit `context`), und was der Agent prüfen sollte (Überschrift exakt wie im Text, eindeutig). Rückkanal `…_rueckmeldung.md`.
2. **CLAUDE.md:** ein Bullet unter *Architecture Notes* (Lerntexte = Library-Dokumente; die zwei Verknüpfungen; der Sprung über `#h=`; warum keine `id`s; Lösch-Mechanik; iOS additiv; Smoke als achter Browser-Smoke in *Test-Suite-Limit*); Baseline-Zahl in *Key Files*.
3. **BACKLOG.md:** LERN-TEXT schließen; Folge-Items **LERN-TEXT-IOS** (Knopf und Launcher-Zeile in der iOS-App; Felder liegen schon an) und **LERN-NACHARBEITEN-SEITE** (P3: wackelnde Karten mit Kontext über Sessions hinweg — nur wenn Oli nach dem ersten Gebrauch danach fragt). Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md` → exit 1.
4. **STATUS.md**, **Memory** (eine Lehre, falls eine entstanden ist — nicht erzwingen), Mintbox-Clone auf den Wrap-Stand (`git pull --ff-only`, Image unverändert).
5. Bericht mit Abweichungen vom Prompt, benannten Eigenschaften (Überschrift umbenannt → Hinweis statt Sprung; Kontext zeigt immer auf ein Dokument desselben Users) und den drei Dingen, die Oli von Hand macht (Skill-Text in claude.ai — liefert der Master; ersten Lerntext anlegen lassen; drei alte DB-Backups).

## Abnahme (Master)
- Suite grün, Mac und Pin; rote Zahl aus Phase 1 unabhängig reproduziert.
- Prod: ein Wegwerf-Dokument mit Überschrift, eine Wegwerf-Karte mit Kontext, Review-Smoke gegen Prod zeigt Knopf nur nach dem Aufdecken und die Nacharbeiten-Liste; Löschen des Dokuments nullt den Kontext (per `GET /api/cards/<id>`); `/api/collections` ist eine Liste; `markdown_render.py` nicht angefasst; Olis Zeilenzahlen unverändert.
