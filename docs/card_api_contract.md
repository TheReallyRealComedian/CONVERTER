# Card-/SR-API-Kontrakt — für die converter-mcp-Erweiterung

> **Wozu**: Der `converter-mcp` (separates Repo `/Volumes/MintHome/CODE/converter-mcp`, Connector „CONVERTER") soll um die R4-LEARN-Tools erweitert werden. Dies ist der **fixe Kontrakt** der CONVERTER-Endpoints, gegen den der MCP gebaut wird. CONVERTER-Seite ist **fertig + deployed**. Out of scope hier: der MCP-Code selbst (Koordinator).

## Auth — zwei Pfade (wichtig)

Der converter-mcp loggt sich heute für die Reads **per Formular ein** (`CONVERTER_USER`/`CONVERTER_PASSWORD` → Session-Cookie). Das bleibt für alle **Read**-Tools. Für die **Write**-Tools kommt **zusätzlich** ein Bearer-Token oben drauf:

- **Reads** → die bestehende **Session** (wie `get_transcript`/`list_audio_transcripts`). `@login_required`.
- **Writes** → **`Authorization: Bearer <CARD_TOKEN>`** (CSRF-exempt, fail-closed). Der Token steht in beiden `.env` (CONVERTER + converter-mcp) und matcht bereits.

## Zu wrappen (8 Tools)

### Writes (Bearer `CARD_TOKEN`)
**`POST /api/cards`** — Karte anlegen. Body (JSON):
- `type` — `"atomic"` | `"generative"` (Pflicht)
- `highlight_id` — int, optional (Provenienz; wird auf Ownership geprüft, fremd/ungültig → 400)
- `front` / `back` / `cloze_text` / `prompt` — Strings, je nach Typ
- `note`, `source_snapshot`, `source_doc_title` — Strings, optional
- `tags` — Liste von Strings (werden normalisiert: **lowercased** + getrimmt, geteiltes Vokabular)
- `collections` — Liste von Strings, optional (LERN-GROUP Achse B): **get_or_create by-name**, **Voll-Ersetzung**, owner-scoped. Normalisierung **case-erhaltend** (Eigennamen wie „Boehringer-Pipeline" — anders als `tags`, die lowercasen): nur Trim + interne Whitespace-Runs kollabiert. Frei anlegbar (max. Agent-Autonomie; Aufräumen = User-UI). `[]` leert die Zuordnung.
- **Validierung (400)**: `atomic` braucht (`front` UND `back`) ODER `cloze_text`; `generative` braucht `prompt`.
- Legt die zugehörige Review-Zeile gleich mit an (`due = jetzt`). → **201** + Karten-JSON (inkl. `tags` + `collections` als `[{id,name}]`).
- Auth-Fehler: **503** (kein `CARD_TOKEN` konfiguriert), **401** (fehlend/falsch).

**`PATCH /api/cards/<id>`** — Karte verfeinern/annotieren: dieselben Felder, plus `state` (`"ok"`|`"wackelt"`), `tags` **und** `collections` ersetzbar (jeweils Voll-Ersetzung; kein Listen-Typ → 400; Key weglassen = unberührt). Fremde Karte → 404.

**`PATCH /api/highlights/<id>/annotate`** — Tags/Notiz auf einem **bestehenden** Highlight setzen/ersetzen/leeren (persistentes Bucket-Tagging; **kein** neues Highlight, **kein** Anker-Edit). Body (JSON):
- `tags` — Liste von Strings, optional: **Voll-Ersetzung** des Tag-Sets (`[]` = alle Tags weg), normalisiert über das **geteilte Vokabular** (`Tag.get_or_create` — dieselben Tag-Rows wie Card-/UI-Tags, kein Parallelsystem). Kein Listen-Typ → 400.
- `note` — String oder `null`, optional: `""` → NULL (Notiz löschen), max 2000 Zeichen (sonst 400).
- **Mind. einer von `tags`/`note` Pflicht** (leerer/keiner → 400 „Nichts zu ändern…").
- `exact`/`prefix`/`suffix` im Body werden **ignoriert** (Anker/Marker über diesen Pfad unveränderbar — der Agent annotiert, bewegt keinen Marker).
- **Ownership**: fremdes **oder** fehlendes Highlight → **404** (nicht 400, nicht 403 — die `<id>` ist die adressierte Ressource, kein Body-Feld).
- Auth-Fehler: **503**/**401** wie die Card-Writes. → **200** + volles Highlight-JSON **inkl. aufgelöster Tags** (`[{id,name}]`).

**`POST /api/tags/parent`** — Tag-Baum bauen (LERN-GROUP Achse A, agent-write; Tool z.B. `set_tag_parent`). Der Agent ordnet ein Thema unter ein anderes ein — **by-name**, **distinct** vom Session-`PATCH /api/tags/<id>` (kein Path-Clash). Body (JSON):
- `tag` — String, Pflicht-non-blank: das einzuordnende Tag. **get_or_create** (legt es an, falls neu).
- `parent` — String **oder** `null`, Pflicht: das Eltern-Tag (**get_or_create**) — `null` **entwurzelt** (`tag` wird Wurzel).
- Beide Namen laufen durch `Tag.get_or_create` → **lowercased** geteiltes Vokabular (dieselben Tag-Rows wie Card-/UI-Tags; „KI" → „ki").
- **Zyklus-Guard**: das Eltern-Tag darf nicht das Tag selbst sein noch in dessen Teilbaum liegen → **400** „Zyklus…" (fängt Selbst-Referenz mit).
- blank `tag`/`parent` → **400**. → **200** + Tag-JSON (inkl. `parent_id`).
- Auth-Fehler: **503**/**401** wie die Card-Writes.
- **Hierarchie lesen**: über das Read-Tool für `GET /api/tags` (`parent_id` + Counts je Tag) — der Agent liest den bestehenden Baum, um konsistent einzuordnen.

### Reads (Session)
- **`GET /api/highlights/recent?since=<ISO-8601>&limit=<n>`** — **globaler** Reader über alle Docs: jüngste Markierungen mit `note`, `tags` `[{id,name}]`, Eltern-`{conversion_id, title}`, `created_at`. (`since` optional, `limit` Default 100/Cap 500, Sort neueste zuerst.) *Das ist der „jüngste Markierungen"-Reader für den Recall-Loop.*
- **`GET /api/cards?state=&highlight_id=&limit=&offset=`** — Karten-Summaries (schlank, ohne `back`/Snapshot).
- **`GET /api/cards/<id>`** — volle Karte.
- **`GET /api/review-state`** — fällige Karten (`due <= jetzt`) als volle Karten + `due_count`/`total_count`.

## Document-Content-Writes (Bearer `CARD_TOKEN`)

> **MCP-DOCWRITE (2026-06-22)**: der converter-mcp kann den **Markdown-Inhalt** einer `Conversion` editieren — **derselbe** Bearer-`CARD_TOKEN` wie die Card-/Highlight-Writes (**kein neuer Token**), **CSRF-exempt**, Ziel-User **server-seitig** (`INGEST_USER`/first(), nie aus dem Request → Anti-Relay). Fremde/fehlende Conversion → **404** (kein Existenz-Leak). Response beider = die **volle aktualisierte Conversion** (`to_dict()`, inkl. `content`) → der Agent verifiziert sein Write direkt. Der Session-Pfad `PUT /api/conversions/<id>` bleibt der UI-/Editor-Weg (CSRF-geschützt, für den MCP unbrauchbar) — **nicht** anfassen. Zwei MCP-Tools, gegen diese Endpoints zu bauen (Koordinator-Scope):

**`update_document`** — **`PATCH /api/conversions/<id>/content`** — Voll-Ersetzung von `content`. Body (JSON):
- `content` — String, **Pflicht + nicht-leer** (Schutz gegen versehentliches Doc-Wipe durch einen Agent-Bug). Leer/fehlend/non-String → **400** „Feld content (nicht-leerer Text) erwartet."
- → **200** + volle Conversion. Auth-Fehler **503** (kein `CARD_TOKEN` konfiguriert), **401** (fehlend/falsch). Fremd/fehlend → **404**.

**`replace_section`** — **`PATCH /api/conversions/<id>/section`** — ein per **Heading** adressierter Abschnitt wird ersetzt (Notion-`replace_section`-analog). Body (JSON):
- `heading` — String, Pflicht-non-blank: der Ziel-Heading-**Text** (führende `#` werden gestrippt; Match **level-agnostisch** auf den Text).
- `content` — String, Pflicht-non-blank: die **neue Section** inkl. eigener Heading (darf die Heading umbenennen; eine Heading wird nicht erzwungen).
- **Abschnitts-Semantik** (ATX-Headings `#`…`######`, **fenced-code-aware** — eine `#`-Zeile in einem Code-Block ist **kein** Heading): die Section = Heading + Body bis zum nächsten Heading mit **gleichem oder höherem Level** (Subsections gehören dazu, bis zum nächsten `#`/gleich-oder-höher).
- **0 Treffer → 404** „Abschnitt nicht gefunden." (unterscheidbar von der Owner-404 „Nicht gefunden."); **>1 Treffer → 409** „Abschnitt mehrdeutig (mehrere Headings gleichen Texts)." (konservativ, nie raten); fehlende/leere Felder → **400**.
- → **200** + volle Conversion. Auth-/Owner-Fehler wie bei `update_document`.

**Nicht v1**: Setext-Headings (`===`/`---`), Multi-Section-/Index-/Zeilen-Range-Adressierung, ETag/Concurrency-Schutz (Single-User; Voll-Ersetzung clobbert bewusst — YAGNI). **CONVERTER-Seite fertig + getestet** (+29 Tests); **kein Schema-Touch, keine Migration, kein neuer Token**.

## NICHT wrappen (UI-only, Session)
- **`POST /api/cards/<id>/review`** (Bewerten again/hard/good/easy) und **`POST /api/cards/<id>/annotate`** (Vertiefen/Notiz) — das macht der **User** in der „Lernen"-Oberfläche, nicht der Agent. Nicht als MCP-Tool exponieren.
- **`DELETE /api/cards/<id>`** (R4-LEARN-FIX) — Karte löschen (`@login_required`, owner-scoped → fremd/fehlend 404, ORM-Cascade nimmt `review` + `card_tags` mit, **200** `{"success": true}`). Der **User** löscht in der „Lernen"-Oberfläche; der Agent **erzeugt/patcht**, löscht nicht. Nicht als MCP-Tool exponieren — falls je Agent-Delete nötig, eigener Token-`DELETE`-Follow-up.

## End-to-end-Kette (zur Einordnung)
Claude-Agent (Session mit dem CONVERTER-Connector) → `recent_highlights`/`list_cards` lesen → Karteninhalt generieren → `create_card` schreiben (Bearer). Der User wiederholt dann im „Lernen"-Tab (Session, ohne MCP).
