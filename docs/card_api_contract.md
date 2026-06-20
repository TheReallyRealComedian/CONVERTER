# Card-/SR-API-Kontrakt — für die converter-mcp-Erweiterung

> **Wozu**: Der `converter-mcp` (separates Repo `/Volumes/MintHome/CODE/converter-mcp`, Connector „CONVERTER") soll um die R4-LEARN-Tools erweitert werden. Dies ist der **fixe Kontrakt** der CONVERTER-Endpoints, gegen den der MCP gebaut wird. CONVERTER-Seite ist **fertig + deployed**. Out of scope hier: der MCP-Code selbst (Koordinator).

## Auth — zwei Pfade (wichtig)

Der converter-mcp loggt sich heute für die Reads **per Formular ein** (`CONVERTER_USER`/`CONVERTER_PASSWORD` → Session-Cookie). Das bleibt für alle **Read**-Tools. Für die **Write**-Tools kommt **zusätzlich** ein Bearer-Token oben drauf:

- **Reads** → die bestehende **Session** (wie `get_transcript`/`list_audio_transcripts`). `@login_required`.
- **Writes** → **`Authorization: Bearer <CARD_TOKEN>`** (CSRF-exempt, fail-closed). Der Token steht in beiden `.env` (CONVERTER + converter-mcp) und matcht bereits.

## Zu wrappen (6 Tools)

### Writes (Bearer `CARD_TOKEN`)
**`POST /api/cards`** — Karte anlegen. Body (JSON):
- `type` — `"atomic"` | `"generative"` (Pflicht)
- `highlight_id` — int, optional (Provenienz; wird auf Ownership geprüft, fremd/ungültig → 400)
- `front` / `back` / `cloze_text` / `prompt` — Strings, je nach Typ
- `note`, `source_snapshot`, `source_doc_title` — Strings, optional
- `tags` — Liste von Strings (werden normalisiert)
- **Validierung (400)**: `atomic` braucht (`front` UND `back`) ODER `cloze_text`; `generative` braucht `prompt`.
- Legt die zugehörige Review-Zeile gleich mit an (`due = jetzt`). → **201** + Karten-JSON.
- Auth-Fehler: **503** (kein `CARD_TOKEN` konfiguriert), **401** (fehlend/falsch).

**`PATCH /api/cards/<id>`** — Karte verfeinern/annotieren: dieselben Felder, plus `state` (`"ok"`|`"wackelt"`), `tags` ersetzbar. Fremde Karte → 404.

### Reads (Session)
- **`GET /api/highlights/recent?since=<ISO-8601>&limit=<n>`** — **globaler** Reader über alle Docs: jüngste Markierungen mit `note`, `tags` `[{id,name}]`, Eltern-`{conversion_id, title}`, `created_at`. (`since` optional, `limit` Default 100/Cap 500, Sort neueste zuerst.) *Das ist der „jüngste Markierungen"-Reader für den Recall-Loop.*
- **`GET /api/cards?state=&highlight_id=&limit=&offset=`** — Karten-Summaries (schlank, ohne `back`/Snapshot).
- **`GET /api/cards/<id>`** — volle Karte.
- **`GET /api/review-state`** — fällige Karten (`due <= jetzt`) als volle Karten + `due_count`/`total_count`.

## NICHT wrappen (UI-only, Session)
- **`POST /api/cards/<id>/review`** (Bewerten again/hard/good/easy) und **`POST /api/cards/<id>/annotate`** (Vertiefen/Notiz) — das macht der **User** in der „Lernen"-Oberfläche, nicht der Agent. Nicht als MCP-Tool exponieren.

## End-to-end-Kette (zur Einordnung)
Claude-Agent (Session mit dem CONVERTER-Connector) → `recent_highlights`/`list_cards` lesen → Karteninhalt generieren → `create_card` schreiben (Bearer). Der User wiederholt dann im „Lernen"-Tab (Session, ohne MCP).
