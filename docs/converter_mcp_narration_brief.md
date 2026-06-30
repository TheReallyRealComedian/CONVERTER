# Developer-Brief an das converter-mcp-Team — NARRATION (treue Vertonung)

> **An**: converter-mcp-Entwickler (Koordinator-Repo).
> **Von**: CONVERTER-Master, 2026-06-30.
> **Worum**: CONVERTER kann jetzt **Erklär-Dokumente treu vertonen** (Sprints NARR-1B/2/3, **live verifiziert** — `narration 76 → ready`, echtes Audio). Ein Agent (Claude, Skill `erklaerbaer-narration`) schreibt eine **Turn-Liste** und triggert die Generierung; CONVERTER rendert async über Gemini-TTS (Cloud-Pfad) und persistiert das Audio als Library-Element. Dieser Brief sagt, **welche Endpoints der converter-mcp wrappt** + **welche Config nötig ist**.

## TL;DR (bitte zuerst lesen)

- **Drei Endpoints** (eine Write + zwei Reads):
  - **`POST /api/narrations`** — **token-authed** (`Authorization: Bearer <NARRATION_TOKEN>`, **eigener Token**, CSRF-exempt, fail-closed). Legt eine `pending`-Vertonung an + enqueued den Render-Job. → **202**.
  - **`GET /api/narrations/<id>`** — **Session** (`@login_required`): Status-Poll (rekonziliert `pending`→`ready`/`failed`).
  - **`GET /api/narrations/<id>/audio`** — **Session**: streamt das fertige WAV.
- **⚠️ Neuer Token**: die Write-Auth nutzt **`NARRATION_TOKEN`**, **nicht** `CARD_TOKEN` (Begründung: eine Vertonung kostet **GCP-Geld pro Call** → unabhängig revozierbar). Der converter-mcp braucht ihn in seiner `.env` (zusätzlich zu `CARD_TOKEN`), und CONVERTER hat ihn in beiden `.env` (Mac + Mintbox). **Fail-closed**: ohne Token antwortet der Endpoint **503**.
- **Auth-Split wie gehabt**: die **Write** über den Token (wie die Card-/Tag-Writes), die **Reads** über die bestehende Session (wie `list_conversions`/`get_transcript`).

## Die drei Endpoints — Kontrakt

### `POST /api/narrations` — Write (Bearer `NARRATION_TOKEN`, CSRF-exempt)
Body (JSON) — der **Turn-Listen-Kontrakt**:
- `mode` — `"single_speaker"` | `"two_speaker"` (Pflicht).
- `voices` — `{label: gemini_voice}` (Pflicht; deckt alle distinct `turns[].speaker` ab; **max. 2** distinct Speaker; Voices = Gemini-Stimmen wie `Kore`/`Puck`/`Zephyr`).
- `turns` — `[{speaker, text}]` (Pflicht, non-leer; `speaker` ∈ `voices`-Keys, `text` non-blank). Wird **wörtlich** gesprochen.
- `title` — optional (degeneriert/leer → server-seitig aus dem Content abgeleitet).
- `language` — optional (Default `de-DE`); `tts_model` — optional (Default `gemini-2.5-flash-tts`); `style_prompt` — optional (Director's-Notes, leakage-frei serverseitig).
- **Validierung (400)**: über `validate_turns` (mode/Speaker-Zahl/voices-Abdeckung/non-blank) — die Fehler-Message nennt den Verstoß.
- → **202** `{"narration_id": int, "job_id": str, "status": "pending"}`.
- Auth-Fehler: **503** (kein `NARRATION_TOKEN` konfiguriert), **401** (fehlend/falsch).

### `GET /api/narrations/<id>` — Read (Session): Status-Poll
- → **200** die volle Conversion (`to_dict()`), inkl. `metadata`: `narration_status` (`pending`/`ready`/`failed`), `duration_seconds`, `error`, `audio_filename`, `speakers`, `transcript`, `tts_model`.
- Der Read **rekonziliert** den Status (DB-freier Worker → die Web-Seite flippt `pending`→`ready`+Dauer / `failed`+error beim Pollen). Also: **so lange pollen, bis `narration_status` terminal** ist.
- Fremd/fehlend → **404**; nicht-`audio_narration` → 404.

### `GET /api/narrations/<id>/audio` — Read (Session): das WAV
- → **200** `audio/wav` (persistent, kein Delete-on-serve). `narration_status != ready` / Datei fehlt → 404; Traversal → 403.

## Der Flow (für den Wrapper)

```
create_narration(turns, voices, mode, title?, language?, tts_model?, style_prompt?)   # Bearer NARRATION_TOKEN
   → {narration_id, job_id, status: "pending"}
loop: get_narration_status(narration_id)   # Session — bis narration_status ∈ {ready, failed}
   → ready  → das Audio ist da (get_narration_audio / Library)
   → failed → metadata.error lesen + dem Agenten zurückgeben (er korrigiert die Turn-Liste)
```

## Empfehlung fürs converter-mcp

1. **`create_narration`** wrappen (`POST /api/narrations`, Bearer `NARRATION_TOKEN`) — Params `turns`, `voices`, `mode`, optional `title`/`language`/`tts_model`/`style_prompt`. Body als echtes JSON (Listen/Dicts, nicht Strings).
2. **`get_narration_status`** wrappen (`GET /api/narrations/<id>`, Session) — der Poll. Tool-Doc: „so lange pollen, bis `narration_status` ∈ {ready, failed}".
3. **`get_narration_audio`** wrappen (`GET /api/narrations/<id>/audio`, Session) — optional, oder die Library-UI/`list_conversions` zeigt das `audio_narration`-Element eh.
4. **`NARRATION_TOKEN`** in die converter-mcp-`.env` (zusätzlich zu `CARD_TOKEN`) — derselbe Wert wie in CONVERTERs `.env`. Ohne ihn → der Wrapper kriegt 503.
5. **Voice-Katalog**: der Skill (`erklaerbaer-narration`) bringt die Stimmen mit; ein optionales `list_gemini_voices`-Read wäre nett, aber nicht nötig (`Kore`/`Puck`/`Zephyr`/`Charon`/… sind im Skill).

## End-to-end-Beweis (Koordinator-Scope)

CONVERTER-seitig live verifiziert (`narration 76`, echtes Audio, Qualität von Oli bestätigt). Nach dem MCP-Wrap: der **Agent** macht den vollen Loop (Doc → Turn-Liste → `create_narration` → poll → Audio in der Library). Der Skill, der das Authoring treibt, liegt in [docs/narration_skill.md](narration_skill.md); die Delivery-Doktrin in [docs/narration_tag_doctrine.md](narration_tag_doctrine.md).

---

*CONVERTER-Seite: NARRATION fertig + deployed + live. **Kein** Schema/Migration; einziger neuer Knopf = `NARRATION_TOKEN` (env, fail-closed). Auth-Split: Write=Token, Reads=Session. Kontrakt-Muster: [docs/narration_reframe.md](narration_reframe.md). Geschwister-Briefs: [docs/converter_mcp_lern_group_brief.md](converter_mcp_lern_group_brief.md), [docs/converter_mcp_tag_cleanup_brief.md](converter_mcp_tag_cleanup_brief.md).*
