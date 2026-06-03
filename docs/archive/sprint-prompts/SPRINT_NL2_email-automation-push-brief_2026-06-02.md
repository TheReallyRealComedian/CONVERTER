# NL2 — AI-Newsletter Push (email-automation → CONVERTER)

> **Self-contained Implementierungs-Brief.** Du arbeitest im **`email-automation`-Repo** (Mintbox: Python 3.12 / FastAPI / async / Postgres / structlog). Ziel: generierte AI-Newsletter zusätzlich zum Notion-Write in die **CONVERTER-Library** pushen. Die CONVERTER-Seite (der Ingestion-Endpoint = der Contract unten) ist **fertig + deployed** — dies implementiert nur die Producer-Seite. Kein Master-Dispatch-Practice nötig; folge `email-automation`s eigener `CLAUDE.md`. Vorschlag: nach Phase 2 kurz stoppen/berichten, dann Phase 3 (Backfill hat eine offene Mechanik-Frage).

---

## Der Contract (CONVERTER-Seite, fix — dagegen implementieren)

```
POST  {converter_ingest_url}      # default: http://host.docker.internal:5656/api/ingest/conversion
Authorization: Bearer <converter_ingest_token>
Content-Type: application/json

{
  "conversion_type": "ai_newsletter",              # PFLICHT, exakt dieser Wert
  "title":           "2026-05-30 - AI Newsletter Analyse",  # PFLICHT
  "content":         "# ... markdown ...",          # PFLICHT — das Original-Markdown des Reports
  "topics":          ["KI-Agenten", "Code-Generierung"],   # optional Liste -> CONVERTER-Tags
  "report_date":     "2026-05-30",                   # optional ISO-Date -> created_at
  "source_id":       "<Notion-Page-ID>"             # optional aber DRINGEND -> idempotenter Dedup
}
```

**Responses**: `201` angelegt · `200 {"deduped": true}` (source_id schon bekannt) · `400` bad body · `401` falscher/fehlender Token · `503` nicht konfiguriert (kein Token / kein User).
**Dedup**: keyed auf `source_id` (CONVERTER legt es in `metadata_json` ab). Re-Push mit gleichem `source_id` = no-op `200`. **→ Pushen ist immer retry-/wiederholungssicher.**

---

## Konventionen (email-automation — einhalten)

- Config: pydantic `BaseSettings` in `src/config.py`; Feature-Flags als `bool`-Felder, default `False`.
- Secrets via Env (`.env`, nie committen).
- Externe Calls: `httpx.AsyncClient` (Vorlage: `src/services/letta.py`, `ollama.py`, `nextcloud.py`).
- Logging: structlog Event-Style (`logger.info("event.name", key=val)`).
- **Jede neue Integration muss per Feature-Flag einzeln abschaltbar sein** (Projekt-Konvention).

---

## Phase 1 — Config + Push-Service

1. **`src/config.py`** (`Settings`): drei Felder ergänzen
   - `converter_push_enabled: bool = False`
   - `converter_ingest_url: str = "http://host.docker.internal:5656/api/ingest/conversion"`
   - `converter_ingest_token: str = ""`
2. **`src/services/converter.py`** (neu) — async Push, spiegelt das httpx-Muster:
   ```python
   import httpx, structlog
   logger = structlog.get_logger()

   class ConverterService:
       def __init__(self, base_url: str, token: str) -> None:
           self.base_url = base_url
           self.token = token

       async def push_newsletter(self, *, title, content, topics, report_date, source_id) -> bool:
           payload = {
               "conversion_type": "ai_newsletter",
               "title": title,
               "content": content,
               "topics": topics or [],
               "report_date": report_date,
               "source_id": source_id,
           }
           try:
               async with httpx.AsyncClient(timeout=15) as client:
                   resp = await client.post(
                       self.base_url,
                       json=payload,
                       headers={"Authorization": f"Bearer {self.token}"},
                   )
               if resp.status_code in (200, 201):
                   logger.info("converter.push.ok", status=resp.status_code,
                               source_id=source_id, deduped=(resp.status_code == 200))
                   return True
               logger.warning("converter.push.failed", status=resp.status_code, source_id=source_id)
               return False
           except Exception as e:                      # NIEMALS raisen — sekundärer Sink
               logger.warning("converter.push.error", error=str(e), source_id=source_id)
               return False
   ```
   **Token nie loggen.** try-except umschließt alles, `push_newsletter` darf **nie** propagieren.
3. **`.env.example`**: `CONVERTER_PUSH_ENABLED=false`, `CONVERTER_INGEST_URL=http://host.docker.internal:5656/api/ingest/conversion`, `CONVERTER_INGEST_TOKEN=` — Werte in `.env` (der Token ist **derselbe** wie CONVERTERs `INGEST_TOKEN`).

---

## Phase 2 — Forward-Hook im Workflow

In **`src/workflows/newsletter.py`**, `execute()`, **Schritt 8 im Notion-Erfolgspfad** (direkt nachdem `page_id = await self.notion.create_newsletter_page(...)` erfolgreich war):

```python
if self.settings.converter_push_enabled and page_id:
    pushed = await self.converter.push_newsletter(
        title=page_title, content=clean_report, topics=topics,
        report_date=today, source_id=page_id,
    )
    result_meta_converter_pushed = pushed   # in result["metadata"] aufnehmen
```

- `ConverterService` wie die anderen Services in `NewsletterWorkflow.__init__` injizieren (siehe wo `notion`/`gemini` durchgereicht werden — den Konstruktions-Punkt in `src/main.py` / der Workflow-Factory mit-anpassen).
- **Nur bei Notion-Erfolg** pushen (`page_id` gesetzt) → `source_id=page_id`. Bei Notion-Fehler (page_id None) **nicht** pushen.
- Push-Resultat in `result["metadata"]` spiegeln (z.B. `"converter_pushed": bool`), analog `notion_page_id`/`notion_error` — für Sichtbarkeit im workflow_run.
- Der `_report_date_override`-Mechanismus (für Backfill historischer Daten) wird durch `today` automatisch mit-respektiert.

---

## Phase 3 — Backfill der bestehenden Notion-Newsletter (one-off)

**Wichtig**: `scripts/backfill_newsletter.py` re-prozessiert *ungesehene Mails* → erzeugt *neue* Pages; es re-exportiert **nicht** die bestehenden ~6 AI_NEWSLETTER-Pages. Die haben SEEN-Mails, der Forward-Hook fängt sie nie. → Separates one-off Script **`scripts/backfill_to_converter.py`**:

1. Notion AI_NEWSLETTER-DB abfragen (`notion_client AsyncClient`, `databases.query`, db-id = `settings.notion_newsletter_db_id`) → Pages listen (page_id, Title, `Report Date`, `Topics`).
2. **Markdown pro Page beschaffen — PHASE-0-ENTSCHEIDUNG (mit Repo-Kontext klären):**
   - **(a) Disk-Reports** `data/newsletter_reports/*.md`: prüfe zuerst die Abdeckung (`ls` + Frontmatter-`date` gegen die Notion-`Report Date` matchen). Wenn vollständig → Markdown von dort lesen (sauberste Quelle, ist das Original). **Bevorzugt, falls Abdeckung passt.**
   - **(b) notion→markdown**: sonst Page-Blocks lesen (`blocks.children.list`, rekursiv) und zu Markdown serialisieren. `src/utils/markdown_to_notion.py` ist die *Inverse* als Referenz — die nötigen Block-Typen sind die der Reports (heading_1/2/3, paragraph, bulleted/numbered_list_item, bold/italic-Annotations).
   - Berichte deine Wahl + warum (Abdeckung der Disk-Reports).
3. Pro Page `ConverterService.push_newsletter(..., source_id=page_id)`. Idempotent → **beliebig oft wiederholbar**. structlog pro Page.

---

## Tests (pytest, `tests/`-Stil + conftest)

- **ConverterService.push_newsletter** (httpx mocken): 201→True; 200→True (deduped); 4xx/5xx/ConnectError→False **und raist nicht**; Token im Header; Body-Shape == Contract.
- **Workflow-Hook**: Flag an + Notion-Erfolg → push mit den richtigen Args (`content=clean_report`, `source_id=page_id`); Flag aus → nicht aufgerufen; Notion-Fehler (page_id None) → nicht aufgerufen; eine Push-Exception **bricht `execute()` nicht**.
- **Backfill**: je nach Phase-0-Wahl — mind. Markdown-Quelle + „push pro Page mit source_id=page_id".

---

## Verify (live)

1. Unit-Tests grün.
2. `.env`: `CONVERTER_INGEST_TOKEN` (= CONVERTERs Wert) + `CONVERTER_PUSH_ENABLED=true`. Newsletter-Workflow triggern → in CONVERTER `/library` erscheint der Newsletter (AI-Newsletter-Badge, Topic-Tags). Re-Run → CONVERTER dedupt (`200`, kein Duplikat).
3. Backfill einmal laufen → die ~6 bestehenden erscheinen.
4. **Netzwerk** (vorab verifiziert — nur Sanity-Check): `host.docker.internal:5656` erreicht CONVERTER aus dem email-automation-Container. Bestätigt: email-automations `docker-compose.yml` hat `extra_hosts: "host.docker.internal:host-gateway"` (wie für Ollama), CONVERTER published `5656:5000` auf `0.0.0.0`. Gegencheck aus dem Container: `curl -s -o /dev/null -w '%{http_code}' -X POST http://host.docker.internal:5656/api/ingest/conversion` → `503`/`401` = Endpoint live (NL1 deployed) · `404` = NL1 noch nicht auf der Mintbox-CONVERTER (→ Voraussetzung 1 unten) · kein Connect = Port/Gateway prüfen.
5. **Robustheit**: CONVERTER-Container stoppen → Newsletter→Notion-Workflow läuft trotzdem sauber durch (Push loggt nur eine Warning). Das ist die Kern-Anforderung.

---

## Out-of-scope

- Den Notion-Write ändern (bleibt 1:1).
- Irgendwas CONVERTER-seitig (fertig — Contract ist fix).
- Andere email-automation-Workflows (Email-Inbox, Tana).
