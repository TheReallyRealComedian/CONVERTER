# Ingestion Contract — `POST /api/ingest/conversion`

**Status**: live since NL1 (2026-06-02). This document is the **interface NL2
(`email-automation`) implements** — the server-to-server push that lands AI
newsletters (and any other allowed conversion type) in the CONVERTER Library.

Direction is **push**: `email-automation` POSTs to CONVERTER. CONVERTER never
pulls. The endpoint is generic — the Bearer token is the trust boundary, and
any `conversion_type` in `ALLOWED_CONVERSION_TYPES` is accepted.

---

## Endpoint

```
POST /api/ingest/conversion
Authorization: Bearer <INGEST_TOKEN>
Content-Type: application/json
```

No session/cookie. No CSRF token (this endpoint is CSRF-exempt — it is the only
one). LAN-only, same as the rest of the app.

## Request body

| Field             | Type        | Required | Notes |
|-------------------|-------------|----------|-------|
| `conversion_type` | string      | **yes**  | Must be in `ALLOWED_CONVERSION_TYPES`. For newsletters: `"ai_newsletter"`. |
| `content`         | string      | **yes**  | The Markdown body. Non-empty. |
| `title`           | string      | no       | Defaults to `"Untitled"`, clipped to 255 chars. |
| `topics`          | string[]    | no       | Mapped to conversion tags (lowercased + trimmed). A non-list value is **ignored** (never an error); non-string / blank / >80-char entries are skipped. |
| `report_date`     | string (ISO)| no       | ISO date or datetime (`"2026-05-30"` or full ISO). Mapped to `created_at`. Unparsable → ignored, `created_at` defaults to now. |
| `source_id`       | string      | no       | Stable idempotency key (see Dedup). Omit and every POST creates a new row. |

### Example

```json
{
  "conversion_type": "ai_newsletter",
  "title": "2026-05-30 - AI Newsletter Analyse",
  "content": "# AI Newsletter\n\n## Themen der Woche\n\n- ...",
  "topics": ["KI-Agenten", "Code-Generierung"],
  "report_date": "2026-05-30",
  "source_id": "notion-2026-05-30-ai-newsletter"
}
```

## Responses

| Code  | When | Body |
|-------|------|------|
| `201` | Created. | The new conversion as `to_dict()` (`id`, `conversion_type`, `title`, `content`, `tag_refs`, `created_at`, …). |
| `200` | Dedup hit — a row with this `source_id` already exists for the target user. **No new row.** | The existing conversion's `to_dict()` plus `"deduped": true`. |
| `400` | Body is not a JSON object, or `content` missing/empty, or `conversion_type` missing / not in `ALLOWED_CONVERSION_TYPES`. | `{"error": "..."}` |
| `401` | `Authorization` header missing, malformed, or token mismatch. | `{"error": "Nicht autorisiert."}` |
| `503` | Endpoint not configured: `INGEST_TOKEN` unset/empty (fail-closed), or no target user resolvable. | `{"error": "..."}` |

A client should treat **both** `201` and `200` as success — `200` means "already
ingested, nothing to do". Retrying a failed POST with the same `source_id` is
safe (idempotent).

## Dedup semantics (`source_id`)

If `source_id` is present, the endpoint checks — scoped to the target user —
whether a conversion already carries that `source_id` in its `metadata_json`.
Hit → `200` + `"deduped": true`, no insert. Miss → insert, storing
`metadata_json = {"ingested": true, "source_id": "<source_id>"}`. No schema
change; `source_id` lives inside the existing `metadata_json` text column.

**Recommendation to NL2**: use a **stable, ideally wildcard-free** `source_id`
— a Notion page ID is ideal (hex + dashes). The matcher neutralises SQL-LIKE
wildcards (`%`, `_`), so ids containing them still dedup correctly, but a clean
opaque id keeps the contract obvious. The same `source_id` must be sent on every
re-push of the same newsletter for idempotency to hold.

## Mappings

- `topics[]` → conversion tags via the shared `conversion_tags` junction
  (`Tag.get_or_create`, lowercased + trimmed, reuses an existing tag row). The
  ingested conversion then shows up under the Library tag-filter `?tag=<topic>`.
- `report_date` → `created_at` (so the Library sorts/dates by the newsletter's
  report date, not the ingestion time).

## Environment

| Var            | Required | Meaning |
|----------------|----------|---------|
| `INGEST_TOKEN` | **yes**  | Shared secret. Unset/empty → endpoint is fail-closed (503). Compared constant-time. Set it in `.env` on the CONVERTER host; rotate by changing it and recreating the container. |
| `INGEST_USER`  | no       | Username to attribute ingested rows to. If set, it must resolve to an existing user (a set-but-missing username is a misconfiguration → 503, never a silent fallback). If unset, the single-user fallback (`User.query.first()`) is used. |

The token is a server secret: never commit it (`.env` is gitignored;
`.env.example` carries only the variable names).
