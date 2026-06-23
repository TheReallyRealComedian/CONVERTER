# Brief an den converter-mcp — Tool `list_conversions` ergänzen

> **An**: wer den `converter-mcp` baut/pflegt (Koordinator-Repo).
> **Von**: CONVERTER-Master, 2026-06-22.
> **TL;DR**: Bitte ein neues MCP-Tool `list_conversions` ergänzen, das den **bereits existierenden** CONVERTER-Endpoint `GET /api/conversions` wrappt. **Keine CONVERTER-Änderung nötig** — rein additives Tool-Wrapping, keine neue Auth.

## Das Problem (warum)

Der Agent kann eine Conversion, die **kein Audio-Transkript** ist (z.B. ein Markdown-Deck oder ein Dokument), aktuell **nicht ohne bekannte ID finden**. Der converter-mcp exponiert dafür nur `list_audio_transcripts`, das hart auf `conversion_type = audio_transcription` filtert. Es fehlt ein Tool, das **nach `lifecycle_status`** (z.B. „alles in der Inbox") und/oder **nach beliebigem Typ** listet. Der pragmatische Workaround bisher: ID nennen lassen oder ab der letzten bekannten ID hochzählen — unschön.

## Die gute Nachricht: CONVERTER-Server ist fertig

Der generische Read-Endpoint `GET /api/conversions` existiert seit dem MCP1-Sprint, ist owner-scoped und kann genau das. **Es ist also nichts am CONVERTER-Server zu bauen** — nur das MCP-Tool fehlt, das ihn aufruft. (Der converter-mcp nutzt für `list_audio_transcripts`/`get_transcript` schon dieselbe Session — der neue Aufruf läuft über genau die.)

## Der zu wrappende Endpoint

```
GET /api/conversions                       Auth: Session (wie list_audio_transcripts), owner-scoped
```

**Query-Parameter** (alle optional):

| Param | Bedeutung |
|---|---|
| `status` | exakter `lifecycle_status`-Filter — einer von `inbox` \| `later` \| `archive`. Ungültig → **400**. |
| `type` | `conversion_type`-Filter — einer von `document_to_markdown` \| `audio_transcription` \| `dialogue_formatting` \| `markdown_input` \| `ai_newsletter`. Ungültig → **400**. |
| `exclude_status` | schließt einen `lifecycle_status` aus (z.B. `archive` → die „unarchivierte" Menge). Ungültig → **400**. |
| `limit` | Seitengröße, Default **100**, Werte > 500 werden auf **500 gecappt** (nicht abgelehnt); non-int / < 1 → **400**. |
| `offset` | Default **0**; non-int / < 0 → **400**. |

Sortierung: `created_at` absteigend (neueste zuerst).

**Response** (200):
```json
{
  "items": [
    {
      "id": 42,
      "title": "Mein Markdown-Deck",
      "conversion_type": "markdown_input",
      "lifecycle_status": "inbox",
      "queue_position": null,
      "is_favorite": false,
      "last_read_percent": 0,
      "source_filename": null,
      "source_mimetype": null,
      "source_size_bytes": null,
      "tag_refs": [{"id": 3, "name": "ki"}],
      "metadata": { },
      "content_length": 1234,
      "content_preview": "die ersten ~300 Zeichen …",
      "created_at": "2026-06-20T10:00:00",
      "updated_at": "2026-06-22T09:00:00"
    }
  ],
  "total": 17,
  "limit": 100,
  "offset": 0
}
```
`total` ist die volle Treffer-Zahl **vor** dem limit/offset-Fenster. Die Item-Summary trägt **keinen** vollen `content` (nur `content_length` + 300-Zeichen-`content_preview`) — für den Volltext s. den Detail-Endpoint unten.

## Volltext einer ID (Companion, existiert ebenfalls)

```
GET /api/conversions/<id>                  Auth: Session, owner-scoped → 404 bei fremd/fehlend
```
Liefert die **volle** Conversion (`to_dict()`, inkl. `content`). Workflow also: `list_conversions` zum Finden der ID → dann den Detail-Call (bzw. das bestehende `get_transcript`-Äquivalent) für den Inhalt.

## Vorgeschlagenes Tool

```
list_conversions(status?, type?, exclude_status?, limit?, offset?)
```
- Parameter 1:1 auf die Query-Params mappen.
- Auth: dieselbe Session wie `list_audio_transcripts` (kein Token).
- Rückgabe: das `{items, total, limit, offset}`-Objekt durchreichen.

### Beispiele
- **Markdown-Decks in der Inbox**: `status=inbox&type=markdown_input`
- **Alles in der Inbox** (jeder Typ): `status=inbox`
- **Alles Unarchivierte**: `exclude_status=archive`
- **Nächste Seite**: `limit=50&offset=50`

## Verhältnis zu den bestehenden Tools

- `list_audio_transcripts` ist im Grunde der Spezialfall `type=audio_transcription` + `exclude_status=archive`. `list_conversions` ist die generische Version. Beide können koexistieren; `list_audio_transcripts` muss nicht angefasst werden.
- `get_transcript`/der Detail-Call bleibt der Weg zum Volltext einer einzelnen ID.

## Eine Nuance: `lifecycle_status` vs. die 4 UI-„Orte"

Der Endpoint filtert die **Roh-Spalte** `lifecycle_status` (`inbox`/`later`/`archive`). Die CONVERTER-UI zeigt seit R2-H **vier abgeleitete Orte** (Inbox · Lese-Liste · Bibliothek · Archiv), berechnet aus `lifecycle_status` **+** `queue_position`:

| UI-Ort | = |
|---|---|
| Inbox | `lifecycle_status = inbox` |
| Archiv | `lifecycle_status = archive` |
| Lese-Liste | `lifecycle_status = later` **und** `queue_position` ≠ null |
| Bibliothek | `lifecycle_status = later` **und** `queue_position` = null |

→ `status=inbox` und `status=archive` treffen die UI-Orte direkt. „Lese-Liste"/„Bibliothek" sind **nicht** als einzelner `status`-Filter abfragbar — aber jedes Item trägt `queue_position` in der Antwort, das Tool/der Agent kann also clientseitig splitten, falls nötig. Für den auslösenden Use-Case („alles in der Inbox finden") reicht `status=inbox` exakt.

## Was der Endpoint (noch) NICHT kann

- **Keine Titel-/Volltext-Suche** (`?search=` o.ä.). Ein bestimmtes Deck per Titel finden = listen + clientseitig matchen. Falls das im Alltag nervt, wäre ein server-seitiger `?search=`-Filter am Endpoint **ein eigener kleiner CONVERTER-Sprint** — aktuell nicht eingeplant, sag dem CONVERTER-Master Bescheid, wenn der Bedarf real wird.

---

*CONVERTER-Seite: komplett, kein offener Punkt. Dieser Brief beschreibt reines converter-mcp-Tool-Wrapping (Koordinator-Scope).*
