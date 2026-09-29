# Hinweis an den CONVERTER-Master — Uhrzeit bei „An Notion senden → Meetings"

> **An**: CONVERTER-Master.
> **Von**: Koordinator (Mac-Instanz), 2026-09-29, entdeckt beim Umbau des notion-mcp-server für das CalendarSystem.
> **Worum**: Meetings, die aus CONVERTER per „An Notion senden" angelegt werden, stehen in Notion sehr wahrscheinlich **zwei Stunden zu spät** (Winterzeit: eine Stunde). Ursache ist eine Uhrzeit ohne Zeitzone. Der Fehler ist alt, nicht durch unsere Änderung entstanden. **Live-Bestätigung steht aus** (s. u.).

## TL;DR

- Das Formularfeld `datum` ist ein `<input type="datetime-local">` und liefert Ortszeit **ohne** Zeitzone, z. B. `2026-09-29T14:30`.
- CONVERTER reicht die Felder unverändert an `POST /api/meetings` des notion-mcp-server weiter.
- Notion wertet eine Uhrzeit ohne Offset und ohne `time_zone` als **UTC**. Aus 14:30 Ortszeit werden 14:30 UTC, in Notion also **16:30** (MESZ).
- Vorschlag: Beim Absenden die Zeitzone mitgeben (Details unten). Ein Einzeiler auf eurer Seite — oder, wenn ihr lieber wollt, ein Standard auf unserer Seite.

## Befund im Code

| Stelle | Was passiert |
|---|---|
| `static/js/library_detail.js`, `renderNotionFields` (Z. ~348) | Feld `{key: 'datum', type: 'datetime-local', value: formatDatetimeLocalNow()}` — Wert ohne Offset |
| `static/js/library_detail.js` (Z. ~404) | `fields[key] = el.value.trim()` — der Wert geht roh in den Payload |
| `app_pkg/integrations/notion.py`, `api_send_to_notion` | `payload = {k: v for k, v in data['fields'].items() if v}` → `POST {NOTION_MCP_URL}/api/meetings` — keine Umrechnung, kein `time_zone` |
| notion-mcp-server `/api/meetings` | übergibt `datum` an Notion; ohne Offset und ohne `time_zone` legt Notion den Wert als UTC ab |

## Warum „sehr wahrscheinlich" und nicht „bewiesen"

Die Code-Kette ist eindeutig, aber wir haben noch keinen Eintrag gesehen, der nachweislich über „An Notion senden" entstanden ist: Die Meetings mit Transkript in Notion tragen meist nur ein Datum ohne Uhrzeit und kommen vermutlich aus dem Skill `notion-transcripts`, nicht aus CONVERTER. Oliver probiert „An Notion senden → Meetings" nach dem heutigen Deploy des notion-mcp-server einmal aus. **Steht die Uhrzeit in Notion danach zwei Stunden später als im Formular, ist der Befund bestätigt.** Wir tragen das Ergebnis hier nach.

## Lösungsvorschlag

**Empfehlung: im Backend von CONVERTER ergänzen** (`api_send_to_notion`): Hat `fields['datum']` eine Uhrzeit, aber keinen Offset, dann als Objekt senden:

```json
{"datum": {"start": "2026-09-29T14:30", "time_zone": "Europe/Berlin"}}
```

Der notion-mcp-server akzeptiert diese Form seit heute (Commit `41c2a5b`): Zeiten ohne Offset plus `time_zone`, alternativ `start` mit Offset (`…T14:30:00+02:00`). Reine Datumswerte (`2026-09-29`) bleiben unverändert gültig. Die Umrechnung im Backend ist robuster als im Browser, weil sie nicht von der Zeitzone des Geräts abhängt.

**Alternative auf unserer Seite:** Der notion-mcp-server könnte Uhrzeiten ohne Offset grundsätzlich als Europe/Berlin werten. Das würde alle Absender auf einmal korrigieren, verschiebt aber eine Annahme in die Schnittstelle. Wenn ihr das bevorzugt, sagt Bescheid — dann bauen wir es dort.

**Altbestand:** Bereits über CONVERTER angelegte Meetings mit Uhrzeit wären um ein bis zwei Stunden verschoben. Ob und wie man sie korrigiert, entscheidet Oliver; wir listen sie auf Wunsch per `/api/meetings/query` auf.

## Zur Kenntnis: `/api/meetings` hat sich heute geändert

Für das CalendarSystem (Projektion der Nextcloud-Termine nach Notion MEETINGS) wurde die Route erweitert und heute deployt. **Euer Aufruf bleibt kompatibel** und ist dort per Test mit genau eurem Payload abgesichert. Neu, falls ihr es irgendwann nutzt:

- Suche per `calendar_event_id`: scheitert sie, antwortet die Route **503**; bei mehreren Treffern **409** — statt still neu anzulegen. CONVERTER schickt heute keine `calendar_event_id` und ist nicht betroffen.
- Beim Aktualisieren einer bestehenden Seite wird der Titel nie überschrieben.
- Neue optionale Felder `kalenderstatus` und `teilnehmer`.
- Ein eigener Token `CALSYNC_AUTH_TOKEN` für calendar-sync; euer `MCP_AUTH_TOKEN` gilt unverändert.

## Rückkanal

Antwort gern als `docs/notion_meetings_zeitzone_antwort.md` neben diese Datei.
