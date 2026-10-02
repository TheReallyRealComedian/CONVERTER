# Auftrag an den CONVERTER-Master — Transkript an ein bestehendes Notion-Meeting hängen

> **An**: CONVERTER-Master.
> **Von**: Koordinator (Mac-Instanz), 2026-10-01, abgestimmt mit Oliver.
> **Worum**: Im Dialog „An Notion senden → Meetings" soll Oliver ein **bestehendes** Meeting aus Notion auswählen können. Beim Senden landet das Transkript im Feld `Transcript` dieser Seite, statt dass wie heute immer eine neue Seite entsteht. Der notion-mcp-server ist dafür vorbereitet (Commit `3e7d5cb`, Deploy folgt, s. u.). Die Arbeit liegt fast ganz bei euch.

## Hintergrund

Seit dem 29.09. projiziert calendar-sync die Termine aus dem Nextcloud-Kalender „Arbeit" nach Notion MEETINGS: Für jedes Meeting im Horizont (heute −14 bis +7 Tage) gibt es eine Seite, verknüpft über `Calendar Event ID` = Nextcloud-UID. Die Transkripte kommen bei euch rein, zu 99 % als Datei über die Audio-Route. Bisher fehlt die Brücke dazwischen. Das ist der erste Schritt von Phase 3 im CalendarSystem-Backlog („CONVERTER ↔ Kalender").

## Olivers Entscheidungen (2026-10-01)

| | Frage | Entscheidung |
|---|---|---|
| A | Seite hat schon ein Transkript | **nach Rückfrage überschreiben** |
| B | Umfang | **nur das Transkript** — keine Zusammenfassung, keine TERMS-Korrekturen. Das macht weiter der Skill `notion-transcripts` |
| C | Rückverweis | **eigenes URL-Feld `CONVERTER`** in MEETINGS (Oliver legt es an) |
| D | Meeting nicht in Notion (ad hoc) | „Neues Meeting anlegen" bleibt wie heute |

## Was schon da ist

**Bei euch:** `recorded_at` (aus dem Dateinamen oder vom Client, mit `recorded_at_source`) und `duration_seconds` stehen in `metadata_json` jeder Audio-Conversion (`app_pkg/audio.py`). `api_send_to_notion` reicht heute für `meetings` die Formularfelder plus `transcript` an `POST /api/meetings` durch, mit euren `MCP_AUTH_TOKEN`.

**Beim notion-mcp-server** (beide Routen akzeptieren euren `MCP_AUTH_TOKEN`):

`POST /api/meetings/query` — nur lesend, Kandidaten:
```json
{"date_from": "2026-09-28", "date_to": "2026-09-30"}
```
liefert alle MEETINGS, deren `Datum` in diesem Zeitraum beginnt (maßgeblich ist der Kalendertag des gespeicherten Beginns, praktisch Ortszeit; Grenzen inklusiv), aufsteigend sortiert, ohne Seiten im Papierkorb:
```json
{"count": 1, "truncated": false, "meetings": [
  {"page_id": "…", "url": "https://www.notion.so/…", "title": "JF Patrick & Oliver",
   "datum": {"start": "2026-09-29T14:30:00.000+02:00", "end": "2026-09-29T15:00:00.000+02:00", "time_zone": null},
   "calendar_event_id": "…", "type": "Meeting", "meeting_link": "…",
   "has_transcript": false, "people_count": 2, "people_count_capped": false,
   "notnotion": false, "converter_link": null}
]}
```
Der Transkripttext wird nie ausgeliefert, nur `has_transcript`. Ganztägige Einträge haben `start` ohne Uhrzeit.

`POST /api/meetings` — **neu ab `3e7d5cb`**, Transkript an bestehende Seite:
```json
{"page_id": "…", "transcript": "…", "converter_link": "https://…/library/123", "replace_transcript": false}
```
- **Kein `title` nötig**, wenn `page_id` gesetzt ist. Bitte auch **kein** `datum`, `type`, `people`, `project`, `summary` mitschicken. Die würden die Seite überschreiben, und die Seite gehört dem Kalender bzw. Oliver.
- Der Server prüft vorher und schreibt sonst **nichts**:

| Status | `code` | Bedeutung |
|---|---|---|
| 200 | — | geschrieben (`created: false`) |
| 409 | `transcript_exists` | Seite hat schon ein Transkript → Rückfrage, dann mit `replace_transcript: true` erneut senden |
| 413 | `transcript_too_long` | mehr als 200.000 Zeichen (Notion-Grenze, etwa 3–4 Stunden Gespräch); Antwort enthält `length` und `max` |
| 400 | — | Seite gehört nicht zu MEETINGS, ungültige Felder, oder Feld `CONVERTER` fehlt noch in der Registry |
| 404 | — | Seite gelöscht oder im Papierkorb |
| 503 | — | Seite konnte nicht geprüft werden — später erneut |

- Updates hängen **nie** Body-Blöcke an (vorher kam ein Markdown-Transkript bei jedem erneuten Senden ein weiteres Mal in den Seiteninhalt). Das Feld `Transcript` ist die einzige Kopie. Die Neuanlage ist unverändert.
- Die volle Schnittstelle steht im README des notion-mcp-server, Abschnitt `POST /api/meetings`.

## Was gebaut werden soll

1. **Zwei Wege im Dialog** (Ziel Meetings): „Bestehendes Meeting" und „Neues Meeting anlegen" (das heutige Formular, unverändert). Für Audio-Conversions ist „Bestehendes Meeting" vorausgewählt, für andere Conversion-Typen bleibt die Neuanlage der Standard.

2. **Auswahlliste:**
   - Bezugszeit ist `recorded_at`; fehlt sie, `created_at` mit sichtbarem Hinweis („Aufnahmezeit unbekannt, Upload-Zeit verwendet").
   - Kandidaten: `query` für Bezugstag −1 bis +1 (Ortszeit Europe/Berlin), sortiert nach Abstand zwischen Meetingbeginn und Bezugszeit.
   - **Vorauswahl** nur, wenn die Bezugszeit zwischen Meetingbeginn −15 Minuten und Meetingende liegt (bei mehreren: das mit dem nächsten Beginn). Sonst keine Vorauswahl.
   - Anzeige je Eintrag: Wochentag, Datum, Uhrzeit (Ortszeit), Titel, Typ. Deutlich markiert: „hat schon Transkript" und „schon mit CONVERTER verknüpft" (`converter_link` gesetzt).
   - Der Tag ist änderbar (z. B. Datumsfeld oder vor/zurück), für Aufnahmen ohne oder mit falscher Zeit.
   - Nicht nach Typ filtern. `notnotion`-Seiten dürfen auswählbar bleiben.

3. **Senden** über euer Backend (Token bleibt serverseitig, `get_owned_conversion` wie bisher): `page_id`, `transcript` (derselbe Inhalt wie heute, `content-source`), `converter_link`.
   - **409 `transcript_exists`** → Rückfrage: „‚<Titel>' (<Datum>) hat schon ein Transkript. Überschreiben?" Bei Ja erneut mit `replace_transcript: true`, bei Nein nichts.
   - **413** → klare Meldung mit Länge und Grenze, nichts gesendet.
   - übrige Fehler → Meldung wie heute.

4. **`converter_link`** = absolute URL der Library-Seite der Conversion (`/library/<id>`) mit der öffentlichen Basis-URL von CONVERTER. Bitte auch beim Weg „Neues Meeting anlegen" mitschicken.

5. **Verknüpfung merken** an der Conversion, z. B. in `metadata_json`: `notion_page_id`, `notion_url`, `calendar_event_id`, `meeting_title` und `meeting_start` (Momentaufnahme), `linked_at`. Bitte auch nach einer Neuanlage (die `page_id` steht in der Antwort).
   - Die Detailseite zeigt dann „verknüpft mit <Titel>, <Datum>" mit Link nach Notion.
   - Erneutes Öffnen des Dialogs wählt dieses Meeting vor; ein Wechsel auf ein anderes bleibt möglich.
   - **Warum die Kalender-UID mitgespeichert wird:** Notion soll mittelfristig abgelöst werden. Die Nextcloud-UID ist der dauerhafte Anker; die Notion-Seite ist nur die heutige Ablage.

## Nicht in diesem Auftrag

- Zusammenfassung, Tasks, TERMS-Korrekturen (bleiben beim Skill `notion-transcripts`)
- Ad-hoc-Termine im Kalender „Arbeit Ad-hoc" anlegen (spätere Phase 3)
- direkter Zugriff auf den Nextcloud-Kalender — Kandidaten kommen vorerst aus Notion
- Änderungen am notion-mcp-server: Falls euch etwas fehlt, bitte als Wunsch in der Antwort, wir bauen es dort

## Reihenfolge und Abhängigkeiten

1. Oliver legt in Notion MEETINGS das Feld **`CONVERTER`** vom Typ **URL** an.
2. Koordinator frischt die Registry auf und deployt den notion-mcp-server (`3e7d5cb`) — Bescheid kommt als Nachtrag hier.
3. Ihr baut und testet gegen die laufende Schnittstelle. Bis Schritt 2 erledigt ist, liefert `converter_link` noch 400; alles andere geht schon nach dem Deploy.

## Abnahme (End-to-end, bitte als kurze Rückmeldung)

1. Audio-Conversion mit `recorded_at` → passendes Meeting vorausgewählt → Senden → Transkript und CONVERTER-Link stehen in Notion, Titel/Datum/Typ/Personen unverändert.
2. Dieselbe Conversion erneut senden → Rückfrage → „Ja" → Transkript ersetzt, **kein** doppelter Inhalt auf der Seite. „Nein" → nichts geändert.
3. Conversion ohne `recorded_at` → Hinweis auf Upload-Zeit, keine falsche Vorauswahl, Tag änderbar.
4. Überlanges Transkript → klare Meldung, nichts geschrieben.
5. Weg „Neues Meeting anlegen" → wie bisher, zusätzlich mit CONVERTER-Link und gemerkter Verknüpfung.

Bitte Fall 1 und 2 an einer **Probe-Seite** testen, nicht an einem echten Meeting mit Inhalt. Oliver kann dafür ein Testmeeting in Notion anlegen.

## Rückkanal

Antwort gern als `docs/notion_meetings_transkript_antwort.md` neben diese Datei.

---

## Nachtrag 2026-10-02 — Schritte 1 und 2 sind erledigt (eingetragen vom CONVERTER-Master)

Der Koordinator hat seine Rückmeldung im Chat an Oli gegeben, nicht als Datei (im CONVERTER-Repo lief parallel ein Sub-Thread). Hier steht, was davon für den Bau zählt, mit der Gegenmessung des Masters.

**Schritt 1 (Oli):** Das URL-Feld `CONVERTER` ist in MEETINGS angelegt. Die Probe-Seite heißt „CONVERTER-Probe" (page id `3ed3f5db-30d2-805b-baec-ec7216b6ffc0`, Datum 2026-10-02 20:35–21:35); `Transcript` und `CONVERTER` sind leer.

**Schritt 2 (Koordinator):** notion-mcp-server `3e7d5cb` ist deployt (Container angelegt 2026-10-02 20:53:28 MESZ, healthy), die Registry für MEETINGS aufgefrischt (hinzugekommen ist nur `CONVERTER`). Der laufende Server merkt sich die Registry bis zu fünf Minuten.

**Vom Koordinator an der Probe-Seite gemessen** (mit dem `MCP_AUTH_TOKEN`; die Tabelle kam im Chat teils abgeschnitten an — die genaue Form der Antworten misst der Sprint selbst nach):

| Aufruf | Status | Antwort |
|---|---|---|
| `page_id` + `transcript` + `converter_link` | 200 | `success: true`, `url`, `message: "Meeting updated"`, `warnings: []` |
| derselbe Aufruf noch einmal | 409 | `error`, `code: "transcript_exists"`, `page_id` |
| mit `replace_transcript: true` | 200 | Transkript ersetzt, nicht angehängt |
| Transkript mit 200 001 Zeichen | 413 | `error`, `code`, `length`, `max: 200000`; nichts geschrieben |
| `converter_link`, solange das Feld der Registry fehlt | 400 | Fehlertext mit dem Auffrisch-Befehl; nichts geschrieben |

Ein Update fasst nur `Transcript` und `CONVERTER` an: Titel, Datum, Type, Kalenderstatus, PEOPLE, PROJECT, Summary, Calendar Event ID und alle übrigen Felder hatten durchgehend dieselbe Prüfsumme, der Seiteninhalt blieb bei denselben drei Blöcken.

**Was vom Brief abweicht:**
- Die Antwort von `query` trägt zusätzlich `date_from`, `date_to` und `only_notnotion`.
- Ein leeres `calendar_event_id` kommt als `""`, nicht als `null`.
- Die Seiten-URL hat die Form `https://app.notion.com/p/…`.
- `transcript: ""` wird als „nicht senden" behandelt, auch mit `replace_transcript: true` — ein Transkript lässt sich über diese API nicht entfernen. `converter_link: ""` leert das Feld.
- Markdown landet roh im Feld `Transcript`.
- Die Grenze ist mit 200 001 Zeichen gemessen, der Fall von genau 200 000 nicht.
- `converter_link` nutzt der Server nirgends als Schlüssel (geschrieben mit `.strip()`, in den Antworten zurückgelesen) — die Form des Links ist allein CONVERTERs Sache.

**Gegenmessung des Masters** (2026-10-02, aus dem CONVERTER-Web-Container, nur lesend): `POST /api/meetings/query` für den Tag → 200; Top-Level `count`, `date_from`, `date_to`, `meetings`, `only_notnotion`, `truncated`; je Eintrag `calendar_event_id`, `converter_link`, `datum`, `has_transcript`, `meeting_link`, `notnotion`, `page_id`, `people_count`, `people_count_capped`, `title`, `type`, `url`. Die Probe-Seite: `has_transcript: false`, `converter_link: null`, `calendar_event_id: ""`, `datum.start` mit Offset `+02:00`, `time_zone: null`.

Hinweis an Oli aus der Rückmeldung, nicht CONVERTERs Sache: der Checkout des notion-mcp-server ist sieben Commits vor `origin`, darunter der deployte.
