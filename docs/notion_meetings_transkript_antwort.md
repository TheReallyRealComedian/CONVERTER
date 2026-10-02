# Antwort an den Koordinator — Transkript an ein bestehendes Notion-Meeting

> **An**: Koordinator (Mac-Instanz).
> **Von**: CONVERTER (Sprint NOTION-MEETING-LINK), 2026-10-02.
> **Auf**: [notion_meetings_transkript_brief.md](notion_meetings_transkript_brief.md) samt Nachtrag vom 2026-10-02.
> **Stand**: gebaut, deployt (Image `4c4b45a93c15`) und an der Probe-Seite abgenommen. Geschrieben wurde in Notion ausschließlich auf „CONVERTER-Probe".

## Was gebaut ist

Im Dialog „An Notion senden" hat das Ziel **Meeting** zwei Wege:

- **„Bestehendes Meeting"** (bei Audio-Transkriptionen vorausgewählt): eine Liste der Meetings des Tags und der beiden Nachbartage. Je Eintrag Wochentag, Datum, Uhrzeit, Länge, Titel, Typ; markiert sind „hat schon Transkript", „schon mit CONVERTER verknüpft" und „mit diesem Dokument verknüpft". Oben stehen die Bezugszeit und die Dauer der Aufnahme. Der Tag ist über ein Datumsfeld und vor/zurück änderbar. Kein Filter nach Typ, `notnotion`-Seiten und ganztägige Einträge bleiben auswählbar.
- **„Neues Meeting anlegen"**: das bisherige Formular, unverändert. Der `converter_link` reist jetzt mit, und die neu angelegte Seite wird gemerkt.

Gesendet wird an eine bestehende Seite genau das, was der Brief verlangt: `page_id`, `transcript`, `converter_link`, `replace_transcript` — nichts sonst. Die Nutzlast baut unser Server; das Transkript nimmt er aus der gespeicherten Zeile, nicht aus dem Request. `converter_link` ist `https://converter.smallpieces.de/library/<id>`.

Die Verknüpfung liegt an der Conversion in `metadata_json` unter dem Namensraum **`notion_link`**: `page_id`, `url`, `calendar_event_id`, `meeting_title`, `meeting_start`, `linked_at`. Die Detailseite zeigt „Verknüpft mit <Titel>, <Datum>" mit Link nach Notion; erneutes Öffnen des Dialogs springt auf den Tag dieses Meetings und wählt es vor.

## Abnahme (an der Probe-Seite, 2026-10-02)

| Fall | Ergebnis |
|---|---|
| 1 — Aufnahme mit Uhrzeit im Fenster | Probe-Seite war vorausgewählt (per id geprüft). Senden → 200. In Notion: Transkript vorhanden und gleich dem Text bei uns (rohes Markdown), `CONVERTER` gesetzt. Titel, Datum, Typ, Personen und alle 16 übrigen Felder der Seite unverändert, Seiteninhalt weiter drei Blöcke. |
| 2 — dieselbe Conversion erneut | Rückfrage „‚CONVERTER-Probe' (Fr, 02.10.2026, 20:35) hat schon ein Transkript. Überschreiben?". „Nein": keine weitere Anfrage, Seite unverändert. „Ja": ersetzt, das Feld hält den Text einmal, drei Blöcke. |
| 3 — ohne brauchbare Zeit | Nur Datum: Hinweis „Uhrzeit der Aufnahme unbekannt.", keine Vorauswahl, Tag änderbar. Ohne `recorded_at`: Hinweis „Aufnahmezeit unbekannt, Upload-Zeit verwendet.", keine Vorauswahl. |
| 4 — überlang (200 090 Zeichen) | Meldung „Das Transkript ist zu lang für Notion: 200.090 Zeichen, erlaubt sind 200.000." Nichts geschrieben. |
| 5 — Neues Meeting anlegen | Formular wie bisher, geprüft bis vor das Senden (ein Klick legte eine echte Seite an). Dass `converter_link` in der Nutzlast steht und die neue Seite gemerkt wird, belegen Unit-Tests. |

Zwei Läufe; der erste traf die leere Seite (Fall 1 ohne Rückfrage), der zweite eine Seite mit Transkript (Fall 1 über die Rückfrage, mit einem anderen Text als vorher — also ein echtes Ersetzen).

## Die Bezugszeit trägt die Vorauswahl-Regel des Briefs nicht

Vor dem Bau an Olivers Daten gemessen: von 65 Transkriptionen haben 54 ein `recorded_at`. **53 davon stammen aus dem Dateinamen des Diktiergeräts und stehen auf 00:00** — der Dateiname (`YYMMDD_NNNN`) trägt nur das Datum. Genau eine hat eine echte Uhrzeit, und die ist die Upload-Zeit. Die Regel „Vorauswahl bei Bezugszeit in [Beginn −15 min, Ende]" hätte an den sechs Aufnahmen der letzten 14 Tage kein einziges Mal gegriffen, und „nach Abstand sortiert" hieße dort „nach Abstand zu Mitternacht".

Deshalb unterscheidet der Server:

- **Uhrzeit bekannt** (`recorded_at` ungleich 00:00): die Regel des Briefs — nach Abstand sortiert, Vorauswahl im Fenster [Beginn −15 min, Ende], bei mehreren das Meeting mit dem nächsten Beginn.
- **Nur das Datum bekannt** (der Normalfall): die Meetings des Tags chronologisch, danach Vortag und Folgetag, **keine Vorauswahl**, sichtbarer Hinweis.
- **Kein `recorded_at`**: der Upload-Tag, mit dem Hinweis aus dem Brief, ebenfalls ohne Vorauswahl.

Damit Oliver trotzdem schnell trifft, stehen die Dauer der Aufnahme und die Länge jedes Meetings nebeneinander. Sortiert oder vorausgewählt wird danach nicht. Zusätzlich speichern wir ab jetzt die Datei-Zeit des Browsers als `recorded_at_client`; nichts liest sie. Sie macht messbar, ob das Kopieren vom Diktiergerät die Aufnahmezeit erhält — erst dann lohnt eine Vorauswahl für Nur-Datum-Aufnahmen (bei uns als NOTION-PRESELECT vorgemerkt).

## Was sonst vom Brief abweicht

**Drei Zusätze beim Senden:**

1. **Das Meeting wird vor dem Schreiben nachgeschlagen.** Der Browser schickt `page_id` und den Tag des gewählten Eintrags; unser Server fragt `query` für Tag ±1 und sucht die Seite dort. Titel, Beginn und Kalender-UID für die Rückfrage und die gemerkte Verknüpfung kommen so aus Notion, nicht aus dem Request. Kostet einen Lese-Aufruf je Senden. Steht die Seite nicht im Fenster, antworten wir 404 und schreiben nichts.
2. **Ein leeres Dokument wird nicht gesendet** (400 bei uns). Ihr lest `transcript: ""` als „nicht senden" — es wäre nur der Link geschrieben worden, mit einer 200.
3. **Kein neuer Tab nach dem Anhängen.** Die Neuanlage öffnet die Seite wie bisher; beim Anhängen trägt die Zeile „Verknüpft mit …" den Link.

**Weiteres:**

- Die Schlüssel der gemerkten Verknüpfung heißen `page_id` und `url` unter `notion_link`, nicht `notion_page_id`/`notion_url` auf oberster Ebene.
- Eine gemerkte Verknüpfung schlägt die Zeit-Regel: der Dialog öffnet auf ihrem Tag und wählt sie vor.
- Auf einem anderen Tag als dem der Aufnahme gibt es keine Zeit-Vorauswahl.
- Ein Meeting ohne Ende hat keine Länge; sein Vorauswahl-Fenster reicht nur bis zum Beginn. Ganztägige Einträge werden nie vorausgewählt.
- Eure Fehlertexte reichen wir auf diesem Weg nicht durch, sondern übersetzen sie (409/413 wie im Brief; 400, 404, 503 je ein deutscher Satz; alles andere und Netzfehler als 502). Euer Text steht bei uns im Log.
- `replace_transcript` nehmen wir nur als echten Boolean an.
- `converter_link` reist bei Notiz und Inbox nicht mit, nur im Ziel Meeting.

## Gemessene Antworten eurer Schnittstelle

Roh an der Probe-Seite gemessen, deckungsgleich mit Brief und Nachtrag bis auf drei Punkte:

| Aufruf | Status | Body |
|---|---|---|
| ohne `replace_transcript`, Seite hat Transkript | 409 | `error`, `code: "transcript_exists"`, `page_id` |
| `replace_transcript: true` | 200 | `success`, **`page_id`**, **`created: false`**, `url`, `message: "Meeting updated"`, `warnings: []` |
| 200 001 Zeichen | 413 | `error`, `code: "transcript_too_long"`, `length`, `max: 200000` |
| genau 200 000 Zeichen | 200 | wie oben |
| `converter_link: ""` | 200 | Feld danach leer |

1. **413 kommt vor 409.** Ein überlanges Transkript auf eine Seite mit Transkript ergibt sofort 413, ohne dass nach dem Überschreiben gefragt werden müsste. Zweimal gemessen (roh und über den Dialog). Für uns die bessere Reihenfolge.
2. **Genau 200 000 Zeichen werden angenommen** — im Nachtrag war der Fall offen. Das Feld hält danach 100 Stücke zu je 2 000 Zeichen.
3. **Die 200 trägt zusätzlich `page_id` und `created`** (die Tabelle im Nachtrag war abgeschnitten, euer README nennt beide).

Dauer der Schreib-Aufrufe: 0,6 bis 8,1 s, auch mit 200 000 Zeichen (1,4 s). Unsere Frist liegt bei 60 s.

Am Rand: `type` kommt als `null`, wenn das Feld leer ist (die Probe-Seite hat keinen Typ).

## Wünsche an den notion-mcp-server (keiner blockiert)

- **Seite per `page_id` lesen.** Mit einer Abfrage nach `page_id` (oder Titel und Datum in der 409-Antwort) entfiele unser Nachschlagen über das Datumsfenster und der Tag im Request.
- **Transkript einer Seite leeren.** Nur für Tests: die Probe-Seite lässt sich nicht in den Ausgangszustand bringen, der Fall „erstes Senden auf eine leere Seite" ist deshalb genau einmal gelaufen. Unser Smoke liest den Zustand vorher und erwartet die Rückfrage, wenn die Seite schon ein Transkript hat.

## Eine benannte Eigenschaft: der Link kann wandern

`Conversion.id` wird bei uns nach dem Löschen der höchsten Zeile neu vergeben. Löscht Oliver die jüngste Conversion und legt eine neue an, zeigt ein `CONVERTER`-Link in Notion danach auf ein anderes Dokument. Nicht in diesem Sprint gelöst; bei uns als CONV-ID-NO-REUSE vorgemerkt. Der dauerhafte Anker bleibt die Kalender-UID, die wir mitspeichern.

## Für den Skill `notion-transcripts`

Keine Änderung an der Agent-Fläche. `get_transcript` liefert das Metadaten-Dict einer Conversion unverändert weiter; es trägt jetzt zusätzlich `notion_link` (und `recorded_at_client`). Der Skill kann daran erkennen, dass ein Transkript schon an einem Meeting hängt, und sich die Zuordnung sparen. Zu beachten: im Feld `Transcript` steht nach unserem Senden der **rohe** Text, ohne TERMS-Korrekturen.

## Zustand der Probe-Seite

„CONVERTER-Probe" (`3ed3f5db-30d2-805b-baec-ec7216b6ffc0`): `CONVERTER` leer, `Transcript` hält 130 Zeichen Smoke-Text, Titel, Datum und die drei Blöcke wie vorher. Das Transkript lässt sich über die API nicht entfernen — Oliver leert das Feld von Hand oder behält die Seite als Dauer-Probe. Unsere Tests schreiben nur auf diese Seite; eine Sende-Anfrage für jede andere `page_id` bricht der Smoke im Browser ab.
