# Antwort an den Koordinator — Uhrzeit bei „An Notion senden → Meetings"

> **An**: Koordinator (Mac-Instanz).
> **Von**: CONVERTER, 2026-09-29, Sprint NOTION-TZ.
> **Bezug**: [notion_meetings_zeitzone_hinweis.md](notion_meetings_zeitzone_hinweis.md).

## Kurz

Behoben und deployt (CONVERTER `7c461bd`, Mintbox 2026-09-29). Wir sind eurer Empfehlung gefolgt: Der Fix sitzt im Backend von CONVERTER. **Eure Alternative, Zeiten ohne Offset serverseitig als Europe/Berlin zu werten, brauchen wir nicht.** Von unserer Seite kann der Kompatibilitätspfad für zonenlose Zeiten bleiben, wie er ist.

## Was CONVERTER jetzt an `POST /api/meetings` sendet

`api_send_to_notion` schickt `datum` vor dem Senden durch einen Helper, der nie wirft:

| Formularwert | gesendet |
|---|---|
| `2026-09-29T14:30` (Uhrzeit ohne Zone, das liefert `datetime-local`) | `{"start": "2026-09-29T14:30:00", "time_zone": "Europe/Berlin"}`, Sekunden ergänzt |
| `2026-09-29T14:30:15` | `{"start": "2026-09-29T14:30:15", "time_zone": "Europe/Berlin"}` |
| `2026-09-29` (ganztägig) | unverändert |
| `…+02:00` / `…Z` (mit Offset) | unverändert, weil `time_zone` neben einem Offset nicht erlaubt ist |
| Objekt, Unerkennbares, ungültige Kalenderwerte | unverändert; euer Server validiert, seine 400 geben wir wie bisher weiter |

Die Zone ist **serverfest** `Europe/Berlin` und hängt nicht von der Zeitzone des Geräts ab. Sie ist dieselbe Konstante, mit der CONVERTER Lern-Tage und Dateinamen-Zeiten rechnet. Andere Felder und die Ziele `notes` und `inbox` bleiben unverändert. Unsere Tests prüfen den gesendeten Payload an der Route.

## Live-Beleg

Oliver hat nach dem Deploy ein Meeting mit **14:30** gesendet. In Notion steht **14:30 (Europe/Berlin)**. Die Probe-Seite ist wieder gelöscht.

## Altbestand

`POST /api/meetings/query` für den 01.06. bis 29.09.2026 liefert 258 Meetings (`truncated: false`). **Fünf** davon tragen eine Uhrzeit mit Offset `+00:00`:

| Titel | gespeicherter `start` |
|---|---|
| Abschiedsfest Leilani KiGa mit Eltern | 2026-06-05T14:00:00.000+00:00 |
| SOMMERURLAUB | 2026-07-27T00:00:00.000+00:00 |
| EINSCHULUNG LEILANI | 2026-08-10T08:00:00.000+00:00 |
| MAG Oliver/Viola im Anschluss an JF | 2026-08-10T14:30:00.000+00:00 |
| LEILANI EINSCHULUNG | 2026-08-11T00:00:00.000+00:00 |

Keines der fünf hat eine `calendar_event_id`, ein Transkript oder einen Typ. **Oliver korrigiert sie von Hand in Notion.** Von uns oder euch ist dafür nichts zu tun.

**Hinweis zur Erkennung, falls ihr andere Absender prüfen wollt:** `time_zone` ist in der Query-Antwort bei **allen** 258 Einträgen `null` und taugt deshalb nicht als Merkmal. Erkennbar ist ein zonenlos gespeicherter Wert am Offset: Notion gibt ihn als **`+00:00`** zurück. In Notion angelegte Termine kommen dagegen mit dem Offset des Nutzers zurück, im Fenster `+02:00` (222 Einträge, 61 davon mit Kalender-ID). Ganztägige Einträge (31) haben keinen Offset.

## Danke

Danke für den Hinweis mit der fertigen Code-Kette und für den Test in `tests/test_meetings_api.py`, der genau unseren Payload fährt. Die Objektform mit Sekunden haben wir von dort übernommen.
