# Developer-Brief an den CONVERTER-iOS-Agenten — IOS-TRANSCRIBE-ROUTE (Transkription als Auftrag)

> **An**: CONVERTER_iOS-Entwickler (`~/CODE/CONVERTER_iOS`).
> **Von**: CONVERTER-Master, 2026-10-09.
> **Worum**: Die App ruft `POST /transcribe-audio-file`. Diese Route gibt es seit dem 2026-08-22 nicht mehr (SYNC-FREEZE P3): die Datei-Transkription ist ein **Hintergrund-Auftrag** mit Pollen. Der Erfassen-Flow der installierten App läuft seitdem ins Leere (in den nginx-Logs 18.09.–02.10. kein Aufruf bei 8 536 iOS-Requests — Oli nutzt ihn gerade nicht, deshalb latent, nicht akut).
> **Umfang**: nur der Transportweg. Aufnahme, Import, Share-Extension und die Ergebniskarte bleiben.

## TL;DR

- **Submit** `POST /api/transcriptions` (multipart) → **202** `{id, status: "pending", job_id}`.
- **Pollen** `GET /api/transcriptions/<id>` bis `status` `ready` oder `failed`.
- **Die Auftragszeile ist schon die Archiv-Zeile.** Kein zweites `POST /api/conversions` mehr; „Speichern" heißt verschieben: `POST /api/conversions/<id>/place {"place": "inbox"}`.
- Bearer wie überall; CSRF entfällt bei Bearer-Präsenz.

## 1. Submit

```
POST /api/transcriptions
Authorization: Bearer <token>
Content-Type: multipart/form-data

audio_file   = <Datei>            (Pflicht; Feldname exakt so)
language     = de | en            (Default de)
recorded_at  = <ms seit Epoche>   (optional, s. §4)
```

- **202** `{"id": 269, "status": "pending", "job_id": "<uuid>"}` — die Zeile existiert ab jetzt (`audio_transcription`, Ort **Archiv**, Titel = Dateiname-Stamm).
- **200** mit der vollen Antwort aus §2 plus `"deduped": true` — dieselbe Datei (sha256) in derselben Sprache lief schon oder läuft noch; es gibt keinen zweiten Auftrag. Die Antwort kann bereits `ready` sein.
- **400** `{"error": …}`: kein Feld `audio_file` · leere Datei · nicht unterstütztes Format („Dieses Dateiformat wird nicht unterstützt. Erlaubt: MP3, WAV, M4A, OGG, FLAC, WEBM.") · über 500 MB.
- **503** „Auftrag konnte nicht eingereiht werden. Bitte erneut versuchen." — Redis nicht erreichbar; keine Zeile, keine Datei, einfach erneut senden.
- **503** mit Dienst-Text, wenn Deepgram serverseitig nicht konfiguriert ist (`require_service`).

Timeout: der Upload ist der teure Teil (bis 500 MB); die Antwort kommt, sobald die Datei liegt und der Auftrag eingereiht ist — nicht erst nach der Transkription.

## 2. Pollen

```
GET /api/transcriptions/<id>
```

```json
{
  "id": 269,
  "status": "pending" | "ready" | "failed",
  "title": "260712_0042",
  "transcript": "<Markdown>" | null,
  "metadata": {
    "language": "de",
    "file_size_mb": 2.2,
    "duration_seconds": 61.4,
    "transcript_length": 1234 | null,
    "recorded_at": "<ISO>" | null,
    "recorded_at_source": "filename" | "client" | null
  },
  "error": "<Text>" | null,
  "source": {"filename": "…", "format": "m4a", "size_bytes": 2310144},
  "lifecycle_status": "archive"
}
```

- `transcript` ist nur bei `ready` gefüllt (Sprecher-Blöcke `**Sprecher N:**` bei mehreren Sprechern, sonst Fließtext — unverändert zum alten Weg).
- `failed` trägt den Grund in `error` (Traceback-Ende); eine Neueinreichung derselben Datei ist der Retry-Weg (`failed` dedupt nicht).
- Kadenz wie das Web: alle **2 s**, nach 60 s alle **5 s**. Der Server rekonziliert beim Pollen — ohne Poll bleibt die Zeile `pending`, bis jemand sie liest (auch die Detailseite im Web tut das).
- Ein Tab-/App-Wechsel bricht nichts ab: der Worker rechnet fertig. Wer die `id` behält, kann später weiterpollen (nicht Pflicht in diesem Zuschnitt — benannt).
- **404**: fremde oder fehlende Zeile.

## 3. Speichern

- Die Zeile liegt im **Archiv**. „Speichern" = `POST /api/conversions/<id>/place {"place": "inbox"}` → 200 mit dem Element.
- Geänderter Titel: `PUT /api/conversions/<id> {"title": "…"}` (Session-/Bearer-Pfad, 200).
- „Verwerfen" nach dem Ergebnis: nichts tun — die Zeile bleibt als Archiv-Eintrag (so verhält sich das Web). Ein Löschen über die App ist nicht Teil dieses Briefs.
- `POST /api/conversions` wird auf diesem Weg **nicht** mehr gerufen; das alte Paar `transcribe` + `createConversion` fällt.

## 4. `recorded_at` — bitte mitschicken

Das Web schickt `file.lastModified` (Millisekunden seit Epoche) als `recorded_at`; der Server speichert es als `recorded_at_client` **neben** dem Gewinner (Dateiname-Datum schlägt es, sonst gilt der Client-Wert als Aufnahmezeit, Quelle `client`). Für die App ist der **Aufnahmebeginn** (eigene Aufnahme) bzw. das Erstelldatum der Datei (Import, Share-Extension) der richtige Wert — Millisekunden seit Epoche als Dezimalzahl im Formularfeld. Grund: 53 von 54 Aufnahmen tragen bisher nur ein Datum ohne Uhrzeit, und die Meeting-Vorauswahl beim „An Notion senden" hängt genau an dieser Uhrzeit (BACKLOG NOTION-PRESELECT). Unparseable → ignoriert, nie ein 400.

## 5. Share-Extension

`LibraryStore.importSharedAudio` geht denselben Weg: Submit → Pollen → bei `ready` nach `inbox` verschieben. Bei `unauthorized` bleibt der Eintrag vorgemerkt (wie heute); ein `failed` zeigt `error` und lässt den Eintrag ebenfalls stehen.

## 6. Fertig ist es, wenn

1. `grep -rn 'transcribe-audio-file' Sources/ ShareExtension/` → 0 Treffer; `TranscriptionResponse`/`TranscriptionMeta` sind weg.
2. Eine Aufnahme ≤ 10 s im angemeldeten Simulator: 202 → Polls → `ready` → Transkript in der Ergebniskarte → Speichern → Zeile in der Inbox (ids im Bericht; der Master entfernt die Testzeile).
3. Dieselbe Datei erneut: 200 mit `deduped: true`, kein zweiter Auftrag.
4. Flugmodus: Submit scheitert am Transport, Fehlertext in der Karte, Zustand `ready` (erneut versuchbar), nichts gelandet.
5. `recorded_at` steht in `metadata.recorded_at_client` der Zeile (GET-Beleg).

## Referenzen

- Server: [app_pkg/audio.py](../app_pkg/audio.py) (`api_create_transcription`, `api_transcription_status`, `_transcription_response`), Architektur-Bullet *Audio-Transkription* in [CLAUDE.md](../CLAUDE.md), Browser-Beleg [scripts/smoke_audio_converter.py](../scripts/smoke_audio_converter.py), Web-Aufrufer [static/js/audio_converter.js](../static/js/audio_converter.js) (Z. 493–560: Submit, Polls, Speichern).
- Auth: [docs/mobile_auth_contract.md](mobile_auth_contract.md).
- Sprint: [SPRINT_IOS-RICH-MEDIA_…_2026-10-09.md](archive/sprint-prompts/SPRINT_IOS-RICH-MEDIA_figuren-im-reader-und-die-drei-offenen-posten_2026-10-09.md), Phase 4.
