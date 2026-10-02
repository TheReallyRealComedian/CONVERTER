# SPRINT NOTION-MEETING-LINK — Transkript an ein bestehendes Notion-Meeting hängen

**Größe**: M–L (4 Phasen: Charakterisieren + Schnitt · Bau · Deploy + Abnahme · Wrap + Antwort) · **Datum**: 2026-10-02 · **Herkunft**: Auftrag des Koordinators vom 2026-10-01, mit Oli abgestimmt — [docs/notion_meetings_transkript_brief.md](../../notion_meetings_transkript_brief.md) **samt Nachtrag vom 2026-10-02** (vollständig lesen, er ist der Kontrakt); Auflagen aus [ARCH-AUDIT](../audit-outputs/AUDIT_ARCHITEKTUR_2026-10-01.md) (V-5, W-8)

## Warum

Seit dem 29.09. legt calendar-sync für jedes Meeting aus Olis Arbeitskalender eine Seite in Notion MEETINGS an (Horizont heute −14 bis +7 Tage). Die Transkripte entstehen bei uns, zu 99 % als Datei über die Audio-Route. „An Notion senden → Meeting" legt heute **immer eine neue Seite** an — neben der, die der Kalender schon erzeugt hat. Gebaut wird die Brücke: im Dialog ein bestehendes Meeting wählen, das Transkript landet im Feld `Transcript` dieser Seite, dazu ein Rückverweis im Feld `CONVERTER`.

## Gegroundeter Ist-Zustand (Master, gemessen 2026-10-02)

**Gegenseite (notion-mcp-server `3e7d5cb`, deployt 2026-10-02 20:53; aus dem CONVERTER-Web-Container gemessen, nur lesend):** `POST {NOTION_MCP_URL}/api/meetings/query` mit `{"date_from", "date_to"}` und `Authorization: Bearer $MCP_AUTH_TOKEN` → 200; Top-Level `count`, `date_from`, `date_to`, `meetings`, `only_notnotion`, `truncated`; je Eintrag `calendar_event_id` (leer = `""`), `converter_link`, `datum` (`start`/`end` mit Offset `+02:00`, `time_zone: null`; ganztägig = `start` ohne Uhrzeit), `has_transcript`, `meeting_link`, `notnotion`, `page_id`, `people_count`, `people_count_capped`, `title`, `type`, `url` (Form `https://app.notion.com/p/…`). Schreibweg und Fehlercodes (200 / 409 `transcript_exists` / 413 mit `length`+`max` / 400 / 404 / 503) stehen im Brief und im Nachtrag; die Tabelle des Koordinators kam teils abgeschnitten an — **die genaue Form der Antworten misst du an der Probe-Seite nach**. Ein Transkript lässt sich über die API nicht entfernen (`transcript: ""` = nicht senden), `converter_link: ""` leert das Feld.

**Probe-Seite:** „CONVERTER-Probe", page id `3ed3f5db-30d2-805b-baec-ec7216b6ffc0`, Datum 2026-10-02 20:35–21:35, `Transcript` und `CONVERTER` leer (gemessen: `has_transcript: false`, `converter_link: null`).

**Unser Code heute:**
- [app_pkg/integrations/notion.py](../../../app_pkg/integrations/notion.py): `api_send_to_notion` reicht `data['fields']` ungefiltert an `POST /api/<target>` (`meetings`/`notes`/`inbox`), nur `datum` läuft durch `normalize_notion_datum` (NOTION-TZ). Fehler der Gegenseite werden mit ihrem Status durchgereicht; die zwei englischen Meldungen („Invalid target", „Failed to reach Notion server.") stehen noch da. Die Vorschläge (`/api/notion/suggestions`) cachen **auch Fehlschläge**: ein non-200 von Notion wird ohne Log zu `{}` bzw. `[]` und 300–3600 s gehalten.
- [static/js/library_detail.js](../../../static/js/library_detail.js) (1 830 Zeilen): die Notion-Gruppe Z. 225–441 (`collectNotionFieldValues`, `restoreNotionFieldValues`, `toggleNotionPanel`, `loadSuggestions`, `selectTarget`, `renderNotionFields`, `sendToNotion`), dazu Z. 6 (`DEFAULT_TARGET`), Z. 29 (`NOTION_TARGET_LABELS`), Z. 32/39–42 (`notionAlertContainer`, `clearNotionAlert`), Z. 1827–1829 (`window`-Exporte). Der Dialog steht in [templates/library_detail.html](../../../templates/library_detail.html) Z. 193–210 mit Inline-`onclick`. Das Transkript nimmt `sendToNotion` aus `#content-source` (= `conversion.content`). `PageData` trägt `conversionType` und `defaultNotionTarget`. Bestätigungen laufen über das native `confirm()` (Z. 189).
- **Kein Test** trifft `api_send_to_notion` oder die Vorschläge (`tests/test_notion_datum.py` prüft nur den Normalisierer). Der Dialog hat keinen Browser-Check.
- **`metadata_json`** wird an zwölf Stellen als Ganz-Blob geschrieben, es gibt keinen Merge-Schreibweg (für `settings_json` gibt es ihn: `learn.write_settings_keys`, ein `json_patch`-UPDATE, LOST-UPDATE). ⚠️ `metadata_json` ist **vom Client schreibbar** (`POST /api/conversions` nimmt eine `metadata`-Tasche) — was daraus gelesen und gerendert wird, ist Eingabe.
- Eine öffentliche Basis-URL kennt der Code nicht.

**Die Bezugszeit — gemessen an Olis Daten, und sie trägt die Regel aus dem Brief nicht:** von 65 Transkriptionen haben 54 ein `recorded_at`; **53 davon stammen aus dem Dateinamen und stehen auf 00:00** (der Dialekt des Diktiergeräts `YYMMDD_NNNN` trägt nur das Datum; `parse_recorded_at_from_filename` setzt dann 00:00 Ortszeit), **genau eine** hat eine echte Uhrzeit (Quelle `client`, und dort ist sie gleich der Upload-Zeit). Der Dateiname schlägt die Zeit des Browsers (`file.lastModified`), die deshalb gar nicht gespeichert wird. Die Regel des Briefs — Vorauswahl, wenn die Bezugszeit in [Beginn −15 min, Ende] liegt — hätte an den sechs Aufnahmen der letzten 14 Tage **null Mal** gegriffen, bei 5 bis 8 Meetings je Tag; „nach Abstand sortiert" hieße dort „nach Abstand zu Mitternacht". Was die Daten dagegen hergeben: `duration_seconds` steht an jeder Aufnahme, und die Länge passt sichtbar (25-min-Aufnahme, ein 25-min-Meeting am selben Tag).

**Baseline:** 1445 passed + 1 skipped, Mac und Pin. Prod-Image `73dd58c0dd31`, Rollback-Tags `pre-sec-dg-token`, `pre-arch-narr5`. Der Web-Dienst läuft seit SEC-DG-TOKEN unter `docker-init` (Browser-Smokes im Web-Container sind damit unkritisch).

## Gesperrte Entscheidungen

1. **Der Kontrakt ist der Brief samt Nachtrag** (Olis Entscheidungen A–D: überschreiben nur nach Rückfrage · nur das Transkript · Rückverweis im Feld `CONVERTER` · ad hoc bleibt Neuanlage). Wo dieser Prompt davon abweicht, steht es hier und gehört in die Antwort an den Koordinator.
2. **Keine Vorauswahl ohne echte Uhrzeit** (Abweichung vom Brief, aus der Messung). Der Server unterscheidet zwei Fälle: *Uhrzeit bekannt* (ein `recorded_at` mit einer Zeit ungleich 00:00:00) → Regel des Briefs: Kandidaten Bezugstag −1 bis +1, nach Abstand zum Meetingbeginn sortiert, Vorauswahl nur bei Bezugszeit in [Beginn −15 min, Ende], bei mehreren das mit dem nächsten Beginn. *Nur das Datum bekannt* (00:00, der Normalfall) → die Meetings des Bezugstags chronologisch, danach die Nachbartage, **keine Vorauswahl**, sichtbarer Hinweis „Uhrzeit der Aufnahme unbekannt". Fehlt `recorded_at` ganz: `created_at` als Bezugstag mit dem Hinweis aus dem Brief („Aufnahmezeit unbekannt, Upload-Zeit verwendet"), ebenfalls ohne Vorauswahl. Falsch vorausgewählt ist schlimmer als nicht vorausgewählt.
3. **Längen zeigen statt raten.** Der Dialog nennt die Dauer der Aufnahme, jeder Kandidat seine Länge. Keine Vorauswahl und keine Sortierung nach Dauer — die Entscheidung trifft Oli, mit beiden Zahlen vor Augen.
4. **Der Server rechnet, das JS zeichnet** (Hausregel seit LEARN-MORE): Reihenfolge, Vorauswahl, Tagesgrenzen und die Anzeige-Texte für Wochentag, Datum und Uhrzeit entstehen im Backend in `LOCAL_TZ`. Im JS keine Zeitzonen-Arithmetik.
5. **Der Sendeweg „bestehendes Meeting" baut seine Nutzlast selbst:** genau `page_id`, `transcript` (serverseitig aus `conversion.content`, nicht aus dem Request), `converter_link`, `replace_transcript` — **nichts sonst** (Titel, Datum, Typ, Personen, Projekt, Zusammenfassung würden die Kalender-Seite überschreiben). `page_id` wird der Form nach geprüft, bevor sie die Gegenseite erreicht. `get_owned_conversion` wie bisher, der Token bleibt im Server.
6. **Fehler der Gegenseite werden übersetzt, nicht durchgereicht:** 409 `transcript_exists` → 409 mit Titel und Datum für die Rückfrage („‚<Titel>' (<Datum>) hat schon ein Transkript. Überschreiben?", bei Ja derselbe Aufruf mit `replace_transcript: true`, bei Nein nichts); 413 → klare Meldung mit Länge und Grenze; 400/404/503 und Netzfehler → deutsche Meldung in Haus-Microcopy. Die zwei englischen Altmeldungen werden dabei deutsch.
7. **`converter_link`** = `PUBLIC_BASE_URL` + `/library/<id>`; `PUBLIC_BASE_URL` als Env mit Compose-Default `https://converter.smallpieces.de`, ohne die Variable `request.url_root`. Er reist auf **beiden** Wegen mit (auch bei „Neues Meeting anlegen"). ⚠️ Benannte Eigenschaft, nicht in diesem Sprint zu lösen: `Conversion.id` wird nach dem Löschen der höchsten Zeile neu vergeben — ein Link in Notion kann danach auf ein anderes Dokument zeigen (Folge-Item im Wrap).
8. **Die Verknüpfung liegt als Namensraum `notion_link` in `metadata_json`** (`page_id`, `url`, `calendar_event_id`, `meeting_title`, `meeting_start`, `linked_at`) und wird über **einen Merge-Schreibweg** geschrieben: ein `json_patch`-UPDATE wie `learn.write_settings_keys`, kein Lesen-Ändern-Schreiben des Blobs. Die elf anderen Ganz-Blob-Schreiber bleiben, wie sie sind (kein Umbau; die Eigenschaft im Bericht benennen). Auch nach einer Neuanlage wird die Verknüpfung gemerkt, wenn die Antwort die Seite nennt. **Gelesen wird `notion_link` als Eingabe:** Text nur als Text-Knoten, die URL wird nur dann ein Link, wenn sie `https` ist und auf einen Notion-Host zeigt.
9. **Zusätzlich gespeichert, nicht benutzt:** die Browser-Zeit der Datei als `recorded_at_client` an der Transkriptions-Zeile, wenn sie beim Upload mitkommt — additiv, ohne jede Logik daran. Sie macht später messbar, ob das Kopieren vom Diktiergerät die Aufnahmezeit erhält.
10. **Neue Oberfläche = neue Datei, fremder Text = Text-Knoten.** Die Kandidatenliste und die Verknüpfungs-Anzeige entstehen in `static/js/library_notion.js` per `createElement`/`textContent` (Meeting-Titel kommen aus Notion). Der wörtlich umgezogene Altcode behält sein `innerHTML` mit `escHtml`.
11. **Geschrieben wird in Notion nur auf die Probe-Seite.** Jeder Test, jeder Smoke und jede Abnahme sendet ausschließlich an page id `3ed3f5db-30d2-805b-baec-ec7216b6ffc0`; ein Skript bricht ab, wenn ein anderes Ziel ausgewählt wäre. Kein echtes Meeting wird beschrieben, auch nicht „zum Test".
12. **Nicht in diesem Sprint:** Transkripte entfernen, Zusammenfassung/Tasks/TERMS (Skill `notion-transcripts`), Ad-hoc-Kalendereinträge, Zugriff auf den Nextcloud-Kalender, Änderungen am notion-mcp-server (Wünsche in die Antwort), Vorauswahl über Dauer oder `recorded_at_client`, id-Wiederverwendung, die anderen `metadata_json`-Schreiber, die Ziele Notiz und Inbox.

**Arbeitsweise:** inline, **kein Workflow, keine Subagenten** ohne Olis ausdrückliches Wort. Commit + Push je Phase, dann Stop + Bericht. Nichts Tragendes im Session-Scratch. Editiert wird nur auf dem Mac. Kein Token, kein Transkript-Inhalt echter Meetings in Ausgaben.

---

# Phase 0 — Charakterisieren, dann schneiden (kein Deploy)

1. **Dialog-Smoke gegen den Stand, der heute in Prod läuft**, im Stil der bestehenden Smokes (`scripts/smoke_notion_dialog.py`, im Web-Container, Wegwerf-User mit eigener Conversion per ORM): Panel öffnen, die drei Ziele umschalten, Felder und ihre Übernahme beim Wechsel, Vorschläge geladen (Datalists gefüllt), Senden-Knopf mit seinen Zuständen — **ohne zu senden**. Er hält fest, was der Dialog heute tut; die Kriterien kommen aus dem gemessenen Verhalten, zweimal gleich gemessen.
2. **Dann der Schnitt:** die sieben Notion-Funktionen samt Konstanten, Alert-Helfern und `window`-Exporten **wörtlich** nach `static/js/library_notion.js` (Funktionskörper byte-gleich, Beleg wie bei ARCH-NARR5), `<script>`-Zeile im Template. Was `library_detail.js` und die neue Datei voneinander brauchen (`CONVERSION_ID`, `conversionTagsState`, `formatDatetimeLocalNow`, `showAlert`, `safeJSON`), im Bericht als Liste.
3. **Vorschläge:** ein non-200 von Notion wird geloggt (WARNING mit Status, ohne Token) und **nicht** gecacht; der leere Rückfall für den Client bleibt.
4. **Gates:** Suite am Mac; die Reader-Gates hängen an seitenglobalen Namen von `library_detail.js` — belegen, dass keiner davon umgezogen ist (Liste der Namen, die `scripts/smoke_reader_media.py` und `scripts/measure_highlight_anchors.py` per `page.evaluate` rufen).

## Stop
Bericht: Smoke-Ergebnis am Prod-Stand · Beleg „byte-gleich" · Abhängigkeits-Liste der neuen Datei · Suite. Dann warten. (Deployt wird Phase 0 zusammen mit Phase 1; der Smoke läuft dann wieder und beweist den Schnitt.)

---

# Phase 1 — Bau (zwei Commits: Backend, Frontend)

## 1a Backend — Tests zuerst, gegen HEAD rot gezeigt

- **Kandidaten:** `GET /api/conversions/<id>/notion-meetings[?day=YYYY-MM-DD]` (Session, Owner-404). Antwort trägt die Bezugs-Angaben (Tag, ob die Uhrzeit bekannt ist, Quelle `recorded_at`/`created_at`, Dauer der Aufnahme), die Kandidaten in fertiger Reihenfolge mit Anzeige-Texten, Länge, `has_transcript`, „schon verknüpft" (`converter_link` gesetzt) und „mit diesem Dokument verknüpft", die Vorauswahl (oder keine) und die gemerkte Verknüpfung dieser Conversion. `day` wird strikt gelesen (sonst 400). Ganztägige Einträge sind auswählbar, tragen keine Länge und sind nie Vorauswahl. Kein Filter nach Typ, `notnotion` bleibt auswählbar. Ist die gemerkte Verknüpfung vorhanden, öffnet der Dialog auf ihrem Tag und wählt sie vor.
- **Senden:** der bestehende Endpunkt bekommt den Weg `page_id` (Entscheidung 5, 6) und auf beiden Wegen den `converter_link`; nach Erfolg der Merge-Schreibweg (Entscheidung 8).
- **`recorded_at_client`** (Entscheidung 9) am Transkriptions-Submit.
- **Tests** (`http_requests` gemockt, kein Netz): Reihenfolge und Vorauswahl für Uhrzeit bekannt / nur Datum / kein `recorded_at`; Grenzen der Vorauswahl (Beginn −15 min, Ende, knapp daneben, zwei Treffer); ganztägig; Tagesparameter; die Nutzlast des Sendewegs ist **genau** die vier Felder, auch wenn der Request mehr mitschickt; `transcript` kommt aus der Zeile; 409 → 409 mit Rückfrage-Daten, mit `replace_transcript` → 200; 413; 400/404/503/Netzfehler; der Merge-Schreibweg lässt fremde Metadaten-Schlüssel stehen (Positivkontrolle: ein Ganz-Blob-Schreiben hätte sie gelöscht); `notion_link` mit einer Nicht-Notion-URL wird nicht als Link ausgeliefert; Owner-404; die Neuanlage ist unverändert bis auf `converter_link`.

## 1b Frontend

- Im Ziel „Meeting" zwei Wege: **„Bestehendes Meeting"** (bei `audio_transcription` vorausgewählt) und **„Neues Meeting anlegen"** (das heutige Formular, unverändert). Die Ziele Notiz und Inbox bleiben, wie sie sind.
- Liste: je Eintrag Wochentag, Datum, Uhrzeit, Titel, Typ, Länge; deutlich markiert „hat schon Transkript" und „schon mit CONVERTER verknüpft"; oben die Dauer der Aufnahme und der Hinweis zur Bezugszeit; der Tag ist änderbar (Datumsfeld, vor/zurück).
- Senden nur mit einer Auswahl; 409 → `confirm()` mit dem Satz aus Entscheidung 6; 413 → Alert mit Länge und Grenze; Erfolg → Toast, die Liste zeigt den neuen Zustand.
- Detailseite: „Verknüpft mit <Titel>, <Datum>" mit Link nach Notion, sichtbar ohne das Panel zu öffnen.
- UI-Texte deutsch, Haus-Microcopy (Fehler höchstens zwei Sätze, Knöpfe höchstens drei Wörter). Helfer aus `_utils.js` nutzen.

**Gates:** Suite am Mac (Abweichung von 1445 + 1 Skip benennen), Container-Suite auf dem Pin (stdin-Rezept), `json_patch` gegen das SQLite des Images.

## Stop
Bericht: die roten Läufe · Antwortform des Kandidaten-Endpunkts (ein Beispiel, mit erfundenen Titeln) · Diff je Datei · Suite Mac und Container · Abweichungen. Dann warten.

---

# Phase 2 — Deploy + Abnahme an der Probe-Seite

1. **Fenster:** Queue leer, Worker idle, 0 `pending`. **DB-Backup** nach dem Rezept in CLAUDE.md, Kopie mit `mode=ro&immutable=1` prüfen. Rollback-Tag `converter-app:pre-notion-meeting-link`; `pre-arch-narr5` und `pre-sec-dg-token` dürfen weg, wenn du vorher im Bericht festhältst, dass der Rückweg auf `pre-arch-narr5` ohnehin die alte Compose-Datei brauchte. **Nie `docker image prune -a`.**
2. `git pull --ff-only`, `docker compose up -d --build` aus dem Projektverzeichnis; `docker inspect`: was neu ist; Login 200; `PUBLIC_BASE_URL` im Web-Container gesetzt.
3. **Die Schreib-Antworten der Gegenseite an der Probe-Seite messen** (Form von 200, 409, 413) und gegen den Nachtrag im Brief halten.
4. **Smoke** `scripts/smoke_notion_dialog.py`, jetzt mit dem neuen Weg, Wegwerf-User, zwei eigene `audio_transcription`-Zeilen per ORM: eine mit Uhrzeit **im** Fenster der Probe-Seite (2026-10-02, 20:35–21:35), eine nur mit Datum. Die fünf Abnahme-Fälle des Briefs:
   1. Uhrzeit im Fenster → die Probe-Seite ist vorausgewählt (**prüfen, dass es die Probe-id ist**; liegt ein echtes Meeting im selben Fenster und gewinnt, die Probe ausdrücklich wählen und das berichten) → Senden → per `query`: `has_transcript: true`, `converter_link` gesetzt; Titel, Datum, Typ, Personen der Seite unverändert.
   2. Dieselbe Conversion erneut → Rückfrage → „Ja" → ersetzt, kein doppelter Inhalt; „Nein" → nichts geändert.
   3. Nur-Datum-Zeile → Hinweis, keine Vorauswahl, Tag änderbar; und eine Zeile ohne `recorded_at` → Hinweis auf die Upload-Zeit.
   4. Überlanges Transkript (über 200 000 Zeichen) → klare Meldung, nichts geschrieben.
   5. „Neues Meeting anlegen" → **nur bis vor das Senden** charakterisieren (es legte eine echte Seite in Olis MEETINGS an); dass `converter_link` in der Nutzlast steht, belegt der Unit-Test. Wenn du es Ende-zu-Ende belegen willst: stoppen und fragen.
   Dazu: die Detailseite zeigt die Verknüpfung, das erneute Öffnen des Dialogs wählt die Probe-Seite vor; der Altbestand des Dialogs (Phase 0) läuft unverändert.
5. **Aufräumen:** `converter_link` der Probe-Seite per API leeren; das Transkript der Probe-Seite lässt sich per API nicht entfernen — im Bericht sagen, dass Oli es von Hand leert oder die Seite als Dauer-Probe behält. Wegwerf-User strikt nach `user_id`. Inventar: DB (Zeilenzahl, `max(id)`, Tokens), Volume, `/tmp`, Mintbox-Checkout sauber.
6. Reader-Gate einmal fahren (`scripts/smoke_reader_media.py`), weil `library_detail.js` und das Template angefasst sind.

Sicherheits-Regeln wie immer: `docker exec` nie mit `-u 0`; DB-Lesen `mode=ro`; keine unversionierte Datei auf der Mintbox.

## Stop
Bericht: Deploy · gemessene Antwortformen der Gegenseite · die fünf Fälle je mit Beleg · Zustand der Probe-Seite danach · Inventar.

---

# Phase 3 — Wrap + Antwort an den Koordinator

- **[docs/notion_meetings_transkript_antwort.md](../../notion_meetings_transkript_antwort.md)** (Rückkanal laut Brief): was gebaut ist; die Abnahme; **die Messung zur Bezugszeit** (53 von 54 nur Datum) mit der Folge für Vorauswahl und Sortierung; was sonst vom Brief abweicht; die gemessenen Antwortformen; Wünsche an den notion-mcp-server, falls beim Bau welche entstanden sind; die benannte Eigenschaft zur id-Wiederverwendung.
- **CLAUDE.md:** ein Bullet (oder die Erweiterung des NOTION-TZ-Bullets): zwei Wege, Kandidaten-Endpunkt, Sendeweg mit fester Nutzlast, `notion_link` und der Merge-Schreibweg, die Regel „`notion_link` ist Eingabe", die Bezugszeit-Messung, die Probe-Seite als einzige Schreib-Zielseite für Tests, der Smoke in der Skript-Liste, `PUBLIC_BASE_URL`, Test-Baseline.
- **STATUS.md**, **BACKLOG.md** (NOTION-MEETING-LINK schließen; ⚠️ Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md`, exit 1 = sauber). Zwei Folge-Items anlegen, beide „benannt, nicht gebaut": **CONV-ID-NO-REUSE** (ids nicht wiederverwenden — betrifft jetzt Links, die in Notion liegen) und **NOTION-PRESELECT** (Vorauswahl für Nur-Datum-Aufnahmen: erst messen, ob `recorded_at_client` die Aufnahmezeit trägt).
- **Brief ans converter-mcp:** keine Agent-Fläche gebaut (Web-Dialog). `Conversion.to_dict()['metadata']` trägt neu `notion_link` und `recorded_at_client` — additiv; vor dem Urteil die Tool-Liste am lebenden Connector lesen und im Wrap sagen, warum kein Brief. Die iOS-App dekodiert aus `metadata` nur ihre eigenen Schlüssel (im Clone nachgesehen).
- **Memory** nur bei übertragbarer Lehre. Kandidat: *eine Auswahlregel an den echten Daten messen, bevor sie gebaut wird — die Bezugszeit, auf der die Vorauswahl stand, gab es in 53 von 54 Fällen nicht.* Vorher prüfen, was `feedback_guard_rail_must_catch_its_own_case` und `reference_dictation_recorder_filename_pattern` schon tragen.
- Commit + Push, Stop + Bericht.

## Nicht-Ziele

Siehe Entscheidung 12. Außerdem: kein Umbau des Dialogs für Notiz und Inbox, keine CSP-Arbeit an den Inline-Handlern (CSP-BASELINE), kein Zerlegen von `library_detail.js` über den Notion-Schnitt hinaus.
