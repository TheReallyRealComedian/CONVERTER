# UX-Patterns + Microcopy: podcast-flow (2026-05-10)

**Methodik:** Stufe 3 der Duan-Kaskade (Duan et al., *Heuristic Evaluation with LLMs*, CHI 2024). Konkrete Patterns + DE-Microcopy auf Basis der Heuristik-Findings aus Stufe 2. Konsolidiert die 16 Stufe-2-Findings auf 12 Pattern-Blöcke (4 konsolidiert + 8 einzeln). Bug-Tickets BT1–BT3 sind in den Patterns ihrer verknüpften Findings mit-adressiert; BT4 (Audio-Blob-URL-Revoke) ist pure Bug-Ticket ohne UX-H-Komponente und nicht Teil von F-4.3 (separater Bug-Sweep oder mit-genommen wenn nahegelegene Patterns berührt werden).
**Quelle Findings:** [docs/ui_findings_podcast_flow_2026-05.md](ui_findings_podcast_flow_2026-05.md)
**Quelle Inventur:** [docs/ui_inventory_podcast_flow_2026-05.md](ui_inventory_podcast_flow_2026-05.md)
**F-1 / F-2 / F-3 Patterns als Referenz:** [docs/ui_patterns_document_converter_2026-05.md](ui_patterns_document_converter_2026-05.md), [docs/ui_patterns_audio_converter_2026-05.md](ui_patterns_audio_converter_2026-05.md), [docs/ui_patterns_library_detail_2026-05.md](ui_patterns_library_detail_2026-05.md)
**Helper-API:** [static/js/_utils.js](../static/js/_utils.js) — `safeJSON(response)`, `fallbackCopyText(text)`, `showAlert(containerEl, level, msg, options?)`, `showToast(msg, options?)`, `formatFileSize(bytes)`, `formatDatetimeLocalNow()`, `confirmIfLong(text, msg, options?)`. CSS-Utility `.sr-only` aus [static/css/style.css:996](../static/css/style.css#L996).
**Komponenten-Basis:** Existierende Neomorphism-Klassen aus [static/css/style.css](../static/css/style.css) — `c-btn`, `c-btn--primary`, `c-input`, `c-card`, `c-alert--danger/success/warning/info` (mit Close-Button + Auto-Dismiss aus F-1 Cluster C), `.toast-notification`, `.hidden`, `.mic-button__spinner` als animierte Spinner-Vorlage, `.sr-only`. Banner-Container `#podcast-alert-container` existiert seit F-2.3 P4 ([templates/audio_converter.html:121](../templates/audio_converter.html#L121)).

**Microcopy-Regeln:** Fehler max 2 Sätze, Empty-State max 3 Sätze, Buttons max 3 Wörter, keine Emojis bei Fehlern, Deutsch durchgängig (Du-Form analog F-1 / F-2 / F-3).
**Aufwand-Skala:** XS / S / M / L (Daumenregel: XS = 1–3 Zeilen, S = ein Handler-Cluster + Microcopy-Sweep, M = Schema-Touch oder neue Mechanik, L = Cross-Stack-Refactor mit Backend-Endpoint + Worker-Touch).
**Impact-Score-Formel:** `Score = Sev × 5 / Aufwand-Gewicht` mit Aufwand-Gewichten XS=1, S=2, M=4, L=8. Höher = besser. Bei konsolidierten Patterns wird die höchste Sev der adressierten Findings genommen (analog F-1.3 / F-2.3 / F-3.3).

**Live-Verifikation-Konvention** (NEU für F-4.3 wegen Master-Walkthrough 2026-05-10): Patterns für F1/F2/F4 tragen `✅ live-verifiziert` mit Verweis auf den Master-Walkthrough (Cancel-Disk-Forensik, Job `9bf48e0a-7fb1-46bb-baac-5d0f88ef12c5`). Patterns für F5/F8/F16 tragen `🔥 Smoke-Pflicht in F4-IMPL` — vor Apply muss F4-IMPL-Sub-Thread per Live-Smoke verifizieren (DevTools-Network-Throttle für F8; Browser-Reload-Sequenz für F5; visueller Vergleich Mic-Spinner vs. Generate-Btn-Icon für F16). Wenn der Smoke zeigt, dass der Befund nicht reproduzierbar ist: Pattern-Apply STOP, Master fragen.

**Cross-Feature-H4-Quote:** 0% (siehe F-4.2 Cross-Feature-Sektion). F-2-Konvergenz ist im podcast-flow bereits sauber durchgezogen — `safeJSON`/`showAlert`/`showToast`/`confirmIfLong` werden vollständig genutzt. Verbleibende H4-Findings (F7, F9) sind interner Natur. Die Patterns hier nutzen die existierenden `_utils.js`-Helper, aber die zusätzliche Reuse-Druck-Begründung aus F-2.3 / F-3.3 entfällt.

---

## Pattern-Blöcke

### Pattern 1: Cancel-Mechanik (Worker-Stop + Frontend-Lüge entfernen)
**Adressiert Findings:** F1 (H1 Sev 4), F2 (H9 Sev 4)
**Adressiert Bug-Ticket:** BT1 (Backend-Cancel via RQ-API plus UI-Microcopy-Korrektur)
**Cluster:** 1 (Cancel-und-Cleanup-Recovery)
**✅ live-verifiziert** (Master-Walkthrough 2026-05-10: Worker lief 1:12 Min nach Cancel-Klick weiter, WAV-File 11.7 MB im Volume, TTS-Token vollständig verbraucht. Frontend-UI „unauffällig" — aktive Lüge.)

- **Pattern:** Cancel triggert tatsächlich einen Backend-Stop (kein reiner Frontend-Polling-Stop wie heute). Drei Sub-Mechaniken kohärent zusammen, weil sie denselben Code-Touch bilden und State-Übergänge `started → cancelling → cancelled/failed` an einer Stelle koordinieren müssen:
  1. **Bestätigungs-Dialog beim Cancel-Klick** (statt direkter Ausführung wie heute) — User-Erwartung „spar mir den Gemini-Credit" wird transparent gemacht: TTS-Token können bereits teilweise konsumiert sein, Cancel ist Best-Effort.
  2. **Backend-Cancel-Endpoint** `POST /podcast-cancel/<job_id>` mit user_id-Match-Check ruft `Job.cancel()` (für `queued`-Jobs) bzw. `send_stop_job_command(connection, job_id)` (für `started`-Jobs) auf. Worker-Code in `tasks.generate_podcast_task` plus Chunking-Loop in `services/gemini/tts.py` braucht **cooperative cancel checks** zwischen TTS-Chunks (`if redis.exists(f"cancel:{job_id}"): raise CancelledError`), damit der Worker mid-pipeline tatsächlich aussteigt — `send_stop_job_command` allein sendet `SIGINT` an den Worker-Prozess, was unter rq 2.x je nach Worker-Konfiguration den Job mid-TTS-Call abrupt beendet. Sub-Thread-Recherche-Punkt für F4-IMPL: prüfen, ob `Worker.kill_horse()`-Pfad nötig ist oder ob `send_stop_job_command` mit cooperative checks ausreicht.
  3. **Frontend-Polling läuft weiter mit Zwischenstate** „Wird abgebrochen …", endgültiger Status (`cancelled`/`failed`) erst nach Worker-Confirmation via `/podcast-status/<job_id>`. Erst dann Banner „Generierung abgebrochen". Der heutige sofort-grüne „Generierung abgebrochen"-Pfad mit Suffix „Backend-Job läuft im Hintergrund weiter" entfällt komplett — er ist die aktive Frontend-Lüge.

- **Visuelle Hinweise:** Bestätigungs-Dialog als zweistufiger In-Page-Dialog (analog F-2.3 P17 Clear-Confirmation), nicht native `confirm()`. Cancel-Button-State während Roundtrip: Text „Wird abgebrochen …" (max 3 Wörter), `disabled=true`, `pointer-events: none`. Banner final über `showAlert(podcastAlertContainer, 'warning', msg)` — Warning-Tint, weil Recovery-Gedanke (User kann erneut starten) und kein hartes Failure.
- **Microcopy** (DE, Du-Form, max 2 Sätze):
  - Confirm-Dialog Stage 1 (default-Cancel-Btn): „Abbrechen"
  - Confirm-Dialog Stage 2 (nach erstem Klick): „Generierung wirklich abbrechen? TTS-Token sind teilweise schon verbraucht."
  - Confirm-Dialog Stage 2 Bestätigen-Btn: „Ja, abbrechen"
  - Confirm-Dialog Stage 2 Verwerfen-Btn: „Weiter generieren"
  - Cancel-Btn während Roundtrip: „Wird abgebrochen …"
  - Banner nach erfolgreichem Cancel (Worker meldet `cancelled`/`failed`): „Generierung abgebrochen."
  - Banner falls Worker trotz Cancel-Signal natürlich fertig wurde (Race-Case): „Generierung wurde noch fertig. Datei wird verworfen." (info-Tint, leitet zu P2 Cleanup-Pfad über)
  - aria-live-Hint bei Stage-2-Open: „Bestätigung erforderlich. Erneut auf Abbrechen klicken zum Bestätigen."
- **Helper-Reuse:** `showAlert(podcastAlertContainer, 'warning', msg)` für den finalen Banner; `confirmIfLong` ist nicht passend (das ist ein Length-Schwellen-Helper) — Confirm-Mechanik wird inline im Cancel-Btn-Handler als kleine State-Machine implementiert (idle → confirm-pending → cancelling → cancelled). Code-Anker: [static/js/audio_converter.js:648-652](../static/js/audio_converter.js#L648-L652) Cancel-Btn-Handler; [static/js/audio_converter.js:854-858](../static/js/audio_converter.js#L854-L858) Cancel-Banner-Pfad; [app_pkg/podcasts.py:138-164](../app_pkg/podcasts.py#L138-L164) Generate-Endpoint (kein Cancel-Endpoint vorhanden); [tasks.py:32-58](../tasks.py#L32-L58) Worker-Task; [services/gemini/tts.py:154-216](../services/gemini/tts.py#L154-L216) Per-Chunk-Loop für cooperative cancel checks.
- **Aufwand:** L — Backend-Endpoint + Worker-cooperative-cancel-Pfad + Frontend-State-Machine + Microcopy-Sweep. RQ-Verhalten (cooperative cancel vs. `Worker.kill_horse()`) muss in F4-IMPL recherchiert werden — siehe „Konstitutiv mit-genommen, falls berührt"-Sektion am Doc-Ende.
- **Impact-Score:** 4 × 5 / 8 = **2.5**
- **Konsolidierung:** F1 (H1 Sichtbarkeit) und F2 (H9 Recovery) entstehen aus derselben Wurzel: Cancel-Btn ist Frontend-Lüge ohne Backend-Stop. Eine Lösung (echter Backend-Cancel + ehrliches Frontend) erfüllt beide Heuristiken — Sichtbarkeit per Roundtrip-State, Recovery per Confirm-Dialog vor Token-Verbrennung. Strukturanalogon zur F-2.3 P2 (Drag-Drop-Lüge): Label/Knopf verspricht Funktion, die nicht erfüllt wird.

---

### Pattern 2: Orphaned-File-Cleanup + Re-Download-Pfad
**Adressiert Findings:** F4 (H9 Sev 3)
**Adressiert Bug-Ticket:** BT3 (File-Cleanup-Strategie TTL statt Sofort-Löschung plus Re-Download-Pfad)
**Cluster:** 1 (Cancel-und-Cleanup-Recovery)
**✅ live-verifiziert** (Master-Walkthrough 2026-05-10: orphaned WAV bleibt im Volume `/var/lib/docker/volumes/converter_podcast_data/_data/` liegen weil Cleanup nur beim Download-Pfad triggert. Disk-Wachstum nach abgebrochenen Jobs.)

- **Pattern:** File-Cleanup-Strategie umstellen von „löschen direkt nach erstem Download" auf TTL-basiert plus Re-Download-Pfad. Drei zusammenhängende Sub-Mechaniken:
  1. **TTL-File-Retention im OUTPUT_DIR** — `podcast_download` löscht nicht mehr direkt nach dem `BytesIO`-Read, sondern lässt das WAV im Volume liegen. Dedizierter Cleanup-Job (cron-Style oder beim nächsten erfolgreichen Generate aufgerufen) entfernt WAVs älter als TTL (z.B. 1 h). Damit ist Re-Download im Browser-Block-Fall möglich.
  2. **Frontend-Re-Download-Btn** — neben dem ersten Download bleibt der Btn mit `href` zum `/podcast-download/<job_id>`-Endpoint sichtbar. Click feuert erneut den Download — Backend kann den File noch ausliefern, weil TTL noch nicht abgelaufen.
  3. **Orphaned-File-Cleanup für Cancel-Race** (verzahnt mit P1) — wenn Worker trotz Cancel-Signal natürlich fertig wurde, schreibt er das WAV-File trotzdem. Der Cancel-Backend-Pfad löscht in seinem Post-Worker-Hook das geschriebene File proaktiv (oder lässt es durch den TTL-Cleanup mitnehmen). Frontend-seitig signalisiert P1 das mit „Datei wird verworfen.".
- **Visuelle Hinweise:** Re-Download-Btn neben dem Erst-Download als sekundärer Button (`c-btn` ohne `--primary`-Tint, dezenter), sichtbar solange Player-Pane sichtbar ist. Bei abgelaufenem TTL feuert der Backend-Endpoint 404 → Frontend zeigt Banner „Datei nicht mehr verfügbar" (siehe Microcopy unten).
- **Microcopy** (DE, Du-Form, max 2 Sätze; Buttons max 3 Wörter):
  - Re-Download-Btn-Label: „Erneut herunterladen"
  - Re-Download-Btn aria-label: „Podcast-Datei erneut herunterladen"
  - Banner bei TTL-Ablauf (404 vom Backend): „Podcast-Datei nicht mehr verfügbar — bitte erneut generieren."
  - Banner bei Browser-Block beim Erst-Download (Folge-UI nach P6 Toast-Guard-Fix, der den falschen Erfolgs-Toast verhindert): „Download wurde vom Browser blockiert. Erneut auf Herunterladen klicken oder Pop-Up-Blocker prüfen."
- **Helper-Reuse:** `showAlert(podcastAlertContainer, 'warning', msg)` für TTL-Ablauf-Banner. Code-Anker: [app_pkg/podcasts.py:230-233](../app_pkg/podcasts.py#L230-L233) `os.unlink`-Stelle (entfällt durch TTL-Strategie); [static/js/audio_converter.js:860-866](../static/js/audio_converter.js#L860-L866) Frontend-Download-Pfad (Re-Download-Btn ergänzen).
- **Aufwand:** M — Backend-TTL-Logik (Cleanup-Job oder Lazy-Check beim nächsten Generate) + Backend-Endpoint-Verhalten umstellen + Frontend-Re-Download-Btn + Verzahnung mit P1 Cancel-Hook. Schema-Touch nicht nötig.
- **Impact-Score:** 3 × 5 / 4 = **3.75**

---

### Pattern 3: Stage-Progress über `job.meta` (Worker-Logger-Hebel)
**Adressiert Findings:** F3 (H1 Sev 3)
**Adressiert Bug-Ticket:** BT2 (Worker pflegt `job.meta` mit Stage-Progress)
**Cluster:** 2 (Async-State-Visibility)

- **Pattern:** Architektur-Hebel aus F-4.1 nutzen: `services/gemini/synthesis.py` und `services/gemini/tts.py` haben bereits 50+ Logger-Lines mit Stage-Markers (Skript-Generierung → TTS-Calls pro Speaker → Per-Chunk-Progress → Konkatenation). Die Stage-Progress-UI wird **nicht** durch eine neue Telemetrie-Schicht erzeugt, sondern indem an den existierenden Log-Sites zusätzlich `job.meta['stage'] = …`-Updates plus `job.save_meta()` geschrieben werden. Backend-Endpoint `/podcast-status/<job_id>` reicht das `job.meta` durch. Frontend rendert es in einem dedizierten Stage-Indicator unter dem Counter-Text statt nur den Wand-Sekunden-Counter zu zeigen.

  Stage-Klassen-Mapping (aus den existierenden Logger-Lines abgeleitet):
  - `filtering` — Skript-Filter-Pass (Markdown-Bereinigung, Speaker-Parse-Vorbereitung)
  - `chunking` — Skript wird in TTS-Chunks geteilt
  - `tts_chunk_X_of_N` — Per-Chunk-TTS-Generierung (X = aktueller, N = total)
  - `concatenating` — pydub-Konkatenation
  - `finalizing` — File-Schreibung in OUTPUT_DIR

- **Visuelle Hinweise:** Stage-Indicator als kleine Sub-Caption unter dem Counter („Generiert … (Ns)") mit der textuellen Stage-Information. Optional ein dezenter Linear-Progress-Balken bei Multi-Chunk-Pipelines (`chunk X/N` lässt sich als `width: (X/N * 100)%` rendern). Default: nur Text — der Counter bleibt sichtbar.
- **Microcopy** (DE, Du-Form, max 3 Wörter pro Stage-Label):
  - `filtering`: „Skript wird gefiltert …"
  - `chunking`: „Wird aufgeteilt …"
  - `tts_chunk_X_of_N`: „Chunk X/N wird gesprochen …"
  - `concatenating`: „Audio wird zusammengefügt …"
  - `finalizing`: „Datei wird abgeschlossen …"
  - aria-live-Hint (höflich): bei jedem Stage-Wechsel die neue Stage als kurzer Text in einer Live-Region — Screenreader-User bekommen Stage-Updates ohne zu fokussieren.
- **Helper-Reuse:** keine neuen `_utils.js`-Helper. Lokale Hilfsfunktion `formatStageLabel(meta)` in `audio_converter.js`. Code-Anker: [tasks.py:32-58](../tasks.py#L32-L58); [services/gemini/tts.py:154-216](../services/gemini/tts.py#L154-L216) per-chunk-loop; [services/gemini/synthesis.py](../services/gemini/synthesis.py) für die bereits existierenden Logger-Lines; [app_pkg/podcasts.py:181-191](../app_pkg/podcasts.py#L181-L191) Status-Endpoint; [static/js/audio_converter.js:837-852](../static/js/audio_converter.js#L837-L852) Polling-Loop.
- **Aufwand:** M — Worker-Code an ~5 Log-Sites zusätzlich `job.meta`-Update + `job.save_meta()`; Backend-Endpoint reicht Meta durch (eine Zeile); Frontend Stage-Indicator-Render plus aria-live. Schema-Touch nicht nötig (RQ-`job.meta` ist persistiert).
- **Impact-Score:** 3 × 5 / 4 = **3.75**
- **Sackgassen-Vermerk:** Wenn `job.save_meta()` im Worker-Subprocess nicht synchron ankommt (RQ-Worker-Subprocess-Konfiguration mit eigenem Redis-Pool kann zeitweise inkonsistent sein), muss F4-IMPL eine Alternative finden — z.B. direkt in Redis-Key `podcast:stage:{job_id}` schreiben statt `job.meta`-Pfad. Architektur-Hebel-Plan A scheitert nicht an der Idee, sondern ggf. an der Synchronisierungs-Latenz.

---

### Pattern 4: Browser-Reload-Recovery via LocalStorage
**Adressiert Findings:** F5 (H1 Sev 3)
**Cluster:** 2 (Async-State-Visibility)
**🔥 Smoke-Pflicht in F4-IMPL** (F-4.1 Test-Anleitung 4: Reload mid-Polling, Worker-Continue-Verhalten plus Output-File-Verbleib verifizieren.)

- **Pattern:** Aktiven `job_id` in `localStorage` unter `podcast.activeJobId` persistieren, sobald `/generate-gemini-podcast` einen `job_id` zurückliefert. Beim Page-Load (`DOMContentLoaded`) prüft `audio_converter.js`, ob ein `activeJobId` vorliegt; falls ja, `/podcast-status/<job_id>`-Roundtrip einmal: ist Status `started`/`processing`/`queued` → Polling-Loop direkt re-attachen, Counter und Stage-Indicator (P3) wieder anzeigen. Bei `finished` → Player-Pane mit Re-Download-Btn (P2) hydraten. Bei `failed`/`cancelled` → Banner mit dem letzten bekannten Status anzeigen, dann Storage-Eintrag räumen. Beim natürlichen Job-Ende (Download-Click oder Failure-Banner-Schließen) wird der Storage-Eintrag entfernt.
- **Visuelle Hinweise:** Beim Re-Attach kurz dezenter info-Banner „Laufende Generierung wiederhergestellt." (Auto-Dismiss 4 s), damit der User nicht überrascht ist, dass der Polling-Loop ohne Click weiterläuft. Ansonsten visuell identisch zum Neu-Generate-Pfad.
- **Microcopy** (DE, Du-Form, max 2 Sätze):
  - Re-Attach-Banner (info, auto-dismiss): „Laufende Generierung wiederhergestellt."
  - Re-Attach-Banner falls Job zwischenzeitlich `failed` (danger, persistent): „Vorherige Generierung ist fehlgeschlagen. Bitte neu starten."
  - Re-Attach-Banner falls Job zwischenzeitlich `cancelled` (warning, persistent): „Vorherige Generierung wurde abgebrochen."
  - Re-Attach-Banner falls Job `finished` (info, auto-dismiss): „Vorherige Generierung ist fertig — Datei zum Download bereit."
- **Helper-Reuse:** `showAlert(podcastAlertContainer, level, msg, { autoDismissMs: 4000 })` für die info-Varianten; `safeJSON` für den Status-Roundtrip. Code-Anker: [static/js/audio_converter.js:837-852](../static/js/audio_converter.js#L837-L852) Polling-Loop, der re-attached wird.
- **Aufwand:** S — drei `localStorage`-Setter/Getter, ein DOMContentLoaded-Branch, Status-Mapping zu Banner-Level. Kein Backend-Touch, kein neues CSS.
- **Impact-Score:** 3 × 5 / 2 = **7.5**
- **Smoke-Mechanik:** Reload während aktiver Generierung — neuer Tab nach Reload zeigt Re-Attach-Banner und Counter läuft weiter. Test-Anleitung 4 aus F-4.1 ist die Vorlage.

---

### Pattern 5: Backend-Status-Differentiation `queued` vs. `started`
**Adressiert Findings:** F6 (H1 Sev 2)
**Cluster:** 2 (Async-State-Visibility)

- **Pattern:** Backend-Endpoint `/podcast-status/<job_id>` mappt RQ-`queued` auf `{"status": "queued"}` und RQ-`started` auf `{"status": "started"}` (statt heute beide auf `{"status": "processing"}`). Frontend-Polling-Loop erweitert die Status-Klassen-Behandlung: `queued` zeigt eine eigene Microcopy „Wartet auf Worker …" — bei langer Wartezeit (>30 s) zusätzlich einen Hinweis-Banner, dass der Worker vermutlich nicht läuft (Diagnostik-Hilfe für Single-User-Single-Worker-Setup). `started` und Stage-Updates aus P3 sind dann der Default-Pfad.
- **Visuelle Hinweise:** keine neuen Komponenten. Counter-Sub-Caption (siehe P3) zeigt zusätzlich „Wartet auf Worker …" während `queued`. Bei >30 s Queue-Zeit erscheint ein dezenter info-Banner.
- **Microcopy** (DE, Du-Form, max 3 Wörter Sub-Caption; max 2 Sätze Banner):
  - Sub-Caption während `queued`: „Wartet auf Worker …"
  - Diagnostik-Banner nach >30 s Queue-Zeit (info): „Worker reagiert nicht. Container-Status prüfen: `docker ps`."
- **Helper-Reuse:** `showAlert(podcastAlertContainer, 'info', msg)` für den Diagnostik-Banner. Code-Anker: [app_pkg/podcasts.py:181-191](../app_pkg/podcasts.py#L181-L191) Status-Endpoint; [static/js/audio_converter.js:837-852](../static/js/audio_converter.js#L837-L852) Polling-Loop.
- **Aufwand:** XS — Backend-Mapping eine Zeile, Frontend-Branch + Microcopy + Timer für Diagnostik-Banner. Konsistent mit F-4.2 Befund 5: Single-User-Setup macht den Befund selten relevant, aber Worker-Restart-Diagnostik ist in der Praxis wertvoll.
- **Impact-Score:** 2 × 5 / 1 = **10.0**

---

### Pattern 6: Download-Toast-Guard (Browser-Block-Detection)
**Adressiert Findings:** F10 (H1 Sev 2)
**Cluster:** 2 (Async-State-Visibility)

- **Pattern:** Click-Handler des `#download-podcast-btn` darf nicht mehr blind grünen Toast feuern. Browser-Block-Detection per `document.visibilityState`-Heuristik oder via `download`-Attribut + `<a>`-Click-Pfad mit `try/catch` um den Programmatic-Click. Heute einfache Variante: Toast wird **nicht** beim Btn-Click ausgelöst, sondern beim `audio`-Element-`load` oder beim erneuten Btn-Click (User hat Datei) — alternativ ein Toast erst nach `setTimeout(2500)` plus `document.hasFocus()`-Check (wenn Tab nicht mehr fokussiert ist, hat Browser Download-Dialog wahrscheinlich geöffnet).
  Pragmatisch für Single-User-LAN-Setup: Toast-Pfad ersetzen durch einen kurzen info-Banner, der vom User aktiv geschlossen wird (statt grünem Toast, der bei Browser-Block lügt). Banner-Level `info` (nicht `success`), Microcopy spricht Browser-Block explizit an.
- **Visuelle Hinweise:** Banner statt Toast. Btn-Style unverändert.
- **Microcopy** (DE, Du-Form, max 2 Sätze):
  - Banner nach Click (info, auto-dismiss 5 s): „Download gestartet. Falls nichts passiert, Pop-Up-Blocker prüfen."
  - aria-live-Hint: identisch zur Banner-Microcopy (höflich).
- **Helper-Reuse:** `showAlert(podcastAlertContainer, 'info', msg, { autoDismissMs: 5000 })` ersetzt den heutigen `showToast('✓ Podcast heruntergeladen')`-Call. Code-Anker: [static/js/audio_converter.js:860-866](../static/js/audio_converter.js#L860-L866) Click-Handler des Download-Btns.
- **Aufwand:** XS — `showToast` → `showAlert` mit anderem Level + Microcopy.
- **Impact-Score:** 2 × 5 / 1 = **10.0**
- **Synergie-Hinweis:** Wenn P2 den Re-Download-Btn ergänzt, wird der Browser-Block-Fall ohnehin via Re-Download-Klick recovered — die Microcopy hier ist die User-Sichtbarkeits-Brücke zur Re-Download-Aktion.

---

### Pattern 7: Loading-State-Konvergenz (Spinner-Animation + Pattern-Einheitlichkeit)
**Adressiert Findings:** F7 (H4 Sev 2), F16 (H1 Sev 1)
**Cluster:** 2 (Async-State-Visibility)
**🔥 Smoke-Pflicht in F4-IMPL** (für F16-Teil — Spinner-Animations-Verhalten ist visuell, code-only-Inventur reicht nicht)

- **Pattern:** Beide Generate-Buttons (`#generate-script-btn` und `#generate-podcast-btn`) bekommen ein einheitliches Loading-Pattern: CSS-Klasse `.is-loading` toggelt einen Spinner (analog zur existierenden `.mic-button__spinner`-Klasse in `style.css`) plus `disabled=true` plus optional Text-Swap. Die heutigen zwei verschiedenen Patterns (`#generate-script-btn` Text-Swap, `#generate-podcast-btn` Hidden-Toggle plus Cancel-Btn-Übernahme) werden auf das CSS-Klassen-Pattern konvergiert:
  - **Skript-Generieren-Btn**: `.is-loading` + Text-Swap auf „Generiert …" (DE-Microcopy aus F-2.3 P12 bereits etabliert) + Spinner-Span vor dem Text.
  - **Podcast-Generieren-Btn**: `.is-loading` triggert die heutige Hidden-Toggle-Logik weiterhin (Btn wird komplett versteckt, Cancel-Btn übernimmt — siehe P1 für Cancel-Mechanik), aber das `.is-loading`-Marker wird konsistent gesetzt, damit `aria-busy=true` einheitlich auf beiden Buttons funktioniert.
  - **F16-Spezifik**: das `.generate-script-btn__icon`-SVG bekommt eine CSS-Animation, die nur unter `.is-loading` aktiv ist (`@keyframes podcast-icon-spin` oder Reuse der `.mic-button__spinner`-Animation). Dadurch wirkt der Btn während Loading visuell aktiver.
- **Visuelle Hinweise:** Spinner-Animation 1 s linear infinite (analog Mic-Spinner). Übergang `.is-loading`-on-toggle 100 ms `opacity`. `aria-busy="true"` während Loading auf beiden Buttons.
- **Microcopy** (DE, Du-Form):
  - keine neuen Microcopy-Strings — Button-Labels und Loading-Texte sind bereits aus F-2.3 P12 in DE.
  - aria-busy-Hint ist via Attribut, kein User-sichtbarer Text.
- **Helper-Reuse:** keine `_utils.js`-Helper-Erweiterung. Reine CSS- + Markup-Konvergenz. Code-Anker: [static/js/audio_converter.js:780-790](../static/js/audio_converter.js#L780-L790) Skript-Generate-Btn-Loading-Toggle; [static/js/audio_converter.js:830-870](../static/js/audio_converter.js#L830-L870) Podcast-Generate-Btn-Loading-Toggle; [static/css/style.css](../static/css/style.css) `.mic-button__spinner` als Vorlage.
- **Aufwand:** S — `.is-loading`-CSS-Klasse + Spinner-Span im Markup beider Buttons + Toggle-Logik in beiden Handler-Branches + CSS-Keyframes (oder Reuse der bestehenden `.mic-button__spinner`-Animation).
- **Impact-Score:** 2 × 5 / 2 = **5.0** (höchste Sev der adressierten Findings: F7 mit Sev 2)
- **Konsolidierung:** F7 (Inkonsistenz Hidden-Toggle vs. Text-Swap) und F16 (Spinner-Icon statisch) gehören thematisch zur selben Pipeline-Stage „Loading-State-Visualisierung der Generate-Buttons". Eine Konvergenz auf CSS-Klassen-Pattern löst beide gleichzeitig — das Hidden-Toggle bleibt für den Podcast-Btn (Cancel-Btn-Übernahme braucht das), aber der Skript-Btn-Spinner und das einheitliche `.is-loading`-Marker konvergieren das Visual-Vokabular.

---

### Pattern 8: Polling-Robustheit (Status-Validation + RQ-Status-Branches)
**Adressiert Findings:** F8 (H9 Sev 2), F15 (H9 Sev 1)
**Cluster:** 3 (Polling- und Defensiv-Robustheit)
**🔥 Smoke-Pflicht in F4-IMPL** (F8 ist code-only — F-4.1 Test-Anleitung 5: Network-Drop in DevTools, Polling-Loop-Verhalten verifizieren)

- **Pattern:** Polling-Loop in `audio_converter.js` wird defensiv robuster:
  1. **`r.ok`-Check vor Body-Parse** — wenn `!r.ok`, dann nicht in den `safeJSON`-Pfad fallen, sondern HTTP-Status klassifizieren: 404 (Job-Result-TTL abgelaufen) → eigener Banner; 5xx → Polling pausieren mit exponential backoff (max 3 Retries), dann Banner.
  2. **Status-Allowlist** — der Loop akzeptiert nur explizite Status-Werte (`queued`, `started`, `finished`, `failed`, `cancelled`, `deferred`, `scheduled`, `stopped`, `canceled`). Unbekannter Status (z.B. RQ-Major-Version-Bump, der neue States einführt) → Banner „Unbekannter Status — bitte erneut versuchen." statt stillem Loop-Abbruch.
  3. **`deferred`/`scheduled`/`stopped`/`canceled`-Branches** — explizit gehandhabt (heute fehlen sie). `deferred`/`scheduled` zeigen „Job wartet auf Bedingung …" (Sub-Caption analog P5); `stopped`/`canceled` zeigen den Cancel-Banner aus P1.
  4. **Frontend-Timeout** — wenn Polling >10 min läuft ohne `finished`/`failed`/`cancelled`, Banner „Generierung dauert länger als erwartet — Worker-Logs prüfen." plus Option zum manuellen Cancel-Trigger (P1).
- **Visuelle Hinweise:** keine neuen Komponenten. Banner über `showAlert(podcastAlertContainer, level, msg)`. Sub-Captions im Counter-Bereich (siehe P3).
- **Microcopy** (DE, Du-Form, max 2 Sätze):
  - 404 (Job-TTL): „Job-Status nicht mehr verfügbar. Bitte neu generieren."
  - 5xx (nach 3 Retries): „Status-Server reagiert nicht. Verbindung prüfen oder Container neu starten."
  - Unbekannter Status: „Unbekannter Job-Status — bitte erneut versuchen."
  - `deferred`/`scheduled` Sub-Caption: „Job wartet auf Bedingung …"
  - Frontend-Timeout (>10 min, info): „Generierung dauert länger als erwartet — Worker-Logs prüfen."
- **Helper-Reuse:** `showAlert(podcastAlertContainer, 'danger', msg)` für 404/5xx; `showAlert(podcastAlertContainer, 'warning', msg)` für unbekannten Status; `showAlert(podcastAlertContainer, 'info', msg)` für Frontend-Timeout. `safeJSON` bleibt für den Body-Parse, aber nur nach `r.ok`-Check. Code-Anker: [static/js/audio_converter.js:837-852](../static/js/audio_converter.js#L837-L852) Polling-Loop; [app_pkg/podcasts.py:181-191](../app_pkg/podcasts.py#L181-L191) Backend-Endpoint.
- **Aufwand:** S — Polling-Loop-Erweiterung (4 Branches + Timeout-Timer + Retry-Counter) + Microcopy. Backend-Status-Mapping für `deferred`/`scheduled`/`stopped`/`canceled` ergänzen (kleiner Backend-Touch).
- **Impact-Score:** 2 × 5 / 2 = **5.0**
- **Konsolidierung:** F8 (Polling-Loop kein Timeout, kein `r.ok`-Check) und F15 (RQ-Statuses ungehandelt) gehören zur selben State-Maschine: Polling-Edge-Cases. Eine Lösung (defensive Robustheit der Polling-Loop) erfüllt beide. F15 wird beim Apply von P1 (Backend-Cancel) live, weil dann `cancelled`-Status real wird — beide Patterns gehören zusammen verifiziert (P1 + P8).

---

### Pattern 9: Backend-Validation-Konvergenz (Allowlists + Max-Length-Hint)
**Adressiert Findings:** F13 (H9 Sev 1), F14 (H9 Sev 1)
**Cluster:** 3 (Polling- und Defensiv-Robustheit)

- **Pattern:** Konvergenz auf das F-013-Allowlist-Niveau für die `/generate-gemini-podcast`-Route plus Frontend-Max-Length-Hint für `raw_text`:
  1. **`/generate-gemini-podcast` Allowlist** — Backend validiert `dialogue` (Schema-Form: Liste von Speaker/Text-Objekten), `language` (Allowlist `['de', 'en', …]`), `tts_model` (Allowlist `['gemini-2.5-flash-preview-tts', 'gemini-2.5-pro-preview-tts']`). Bei kaputtem Input 400-Antwort mit DE-JSON-Body statt späterem Worker-Crash.
  2. **`/format-dialogue-with-llm` `raw_text`-Max-Length** — Backend validiert `len(raw_text) <= MAX_RAW_TEXT_CHARS` (z.B. 50 000 Zeichen). Frontend zeigt einen kleinen Counter unter der `#podcast-script`-Textarea (oder dem `raw_text`-Source-Input) — bei >80% Limit dezent gelb, bei >100% rot mit Banner.
- **Visuelle Hinweise:** Counter unter der Textarea, dezent grau bis 80%, gelb bei 80-100%, rot bei >100%. Banner via `showAlert(podcastAlertContainer, 'danger', msg)` bei 400-Antwort vom Backend.
- **Microcopy** (DE, Du-Form, max 2 Sätze):
  - Counter-Format: „{n}/{max} Zeichen"
  - Banner Backend-400 für ungültiges `language`/`tts_model`/`dialogue`: „Ungültiger Eingabewert. Bitte Sprache, TTS-Modell und Skript-Format prüfen."
  - Banner Frontend-Pre-Submit (Max-Length): „Quelltext ist zu lang. Maximum: 50 000 Zeichen."
  - Banner Backend-400 für Max-Length-Überschreitung (Race-Case bei großen Pastes): „Quelltext ist zu lang. Bitte kürzen."
- **Helper-Reuse:** `showAlert(podcastAlertContainer, 'danger', msg)` für die Banner. Code-Anker: [app_pkg/podcasts.py:138-164](../app_pkg/podcasts.py#L138-L164) `/generate-gemini-podcast`-Endpoint; F-013-Allowlist-Konvention für `/format-dialogue-with-llm` als Vorlage; [templates/audio_converter.html](../templates/audio_converter.html) für Counter-Span unter der Textarea.
- **Aufwand:** S — Backend-Allowlist-Validierung (analog F-013-Pfad) + Frontend-Counter-Render + Microcopy. Single-Source-of-Truth für Max-Length-Konstante (`app_pkg/config.py`).
- **Impact-Score:** 1 × 5 / 2 = **2.5**
- **Konsolidierung:** F13 (raw_text Max-Length) und F14 (Allowlist-Gap auf `/generate-gemini-podcast`) sind zwei Defensiv-Lücken auf demselben Backend-Validation-Hebel. Eine Lösung (F-013-Niveau auf beide Endpoints anwenden) erfüllt beide.

---

### Pattern 10: Skript-Format-Hilfe (Recognition statt Recall)
**Adressiert Findings:** F11 (H6 Sev 2)
**Cluster:** 4 (Speaker-Format-Hilfe und Edit-Verhalten)

- **Pattern:** Skript-Editor (`#podcast-script`-Textarea) bekommt eine inline-Schema-Hilfe statt nur Placeholder-Beispiel:
  1. **Sichtbare Format-Erklärung** unter der Textarea (kleine Hint-Zeile in dezentem Text-Tint): zeigt das exakte Format `Sprecher [stil]: Text` plus die unterstützten Sprecher-Namen plus den Skip-Branch für `#`/`**`-Zeilen.
  2. **Inline-Validierungs-Hint** — beim Blur (`change`-Event) der Textarea wird das Skript pre-parsed (gleiche Logik wie der Generate-Btn-Handler). Bei Empty-`dialogue[]`-Ergebnis erscheint ein dezenter Warning-Banner mit konkretem Hinweis auf die erste fehlerhafte Zeile.
  3. **Voice-Map-Sichtbarkeit** — die heute hardcoded Voice-Map (`{Kate→Zephyr, Max→Charon, default→Kore}`) wird in der Hint-Zeile erwähnt, damit User die Speaker-Namen kennt, ohne den Code lesen zu müssen.
- **Visuelle Hinweise:** Hint-Zeile unter Textarea, kleiner Schrift-Grad, Text-Tint `var(--nm-text-faint)`. Warning-Banner via `showAlert(podcastAlertContainer, 'warning', msg)` mit Auto-Dismiss 8 s (User braucht Zeit zum Lesen).
- **Microcopy** (DE, Du-Form, max 3 Sätze für Hint, max 2 Sätze für Banner):
  - Hint-Zeile (3 Sätze): „Format: `Sprecher [stil]: Text` pro Zeile. Sprecher: Kate, Max oder eigener Name (Default-Stimme: Kore). Zeilen mit `#` oder `**` werden ignoriert."
  - Validierungs-Banner bei Empty-`dialogue[]`: „Skript konnte nicht gelesen werden. Format prüfen: `Sprecher [stil]: Text`."
  - Validierungs-Banner mit Zeilen-Hinweis: „Skript konnte nicht gelesen werden — Zeile {n}: kein `:`-Trenner gefunden."
- **Helper-Reuse:** `showAlert(podcastAlertContainer, 'warning', msg)` für Validierungs-Banner. Code-Anker: [static/js/audio_converter.js:740-760](../static/js/audio_converter.js#L740-L760) Generate-Btn-Handler-Parser.
- **Aufwand:** M — Hint-Zeile + Inline-Validierungs-Pre-Parse + Banner-Microcopy + Voice-Map-Sichtbarkeit. Wenn Voice-Map künftig konfigurierbar wird (Helper-Vorschlag-Pfad), ist das ein größerer Touch — bleibt hier explizit auf Hint-Zeile beschränkt.
- **Impact-Score:** 2 × 5 / 4 = **2.5**

---

### Pattern 11: Skript-Textarea-Readonly während Generierung
**Adressiert Findings:** F12 (H6 Sev 1)
**Cluster:** 4 (Speaker-Format-Hilfe und Edit-Verhalten)

- **Pattern:** Helper `setPodcastScriptReadonly(boolean)` analog zu F-2.3 P10 (`setLiveTextareaReadonly`) — `#podcast-script` wird `readonly=true` während der Generierung läuft (Worker arbeitet mit eingefrorener `dialogue[]`-Snapshot, User-Edits hätten keine Auswirkung). Nach `finished`/`failed`/`cancelled` wird `readonly=false` zurückgesetzt. Visual-Hint analog F-2.3 P10: graue Border-Tint während readonly, Pulse-Übergang 200 ms beim Stop.
- **Visuelle Hinweise:** `readonly`-Style: graue Border-Tint (`var(--nm-tint-faint)`), Cursor unverändert (User soll lesen können), Tooltip beim Hover. Übergang zum editierbaren State per Border-Color-Pulse 200 ms beim Job-Ende.
- **Microcopy** (DE, Du-Form):
  - Tooltip auf der readonly Textarea während Generierung: „Während der Generierung schreibgeschützt"
  - Subtiler Status-Hint unter Textarea (dezent kursiv): „Generierung läuft — Skript-Edit erst nach Stop oder Abschluss."
- **Helper-Reuse:** keine `_utils.js`-Helper. Lokale Hilfsfunktion `setPodcastScriptReadonly(boolean)` in `audio_converter.js`. Code-Anker: [static/js/audio_converter.js:830-870](../static/js/audio_converter.js#L830-L870) Generate-Btn-Handler-Sequenz.
- **Aufwand:** XS — Helper-Funktion (3 Zeilen) + Toggle-Aufrufe an Job-Start/-Ende.
- **Impact-Score:** 1 × 5 / 1 = **5.0**

---

### Pattern 12: Tab-Disabled-Konsistenz zwischen Markup und JS
**Adressiert Findings:** F9 (H4 Sev 2)
**Cluster:** kein Cluster (intra-feature H4)

- **Pattern:** Tab-Switching-Logik in `audio_converter.js` prüft `aria-disabled="true"` plus `aria-disabled === 'true'` vor Click-Switch. Wenn der Tab als disabled markiert ist (z.B. weil `gemini_api_key_set === false` für den Podcast-Tab), bricht der Click-Handler ab und zeigt einen kurzen `info`-Toast statt des Pane-Switchs. Optional CSS `.tab-btn[aria-disabled="true"] { pointer-events: none; opacity: 0.5; }` als zusätzlicher Schutz.
- **Visuelle Hinweise:** disabled-Tab visuell deutlicher (via CSS-Rule oben) — Opacity reduziert, Cursor `not-allowed`, Hover-Style off.
- **Microcopy** (DE, Du-Form, max 2 Sätze):
  - Toast bei Click auf disabled Tab (info, auto-dismiss 4 s): „Service nicht konfiguriert."
  - Tooltip auf disabled-Tab-Btn (heute schon `title="Service nicht konfiguriert"`): bleibt unverändert.
- **Helper-Reuse:** `showToast(msg, { level: 'info' })` für den Hinweis. Code-Anker: [static/js/audio_converter.js:35-54](../static/js/audio_converter.js#L35-L54) Tab-Switching-Loop.
- **Aufwand:** XS — Click-Handler-Branch + CSS-Rule + Microcopy.
- **Impact-Score:** 2 × 5 / 1 = **10.0**

---

## Bug-Tickets ohne UX-H-Komponente (nicht in F-4.3)

Aus den Stage-2-Bug-Tickets ist eines explizit nicht in F-4.3 adressiert (Sprint-Prompt Out-of-scope):

- **BT4: Audio-Blob-URL wird nicht via `URL.revokeObjectURL` freigegeben.** Heute User-unsichtbar (reine Memory-Hygiene), kein UX-H-Aspekt — H1/H4/H6/H9 treffen alle nicht. Code-Anker: [static/js/audio_converter.js:866-871](../static/js/audio_converter.js#L866-L871). **Pure Bug-Ticket, kein UX-H-Aspekt.** Gehört in einen Sammel-Bug-Pass oder wird mit-genommen, wenn der Download-Pfad für P2 (Re-Download-Btn) ohnehin angefasst wird — der `URL.revokeObjectURL`-Aufruf passt thematisch in den Re-Download-Lifecycle.

BT1–BT3 sind in den Patterns ihrer verknüpften Findings adressiert (siehe Pattern-Block-Header):
- BT1 ↔ P1 (Cancel-Mechanik)
- BT2 ↔ P3 (Stage-Progress)
- BT3 ↔ P2 (File-Cleanup + Re-Download)

---

## Cluster-Vorbereitung für Implementation

**Drei-Cluster-Default — Cluster I = 2 Patterns, Cluster II = 5 Patterns, Cluster III = 5 Patterns.**

**F-3-IMPL-Lehre angewendet:** Bei 12 Patterns liegt der Sprint im 1-Sprint-mit-Sub-Batches-Bereich (Schmerzgrenze 12). 3 Sub-Batches in einem F4-IMPL-Sprint sind pragmatisch, falls die Cluster disjunkt bleiben. Falls F4-IMPL-Sub-Thread Cluster I als zu schwer empfindet (L-Aufwand auf P1, RQ-Recherche), kann Cluster I als eigener Sprint vorgezogen werden.

### Cluster I (Cancel-und-Cleanup-Recovery)

Patterns: **P1, P2** — 2 Patterns.

Begründung Gruppierung: beide adressieren Cluster 1 aus den Findings (Cancel-Lüge + File-Cleanup). Beide tragen `✅ live-verifiziert` aus dem Master-Walkthrough. Code-Touch ist Backend-heavy (Worker-cooperative-cancel + neuer Cancel-Endpoint + TTL-Cleanup-Strategie + Re-Download-Pfad). RQ-Recherche-Punkt aus Sprint-Prompt-Sektion „Konstitutiv mit-genommen": rq 2.x cooperative cancel via redis-key-poll vs. `Worker.kill_horse()`-Emulation — F4-IMPL-Sub-Thread löst das auf.

**Smoke-Sequenz vor Apply:** Test-Anleitung 3 aus F-4.1 wiederholen — Generate-Flow starten, ~30 s warten, Cancel klicken → Worker-Stop verifizieren via `docker logs converter-worker -f`. Test-Anleitung 7 für Re-Download-Pfad. Beides ist durch Master-Walkthrough live-verifiziert, F4-IMPL braucht den Smoke vor Apply nicht erneut, sondern als Verifikation **nach** Apply.

### Cluster II (Async-State-Visibility)

Patterns: **P3, P4, P5, P6, P7** — 5 Patterns.

Begründung Gruppierung: alle adressieren Cluster 2 aus den Findings (Async-State-Visibility). P3 ist die größte Investition (M, `job.meta`-Hebel mit potentieller Sackgasse). P4 trägt `🔥 Smoke-Pflicht`. P7 trägt `🔥 Smoke-Pflicht` für den F16-Spinner-Animations-Teil. P5 + P6 sind XS — gut als Aufwärm-Items zu Beginn des Clusters. Reihenfolge-Empfehlung: P5 → P6 → P4 → P7 → P3 (XS zuerst, M am Ende mit RQ-Subprocess-Sync-Verifikation).

**Smoke-Sequenz vor Apply:** Test-Anleitung 4 (Browser-Reload) für P4. Visueller Vergleich Mic-Spinner vs. Generate-Btn-Icon-Animation für P7. Test-Anleitung 2 (Multi-Chunk) für P3-Stage-Progress-Visualisierung.

### Cluster III (Defensiv-Robustheit + Skript-Hilfe + Tab-Disabled)

Patterns: **P8, P9, P10, P11, P12** — 5 Patterns.

Begründung Gruppierung: Mischung aus Defensiv-Patterns (P8, P9, P12), Skript-Editor-Polish (P10, P11). Niedrige Sev (1-2), kein Smoke-Pflicht-Druck (außer P8). Reihenfolge frei wählbar; sinnvoll: P12 (XS, isoliert) zuerst, dann P11 (XS, isoliert), dann P8 (S, mit P1-Verzahnung weil F15 dann live wird), dann P9 (S, Backend-Touch), dann P10 (M, Skript-Hilfe).

**Hinweis F15-Verzahnung:** P8 löst F15 (RQ-Statuses ungehandelt) mit. F15 wird live, sobald P1 (Backend-Cancel) implementiert ist — `cancelled`-Status existiert dann real. P8 sollte daher entweder gleichzeitig mit oder direkt nach Cluster I gemerged werden, nicht zu lange später.

**Smoke-Sequenz vor Apply:** Test-Anleitung 5 (Network-Drop in DevTools) für P8.

### Zwei-Cluster-Empfehlung (falls Cluster I als zu schwer empfunden wird)

P1 ist der schwerste einzelne Pattern-Block in F-4.3 (L-Aufwand wegen Worker-cooperative-cancel + Backend-Endpoint + Frontend-State-Machine). Falls F4-IMPL-Sub-Thread P1 als eigenen Sprint vorziehen will:

- **Sprint F4-IMPL-A (Cancel-Sprint):** P1, P2 — 2 Patterns. Eigener Sprint mit RQ-Recherche-Phase 0.
- **Sprint F4-IMPL-B (Async-State + Polish):** P3, P4, P5, P6, P7, P8, P9, P10, P11, P12 — 10 Patterns. Sub-Batch-Strategie analog F-3-IMPL.

Master entscheidet beim Schreiben der F4-IMPL-Sprint-Prompts, ob 1-Sprint-mit-3-Sub-Batches oder 2-Sprint-Split.

---

## Top-5 Quick-Wins

**Aufwand-Gewicht:** XS=1, S=2, M=4, L=8. Score = Sev × 5 / Aufwand-Gewicht. Höher = besser.

| Rang | Pattern # | Adressiert | Sev | Aufwand | Impact-Score | Quick-Win |
|------|-----------|------------|-----|---------|--------------|-----------|
| 1 | P5 | F6 — Backend-Status `queued` vs. `started` | 2 | XS | 10.0 | ★ Top-5 |
| 2 | P6 | F10 — Download-Toast-Guard | 2 | XS | 10.0 | ★ Top-5 |
| 3 | P12 | F9 — Tab-Disabled-Konsistenz | 2 | XS | 10.0 | ★ Top-5 |
| 4 | P4 | F5 — Browser-Reload LocalStorage | 3 | S | 7.5 | ★ Top-5 |
| 5 | P11 | F12 — Skript-Textarea-Readonly | 1 | XS | 5.0 | ★ Top-5 |
| 6 | P7 | F7, F16 — Loading-State-Konvergenz | 2 | S | 5.0 | |
| 7 | P8 | F8, F15 — Polling-Robustheit | 2 | S | 5.0 | |
| 8 | P2 | F4 — File-Cleanup + Re-Download | 3 | M | 3.75 | |
| 9 | P3 | F3 — Stage-Progress (`job.meta`) | 3 | M | 3.75 | |
| 10 | P1 | F1, F2 — Cancel-Mechanik | 4 | L | 2.5 | |
| 11 | P9 | F13, F14 — Backend-Validation | 1 | S | 2.5 | |
| 12 | P10 | F11 — Skript-Format-Hilfe | 2 | M | 2.5 | |

**Top-5 Quick-Wins:**

1. **P5 — Backend-Status-Differentiation `queued` vs. `started`** (10.0): Backend-Mapping eine Zeile, Frontend-Branch eine Sub-Caption. Diagnostik-Wert für Single-User-Worker-Restart-Pfade. XS-Aufwand mit klarem Hebel.
2. **P6 — Download-Toast-Guard** (10.0): einzige Microcopy-Änderung — `showToast`-Call durch `showAlert` ersetzen. Beseitigt eine Sev-2-System-Status-Lüge. Synergie mit P2 Re-Download-Pfad.
3. **P12 — Tab-Disabled-Konsistenz** (10.0): Click-Handler-Branch plus eine CSS-Rule. Schließt einen Markup-vs-JS-Konsistenz-Bruch im Service-Gate-Pfad.
4. **P4 — Browser-Reload-Recovery via LocalStorage** (7.5): höchster Sev (3) der XS/S-Patterns. `localStorage`-Persistierung des `job_id` plus Re-Attach-Branch. Smoke-Pflicht.
5. **P11 — Skript-Textarea-Readonly** (5.0): Helper analog F-2.3 P10 für Live-Textarea, eine Toggle-Funktion plus zwei Aufrufe. Schließt Recognition-Bruch „Eingabe-State eingefroren während Maschine arbeitet".

P1 (Cancel-Mechanik), P2 (File-Cleanup) und P3 (Stage-Progress) liegen mit niedrigerem Score, sind aber die schwersten Pflicht-Fixes des Sprints (Sev 4 / Sev 3, Cluster 1 + 2). Sie gehören nicht in die Quick-Wins, sondern in die Pflicht-Cluster — ihr Score reflektiert den Aufwand, nicht ihre Priorität.

---

## Smoke-Pflicht-Übersicht

### `✅ live-verifiziert` (Master-Walkthrough 2026-05-10)

| Pattern | Adressiert | Walkthrough-Befund |
|---------|-----------|-------------------|
| **P1** | F1 (Cancel-Lüge — Worker läuft weiter) + F2 (Frontend-Status zeigt erfolgreichen Cancel) | Worker lief 1:12 Min nach Cancel weiter, TTS-Token vollständig verbraucht, WAV 11.7 MB im Volume; Frontend-UI „unauffällig" — aktive Lüge bestätigt. |
| **P2** | F4 (File-Cleanup-vs-Re-Download) | Orphaned WAV bleibt im Volume liegen weil Cleanup nur beim Download-Pfad triggert — Disk-Wachstum nach abgebrochenen Jobs bestätigt. |

**Anzahl live-verifizierter Patterns:** 2 (von 12). Adressiert 3 Findings (F1, F2, F4). F4-IMPL-Sub-Thread braucht für diese Patterns keinen vorab-Smoke, sondern den Verifikations-Smoke nach Apply (Worker-Stop reproduzierbar, TTL-Cleanup-Job läuft).

### `🔥 Smoke-Pflicht in F4-IMPL`

Patterns für `⚠️ code-only`-Findings, die Master-Walkthrough nicht abgedeckt hat:

| Pattern | Adressiert | Smoke-Mechanik (vor Apply) |
|---------|-----------|----------------------------|
| **P4** | F5 (Browser-Reload = `job_id` verloren) | F-4.1 Test-Anleitung 4: Generate starten → Reload mid-Polling → Worker-Continue verifizieren plus Output-File-Verbleib im Volume. |
| **P7** | F16 (Spinner-Icon statisch — F7-Teil ist live) | Visueller Vergleich Mic-Button-Spinner-Animation vs. `#generate-script-btn`-Icon (heute statisch); CSS-Animation-Ergebnis live im Browser prüfen. |
| **P8** | F8 (Polling-Loop kein Timeout, kein `r.ok`-Check; F15 ist nicht code-only-markiert) | F-4.1 Test-Anleitung 5: DevTools-Network-Throttle auf „Offline" mid-Polling → Loop-Verhalten beim 5xx/Timeout konkret prüfen. |

**Anzahl Smoke-Pflicht-Patterns:** 3 (von 12). Adressiert 3 verbleibende `⚠️ code-only`-Findings (F5, F8, F16). F7 ist nicht code-only und wird über P7 als Konvergenz-Pattern mit-adressiert; F15 ist nicht code-only-markiert (theoretischer Defensiv-Lücken-Befund).

**Anzahl Patterns ohne Smoke- oder Live-Tag:** 7 (P3, P5, P6, P9, P10, P11, P12) — statisch verifizierbar oder Pattern-Apply-Verifikation als Default-Smoke ausreichend.

---

## Helper-Vorschläge (für F4-IMPL-Sub-Thread zur Entscheidung)

Beim Pattern-Schreiben ist ein möglicher neuer `_utils.js`-Helper aufgefallen, der für künftige Wiederverwendung sinnvoll wäre — **nicht** still im jeweiligen Pattern mit-anlegen, sondern F4-IMPL-Sub-Thread entscheidet, ob der Helper im Pattern-Cluster mit-implementiert oder als separater Helper-Cluster vorgezogen wird:

- **`confirmInPlace(triggerEl, options)`** — generischer In-Page-Two-Stage-Confirm-Helper analog zur F-2.3 P17 Clear-Confirmation-Mechanik. Heute in P1 (Cancel-Confirmation) und implizit in F-2.3 P17 verwendet. Wenn künftig weitere Confirm-Stages auftauchen (z.B. Library-Bulk-Delete), wird der Helper Cross-Feature relevant. Bis dahin reicht die inline State-Machine-Variante in `audio_converter.js`.

**Disposition:** Helper bleibt im jeweiligen Pattern-Block als „Helper-Vorschlag" markiert; F4-IMPL-Sub-Thread entscheidet beim Cluster-Schnitt.

---

## Architektur-Hebel-Bedenken (für F4-IMPL-Sub-Thread)

Aus dem Sprint-Prompt-Abschnitt „Konstitutiv mit-genommen, falls berührt":

- **P1 RQ-Cooperative-Cancel-Recherche-Punkt:** Sprint-Prompt-Frage „kann rq 2.x cooperative cancel via redis-key-poll, oder müssen wir `Worker.kill_horse()` emulieren?". F-4.3 löst das **nicht** — F4-IMPL-Sub-Thread macht die Recherche und entscheidet. Pattern-Block beschreibt beide Varianten als gleichberechtigte Sub-Optionen. Code-Touch-Implikation: cooperative cancel via redis-key-poll ist deutlich einfacher (mid-TTS-Chunk-Check), `Worker.kill_horse()` braucht Worker-Konfigurations-Anpassung.
- **P3 `job.meta`-Subprocess-Sync-Risiko:** RQ-Worker-Subprocess-Konfiguration mit eigenem Redis-Pool kann zeitweise inkonsistente `job.meta`-Sichten erzeugen. Architektur-Hebel-Plan A scheitert nicht an der Idee, sondern ggf. an der Sync-Latenz. Plan B: direkt in Redis-Key `podcast:stage:{job_id}` schreiben statt `job.meta`-Pfad. F4-IMPL-Sub-Thread verifiziert mit dem ersten echten Multi-Chunk-Lauf, ob Plan A reicht.

---

**Schweregrad-Skala (aus Stufe 2):**
1. kosmetisch (kaum spürbar)
2. gering (nur in Edge-Cases störend)
3. mittel (regelmäßig spürbar, frustrierend)
4. kritisch (verhindert/verfälscht die primäre Aufgabe oder produziert falsche Ergebnisse / Datenverlust- oder Cost-Pfad)
