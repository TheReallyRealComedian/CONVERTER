# UX-Heuristik-Findings: podcast-flow (2026-05-10)

**Methodik:** Stufe 2 der Duan-Kaskade (Duan et al., *Heuristic Evaluation with LLMs*, CHI 2024). Heuristisches Review der strukturierten Inventur aus Stufe 1.
**Quelle:** [docs/ui_inventory_podcast_flow_2026-05.md](ui_inventory_podcast_flow_2026-05.md) — 17 echte interaktive Elemente plus 3 Async-State-Pseudo-Elemente, 19 Befunde plus 1 Meta-Beobachtung (Befund 20).
**Heuristiken:** Nielsen H1 (Sichtbarkeit des Systemzustands), H4 (Konsistenz und Standards), H6 (Wiedererkennen statt Erinnern), H9 (Fehlermeldungen / Hilfe bei Fehlern).
**Produkt-Kontext:** Single-User (Oliver), LAN-only, login-protected. **`podcast-flow` ist der Long-Running-Async-Flow** — Quelltext einfügen → Mode wählen → Generieren → warten (Minuten-Bereich, Multi-Chunk bei langen Skripten) → Download. **Long-running async ist der zentrale UX-Reibungspunkt**: H1 (System-Status) und H9 (Error-Recovery) dominieren wegen der 5/6 unzureichenden Async-State-Klassen aus F-4.1. Daily-Usage-Schmerz-Gewichtung relevant: wenn Oliver Podcasts häufig generiert (Reader-Workflow + Audio-Konsumption im Readwise-Ersatz-Kontext), schlagen async-Status-Lücken besonders zu — Cancel-Lüge (Worker brennt CPU+Token weiter), fehlender Stage-Progress (User weiß nicht ob 30s oder 8min noch warten), File-Cleanup-Disk-Wachstum.

**Cross-Feature-H4 niedrig erwartet:** F-4.1 Befund 20 hat dokumentiert, dass der `audio_converter.js`-Podcast-Block die Helper-Konvergenz aus F-2 sauber durchgezogen hat (`safeJSON` plus `showAlert` plus `showToast` plus `confirmIfLong` durchgehend, kein `alert()`-Pfad, keine drei konkurrierenden Error-UI-Patterns wie ursprünglich in F-2.1). Master-Annotation hat 15-25% Cross-Feature-H4-Quote vorhergesagt — Sub-Thread soll keine künstlichen H4-Findings konstruieren. **Tatsächliches Ergebnis nach Heuristik-Filter: 0% reine Cross-Feature-H4-Brüche** (siehe Cross-Feature-Sektion und Begründung unten).

**Live-Walkthrough-Hinweis:** F-4.1 ist Code-only-Inventur (Coverage geschätzt 65%). 5 Code↔live-Divergenz-Verdachte und 7 Live-Walkthrough-Test-Anleitungen wurden in F-4.1 dokumentiert. Findings, deren visueller Effekt aus reinem Code-Reading nicht endgültig beurteilbar ist (z.B. Worker-Verhalten nach Cancel, Network-Drop-Recovery beim Polling, Browser-Download-Block-Verhalten), sind in der Severity-Spalte mit `⚠️ code-only` gekennzeichnet — Master kann zwischen F4-REVIEW und F4-PATTERNS Walkthrough-Nachreichung erwägen, gerade weil Async-Übergänge in F4-IMPL schwer nachzu-Smoke-en sind (spezifische Multi-Chunk-/Network-Drop-/Cancel-mid-Setups).

---

## Findings (sortiert absteigend nach Schweregrad)

| #   | Element / Befund | Problem (1–2 Sätze) | Heuristik | Schweregrad (1–4) | Inventur-Anker | Disposition |
|-----|------------------|---------------------|-----------|-------------------|----------------|-------------|
| F1  | Cancel-Btn ist Frontend-Lüge — Worker läuft weiter (Inventur Befund 9; #20) | Click auf "Abbrechen" setzt nur `podcastCancelRequested = true` und stoppt das Frontend-Polling. Der RQ-Worker arbeitet bis zur natürlichen Job-Fertigstellung weiter (kann Minuten dauern), generiert das WAV-File, schreibt es in `OUTPUT_DIR` — wo es nie heruntergeladen wird und liegen bleibt. **Strukturanalogon zur F-2.1 „Drag-Drop-Lüge"**: Label/Knopf verspricht eine Funktion, die JS/Backend nicht erfüllt. Sichtbarer System-Status („Generierung abgebrochen") widerspricht der Realität (Worker brennt CPU plus Gemini-TTS-Token plus Disk weiter). | H1 | **4** ⚠️ code-only | Inventur Befund 9 | Finding + Bug-Ticket BT1 |
| F2  | Cancel-Recovery + Cost-Verschwendung (siehe F1) | Da der Frontend-Cancel keinen `Job.cancel()`/`send_stop_job_command` auslöst, fehlt jede Recovery-Möglichkeit für den Worker („wirklich abbrechen, Token sparen"). User-Erwartung beim Cancel ist „spar mir den Gemini-Credit"; Realität ist „Credit voll konsumiert plus Disk wächst". Frontend-Microcopy „Backend-Job läuft im Hintergrund weiter" ist ehrlich, aber kein Recovery-Pfad. | H9 | **4** ⚠️ code-only | Inventur Befund 9 | Finding + Bug-Ticket BT1 |
| F3  | Stage-Progress fehlt — Worker pflegt kein `job.meta` (Inventur Befund 4; #19/#23) | Der Worker (`tasks.generate_podcast_task` plus `services/gemini/tts.py` Chunking-Loop) macht ausschließlich `logger.info` für Stage-Events (filter → split → chunking-decision → per-chunk-progress → concat). RQ stellt `job.meta` als persistierten Dict bereit, der vom Frontend per `/podcast-status/<job_id>` ausgelesen werden könnte — wird aber nie gepflegt. User sieht nur einen Wand-Sekunden-Counter im Cancel-Btn-Text („Abbrechen (Ns)"); ob 50%, 90% oder Stuck bei Multi-Chunk-Podcast (z.B. 5 Chunks à ~30s) ist nicht erkennbar. **Daily-Usage-Schmerz hoch**: Long-Running-Job ohne Fortschritts-Anzeige ist klassischer H1-Bruch und wird täglich beim Generieren spürbar. | H1 | **3** | Inventur Befund 4 | Finding + Bug-Ticket BT2 |
| F4  | File-Cleanup direkt nach erstem Download — kein Re-Download (Inventur Befund 18; #28/Backend `/podcast-download`) | `podcast_download` macht `os.unlink(real_path)` direkt nach `BytesIO`-Read. Wenn der Browser-Download fehlschlägt (Pop-Up-Blocker, Container-Restart zwischen Generate und Download, Network-Glitch beim Download-Roundtrip), kann der User nicht erneut auf `/podcast-download/<job_id>` triggern — der File ist weg, RQ-Job ist `finished` (nicht failed), aber `job.result` zeigt auf einen non-existenten Pfad → 404. Frontend-seitig gibt es keinen Re-Download-Knopf — der User muss den Generate-Flow erneut anstoßen, **TTS-Credit erneut verbrennen**. Recovery-Pfad fehlt komplett. | H9 | **3** ⚠️ code-only | Inventur Befund 18 | Finding + Bug-Ticket BT3 |
| F5  | Browser-Reload während Polling = `job_id` verloren (Inventur Befund 10; #21) | Kein LocalStorage/Cookie für laufenden `job_id`. User reloadet versehentlich → leerer Podcast-Pane → Worker läuft im Hintergrund weiter → File landet in `OUTPUT_DIR` und ist verwaist. **Sichtbarer System-Status verschwindet komplett**, obwohl der Job weiterläuft. Selbe Disk-Wachstums- und Credit-Verschwendungs-Familie wie F1/F2. | H1 | **3** ⚠️ code-only | Inventur Befund 10 | nur Finding |
| F6  | `queued` und `started` werden zu `processing` konflatiert (Inventur Befund 5; #22) | Backend-Endpoint `/podcast-status/<job_id>` mappt RQ-`queued` und RQ-`started` beide auf `{"status": "processing"}`. User-relevante Information geht verloren: „Worker hat den Job nicht aufgegriffen" (queued, kann auf Worker-Restart hindeuten) vs. „Worker arbeitet aktiv". Im Single-User-Setup mit einem Worker selten sichtbar, aber bei Worker-Restart oder Worker-Crash zwischen den Jobs würde der User keinen Hinweis bekommen, dass der Job hängt statt arbeitet. | H1 | **2** | Inventur Befund 5 | nur Finding |
| F7  | Loading-Pattern-Inkonsistenz: `.hidden`-Toggle vs. Text-Swap (Inventur Befund 8; #12 vs. #19) | `#generate-script-btn` zeigt Loading via Text-Swap „Generiert …" plus `.disabled`. `#generate-podcast-btn` zeigt Loading via `.hidden`-Toggle plus Cancel-Btn-Übernahme. **Innerhalb derselben Pane zwei verschiedene Loading-Patterns**. F-2 hat showAlert/showToast/Helper konvergiert, aber nicht die Loading-Toggle-Konvention. **Kein Cross-Feature-H4-Bruch** (beide Patterns existieren auch sonst in der App), sondern interner H4-Bruch innerhalb des Podcast-Tabs. | H4 | **2** | Inventur Befund 8 | nur Finding |
| F8  | Polling-Loop kein Frontend-Timeout, kein `r.ok`-Check (Inventur Befund 6; #21) | `while (status === 'pending' || 'processing')`-Loop fragt alle 2s `/podcast-status/<id>` ab und prüft nur `if (status === 'failed')`. Bei `r.ok===false` (z.B. 404 NoSuchJobError nach RQ-result-TTL-Ablauf, oder 500 transport-error) wird `r.json()` per `safeJSON` geparst; wenn der Body `{"error": "..."}` statt `{"status": "..."}` enthält, wird `status` zu `undefined`, der Loop bricht ab und fällt in den Download-Pfad mit ungültigem `job_id` → throw → generischer „Podcast-Generierung fehlgeschlagen"-Banner. Recovery-Anleitung spricht nicht den eigentlichen Fehler aus (Polling-Drop ≠ Generation-Failure). | H9 | **2** ⚠️ code-only | Inventur Befund 6 | nur Finding |
| F9  | Tab-Button "Text zu Podcast" `aria-disabled` ohne JS-Click-Block (Inventur Befund 1; #1) | Template setzt `aria-disabled="true"` plus `title="Service nicht konfiguriert"` wenn `!gemini_api_key_set`. JS in `audio_converter.js` (Zeilen 35-54) iteriert über alle `.tab-btn` und switcht ohne Disabled-Check — Pane wird trotzdem aktiviert, User sieht den Service-Gate-Banner und disabled-Buttons. **Markup verspricht Disabled-State, JS bricht das Versprechen** — interner H4-Bruch zwischen Template und JS. Im Single-User-Setup mit gesetztem Gemini-Key selten relevant; bei Service-Config-Issue spürbar. | H4 | **2** | Inventur Befund 1 | nur Finding |
| F10 | Download-Toast feuert immer grün, auch ohne echten Download (Inventur Befund 19; #28) | Click-Handler des `#download-podcast-btn`: `if (!downloadPodcastBtn.getAttribute('href')) return; showToast('✓ Podcast heruntergeladen')`. Guard nur gegen leeren `href`. Wenn Browser den Download blockt (Pop-Up-Blocker, `download`-Attribut nicht respektiert, Blob-URL revoked), feuert der Toast trotzdem grün-success — System-Status-Lüge. | H1 | **2** | Inventur Befund 19 | nur Finding |
| F11 | Skript-Parsing fragil — User muss Format perfekt erinnern (Inventur Befund 16; #19) | Parser in `generate-podcast-btn`-Handler erwartet exakt `Sprecher [stil]: Text` pro Zeile, mit `:`-Trenner. Skip-Branch für `#`/`**`-Zeilen, voiceMap hardcoded auf {Kate→Zephyr, Max→Charon, default→Kore}. Edge-Cases (Markdown-Sections wie `# Intro`, List-Items mit `:`, Sprecher-Namen mit Sonderzeichen, Zeilen ohne `:`-Trenner) gehen entweder in den Skip-Branch oder produzieren invaliden Speaker — Empty-`dialogue[]` triggert dann „Skript konnte nicht gelesen werden"-Banner. **Recognition-over-Recall**: User muss das Format aktiv aus Kopf reproduzieren, der Editor bietet keine Schema-Hilfe (außer dem Placeholder-Beispiel). | H6 | **2** | Inventur Befund 16 | nur Finding |
| F12 | Skript-Textarea editable während Podcast-Generierung (Inventur Befund 15; #13) | `#podcast-script` ist während des laufenden Generate-Jobs nicht readonly. User kann mid-pipeline tippen — hat **keine** Auswirkung auf den laufenden Job (Worker hat sein eigenes `dialogue[]` schon bekommen), kann aber User verwirren („ich habe es geändert, warum klingt der Podcast wie vorher?"). F-2 hat das Live-Transcript-Pattern (`setLiveTextareaReadonly`) etabliert; analog hier denkbar. **Recognition-over-Recall**: User-Erwartung „Eingabe-State ist eingefroren während die Maschine arbeitet" wird gebrochen. | H6 | **1** | Inventur Befund 15 | nur Finding |
| F13 | `raw_text`-Eingabe kein Max-Length-Hint, kein Backend-Check (Inventur Befund 13; #11) | Backend `/format-dialogue-with-llm` validiert nur `not raw_text or not raw_text.strip()`. Bei sehr großem Quelltext (100k+ Zeichen, Buch-Auszug) würde der LLM-Call lange dauern, Token-Limit erreichen oder fehlschlagen. Frontend hat ebenfalls keinen Max-Length-Hint. **Recovery-Anleitung fehlt**: User bekommt im Failure-Fall nur den generischen „Skript-Generierung fehlgeschlagen"-Banner, ohne Hinweis „Quelltext zu groß, kürzen". | H9 | **1** | Inventur Befund 13 | nur Finding |
| F14 | `/generate-gemini-podcast` keine Allowlist auf `language`/`tts_model` (Inventur Befund 12; #19/Backend) | Backend-Route validiert `dialogue` plus `language` plus `tts_model` nicht (im Gegensatz zu `/format-dialogue-with-llm`, das nach F-013 vollständige Allowlists hat). `tts_model` wird im Worker auf Default zurückgefallen, `language` läuft komplett ungeprüft durch. Bei kaputtem Input crasht der Worker später → `failed`-Status → generischer Banner. **Recovery-Pfad spricht nicht den eigentlichen Fehler aus**. Konsistenz-Gap mit F-013, der die Allowlist-Konvention für die andere Route etabliert hat. | H9 | **1** | Inventur Befund 12 | nur Finding |
| F15 | RQ-Statuses `deferred`/`scheduled`/`stopped`/`canceled` ungehandelt (Inventur Befund 7; #22) | Backend mappt nur `queued`/`started`/`finished`/`failed` und reicht andere RQ-Statuses raw als `{"status": <raw>}` durch. Frontend hat dafür keinen Branch, der Loop bricht ab (siehe F8-Mechanik). Praktisch tritt das nur bei `Job.cancel()`/`send_stop_job_command` (von außen) ein — was heute nirgends im Code ausgelöst wird; aber wenn F1/BT1 (Backend-Cancel) nachgereicht wird, würde dieser Pfad live werden, ohne Frontend-Handling. **Defensiv-Lücke**, heute nicht ausnutzbar. | H9 | **1** | Inventur Befund 7 | nur Finding |
| F16 | Loading-Spinner-Icon ist statisch (Inventur Befund 14; #12) | `#generate-script-btn` enthält ein SVG (Sun/Spark-Icon) in `.generate-script-btn__icon`. Während Loading wird nur `.generate-script-btn__label`-Text auf „Generiert …" gewechselt; **kein CSS-Animation auf dem SVG** (z.B. `animation: spin 1s linear infinite`). Visuell wirkt der Button weniger aktiv als ein animierter Spinner. `.mic-button__spinner` existiert bereits als Vorlage in `style.css` (für den Mic-Loading-State). **Cross-Feature-Konvention nicht ganz übernommen** — fällt aber unter „interne Stil-Verfeinerung", nicht echter Cross-Feature-H4-Bruch. | H1 | **1** ⚠️ code-only | Inventur Befund 14 | nur Finding |

---

## Reine Bug-Tickets (ohne eigenständiges Heuristik-Finding **oder** als Implementations-Anker für Findings — separates Ticket-Material)

Diese Befunde sind teils in den Findings oben **bereits aus UX-Sicht erfasst** (BT1–BT3 als finding-linked Bug-Tickets; vergleichbar zu F-3.2's BT1–BT6) und brauchen zusätzlich konkreten Code-/Backend-Implementation. BT4 ist **rein Code-Hygiene** ohne UX-H-Aspekt (vergleichbar zu F-3.2's BT7/BT8).

- **BT1: Backend-Cancel via RQ-API plus UI-Microcopy-Korrektur.** Frontend-Cancel triggert heute nur `podcastCancelRequested = true`. Bug-Fix-Pfad: `Job.cancel()` (für queued-Jobs) plus `send_stop_job_command(connection, job_id)` (für running-Jobs) im Frontend-Pfad oder via dediziertem Backend-Endpoint `POST /podcast-cancel/<job_id>` aufrufen. Cleanup: zur Stage-3-Pattern-Sprint-Diskussion gehört zusätzlich, ob `OUTPUT_DIR`-orphaned-Files (durch Browser-Reload, Network-Drop, alte Cancel-Lüge-Bestände) per TTL/Garbage-Collection-Job aufgeräumt werden. → siehe Findings F1, F2. Code-Anker: [static/js/audio_converter.js:648-652](../static/js/audio_converter.js#L648-L652) Cancel-Btn-Handler; [static/js/audio_converter.js:854-858](../static/js/audio_converter.js#L854-L858) Cancel-Banner-Pfad; [app_pkg/podcasts.py:138-164](../app_pkg/podcasts.py#L138-L164) Generate-Endpoint (kein Cancel-Endpoint vorhanden). Reproduktion: Generate-Flow starten, ~30s warten, Cancel klicken, dann `docker logs converter-worker -f` für 2 Min beobachten — Worker arbeitet weiter, File landet in `OUTPUT_DIR`. (F-4.1 Test-Anleitung 3.) Vorgeschlagener Fix-Pfad: dedizierter `POST /podcast-cancel/<job_id>`-Endpoint mit user_id-Match-Check plus `send_stop_job_command`; Frontend-Microcopy auf „Wird abgebrochen …" während des Cancel-Roundtrips, dann „Generierung abgebrochen" (ohne den missverständlichen „Backend-Job läuft im Hintergrund weiter"-Suffix).
- **BT2: Worker pflegt `job.meta` mit Stage-Progress.** `tasks.generate_podcast_task` plus `services/gemini/tts.py` Chunking-Loop machen heute nur `logger.info`. Bug-Fix-Pfad: `job.meta['stage'] = …`-Updates plus `job.save_meta()` an den existierenden Log-Sites (filter-done, chunking-decision, chunk-x-of-y-start, chunk-x-of-y-done, concat-start, concat-done). Backend-Endpoint `/podcast-status/<job_id>` reicht das Meta dann mit; Frontend rendert es in einem dedizierten Stage-Indicator (statt nur im Counter-Text). Architektur-Hebel: F-4.1 Out-of-Scope-Hinweis Befund 4 — `services/gemini/synthesis.py` und `services/gemini/tts.py` haben 50+ existierende Logger-Lines mit Per-Chunk-Progress plus Safety-Ratings plus Token-Counts plus Estimated-Remaining-Time, die als job.meta-Update-Source dienen können (keine neue Telemetrie-Investition). → siehe Finding F3. Code-Anker: [tasks.py:32-58](../tasks.py#L32-L58); [services/gemini/tts.py:154-216](../services/gemini/tts.py#L154-L216) per-chunk-loop; [app_pkg/podcasts.py:181-191](../app_pkg/podcasts.py#L181-L191) Status-Endpoint; [static/js/audio_converter.js:837-852](../static/js/audio_converter.js#L837-L852) Polling-Loop. Reproduktion: F-4.1 Test-Anleitung 2 (Multi-Chunk mit ~120-Zeilen-Skript) — Counter wächst, aber kein Hinweis auf Stage. Vorgeschlagener Fix-Pfad: in Stage-3-Pattern-Sprint entscheiden ob Stage-Texte in JSON via Backend-Endpoint oder als simple Strings durchgereicht werden; UI-Element kann ein Sub-Caption unter dem Counter sein („Chunk 3/5 wird generiert").
- **BT3: File-Cleanup-Strategie (TTL statt Sofort-Löschung) plus Re-Download-Pfad.** `podcast_download` macht `os.unlink` direkt nach `BytesIO`-Read. Bug-Fix-Pfad-Optionen: (a) File mit TTL (z.B. 1h) im OUTPUT_DIR lassen, dedizierter Cleanup-Job via cron oder beim nächsten erfolgreichen Generate; (b) Frontend-seitig den Blob im Memory cachen plus Re-Download-Btn neben dem ersten Download anzeigen (nutzt die bereits in F4 erwähnte Browser-cached-Blob-URL aus). Falls (a): Backend-Endpoint kann dann gleich Re-Download-Pfad anbieten ohne erneuten Generate-Flow. → siehe Finding F4. Code-Anker: [app_pkg/podcasts.py:230-233](../app_pkg/podcasts.py#L230-L233) `os.unlink`-Stelle; [static/js/audio_converter.js:860-866](../static/js/audio_converter.js#L860-L866) Frontend-Download-Pfad. Reproduktion: F-4.1 Test-Anleitung 7 — erfolgreichen Podcast generieren, Download-Dialog abbrechen, erneut Download-Btn klicken — File ist weg. Vorgeschlagener Fix-Pfad: in Stage-3-Pattern-Sprint Wahl zwischen TTL-File-Cleanup vs. Memory-Blob-Re-Download oder beidem gleichzeitig.
- **BT4: Audio-Blob-URL wird nicht via `URL.revokeObjectURL` freigegeben.** [static/js/audio_converter.js:866-871](../static/js/audio_converter.js#L866-L871): `const audioUrl = URL.createObjectURL(audioBlob); podcastAudioSource.src = audioUrl; downloadPodcastBtn.href = audioUrl`. Bei mehrfacher Generation in derselben Tab-Session sammeln sich Blobs im Browser-Memory. **Aus F-4.2 herausgenommen, weil keine UX-Heuristik-Komponente** (User sieht heute keinen Unterschied; reine Memory-Hygiene). Inventur-Befund 17 ist redundant mit Befund 11 — wird hier in BT4 absorbiert (selbe Wurzel, keine doppelte Disposition). Code-Anker: [static/js/audio_converter.js:866-871](../static/js/audio_converter.js#L866-L871). Reproduktion: 5–10× hintereinander Generate-Flow durchlaufen, DevTools-Memory-Profile prüfen → Blob-Akkumulation. Vorgeschlagener Fix-Pfad: vor jeder neuen `URL.createObjectURL`-Erzeugung den vorherigen `URL.revokeObjectURL(audioUrl)` aufrufen, auch beim Cancel-Pfad und beim Browser-Tab-Close (`beforeunload`-Listener).

---

## Aus F-4.2 ausgenommene Inventur-Befunde

Zwei Inventur-Befunde fielen beim Heuristik-Filter heraus, mit expliziter Begründung (analog F-3.2-Konvention für ausgenommene Items):

- **Befund 3 (Legacy `/generate-podcast` Google-TTS-Pfad — Dead-Code-Kandidat)**: aus dem Heuristik-Review explizit ausgenommen laut Sprint-Prompt — **gehört zu einer Hygiene-Welle**. Die Removal-Decision (Route plus `google_tts_service`-Singleton plus zugehörige `/api/get-google-voices`-Route plus Voice-Picker-Code) ist keine UX-H-Frage. Master entscheidet in einer Doc-/Code-Hygiene-Welle, ob der Pfad entfernt wird; Inventur-Anker [app_pkg/podcasts.py:72-136](../app_pkg/podcasts.py#L72-L136).
- **Befund 2 (F-2.1-Doc-Korrektur Service-Gate-Banner-Verhalten)**: aus dem Heuristik-Review explizit ausgenommen laut Sprint-Prompt — **gehört zu einer Doc-Hygiene-Welle**. Die F-2.1-Inventur (2026-05-03) hatte die globale-Disable-Wirkung des Service-Gate-Banners dokumentiert, der aktuelle Code lässt das Pane mit disabled-Controls sichtbar. Master pflegt das in der Doc-Hygiene-Welle nach (bereits als P3-Item im BACKLOG.md notiert nach F4-PICK-Close). Hier kein eigenes Finding, weil das Service-Gate-Banner-Verhalten **konsistent zwischen Live/File/Podcast-Tabs** ist und keine UX-H-Lücke spezifisch für podcast-flow erzeugt.

Außerdem absorbiert / als Meta:
- **Befund 17 (Audio-Blob-URL nicht revoked, redundant mit Befund 11)**: in BT4 absorbiert (selbe Wurzel als „doppelt aufgenommen" in F-4.1 markiert). Eine Disposition.
- **Befund 20 (Helper-Reuse-Beobachtung — solide)**: positive Meta-Beobachtung, kein eigenständiges Finding. In der Cross-Feature-H4-Sektion unten als Begründung der niedrigen Quote referenziert.

---

## Cross-Feature-H4-Sektion

**Cross-Feature-Konvergenz-Quote: 0% (0 von 16 Findings).** Drastisch niedriger als F-2.2 (~41%) und F-3.2 (~35%), und auch unter der Master-Erwartung von 15-25%. **Begründung:**

F-4.1 Befund 20 hat dokumentiert, dass der `audio_converter.js`-Podcast-Block die Helper-Konvergenz aus F-2 sauber durchgezogen hat:
- `safeJSON` ([static/js/audio_converter.js:750-754](../static/js/audio_converter.js#L750-L754); [:830-835](../static/js/audio_converter.js#L830-L835); [:846](../static/js/audio_converter.js#L846)) — vollständig
- `showAlert(level)` ([static/js/audio_converter.js:712-715, 758-760, 772-775, 808-811, 855-857, 877-879](../static/js/audio_converter.js#L712-L715)) — vollständig
- `showToast` ([static/js/audio_converter.js:657](../static/js/audio_converter.js#L657)) — punktuell für Download-Confirm
- `confirmIfLong` ([static/js/audio_converter.js:700-703](../static/js/audio_converter.js#L700-L703)) — punktuell für Reset-Prompt
- **kein** `alert()`-Pfad
- **keine** drei konkurrierenden Error-UI-Patterns wie in der ursprünglichen F-2.1

Die verbleibenden H4-Findings sind **interner Natur** (innerhalb des Podcast-Tabs), nicht Cross-Feature:
- **F7 (Loading-Pattern Hidden vs. Text-Swap)**: zwei Loading-Patterns innerhalb derselben Pane. Beide Patterns existieren auch sonst in der App (Hidden-Toggle bei Mic-Recording-State, Text-Swap bei Transcribe-File-Btn) — es gibt **keine** klare App-übergreifende Loading-Convention, die hier verletzt würde. Interner H4-Bruch.
- **F9 (Tab-aria-disabled vs. JS-Click-Block)**: Konsistenz-Bruch zwischen Markup-Versprechen (Template) und JS-Verhalten — also intra-feature, nicht inter-feature.
- **F16 (Spinner-Icon statisch)**: leichte Cross-Feature-Tendenz (`.mic-button__spinner` existiert bereits als animierte Vorlage in `style.css`), aber das `#generate-script-btn`-Icon hat eine andere Funktion (decorative SVG, kein Spinner) — kein direkter Konvention-Bruch, eher eine fehlende Polish-Anwendung.

**Bestätigung der Master-Annotation**: Sub-Thread sollte „keine künstlichen H4-Findings konstruieren". Die 0%-Quote ist erwartet und dokumentiert die positive F-2-Konvergenz-Wirkung; gleichzeitig ist die **Verteilungs-Verschiebung zu H1 und H9** (Async-Heuristik-Sub-Sektion unten) der substantielle Kern dieses Reviews.

---

## Async-Heuristik-Sub-Sektion (NEU für podcast-flow)

Verteilung der Findings über die 5/6 unzureichenden Async-State-Klassen aus F-4.1. **10 von 16 Findings (62.5%) sind Async-spezifisch** — entweder direkte State-Klassen-Lücken oder Polling-Edge-Cases. Diese Sub-Sektion hilft F4-PATTERNS, die Patterns nach Async-State-Mapping zu strukturieren.

| Async-State-Klasse | Findings | Bug-Tickets | Heuristik dominiert | Daily-Usage-Schmerz |
|--------------------|----------|-------------|---------------------|---------------------|
| **`cancelled` (Frontend-Lüge — Worker läuft weiter)** | F1 (Sev 4 H1), F2 (Sev 4 H9) | BT1 | H1 + H9 (geteilt) | Hoch (Cancel ist täglich erreichbarer Pfad bei Long-Running) |
| **`stage-progress` fehlt komplett (`job.meta` ungenutzt)** | F3 (Sev 3 H1) | BT2 | H1 | Hoch (jeder Multi-Chunk-Job spürt die Counter-vs-Stage-Lücke) |
| **`queued` und `started` zu `processing` konflatiert** | F6 (Sev 2 H1) | — | H1 | Niedrig (Single-User-Setup, ein Worker, queued-vs-started selten relevant) |
| **Polling-Edge-Cases (kein Frontend-Timeout, kein r.ok-Check, RQ-Statuses ungehandelt)** | F8 (Sev 2 H9), F15 (Sev 1 H9) | — | H9 | Niedrig–Mittel (passiert nur bei kaputten Backend-Responses oder Network-Drop) |
| **Browser-Reload während Polling (`job_id` flüchtig)** | F5 (Sev 3 H1) | — | H1 | Mittel (versehentliche Reloads passieren; Worker arbeitet trotzdem weiter — selbe Cost-Familie wie F1) |
| **File-Cleanup post-finished (kein Re-Download)** | F4 (Sev 3 H9) | BT3 | H9 | Mittel (selten Browser-Block, aber wenn passiert kostet TTS-Re-Generation Credit) |
| **Download-Toast immer grün post-finished** | F10 (Sev 2 H1) | — | H1 | Niedrig (visuell-Lüge, aber selten konsequenz-relevant) |
| **Loading-Pattern-Inkonsistenz (während async)** | F7 (Sev 2 H4) | — | H4 (interner Bruch) | Niedrig |

**Nicht-Async-Findings (6 von 16):** F9 (Tab-aria-disabled, intra-feature), F11 (Skript-Parsing, H6), F12 (Skript-Textarea-editable-mid-pipeline, H6 — adjacent zu Async aber nicht direkt async-state), F13 (raw_text Max-Length), F14 (Allowlist-Gap), F16 (Spinner-Icon).

**Pattern-Implikation für F4-PATTERNS:** Cluster-Strategie sollte die Async-State-Klassen als Leitachse nehmen. Cancel-Recovery (F1+F2+BT1) und Stage-Progress (F3+BT2) sind die zwei größten Pattern-Investitionen mit Daily-Usage-Schmerz; File-Cleanup (F4+BT3) ist die dritte. Die Polling-Edge-Cases (F5/F6/F8/F10/F15) lassen sich als kleinerer „Polling-Robustheit"-Pattern bündeln.

---

## Schwerpunkt-Cluster

Vier thematische Cluster, in denen sich die schweren Findings konzentrieren — analog F-2.2's „Drag-Drop-Lüge / 11+ alert() / Config-Error-Global / Englische Strings" und F-3.2's „Silent-Failure-Familie / Notion-Side State-Wipe / Cross-Feature-Helper-Drift":

### Cluster 1 — Cancel-und-Cleanup-Recovery (F1, F2, F4; Sev 3–4 ⚠️ code-only)

**Daily-Usage-Schmerz hoch.** Drei Findings, die auf demselben Async-Pipeline-Anti-Pattern beruhen: Frontend-Visual entkoppelt sich von Backend-Realität.
- **F1 + F2 (Cancel-Lüge)**: Cancel-Btn stoppt Worker nicht. Strukturanalogon zur F-2.1-Drag-Drop-Lüge, identische Schmerz-Klasse (Label/Knopf verspricht Funktion, die nicht erfüllt wird). Worker brennt CPU plus TTS-Token plus Disk weiter.
- **F4 (File-Cleanup-vs-Re-Download)**: `os.unlink` direkt nach erstem Download macht Re-Download unmöglich, falls Browser den Download blockt. Recovery-Pfad fehlt komplett — User muss neu generieren plus Credit erneut verbrennen.

**Daily-Usage**: Cancel ist täglich erreichbarer Pfad bei Long-Running; File-Cleanup-Schmerz seltener aber teurer pro Vorfall.
**Fix-Pfad in Stage 3**: BT1 (Backend-Cancel via RQ-API plus UI-Microcopy-Korrektur) und BT3 (File-TTL plus Re-Download) als unabhängige Bug-Tickets vorab fixbar; Pattern-Cluster „Cancel-und-Cleanup-Recovery" konsolidiert UX-Aspekt mit Microcopy-Update („Wird abgebrochen …" während Cancel-Roundtrip; Re-Download-Btn neben Erst-Download). Plus optional: orphaned-OUTPUT_DIR-Files via Garbage-Collection-Job aufräumen.

### Cluster 2 — Async-State-Visibility (F3, F5, F6, F10, F16; Sev 1–3)

**Daily-Usage-Schmerz mittel-hoch wegen F3.** Fünf Findings, die System-Status während/nach der Async-Pipeline unterkommunizieren.
- **F3 (Stage-Progress fehlt)**: zentralster Daily-Usage-Schmerz dieses Clusters. Worker pflegt kein `job.meta`, User sieht nur Wand-Sekunden-Counter ohne Kontext.
- **F5 (Browser-Reload = job_id verloren)**: System-Status verschwindet komplett, obwohl Job weiterläuft. Selbe Disk-/Credit-Familie wie F1.
- **F6 (queued+started konflatiert)**: User-relevante Worker-Status-Information geht im Backend verloren.
- **F10 (Download-Toast immer grün)**: System-Status-Lüge bei Browser-Block.
- **F16 (Spinner-Icon statisch)**: Polish-Item, fällt visuell weniger aktiv auf.

**Fix-Pfad in Stage 3**: BT2 (`job.meta`-Stage-Progress aus den existierenden Worker-Logger-Lines, F-4.1-Architektur-Hebel) ist die größte Pattern-Investition; F5/F6 lassen sich als kleinere Pattern-Items integrieren (LocalStorage für `job_id`, Backend-Endpoint differenziert queued/started); F10/F16 als Polish-Microcopy/CSS-Animation in einem dedizierten Polish-Pattern.

### Cluster 3 — Polling- und Defensiv-Robustheit (F8, F13, F14, F15; Sev 1–2)

**Daily-Usage-Schmerz niedrig**, aber wichtige Defensiv-Lücken die bei Edge-Cases zu generischen Failure-Bannern statt spezifischen Recovery-Hinweisen führen.
- **F8 (Polling-Loop kein Timeout, kein r.ok-Check)**: bei kaputten Backend-Responses fällt der Loop in den Download-Pfad mit ungültigem `job_id`. Generischer „Generierung fehlgeschlagen"-Banner statt „Polling-Drop, Verbindung prüfen".
- **F13 (raw_text kein Max-Length)**: bei sehr großem Quelltext Token-Limit-Failure ohne Vorab-Hinweis.
- **F14 (Allowlist-Gap auf `/generate-gemini-podcast`)**: Konsistenz-Gap mit F-013-Allowlist-Konvention; bei kaputtem Input crasht Worker später.
- **F15 (RQ-Statuses ungehandelt)**: heute nicht ausnutzbar, aber wenn BT1 (Backend-Cancel) implementiert wird, würde `canceled`-Status live werden ohne Frontend-Branch.

**Fix-Pfad in Stage 3**: kleines „Polling-Robustheit"-Pattern für F8+F15 (status-Validation, deferred/canceled-Branches); F13+F14 als kleines „Backend-Validation-Konvergenz"-Item auf F-013-Niveau bringen.

### Cluster 4 — Speaker-Format-Hilfe und Edit-Verhalten (F11, F12; Sev 1–2 H6)

**Daily-Usage-Schmerz mittel** wenn Skript manuell editiert/geschrieben wird.
- **F11 (Skript-Parsing fragil)**: Recognition-over-Recall-Bruch — User muss `Sprecher [stil]: Text`-Format aktiv aus Kopf reproduzieren, kein Schema-Editor.
- **F12 (Skript-Textarea editable mid-pipeline)**: User-Erwartung „Eingabe-State eingefroren während Maschine arbeitet" wird gebrochen (analog zu F-2's `setLiveTextareaReadonly`-Pattern).

**Fix-Pfad in Stage 3**: Pattern „Skript-Editor-Hilfe" mit Format-Hint-Verbesserung (z.B. inline-Validation bei Eingabe, Voice-Map-Picker statt Format-Erinnerung); plus `setPodcastScriptReadonly(true)`-Helper analog zu Live-Tab.

---

## Disposition-Verteilung

- **Nur Findings (kommen in F4-PATTERNS):** 12 — F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16
- **Findings + Bug-Tickets (kommen in F4-PATTERNS **plus** separates Bug-Ticket):** 4 Findings → **3 unique Bug-Tickets** — F1+F2 (BT1), F3 (BT2), F4 (BT3)
- **Nur Bug-Tickets (kommen **nicht** in F4-PATTERNS):** 1 — BT4 (Audio-Blob-URL-Revoke, kein UX-H-Aspekt; Inventur-Befund 11 plus 17 absorbiert)
- **Aus F-4.2 ausgenommen (Hygiene-/Doc-Welle):** 2 Inventur-Befunde — Befund 3 (Legacy `/generate-podcast` Dead-Code), Befund 2 (F-2.1-Doc-Korrektur Service-Gate)

**Inventur-Befund-Coverage (alle 19 plus 1 Meta disponiert):**
- Befund 1 → F9
- Befund 2 → ausgenommen (F-2.1-Doc-Korrektur, P3-BACKLOG-Item)
- Befund 3 → ausgenommen (Hygiene-Welle Dead-Code)
- Befund 4 → F3 + BT2
- Befund 5 → F6
- Befund 6 → F8
- Befund 7 → F15
- Befund 8 → F7
- Befund 9 → F1 (H1) + F2 (H9) + BT1
- Befund 10 → F5
- Befund 11 → BT4 (kein UX-H-Aspekt — pure Memory-Hygiene)
- Befund 12 → F14
- Befund 13 → F13
- Befund 14 → F16
- Befund 15 → F12
- Befund 16 → F11
- Befund 17 → in BT4 absorbiert (redundant mit Befund 11; F-4.1 hat das transparent als „doppelt aufgenommen" markiert)
- Befund 18 → F4 + BT3
- Befund 19 → F10
- Befund 20 (Meta-Beobachtung Helper-Reuse) → in Cross-Feature-H4-Sektion oben absorbiert (Begründung der 0%-Quote)

**Abweichungen von F-4.1-Disposition (begründet):**
- **Befund 11 (Audio-Blob-URL-Revoke):** F-4.1 schlug „nur Finding" vor. F-4.2 ordnet als „nur Bug-Ticket" ein, weil **keine UX-Heuristik-Komponente** vorhanden ist (User sieht heute keinen Unterschied; reine Memory-Hygiene; H1/H4/H6/H9 treffen alle nicht). Filtert sich beim Heuristik-Filter heraus, gehört zur Code-Hygiene-Welle. Analog zu F-3.2's Behandlung von Befund 9 (textarea-escape) und Befund 14 (window.open-noopener).
- **Befund 17 (Audio-Blob-URL nicht revoked, redundant):** F-4.1 hatte das transparent als „doppelt aufgenommen aus Code-Review-Sicht" markiert. F-4.2 absorbiert in BT4 — eine Disposition. Konsistent mit F-4.1's eigener „eine Disposition"-Hinweis.
- **Inventur-Befund-Aufteilung:** F-4.1 hatte 19 nummerierte Befunde + 1 Meta. F-4.2 produziert 16 Findings + 4 Bug-Tickets, weil zwei Befunde als „aus F-4.2 ausgenommen" katalogisiert sind (Befund 2, Befund 3), ein Befund in 2 Heuristik-Reihen aufgespalten wurde (Befund 9 → F1+F2), zwei Befunde absorbiert wurden (Befund 17 in BT4, Befund 20 in Cross-Feature-Sektion), und ein Befund zu Bug-only fiel (Befund 11).

---

## Zusammenfassung

- **Heuristik-Findings gesamt:** 16
- **Davon Schweregrad 4 (kritisch):** 2 (F1, F2 — beide zur Cancel-Lüge; Datenverlust-/Cost-Pfad)
- **Davon Schweregrad 3:** 3 (F3 Stage-Progress fehlt, F4 File-Cleanup-vs-Re-Download, F5 Browser-Reload-job_id-Verlust)
- **Davon Schweregrad 2:** 6 (F6 queued+started-Konflation, F7 Loading-Pattern-Inkonsistenz, F8 Polling-Robustheit, F9 Tab-aria-disabled, F10 Download-Toast-immer-grün, F11 Skript-Parsing-Fragilität)
- **Davon Schweregrad 1:** 5 (F12 Skript-Textarea-editable-mid-pipeline, F13 raw_text-Max-Length, F14 Allowlist-Gap, F15 RQ-Statuses-ungehandelt, F16 Spinner-Icon-statisch)
- **Reine Bug-Tickets (mit Ticket-Material):** 4 (BT1 Backend-Cancel-via-RQ-API, BT2 Stage-Progress-job.meta, BT3 File-Cleanup-Strategie-plus-Re-Download, BT4 Audio-Blob-URL-Revoke) — davon 3 mit Finding-Verknüpfung (BT1↔F1+F2, BT2↔F3, BT3↔F4), 1 pure Bug ohne H-Aspekt (BT4)
- **Cross-Feature-H4-Findings:** **0 von 16 (0%)** — drastisch niedriger als F-2.2 (~41%) und F-3.2 (~35%) und auch unter Master-Erwartung 15-25%. Begründung: F-4.1 Befund 20 hat die saubere Helper-Konvergenz aus F-2 dokumentiert; verbleibende H4-Findings (F7, F9, teilweise F16) sind interner Natur, nicht Cross-Feature. Verteilungs-Verschiebung zu H1 (5 Findings) und H9 (5 Findings) ist die substantielle Kern-Beobachtung.
- **Async-spezifische Findings:** 10 von 16 (62.5%) — entweder direkte State-Klassen-Lücken (cancelled, stage-progress, queued+started) oder Polling-Edge-Cases (Reload, Network-Drop, RQ-Statuses, Frontend-Timeout) oder post-finished State (File-Cleanup, Download-Toast).
- **`⚠️ code-only`-markierte Findings:** 6 (F1, F2, F4, F5, F8, F16) — Findings, deren visueller Effekt aus reinem Code-Reading nicht endgültig beurteilbar ist.

**Heuristik-Verteilung:** H1: 6 Findings (F1, F3, F5, F6, F10, F16). H4: 2 Findings (F7, F9). H6: 2 Findings (F11, F12). H9: 6 Findings (F2, F4, F8, F13, F14, F15). **H1 und H9 dominieren wie Master vorhergesagt** wegen der 5/6 unzureichenden Async-State-Klassen — gleichgewichtige 6-6-Verteilung zwischen System-Status (H1) und Error-Recovery (H9), wobei das Cancel-Lüge-Finding-Paar (F1+F2) genau diese Doppel-Achse spannt: H1 für die Lüge ggü. der Worker-Realität, H9 für den fehlenden Recovery-Pfad. 16 Findings total (6+2+2+6).

**Schweregrad-Skala:**
1. kosmetisch (kaum spürbar)
2. gering (nur in Edge-Cases störend)
3. mittel (regelmäßig spürbar, frustrierend)
4. kritisch (verhindert/verfälscht die primäre Aufgabe oder produziert falsche Ergebnisse / Datenverlust- oder Cost-Pfad)

**Master-Walkthrough-Empfehlung vor F4-PATTERNS:** Ja, gerade weil Async-Übergänge in F4-IMPL schwer nachzu-Smoke-en sind. Priorisierte Walkthrough-Sequenz aus den 7 F-4.1-Test-Anleitungen:
- **Test-Anleitung 3 (Cancel-mid-Generation plus `docker logs converter-worker -f` Disk-Forensik)** für F1+F2 — höchste Priorität, weil Sev-4-Schmerz und BT1-Implementation ohne Verifikation der konkreten Worker-Verhalten-Charakteristik schwierig zu kalibrieren ist.
- **Test-Anleitung 7 (Re-Download-Pfad mit Pop-Up-Blocker)** für F4 — verifiziert ob Browser-cached-Blob-URL den Re-Download trotz Server-Side-Cleanup ermöglicht oder nicht (BT3-Fix-Pfad-Wahl hängt davon ab).
- **Test-Anleitung 4 (Browser-Reload mid-Polling)** für F5 — verifiziert Worker-Continue-Verhalten plus Output-File-Verbleib.
- **Test-Anleitung 2 (Multi-Chunk-Podcast)** für F3 — visualisiert die Counter-vs-Stage-Lücke konkret, hilft F4-PATTERNS bei der Stage-Indicator-UX-Designentscheidung.
- **Test-Anleitung 5 (Network-Drop in DevTools)** für F8 — verifiziert Polling-Robustheit-Verhalten konkret.

Wenn Master-Bandwidth knapp ist: F4-PATTERNS kann auch ohne Walkthrough beginnen, aber dann sollten die Pattern-Vorschläge für Cluster 1 und Cluster 2 das `⚠️ code-only`-Risiko explizit in den Phasen-Stops markieren, damit ein Implementierungs-Smoke vor Merge zwingend ist (analog F3-IMPL Smoke-Pflicht-Patterns P1/P3/P4/P5).
