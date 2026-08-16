# SPRINT DOC-LOCAL — der lokale Pfad bekommt eine echte Engine

**Größe**: L (3 Phasen) · **Datum**: 2026-08-16 · **Vorhaben**: DOC-SVC

## Warum

Der Dienst hat seit DOC-ENGINE die gemessenen Sieger hinter sich — für Office, Web und Cloud-PDF. Der **lokale** PDF-Pfad ist dagegen bis heute die rohe PyMuPDF-Textebene: beweisbar deterministisch je Seite, und auf Scans **leer**. Das war bewusst so gebaut (die Degradation braucht ein Ziel, und ein ehrlich schwaches ist besser als ein fehlendes), aber es ist der letzte Punkt, an dem der Dienst unter dem liegt, was gemessen wurde.

Dieser Sprint tauscht ihn gegen **mineru** aus. Damit steht auf beiden Seiten des Kostendeckels eine echte Engine, und ein Auftrag mit `mode=lokal` liefert erstmals brauchbare Ergebnisse auf gescannten Seiten.

**Er tauscht nur die Engine.** Das Retten von Seiten-Routing und Multi-Page-Merge aus dem Eigenbau und das Abschalten der fünf Detektoren sind ein **eigener Sprint** (DOC-ROUTE) — zusammen wäre das XL, und die Splitting-Regel der Working Practice greift.

## Gegroundeter Ist-Zustand (Master — nicht neu herleiten, alles am 2026-08-16 gemessen)

**Die GPU ist bereit.** Auf der Mintbox: RTX A2000 12 GB, Treiber 580.126.18, `nvidia-container-toolkit` 1.20.0-1 installiert, `nvidia` als Runtime in Docker registriert (Default bleibt `runc`), GPU-Testcontainer läuft. ⚠️ Ältere Notizen im Repo sagen „kein Passthrough" — **das ist überholt**, vermutlich seit dem Bake-off eingerichtet.

**`mineru:latest` liegt schon auf der Mintbox**, **29,7 GB**. Das ist der Grund für einen eigenen Container statt eines Merges ins App-Image, das IMG-SLIM gerade mühsam auf 9 GB gebracht hat.

**Die gemessene Invokation** (aus `corpus/bakeoff/harness/adapters.py::run_mineru_vlm`, mineru 3.4.4):

```
mineru -p /in/<datei>.pdf -o /out -b vlm-engine
```

⚠️ Das Backend heißt `vlm-engine`. Die 2.x-Doku sagt noch `vlm-vllm-engine` — der Bake-off hat das live gegen `--help` verifiziert.

**Die `docker run`-Mechanik ist im Harness schon teuer gelernt** (`_docker_convert`) und **wird repliziert, nicht neu erfunden**:

- `--gpus all --shm-size 16g`
- `-v <quellverzeichnis>:/in:ro -v <ausgabe>:/out`
- `-v <hf-cache>:/models -e HF_HOME=/models -e MINERU_MODEL_SOURCE=huggingface` — auf der Mintbox liegt der Cache unter `/home/oliver/.cache/huggingface` (heute so benutzt, funktioniert). **Ohne persistenten Cache lädt jeder Lauf die Modelle neu.**
- Der Container läuft **als root**. `--user` scheitert an fehlenden passwd-Einträgen (`getpwuid(): uid not found`, live getroffen). Der Gegenpart — root-eigene Dateien in `/out`, die der Aufrufer nicht lesen darf — wird danach mit einem `busybox chown -R <uid>:<gid> /out` aufgelöst. ⚠️ **chown, nicht chmod**: mit `chmod a+rX` starb das Aufräumen des Temp-Verzeichnisses an EPERM.

**Die Kostenkurve**, linear gefittet über die Bake-off-Läufe von 2 bis 280 Seiten: **≈61 s fixer Modell-Start + ≈2,5 s je Seite.** Belege: 2 Seiten 66 s · 12 Seiten 99 s · 20 Seiten 102 s · 54 Seiten 212 s · 280 Seiten 766 s. VRAM-Spitze 6,5 GB von 12.

**Die Qualität, ehrlich**: mineru liegt auf der Paper-Goldseite bei **0,9551** gegen 0,9809 bei gemini, hat aber die **besten Tabellenzellen des ganzen Feldes (0,916)** und verliert systematisch **etwa fünf Fußnoten pro Dokument**. Es ist der einzige lokale Allrounder mit Tempo. Es ist **nicht** der Cloud-Pfad in grün — es ist ein guter zweiter.

**mineru liefert Seiten-Attribution selbst.** Heute verifiziert am 2-Seiten-Schnitt `corpus/bakeoff/derived/03_gold-seiten.pdf`: die Ausgabe enthält `<name>_content_list.json` mit **`page_idx` je Element** (55 Einträge Seite 0, 55 Seite 1), dazu `bbox`, `type`, `text`, `text_level`. Damit ist der `page_fn`-Vertrag ohne Textzerschneiden erfüllbar.

⚠️ **Und mineru liefert die Mittel zum stillen Aufräumen gleich mit.** Die `type`-Verteilung desselben Laufs: `text` 102, **`header` 4, `footer` 2, `page_number` 2**. Wer diese drei Typen wegfiltert — was viele Pipelines tun —, entfernt genau die wiederholten Kopf- und Fußzeilen, die Oli am 2026-08-16 ausdrücklich behalten wollte („alles rein, 1:1 Kopie") und die **Bewertungsregel 4** als Messgegenstand festschreibt. **Nicht filtern.**

**In `03_gold-seiten` kommt kein `table`-Typ vor** (das Dokument ist eine einspaltige Liste). Wie mineru Tabellen in die `content_list` schreibt, ist damit **ungemessen** — auf einem tabellentragenden Belegexemplar prüfen (`corpus/04_verbundene-zellen/` oder `01`), bevor die Zusammensetzung darauf baut.

**Der Backend-Vertrag** steht unverändert in [services/document_pipeline.py](../../../services/document_pipeline.py): `page_fn(page_index_0based) → {markdown, origin, cost_eur}`, zwei Callables je Lauf. Der heutige `local_page` ist die PyMuPDF-Textebene in [services/pdf_cloud.py](../../../services/pdf_cloud.py) — **den ersetzt dieser Sprint**.

## Gesperrte Entscheidungen

1. **Docker-Socket im Worker** (Oli, 2026-08-16). Der Worker startet ein **Geschwister**-Container, nicht ein verschachteltes. Gewählt gegen einen `mineru-api`-Sidecar, weil: es ist die gemessene Invokation, und die GPU ist **nur während des Auftrags** belegt — ein Dauer-Sidecar hielte 6,5 GB von 12 GB dauerhaft und kollidierte mit Olis ComfyUI/LoRA-Nutzung. Preis, bewusst akzeptiert: 61 s Modell-Start pro Auftrag (bei einem Hintergrund-Job unkritisch) und ⚠️ **der Docker-Socket ist root-äquivalent auf dem Host** — das gehört **benannt** ins Kontrakt-Doc und in CLAUDE.md, nicht versteckt.
2. **Die Invokation wird wörtlich repliziert.** Abweichende Aufrufe entwerten die Messung (Lehre `reference_measured_winner_version_gap`). Wenn dir ein anderer Aufruf besser scheint: **messen und berichten**, nicht setzen.
3. **Ein mineru-Lauf je gebrauchtem Seitenbereich, nicht je Seite.** Bei 61 s Startzeit wäre ein Aufruf pro Seite absurd. Der `local_page`-Callable bleibt nach außen seitenweise, dahinter liegt **ein memoisierter Lauf über einen Teil-PDF genau der Seiten, die gebraucht werden** (beim Mid-flight-Umsprung also Seite N bis Ende, nicht das ganze Dokument).
4. **Die Herkunft ist `modell`, die Kosten sind 0,00 €.** mineru ist ein VLM. ⚠️ **Damit ändert sich die Bedeutung von `mode=lokal`**: bis heute heißt lokal *beweisbar deterministisch*, danach heißt es *lokales Modell, kein Geld*. Das ist ehrlich (die Herkunft je Seite sagt es), aber es ist eine **Vertragsänderung in der Bedeutung** und muss ins Kontrakt-Doc. Ein künftiger `mode=deterministisch` für Aufrufer, die Determinismus brauchen, ist **nicht** Teil dieses Sprints — aber als Möglichkeit zu benennen.
5. **`header`, `footer`, `page_number` werden nicht gefiltert** (Punkt oben, Regel 4).

---

# Phase 1 — Das Backend

## 1.1 Das Modul

Nach dem Muster von [services/pdf_cloud.py](../../../services/pdf_cloud.py): pures Modul, schwere Imports in den Funktionen, Docstring trägt die Begründung. Es liefert den `local_page`-Callable des Vertrags.

Aufgaben: Teil-PDF der gebrauchten Seiten schneiden · Container starten (Invokation und Mechanik oben) · `content_list.json` lesen · nach `page_idx` gruppieren · je Seite Markdown zusammensetzen · memoisieren.

## 1.2 Was zu messen ist, statt es anzunehmen

- **Tabellen**: wie stehen sie in der `content_list`? Auf einem tabellentragenden Exemplar prüfen. Falls die `content_list` Tabellen nicht in verwertbarer Form trägt, ist die Alternative, mineru's fertiges `.md` an den Seitengrenzen zu schneiden — dann **belege**, dass die Grenzen zuverlässig sind. Nicht raten.
- **Zusammensetzung gegen Gold**: fahr das Ergebnis gegen `corpus/gold/03.md` mit `score_gold.py` (Klasse 03 ist seit 2026-08-16 registriert) und gegen `01.gold`. Die Bake-off-Zahl für mineru auf `01.gold` ist **0,9551** — kommst du deutlich darunter raus, liegt es an deiner Zusammensetzung, nicht an mineru.

## 1.3 Der Fehlerpfad

Der Deckel degradiert cloud→lokal. Fällt **lokal** aus (GPU belegt, Container-Fehler, Timeout), gibt es kein weiteres Ziel. Fall zurück auf die PyMuPDF-Textebene mit einer benannten **`backend_fallback`**-Degradation — den Code gibt es seit DOC-ENGINE P1, und das Muster ist dort schon begründet: ein harter Fail wäre eine Fähigkeits-Regression.

## 1.4 Belege

⚠️ **Dieser Sprint kann nicht auf dem Mac verifiziert werden** — dort ist keine NVIDIA-GPU. Das heißt **nicht**, dass auf der Mintbox gearbeitet wird. Die Regel bleibt: **editiert wird ausschließlich auf dem Mac**, jede Änderung geht über `origin`, auf der Mintbox nur `git pull` + `docker compose up -d --build` + Smoke. Kein Editieren dort, **keine unversionierten Dateien zurücklassen** (der Bake-off hat dort 72 liegengelassen und Olis nächsten Deploy blockiert).

Beleg für diese Phase: der Zwei-Seiten-Schnitt und ein tabellentragendes Exemplar durch das Modul, Scores gegen Gold.

## Stop
`pytest tests/` grün (Baseline **923**). **Commit + Push** `feat(DOC-LOCAL): mineru-Backend fuer den lokalen PDF-Pfad (P1)`. Dann warten.

---

# Phase 2 — Verdrahtung und Betrieb

## 2.1 Der Geschwister-Container

⚠️ **Die Falle, die dich sonst Stunden kostet**: Der Worker spricht über den Socket mit dem **Host**-Docker. Alle Pfade in `-v` sind **Host**-Pfade, nicht Pfade im Worker-Container. `/app/output_podcasts/doc_conversions/source_134.pdf` existiert auf dem Host **nicht** unter diesem Namen — es liegt im Volume `converter_podcast_data`. Löse das bewusst und benenne die Lösung im Bericht (Volume-Mountpoint auflösen, oder ein Host-Pfad-Bind, oder den Teil-PDF an einen beiden bekannten Ort schreiben).

## 2.2 Compose

Socket in den Worker, HF-Cache als persistentes Volume oder Host-Bind. Das `mineru:latest`-Image muss auf dem Host liegen — es wird **nicht** gebaut. Prüfe, ob es auf der Mintbox versionsfest ist (`latest` ist ein bewegliches Ziel; wenn ein Digest-Pin billig ist, nimm ihn — die Messung gilt für 3.4.4).

## 2.3 Der Zeit-Umschlag

`doc_convert_job_timeout_for(pages)` ist für die Cloud-Deadline kalibriert. Die neue Kurve ist **61 s + 2,5 s je Seite**; bei `mode=lokal` muss der Umschlag das tragen — 280 Seiten sind ~13 Minuten. Rechne es aus, setz es, benenne es.

## 2.4 Belege

Durch den **echten** Dienst auf der Mintbox: ein Scan-Exemplar (`05_scan-sauber` — das ist der Fall, den der alte lokale Pfad **leer** zurückgab) und `12_grosses-pdf` mit `mode=lokal`. Beim zweiten ist die Antwort mit ihrer Herkunft je Seite und der Laufzeit zu zeigen.

## Stop
Beide Läufe belegt. **Commit + Push** `feat(DOC-LOCAL): Verdrahtung, Compose und Zeit-Umschlag (P2)`. Dann warten.

---

# Phase 3 — Wrap

- **Kontrakt-Doc** [docs/document_api_contract.md](../../document_api_contract.md): §10 um den lokalen Pfad ergänzen, **und die Bedeutungsänderung von `mode=lokal` deutlich benennen** (Herkunft `modell` statt `deterministisch`, Kosten weiterhin 0,00 €; ein künftiger `mode=deterministisch` als Möglichkeit, nicht als Zusage).
- **CLAUDE.md** (Architecture Notes, eigener DOC-LOCAL-Bullet; der DOC-ENGINE-Bullet sagt „bis DOC-LOCAL" — nachziehen), **STATUS.md**, **BACKLOG.md** (Bullet-Guard).
- **Der Docker-Socket gehört sichtbar dokumentiert**, mit der Begründung aus den gesperrten Entscheidungen — nicht als Fußnote.
- **Memory**: die veraltete Notiz zum fehlenden GPU-Passthrough korrigieren (`reference_mintbox_gpu_unstructured_cpu_path`); neue Lehren, falls übertragbar. Nach dem Schreiben mit `ls` prüfen, dass Datei und Index-Zeile zusammenpassen.
- **Im Bericht benennen**: wie Tabellen aus der `content_list` kommen · wie das Host-Pfad-Problem gelöst wurde · welchen Zeit-Umschlag du gesetzt hast · welchen Score deine Zusammensetzung gegen `01.gold` und `03.gold` erreicht · ob `mineru:latest` gepinnt werden konnte.

## Nicht-Ziele

- **Kein Seiten-Routing, kein Multi-Page-Merge, kein Abschalten der fünf Detektoren.** Das ist DOC-ROUTE. `services/pdf_extraction/` bleibt in diesem Sprint **unberührt** — es trägt weiterhin `/transform-document` und den Textebenen-Rückfall.
- **Kein** Umbau von `document_pipeline.py`, der Antwortform oder des Kontrakt-Vertrags (die Bedeutungs-Fußnote zu `mode=lokal` ist eine Doku-Ergänzung, keine Formänderung).
- **Kein** Anfassen des Cloud-Pfads.
- **Kein** `mineru-api`-Sidecar (Entscheidung 1).
- **Kein** `mode=deterministisch`.
- ⚠️ **Editiert wird nur auf dem Mac.** Die Mintbox ist Runtime — Deploy und Smoke ja, Arbeitsplatz nein, keine unversionierten Dateien zurücklassen.
