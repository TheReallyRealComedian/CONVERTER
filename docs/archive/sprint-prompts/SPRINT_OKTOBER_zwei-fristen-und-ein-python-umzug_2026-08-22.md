# SPRINT OKTOBER — zwei Fristen prüfen, dann der Python-Umzug

**Größe**: L (3 Phasen) · **Datum**: 2026-08-22 · **Vorhaben**: Betrieb

## Warum

Zwei datierte Abhängigkeiten laufen im Oktober 2026 ab. Beide sind die Fehlerklasse, die CONVERTER schon einmal zwei Monate stillen Ausfall gekostet hat: `gemini-2.0-flash` wurde abgeschaltet, der 404 wurde gefangen, der Pfad degradierte auf eine leere Textebene, und die einzige Spur war eine WARNING-Zeile.

⚠️ **Die Backlog-Notiz zu Frist 1 hat eine Lücke, die dieser Sprint zuerst schließt.** Sie sagt „CONVERTER nutzt `gemini-2.5-flash` nicht" — das stimmt für den PDF-Pfad (`gemini-3.6-flash` seit DOC-FIX). Sie übersieht aber, dass die **Narration** auf **`gemini-2.5-flash-tts`** läuft ([services/narration_render.py:46](../../../services/narration_render.py)). Ob die TTS-Variante dasselbe Abschaltdatum teilt, ist **unbekannt und muss geprüft werden**.

## Gegroundeter Ist-Zustand (Master — nicht neu herleiten)

**Frist 1 — 16.10.2026, `gemini-2.5-flash`.** Live in Benutzung sind heute genau zwei Modellnamen:
`gemini-3.6-flash` ([services/pdf_cloud.py:60](../../../services/pdf_cloud.py), env `PDF_VISION_MODEL`) und `gemini-2.5-flash-tts` ([services/narration_render.py:46](../../../services/narration_render.py), env `NARRATION_TTS_MODEL`, **Cloud-TTS-Name, nicht genai**). ⚠️ Der zweite ist der offene Punkt. Beide sind env-overridable — ein Wechsel ist eine `.env`-Zeile, **wenn** man rechtzeitig weiß, worauf.

**Frist 2 — 04.10.2026, Python 3.10.** Der Container läuft **Python 3.10.12** auf `mcr.microsoft.com/playwright/python:v1.44.0-jammy` ([Dockerfile:2](../../../Dockerfile)). Die Google-Client-Bibliotheken warnen bei jedem Aufruf, dass sie nach diesem Datum keine neuen Versionen mehr für 3.10 liefern. ⚠️ **Das ist keine Klippe**: nichts hört auf zu funktionieren, die Pins laufen weiter. Was endet, sind **Updates** — und die betroffenen Bibliotheken (`google-cloud-texttospeech` für die Narration, `google-genai` für den PDF-Pfad) sind netzwerkzugewandt und tragen Olis Zugangsdaten. Der Preis fällt an, wenn eine Sicherheitskorrektur nur noch für neuere Python-Versionen kommt.

**Die Python-Version hat schon einmal eine Wahl eingeschränkt**: der CVE-PDF-Eintrag im BACKLOG hält fest, dass `unstructured` auf 0.18.32 „gecappt auf Python-3.10-Container" wurde.

**Tragende Pins mit eigenen Verträgen** — bei einem Python-Wechsel müssen sie alle neu aufgelöst werden, und bei diesen fällt die Prüfung nicht optional aus:
- `nh3==0.2.18` — die CARD-SVG-Doktrin steht darauf, dass ammonia `viewBox` **nicht** kleinschreibt; Sentinel-Test in [tests/test_svg_sanitize.py](../../../tests/test_svg_sanitize.py).
- `fsrs==6.3.1` — `learning_steps`/`relearning_steps` sind explizit gesetzt, Sentinel in [tests/test_scheduler.py](../../../tests/test_scheduler.py); ein geänderter Library-Default würde sonst still das Lernverhalten verschieben.
- `asgiref==3.8.1` — [app_pkg/asgi.py](../../../app_pkg/asgi.py) hält asgirefs WSGI-Übersetzung byte-gleich, der Sentinel in [tests/test_asgi_adapter.py](../../../tests/test_asgi_adapter.py) pinnt `run_wsgi_app.func is` Upstream. (Gemessen in SYNC-FREEZE: 3.12.1 hat denselben Leak, unser Adapter trägt auf beiden.)
- `Flask-WTF==1.2.1` — die CSRF-Inversion **repliziert** dessen Guard-Kette; Sentinels in `tests/test_csrf_inversion.py`.
- `torch==2.12.1+cpu` — der IMG-SLIM-Pin läuft vor dem `requirements.txt`-Install und hält die CUDA-Wheels draußen; für die Ziel-Python-Version müssen `+cpu`-Wheels existieren.
- `deepgram-sdk==7.1.0` — kennt **keinen** typisierten `diarize_model`-kwarg; DIARIZE reicht `v2` als Query-Parameter durch. Ein Bump könnte das ändern.
- `PyMuPDF==1.24.1` — trägt beide PDF-Pfade und die Korpus-Werkzeuge.

**Das Basis-Image bringt Chromium mit** — beide Browser-Smokes ([scripts/smoke_document_converter.py](../../../scripts/smoke_document_converter.py), [scripts/smoke_audio_converter.py](../../../scripts/smoke_audio_converter.py)) laufen **im** Web-Container und hängen daran. Ein Image-Wechsel bewegt auch die Playwright-Version, und die trägt den Markdown→PDF-Pfad ([app_pkg/markdown.py](../../../app_pkg/markdown.py)) samt KaTeX-Assets.

**Der mineru-Container ist unbetroffen** — eigenes Image, eigene Python-Version, über den Socket gestartet.

**Testlage**: 985 grün. Die Suite rendert keine Templates; für die UI gibt es die zwei Browser-Smokes. `pandoc`, `soffice` und die docker-CLI kommen als Binaries ins Image, nicht über pip.

## Gesperrte Entscheidungen

1. **Erst prüfen, dann umziehen.** Phase 1 klärt beide Fristen an der **Quelle**. ⚠️ Verlass dich **nicht** auf die Daten in diesem Prompt oder im BACKLOG — sie sind vom 2026-07-31 und können falsch oder überholt sein. Das ist der ganze Punkt der Übung.
2. **Der Narrations-Pfad wird live geprüft, nicht gelesen.** Ein totes Modell zeigt sich als 404 im Aufruf, nicht in einer Änderungsliste. Ein echter Render ist der Beleg.
3. **Die Sentinel-Tests sind das Sicherheitsnetz des Umzugs.** Schlägt einer an, ist das ein **Befund**, kein Hindernis — er gehört in den Bericht, und der Vertrag dahinter wird neu verifiziert, nicht der Test angepasst.
4. **Kein Upgrade „bei der Gelegenheit"**, das nicht der Python-Wechsel erzwingt. Wer eine Version anfasst, die auch auf der Ziel-Version in der alten Fassung läuft, macht den Umzug unnötig groß und die Fehlersuche unmöglich.

---

# Phase 1 — Die beiden Fristen prüfen

## 1.1 Frist 1: Was stirbt am 16.10., und betrifft es uns?

Klär an der offiziellen Quelle: gilt die Abschaltung von `gemini-2.5-flash` **auch** für **`gemini-2.5-flash-tts`**? Und was ist der benannte Nachfolger für die TTS-Fläche?

⚠️ **Und prüf es live**: ein echter Narrations-Render über den Dienst, heute. Wenn er läuft, wissen wir, dass das Modell **jetzt** noch antwortet — und der Sprint weiß, ob er einen Wechsel vorbereiten muss oder eine Beobachtung.

Falls ein Wechsel nötig ist: er ist eine `.env`-Zeile (`NARRATION_TTS_MODEL`). ⚠️ Der Cloud-TTS-Modellname **unterscheidet sich** vom genai-Namen (deshalb `gemini-2.5-flash-tts` ohne `-preview-`-Infix) — der Nachfolger muss gegen die **Cloud-TTS**-Fläche verifiziert werden, nicht gegen die genai-Doku.

## 1.2 Frist 2: Was genau endet am 04.10.?

Klär, was die Google-Bibliotheken tatsächlich zusagen: keine neuen Releases für 3.10, oder mehr? Und welche Python-Version ist das sinnvolle Ziel — die, die das Playwright-Basis-Image in einer aktuellen Fassung mitbringt.

**Prüf die Wheel-Lage für die Ziel-Version**, bevor irgendwer ein Image baut: existieren `torch==2.12.1+cpu` und die übrigen Pins dort? Wo nicht, ist das ein benannter Teil des Umzugs, keine Überraschung mitten drin.

## 1.3 Der Bericht entscheidet den Umfang

Je Frist: was gilt wirklich, mit Quelle · betrifft es uns · was ist zu tun. Wenn Frist 1 uns nicht betrifft, sag das und lass es. Wenn sie uns betrifft, ist der Modellwechsel Teil von Phase 2.

## Stop
Beide Fristen geklärt, Narration live belegt. Bericht — Code nur, falls schon eine `.env`-Zeile fällig ist. Dann warten.

---

# Phase 2 — Der Umzug

## 2.1 Basis-Image und Pins

Neues Playwright-Python-Image, `requirements.txt` auf der Ziel-Version auflösen, `torch`-Vorab-Pin anpassen. Der pandoc-Release-deb, die docker-CLI und die apt-Pakete sind arch- und python-unabhängig — nicht anfassen, außer der Bau bricht.

⚠️ **Ein Pin nach dem anderen bewegen, nicht alle zugleich.** Wenn am Ende etwas kaputt ist, muss die Ursache auffindbar bleiben.

## 2.2 Das Sicherheitsnetz abarbeiten

`pytest tests/` (Baseline **985**) ist die erste Stufe, aber sie deckt nicht, was zählt. Namentlich zu prüfen:
- Die vier Sentinel-Verträge oben (nh3-camelCase, fsrs-Steps, asgiref-Adapter, Flask-WTF-Guard-Kette). Schlägt einer an: Befund in den Bericht.
- **Beide Browser-Smokes** im neuen Container — sie prüfen zugleich, dass Chromium und Playwright im neuen Image tragen.
- **Ein Markdown→PDF-Render mit Mathematik** — die KaTeX-Assets und `page.pdf()` hängen an der Playwright-Version. Optisch vergleichen, nicht nur auf Exit-Code.
- **Eine echte Dokument-Konvertierung je Backend** (pandoc, markitdown, trafilatura, unstructured, PDF cloud, PDF lokal). Der lokale Pfad prüft nebenbei, dass die docker-CLI und der Geschwister-Container-Start im neuen Image funktionieren.
- **Eine Transkription** und **ein Narrations-Render** — die beiden SDK-Flächen, um deretwillen der Umzug stattfindet.

## 2.3 Image-Größe

Nach dem Umzug messen und berichten (Stand heute: 5,88 GB). Ein Basis-Image-Wechsel kann sie in beide Richtungen bewegen; unkommentiert soll sie nicht bleiben.

## Stop
Suite grün, alle Belege aus 2.2 gezeigt. **Commit + Push** `build(OKTOBER): Python-Umzug auf <version> (P2)`. Dann warten.

---

# Phase 3 — Wrap

- **Ein Werkzeug gegen die nächste stille Frist.** Der ganze Vorgang existiert, weil ein abgeschaltetes Modell nur als WARNING sichtbar wurde. Ein kleines Skript unter `scripts/`, das die **konfigurierten** Modellnamen einmal real anspricht und laut wird, wenn einer nicht mehr antwortet, schließt genau diese Lücke — billig, und es prüft die Wirklichkeit statt eine Notiz. Docstring mit Laufanleitung, wie bei den Smokes.
- **CLAUDE.md**: Python-Version, Basis-Image, Modellnamen und die Pin-Begründungen stehen an mehreren Stellen — nachziehen, nicht ergänzen.
- **STATUS.md**, **BACKLOG.md** (Bullet-Guard): das Fristen-Item schließen; falls Frist 1 uns nicht betraf, **das** als Ergebnis festhalten statt es kommentarlos zu streichen.
- **Memory**, falls übertragbar; nach dem Schreiben mit `ls` prüfen, dass Datei und Index-Zeile zusammenpassen.
- **Im Bericht benennen**: was die beiden Fristen wirklich besagen, mit Quelle · welche Pins sich bewegen mussten und welche nicht · ob ein Sentinel angeschlagen hat · die Image-Größe vorher/nachher.

## Nicht-Ziele

- **Keine** Feature-Arbeit, **keine** Aufräum-Upgrades, die der Python-Wechsel nicht erzwingt.
- **Kein** Anfassen von mineru (eigenes Image) und **kein** Engine-Generation-Bump (die Konvertierungs-Ergebnisse ändern sich nicht — ⚠️ **es sei denn, ein Backend-Bump ändert sie doch**; dann bumpen und sagen warum).
- **Kein** Umbau der CSRF-Inversion oder des ASGI-Adapters — sie werden **verifiziert**, nicht neu entworfen.
- ⚠️ **Editiert wird nur auf dem Mac.** Die Mintbox ist Runtime — Deploy und Smoke ja, Arbeitsplatz nein, keine unversionierten Dateien zurücklassen; Wegwerf-User strikt nach `user_id` abräumen.
- ⚠️ **Vor dem ersten Deploy ein Prod-DB-Backup** nach dem WAL-Rezept in CLAUDE.md (`sqlite3.backup()` im Container, **nicht** nacktes `docker cp` der `.db`).
