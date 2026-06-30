# Sprint IMG-SLIM — CPU-only-Torch-Pin (Image entschlacken) (M, 3 Phasen)

> **Executor-Doc.** Phasen nacheinander, nach jeder **Stop + Bericht**, auf Sign-off warten. **Phase 1 ist ein hartes Mess-Gate** — NICHT zur Implementierung übergehen, bevor die gemessene Ersparnis das M-Effort rechtfertigt + Master/Oli ge-go't haben. Arbeitsverzeichnis `/Users/olivergluth/CODE/CONVERTER` (Mac = Source-of-Truth). Backlog-Item **MAC1-FOLLOWUP-B**.
>
> **Kontext**: Das Docker-Image trägt die NVIDIA-CUDA-Wheels von PyTorch mit, die via `unstructured[all-docs]` reinkommen. Sie sind **bewiesen tote Bytes**: die Mintbox hat zwar eine RTX A2000, aber (1) der Container hat **kein** GPU-Passthrough und (2) der Code fährt `partition(strategy="fast")` (kein Torch-Modell) / PDFs über Gemini → **Torch-Neural-Inferenz läuft nie**. Memory `reference_mintbox_gpu_unstructured_cpu_path`. Ziel: `torch==…+cpu` pinnen → CUDA-Wheels raus, **verhaltensneutral**.

## ⚠️ Zwei korrigierte Fakten (Master-gegroundet 2026-06-30) — lies zuerst

1. **Die „4,82 GB" im Backlog sind falsch.** Das lokale `converter-app:latest` ist **15,5 GB**. Der reale Slim-Gewinn (nur die CUDA-Wheels) ist daher ein *kleinerer Anteil* eines großen Images (Playwright-Base + Browser, `libreoffice`/`tesseract`/`pandoc`/`ffmpeg`/`ghostscript`, NLTK-Daten sind ebenfalls dick). **Phase 1 misst die echte Ersparnis, bevor irgendwas geändert wird.**
2. **Plattform-Nuance.** Die NVIDIA-CUDA-Wheels (`nvidia-cu12-*`) gibt es **nur für linux/amd64**, nicht arm64. Baut der Mac (Apple Silicon) per Default **arm64**, hat das *lokale* Image **gar keine** CUDA-Bloat — die liegt nur im **amd64-Mintbox-Image** (das deployte). Heißt: messen + verifizieren muss **amd64**-repräsentativ sein, nicht am evtl. arm64-Mac-Image.

## Verifizierte Code-Fakten

- **Dockerfile** ([Dockerfile](Dockerfile)): Base `mcr.microsoft.com/playwright/python:v1.44.0-jammy` (Python 3.10). Torch kommt via **Z.23** `RUN pip install --no-cache-dir --timeout=600 --retries=5 -r requirements.txt` (transitiv über `unstructured[all-docs]==0.18.32`, [requirements.txt:10](requirements.txt)). NLTK-Heredoc Z.26–58 (nutzt `nltk`, **nicht** torch). `COPY . .` Z.60.
- **Kein direkter Torch-Use in unserem Code** (zu bestätigen in P1 per grep): Extraktion ([app_pkg/documents.py:65](app_pkg/documents.py)) ruft `partition(strategy="fast")` (ML-frei), PDFs → `PDFExtractionService` → Gemini. Deepgram/Gemini/Google-TTS sind APIs. Torch wird nur **von unstructured importiert**, nie für Inferenz gefahren.
- **Beide Container teilen das Image**: [docker-compose.yml](docker-compose.yml) `markdown-converter` baut `converter-app:latest`, `worker` re-used dasselbe → Slim hilft beiden.
- **Mac kann bauen** (Docker-Daemon läuft, v28.5.2) — aber ein amd64-Build unter Emulation ist **langsam**; ggf. `run_in_background` + Monitor.

## Phase 1 — Messen & Gate (HART, kein Code-Change)

**Ziel: die echte amd64-Ersparnis kennen, bevor wir M-Effort investieren.**
1. **Platform + Wheel-Bestand des bestehenden Images** (billig, kein Rebuild):
   - `docker image inspect converter-app:latest --format '{{.Architecture}} {{.Size}}'` → arm64 oder amd64?
   - `docker run --rm converter-app:latest sh -lc "python -c 'import torch,platform; print(torch.__version__, platform.machine())'; pip list 2>/dev/null | grep -iE 'torch|nvidia'"` → ist `torch` ein CUDA-Build? Sind `nvidia-*`-Wheels da? Wie groß? (`du -sh` auf die site-packages-`nvidia*`/`torch`-Verzeichnisse — Pfad via `python -c "import site;print(site.getsitepackages())"`).
2. **Falls das lokale Image arm64 ist** (→ keine CUDA-Wheels lokal): die Bloat ist amd64-only. Dann **einen amd64-Probe-Build** für die echte Zahl: `docker build --platform linux/amd64 -t converter-app:amd64-probe .` (ggf. `run_in_background`), danach dieselbe `pip list`/`du`-Messung im amd64-Image. **Oder** — falls amd64-Build-auf-Mac unzumutbar langsam — die Messung als **Mintbox-Schritt** an Oli übergeben (präzises Kommando liefern) statt zu raten.
3. **Torch-Zielversion bestimmen**: welche torch-Version zieht `unstructured[all-docs]==0.18.32`? (aus der `pip show torch`-Ausgabe im Image). Bestätigen, dass es den **`+cpu`-Wheel** für **genau diese Version**, **cp310**, **linux/amd64** auf `https://download.pytorch.org/whl/cpu` gibt (sonst Versions-Konflikt mit unstructured).
4. **Direkter Torch-Use ausschließen**: `grep -rnE "import torch|from torch" app_pkg/ services/ tasks.py worker.py app.py` → erwartet **leer** (nur unstructured nutzt torch intern).

**Stop + Bericht**: echte amd64-Image-Größe, CUDA-Wheel-Gewicht (= projizierte Ersparnis), Torch-Zielversion + `+cpu`-Wheel-Existenz bestätigt, kein direkter Torch-Use. **Go/No-Go-Empfehlung**: lohnt der Slim (z.B. Ersparnis ≥2 GB) das M-Effort + die Verify-Komplexität? **Warten auf Sign-off.**

## Phase 2 — Implementieren (nur nach Go)

1. **Dockerfile**: **vor** Z.23 (dem `requirements.txt`-Install) einen CPU-Torch-Install einfügen:
   ```dockerfile
   # CPU-only PyTorch (die CUDA-Wheels sind tote Bytes — kein GPU-Passthrough + ML-freier Pfad)
   RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==<ZIELVERSION>
   ```
   Damit ist `torch` schon befriedigt, wenn `unstructured` installiert → pips Resolver zieht **nicht** die CUDA-Variante. (Falls der Resolver bei Z.23 dennoch CUDA-Torch nachziehen will: auf eine **constraints**-Datei oder `PIP_EXTRA_INDEX_URL`/`--extra-index-url` ausweichen — robust machen, nicht raten.)
2. **Build** (amd64, repräsentativ): `docker build --platform linux/amd64 -t converter-app:slim .`. Im gebauten Image **beweisen**: `pip list | grep -iE 'torch|nvidia'` → `torch …+cpu`, **keine** `nvidia-*`-Wheels mehr.
3. **Größe vorher/nachher** belegen (`docker images …`).

**Stop + Bericht** (alte vs. neue amd64-Größe, `+cpu` bestätigt, nvidia-Wheels weg).

## Phase 3 — Verifizieren (Container-Smoke) + Wrap

**pytest beweist hier NICHTS** (es mockt `partition` → fängt einen Torch-Import-Break nicht). Der echte Gate ist der Container-Smoke:
1. **Import-Smoke** im slim-Image: `docker run --rm converter-app:slim python -c "import torch; print(torch.__version__, torch.cuda.is_available()); from unstructured.partition.auto import partition; print('partition-import ok')"` → `+cpu`, `False`, ok.
2. **Echter Extraktions-Smoke**: ein kleines **`.docx`** (und wenn möglich `.pptx`) durch `partition(filename=…, strategy="fast")` im Container jagen (direkter `docker run`-Einzeiler oder über die laufende App-Route) → liefert Elemente/Markdown, kein Crash. (Das ist der Pfad, den der Torch-Swap riskiert.)
3. **PDF-Pfad unberührt bestätigen**: PDFs gehen über Gemini, nicht torch — der Import/Route-Pfad muss intakt sein (kein Live-Gemini-Call nötig).
4. **pytest** `tests/` grün (612, Sanity).
5. **Wrap**:
   - **BACKLOG.md** MAC1-FOLLOWUP-B → ☑ done (echte Vorher/Nachher-Größe statt „4,82 GB"; die falsche Zahl explizit korrigieren). **STATUS.md** Eintrag (pytest 612 unverändert; Dep-Mechanik = CPU-Torch-Pin).
   - **CLAUDE.md** falls Image/Build erwähnt: CPU-Torch-Pin notieren.
   - **Memory** `reference_mintbox_gpu_unstructured_cpu_path` ergänzen (echte Image-Größen + der Resolver-Pin-Mechanismus + die arm64/amd64-Plattform-Nuance) **oder** eigene `reference_cpu_torch_pin_past_resolver` falls die Technik (CPU-Wheel vor dem Extra-Resolver pinnen) als wiederverwendbar taugt.
   - **Bullet-Guard** vor BACKLOG/STATUS-Commit (`grep -nE '(- \*\*.*){2,}'`).
6. **Deploy-Notiz** (Olis Schritt): Mintbox `git pull` + `docker compose up -d --build` rebuildt amd64 mit CPU-Torch → kleineres Image. Der lokale amd64-Verify de-riskt den Deploy. **Kein** Schema/Token; der einzige Touch ist der Dockerfile-Torch-Pin.

**Stop + Schluss-Bericht.**

## Bewusst NICHT (Scope-Grenze)

- **Kein** GPU-Passthrough, **kein** `strategy="hi_res"` (die GPU-*Nutzung* ist verworfen — separater Scope).
- **Kein** Anfassen der Playwright-Base / `libreoffice` & Co. (eigene, größere Slim-Frage — falls Phase 1 zeigt, dass die der wahre Brocken sind: als **separates Item** vermerken, nicht hier mit-slimmen).
- **Kein** Python-Bump (3.10→3.11 ist ein eigenes Item).

## Akzeptanz

- [ ] **P1**: echte amd64-Image-Größe + CUDA-Wheel-Gewicht gemessen, Torch-Zielversion + `+cpu`-Wheel bestätigt, kein direkter Torch-Use, **Go/No-Go** mit echten Zahlen — Sign-off vor P2.
- [ ] **P2**: Dockerfile-CPU-Torch-Pin, amd64-Build grün, `pip list` zeigt `torch+cpu` + **keine** `nvidia-*`-Wheels.
- [ ] **P3**: Import-Smoke + **echter docx/pptx-Extraktions-Smoke** im slim-Image grün, PDF-Pfad intakt, pytest 612; Größe vorher/nachher belegt; Docs/Memory/BACKLOG (4,82-GB-Zahl korrigiert) + Bullet-Guard.
- [ ] Verhaltensneutral (gleiche Extraktion), **kein** Schema/Token; Deploy = reiner `up -d --build`.
