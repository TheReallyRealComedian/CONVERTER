# Sprint KINDLE-MATH — Math im EPUB→Kindle-Pfad (Server-Render LaTeX→MathML) (M, 3 Phasen)

> **Executor-Doc.** Phasen nacheinander, nach jeder **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün (Baseline **534**). Du committest jede Phase selbst (Hash + push), **fokussiert**. Arbeitsverzeichnis `/Users/olivergluth/CODE/CONVERTER`. **Backend + Docs**, kein UI, **kein Schema-Touch**, **ein** neuer Dep (`latex2mathml`, pure-Python). Backend pytest-getestet (Mail/UI-frei).
>
> ⚠️ **Done-Gate jenseits pytest**: ein **echter Kindle-Geräte-Smoke durch Oli** ist Pflicht, BEVOR der Sprint wirklich „done" ist — pytest rendert kein Gerät (CLAUDE.md Test-Suite-Limit). Dazu am Ende mehr.

## Warum & Entscheidungen (gesetzt — aus Research+Design-Workflow, Findings live-verifiziert)

MATH-RENDER hat Math in Reader/Preview/PDF via KaTeX gelöst, aber **EPUB blieb offen**: E-Reader führen kein verlässliches JS aus, also bleiben die `math-inline`/`math-display`-Spans im EPUB als roher LaTeX-Text stehen. Fix: **server-seitig LaTeX→MathML** beim EPUB-Bau, eingebettet als **EPUB3-MathML**.

- **MathML-first via pure-Python `latex2mathml`** (kein Node, kein TeX, kein Playwright-im-Request). Ein Post-Pass über die Spans im fertigen HTML-Body, **bevor** er in den EPUB-Chapter geht. **Begründung**: MathML ist korrekt, reflowbar, font-skalierbar, screenreader-zugänglich auf In-App-Reader / Apple Books / modernen E-Readern — und auf Kindle **nie schlechter** als der heutige Roh-LaTeX-Status-quo. Eine echte LaTeX→Bild-Lösung (robuster auf alten Kindles) ist **L** an spekulativer Infra für eine **noch nicht beobachtete** Kindle-Schwäche → premature optimization.
- **Bild-Pfad flag-gegated, aber NICHT gebaut**: `EPUB_MATH_MODE` (Default `mathml`). Der Bild-Escape-Hatch (Playwright→PNG) wird **erst** gebaut, wenn Olis **Geräte-Smoke** beweist, dass Kindle MathML unzumutbar rendert. Das Flag dient zugleich als **Kill-Switch** (`EPUB_MATH_MODE=off` → heutiges Verhalten ohne Code-Change).
- **Per-Gleichung-Fallback PFLICHT**: `latex2mathml` **wirft** bei kaputtem/partiellem LaTeX (`NoAvailableTokensError`/`MissingEndError`/… — anders als KaTeX `throwOnError:false`). Eine einzige schlechte Formel ohne try/except crasht den ganzen EPUB-Bau → 502 beim Senden. Auf Exception **den Original-Span (sichtbarer Roh-LaTeX) stehen lassen**.
- **`alttext`=LaTeX** auf jedem `<math>` (Accessibility-/Recovery-Boden, MathML-nativ, harmlos für Reader die's ignorieren).
- **Verworfen**: in-file MathML+`altimg` / `epub:switch` (Reader inkl. Kindle ignorieren `altimg`; `epub:switch` ist seit EPUB 3.1 deprecated → zahlt für zwei Renderer, Kindle zeigt keinen gut).
- **Pures Modul** `services/epub_math.py` (wie `epub_service`/`kindle_service` — kein SDK-Singleton), unit-testbar als reiner String→String-Transform. **`app_pkg/kindle.py` bleibt unberührt** (ruft schon `build_epub(title, render_markdown_to_html(content))`).

## Verifizierte Code-Fakten (Master-gegroundet + Workflow-live-verifiziert)

- **Span-Format** ([app_pkg/markdown_render.py:37-42](app_pkg/markdown_render.py)): `<span class="math-inline">{escapeHtml(latex.strip())}</span>` (für `$…$`) bzw. `class="math-display"` (für `$$…$$`). `escapeHtml` macht **nur** `< > & "` → Entities → der **rohe LaTeX ist via lxml `el.text_content()` rückgewinnbar** (lxml dekodiert die Entities). Der `math-display`-Span ist **bare** (kein `<p>`/`<div>`-Wrap).
- **nh3 läuft UPSTREAM** in `render_markdown_to_html` (erlaubt `span`/`class`). Der EPUB-Pfad hat **keinen zweiten Sanitizer** ([services/epub_service.py](services/epub_service.py) sanitized nicht) → MathML hier zu injizieren ist safe; **nh3 NICHT erneut** auf dem transformierten Body laufen lassen (es würde `<math>` strippen).
- **EPUB-Bau** ([services/epub_service.py](services/epub_service.py)): Single-Chapter **EPUB3**; der Seam ist **Zeile 49** `inner = (html_body or '').strip() or '<p></p>'`, dann `chapter.content = f'<html><body>{inner}</body></html>'` (Z.50). `EpubHtml(...)` wird in **Z.43** erzeugt.
- **`latex2mathml` (live-verifiziert)**: `from latex2mathml.converter import convert`; `convert(latex, display='inline'|'block')` → namespaced `<math>` mit gebackenem `display`-Attribut. **Wirft** bei kaputtem LaTeX. Default `display='inline'` für ALLES (auch `$$…$$`) → auf dem `math-display`-Zweig **explizit `display='block'`** übergeben.
- **ebooklib (live-verifiziert)**: **`chapter.properties.append('mathml')`** emittiert `properties="mathml"` im OPF-Manifest-Item UND das MathML übersteht ebooklibs lxml-Serialisierung. **`EpubHtml(..., properties=['mathml'])` als Konstruktor-kwarg → TypeError** (zwei Design-Vorschläge lagen hier falsch — **nur** die Attribut-Form nutzen).
- **`lxml`** ist **kein neuer Dep** (schon transitiv via ebooklib) — in `services/epub_math.py` aber **explizit importieren**.
- **Einziger neuer Dep**: `latex2mathml` (pure-Python, MIT). Lokal `3.79.0` importierbar, Web-Finding nennt `3.81.0` (2026-04-15) → **pinnen, was auf dem Mintbox-Docker-Build sauber installiert** (verifizieren).
- **Für den DEFERRED Bild-Pfad** (nicht dieser Sprint): Playwright 1.44 + vendored KaTeX 0.16.11 + `_katex_pdf_assets()`/`_KATEX_RENDER_JS` in [app_pkg/markdown.py](app_pkg/markdown.py) — `page.pdf()` → `element.screenshot()` tauschen, **PNG** data-URI `<img>` (NICHT SVG — Kindle ist SVG-feindlich), `alt`=LaTeX. Die Maschinerie erst in ein `services/`-Modul heben.

## Phase 1 — Pures Transform-Modul `services/epub_math.py` + Tests

1. **`services/epub_math.py`** (pures Modul, kein Flask/SMTP): `latex_spans_to_mathml(html_body: str) -> tuple[str, bool]` → `(transformed_body, has_math)`.
   - Parse mit **`lxml.html.fragment_fromstring(html_body, create_parent='div')`** (vermeidet den impliziten `html/body`-Shell); am Ende **nur den Inner-Fragment-Content** zurück-serialisieren (nicht den `create_parent`-Wrapper).
   - Für jedes Element mit class `math-inline` **oder** `math-display`:
     - `latex = el.text_content()` (lxml dekodiert die 4 Entities → exakter LaTeX).
     - **leer/whitespace** → Original-Span unberührt lassen, weiter.
     - `try: mathml = convert(latex, display='block' if 'math-display' in classes else 'inline')` — **`except Exception:`** Original-Span unberührt lassen (sichtbarer Roh-LaTeX), weiter.
     - Erfolg: das `<math>`-String parsen, **`alttext=latex`** drauf setzen, den **Span im Baum ersetzen**. Für `math-display`: das `<math>` in einen **Block-Container** (`<p class="math-display">`) wrappen (Source-Span ist bare → sonst säße Display-Math inline).
   - `has_math = True` nur wenn **≥1** `<math>` emittiert wurde.
2. **Tests** `tests/test_epub_math.py` (reiner String→String, kein Flask/SMTP/EPUB):
   - `math-inline` → `<math` mit `display="inline"`; `math-display` → `display="block"` **und** block-gewrappt (nicht inline-im-Fluss).
   - **malformed LaTeX** (z.B. `\frac{1}{`) → der sichtbare Original-Span bleibt, **wirft NICHT** (load-bearing Safety-Test — ohne try/except 502t der ganze Send).
   - leer/whitespace-LaTeX → Span bleibt; **math-freier Body → `(body, False)` byte-identisch**; Nicht-Math-HTML rund um die Spans bleibt erhalten (Round-Trip-Integrität — fängt lxml-Serialisierungs-Korruption).
3. Pre-Flight `pytest tests/` grün.

**Stop + Bericht.**

## Phase 2 — Wiring in `epub_service.py` + OPF-Property + Flag-Gate

1. In [services/epub_service.py](services/epub_service.py): `EPUB_MATH_MODE` via `os.environ.get('EPUB_MATH_MODE', 'mathml')` lesen.
   - **Vor** dem `inner = …`-Seam (Z.49): wenn `mode == 'mathml'` → `html_body, has_math = latex_spans_to_mathml(html_body)`. Sonst (`off`/`image`/unbekannt) → passthrough (heutiges Verhalten; `image` ist der dokumentierte-aber-ungebaute Escape-Hatch). Den Empty-Body-Placeholder-Guard **nach** dem Transform lassen.
   - Wenn `has_math` → **`chapter.properties.append('mathml')`** (Attribut-Form, **nicht** Konstruktor-kwarg — wirft TypeError). Wenn nicht → `properties` unberührt (math-freie EPUBs bleiben byte-stabil vs. heute → Regressions-Sicherheit).
2. **Tests** in `tests/test_epub_service.py` (das bestehende `_read_back`/zip-Inspect nutzen):
   - `$$…$$`-Body → EPUB-`chapter.xhtml` enthält `<math` UND OPF-Manifest-Item trägt `properties="mathml"`.
   - math-freier Body → **kein** `properties="mathml"` (Byte-Stabilitäts-Regressions-Guard).
   - bestehende Title/Content-Round-Trip- + Empty/None-Body-Tests bleiben grün.
3. Bestätigen: **`app_pkg/kindle.py` braucht NULL Änderung** (ruft schon `build_epub(title, render_markdown_to_html(content))`).
4. Pre-Flight `pytest tests/` grün.

**Stop + Bericht.**

## Phase 3 — requirements + Docs + Wrap-up

1. **`requirements.txt`** — `latex2mathml` nach der `ebooklib`-Zeile zufügen (Kommentar: EPUB/Kindle-Math). Pin verifizieren (installiert sauber auf dem Mintbox-Build).
2. **`docs/kindle.md`** — Server-seitiger MathML-Pass beim Bau, das `EPUB_MATH_MODE`-Flag (`mathml` Default, `off` Kill-Switch, `image` dokumentiert-aber-ungebaut), per-Gleichung-Fallback-auf-sichtbaren-LaTeX.
3. **`CLAUDE.md`** — Architecture-Notes Kindle-Bullet: server-seitiges MathML im EPUB-Pfad ergänzen.
4. **STATUS.md** + **BACKLOG.md** — P3-Item „EPUB/Kindle-Math" schließen (Entscheid: MathML-first + flag-gegateter, ungebauter Bild-Escape-Hatch); **Bullet-Guard** (`grep -nE '(- \*\*.*){2,}' BACKLOG.md`).
5. **Memory** — `reference_math_protect_then_render` **erweitern** (oder ein Geschwister `reference_*`): MathML-first im EPUB geshippt via pure-Python `latex2mathml`; Bild-Pfad deferred hinter `EPUB_MATH_MODE`; die zwei Gotchas (**ebooklib properties-Attribut-nicht-kwarg**, **latex2mathml-wirft-→-try/except-Pflicht**); `alttext`-Boden; `altimg`/`epub:switch` verworfen. MEMORY.md-Pointer falls neu.
6. **Sprint-Prompt-Doc** (dieses File) mit eincheckbar.
7. Finaler `pytest`.

**Stop + Schluss-Bericht** — inkl. Deploy-Kette: Mac push → Mintbox `git pull` + `docker compose up -d --build` (**neuer Dep `latex2mathml` → Image rebuildt; kein Schema, keine Migration, kein neuer Token**). **Danach Olis Geräte-Smoke** (s.u.) als finaler Done-Gate.

## Done-Gate: Geräte-Smoke (Oli, nach Deploy)

pytest kann kein Gerät rendern. **Oli sendet ein echtes Math-Dokument an seinen Kindle** (Gerät **und** Kindle-App) und prüft, ob das MathML zumutbar rendert.
- **Rendert gut** → Sprint wirklich done; Bild-Pfad bleibt ungebaut.
- **Rendert schlecht** → triggert einen **separaten L-Follow-on** (`EPUB_MATH_MODE=image`): Playwright+vendored-KaTeX→PNG data-URI `<img>` (`alt`=LaTeX), die Maschinerie aus `app_pkg/markdown.py` erst in ein `services/`-Modul heben. **Nicht** Teil dieses Sprints.

## Bewusst NICHT (Scope-Grenze)

- **Kein** Bild-Render-Pfad bauen (deferred hinter dem Flag bis zum Geräte-Smoke).
- **Kein** `altimg`/`epub:switch`-In-File-Hybrid (verworfen).
- **Kein** Node/TeX/Playwright-im-Request, **kein** zweiter Sanitizer auf dem EPUB-Body.
- **Kein** Schema-Touch, kein neuer Token; einziger Dep = `latex2mathml`.
- **`app_pkg/kindle.py`**, `services/kindle_service.py`, der Reader/Preview/PDF-KaTeX-Pfad — **unberührt**.

## Akzeptanz

- [ ] `services/epub_math.py` `latex_spans_to_mathml` (pure): Spans→EPUB3-MathML, `display='block'` für display + Block-Wrap, `alttext`=LaTeX, **per-Gleichung try/except → Fallback auf sichtbaren Span**, math-frei → byte-identisch + `has_math=False`.
- [ ] `epub_service.py` ruft den Transform unter `EPUB_MATH_MODE=='mathml'` (Default), setzt `chapter.properties.append('mathml')` nur bei `has_math`; `off`/`image` = passthrough; `kindle.py` unverändert.
- [ ] Tests: `test_epub_math.py` (inkl. malformed-LaTeX-wirft-nicht + Round-Trip) + `test_epub_service.py` (`<math`+`properties=mathml` bei Math, abwesend bei math-frei). `pytest` grün ≥ 534 + neue.
- [ ] `requirements.txt`-Pin verifiziert; docs/kindle.md + CLAUDE.md + STATUS/BACKLOG + Memory aktualisiert; fokussierte Commits.
- [ ] **Geräte-Smoke durch Oli** als Done-Gate notiert (nicht vom Sub-Thread leistbar).
