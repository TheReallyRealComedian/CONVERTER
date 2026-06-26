# Sprint MATH-RENDER — LaTeX-Mathe rendern (KaTeX) in Reader + Preview + PDF (M, 3 Phasen)

> **Executor-Doc.** Phasen nacheinander, nach jeder **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün (Baseline **435**). Du committest jede Phase selbst (Hash + push). Arbeitsverzeichnis `/Users/olivergluth/CODE/CONVERTER`. **Frontend-lastig** → Live-Smoke (echte gerenderte Formeln, dark+light, 0 Console-Errors) ist der Gate; pytest deckt nur den Server-Renderer.

## Warum

Markdown mit LaTeX-Mathe (`$...$` inline, `$$...$$` Block) wird **roh** angezeigt — `\frac{dC}{dt}`, `C_{\text{in}}` etc. erscheinen als Quelltext (Oli-Screenshot, Reader). Der geteilte Renderer hat kein Mathe-Plugin, im Frontend kein KaTeX.

## Entscheidungen (gesetzt)

- **Ansatz**: Mathe **schützen** (Server tokenisiert sie, bevor Markdown sie zerlegt) **+ rendern** (KaTeX client). Reine client-seitige Lösung wäre fragil (Markdown frisst `_`/`\`/`{}` vorher).
- **Flächen**: **Reader** (`library_detail`) · **markdown-converter Live-Preview** · **PDF-Export** (Playwright). **EPUB/Kindle out** (Kindle-JS unzuverlässig → eigener Folge-Sprint, server-render/MathML).
- **Server-Plugin**: `dollarmath` aus **`mdit-py-plugins`** (neuer pip-Dep) im geteilten `render_markdown_to_html`.
- **KaTeX-Assets**: **vendored** unter `static/vendor/katex/` (CSS+JS+Fonts) — zuverlässig in Playmwright + offline, kein CDN-zur-Render-Zeit. (CDN wäre die leichtere Alternative, aber für den PDF-Export riskanter.)

## Verifizierte Code-Fakten (Master-gegroundet)

- **Geteilter Renderer**: `render_markdown_to_html` ([app_pkg/markdown_render.py](app_pkg/markdown_render.py)) — `MarkdownIt('default', {...})` + `nh3.clean(rendered, tags=_ALLOWED_TAGS, attributes=_ALLOWED_ATTRIBUTES)`. **Kein** `.use(...)`. `_ALLOWED_TAGS` enthält `span`+`div`; `_ALLOWED_ATTRIBUTES['*'] = {'class','id','style'}` → **class-getaggtes** Mathe-Markup übersteht nh3 **ohne** Allow-List-Änderung (Display-Modus über die **Klasse**, nicht ein `data-`-Attr).
- **Nutzer des geteilten Renderers**: Reader (`library.py:350` → `library_detail.html:65` `{{ content_html | safe }}` in `.reader-view`) · PDF-Flow (`markdown.py:141` `render_markdown_to_html` → `full_html` → Playwright `set_content` → `page.pdf()`) · EPUB (`epub_service.build_epub`, **hier out**).
- **Preview ist separat**: client `markdownit(...)` (CDN) in [static/js/markdown_converter.js](static/js/markdown_converter.js) → `buildIframeDoc` baut `srcdoc`. **Eigener** Mathe-Pfad nötig (client markdown-it-Plugin + KaTeX).
- **PDF-`full_html`**: in `convert_markdown` ([app_pkg/markdown.py](app_pkg/markdown.py)) als f-String mit `<head><style>…</style></head><body>{html_content}</body>` — hier KaTeX-CSS/JS + Render-Aufruf injizieren; Playwright lädt schon Webfonts (`document.fonts.ready`), hat also einen vollen Chromium.
- **`mdit-py-plugins` ist NICHT installiert** (Import schlägt fehl) → neuer Dep.

## Phase 1 — Server-Mathe-Schutz + Reader-Rendering

1. **Dep** `mdit-py-plugins` (gepinnt) in `requirements.txt`.
2. **`render_markdown_to_html`** ([markdown_render.py](app_pkg/markdown_render.py)): `from mdit_py_plugins.dollarmath import dollarmath_plugin`; `_md.use(dollarmath_plugin, …)`. **Ziel-Output** (damit nh3 + client-KaTeX greifen): inline → `<span class="math-inline">…rohes latex…</span>`, block → `<span class="math-display">…</span>` (oder `<div>`). Den Plugin-`renderer` entsprechend setzen (oder Output nachmappen). **Konservativ konfigurieren** gegen Streu-`$` (Preise): keine Space-adjazenten Delimiter als Mathe werten (`$ 5 $`-Text bleibt Text).
   - **nh3**: prüfen, dass das class-getaggte Markup durchgeht (span/div/class sind erlaubt → i.d.R. **keine** Allow-List-Änderung); das rohe latex überlebt als Text-Content (KaTeX liest `textContent`).
3. **Tests** (`tests/test_markdown_render.py` o.ä.): `$$\frac{dC}{dt}$$` → `math-display`-Span mit **intaktem** `\frac{dC}{dt}` (nicht markdown-zerlegt); `$C_{\text{in}}$` → `math-inline` mit intaktem `_{\text{in}}` (kein `<em>`); **Streu-`$`** („kostet 5$ und 10$") bleibt Text, wird **nicht** Mathe; Nicht-Mathe-Content unverändert; nh3 sanitisiert weiter (kein XSS via Mathe). `pytest` grün ≥ 435.
4. **KaTeX-Assets** vendoren unter `static/vendor/katex/` (CSS+JS+Fonts; Version pinnen).
5. **Reader**: KaTeX-CSS einbinden + ein Render-Script (im Reader-Kontext, nach Content-Render): `document.querySelectorAll('.math-inline, .math-display').forEach(el => katex.render(el.textContent, el, {displayMode: el.classList.contains('math-display'), throwOnError: false}))`. Greift auf der normalen Detailseite **und** im Reader-Mode (gleicher DOM). Wo einbinden: `library_detail.html` (Assets) + ein kleiner Block in `library_detail.js` (Render on load).
6. **Live-Smoke Reader** (lokale Instanz/echtes Doc, MacChrome **dark+light**, **0 Console-Errors**): Olis exakter Inhalt (`$$\text{Akkumulation} = …$$`, `$$\frac{dC}{dt} = \frac{F}{V}\,(C_{\text{in}} - C) + r$$`, inline `$C$`/`$F$`/`$V$`) → **gerenderte Formeln** statt Quelltext; KaTeX-Farbe erbt den Text (lesbar in beiden Themes).

**Stop + Bericht.** (Damit ist Olis Hauptschmerz — der Reader — gelöst.)

## Phase 2 — markdown-converter Preview + PDF-Export

1. **Preview** ([markdown_converter.js](static/js/markdown_converter.js)): den client-`markdownit` um ein Mathe-Plugin erweitern (`markdown-it-texmath` o.ä., das per KaTeX **rendert**), KaTeX im Parser-Kontext laden; in `buildIframeDoc` die **KaTeX-CSS** in den iframe-`<head>` injizieren (die gerenderte KaTeX-HTML kommt aus dem Parent → iframe braucht nur das CSS, kein JS). Smoke: `$$\frac{a}{b}$$` im Editor tippen → Preview zeigt **live** die gerenderte Formel; bestehender Preview (Themes, Orientation, Dark) unberührt.
2. **PDF** ([markdown.py](app_pkg/markdown.py) `convert_markdown`): in `full_html` die **KaTeX-CSS** (`<head>`) + **KaTeX-JS** einbinden und nach `set_content` (vor `page.pdf()`) die Mathe-Spans rendern — `await page.evaluate(<render-all .math-inline/.math-display via katex.render>)` (KaTeX rendert **synchron** → keine Race). Reuse der vendored Assets (per `file://`/Static-Pfad in den Playwright-Kontext, oder als Inline-Inject). Smoke: ein Deck mit Mathe exportieren → **PDF enthält gerenderte Formeln** (nicht `$$…$$`).
3. `node --check` der berührten JS; `pytest` grün.

**Stop + Bericht.**

## Phase 3 — Wrap

1. **STATUS.md** + **BACKLOG.md**: MATH-RENDER ☑ done mit Hashes; **EPUB/Kindle-Math als P3-Folge-Item** notieren (server-render/MathML, Kindle-JS unzuverlässig). „Aktiv offen" → Web-Article-Save (P2). **Bullet-Guard.**
2. **Doc**: kurzer Eintrag (z.B. `docs/reader_architecture.md` oder neue Notiz) — Mathe-Pipeline: dollarmath-**Schutz** im geteilten Renderer (class-getaggte Spans) + **KaTeX-Render** pro Fläche (Reader-JS · Preview-iframe · Playwright-evaluate); EPUB bewusst offen.
3. **Memory** (`reference_*`): das wiederverwendbare Muster — „Mathe **schützen** (tokenize vor Markdown) **dann** rendern (KaTeX client)", die class-getaggten Spans übersteh­en nh3 ohne Allow-List-Touch, der geteilte-Renderer-Hebel (Reader+PDF teilen den Schutz, KaTeX pro Fläche), EPUB-JS-Grenze. MEMORY.md-Pointer.
4. Finaler `pytest`.

**Stop + Schluss-Bericht** — inkl. Olis Deploy-Schritt: Mintbox `git pull` + `docker compose up -d --build` (neuer Dep `mdit-py-plugins` + vendored static → Image baut neu; **keine Migration**), Browser-Hard-Reload.

## Bewusst NICHT (Scope-Grenze)

- **EPUB/Kindle-Math** — Kindle-JS rendert KaTeX nicht zuverlässig; echtes EPUB-Math bräuchte Server-Render (KaTeX→MathML) → **eigener Folge-Sprint**. (Nach diesem Sprint zeigt das EPUB die Mathe als rohes latex in Spans — wie bisher roh, nicht schlechter.)
- **Kein** Markdown-/Content-Touch — nur Render-Pipeline.
- **Kein** Schema-Touch, kein Backend-Endpoint.
- **Konservative `$`-Regeln** — lieber eine Formel nicht erkennen als Streu-`$` (Preise) zerschießen.

## Akzeptanz

- [ ] `$...$`/`$$...$$` rendern als echte Formeln im **Reader**, in der **markdown-converter-Preview** und im **PDF-Export**
- [ ] Mathe ist vor Markdown-Zerlegung geschützt (Server-`dollarmath`) — `C_{\text{in}}`/`\frac{}{}` bleiben intakt
- [ ] Streu-`$` (Preise) wird **nicht** zu Mathe; Nicht-Mathe-Content unverändert; nh3 sanitisiert weiter
- [ ] KaTeX-Assets vendored, in beiden Themes lesbar (Farbe erbt Text)
- [ ] `pytest` grün ≥ 435 + neue Tests; Live-Smoke dark+light, 0 Console-Errors
- [ ] EPUB-Math als Folge-Item notiert (nicht in diesem Sprint)
