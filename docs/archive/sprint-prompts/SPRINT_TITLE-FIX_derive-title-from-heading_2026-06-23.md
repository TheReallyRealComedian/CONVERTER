# Sprint TITLE-FIX — Titel aus erster Überschrift ableiten (statt erster Zeile) (S, 2 Phasen)

> **Executor-Doc.** Phasen nacheinander, nach jeder **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün (Baseline **393**). Du committest jede Phase selbst (Hash + push). Arbeitsverzeichnis `/Users/olivergluth/CODE/CONVERTER`. Working-Practice in `CLAUDE.md`.

## Warum

PDF→Markdown-Decks beginnen mit `<!-- Seite 1 -->` (Seiten-Marker) — die Titel-Ableitung „erste Zeile" macht daraus den Titel `<!-- Seite 1 -->`. In der Library (und für den converter-mcp-`list_conversions`-Finder) sind die Decks dann per Titel ununterscheidbar, obwohl direkt nach dem Kommentar eine echte `#`-Überschrift steht (z.B. „Addiction to product expression"). **Inhalt bleibt unangetastet** — nur die Titel-Ableitung wird schlauer.

## Entscheidungen (gesetzt, Oli 2026-06-23)

- **A** Server-seitige Smart-Ableitung in **beiden** POST-Endpoints (`api_create_conversion` + `api_ingest_conversion`) — greift nur bei **degeneriertem** Titel.
- **B** Client-seitige Ableitung in `saveMarkdownToLibrary` (markdown-converter-UI) mitziehen.
- **C** Backfill-Script für die ~24 bestehenden falsch betitelten Decks.
- **Titel = erste `#`-Überschrift** (Emphasis-Marker gestrippt), Fallback **erste Nicht-Kommentar-/Nicht-Leer-Zeile**, letzter Fallback `Untitled`.
- **Reuse**: der `derive_title`-Helper nutzt den **fenced-code-aware** `_iter_headings` aus `services/markdown_sections.py` (nach HTML-Kommentar-Strip).
- **Kein Schema-Touch, keine Migration.**

## Verifizierte Code-Fakten (Master-gegroundet)

- **Server speichert Titel wörtlich, leitet nichts ab**: `api_create_conversion` ([app_pkg/library.py:452](app_pkg/library.py)) `title = data.get('title', 'Untitled')[:255]`; `api_ingest_conversion` ([app_pkg/ingest.py:152](app_pkg/ingest.py)) `title = (data.get('title') or 'Untitled')[:255]`.
- **Client leitet aus erster Zeile ab**: [static/js/markdown_converter.js:87-88](static/js/markdown_converter.js) `firstLine = content.split('\n')[0].replace(/^#+\s*/,'').trim(); title = firstLine.substring(0,100) || 'Untitled Markdown'`.
- **Reuse-Punkt**: `_iter_headings(lines)` → `(i, level, text)`, fenced-code-aware ([services/markdown_sections.py:48](services/markdown_sections.py)); `_HEADING_RE`/`_FENCE_RE` ebd.
- **Backfill-Vorlage**: [scripts/backfill_recorded_at.py](scripts/backfill_recorded_at.py) — argparse `--apply`, dry-run-Default mit `db.session.rollback()`, **importiert** die Runtime-Funktion aus `app_pkg`, idempotent (Memory `reference_tag_vocab_central_gate_plus_backfill_script`).
- **Conversion.title** = `String(255)`; `content` = Text (Markdown bei den relevanten Typen).

## Phase 1 — `derive_title` + Server-Endpoints + Client (B) + Tests

1. **`derive_title(markdown_text: str) -> str`** in `services/markdown_sections.py` (co-located, reuse `_iter_headings`), pure:
   - **HTML-Kommentare strippen** zuerst: `re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)` (fängt einzeilige, mehrzeilige **und** mehrere — wichtig, sonst würde ein `#` *innerhalb* eines mehrzeiligen Kommentars als Heading erkannt; `_iter_headings` kennt nur Code-Fences, keine HTML-Kommentare).
   - Dann **erste Überschrift** via `_iter_headings(stripped.split('\n'))` → deren `text`, **umschließende Emphasis-Marker strippen** (`text.strip().strip('*_ ').strip()` → `# *Optimization…*` ergibt „Optimization…").
   - Sonst **erste nicht-leere Zeile** des gestrippten Contents (`.strip()`).
   - Sonst `'Untitled'`. **Nicht** truncaten (Caller macht `[:255]` bzw. `substring(0,100)`).
2. **`_is_degenerate_title(title) -> bool`** (gleiche Datei): `t=(title or '').strip()` → `not t` **oder** `t.lower() in ('untitled','untitled markdown')` **oder** `t.startswith('<!--')`.
3. **Beide Server-Endpoints** verdrahten (Import aus `services.markdown_sections`): den geposteten Titel nehmen, **außer er ist degeneriert** → dann ableiten:
   ```python
   posted = data.get('title')
   title = (derive_title(content) if _is_degenerate_title(posted) else posted)[:255]
   ```
   (Gilt für **alle** `conversion_type` — Trigger ist allein der degenerierte Titel; ein echter Client-Titel bleibt **wörtlich** erhalten.)
4. **Client (B)** [static/js/markdown_converter.js](static/js/markdown_converter.js): kleinen `deriveTitle(content)`-Helper — HTML-Kommentare per Regex strippen, erste `#`-Heading-Zeile (`/^#{1,6}\s+(.+)/m` auf dem gestrippten Text, Emphasis raus), sonst erste nicht-leere Zeile, `substring(0,100)`, Fallback `'Untitled Markdown'`. In `saveMarkdownToLibrary` die alte First-Line-Logik ersetzen.
5. **Tests** (`tests/test_markdown_titles.py` **oder** in `tests/test_markdown_sections.py`): `derive_title` — Heading nach einem Kommentar / nach **mehrzeiligem** Kommentar / nach mehreren Kommentaren; Emphasis gestrippt; kein Heading → erste Nicht-Kommentar-Zeile; **`#` in mehrzeiligem HTML-Kommentar wird NICHT Titel**; **`#` in Code-Fence wird nicht Titel** (reuse-Beleg); leerer Content → `'Untitled'`. **Endpoint-Tests**: `POST /api/conversions` mit degeneriertem Titel (`<!-- … -->`, `''`, `'Untitled'`) → Server leitet aus Content ab; mit **echtem** Titel → wörtlich erhalten; Ingest-Endpoint analog (Token-Scaffolding aus `test_ingest.py`). `pytest` grün ≥ 393.
6. **Verify**: `node --check static/js/markdown_converter.js`. **Live-Smoke** (markdown-converter, lokal): Content `"<!-- Seite 1 -->\n\nGTM\n\n# Echte Überschrift"` einfügen → „In Library speichern" → die Library-Karte trägt **„Echte Überschrift"** (nicht `<!-- Seite 1 -->`, nicht „GTM").

**Stop + Bericht.**

## Phase 2 — Backfill (C) + Wrap

1. **`scripts/backfill_titles.py`** (Muster `backfill_recorded_at.py`): **importiert** `derive_title` + `_is_degenerate_title` aus `services.markdown_sections` (nie reimplementieren). Query: `Conversion`-Rows mit **degeneriertem** Titel (konservativ — primär `title LIKE '<!--%'`, plus leer/`Untitled`/`Untitled Markdown`). Pro Row `neu = derive_title(content)[:255]`; **nur** schreiben, wenn `neu` sich vom alten unterscheidet **und** nicht selbst degeneriert ist. **Dry-run-Default** + `--apply`; Dry-run via `rollback()` (Vorhersage == Apply); idempotent (2. `--apply` = no-op, da Titel dann nicht mehr degeneriert); pro Row `id: alt → neu` printen.
2. **Tests** (`tests/test_backfill_titles.py` oder inline): dry-run schreibt nichts; `--apply` re-titelt eine degenerierte Row (synthetisch, Muster wie der recorded_at-Backfill-Test); eine **echt betitelte** Row bleibt unangetastet; 2. `--apply` = no-op.
3. **Wrap**: `STATUS.md` + `BACKLOG.md` (TITLE-FIX ☑ done mit Hashes; „Aktiv offen"/„Aktueller Sprint" pflegen; **Bullet-Guard**). **Memory** optional (deine Einschätzung — `derive_title` aus erster Heading, HTML-Kommentar + fenced-code-aware, degenerierter-Titel-Trigger, Backfill-Reuse; Treiber war die converter-mcp-`list_conversions`-Findbarkeit). Finaler `pytest`.

**Stop + Schluss-Bericht** — inkl. **Olis Real-Welt-Schritte**:
> 1. **Mintbox-Deploy**: `git pull` + `docker compose up -d --build` (Frontend+Backend-Touch, **keine Migration, kein Dep**), Browser-Hard-Reload.
> 2. **Backfill der Bestands-Decks** auf der Mintbox: `docker compose exec markdown-converter python scripts/backfill_titles.py` (Dry-run, prüfen) → dann `--apply`. (Die ~24 realen Decks liegen in der Prod-DB — der echte `--apply` ist Olis Schritt.)

## Bewusst NICHT (Scope-Grenze)

- **Content unangetastet** — die `<!-- Seite N -->`/`<!-- Grafik: … -->`-Kommentare bleiben (Reader/EPUB brauchen sie).
- **Server leitet nur bei degeneriertem Titel ab** — ein echter, vom Client gesetzter Titel bleibt wörtlich (kein Clobbern).
- **Backfill konservativ** — nur degenerierte Titel, kein Touch an echten.
- **ATX-Headings** (reuse `markdown_sections`); Setext (`===`/`---`) nicht unterstützt (konsistent mit dem Parser).
- **Kein Schema-Touch, keine Migration, kein neuer Endpoint.**

## Akzeptanz

- [ ] `derive_title`: erste `#`-Überschrift (Emphasis gestrippt), **HTML-Kommentar- + fenced-code-aware**, Fallback erste Nicht-Kommentar-Zeile → `Untitled`
- [ ] beide Server-Endpoints leiten bei degeneriertem Titel aus dem Content ab; **echter Titel bleibt wörtlich**
- [ ] markdown-converter-Save leitet genauso ab (Smoke: echte Überschrift wird Kartentitel)
- [ ] Backfill re-titelt die degenerierten Bestands-Decks (dry-run-Default, idempotent, importiert die Runtime-Funktion)
- [ ] `pytest` grün ≥ 393 + neue Tests; `node --check` grün; **Content nie verändert**
