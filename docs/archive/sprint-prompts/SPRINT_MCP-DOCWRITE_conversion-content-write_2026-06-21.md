# Sprint MCP-DOCWRITE — Conversion-Content schreiben (update_document + replace_section) (L, 3 Phasen)

> **Executor-Doc.** Phasen nacheinander, nach jeder **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün (Baseline **364**). Du committest jede Phase selbst (eigener Hash + push). Arbeitsverzeichnis `/Users/olivergluth/CODE/CONVERTER`. Working-Practice in `CLAUDE.md` (Sektion „Working Practice").

## Ziel & Entscheidungen (gesetzt — kein Workshop, nicht neu aufmachen)

Der **converter-mcp** braucht zwei Dokument-Write-Tools (Oli 2026-06-21, „beide", Notion-grade). CONVERTER-Seite = die **token-authentifizierten Endpoints**; das MCP-Tool-Wrapping ist Koordinator-Scope (wie bei den Card-Tools). Entscheidungen (Master):

- **Zwei Endpoints, beide token-authed, CSRF-exempt** (Agent hat keine Session/CSRF), Empfänger… äh, **Ziel-User server-seitig** via Token-Target (`INGEST_USER`), fremde/fehlende Conversion → **404**:
  - **`update_document`** = `PATCH /api/conversions/<id>/content` — Voll-Ersetzung von `content`.
  - **`replace_section`** = `PATCH /api/conversions/<id>/section` — ein per **Heading** adressierter Abschnitt wird ersetzt.
- **Token = der bestehende `CARD_TOKEN`** (der generische Agent-Write-Token; **kein** neuer `.env`-Key). Auth-Gate **wiederverwenden** (s.u.).
- **Kein UI** — das ist eine Agent-API (wie Card-Writes/Highlight-Annotate). Der menschliche Schreib-Pfad (Markdown-Editor, „Im Editor öffnen") existiert schon und bleibt unangetastet.
- **Kein Schema-Touch** (`content`/`updated_at` existieren; `updated_at` bumpt via `onupdate`).

## Verifizierte Code-Fakten (Master-gegroundet — bau darauf)

- **Pfad-Kollision vermieden**: `PUT /api/conversions/<id>` ([app_pkg/library.py:509](app_pkg/library.py), `api_update_conversion`, `@login_required`) macht heute Partial-Writes inkl. `content` — **Session + CSRF-geschützt**, für den MCP unbrauchbar (kein CSRF-Token im HTTP-Client). Deshalb **eigene Sub-Pfade** `/content` + `/section` (kein Method+Path-Clash). Den Session-PUT **nicht** anfassen.
- **Auth-Gate 1:1 wiederverwendbar**: `_authorize_card_write()` ([app_pkg/cards.py:113](app_pkg/cards.py)) ist der generische Agent-Write-Gate (CARD_TOKEN, fail-closed 503, constant-time 401, Ziel-User `INGEST_USER`/first()), schon von Card-Writes **und** Highlight-Annotate genutzt. **Im neuen Modul importieren + neutral aliasen**: `from .cards import _authorize_card_write as _authorize_agent_write` (null Churn an cards.py, sauberer Read am Use-Site, eine Quelle der Wahrheit). **Nicht** reimplementieren, **nicht** cards.py umbenennen (Scope-Schutz).
- **Owner-Scope-Muster** (Token-Target, 404 für fremd): wie die Card-PATCH-View — `Conversion.query.filter_by(id=conversion_id, user_id=target.id).first()`; `None → 404 'Nicht gefunden.'` (kein Existenz-Leak, **nicht** 403/400).
- **Response-Shape**: `Conversion.to_dict()` ([models.py:74](models.py)) enthält `content` (voll) → nach dem Write `jsonify(conversion.to_dict())` zurückgeben (der Agent verifiziert).
- **CSRF-exempt-Registry**: `app.extensions['csrf'].exempt(view)` (Muster cards.py-Ende, drei bestehende Einträge) — die zwei neuen Views ergänzen.
- **Factory** für Tests: `_make_conversion(app, user_id, title=…, ctype=…)` (vgl. [tests/test_cards.py:16](tests/test_cards.py)); Token-Test-Scaffolding (`_card_auth`, `CARD_TOKEN='r4-test-card-token-9b2e'`, `monkeypatch.setenv`) ebenfalls dort.
- **Registry**: [app.py](app.py) ruft `…_module.register(app)` (Zeilen ~63–74) — neues Modul ergänzen (Import + `register`).

## Phase 1 — Section-Parser (`services/markdown_sections.py`) + Tests

**Das ist das Herz des Sprints** (die Endpoint-Schicht in Phase 2 ist Routine). Pure, Flask-frei, gründlich getestet.

`replace_section(markdown_text: str, heading: str, new_section: str) -> str` — gibt das neue Volltext-Markdown zurück; wirft `SectionNotFound` bzw. `SectionAmbiguous` (eigene Exceptions im Modul, der Endpoint mappt sie auf 404/409).

**Algorithmus** (ATX-Headings `#`…`######`; **Setext `===`/`---` ist out-of-scope v1** — im Docstring vermerken):
1. **Ziel normalisieren**: `target = heading.lstrip('#').strip()`.
2. Zeilenweise gehen, **Fenced-Code-State tracken**: eine Zeile, die (ggf. mit Leading-Whitespace) mit ` ``` ` **oder** `~~~` beginnt, toggelt „in Fence". **⚠️ Kern-Korrektheit: `#`-Zeilen INNERHALB eines Fence sind KEINE Headings** (ein `# kommentar` in einem Python-Block darf nicht als Abschnitt erkannt werden). Nur außerhalb von Fences nach Headings suchen.
3. Heading-Zeile = `^(#{1,6})\s+(.+?)\s*#*\s*$` (Closed-ATX-Trailing-`#` mitstrippen) → `level = len(group1)`, `text = group2.strip()`.
4. Alle Heading-Matches mit `text == target` sammeln: **0 → `SectionNotFound`**, **>1 → `SectionAmbiguous`** (konservativ, nie raten — wie das `len==1`-Gate beim Diktat-Parser).
5. Matched Heading bei Zeile `i`, Level `L`. **Section-Ende** = erste Zeile > `i`, die (außerhalb Fence) eine Heading-Zeile mit **Level ≤ L** ist (Subsections gehören also zur Section) — sonst EOF.
6. `lines[i:end]` (Heading **inklusive**) durch `new_section` ersetzen → der Agent liefert die **neue Section inkl. eigener Heading** (kann die Heading also umbenennen; `new_section` muss keine Heading enthalten, wird nicht erzwungen). Sauber rejoinen — **kein Verkleben** von Zeilen, Trailing-Newline-Verhalten in Tests absichern.

**Tests** (`tests/test_markdown_sections.py`) — die Korrektheit lebt hier:
- Single-Match-Replace (Heading+Body ersetzt, Rest unberührt);
- Section **mit Subsections** (`##` unter `#` gehört dazu, bis zum nächsten `#`/gleich-oder-höher);
- Section am **Doc-Ende** (bis EOF);
- **erste** Section / **letzte** Section / mittige;
- **Fenced-Code-Falle**: ein ` ``` `-Block mit einer `# foo`-Zeile wird **nicht** als Heading erkannt (sowohl als Such-Ziel als auch als fälschliche Section-Grenze) — beide Richtungen testen;
- **Ambiguität**: zwei Headings gleichen Texts → `SectionAmbiguous`;
- **Not-Found**: Ziel-Heading fehlt → `SectionNotFound`;
- gleiche Heading-Text auf verschiedenen Levels (z.B. `# Intro` und `### Intro`) → ambiguous (Text-Match level-agnostisch);
- `new_section` ohne Heading (reiner Body) wird sauber eingespleißt;
- Heading mit Trailing-`#` (Closed ATX) matcht.

`pytest tests/` grün ≥ 364. **Stop + Bericht** (inkl. dem Fenced-Code-Beleg).

## Phase 2 — Endpoints (`app_pkg/docwrite.py`) + Register + Tests

Neues Modul `app_pkg/docwrite.py` mit `register(app)`. Beide Views: `target, err = _authorize_agent_write(); if err: return err` → `conv = Conversion.query.filter_by(id=conversion_id, user_id=target.id).first()`; `None → 404`. Body non-dict → 400. **Nicht** CSRF-exempt vergessen (beide Views registrieren).

1. **`update_document`** — `PATCH /api/conversions/<int:conversion_id>/content`:
   - Body `{content: str}`. **`content` Pflicht + non-blank-String** (gegen versehentliches Doc-Wipe durch einen Agent-Bug) → sonst 400 `'Feld content (nicht-leerer Text) erwartet.'`.
   - `conv.content = content`; `db.session.commit()` (updated_at bumpt). → 200 `jsonify(conv.to_dict())`.
2. **`replace_section`** — `PATCH /api/conversions/<int:conversion_id>/section`:
   - Body `{heading: str, content: str}` — beide Pflicht-non-blank-Strings (sonst 400). (`content` = die neue Section.)
   - `try: new_text = replace_section(conv.content, heading, content)`
     - `except SectionNotFound: return 404 'Abschnitt nicht gefunden.'`
     - `except SectionAmbiguous: return 409 'Abschnitt mehrdeutig (mehrere Headings gleichen Texts).'`
   - `conv.content = new_text`; commit. → 200 `jsonify(conv.to_dict())`.
3. **Registrieren**: in [app.py](app.py) Import + `docwrite_module.register(app)` (Reihe ~63–74); die zwei `csrf.exempt(...)` am Modul-Ende (Muster cards.py).

**Tests** (`tests/test_docwrite.py`) — Token-Scaffolding wie test_cards.py:
- **Auth**: fehlt → 503; falsch → 401 (beide Endpoints).
- **Owner**: fremde/fehlende Conversion → 404 (beide).
- **update_document**: Erfolg → 200 + `content` ersetzt (DB-Probe), Response-`to_dict` trägt neuen Content; leer/fehlend/non-str `content` → 400.
- **replace_section**: Erfolg → 200, genau die Ziel-Section ersetzt (DB-Probe, Rest unberührt); Heading fehlt → 404; mehrdeutig → 409; fehlende Felder → 400.
- **CSRF-exempt-Registry**: beide neuen Views in `csrf._exempt_views` (Probe wie `test_only_card_write_views_are_csrf_exempt`).

`pytest tests/` grün. **Stop + Bericht.**

## Phase 3 — Wrap

1. **`docs/card_api_contract.md`** — neue Sektion **„## Document-Content-Writes (Bearer `CARD_TOKEN`)"** (eine Stelle für alle converter-mcp-Write-Kontrakte, die der Koordinator schon liest): die zwei Endpoints, Bodies, Status-Codes (200/404/409/400/503/401), **server-seitiger Ziel-User**, Response = volle Conversion. Kurz notieren: Token = derselbe `CARD_TOKEN`; Tools = `update_document` / `replace_section`.
2. **STATUS.md** + **BACKLOG.md**: MCP-DOCWRITE ☑ done mit Hashes (Muster wie KINDLE); „Aktiv offen"-Block → **READER-ADJ als nächstes (letztes) P1**, dann P2 Web-Article-Save. STATUS „Aktueller Sprint" = MCP-DOCWRITE done, KINDLE → Vorheriger. **Bullet-Guard** (`grep -nE '(- \*\*.*){2,}' BACKLOG.md`, Memory `reference_markdown_bullet_delete_newline`).
3. **Memory** (`reference_*`): ja — **Markdown-Section-Replace** als reusable Faktum: heading-adressiert, **fenced-code-aware** (`#` im Code-Block ≠ Heading), level-aware Boundaries (Section = Heading + Body bis zum nächsten ≤-Level-Heading), **ambiguity-konservativ** (>1 Match → Fehler, nie raten), ATX-only v1. Plus: dritte Agent-Write-Surface über denselben `_authorize_card_write`-Gate (Card / Highlight-Annotate / Doc) — der Gate ist generisch trotz card-y Name. MEMORY.md-Pointer.
4. Finaler `pytest tests/` grün.

**Stop + Schluss-Bericht** — inkl. **Olis/Koordinator-Deploy-Kette**:
> Mac push → Mintbox `git pull` + `docker compose up -d --build` (CONVERTER, **keine Migration**, kein neuer Token — `CARD_TOKEN` steht schon) → dann converter-mcp die zwei Tools `update_document` + `replace_section` gegen die neuen Endpoints bauen/deployen (Koordinator-Scope). End-to-End-Beweis = Koordinator nach Deploy.

## Bewusst NICHT (Scope-Grenze)

- **Kein** neuer Token / `.env`-Key (Reuse `CARD_TOKEN`).
- **Kein** Anfassen des Session-`PUT /api/conversions/<id>` (bleibt der UI-/Editor-Pfad).
- **Kein** UI (Agent-API; der menschliche Editor existiert).
- **Kein** Setext-Heading-Support (`===`/`---`) v1 (ATX only; im Docstring vermerkt).
- **Kein** Multi-Section-/Regex-/Range-Adressierung v1 (genau **eine** per-Heading-Section pro Call; mehrdeutig → 409).
- **Kein** Concurrency-/ETag-Schutz v1 (Single-User; Voll-Ersetzung clobbert bewusst — YAGNI, ggf. späterer Follow-up).
- **Kein** Schema-Touch.

## Akzeptanz

- [ ] `update_document` (`PATCH …/content`) ersetzt `content` voll, token-authed, owner-404, fail-closed 503, leerer Content → 400
- [ ] `replace_section` (`PATCH …/section`) ersetzt genau die per Heading adressierte Section (inkl. Subsections), Rest unberührt
- [ ] Section-Parser ist **fenced-code-aware** (`#` im Code-Block ≠ Heading) und **ambiguity-konservativ** (>1 Match → 409, 0 → 404)
- [ ] beide Endpoints token-authed (CARD_TOKEN, 503/401), CSRF-exempt (Registry-Probe grün), fremde Conversion → 404
- [ ] Response = volle aktualisierte Conversion (`to_dict()`)
- [ ] Session-`PUT` unangetastet; kein Schema-Touch, keine Migration, kein neuer Token
- [ ] `pytest tests/` grün ≥ 364 + neue Tests; `card_api_contract.md` um die Document-Writes ergänzt
