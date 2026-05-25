# Sprint R1-B-B — Highlight-Notes + Sidebar

**Datum**: 2026-05-25

**Ziel**: Zwei zusammengehörige Funktionen, die R1-B-A natürlich vervollständigen — (1) **Note pro Highlight** (Single-Field, optionaler Text), (2) **Sidebar-Liste** mit allen Highlights des Doc. Die Sidebar löst gleichzeitig die R1-B-A-Cross-Format-UX-Lücke: Cross-Format-Highlights (in der DB persistiert aber nicht im DOM gerendert) werden über die Sidebar wieder zugänglich.

**Vorbedingung**:
- Pytest 81/81 grün auf `main` (zuletzt R1-B-A done, commit `12d2c6d`).
- Architektur-Memo [docs/reader_architecture.md](docs/reader_architecture.md) hat Knoten 3 entschieden: Single-`note`-Feld am Highlight.
- Container-DB enthält bereits R1-B-A-Smoke-Artefakte als **gewünschte Smoke-Foundation**:
  - **doc 4** (`audio_transcription` „Sprecher-Dialog HOST/GAST") hält den Cross-Format-Highlight `id=4` (`exact = "HOST: Willkommen "`, DB-row, DOM-unsichtbar). Sidebar muss diesen Eintrag anzeigen + UX für „nicht-im-Text-scrollbar" liefern.
  - **doc 7** „Multi-Anker-Smoke" mit 2 disambiguierten Highlights als Multi-Occurrence-Demo (id=5, id=6).
  - **doc 2** Quartalsbericht mit 2 normalen Highlights.
- Frontend-Foundation aus R1-B-A: `library_detail.js` hat `loadHighlights`, `applyHighlight`, `locateHighlightOffset`, `readerRawText`, Action-Popover-Pattern. Wiederverwendbar.
- **Schema-Migration ist nicht-trivial**: das Projekt hat kein Alembic/Flask-Migrate, `db.create_all()` patcht keine existierende Tabelle. `highlight`-Tabelle wurde in R1-B-A angelegt und hält bereits Daten. Spalten-Add braucht **idempotenten Inline-Migration-Helper** — Mechanik unter Phase 1 spezifiziert.

**Out-of-scope**:
- **R1-B-C** — Highlight-Tags + Tag-Foundation (`Tag` + `highlight_tags`-Junction).
- **R2-A** — `conversion_tags`-Junction + CSV-Migration der existierenden `Conversion.tags`-Spalte.
- **R2-B** — Filtered Views, Reading-Progress.
- **Multi-Node-Wrap für Cross-Format-Highlights** (Cross-Format bleibt im DOM unsichtbar, Sidebar zeigt sie aber an).
- **Sidebar-Sortierung / -Filter** — Sidebar zeigt alle Highlights chronologisch (`created_at asc`), kein Sort-Picker, kein Tag-Filter (kommt in R2).
- **Note-Markdown-Render**: Notes sind plain text, kein Markdown-Render in der Note-Anzeige. YAGNI.

---

## Phase 1 — Implementation

Pre-Flight:

1. `git status -s` → clean tree erwartet.
2. `pytest tests/` → 81/81 grün als Baseline.
3. Mac-Stack live. Smoke-User `smoke/smokepass123`. Test-Docs aus R1-B-A in der DB (doc 4 + doc 7 + doc 2 mit Highlights).
4. **Architektur-Memo Knoten 3** lesen — Single-`note`-Feld am Highlight.

### Files

```
models.py                     # EDIT — Highlight.note + to_dict erweitert
app_pkg/__init__.py           # EDIT — idempotenter ALTER-TABLE-Helper nach create_all
app_pkg/highlights.py         # EDIT — PATCH-Route, Note-Validation
templates/library_detail.html # EDIT — Sidebar-Highlights-Sektion + Action-Popover erweitert um Note-Editor
static/js/library_detail.js   # EDIT — Note-Edit-Logik im Popover, Sidebar-Render bei loadHighlights, Sidebar-Click-Navigation
static/css/style.css          # EDIT — Note-Editor-Block, Sidebar-Highlights-Block
tests/test_highlights.py      # EDIT — 4-5 neue Tests für PATCH + Note-Persistenz
```

Sieben Files, alle Edits (keine Neuanlagen).

### Mechanik

**1. Schema-Touch ([models.py](models.py))**:

Eine Zeile an `Highlight`:

```python
note = db.Column(db.Text, nullable=True)
```

Und `to_dict` erweitern um `'note': self.note`.

**2. Idempotenter Migration-Helper ([app_pkg/__init__.py](app_pkg/__init__.py))**:

Direkt nach `db.create_all()` in `create_app`:

```python
# R1-B-B: Spalten-Migration für bestehende Tabellen, die create_all() nicht patcht.
# Idempotent — checkt erst, dann ALTERt nur wenn nötig.
from sqlalchemy import inspect, text
inspector = inspect(db.engine)
if 'highlight' in inspector.get_table_names():
    cols = {c['name'] for c in inspector.get_columns('highlight')}
    if 'note' not in cols:
        db.session.execute(text('ALTER TABLE highlight ADD COLUMN note TEXT'))
        db.session.commit()
        app.logger.info("R1-B-B: highlight.note column added via ALTER TABLE")
```

Begründung der Inline-Lösung: kein Alembic im Projekt, keine separaten Migration-Files, idempotent gegen wiederholtes Container-Start, Log-Line dokumentiert den Lauf. **Pattern wiederverwendbar** für künftige Spalten-Adds (z.B. R1-B-C falls auch ein Note-ähnliches Feld an einer existierenden Tabelle braucht). Sub-Thread soll erwägen, das in einen kleinen Helper `_run_pending_migrations(app, db)` umzuziehen wenn er saubere Struktur sieht — aber Inline ist auch OK für jetzt.

SQLite-Detail: `ALTER TABLE ADD COLUMN` ohne NOT NULL/DEFAULT funktioniert direkt, kein Table-Rebuild nötig. Existing rows bekommen `NULL` als note — gewünscht.

**3. Backend-PATCH-Route ([app_pkg/highlights.py](app_pkg/highlights.py))**:

Neue Route im `register(app)`-Block:

```
PATCH /api/highlights/<int:highlight_id>
Body: {"note": "..."} oder {"note": null}
Validation:
  - data must be dict
  - 'note' key must exist (Sub-Thread entscheidet: optional oder pflicht — Empfehlung: optional, fehlend = no-op)
  - note value: string oder null
  - note length <= 2000 chars wenn string
  - Ownership: highlight.conversion.user_id == current_user.id, sonst 404
Response: 200 + highlight.to_dict()
```

Konventions-Disposition: optional vs. pflicht für den `note`-Key — Empfehlung **pflicht** (Body muss `note` enthalten), damit der Endpoint atomar bleibt und keine versteckten no-ops triggert. Sub-Thread entscheidet.

**4. Template-Touch ([templates/library_detail.html](templates/library_detail.html))**:

Zwei Anpassungen:

*4a. Sidebar-Highlights-Sektion* — neuer Block in der Sidebar (zwischen Notion-Send und Details, oder am Anfang der Sidebar — Sub-Thread disponiert nach Sidebar-Hierarchie):

```html
<div class="c-surface--flat p-4">
  <h6 class="text-[11px] font-semibold uppercase tracking-wider text-neo-faint mb-3">
    Markierungen <span id="highlight-count" class="text-neo-faint">(0)</span>
  </h6>
  <div id="highlight-list" class="flex flex-col gap-2">
    <p class="text-sm text-neo-faint italic" id="highlight-list-empty">Noch keine Markierungen.</p>
  </div>
</div>
```

Der `#highlight-list`-Container wird von JS bei `loadHighlights` befüllt.

*4b. Action-Popover erweitert um Note-Editor* — der bestehende `#highlight-action-popover` aus R1-B-A bekommt eine Textarea-Sektion:

```html
<div id="highlight-action-popover" ...>
  <textarea id="highlight-note-input"
            class="c-input w-full text-sm"
            rows="3"
            maxlength="2000"
            placeholder="Notiz hinzufügen..."></textarea>
  <div class="flex gap-2 mt-2">
    <button type="button" class="c-btn c-btn--primary text-xs" onclick="saveHighlightNote()">Speichern</button>
    <button type="button" class="c-btn c-btn--danger text-xs" onclick="deleteHighlight()">Löschen</button>
  </div>
</div>
```

Sub-Thread soll bestehende `c-btn`/`c-input`-Klassen wiederverwenden, **nicht** neu erfinden.

**5. JS-Logik ([static/js/library_detail.js](static/js/library_detail.js))**:

Erweiterungen am bestehenden Highlight-Block:

*5a. `loadHighlights`* erweitert: nicht nur DOM-Apply, sondern auch Sidebar-Render. Zentrale Liste `state.highlights = []` (oder direkt im Module-Scope) hält die geladenen Highlights, beide Render-Pfade lesen daraus.

*5b. `renderHighlightList()`* — neue Funktion:
- Iteriert `state.highlights`, baut Cards mit:
  - Snippet (erste 80 Chars von `exact`)
  - Note-Preview wenn vorhanden (erste 60 Chars)
  - Click-Handler → `scrollToHighlight(id)`
- Für **Cross-Format**-Highlights (kein zugehöriger DOM-Span): visueller Marker (kleines Icon oder gedimmtes Styling) + Click triggert Toast „Markierung über Formatierungsgrenze, im Text nicht direkt anspringbar". Sub-Thread entscheidet UX-Detail.
- Empty-State wenn `state.highlights.length === 0`: zeigt `#highlight-list-empty`-Default.

*5c. `scrollToHighlight(id)`*:
- Findet `span.highlight[data-highlight-id="id"]` im Reader-View.
- Wenn gefunden: `element.scrollIntoView({behavior: 'smooth', block: 'center'})` + temporäre `.highlight-flash`-Klasse für 1s (Pulse-Animation als visueller Cue).
- Wenn nicht gefunden (Cross-Format): Toast wie oben.

*5d. Action-Popover-Erweiterung*:
- Beim Open (Click auf Span): Textarea mit `highlight.note || ''` vorbefüllen, dataset für die aktive `highlight_id` setzen.
- `saveHighlightNote()`: PATCH-Call, bei Success → state-Update + Sidebar-Re-Render, Popover schließen, Toast „Notiz gespeichert.".
- `deleteHighlight()` bleibt wie in R1-B-A, plus Sidebar-Re-Render.

*5e. `saveCurrentSelection` (POST)* bleibt unverändert. Aber: nach Success muss die neu erstellte Highlight in `state.highlights` einsortiert und Sidebar re-rendered werden — Sub-Thread integriert das.

**6. CSS ([static/css/style.css](static/css/style.css))**:

Neuer Block nach HIGHLIGHT-Section. Mindest-Set:

```css
/* HIGHLIGHT SIDEBAR */
.highlight-card {
  padding: 0.625rem 0.75rem;
  border-radius: var(--nm-radius-sm);
  background: var(--nm-bg);
  cursor: pointer;
  box-shadow: var(--nm-raised-sm);
  font-size: 0.8125rem;
  line-height: 1.4;
}
.highlight-card:hover { background: var(--nm-tint-accent); }
.highlight-card__snippet {
  color: var(--nm-text);
  margin-bottom: 0.25em;
}
.highlight-card__note {
  color: var(--nm-text-muted);
  font-style: italic;
}
.highlight-card--cross-format {
  opacity: 0.7;
  /* Sub-Thread: ggf. zusätzliches Icon-Marker via ::before */
}

/* NOTE EDITOR im Action-Popover */
.highlight-action-popover__note {
  /* falls strukturierter Spacing nötig */
}

/* HIGHLIGHT FLASH (scroll-to-Cue) */
@keyframes highlight-flash {
  0% { background-color: var(--nm-tint-highlight); }
  50% { background-color: var(--nm-tint-accent); }
  100% { background-color: var(--nm-tint-highlight); }
}
.reader-view span.highlight[data-highlight-id].highlight-flash {
  animation: highlight-flash 1s ease-in-out;
}
```

Sub-Thread soll den TOC-Eintrag erweitern (oder bestehenden HIGHLIGHT-Block sub-strukturieren), Token-Reuse beibehalten.

**7. Tests ([tests/test_highlights.py](tests/test_highlights.py))**:

Neue Tests (auf das R1-B-A-Pattern aufsetzen):

| Test | Erwartung |
|---|---|
| PATCH mit valid note → 200, note persistiert | grün |
| PATCH mit `note: null` → 200, note ist NULL in DB | grün |
| PATCH mit note > 2000 chars → 400 | grün |
| PATCH fremden Highlight → 404 | grün |
| PATCH non-dict body → 400 | grün |
| GET-Liste enthält note-Feld in jedem Highlight-Dict | grün |
| (optional) PATCH ohne `note`-Key im Body → 400 oder no-op (je nach Sub-Thread-Disposition) | grün |

Mindestens 5 neue Tests. Pytest auf **86+** grün.

### Code-Quality-Gates

- UI-Strings deutsch: „Markierungen", „Notiz hinzufügen...", „Notiz gespeichert.", „Markierung über Formatierungsgrenze, im Text nicht direkt anspringbar." — alle ohne Emoji, max 2 Sätze.
- `showToast` für Banner (success/warning, nicht alert()).
- Helper-Reuse: `get_owned_conversion` wie in R1-B-A, `c-btn`/`c-input`-Klassen-Reuse.
- CSRF transparent über base.html-fetch-Wrapper (wie R1-B-A bewiesen).
- Live-Smoke nach Frontend-Änderung Pflicht.
- Pytest 86+/86+ grün vor Phase-Ende.

### Phase-1-Stop

Nach Phase 1: STOP — Bericht. ALTER-TABLE-Migration-Disposition (Inline-Block oder Helper-Function), PATCH-Body-Validierung (note-Key pflicht oder optional), Sidebar-Position in der Sidebar-Hierarchie, Cross-Format-Sidebar-UX-Pattern, Test-Count vorher/nachher.

---

## Phase 2 — Verify

**Pytest** (im rebuild Container):

1. `docker compose exec markdown-converter pytest tests/` → 86+/86+ grün.

**Migration-Idempotenz-Smoke**:

2. Container neu starten (`docker compose restart markdown-converter`) → Log darf nur einmal die R1-B-B-ALTER-Line zeigen, danach nie wieder (Idempotenz verifiziert).

**Live-Smoke** (Browser, Smoke-User, Test-Docs aus R1-B-A):

3. **doc 2 Quartalsbericht** öffnen. Sidebar zeigt 2 Highlight-Cards (Snippets der zwei existierenden Highlights), `(2)`-Counter, keine Notes.
4. **Note hinzufügen**: Click auf einen Highlight im Reader → Popover öffnet → Textarea leer → Text eingeben („Wichtiges Detail.") → Speichern → Toast „Notiz gespeichert." → Sidebar-Card zeigt jetzt Note-Preview.
5. **Note ändern**: zweiter Click auf denselben Highlight → Textarea vorbefüllt mit aktueller Note → ändern → Speichern → Sidebar-Card updated.
6. **Note löschen**: Textarea leeren → Speichern → DB: note=NULL, Sidebar-Card: keine Note-Preview mehr.
7. **Sidebar-Click → Scroll**: zweiten Highlight in der Sidebar anklicken → Reader scrollt smooth zur Stelle, Highlight-Span pulst 1s.
8. **doc 4 Cross-Format-Sidebar**: doc 4 öffnen → Sidebar zeigt den Cross-Format-Highlight als Card (visueller Marker für „nicht-im-DOM"), Click → Toast statt Scroll. **Das ist der Kern-UX-Lücken-Schluss aus R1-B-A**, bitte ausführlich verifizieren.
9. **doc 7 Multi-Anker**: doc 7 öffnen → 2 Sidebar-Cards für die zwei Anker-Highlights, Click führt zur jeweils richtigen Stelle (Disambiguation hält).
10. **Highlight löschen via Popover** (bleibt aus R1-B-A): Click → Popover → Löschen → Span weg, Sidebar-Card weg, Counter auf -1.
11. **Neuer Highlight + Sidebar live**: neuen Text in doc 2 markieren → Save → Sidebar-Card erscheint ohne Reload, Counter auf +1.
12. **Cross-Doc-Isolation**: doc 1 öffnen → Sidebar zeigt 0 Highlights („Noch keine Markierungen."), kein Bleed aus doc 2.
13. **Dark-Mode**: Theme-Toggle → Sidebar-Cards + Note-Editor sehen sauber aus, kein Kontrast-Bruch, Highlight-Flash-Animation läuft auch im Dark-Mode.

**Edge-Case-Smoke**:

14. Note mit 2001 Chars eingeben → Frontend cappt bei 2000 (via `maxlength`) ODER PATCH returnt 400 mit klarer Fehlermeldung — Sub-Thread klärt welcher Pfad genommen wurde.
15. Schnelles Doppel-Click auf Highlight (Race-Condition-Smoke): Popover öffnet einmal sauber, Textarea vorbefüllt korrekt.

Nach Phase 2: STOP — Bericht. Welche der 15 Pfade grün/gelb/rot, Cross-Format-Sidebar-UX-Wirkung subjektive Einschätzung, Performance bei Sidebar-Re-Render bei N=10+ Highlights (bei doc 2/4/7 nicht messbar, aber wenn Sub-Thread Massentest macht: Beobachtung).

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- Ein Commit. Subject z.B. „R1-B-B: highlight-notes + sidebar -- schema-add, PATCH, sidebar-list, scroll-to-highlight".
- Body soll erwähnen: ALTER-TABLE-Migration-Mechanik (idempotenter Inline-Block), PATCH-Route, Sidebar-Liste mit Cross-Format-Pattern (geschlossener UX-Lücken-Punkt), Scroll-to-Highlight mit Flash-Cue.
- Branch direkt auf `main`. Push direkt nach Commit.

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute.

**Zusätzlich für R1-B-B**:
- Wenn der ALTER-TABLE-Helper beim ersten Container-Start nicht idempotent läuft (z.B. zweiter Start schmeißt Exception statt no-op): **sofort STOP**, Master fragen. Schema-Migration-Bug ist deploy-blocking.
- Wenn die Sidebar bei N>20 Highlights merklich langsam re-rendert (Sub-Thread benchmarkt das wenn möglich): nicht im Sprint umstellen, im Bericht erwähnen — Performance-Sprint später wenn nötig.
- Wenn das Cross-Format-Sidebar-UX-Pattern beim Smoke unklar wirkt (Toast vs. Icon vs. gedimmtes Styling): Bericht-Item für Master-Entscheidung am Phase-Ende.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**S–M** — eine Schema-Migration (additive ALTER TABLE), ein PATCH-Endpoint, Sidebar-Komponente neu (~80 LOC JS + CSS-Block), Action-Popover-Erweiterung (~50 LOC JS), 5+ neue Tests. Wenn Sidebar-Performance oder Cross-Format-UX-Pattern signifikant mehr Arbeit ergeben als hier eingeplant: eskalation auf M und Master entscheidet ob splitten (z.B. „R1-B-B-1 Notes-Only" + „R1-B-B-2 Sidebar"). Default ist alles in einem Sprint.

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim Sidebar-Render auffällt, dass die aktuelle Sidebar-Hierarchie eine Polish-Schwäche hat (z.B. „Tags" als Block ist visuell zu schwach abgegrenzt): leichter Style-Tweak im selben Sprint OK, kein Refactor.
- Wenn der idempotente Migration-Helper als wiederverwendbares Pattern aussieht (z.B. R1-B-C wird ihn auch brauchen): kleiner Refactor in eine Helper-Function `_run_pending_migrations(app, db)` ist OK, **aber** dann beide Migration-Steps (R1-B-B note-Add plus ggf. R1-B-C-Vorbereitung) müssen registriert sein. Sub-Thread disponiert.
- Wenn die Note-Anzeige in der Sidebar bei sehr langen Notes (>500 chars) den Card-Layout sprengt: graceful Truncation im JS (z.B. via `_utils.js`-Helper falls dort schon einer existiert) — sonst inline.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „R1-B-B ☑ done 2026-05-25 → commit `<hash>` (Highlight-Notes + Sidebar: Schema-Add `note TEXT` via idempotenten ALTER-TABLE-Inline-Helper in `app_pkg/__init__.py` post-create_all, PATCH `/api/highlights/<id>` mit note-Validierung, Sidebar-Highlights-Liste in `library_detail.html` mit Snippet + Note-Preview + Cross-Format-Marker, Action-Popover erweitert um Textarea + Speichern/Löschen, Scroll-to-Highlight mit Flash-Animation, neue Tests im test_highlights-Modul). Pytest 86+/86+ grün. Cross-Format-UX-Lücke aus R1-B-A geschlossen via Sidebar. 15-Schritte-Smoke abgeschlossen mit Beobachtungen <…>. **Master-Aktivität nächste**: R1-B-C-Sub-Thread starten."
- **BACKLOG.md**: R1-B-B in Erledigt; R1-B-B-Bullet aus P1-Inbox raus; R1-B-C bleibt P1-Item, Smoke-Foundation-Hinweise zu doc 4 / doc 7 / doc 2 ggf. updated falls neue Smoke-Artefakte (Notes) hinzugekommen sind, die R1-B-C wiederverwenden kann.
- **Memory**: wenn der ALTER-TABLE-Inline-Helper als Pattern für künftige Schema-Touches reift: `reference_inline_sqlite_migration.md` — die `inspect(db.engine).get_columns()`-Mechanik plus idempotenter Add-Column-Loop ist im CONVERTER-Stack vermutlich Single-Source-of-Truth-Kandidat. Nichts erzwingen, Sub-Thread-Disposition.

---

## Phase-0-Entscheidungen

_(Phase 0 nicht aktiviert — Architektur-Knoten 3 ist im READER-PLAN-Workshop 2026-05-25 geklärt (Single-`note`-Feld). Migration-Mechanik = idempotenter Inline-ALTER-TABLE-Block, weil kein Migration-Framework im Projekt. Sidebar-Mechanik = chronologische Liste mit Snippet + Note-Preview, Cross-Format-Marker als visueller Pattern. Keine offene Workshop-Frage.)_
