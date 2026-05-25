# Sprint R1-B-C — Highlight-Tags + Tag-Foundation

**Datum**: 2026-05-25

**Ziel**: Tag-System für Highlights — Tag-Tabelle als Per-User-Namespace, `highlight_tags`-Junction, Tag-API, Tag-Picker im Highlight-Edit-Popover, Tag-Chips in der Sidebar-Card, minimale Tag-Manager-Page mit Übersicht + Delete. Letzter R1-B-Sub-Sprint, schließt den R1-Cluster Reader-Core ab. Foundation für R2-A (CSS-Migration der `Conversion.tags`-Spalte in `conversion_tags`-Junction).

**Vorbedingung**:
- Pytest 90/90 grün auf `main` (zuletzt R1-B-B done, commit `5b33f75`).
- Architektur-Memo [docs/reader_architecture.md](docs/reader_architecture.md) Knoten 4 ist Source-of-Truth: `Tag`-Tabelle mit `user_id`-FK + unique(user_id, name), zwei Junction-Tabellen `conversion_tags` + `highlight_tags`. **R1-B-C touched nur `highlight_tags` plus `Tag`-Tabelle**. `conversion_tags`-Junction + CSV-Migration bleibt R2-A vorbehalten.
- Container-DB hält R1-B-A/B-B-Smoke-Artefakte als Tag-Smoke-Foundation:
  - **doc 2 „Quartalsbericht"** mit 4 Highlights (2 normal + 2 cross-format) — perfekt für „mehrere Highlights, mehrere Tags".
  - **doc 7 „Multi-Anker-Smoke"** mit 2 disambiguierten Highlights.
  - **doc 4 „Sprecher-Dialog"** mit 1 cross-format Highlight in Sidebar erreichbar.
- Frontend-Foundation aus R1-B-B: `library_detail.js` hat `highlightsState`, `renderHighlightList`, Action-Popover-Pattern, `getCsrfToken`-Free via base.html-Wrapper.
- **Schema-Migration ist trivial**: beide neuen Tabellen werden via `db.create_all()` automatisch angelegt, **kein `_run_pending_migrations`-Touch nötig** (Memory `reference_inline_sqlite_migration.md` ist nur relevant für Spalten-Adds an bestehenden Tabellen).
- Existierende `Conversion.tags`-CSV-Spalte ([models.py:35](models.py:35)) bleibt in R1-B-C **komplett unangetastet**.

**Out-of-scope**:
- **R2-A** — `conversion_tags`-Junction + CSV-Migration der `Conversion.tags`-Spalte.
- **R2-B** — Filtered Views, Reading-Progress, Tag-basierte Sidebar-Filter in der Library-List.
- **Tag-Rename** — YAGNI. Wenn ein Tag falsch geschrieben wurde, löscht der User ihn und tagged neu. Sprint-Prompt-Default: kein Rename-Endpoint, keine Rename-UI.
- **Tag-Color-Coding** — overengineered, kommt vielleicht in einem Polish-Sprint.
- **Tag-Bulk-Operations** (multi-select + bulk-attach/detach) — eigener Sprint später.
- **Tag-Suche / Tag-Cloud / Tag-Statistik** über Highlight-Count hinaus — R2-B.
- **Tag-Click in der Tag-Manager-Page → Filter-View** — Filtered Views kommen in R2-B; in R1-B-C ist der Tag-Card-Click ein No-Op oder Tooltip „Filter kommt mit R2-B".

---

## Phase 1 — Implementation

Pre-Flight:

1. `git status -s` → clean tree erwartet.
2. `pytest tests/` → 90/90 grün als Baseline.
3. Mac-Stack live. Smoke-User + Test-Docs vorhanden.
4. **Architektur-Memo Knoten 4** lesen — `Tag` + zwei Junction-Tabellen, `user_id`-Scoping.

### Files

```
models.py                     # EDIT — Tag-Klasse + highlight_tags-Junction-Table + Highlight.tags-Relationship
app_pkg/tags.py               # NEU — register(app) mit 4 Routes
app_pkg/highlights.py         # EDIT — to_dict erweitert um tags-Liste
app.py                        # EDIT — tags_module.register(app) registrieren
templates/library_detail.html # EDIT — Tag-Picker im Action-Popover, Tag-Chips in Sidebar-Card via JS-Render
templates/tags.html           # NEU — Tag-Manager-Page mit Liste + Delete
static/js/library_detail.js   # EDIT — Tag-Picker-Logic, Sidebar-Card-Tag-Chips, datalist-Autocomplete
static/js/tags.js             # NEU — Tag-Manager-Page-Logic (Delete-Confirmation, Auto-Refresh)
static/css/style.css          # EDIT — Tag-Chip-Block, Tag-Picker-Block, Tag-Manager-Layout
tests/test_tags.py            # NEU — 10+ Tests für Tag-API
```

Zehn Files: vier Neuanlagen (Backend-Modul + Template + JS + Tests), sechs Edits.

### Mechanik

**1. Schema-Touch ([models.py](models.py))**:

Neue `Tag`-Klasse:

```python
class Tag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    name = db.Column(db.String(80), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    __table_args__ = (
        db.UniqueConstraint('user_id', 'name', name='uq_tag_user_name'),
    )

    def to_dict(self, highlight_count=None):
        out = {'id': self.id, 'name': self.name, 'created_at': self.created_at.isoformat() if self.created_at else None}
        if highlight_count is not None:
            out['highlight_count'] = highlight_count
        return out
```

Junction-Tabelle als `db.Table` (kein extra Felder nötig):

```python
highlight_tags = db.Table(
    'highlight_tags',
    db.Column('highlight_id', db.Integer, db.ForeignKey('highlight.id'), primary_key=True),
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id'), primary_key=True),
)
```

Plus M:N-Relationship am `Highlight`-Model:

```python
tags = db.relationship('Tag', secondary=highlight_tags, lazy='joined', backref='highlights')
```

`lazy='joined'` damit `highlight.to_dict()` die Tags ohne extra Query mitziehen kann.

User-Backref auf `tags` nicht zwingend — die Tag-Liste pro User wird per Query im API gemacht.

**2. Backend-Modul ([app_pkg/tags.py](app_pkg/tags.py))**:

Folgt dem `register(app)`-Pattern. Vier Routes:

| Route | Methode | Body | Response |
|---|---|---|---|
| `/api/tags` | GET | — | 200 + Liste aller Tags des current_user mit `highlight_count`, sortiert by name |
| `/api/highlights/<int:highlight_id>/tags` | POST | `{"name": str}` | 201 + Tag.to_dict() — find-or-create + Attach |
| `/api/highlights/<int:highlight_id>/tags/<int:tag_id>` | DELETE | — | 200 + `{"success": true}` — Detach (Tag bleibt) |
| `/api/tags/<int:tag_id>` | DELETE | — | 200 + `{"success": true}` — Tag komplett löschen + alle Junctions (cascade) |

Validierung:
- Tag-Name: non-empty, **trim + lowercase** beim Empfang (Master-Disposition für Disambiguation — „KI" und „ki" und „KI " sind dasselbe), max 80 chars.
- find-or-create: bevor neue Tag-Row, query nach `(user_id=current_user.id, name=normalized_name)`. Wenn existiert: reuse, sonst create.
- Ownership: alle Routes via User-Scoping. Tag muss `tag.user_id == current_user.id` haben, Highlight muss `highlight.conversion.user_id == current_user.id` haben.
- Attach-Idempotenz: wenn Tag bereits am Highlight hängt, **no-op** mit 200 statt 409 (User-Friendly, kein Double-Add-Crash bei schnellen Clicks).
- Tag komplett-DELETE: cascade auf `highlight_tags`-Junction via `Tag.highlights`-Backref iterativ leeren, dann `db.session.delete(tag)`. Sub-Thread disponiert wenn SQLAlchemy-Cascade-Option sauberer ist (z.B. `db.relationship(..., cascade='all, delete-orphan')` am `tags`-Relationship — funktioniert mit `secondary` Junction).

**Tag-Normalisierung-Memory-Hinweis**: lowercase+trim ist eine Design-Wahl, **nicht eine technische Pflicht**. Sub-Thread soll im Sprint-Bericht erwähnen ob die Smoke-Wirkung passt. Alternative wäre case-preserving (Tag „KI" und „ki" sind zwei Tags). Bei Reader-Replacement-Workflow ist case-insensitive-Disambiguation pragmatischer (Tippfehler-Resilienz), aber wenn Oliver später anders denkt: Folge-Sprint-Item.

**3. Highlight.to_dict-Erweiterung ([app_pkg/highlights.py](app_pkg/highlights.py))**:

Aktuell returnt `to_dict()` `{id, conversion_id, exact, prefix, suffix, note, created_at}`. Erweitern um:

```python
'tags': [t.to_dict() for t in self.tags]
```

Damit `GET /api/conversions/<id>/highlights` direkt die Tag-Liste pro Highlight mitliefert — kein N+1-Roundtrip.

**4. App-Registrierung ([app.py](app.py))**:

Neue Zeile: `from app_pkg import tags as tags_module` plus `tags_module.register(app)` nach `highlights_module.register(app)`.

**5. Template-Touch ([templates/library_detail.html](templates/library_detail.html))**:

*5a. Action-Popover erweitert um Tag-Sektion*:

```html
<!-- nach der Note-Textarea + Buttons-Reihe, vor closing div -->
<hr class="my-2 border-nm-text-muted opacity-40">
<div class="highlight-tag-section">
  <label class="text-[11px] uppercase tracking-wider text-nm-text-muted">Tags</label>
  <div id="highlight-tag-chips" class="flex flex-wrap gap-1 mt-1"></div>
  <div class="flex gap-1 mt-2">
    <input type="text" id="highlight-tag-input" list="tag-suggestions"
           placeholder="Tag hinzufügen..." class="c-input text-sm flex-1"
           autocomplete="off" maxlength="80">
    <button type="button" class="c-btn text-xs" onclick="addTagToHighlight()">+</button>
  </div>
  <datalist id="tag-suggestions"></datalist>
</div>
```

Browser-native `<datalist>` als Autocomplete-Quelle, befüllt aus `GET /api/tags`.

*5b. Sidebar-Card Tag-Chips-Slot*: bestehende Sidebar-Card-Render-Funktion bekommt einen `<div class="highlight-card__tags">` per JS injiziert. Max 3 Chips sichtbar, „+N" wenn mehr. Layout: unter dem Note-Preview, vor dem Card-Border.

**6. Tag-Manager-Page ([templates/tags.html](templates/tags.html))**:

Neue Page, gerendert via neue Route `/tags` (in `app_pkg/tags.py`-Modul registriert):

```html
{% extends "base.html" %}
{% block title %}Tags – Library{% endblock %}
{% block content %}
<div class="flex-1 p-6 lg:p-8 overflow-auto">
  <div class="max-w-4xl mx-auto">
    <a href="{{ url_for('library') }}" class="text-sm text-nm-text-muted hover:text-nm-text no-underline transition-colors">&larr; Zurück zur Library</a>
    <h1 class="text-2xl font-semibold mt-2 mb-6">Tags</h1>
    <div id="tag-list" class="grid gap-2"></div>
    <p id="tag-list-empty" class="text-nm-text-muted italic hidden">Noch keine Tags. In der Library-Detailansicht einen Highlight markieren und Tags hinzufügen.</p>
  </div>
</div>
{% endblock %}
{% block scripts %}
<script src="{{ url_for('static', filename='js/tags.js') }}"></script>
{% endblock %}
```

Plus ein Link in `library.html` (Library-List-Header) zu `/tags` — „Tags verwalten" als knapper Sekundär-Link rechts oben oder neben dem Sort-Picker. Sub-Thread positioniert nach Sidebar-Layout.

**7. JS — Tag-Logic in library_detail.js + tags.js**:

*7a. `static/js/library_detail.js` Erweiterung*:

Neue Funktionen:
- `loadTagSuggestions()` — fetch `GET /api/tags`, befüllt `<datalist id="tag-suggestions">` mit `<option value="name">`.
- `addTagToHighlight()` — liest Input, POST an `/api/highlights/<id>/tags`, on success: state-Update für den aktiven Highlight (push tag in `highlight.tags`), re-render Popover-Chips + Sidebar-Card.
- `removeTagFromHighlight(tagId)` — DELETE-Call, state-Update, re-render.
- `renderHighlightTagChips(highlight)` — baut Chips für Popover + Sidebar-Card. Sidebar-Card-Chips sind kompakter (kein X-Button per Click in der Sidebar — Detach läuft nur über Popover).
- Popover-Open: lädt aktuelle Tags + Suggestions-Datalist.

*7b. `static/js/tags.js` — Tag-Manager-Page*:

- `loadTags()` — fetch `GET /api/tags`, render Liste mit Tag-Name + highlight_count + Delete-Button.
- `deleteTag(tagId)` — Confirmation-Dialog (`confirm()` native ist OK für R1-B-C, kein Modal-Refactor), DELETE-Call, reload-List bei success.

**8. CSS ([static/css/style.css](static/css/style.css))**:

Neue Blöcke (TOC-Eintrag aktualisieren):

```css
/* TAG CHIP (Highlight-Tags in Popover + Sidebar) */
.tag-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.25em;
  padding: 0.15em 0.5em;
  border-radius: 999px;
  background: var(--nm-tint-accent);
  color: var(--nm-text);
  font-size: 0.75rem;
  font-weight: 500;
}
.tag-chip__remove {
  border: none;
  background: transparent;
  color: var(--nm-text-muted);
  cursor: pointer;
  font-size: 1em;
  line-height: 1;
  padding: 0;
}
.tag-chip__remove:hover { color: var(--nm-text); }
.tag-chip--compact {
  /* sidebar-card version: smaller, no remove button */
  font-size: 0.6875rem;
  padding: 0.1em 0.4em;
}

/* TAG MANAGER PAGE */
.tag-manager-card {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 1rem;
  border-radius: var(--nm-radius-sm);
  background: var(--nm-bg);
  box-shadow: var(--nm-raised-sm);
}
.tag-manager-card__count {
  color: var(--nm-text-muted);
  font-size: 0.875rem;
}
```

Sub-Thread soll Token-Reuse beibehalten und im TOC „TAG CHIP" + „TAG MANAGER" als Sektionen einsortieren.

**9. Tests ([tests/test_tags.py](tests/test_tags.py))**:

Neuer Test-Modul. Fixture-Pattern aus `test_highlights.py` übernehmen. Mindest-Cases:

| Test | Erwartung |
|---|---|
| GET /api/tags empty → 200 + [] | grün |
| GET /api/tags nach Seed → 200 + sortierte Liste mit highlight_count | grün |
| POST tag to highlight (new) → 201, Tag erstellt, attached | grün |
| POST tag to highlight (existing, anderer Highlight) → 201, Tag reused, attached | grün |
| POST tag to highlight (already attached) → 200 no-op, kein 409 | grün |
| POST tag-name normalisiert (lowercase+trim) | grün |
| POST empty/whitespace-only/oversized name → 400 | grün |
| POST foreign highlight → 404 | grün |
| DELETE /api/highlights/<id>/tags/<tag_id> → 200, Junction weg, Tag bleibt | grün |
| DELETE detach foreign-user-Tag → 404 | grün |
| DELETE /api/tags/<id> komplett → 200, Tag weg, alle Junctions weg | grün |
| DELETE foreign Tag → 404 | grün |
| User-Scoping: User A erstellt Tag „KI", User B sieht ihn nicht in GET /api/tags | grün |

Mindestens 10 Tests. Pytest auf **100+** grün.

### Code-Quality-Gates

- UI-Strings deutsch: „Tag hinzufügen...", „Tags", „Tags verwalten", „Tag löschen?", „Noch keine Tags.", „Tag entfernt." — alle max 2 Sätze, keine Emoji.
- `showToast` für Banner.
- CSRF transparent via base.html-fetch-Wrapper.
- Helper-Reuse: `get_owned_conversion` für Highlight-Ownership-Check via highlight.conversion-Backref.
- Tag-Chips im Sidebar nur Read-Only (kein Detach-X-Button), Detach läuft über den Popover.
- `lazy='joined'` an Highlight.tags damit GET-Highlights die Tags atomic mitliefert.
- Live-Smoke nach Frontend-Änderung Pflicht.
- Pytest 100+/100+ grün vor Phase-Ende.

### Phase-1-Stop

Nach Phase 1: STOP — Bericht. Tag-Name-Normalisierung (lowercase+trim akzeptiert, oder case-preserving?), Attach-Idempotenz-Pattern (no-op 200 ist OK?), Tag-Manager-Page-Position des Library-Links, CSS-Token-Wahl für Tag-Chip-Background (--nm-tint-accent vs. eigener Token), Test-Count vorher/nachher.

---

## Phase 2 — Verify

**Pytest**:

1. `docker compose exec markdown-converter pytest tests/` → 100+/100+ grün.

**Live-Smoke** (Browser, Smoke-User, Test-Docs aus R1-B-A/B-B):

2. **doc 2 öffnen** → Sidebar-Cards mit 4 Highlights, alle ohne Tags.
3. **Erster Tag**: Highlight #1 anklicken → Popover öffnet → Tag-Input „KI" tippen → + → Toast „Tag hinzugefügt." → Chip erscheint in Popover + Sidebar-Card.
4. **Datalist-Autocomplete**: Highlight #2 anklicken → Tag-Input „k" tippen → Browser zeigt „ki"-Suggestion aus datalist.
5. **Existing-Tag wiederverwenden**: Highlight #2 Tag-Input „KI" + → Toast „Tag hinzugefügt." → derselbe Tag (ID identisch) attached an Highlight #2 + Sidebar-Card.
6. **Normalisierung**: Highlight #3 Tag-Input „  Produkt  " (mit Whitespace) → wird zu „produkt" → Chip „produkt".
7. **Idempotenz**: Highlight #1 Tag-Input „KI" nochmal + → 200 no-op, kein Duplikat, kein 409, kein Toast (oder neutraler Hinweis „Tag bereits vorhanden").
8. **Detach via Popover**: Highlight #1 Tag-Chip „KI" X-Click → DELETE → Chip weg in Popover + Sidebar-Card.
9. **Tag-Manager-Page**: Library-List öffnen → Link „Tags verwalten" → /tags → Liste zeigt „ki" (2 Highlights) + „produkt" (1 Highlight).
10. **Tag-Komplett-Löschen**: /tags → Delete-Button bei „ki" → Confirmation → Confirm → Tag verschwindet aus Liste, „produkt" bleibt. Zurück zu doc 2 → die zwei „KI"-Chips sind weg an allen Highlights, „produkt" Chip an Highlight #3 bleibt.
11. **Cross-Doc-Isolation**: doc 4 öffnen → Sidebar zeigt den cross-format-Highlight ohne Tags. Tag „test" anhängen → in doc 4 sichtbar, in doc 2 nicht.
12. **User-Scoping**: ein zweiter User wird via Flask-CLI angelegt (`create-user`), eingeloggt → GET /api/tags muss leer sein. (Optional, wenn Zweit-User-Setup zu aufwendig: Test mit Pytest-Coverage ist genug.)
13. **Sidebar-Tag-Chips**: max 3 sichtbar, „+N" bei mehr — Test mit Highlight, das 5 Tags hat.
14. **Dark-Mode**: Theme-Toggle → Tag-Chips lesbar, Tag-Manager-Page sauber, kein Kontrast-Bruch.

**Edge-Case-Smoke**:

15. Tag-Name mit Sonderzeichen („ich/du", „ai+ml", „2026") → akzeptiert, kein Encoding-Bug.
16. Tag-Name mit Emoji („🔥KI") → akzeptiert oder graceful normalisiert, kein Crash.
17. Sehr lange Tag-Name (80 chars) → akzeptiert, 81 chars → 400.
18. Tag-Picker bei cross-format-Highlight in doc 4: funktioniert ohne Sonderbehandlung (Highlight ist in DB, Tag-Operation läuft normal).

Nach Phase 2: STOP — Bericht. Tag-Normalisierung-Wirkung (Tippfehler-Resilienz vs. Case-Verlust), Tag-Manager-Page-UX, Smoke-Pfade Übersicht.

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- Ein Commit. Subject z.B. „R1-B-C: highlight-tags + tag-foundation -- Tag-Table, highlight_tags-Junction, Tag-API, Tag-Picker, Tag-Manager-Page".
- Body soll erwähnen: Schema (zwei neue Tabellen, kein Migration-Helper-Touch), 4 API-Routes, Tag-Normalisierung lowercase+trim, find-or-create-Pattern, Tag-Chips in Sidebar-Cards (Read-Only) plus Popover (Edit), Tag-Manager-Page, R1-Cluster-Abschluss.
- Branch direkt auf `main`. Push direkt nach Commit.

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute.

**Zusätzlich für R1-B-C**:
- Wenn beim Tag-Detach-Pattern (DELETE /api/highlights/<id>/tags/<tag_id>) die SQLAlchemy-Mechanik mit `secondary=`-Junctions Reibung erzeugt (z.B. nicht direkt `highlight.tags.remove(tag)` funktioniert): **STOP**, Master fragen. SQLAlchemy-M:N-Patterns können je nach Version subtil sein.
- Wenn Tag-Manager-Page-Link in der Library-List-Header layout-bruch auslöst (Sidebar oder Sort-Picker): nicht aggressiv refactorn, Bericht-Item.
- Wenn die Tag-Normalisierung lowercase+trim sich beim Smoke „falsch" anfühlt (z.B. „KI" wirkt natürlicher als „ki" in der Sidebar): Bericht-Item, Master entscheidet ob auf case-preserving wechseln. **Pre-Commit-Patch erlaubt** wenn Sub-Thread sicher ist und Tests entsprechend angepasst werden.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**M** — zwei neue Tabellen (trivial via create_all), 4 Backend-Routes + 1 Page-Route, Tag-Picker im Popover, Tag-Chips in Sidebar-Card, Tag-Manager-Page komplett neu, 10+ Tests. Größter Risiko-Punkt: SQLAlchemy-`secondary`-Relationship-Bedienung beim Attach/Detach, falls die `lazy='joined'`-Option Reibung mit dem to_dict-Render erzeugt. Wenn das eskaliert: M → L, Master entscheidet ob splitten (z.B. „R1-B-C-1 Tag-Schema + Highlight-Tag-API" + „R1-B-C-2 Tag-Manager-Page").

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim Sidebar-Tag-Chip-Render auffällt, dass `renderHighlightList` Performance-Verbesserung lohnt (z.B. DocumentFragment statt direkter appendChild für N Tag-Chips × N Cards): kleiner Optimization-Touch im selben Sprint OK.
- Wenn die existierende `Conversion.tags`-CSV-Spalte beim Schemen-Walk auffällt: **nicht touchen** — das ist R2-A. Bericht-Item OK.
- Wenn `library.py` einen Link zu `/tags` braucht und der Header dort gerade unstrukturiert ist: leichter Style-Tweak OK, kein Refactor.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „R1-B-C ☑ done 2026-05-25 → commit `<hash>` (Tag-Foundation: Tag-Tabelle per-User-Namespace mit unique(user_id, name), highlight_tags-Junction-Table, 4 API-Routes GET-tags / POST-attach / DELETE-detach / DELETE-tag-complete, Tag-Picker im Action-Popover mit datalist-Autocomplete, Tag-Chips in Sidebar-Cards Read-Only, Tag-Manager-Page /tags mit Liste + Delete, Tag-Normalisierung lowercase+trim für Tippfehler-Resilienz, find-or-create-Pattern beim Attach, neuer Test-Modul tests/test_tags.py mit N Tests). Pytest 100+/100+ grün. R1-Cluster Reader-Core damit komplett: R1-A Reading-View + R1-B-A Highlight-Core + R1-B-B Notes+Sidebar + R1-B-C Tag-Foundation. **Master-Aktivität nächste**: R2-Cluster Library-Power planen (R2-A conversion_tags-Junction + CSV-Migration, R2-B Filtered Views + Reading-Progress)."
- **BACKLOG.md**: R1-B-C in Erledigt-Liste; R1-B-C-Bullet aus P1-Inbox raus; R1-Cluster-Header-Text auf „R1-Cluster Reader-Core abgeschlossen 2026-05-25" updated. **Neue R2-Cluster-Sektion** anlegen (R2-A + R2-B als P1-Items) — Spec analog R1-B aus Architektur-Memo. Oder: Master macht R2-Planung im nächsten Burst, Sub-Thread macht nur R1-B-C-Closing. **Default**: Sub-Thread schließt R1-B-C, eröffnet R2-Sektion mit Platzhalter-Bullets (R2-A + R2-B knapp), Master verfeinert.
- **Memory**: wenn die Tag-Normalisierung (lowercase+trim) als wiederverwendbares Pattern reift (z.B. R2-A wird die `Conversion.tags`-CSV ebenfalls so normalisieren müssen für Konsistenz): `reference_tag_normalization.md` mit Begründung. Nichts erzwingen.

---

## Phase-0-Entscheidungen

_(Phase 0 nicht aktiviert — Knoten 4 im READER-PLAN-Workshop 2026-05-25 geklärt (Tag-Tabelle + zwei Junction-Tabellen, Per-User-Namespace, beidseitige FK-Integrität). Schema-Migration trivial (zwei neue Tabellen via create_all). Tag-Normalisierung-Wahl lowercase+trim ist Master-Vorgabe, Sub-Thread soll im Smoke die Wirkung beurteilen. Frontend-UX (Tag-Picker mit datalist + Sidebar-Card-Chips + Manager-Page) ist mechanisch klar.)_
