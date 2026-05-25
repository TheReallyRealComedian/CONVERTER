# Sprint R1-B-A — Highlight-Core

**Datum**: 2026-05-25

**Ziel**: Foundation für Reader-Highlights — Schema, Backend-API, Frontend-Selektion-UX, Save, Re-Apply beim Doc-Load. **Kein Note, keine Tags** — die kommen in R1-B-B und R1-B-C. Output: User markiert Text im Reader-View, Highlight wird persistiert und erscheint beim Reload als gelb-tönter Span.

**Vorbedingung**:
- Pytest 71/71 grün auf `main` (zuletzt R1-A done, commit `c84e469`).
- Architektur-Memo [docs/reader_architecture.md](docs/reader_architecture.md) ist Source-of-Truth für Schema-Entscheidungen. **Lesen vor Phase 1.**
- Foundation-Voraussetzungen liefert R1-A:
  - `<article class="reader-view">` rendert Markdown lesbar (kein `<pre>`-Block mehr).
  - `<script type="text/markdown" id="content-source">` enthält den raw Markdown byte-genau (für Future-Features, in R1-B-A nicht zwingend gebraucht — siehe Anker-Mechanik).
  - `script_safe`-Filter mit `Markup`-Return ist live.
- Keine Migration-Framework (Alembic/Flask-Migrate) im Projekt — `db.create_all()` in [app_pkg/__init__.py:76](app_pkg/__init__.py) legt neue Tabellen beim nächsten Container-Start automatisch an. **Neue Tabelle ist trivial, neue Spalte an `Conversion` wäre manuelles ALTER** (deshalb hier nur neue Tabelle).
- Service-Singleton-Pattern aus [CLAUDE.md](CLAUDE.md#Architecture-Notes) und Routing-Pattern (`register(app)`-Function, **kein** Blueprint) — neue Routes folgen demselben Pattern.

**Out-of-scope**:
- **R1-B-B**: `note`-Feld am Highlight + Note-UI. Schema-Spalte wird hier **nicht** vorab angelegt — Note ist `ALTER TABLE Highlight ADD COLUMN note TEXT` in R1-B-B, sauber separiert.
- **R1-B-C**: Tabellen `Tag` + Junction `highlight_tags`, Tag-Picker.
- **R2-A**: `conversion_tags`-Junction + CSV-Migration der existierenden `Conversion.tags`. R1-B-A touched `Conversion`-Model **nur additiv** (eine Relationship-Zeile für `highlights`-Backref).
- **Highlight-Liste in der Sidebar** — gehört zu R1-B-B (Note-Anzeige).
- **Highlight-Edit** über die Erstanlage hinaus (Update von `exact`/`prefix`/`suffix` ist sinnlos, würde Anker zerstören). Nur Delete + Create.

---

## Phase 1 — Implementation

Pre-Flight:

1. `git status -s` → clean tree erwartet.
2. `pytest tests/` → 71/71 grün als Baseline.
3. Mac-Stack live via `docker compose up --build`. Live-Smoke am Ende auf `http://localhost:5656/library/<id>`. Smoke-User `smoke/smokepass123` plus die Test-Docs aus R1-A liegen vor.
4. **READER-Architecture-Memo lesen** ([docs/reader_architecture.md](docs/reader_architecture.md)) — insbesondere Knoten 2 (Anker-Mechanik) und das ER-Diagramm.

### Files

```
models.py                          # EDIT — neue Highlight-Tabelle + Relationship am Conversion
app_pkg/highlights.py              # NEU — register(app) mit 3 API-Routes
app.py                             # EDIT — highlights_module.register(app) registrieren
templates/library_detail.html      # EDIT — schwebender Highlight-Button, optional Highlight-Container
static/js/library_detail.js        # EDIT — initHighlights, mouseup-Handler, save/apply/load
static/css/style.css               # EDIT — .highlight-Span-Styling, .highlight-button-Floater
tests/test_highlights.py           # NEU — POST/GET/DELETE + Ownership-Tests
```

Sieben Files, davon vier Edits an bestehendem Code, drei Neuanlagen (Modul + Tests + CSS-Block).

### Mechanik

**1. Schema-Touch ([models.py](models.py))**:

Neue Klasse `Highlight` nach `Conversion`:

```python
class Highlight(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversion_id = db.Column(db.Integer, db.ForeignKey('conversion.id'), nullable=False, index=True)
    exact = db.Column(db.Text, nullable=False)
    prefix = db.Column(db.Text, default='')
    suffix = db.Column(db.Text, default='')
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            'id': self.id,
            'conversion_id': self.conversion_id,
            'exact': self.exact,
            'prefix': self.prefix,
            'suffix': self.suffix,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
```

Plus am bestehenden `Conversion`-Model eine Backref-Relationship:

```python
highlights = db.relationship('Highlight', backref='conversion',
                             cascade='all, delete-orphan', lazy='dynamic')
```

`cascade='all, delete-orphan'` ist wichtig: wenn ein User eine Conversion löscht, sollen die zugehörigen Highlights automatisch mit.

**2. Backend-Modul ([app_pkg/highlights.py](app_pkg/highlights.py))**:

Folgt dem `register(app)`-Pattern aus den anderen Modulen. Importiert `get_owned_conversion` aus `library.py` für Ownership-Check.

Drei Routes:

| Route | Methode | Auth | Body | Response |
|---|---|---|---|---|
| `/api/conversions/<int:conversion_id>/highlights` | POST | login_required | `{"exact": str, "prefix": str, "suffix": str}` | 201 + Highlight.to_dict() |
| `/api/conversions/<int:conversion_id>/highlights` | GET | login_required | — | 200 + Liste von Highlights, sortiert by `created_at asc` |
| `/api/highlights/<int:highlight_id>` | DELETE | login_required | — | 200 + `{"success": true}` |

Validierung:
- `POST`: `exact` muss non-empty sein und `<= 5000` chars (Schutz gegen Doc-as-Highlight-Missbrauch). `prefix`/`suffix` jeweils `<= 200` chars.
- `POST` + `GET`: `get_owned_conversion(conversion_id)` prüft Ownership.
- `DELETE`: laden via `Highlight.query.get_or_404`, dann prüfen `highlight.conversion.user_id == current_user.id`. Wenn nicht: 404 (nicht 403 — wir wollen Existenz nicht leaken).

CSRF: alle drei Routes sind innerhalb des CSRF-geschützten Bereichs. JS muss den CSRF-Token mitschicken (Pattern aus `library_detail.js` `sendToNotion` — `X-CSRFToken`-Header).

**3. App-Registrierung ([app.py](app.py))**:

Neue Zeile: `from app_pkg import highlights as highlights_module` plus `highlights_module.register(app)` nach `library_module.register(app)`.

**4. Frontend-Selektion-UX**:

**Schwebender Highlight-Button** ([templates/library_detail.html](templates/library_detail.html)):

Ein unsichtbarer Button im Document-Container, der bei aktiver Selection eingeblendet wird. Beispiel-Markup:

```html
<button type="button" id="highlight-create-btn"
        class="highlight-create-btn"
        style="display: none;"
        aria-label="Auswahl markieren">
    Markieren
</button>
```

CSS-Position: `position: absolute`, JS positioniert dynamisch über die Selection-Rect.

**JS-Mechanik** ([static/js/library_detail.js](static/js/library_detail.js)):

Neue Funktionen am Datei-Ende, registriert via DOM-Ready oder direkt am Ende des Scripts:

```js
function initHighlights() {
  loadHighlights();
  attachSelectionListener();
}

function attachSelectionListener() {
  const reader = document.querySelector('.reader-view');
  document.addEventListener('selectionchange', () => positionHighlightButton(reader));
  document.getElementById('highlight-create-btn').addEventListener('click', saveCurrentSelection);
}

function positionHighlightButton(reader) {
  const sel = window.getSelection();
  const btn = document.getElementById('highlight-create-btn');
  if (!sel || sel.isCollapsed || !reader.contains(sel.anchorNode)) {
    btn.style.display = 'none';
    return;
  }
  const rect = sel.getRangeAt(0).getBoundingClientRect();
  btn.style.top = `${window.scrollY + rect.top - 36}px`;
  btn.style.left = `${window.scrollX + rect.left}px`;
  btn.style.display = 'inline-flex';
}

async function saveCurrentSelection() {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed) return;
  const exact = sel.toString();
  const { prefix, suffix } = extractContext(sel);
  sel.removeAllRanges();
  document.getElementById('highlight-create-btn').style.display = 'none';

  const csrf = await getCsrfToken();  // wiederverwendbar aus sendToNotion-Pattern
  const resp = await fetch(`/api/conversions/${window.PageData.conversionId}/highlights`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json', 'X-CSRFToken': csrf},
    body: JSON.stringify({exact, prefix, suffix}),
  });
  if (!resp.ok) { showToast('Markierung speichern fehlgeschlagen.', 'danger'); return; }
  const highlight = await resp.json();
  applyHighlight(highlight);
  showToast('Markiert.', 'success');
}

function extractContext(selection, contextLength = 32) {
  // Anker-Felder kommen aus dem RENDERED HTML-Text, NICHT aus dem raw Markdown
  // (siehe docs/reader_architecture.md Knoten 2).
  // Walk im Reader-View-Text vor und nach der Selection.
  const range = selection.getRangeAt(0);
  const reader = document.querySelector('.reader-view');
  const fullText = reader.innerText;
  const exact = selection.toString();
  // Suche-Position approximieren: pre-Range-Text-Länge messen
  const preRange = document.createRange();
  preRange.setStart(reader, 0);
  preRange.setEnd(range.startContainer, range.startOffset);
  const preLen = preRange.toString().length;
  const prefix = fullText.slice(Math.max(0, preLen - contextLength), preLen);
  const suffix = fullText.slice(preLen + exact.length, preLen + exact.length + contextLength);
  return {prefix, suffix};
}

async function loadHighlights() {
  const resp = await fetch(`/api/conversions/${window.PageData.conversionId}/highlights`);
  if (!resp.ok) return;
  const highlights = await resp.json();
  highlights.forEach(applyHighlight);
}

function applyHighlight(highlight) {
  const reader = document.querySelector('.reader-view');
  // Walk Text-Nodes, finde `exact` mit Prefix/Suffix-Disambiguation,
  // wrappe mit <span class="highlight" data-highlight-id="...">.
  // Implementation darf Sub-Thread frei wählen — Empfehlung: TreeWalker + Range-API.
  // Bei Mehrfach-Match: Prefix-/Suffix-Vergleich, sonst erstes Vorkommen.
}

initHighlights();
```

Sub-Thread soll die `applyHighlight`-Implementation pragmatisch lösen. **Edge-Cases akzeptabel für R1-B-A**:
- Mehrfach-Match mit identischem Prefix/Suffix → erstes Vorkommen.
- `exact`-Text gespannt über mehrere DOM-Elemente (z.B. einen `<strong>`-Block hinweg) → korrekt wrappen ist non-trivial, **erster Sprint-Versuch darf das gracefully ignorieren** (zeigt Toast „Markierung über Formatierungsgrenze nicht unterstützt"). Im Bericht erwähnen, bei häufiger Reibung in Folge-Sprint lösen.

**5. CSS** ([static/css/style.css](static/css/style.css)):

Neuer Block nach Reader-View-Section:

```css
/* HIGHLIGHT */
.highlight {
  background-color: var(--nm-tint-highlight, rgba(255, 220, 0, 0.4));
  padding: 0.1em 0;
  border-radius: 2px;
  cursor: pointer;
}
.highlight-create-btn {
  position: absolute;
  z-index: 100;
  padding: 0.4em 0.8em;
  font-size: 0.8125rem;
  background: var(--nm-bg);
  color: var(--nm-text);
  border: 1px solid var(--nm-text-muted);
  border-radius: var(--nm-radius-sm);
  box-shadow: var(--nm-shadow-floating, 0 4px 12px rgba(0,0,0,0.15));
  cursor: pointer;
}
.highlight-create-btn:hover {
  background: var(--nm-text-muted);
  color: var(--nm-bg);
}
```

`--nm-tint-highlight` neuer Token im `:root` (Light: `rgba(255, 220, 0, 0.4)` Gelb-tönt) plus `[data-global-theme="dark"]` (Dark: `rgba(180, 145, 0, 0.4)` gedimmtes Gelb). Sub-Thread soll im `style.css`-TOC einen neuen Eintrag „HIGHLIGHT" eintragen, falls die TOC-Konvention das verlangt.

Falls `--nm-shadow-floating` nicht existiert: hardcoden mit Fallback wie oben.

**6. Tests ([tests/test_highlights.py](tests/test_highlights.py))**:

Neuer Test-Modul. Pattern aus bestehenden Tests übernehmen (Fixture für User+Conversion, In-Memory-DB). Mindest-Cases:

| Test | Erwartung |
|---|---|
| POST mit valid body → 201, highlight persistiert, response enthält id | grün |
| POST mit leerem exact → 400 | grün |
| POST mit exact > 5000 chars → 400 | grün |
| POST auf fremde Conversion → 404 | grün |
| GET auf eigene Conversion → 200 + Liste mit erstellten Highlights | grün |
| GET auf fremde Conversion → 404 | grün |
| DELETE eigenen Highlight → 200, danach 404 beim GET-Versuch | grün |
| DELETE fremden Highlight → 404 | grün |
| Conversion löschen → zugehörige Highlights mit-gelöscht (cascade-Test) | grün |

Mindestens 8 Tests. `pytest tests/` muss von 71 auf **79+** grün hochgehen.

### Code-Quality-Gates

- UI-Strings deutsch: „Markieren", „Markiert.", „Markierung speichern fehlgeschlagen.", „Markierung über Formatierungsgrenze nicht unterstützt." — alle ohne Emoji, ohne „Error:".
- `showToast` für Banner (nicht `alert()`).
- CSRF-Token via bestehendes Pattern (`X-CSRFToken`-Header) — nicht neu erfinden.
- Helper-Reuse: `get_owned_conversion()` aus `library.py` importieren statt re-implementieren.
- Live-Smoke nach Frontend-Änderung **Pflicht**.
- Pytest 79+/79+ grün vor Phase-Ende.

### Phase-1-Stop

Nach Phase 1: STOP — Bericht. Welche `applyHighlight`-Strategie gewählt (eigener Tree-Walker oder Library), wie Multi-Match-Disambiguation gelöst, ob `--nm-tint-highlight`-Token eingeführt wurde oder rgba-Inline, ob neue Tests grün und welcher Edge-Case (Cross-Format-Selection) im Bericht steht.

---

## Phase 2 — Verify

**Pytest**:

1. `docker compose exec markdown-converter pytest tests/` → 79+/79+ grün (71 baseline + neue Tests).

**Live-Smoke** (Browser auf `http://localhost:5656/library/<id>`, Login mit `smoke/smokepass123`):

2. Reader-View aufrufen für das `markdown_input`-Smoke-Doc (rich content aus R1-A-Smoke).
3. **Selektion + Save**: Text mit Maus markieren → schwebender „Markieren"-Button erscheint nahe der Selection → klicken → Toast „Markiert." → Span-Background wird gelb-tönt.
4. **Page-Reload**: Markierung muss wieder erscheinen, an exakt derselben Stelle.
5. **Mehrere Highlights**: 2-3 verschiedene Stellen markieren, jeder mit Reload-Persistenz.
6. **Cross-Doc-Isolation**: zweites Doc öffnen, Highlights vom ersten dürfen nicht durchschlagen.
7. **Delete**: irgendein UI-Pfad zum Delete (z.B. Right-Click oder Highlight-Click → Bestätigungs-Toast „Markierung löschen?" → ja → entfernt). Sub-Thread entscheidet UX-Pattern für Delete, Empfehlung: Click auf Highlight zeigt einen kleinen Action-Popover.

Wenn Delete-UX schwierig ist und Sub-Thread sich verbeißt: **Fallback** — Delete-Funktion exposen via Browser-Console (`window.deleteHighlight(id)`) plus Sidebar-Liste in R1-B-B als richtigem UX-Pfad. Sub-Thread berichtet die Disposition.

**Edge-Case-Smoke**:

8. Cross-Format-Selection: markiere Text der einen `<strong>`-Block überspannt — Toast „nicht unterstützt" oder graceful Save mit eingeschränktem Re-Apply. Bericht-Item.
9. Mehrfach-Vorkommen: markiere ein Wort, das mehrfach im Dokument vorkommt → Save → Reload → korrekte Stelle highlighted (Prefix/Suffix-Disambiguation).
10. Dark-Mode-Toggle: Highlight im Dark-Mode muss noch sichtbar sein, kein Kontrast-Bruch.

Nach Phase 2: STOP — Bericht. Liste der gesmokten Pfade, Edge-Case-Ergebnisse, Browser-Screenshots optional, Token-Disposition (rgba inline vs. `--nm-tint-highlight`).

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- Ein Commit. Subject-Vorschlag: „R1-B-A: highlight-core — schema, api, selection-ux, save+reapply".
- Body soll erwähnen: Schema (Highlight-Tabelle), die drei API-Routes, Anker-Mechanik (Text-Quote + Prefix/Suffix aus HTML-Text), Edge-Case-Disposition (Cross-Format-Selection, Multi-Match), Test-Count vorher/nachher.
- Branch direkt auf `main`. Push direkt nach Commit.

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute.

**Zusätzlich für R1-B-A**:
- Wenn die `applyHighlight`-Implementation auf eine externe NPM-Library zurückgreifen will (z.B. `mark.js`): **STOP**, Master fragen. CLAUDE.md Code-Stil sagt Vanilla-JS, eine neue Dependency braucht Master-Sign-off.
- Wenn pytest nach dem Schema-Touch rot wird (z.B. weil `db.create_all()` auf bestehender DB einen Highlights-Conflict hat oder cascade-Tests divergieren): **sofort STOP**, Master fragen — Schema-Touches sind kritisch.
- Wenn die Anker-Mechanik in der Praxis (Smoke) brüchig wird (z.B. Selection durch eine Tabellenzeile spannt nichts Sinnvolles): Bericht, **nicht selbst lösen** — eskalation auf M oder Folge-Sprint.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**M** — neue Tabelle + neuer Backend-Modul (3 Routes) + Frontend-Selektion-UX (ca. 150 LOC neuer JS) + CSS-Block + neuer Test-Modul (8+ Tests). Schema-Touch ist gering (eine Tabelle, additive Relationship), Service-Code-Touch null. Wenn Phase 2 Edge-Cases zeigen, die Cross-Format-Selection einen eigenen Lösungs-Loop erzwingen: eskalation auf L und Master entscheidet ob splitten in „R1-B-A-1 Selection-Core" plus „R1-B-A-2 Cross-Format-Resilience".

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim Highlight-Rendering auffällt, dass `.reader-view`-Typografie eine Verbesserung braucht (z.B. `.highlight` interferiert mit `<code>`-Padding): kleiner Style-Tweak im selben Sprint OK.
- Wenn `library_detail.js` Module-Pattern aufzeigt (z.B. einzige IIFE oder einzige Skript-Insel), kann der Sub-Thread eine sanfte Strukturierung erwägen. Aber **kein Refactor** — das ist eigener Sprint.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „R1-B-A ☑ done 2026-05-25 → commit `<hash>` (highlight-core: Tabelle Highlight mit conversion_id-FK plus exact/prefix/suffix-Anker im W3C-Annotation-Style, neuer Modul app_pkg/highlights.py mit 3 API-Routes POST/GET/DELETE, Frontend-Selektion-UX mit schwebendem Markieren-Button und Re-Apply beim Doc-Load, --nm-tint-highlight-Token oder rgba-Inline-Disposition, neuer Test-Modul tests/test_highlights.py mit N Tests). Pytest 79+/79+ grün. Edge-Cases <kurz>. Foundation für R1-B-B Highlight-Notes (Phase-1-Ready, wartet auf Sign-off)."
- **BACKLOG.md**: R1-B-A in Erledigt-Liste; R1-B-A-Bullet aus P1-Inbox raus; R1-B-B als nächstes P1-Item bleibt.
- **Memory**: wenn `applyHighlight`-Pattern eine wiederverwendbare Lehre ergibt (z.B. „TreeWalker + Range API für Text-Wrap funktioniert besser als Markdown-Substitution"): `reference_dom_text_anchoring.md` oder ähnlich. Nichts erzwingen.

---

## Phase-0-Entscheidungen

_(Phase 0 nicht aktiviert — alle Architektur-Knoten im READER-PLAN-Workshop 2026-05-25 geklärt, Entscheidungen persistiert in [docs/reader_architecture.md](docs/reader_architecture.md). Anker-Mechanik = W3C-Style aus HTML-Text. Schema = neue Tabelle. CSRF-Pattern und Helper-Reuse-Pflicht aus CLAUDE.md.)_
