# Sprint R1-A — Reading-View Foundation

**Datum**: 2026-05-25

**Ziel**: Den `<pre>`-Monospace-Block in `library_detail.html` durch eine **server-side gerenderte Markdown-Reading-View** mit lesefreundlicher Typografie ersetzen. Foundation für R1-B (Highlight-Selektion) — auf einem `<pre>`-Block ist Highlight-UX katastrophal, deshalb kommt R1-A zuerst. Kein Schema-Touch, kein Frontend-State-Bruch.

**Vorbedingung**:
- Pytest 71/71 grün auf `main` (zuletzt nach MAC1, commit `2c9987a`).
- Bestehende `MarkdownIt`-Konfiguration in [app_pkg/markdown.py:97-100](app_pkg/markdown.py:97) (mit `breaks`, `html`, `highlight_code`-Callback via pygments) plus `nh3.clean`-Allow-List in [app_pkg/markdown.py:161-183](app_pkg/markdown.py:161) sind die Quelle der Wahrheit für Markdown-Rendering im Projekt.
- Renderziel: `library_detail.html` Zeile 39-40 (der `<pre>`-Block mit Klasse `detail-content-text`).
- `library_detail.js` liest an **4 Stellen** den Raw-Content via `document.querySelector('.detail-content-text').textContent` (Zeilen 108, 114, 124, 348) — Copy, Download, Notion-Send, ein weiterer Use-Case. Diese Lese-Pfade müssen **weiterhin den raw Markdown-Quelltext bekommen**, nicht den HTML-gestrippten Render.
- Conversion-Types laut [app_pkg/library.py:11-16](app_pkg/library.py:11): `document_to_markdown`, `audio_transcription`, `dialogue_formatting`, `markdown_input`. Alle vier werden im selben Detail-Template angezeigt und alle vier sollen durch den Reading-View-Renderer laufen.
- Olivers Entscheidung 2026-05-25: `<pre>`-Block **komplett ersetzen** (kein Toggle Read/Source), **Backend-Render** (markdown-it + nh3 wiederverwenden).

**Out-of-scope**:
- **Highlight-Selektion** — R1-B, separater Sprint nach R1-A.
- **Schema-Touches** an `Conversion`-Model oder neue Tabellen — gehört zum READER-PLAN-Workshop, nicht hier.
- **Reading-Settings** (Schriftgröße, Theme, Dark/Light-Mode) — R6 Polish-Cluster.
- **Reading-Progress-Tracking** — R2-B.
- **Read/Source-Toggle** — explizit verworfen (Oliver: „komplett ersetzen").
- **`markdown_converter.html`** Touches — die Markdown→PDF-Pipeline bleibt unverändert. R1-A betrifft nur den Library-Detail-Pfad.

---

## Phase 1 — Implementation

Pre-Flight:

1. `git status -s` → clean tree erwartet.
2. `pytest tests/` → 71/71 grün als Baseline.
3. Mac-Stack läuft via `docker compose up --build` (MAC1 ist live). Live-Smoke am Ende auf `http://localhost:5656/library/<id>`.

**Files**:

```
app_pkg/markdown_render.py    # NEU — DRY-Helper, Markdown→HTML+sanitization wiederverwendbar
app_pkg/markdown.py           # EDIT — auf den neuen Helper umstellen, keine Duplikation
app_pkg/library.py            # EDIT — library_detail-Route ruft den Helper auf, rendert content_html
templates/library_detail.html # EDIT — <pre>-Block ersetzt durch gerendertes HTML + hidden raw-source
static/css/style.css          # EDIT — Reader-Layout-Block (.reader-view, Typografie)
static/js/library_detail.js   # EDIT — die 4 raw-content-Reads umstellen auf die neue Source
```

Sechs Files, vier davon Edits an bestehendem Code, zwei Neuanlagen (Modul + CSS-Block).

### Mechanik

**1. Helper-Modul `app_pkg/markdown_render.py`**:

- Exportiert eine Funktion `render_markdown_to_html(markdown_text: str) -> str`.
- Erstellt eine Modul-Level-`MarkdownIt`-Instanz mit denselben Optionen wie `markdown.py:97-100` (`breaks: True, html: True, highlight: highlight_code`).
- `highlight_code` wird ebenfalls hier definiert (oder aus `markdown.py` umgezogen — Sub-Thread entscheidet, je nachdem ob `markdown.py` die Funktion sonst noch braucht). Modul-Header sagt: „Single source of truth für Markdown→HTML im Projekt."
- Nach dem MarkdownIt-Render läuft `nh3.clean(...)` mit derselben Tags+Attributes-Allow-List wie `markdown.py:161-183`. Allow-List ebenfalls ins Helper-Modul ziehen, beide Call-Sites darauf umstellen.
- Empty/None-Content: Funktion gibt leeren String zurück (nicht `<p></p>` oder ähnliches). Template-Side prüft auf leer und zeigt Empty-State.

**2. `app_pkg/markdown.py` Refactor**:

- `md`-Instanz, `highlight_code`, `nh3.clean`-Call: auf `render_markdown_to_html`-Helper umstellen.
- `_wrap_wide_tables` bleibt wo es ist (ist Markdown-Converter-spezifisch für Landscape-PDF).
- Convert-Markdown-Pfad muss weiterhin identisches Verhalten haben — pytest soll grün bleiben.

**3. `app_pkg/library.py` Route-Touch**:

- `library_detail`-Route ([library.py:100-105](app_pkg/library.py:100)) ruft `render_markdown_to_html(conversion.content)` und übergibt das Resultat als `content_html` an `render_template`.
- Keine Filter auf `conversion_type` — alle vier Types laufen durch denselben Renderer. Plain-Text (z.B. Deepgram-Transkript) wird zu `<p>`-Blöcken, das ist akzeptiert.
- API-Routes `api_create_conversion`, `api_update_conversion`, `api_delete_conversion` bleiben **unangetastet** — die geben weiterhin `to_dict()` mit `content` raw zurück, kein HTML.

**4. `templates/library_detail.html` Template-Touch (Zeilen 39-41)**:

Vor:
```html
<div class="c-surface overflow-auto max-h-[calc(100vh-280px)]" id="content-body">
    <pre class="detail-content-text p-5 m-0 font-mono text-sm leading-relaxed whitespace-pre-wrap break-words text-neo-muted">{{ conversion.content }}</pre>
</div>
```

Nach (Empfehlung, Sub-Thread darf strukturell leicht abweichen wenn begründet):
```html
<div class="c-surface overflow-auto max-h-[calc(100vh-280px)]" id="content-body">
    <article class="reader-view p-6">
        {% if content_html %}{{ content_html | safe }}{% else %}<p class="text-neo-faint italic">Kein Inhalt.</p>{% endif %}
    </article>
    <script type="text/markdown" id="content-source">{{ conversion.content }}</script>
</div>
```

**Hidden raw-source Pattern** (das ist der kritische Punkt):
- `<script type="text/markdown">` wird vom Browser **nicht ausgeführt** (unbekannter MIME-Type) und **nicht gerendert**.
- `.textContent` liefert den raw Markdown zurück — exakt was die 4 JS-Lese-Stellen brauchen.
- **Encoding-Hazard**: wenn `conversion.content` einen `</script>`-String enthält, bricht der HTML-Parser. Lösung: ein kleiner Jinja2-Filter oder Inline-Replace, der `</script>` zu `<\/script>` macht. Sub-Thread implementiert das defensiv (Filter-Variante bevorzugt — registrierbar in `app_pkg/__init__.py:_register_template_filters`). Test: ein Doc mit String `</script>` im Content speichern und Detail-View aufrufen → darf nicht brechen.

**5. `static/css/style.css` Reader-Layout-Block**:

Neuer Block am Ende der CSS (oder im passenden TOC-Abschnitt, falls TOC-Konvention existiert — kurz Header von `style.css` prüfen für Konvention). Mindest-Set:

```css
.reader-view {
  max-width: 70ch;       /* Lesezeilenlänge */
  margin: 0 auto;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  font-size: 1.0625rem;   /* leicht > 16px */
  line-height: 1.65;
  color: var(--neo-text);
}
.reader-view h1, .reader-view h2, .reader-view h3, .reader-view h4 { line-height: 1.25; margin-top: 1.5em; margin-bottom: 0.5em; }
.reader-view h1 { font-size: 1.875rem; }
.reader-view h2 { font-size: 1.5rem; }
.reader-view h3 { font-size: 1.25rem; }
.reader-view p { margin: 0 0 1em; }
.reader-view ul, .reader-view ol { padding-left: 1.5em; margin: 0 0 1em; }
.reader-view li { margin-bottom: 0.25em; }
.reader-view blockquote { border-left: 3px solid var(--neo-border); padding-left: 1em; color: var(--neo-faint); font-style: italic; margin: 1em 0; }
.reader-view pre { background: var(--neo-surface); padding: 1em; border-radius: 4px; overflow-x: auto; font-size: 0.875rem; }
.reader-view code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.9em; padding: 0.1em 0.3em; background: var(--neo-surface); border-radius: 3px; }
.reader-view pre code { padding: 0; background: transparent; }
.reader-view a { color: var(--neo-link, #2563eb); text-decoration: underline; text-underline-offset: 2px; }
.reader-view table { border-collapse: collapse; margin: 1em 0; width: 100%; }
.reader-view th, .reader-view td { border: 1px solid var(--neo-border); padding: 0.5em 0.75em; text-align: left; }
.reader-view img { max-width: 100%; height: auto; border-radius: 4px; }
.reader-view hr { border: none; border-top: 1px solid var(--neo-border); margin: 2em 0; }
```

Sub-Thread soll die echten Variablen-Namen aus `style.css`-Header prüfen (`--neo-text`, `--neo-faint`, `--neo-border`, `--neo-surface` sind Vermutungen aus den Tailwind-Klassen im Template — falls die CSS-Variablen anders heißen, anpassen). Wenn die CSS-Datei TOC-Konvention hat, neuen Eintrag „READER-VIEW" am richtigen Platz einsortieren.

**6. `static/js/library_detail.js` JS-Touch**:

Vier Stellen ändern (Zeilen 108, 114, 124, 348 — siehe `grep`-Output). Vorher:

```js
const text = document.querySelector('.detail-content-text').textContent;
```

Nachher:

```js
const text = document.getElementById('content-source').textContent;
```

Wenn die Klasse `.detail-content-text` an mehr Stellen referenziert ist (z.B. in CSS außerhalb dieser Datei oder in Tests): `grep` machen und ggf. Querverweise mitziehen oder dokumentieren.

### Code-Quality-Gates

- `pytest tests/` muss 71/71 grün bleiben — Backend-Charakterisierungstests touchen den Markdown-Render-Pfad über `convert_markdown`. Wenn die Tests rot werden: STOP, Master fragen.
- UI-Strings deutsch (Empty-State: „Kein Inhalt.").
- Helper-Reuse: `markdown.py` und `library.py` müssen **dieselbe** MarkdownIt+nh3-Logik benutzen, nicht duplizieren. Wenn das zu Reibung führt (z.B. weil `markdown.py` Pipeline-spezifische Schritte braucht): Sub-Thread dokumentiert die Begründung in der Pull-Request-Beschreibung.
- Live-Smoke nach Template-Änderung **Pflicht** (Test-Suite rendert keine Templates, das ist im CLAUDE.md-Stil dokumentiert).
- Keine `alert()`-Calls. Keine Inline-Styles statt CSS-Klassen.

Nach Phase 1: STOP — Bericht. Welche Helper-Disposition gewählt (neuer Modul oder Exportfunktion direkt aus `markdown.py`), wie `</script>`-Escape gelöst, welche CSS-Variablen tatsächlich gemerged wurden.

---

## Phase 2 — Verify

**Pytest**:

1. `docker compose exec markdown-converter pytest tests/` → 71/71 grün.

**Live-Smoke** (Mac-Stack auf `http://localhost:5656`, Login mit `smoke/smokepass123` oder Olivers User):

Smoke pro Conversion-Type — mindestens **ein** Dokument je Type rendern und im Browser inspizieren. Wenn die DB leer ist, schnell ein paar Test-Docs anlegen (via UI: Markdown-Converter, Document-Converter, Audio).

| Type | Smoke-Erwartung |
|---|---|
| `markdown_input` | Reichhaltiges Markdown (Headings, Listen, Codeblöcke, Tabelle, Link) → vollständig gerendert, Typografie sauber, Lese-Spalte ~70ch breit, Codeblock mit pygments-Syntax-Highlight. |
| `document_to_markdown` | Konvertierter Dokumenten-Inhalt (vermutlich Headings + Absätze) → sauber lesbar, keine HTML-Artefakte. |
| `audio_transcription` | Plain-Text-Transkript (vermutlich keine Markdown-Struktur) → wird zu Paragraphen, keine `<pre>`-Anmutung mehr. |
| `dialogue_formatting` | Markdown-Dialog mit Speaker-Labels (z.B. `**HOST:**`) → Sprecher-Labels bold, sauber lesbar. |

**Encoding-Smoke**:

5. Ein Doc mit Markdown anlegen, das den String `</script>` im Content enthält (z.B. via Markdown-Converter eine Markdown-Datei mit ```` ```html <script>...</script> ``` ```` reinwerfen oder via SQL direkt). Detail-View aufrufen → **darf nicht brechen**. Erwartung: Codeblock wird sauber gerendert, raw-source-Script-Block bleibt intakt.

**JS-Reuse-Smoke**:

6. Im Browser auf einem Detail-View:
   - Button **Kopieren** → Clipboard enthält raw Markdown (nicht HTML-strip). Prüfung: Markdown in einen Text-Editor pasten, `**bold**` muss als `**bold**` ankommen.
   - Button **Als .txt herunterladen** → die heruntergeladene Datei enthält raw Markdown.
   - Notion-Send (falls Notion lokal nicht läuft, mindestens das **vorbereitete Payload** im DevTools-Network-Tab prüfen — `content`-Feld muss raw Markdown enthalten).

**Empty-State-Smoke**:

7. Ein Doc mit leerem Content erzeugen (via API oder via leerer Markdown-Eingabe falls möglich) → Detail-View zeigt „Kein Inhalt." statt leerer Article-Container.

Nach Phase 2: STOP — Bericht. Liste der gesmokten Conversion-Types, Encoding-Smoke-Ergebnis, JS-Reuse-Smoke-Ergebnis, Browser-Screenshots-Pfade falls Sub-Thread welche gemacht hat (optional).

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- **Empfehlung**: ein Commit. Alle 6 Files gehören logisch zusammen (Reading-View-Foundation als Einheit).
- Subject z.B. „R1-A: reading-view foundation — markdown render statt pre block in library detail"
- Branch direkt auf `main`. Push direkt nach Commit (Single-User-Single-Instance-Repo, kein Sign-off-Gate).
- Vor Commit `git status` checken — alle 6 erwarteten Files im Diff, sonst nichts.

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute.

**Zusätzlich für R1-A**: wenn beim Pytest-Run nach dem `markdown.py`-Refactor Tests rot werden (Charakterisierungstests touchen `convert_markdown`): **sofort STOP**, Master fragen. Markdown-Render-Pfad ist Helper-Reuse-sensitiv, und wenn die Refaktorierung etwas subtil ändert (z.B. nh3-Reihenfolge), kann das andere Outputs erzeugen.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**M** — neuer Helper-Modul, ein Refactor (markdown.py auf Helper umgestellt), Route-Touch, Template-Touch, CSS-Block, 4 JS-Stellen umgestellt. Kein Schema-Touch, kein neuer Test-Block zwingend. Wenn beim Refactor von `markdown.py` Test-Brüche auftauchen oder die CSS-Variablen-Annahme falsch ist und größere Style-Touches nötig werden: eskaliert auf L und Master entscheidet ob splitten.

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim CSS-Touch auffällt, dass die `style.css`-TOC veraltet ist (z.B. ein älterer Block ohne TOC-Eintrag): **nicht** im Sprint mit-fixen, nur im Bericht erwähnen.
- Wenn `library_detail.js` an anderen Stellen Annahmen über die `.detail-content-text`-Klasse macht (z.B. Selektor-Suche, Event-Handler): zusätzliche Touches als Teil dieses Sprints in Ordnung, weil sie zur selben semantischen Einheit gehören.
- Wenn der Refactor von `markdown.py` einen unused-import übrig lässt: aufräumen.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „R1-A ☑ done 2026-05-25 → commit `<hash>` (reading-view-foundation: markdown-render + reader-layout in library_detail, helper-modul markdown_render.py konsolidiert markdown.py + library.py auf einen Renderer). 4 Conversion-Types gesmoked, Encoding-Smoke `</script>` clean, JS-Reuse-Smoke Copy/Download/Notion clean, Pytest 71/71 grün."
- **BACKLOG.md**: neuen Eintrag oben („R1-Cluster Reader-Core") falls noch nicht da; R1-A in Erledigt; R1-B als P1-Item aufnehmen (Highlight-Selektion + Speicherung + Notes + Tags, Größe L, braucht READER-PLAN-Workshop für Schema-Entscheidung).
- **Memory**: wenn der `</script>`-Escape-Pattern als wiederverwendbares Pattern auftaucht oder das CSS-Variablen-Mapping-Mismatch eine Lehre für künftige UI-Sprints ist: `reference_*.md` oder `feedback_*.md`. Nichts erzwingen.

---

## Phase-0-Entscheidungen

_(Phase 0 nicht aktiviert — Mechanik klar: Backend-Render mit markdown-it+nh3 (Olivers Entscheidung), `<pre>` komplett ersetzen (Olivers Entscheidung), hidden `<script type="text/markdown">` als raw-source-Pattern für JS-Reads (Master-Vorgabe), Helper-Modul `markdown_render.py` als DRY-Anker (Master-Vorgabe). Keine offene Workshop-Frage.)_
