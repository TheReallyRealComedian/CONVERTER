# UI-Inventur: library List-View (2026-05-10)

**Methodik:** Stufe 1 der Duan-Kaskade (analog F-1.1 / F-2.1 / F-3.1 / F-4.1 / F-5.1). **Code-only** Inventur — kein Live-Walkthrough in diesem Sub-Thread; visuell-laufzeitabhängige States sind als „Code-deduced, nicht live verifiziert" markiert. `library` (List-View) ist **Geschwister-Feature zu `library_detail`** (F-3-Welle): selbe `ConversionHistory`-Datenklasse, andere View-Klasse — List statt Detail. Daher die zusätzliche **F-3-Korrespondenz-Spalte** in der Element-Tabelle und eine Übersicht am Doc-Ende, welche der 15 F-3-Patterns (P1–P15) hier direkt / teil / nicht anwendbar sind.

**Feature-Kontext (aus CLAUDE.md + Memory `project_readwise_replacement.md`):** Single-User-App, LAN-only, login-protected. Primäre `library` List-View-Aufgabe: Konvertierungs-Historie übersichtlich anzeigen mit Filter (Type / Favorites), Sortierung (newest / oldest / title), Volltext-Suche (title / content / tags), Pagination (20 per page) und Navigation zu Detail. Per-Card-Quick-Actions: Favorite-Toggle, Copy-Preview, Delete. Memory-Gewicht: Library ist zentraler Reader-Ersatz für Readwise-Replacement — daily-usage-Schmerz hoch.

**Quellen:** [templates/library.html](templates/library.html), [static/js/library.js](static/js/library.js), [app_pkg/library.py](app_pkg/library.py) (`library`-Endpoint + `api_*`-Endpoints, geteilt mit `library_detail`), [templates/base.html](templates/base.html), [static/js/_utils.js](static/js/_utils.js), [static/css/style.css](static/css/style.css) (für State-Klassen `.c-btn`, `.c-btn--primary`, `.c-btn--danger`, `.c-card`, `.c-card:hover`, `.c-input`, `.c-tag`, `.type-badge`, `.favorite-btn`, `.favorite-btn.active`, `.neo-separator-top`, `.toast-notification`).

**Bereits durch frühere Sprints erfüllt — NICHT als Inventur-Befund aufnehmen** (Header-Verweis):
- **F-005 Path-Traversal-Guard** (SEC-Sprint) auf `api_*`-Endpoints.
- **F-007 `secure_filename(None)`-Guard** (HYG-Sprint) — n/a für library API (kein filename).
- **F-008 Logging-Sites mit `exc_info=True`** (HYG-Sprint) — n/a für library API (kein dedizierter logger-Call hier).
- **F-011 `@require_service`-Decorator** (HYG-Sprint) — `api_*`-Endpoints auf DB-only, kein externer SDK-Service.
- **F-013 Input-Allowlists** (SEC-Sprint) — `api_create_conversion` prüft `ALLOWED_CONVERSION_TYPES` ([app_pkg/library.py:91](app_pkg/library.py#L91)), `api_update_conversion` prüft `isinstance(data, dict)` ([app_pkg/library.py:116](app_pkg/library.py#L116)).
- **F-3-Welle** (F-3.1 / F-3.2 / F-3.3 / F-3-IMPL Sub-Batches A/B/C) ist abgeschlossen für `library_detail` — als Code-Anker und Helper-Reuse-Quelle relevant, **kein library_detail-Touch in F-6.1**.
- **F-006 markdown Backend-Whitelist** — markdown_converter, kein library-Bezug.

---

## Element-Tabelle

Legende States: ✓ vorhanden im Code · ✗ fehlt · ? unklar/teils · n/a nicht zutreffend
Live-Spalte: ✓ live verifiziert · ✗ live nicht beobachtet · ↯ Differenz Code↔live · n/a nicht prüfbar (rein statisch)
F-3-Korrespondenz: **direkt** = F-3-Pattern direkt anwendbar · **teil** = F-3-Pattern mit Modifikation · **list-spezifisch** = kein F-3-Korrespondent · **bereits-erfüllt** = F-3-Pattern in List-View strukturell schon adressiert · **n/a** = Pattern auf List-View nicht anwendbar

Regions: **Layout** = `base.html` Sidebar/Header · **Filter-Bar** = `.c-surface` mit Filter/Search/Sort/Submit · **Results-Grid** = Liste der `.c-card`-Karten · **Card** = einzelne Conversion-Karte (Element pro Karte gelistet, Mengen-Multiplikator je nach Server-Pagination) · **Pagination** = Prev/Page-Numbers/Next · **Empty-State** = Fallback-Block ohne Conversions · **Feedback** = Toast / `confirm()` / sonstige System-Reaktionen

| #  | Region | Element-Typ | Label/Text | Aktion | default | hover | focus / focus-visible | disabled | loading | error | success | empty | F-3-Korrespondenz | Live verifiziert |
|----|--------|-------------|------------|--------|---------|-------|-----------------------|----------|---------|-------|---------|-------|-------------------|------------------|
| 1  | Layout | Button | Theme-Toggle (Sun/Moon-Icon) | Toggle `data-global-theme` light↔dark + persist `localStorage` | ✓ | ✓ (`.theme-toggle-btn:hover`) | ? (kein eigenes `:focus-visible`-Ring) | n/a | n/a | n/a | n/a | n/a | F-3 #1 (identisch via base.html) | n/a (Code-only) |
| 2  | Layout | Link | "File Transformer" (Brand) | → `url_for('markdown_converter')` | ✓ | ✓ (`hover:text-neo-accent`) | ? | n/a | n/a | n/a | n/a | n/a | F-3 #2 (identisch) — **EN-String** | n/a (Code-only) |
| 3  | Layout | Link | "Markdown to PDF" (Sidebar-Nav) | → `/markdown-converter` | ✓ | ✓ | ? | n/a | n/a | n/a | n/a | n/a | F-3 #3 (identisch) — **EN-String** | n/a (Code-only) |
| 4  | Layout | Link | "Document Converter" (Sidebar-Nav) | → `/document-converter` | ✓ | ✓ | ? | n/a | n/a | n/a | n/a | n/a | F-3 #4 (identisch) — **EN-String** | n/a (Code-only) |
| 5  | Layout | Link | "Audio Converter" (Sidebar-Nav) | → `/audio-converter` | ✓ | ✓ | ? | n/a | n/a | n/a | n/a | n/a | F-3 #5 (identisch) — **EN-String** | n/a (Code-only) |
| 6  | Layout | Link | "Mermaid Converter" (Sidebar-Nav) | → `/mermaid-converter` | ✓ | ✓ | ? | n/a | n/a | n/a | n/a | n/a | F-3 #6 (identisch) — **EN-String** | n/a (Code-only) |
| 7  | Layout | Link | "Library" (Sidebar-Nav, **aktiv** auf dieser Seite) | → `/library`; aktiver State via `neo-nav-active`-Klasse, **wird in `base.html` per `'/library' in request.path` gesetzt** ([templates/base.html:84](templates/base.html#L84)) — deckt sowohl `/library` als auch `/library/<id>` ab | ✓ + `neo-nav-active` | ✓ | ? | n/a | n/a | n/a | n/a | n/a | F-3-Pattern P11 **bereits-erfüllt** (path-Match deckt List + Detail beide ab, ist die Stelle die F-3-IMPL als P11 dokumentiert hat) — **EN-String** | n/a (Code-only) |
| 8  | Layout | Link | "Logout" | → `/logout` | ✓ | ✓ (`hover:text-neo-text`) | ? | n/a | n/a | n/a | n/a | n/a | F-3 #8 (identisch) — **EN-String** | n/a (Code-only) |
| 9  | Layout | Button | Mobile-Sidebar-Toggle (Hamburger) | Sidebar ein/ausblenden (mobile) | ✓ | ? | ? | n/a | n/a | n/a | n/a | n/a | F-3 #9 (identisch) | n/a (Code-only) |
| 10 | Filter-Bar | Form-Wrapper | `<form method="GET" action="{{ url_for('library') }}">` | umschließt alle Filter-Felder; submit → Page-Reload mit URL-Query-Params; **alle Filter-State lebt im URL-Query, kein localStorage** | ✓ | n/a | n/a | n/a | n/a | n/a | n/a | n/a | list-spezifisch (List-View-State im URL statt clientseitig) | n/a (Code-only) |
| 11 | Filter-Bar | Select | `name="type"` (`.c-input`, `onchange="this.form.submit()"`, `title="Filter by type"`) — Optionen: "All Types", "Document", "Audio", "Dialogue", "Markdown" mit `{% if current_type == ... %}selected` | Auto-Submit on `change` → reload mit `?type=<val>`-Param; Backend filtert `Conversion.conversion_type==val` ([app_pkg/library.py:43-44](app_pkg/library.py#L43-L44)) | ✓ | n/a | ✓ (`select.c-input:focus`) | n/a | ✗ kein Loading-State (Roundtrip ist Form-GET, Browser-Reload macht Visual-Reset implizit) | n/a (kein Error-Pfad für `type=ungültig`; Backend `filter_by` matched einfach nicht und liefert leere Liste) | ✓ implizit (Reload mit gefilterter Liste) | n/a | list-spezifisch (kein F-3-Korrespondent — Detail-View hat keinen Filter); **F-3-Pattern P6 DE-Microcopy direkt anwendbar** auf Optionen + `title`-Attribut — **EN-Strings**: "All Types", "Document", "Audio", "Dialogue", "Markdown", `title="Filter by type"` | n/a (Code-only) |
| 12 | Filter-Bar | Input (Text) | `name="search"` (`.c-input w-full`, `placeholder="Search title, content, tags..."`, `value="{{ current_search }}"`) | **Kein auto-submit** — User muss "Search"-Submit-Btn (#15) klicken oder Enter im Feld drücken. Backend SQL-escapet `%`, `_`, `\` ([app_pkg/library.py:48](app_pkg/library.py#L48)) und macht `ilike` auf title/content/tags ([:49-55](app_pkg/library.py#L49-L55)). | ✓ | n/a | ✓ (`.c-input:focus`) | n/a | ✗ kein Loading | n/a (kein Error-Pfad) | ✓ implizit (Reload mit Treffern) | ✓ Placeholder "Search title, content, tags..." (EN) | list-spezifisch — **EN-Placeholder**; ↯ Inkonsistenz mit #11/#13/#14 (alle anderen Filter sind auto-submit) — siehe Befund 8 | n/a (Code-only) |
| 13 | Filter-Bar | Checkbox + Label | `name="favorites"` `value="1"` (`onchange="this.form.submit()"`) mit `<span>Favorites</span>`-Label; `{% if current_favorites %}checked{% endif %}` | Auto-Submit on `change` → reload mit `?favorites=1` oder ohne; Backend `== '1'` ([app_pkg/library.py:36](app_pkg/library.py#L36)) → filter_by `is_favorite=True` ([:46](app_pkg/library.py#L46)) | ✓ | n/a | ? (Browser-Default-Outline auf Checkbox) | n/a | ✗ kein Loading | n/a | ✓ implizit | n/a | list-spezifisch — **EN-Label** "Favorites" | n/a (Code-only) |
| 14 | Filter-Bar | Select | `name="sort"` (`.c-input`, `onchange="this.form.submit()"`, `title="Sort order"`) — Optionen: "Newest", "Oldest", "Title A-Z" | Auto-Submit on `change` → reload mit `?sort=<val>`-Param; Backend `order_by` ([app_pkg/library.py:57-62](app_pkg/library.py#L57-L62)); default `newest`. | ✓ | n/a | ✓ | n/a | ✗ kein Loading | n/a | ✓ implizit | n/a | list-spezifisch; **F-3-Pattern P6 DE-Microcopy direkt anwendbar** auf Optionen + `title`-Attribut — **EN-Strings**: "Newest", "Oldest", "Title A-Z", `title="Sort order"` | n/a (Code-only) |
| 15 | Filter-Bar | Button (Submit) | "Search" (`.c-btn.c-btn--primary.py-1.5.px-4.text-sm`, `type="submit"`) | submit-Form → Page-Reload mit gesetzten Query-Params; **redundant für #11/#13/#14 die schon auto-submitten**, primär für #12 (Search-Input ohne auto-submit) und für Enter-Key-Hit auf #12 | ✓ | ✓ (`.c-btn:hover`) | ✓ (`.c-btn:focus-visible`) | n/a | ✗ kein Loading | n/a | ✓ implizit | n/a | list-spezifisch — **EN-String** "Search" | n/a (Code-only) |
| 16 | Results-Grid | Container | `<div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">` (nur wenn `conversions`-Liste truthy; conditional auf `{% if conversions %}` ab [templates/library.html:38](templates/library.html#L38)) | rein dekorativer Wrapper; iteriert `conversions` Server-side, ein `.c-card` pro Eintrag | ✓ (wenn Liste nicht leer) | n/a | n/a | n/a | n/a | n/a | n/a | ✓ wenn Liste leer → Empty-State-Block (#28) statt Grid | list-spezifisch | n/a (Code-only) |
| 17 | Card | Container | `<div class="c-card flex flex-col" data-id="{{ conv.id }}">` — pro Conversion-Eintrag eine Karte mit `data-id` für JS-Targeting | rein dekorativer Wrapper; `.c-card:hover` lifted die Karte (`translateY(-2px)` + `box-shadow: var(--nm-raised-lg)`, [static/css/style.css:242-245](static/css/style.css#L242-L245)) | ✓ | ✓ (`.c-card:hover` mit translateY) | n/a (Container nicht selbst fokussierbar) | n/a | n/a | n/a | n/a | n/a | list-spezifisch (keine analoge Card-Struktur in Detail-View) | n/a (Code-only); ↯ Card-`hover` lift bewegt die Karte; bei mausgeführtem Klick-Toggle des Favorite-Buttons im oberen Bereich könnten gleichzeitige `:hover`-Lift + Klick einen Mis-Hit produzieren (Live-Walkthrough-Lücke) |
| 18 | Card | Span (dekorativ) | `.type-badge.type-{{ conv.conversion_type }}` mit Text "Document" / "Audio" / "Dialogue" / "Markdown" / Fallback `conv.conversion_type` | rein dekorativ; CSS-Tinting via `.type-document_to_markdown` etc. ([static/css/style.css:1050+](static/css/style.css#L1050)) | ✓ | n/a | n/a | n/a | n/a | n/a | n/a | n/a | F-3 #12 (identisch via selbes type-badge-Pattern) — **EN-Strings**: "Document", "Audio", "Dialogue", "Markdown" (identisch zu F-3-Befund 4 — DE-Pass via P6 deckt beide ab) | n/a (Code-only) |
| 19 | Card | Button | `.favorite-btn` (Glyph `★`/`☆`, `title="Toggle favorite"`, `onclick="toggleFavorite({{ conv.id }}, this)"`, `{% if conv.is_favorite %}active{% endif %}`) | `toggleFavorite(id, btn)` → `PUT /api/conversions/<id>` JSON `{is_favorite: !current}`. Bei `r.ok`: Klassen-Toggle `.active` + innerHTML-Swap `&#9733;`/`&#9734;`. **Kein Toast, keine Bestätigung, kein Error-Pfad** ([static/js/library.js:3-15](static/js/library.js#L3-L15)). Bei Fehler: **silent** (kein `.catch`, kein `r.ok===false`-Branch, kein `safeJSON`-Wrap). | ✓ (Glyph je nach State) | ✓ (`.favorite-btn:hover` color-shift) | ? (kein eigenes `:focus-visible`) | n/a | ✗ kein Loading-State während PUT | ✗ kein Error-State (silent fail) | ✓ Klassen-Toggle + Glyph-Swap | n/a | F-3-Pattern P1 **direkt** (Auto-Save-silent-fail-Familie — identische Mechanik wie `toggleFavorite` in `library_detail.js`) + F-3-Pattern P14 **direkt** (Helper-Reuse-Konvergenz `safeJSON` für Login-Redirect-HTML-Detection); **EN-`title`** "Toggle favorite" → P6 DE-Pass | n/a (Code-only); ↯ Card-hover-lift gleichzeitig — siehe Anmerkung #17 |
| 20 | Card | Link | `<a href="{{ url_for('library_detail', conversion_id=conv.id) }}" class="px-4 py-2 flex-1 no-underline">` enthält `<h5>` (Title), optional `<div>` (Source-Filename) und `<p>` (Content-Preview clipped 200 chars + `...`-Suffix wenn länger) | Navigation zur Detail-View; rein server-rendered, kein JS-Hook | ✓ | ✓ (Browser-Default für `<a>`, ohne explizites CSS) | ✓ (Browser-Default-Outline) | n/a | n/a | n/a | ✓ implizit (Navigation) | n/a (Title ist Server-required mit Default „Untitled") | list-spezifisch | n/a (Code-only); ↯ `conv.content[:200]` ist ein Python-String-Slice — Bytes-aware? Nein, ist String-Slice. Bei Multibyte-Chars könnte das `...`-Suffix-Pattern unklar erscheinen wenn der Cut mitten in einem Word liegt (Live-Walkthrough-Lücke) |
| 21 | Card | Span (Datum) | `<span class="text-[11px] text-neo-faint" title="{{ conv.created_at.strftime('%Y-%m-%d %H:%M') }}">{{ conv.created_at.strftime('%d %b %Y, %H:%M') }}</span>` — Visible: z.B. "10 May 2026, 13:42"; Tooltip: ISO-Format | rein dekorativ; **`%b` ist Locale-abhängig** — Container-Locale ist nicht erzwungen, bei `LC_ALL=C` (Docker-Default) liefert es englische Kürzel "May", "Jun", … | ✓ | n/a (Span hat keine Hover-Animation, aber `title`-Attribut zeigt Browser-Tooltip on hover) | n/a | n/a | n/a | n/a | n/a | n/a | F-3-Pattern P5 **teil** (Datum-Lokalisierung) — F-3 hat das Problem mit UTC-Datum in `<input type="datetime-local">` adressiert; hier ist es ein anderer Mechanismus (Server-side strftime mit Locale-abhängiger Monatsabkürzung) — **EN-Monatskürzel via `%b`** (siehe Befund 11) | n/a (Code-only); ↯ Live-Locale-Verhalten je nach Docker-Image-Locale unklar — siehe Befund 11 |
| 22 | Card | Span (Tag-Chip, wiederholt) | `<span class="c-tag text-[10px] px-1.5 py-0.5 rounded-full text-neo-muted">{{ tag.strip() }}</span>` für jedes nicht-leere Element aus `conv.tags.split(',')` | rein dekorativ; **Tag-Chip-Visualisierung über `.c-tag`-Klasse ([static/css/style.css:1108-1110](static/css/style.css#L1108-L1110))** — also rendering ist hier **bereits chip-style**, im Gegensatz zur Detail-View die noch Plain-CSV-Input rendert | ✓ (nur wenn `conv.tags` truthy) | n/a | n/a | n/a | n/a | n/a | n/a | ✓ Block nicht gerendert wenn keine Tags | F-3-Pattern P9 **bereits-erfüllt** (List-View rendert Tag-Chips schon mit `.c-tag`-Klasse; Detail-View P9 wird dasselbe Rendering einführen — interessante Inversion: List-View ist hier voraus) | n/a (Code-only) |
| 23 | Card | Button | "Copy" (`.c-btn.text-xs.py-1.px-2`, `onclick="copyContent({{ conv.id }})"`, `title="Copy content"`) | `copyContent(id)` → liest `card.querySelector('.line-clamp-3').textContent` (**das ist die clipped 200-char-Preview, nicht der Full-Content!**) → `fallbackCopyText(...)` → bei Erfolg: `showToast('Content copied to clipboard')`, bei Fehler: `showToast('Copy failed')` ([static/js/library.js:17-25](static/js/library.js#L17-L25)) | ✓ | ✓ (`.c-btn:hover`) | ✓ (`.c-btn:focus-visible`) | n/a | ✗ kein Loading-State (clipboard sync) | ✓ Toast "Copy failed" (EN, aber Toast-Level=default `success` — **falsch tönend**, F-3 Befund 13 / P8 1:1) | ✓ Toast "Content copied to clipboard" (EN, Level=success) | n/a | F-3 #14 (Detail-View "Copy to Clipboard") + Pattern **P8 direkt** (Toast-Level für Fehler-Pfad) — **EN-Strings** "Copy", `title="Copy content"`, "Content copied to clipboard", "Copy failed"; **Bug-Kandidat**: kopiert nur 200-char-Preview statt Full-Content (siehe Befund 1) | n/a (Code-only); ↯ `.line-clamp-3` selector resolves zu `<p>` mit clipped Inhalt — Befund 1 |
| 24 | Card | Button (Danger) | "Delete" (`.c-btn.c-btn--danger.text-xs.py-1.px-2`, `onclick="deleteConversion({{ conv.id }}, this)"`, `title="Delete"`) | `deleteConversion(id, btn)` → `confirm('Delete this conversion? This cannot be undone.')` (EN, browser-native) → bei OK: `DELETE /api/conversions/<id>` → bei `r.ok`: `card.style.opacity=0` + scale-Transform + nach 200 ms `card.remove()` ([static/js/library.js:27-38](static/js/library.js#L27-L38)). **Kein Error-Feedback bei DELETE-Fail** (silent), **kein Loading-State** während Roundtrip, **kein `safeJSON`-Wrap** für Login-Redirect-HTML-Detection. | ✓ | ✓ (`.c-btn:hover` über `.c-btn--danger`-Variation) | ✓ (`.c-btn:focus-visible`) | n/a | ✗ kein Loading-State (Button bleibt klickbar während DELETE) | ✗ kein Error-State (silent fail) | ✓ Card visuell entfernt | n/a | F-3 #17 (Detail-View "Delete") + Pattern **P3 direkt** (Delete-silent-fail-Familie) + Pattern **P14 direkt** (Helper-Reuse-Konvergenz `safeJSON`); **EN-confirm-String** identisch zu F-3 Detail-View → P6 DE-Pass deckt beide ab — **EN-Strings** "Delete", `title="Delete"`, `confirm('Delete this conversion? This cannot be undone.')` | n/a (Code-only); ↯ Card-Remove ohne aria-live-Update — Screenreader bekommt keinen Hinweis dass die Karte entfernt wurde (Live-Walkthrough-Lücke + a11y-Befund 15) |
| 25 | Pagination | Anchor | "Prev" (`.c-btn.text-sm.py-1.5.px-3.no-underline`, conditional auf `pagination.has_prev`) | → reload mit `?page=<prev_num>` plus alle aktuellen Filter (`type`, `search`, `favorites`, `sort`) als Query-Params re-injected | ✓ (nur wenn `has_prev`) | ✓ | ✓ | n/a (nicht-vorhanden statt disabled) | n/a | n/a | ✓ implizit (Reload) | n/a | list-spezifisch — **EN-String** "Prev"; ↯ URL hat `favorites=''` als leeren String wenn nicht aktiv → URL-Artifact `?favorites=` sichtbar (siehe Befund 9) | n/a (Code-only) |
| 26 | Pagination | Anchor + Ellipsis-Span | Page-Number-Links via `pagination.iter_pages(left_edge=1, right_edge=1, left_current=2, right_current=2)`; aktive Seite `.c-btn.c-btn--primary`, andere `.c-btn`; `None`-Slots der Iter-Liste rendern `<span>...</span>` | → reload mit `?page=<p>` + Filter-Params | ✓ + aktive Page `c-btn--primary` | ✓ | ✓ | n/a | n/a | n/a | ✓ implizit | n/a | list-spezifisch | n/a (Code-only) |
| 27 | Pagination | Anchor | "Next" (`.c-btn.text-sm.py-1.5.px-3.no-underline`, conditional auf `pagination.has_next`) | → reload mit `?page=<next_num>` + Filter-Params | ✓ | ✓ | ✓ | n/a | n/a | n/a | ✓ implizit | n/a | list-spezifisch — **EN-String** "Next" | n/a (Code-only) |
| 28 | Empty-State | Container | `<div class="text-center py-16 text-neo-muted">` mit `<h5>` "No saved conversions yet" + `<p>` "Use \"Save to Library\" on any converter page to start building your library." | rein dekorativ, gerendert wenn `{% if conversions %}`-Branch falsy ist | n/a (nur als Empty-Fallback gerendert) | n/a | n/a | n/a | n/a | n/a | n/a | ✓ Block sichtbar bei leerer Liste | list-spezifisch — **EN-Strings** beide; ↯ unterscheidet **nicht** zwischen „nie eine Conversion gespeichert" und „aktueller Filter liefert keine Treffer" — siehe Befund 12 | n/a (Code-only) |
| 29 | Feedback | Toast (`.toast-notification`) | dynamisch via `showToast(message, opts)` aus `_utils.js` — nur Copy-Erfolg + Copy-Fehler-Pfade ([static/js/library.js:21,23](static/js/library.js#L21)) | Singleton-Toast: alte Toasts entfernt, neuer mit `.show`-Klasse, auto-dismiss nach 2.5 s default | ✓ idle | n/a | n/a | n/a | n/a | ✗ alle showToast-Calls auf List-View ohne `level: 'danger'` → "Copy failed" rendert als grüner Success-Toast (siehe Befund 4) | ✓ | ✓ idle | F-3 #30 (selbes Toast-System) + Pattern **P8 direkt** für Copy-Fail-Level | n/a (Code-only); **alle Toasts auf List-View sind level=success unabhängig vom Inhalt** |
| 30 | Feedback | Browser-native `confirm()` | "Delete this conversion? This cannot be undone." (EN) | nur Delete-Pfad (#24). Es wird **nicht** das `confirmIfLong`-Helper aus `_utils.js` verwendet (Threshold-Logik ist hier auch unpassend — User soll **jeden** Delete bestätigen). | ✓ überall (Browser-Default) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | F-3 #31 (identische Mechanik, identischer EN-String) → Pattern **P3 direkt** (Microcopy-Konvergenz mit Detail-View) — **EN-String, Browser-native** | n/a (Code-only) |

---

## Zusammenfassung

- **Gesamtzahl interaktive Elemente:** 21 (Layout #1-#9, Filter-Bar #11–#15 als 5 distinkte Interaktions-Targets, Card-Interaktionen #19/#20/#23/#24 als 4, Pagination #25/#26/#27 als 3). Container/dekorativ/state-system Zeilen (#10 Form-Wrapper, #16 Results-Grid-Container, #17 Card-Container, #18 type-badge, #21 Datum-Span, #22 Tag-Chips, #28 Empty-State, #29 Toast-Singleton, #30 Browser-confirm) zur Vollständigkeit gelistet — Gesamt-Tabellenzeilen: 30. Im erwarteten Bereich (Sprint-Prompt 20-40, F-3.1's 21 bis F-5.1's 32).

- **Im Code identifizierte fehlende States (✗):**
  - **Filter-Selects/Checkbox/Search-Input (#11–#14) haben keinen Loading-State** beim Auto-Submit — Roundtrip ist GET-Reload, Browser-Default-Loading kommt zur Anwendung; kein Skeleton, kein „lade gefilterte Liste …"-Hint. Bei langsamer DB könnte die ms zwischen Klick und Reload als „nichts passiert" wirken.
  - **Search-Input (#12) hat keinen Live-Search / kein Debouncing** — Submit nur via #15-Button oder Enter-Key; alle anderen Filter sind auto-submit. **UX-Inkonsistenz** (siehe Befund 8).
  - **Favorite-Btn (#19) hat keinen Loading- und keinen Error-State** — `toggleFavorite()` toggelt nur bei `r.ok`, fängt Network-Failures gar nicht und zeigt keine Bestätigung (siehe Befund 2).
  - **Favorite-Btn (#19) hat kein eigenes `:focus-visible`-Style** — identisch zu F-3 Befund auf Detail-View.
  - **Delete-Btn (#24) hat keinen Loading-State** während DELETE-Roundtrip und keinen Error-State bei DELETE-Fail (silent — Card bleibt sichtbar, kein Hinweis warum) (siehe Befund 3).
  - **Copy-Btn (#23) Toast-Level bei Failure ist `success`** — `showToast('Copy failed')` ohne `opts.level='danger'` (siehe Befund 4).
  - **Card-Remove nach Delete (#24) hat keine aria-live-Region** — Screenreader bekommt keinen Hinweis dass die Karte entfernt wurde (siehe Befund 15).
  - **Empty-State (#28) ist nicht filter-aware** — selbe Message für „keine Conversions in der DB" und „aktueller Filter liefert keine Treffer" (siehe Befund 12).
  - **Pagination-Links bauen URL mit leerem `favorites=''`-String** wenn nicht aktiv → URL-Artifact (siehe Befund 9).
  - **Kein Banner-Mountpoint für `showAlert`** im Template — alle Fehler-Pfade landen entweder im Toast (mit falschem Level) oder silent (siehe Befund 5).

- **Code↔live-Divergenzen-Verdacht (↯, Code-deduced):**
  - **`%b`-Monatskürzel locale-abhängig** (#21): bei `LC_ALL=C` (Docker-Default) liefert Python's `strftime('%b')` englische Kürzel ("May", "Jun"). Container-Locale ist auf Mintbox de_DE (laut CLAUDE.md), aber im **Docker-Container** wahrscheinlich `C` oder `C.UTF-8`. Master-Walkthrough-Verifikation: was steht real im Browser? (siehe Befund 11).
  - **Card-Hover-Lift gleichzeitig mit Favorite-Klick** (#17/#19): `.c-card:hover { translateY(-2px) }` läuft sobald Maus die Card berührt; der Favorite-Btn sitzt im oberen Card-Bereich — Klick bei laufender Translate-Animation kann theoretisch einen Mis-Hit produzieren wenn die Maus zwischen Pointer-Down und -Up die Translate-bewegte Hitbox verlässt. Live-Walkthrough-Lücke, vermutlich Browser-robustes Pointer-Capture macht das unkritisch.
  - **`conv.content[:200]`-Cut** (#20): Python-String-Slice bei Multibyte-Codepoints ist sauber (Slice operiert auf Code-Units in CPython 3, also einzelne Codepoints, nicht Bytes); aber **bei Cut mitten in einer Tabelle / Code-Block / Liste** kann das `...`-Suffix verwirrend sein. Kosmetisch.
  - **Copy-Btn kopiert nur 200-char-Preview** (#23): `.line-clamp-3.textContent` ist exakt der gleiche Server-clipped Inhalt aus #20, nicht der DB-Full-Content. **Bug** — siehe Befund 1.

- **Unverifizierbare States (?):**
  - `:focus-visible`-Styles auf Sidebar-Links (#3–#8), Brand-Link (#2), Theme-Toggle (#1), Mobile-Sidebar-Toggle (#9), Favorite-Btn (#19), Checkbox (#13) — Code zeigt keine expliziten Regeln, Browser-Default-Outline kommt zur Anwendung.
  - Verhalten bei sehr großer Conversion-Liste (>1000 Einträge total) — Pagination cap auf 20 per page klemmt es, aber Aggregate-Query auf SQLite mit Volltext-`ilike` über `content` könnte bei großen Datasets langsam werden. Performance-Frage, nicht Code-Reading.
  - Browser-Verhalten bei `confirm()` über Tastatur — Browser-Default.

- **Unterschiede zur Detail-View (F-3.1):**
  - **Keine Edit-Pfade** für Title/Tags/Content — alle Edits gehen über Detail-View. Auto-Save-Pattern P1/P2 sind hier **nur über Favorite-Btn** relevant (single-field-PUT).
  - **Kein Notion-Pfad** — entfällt P4/P5/P7/P10.
  - **Kein Page-Title-Update** — entfällt P13.
  - **Tag-Chip-Rendering bereits vorhanden** (P9 bereits-erfüllt — interessante Inversion: List-View ist hier voraus).
  - **Sidebar-Active-State bereits korrekt** für List + Detail dank path-Match in `base.html:84` (P11 bereits-erfüllt).
  - **List-spezifische States**: Filter / Sort / Search / Pagination (alle URL-Param-getrieben) + Empty-State + Per-Card-Quick-Actions (Favorite-Toggle / Copy-Preview / Delete).

---

## List-View-States-Sub-Sektion

Mapping der List-View-spezifischen State-Klassen (Master-Annotation 4):

### Sortierung-State

- **Mechanik**: URL-Query-Param `?sort=newest|oldest|title`, default `newest` ([app_pkg/library.py:37](app_pkg/library.py#L37)).
- **UI-Repräsentation**: `<select name="sort">` ([templates/library.html:28-32](templates/library.html#L28-L32)) mit `{% if current_sort == ... %}selected{% endif %}`-Markern.
- **Persistierung**: rein URL-basiert; **kein localStorage**, **kein Server-Session-State**. Reload ohne Query → default `newest`.
- **Helper-Reuse-Status**: kein `saveViewState`/`loadViewState` (Design-Wahl: URL ist die Persistierungs-Schicht). Siehe Master-Annotation-2-Antwort in Helper-Reuse-Spuren-Sektion.

### Filter-State (Type + Favorites)

- **Mechanik**: URL-Query-Params `?type=<conversion_type>` + `?favorites=1`, beide optional ([app_pkg/library.py:34-36](app_pkg/library.py#L34-L36)).
- **UI-Repräsentation**: `<select name="type">` (#11) + `<input type="checkbox" name="favorites" value="1">` (#13). Beide `onchange="this.form.submit()"`.
- **Backend-Filter**: ([app_pkg/library.py:43-46](app_pkg/library.py#L43-L46)) — `conversion_type` matched gegen `ALLOWED_CONVERSION_TYPES`-Set ohne Validation (jeder String wird gegen DB gehalten; SQL ist sicher, nur Result-Set leer wenn ungültig); `favorites` ist striktes `== '1'`-Match.
- **Persistierung**: URL only.

### Suche-State

- **Mechanik**: URL-Query-Param `?search=<query>`, default empty string ([app_pkg/library.py:35](app_pkg/library.py#L35)).
- **UI-Repräsentation**: `<input type="text" name="search">` (#12) **ohne `onchange="this.form.submit()"`** — Submit nur via "Search"-Button (#15) oder Enter im Feld.
- **Backend**: SQL-LIKE-Escape für `%`, `_`, `\` ([app_pkg/library.py:48](app_pkg/library.py#L48)) plus `ilike` über title/content/tags mit `OR` ([:49-55](app_pkg/library.py#L49-L55)).
- **Debouncing**: keins — Submit ist explizit.
- **Inkonsistenz** vgl. #11/#13/#14: Filter/Sort sind auto-submit, Search ist explizit-submit. Siehe Befund 8.

### Bulk-Selektion-State

- **Mechanik**: **existiert nicht**. Keine Checkboxes pro Card, keine Select-All-Logik, keine Bulk-Action-Toolbar.
- **Code-Anker**: kein Code für Bulk-Selektion in `library.js` / `library.html`.
- **Helper-Reuse-Status**: `confirmInPlace` aus F-4-IMPL ist **n/a** auf der List-View, weil kein Bulk-Delete-Pfad existiert. Per-Card-Delete (#24) verwendet rohes `confirm()` und ist kein Bulk-Use-Case. Master-Annotation-2-Antwort.

### Pagination-State

- **Mechanik**: URL-Query-Param `?page=<int>`, default `1` ([app_pkg/library.py:38](app_pkg/library.py#L38)); SQLAlchemy `query.paginate(page=page, per_page=20, error_out=False)` ([app_pkg/library.py:64](app_pkg/library.py#L64)).
- **UI-Repräsentation**: `pagination.iter_pages(...)` mit Prev/Next-Links und Page-Number-Anchors (#25/#26/#27), `c-btn--primary` markiert aktive Seite.
- **Per-Page-Size**: hardcoded `20`, **nicht UI-konfigurierbar**.
- **Infinite-Scroll vs. Numeric-Pages**: numeric.

### Empty-State

- **Mechanik**: `{% if conversions %}` ([templates/library.html:38](templates/library.html#L38)) — gerendert wenn `pagination.items` truthy ist. Else-Branch (#28) zeigt Fixed-Message.
- **Inhalt**: "No saved conversions yet" + Hinweis auf "Save to Library" auf Converter-Pages.
- **Filter-Awareness**: **keine** — selbe Message für „nie was gespeichert" und „Filter liefert nichts" (siehe Befund 12).

### Page-Load / Initial-State

- **Mechanik**: Server-side Render mit URL-Query-Params → SQLAlchemy-Query → `render_template` mit `conversions`, `pagination`, `current_type`, `current_search`, `current_favorites`, `current_sort` als Context-Vars.
- **Re-Hydration**: alle Filter-Werte (`current_*`) werden ins Template re-injected damit Selects/Checkboxes/Search die aktuelle Filter-Anzeige korrekt darstellen.

---

## Helper-Reuse-Spuren

Pro etabliertem Helper aus `_utils.js` plus `file_size`-Jinja-Filter aus `app_pkg/__init__.py` — Status auf List-View:

- **`fallbackCopyText`** ✓ **genutzt** in `copyContent()` ([static/js/library.js:20](static/js/library.js#L20)) — sauber.
- **`showToast`** ✓ **genutzt** in `copyContent()`-Success-/Failure-Pfaden ([static/js/library.js:21,23](static/js/library.js#L21)) — **aber Failure-Pfad ohne `level: 'danger'`** → falsches Toast-Tönen (Befund 4, P8-Konvergenz).
- **`showAlert`** ✗ **nicht genutzt** in `toggleFavorite()` (silent fail bei API-Error) und `deleteConversion()` (silent fail bei DELETE-Fail) — keine Banner-Mountpoint-Container im Template (siehe Befund 5, P15-Konvergenz).
- **`safeJSON`** ✗ **nicht genutzt** in `toggleFavorite()` und `deleteConversion()` — Session-Expired (Login-Redirect-HTML als 200) würde bei `toggleFavorite` als `r.ok===true` durchgehen und den Glyph fälschlich togglen ohne dass etwas gespeichert wurde. Bei `deleteConversion` analog: Card würde aus dem DOM entfernt ohne DB-Delete.
- **`formatFileSize`** **n/a** — kein File-Size-Display auf den Cards. Server-Filter `file_size` ist deshalb auch **n/a für List-View**, wird aber bestehend für Detail-View genutzt (F-3.IMPL P12).
- **`formatDatetimeLocalNow`** **n/a** — kein `datetime-local`-Input auf List-View; Datum wird Server-side strftime-formatiert (Card #21).
- **`confirmIfLong`** **n/a** — Delete-Konfirmation soll **jeden** Delete bestätigen (kein Threshold-Bypass sinnvoll für eine Card-Lösch-Aktion); aktuell raw `confirm()` (#24/#30). Threshold-Logik passt nicht zur Semantik der Card-Lösch-Aktion.
- **`loadViewState` / `saveViewState`** (Master-Annotation 2, zweite Call-Site-Frage): **n/a auf List-View** mit **expliziter Begründung** — der gesamte View-State der List-View (Sortierung / Filter / Favorites-Checkbox / Search-Query / Page-Index) wird **bereits über URL-Query-Params persistiert**, was eine andere Persistierungs-Schicht ist als localStorage. Vorteile der URL-Persistierung: bookmark-bar, sharable, browser-back-restoriert State, kein localStorage-Quota-Risk. Falls F6-PATTERNS argumentiert dass List-View beim **Default-Reload ohne Query-Params** trotzdem den letzten User-State wiederherstellen soll (Reader-Mode-Analogie), wäre `saveViewState` eine zweite Call-Site — aber das ist eine **Design-Entscheidung**, kein Code-deduzierter Befund. → **als Helper-Reuse-Spur dokumentiert mit „n/a-mit-Begründung", kein H4-Konvergenz-Befund automatisch**.
- **`confirmInPlace` aus F-4-IMPL** (Master-Annotation 2, zweite Call-Site-Frage): **n/a auf List-View** mit **expliziter Begründung** — `confirmInPlace` ist die Idle→Confirm-Pending→Cancelling-State-Machine aus `audio_converter.js` für **mid-flight-cancel** eines laufenden RQ-Jobs, kein Generic-Bulk-Delete-Helper. Die List-View hat **keinen Bulk-Delete-Pfad** (siehe List-View-States-Sub-Sektion → Bulk-Selektion-State); per-Card-Delete (#24) verwendet rohes `confirm()` und ist semantisch näher zu F-3 #17 (Detail-View Delete) als zu F-4 Cancel. → **als Helper-Reuse-Spur dokumentiert mit „n/a-mit-Begründung", kein Helper-Extraktion-Trigger**.

### Code-deduzierte H4-Konvergenz-Lage

Die Helper-Vermisst-Stellen sind dieselben wie in `library_detail.js` aus F-3.1 (silent-fail in `toggleFavorite` / `deleteConversion` / Toast-Level-Failure). F-3-IMPL hat Sub-Batch A diese Stellen in Detail-View geschlossen — die **Inversion** auf List-View ist die Stelle wo F6-IMPL den Pattern-Übertrag macht. Identische Code-Patches strukturell.

---

## F-3-Korrespondenz-Übersicht

Mapping der 15 F-3-Patterns aus [docs/ui_patterns_library_detail_2026-05.md](ui_patterns_library_detail_2026-05.md) auf library List-View:

### Direkt anwendbar (selbes Pattern, anderer Code-Anker)

| F-3-Pattern | Beschreibung F-3 | Übertragbarkeit auf library List-View | Verweis |
|-------------|-------------------|----------------------------------------|---------|
| **P1 — Auto-Save Title/Tags + Favorite silent-fail** | `updateField` / `toggleFavorite` in `library_detail.js` ohne `.catch` / `r.ok===false`-Branch | **direkt** für `toggleFavorite` in `library.js` — identische Mechanik, identische `.catch`/`r.ok`-Lücken, identischer Showalert-Banner-Mountpoint-Bedarf (siehe Befund 2 und P15-Verzahnung) | Befund 2 |
| **P3 — Delete silent-fail** | `deleteConversion` in `library_detail.js` ohne Error-Pfad + EN-confirm | **direkt** für `deleteConversion` in `library.js` — identische Mechanik plus zusätzlicher a11y-Aspekt (aria-live für Card-Remove, siehe Befund 15) | Befund 3, Befund 15 |
| **P6 — DE-Microcopy-Pass flächendeckend** | ~18 EN-Strings auf Detail-View | **direkt** für List-View EN-Strings: Filter-Optionen (#11), Search-Placeholder (#12), Favorites-Label (#13), Sort-Optionen (#14), Search-Button (#15), Type-Badge-Texte (#18 — gemeinsam mit Detail-View), Favorite-`title` (#19), Card-Buttons (#23/#24) + Tooltips, confirm-Text (#24/#30), Pagination-Texte (#25/#27), Empty-State (#28), Toast-Texte (#23/#29). Sammel-Befund 6. | Befund 6 |
| **P8 — Toast-Level pro Call-Site korrekt setzen** | `showToast` ohne `level: 'danger'` für Error-Pfade in Detail-View | **direkt** auf #23 Copy-Failure-Pfad ([static/js/library.js:23](static/js/library.js#L23)) — identisches Issue, identischer Fix | Befund 4 |
| **P14 — `loadSuggestions` HTTP-Status-Check via `safeJSON`** | Detail-View `loadSuggestions` parsed `r.json()` ohne 4xx/5xx-Branch und ohne Login-Redirect-HTML-Detection | **direkt** auf `toggleFavorite` / `deleteConversion` in `library.js` — `r.ok`-Check fehlt komplett, ebenfalls keine `safeJSON`-Login-Redirect-Detection; identischer Fix-Mechanik mit kombiniert P1/P3-Sub-Batch | Befund 2, 3 |
| **P15 — Banner-Mountpoint-Container im Template** (struktureller Vorbedingung-Fix) | Detail-View brauchte `#detail-alert-container` + `#notion-alert-container` als Banner-Mountpoints für `showAlert`-Calls | **direkt** — List-View hat **gar keinen** Banner-Mountpoint im Template; F6-IMPL muss einen einführen (z.B. `#library-alert-container` über der Filter-Bar) bevor P1/P3-Banner-Calls überhaupt rendern können | Befund 5 |

### Teil-übertragbar (Pattern existiert, Mechanik anders)

| F-3-Pattern | Beschreibung F-3 | Übertragbarkeit auf library List-View | Verweis |
|-------------|-------------------|----------------------------------------|---------|
| **P5 — Datum-Default lokal statt UTC** | Detail-View `formatDatetimeLocalNow`-Helper für Notion-`<input type="datetime-local">`-Pre-Population | **teil** — andere Stelle: List-View hat **Server-side strftime** für Card-Datum (#21), nicht JS-Pre-Population. Aber **selbe Klasse von Problem** (Lokalisierung). Lösung wäre Backend-Locale-Setting oder DE-Monatskürzel-Map im Jinja-Filter — **andere Mechanik**, nicht der `_utils.js`-Helper | Befund 11 |
| **P2 — Auto-Save Pending-State sichtbar machen** | Detail-View Title/Tags-Inputs mit `.c-input--dirty`-inset-shadow + Flush-on-Hide | **teil** — List-View hat **keinen** Title/Tags-Edit-Pfad; aber Favorite-Toggle (#19) hätte einen ähnlichen Pending-State-Anlass für die ms zwischen Klick und Server-Response. Aufwand und UX-Wert eher gering — Folde-Kandidat | F6-PATTERNS-Diskussion |

### Bereits-erfüllt (Pattern strukturell schon adressiert)

| F-3-Pattern | Beschreibung F-3 | Status auf library List-View | Verweis |
|-------------|-------------------|------------------------------|---------|
| **P9 — Tags-Input mit Chip-Visualisierung** | Detail-View `renderTagChips`-Pattern für CSV-Tags | **bereits-erfüllt** — List-View rendert Tag-Chips **schon** mit `.c-tag`-Klasse Server-side ([templates/library.html:67-71](templates/library.html#L67-L71)), siehe Element #22. Interessante Inversion: List-View ist hier voraus. | Element #22 |
| **P11 — Sidebar-Active-State** | F-3-IMPL hat den Sidebar-Active-State über `'/library' in request.path`-Match in `base.html:84` etabliert (deckt List + Detail beide ab) | **bereits-erfüllt** — selber path-Match macht List-View-Sidebar-Active korrekt ([templates/base.html:84](templates/base.html#L84)), siehe Element #7 | Element #7 |
| **P12 — File-Size mit KB/B-Fallback (Server-side)** | Detail-View nutzt `file_size`-Jinja-Filter ([app_pkg/__init__.py:114-123](app_pkg/__init__.py#L114-L123)) | **bereits-erfüllt zwar als Helper-Existenz**, aber **n/a für List-View Display** — Cards zeigen kein File-Size; Helper bleibt für Detail-View. Nicht-Anwendbar mangels Display-Anlass. | n/a |

### Nicht anwendbar (List-View hat das Feature nicht)

| F-3-Pattern | Begründung |
|-------------|-----------|
| **P4 — Notion-Form State-Preservation** | Keine Notion-Form auf List-View (Notion-Send ist Detail-only). |
| **P7 — Notion-Submit Persistent Error-Banner** | s.o. — kein Notion-Submit-Pfad. |
| **P10 — Notion-Toggle aria-expanded/aria-controls** | s.o. — kein Notion-Toggle-Disclosure. |
| **P13 — Page-`<title>` aktualisieren nach Title-Edit** | List-View hat keinen Title-Edit-Pfad; `<title>Library</title>` bleibt statisch. |

### List-spezifische Befunde ohne F-3-Korrespondent

- **Befund 1** — Copy-Btn kopiert nur 200-char-Preview statt Full-Content (Bug)
- **Befund 7** — Search-Input ohne Live-Search / Debouncing (UX-Inkonsistenz)
- **Befund 8** — Search-Input nicht auto-submit vs. Type/Favorites/Sort sind auto-submit
- **Befund 9** — Pagination-URL mit leerem `favorites=''`-Artifact
- **Befund 10** — Type-Filter Backend-Validation fehlt (akzeptiert beliebige Strings, liefert nur leere Liste)
- **Befund 11** — `%b`-Monatskürzel Locale-abhängig im Card-Datum
- **Befund 12** — Empty-State nicht filter-aware
- **Befund 13** — Per-Page-Size 20 hardcoded, kein UI-Toggle
- **Befund 14** — Keine Live-Region für Card-Remove-aria-Announcement
- **Befund 15** — Delete-Animation ohne aria-live

### Verteilung

- **direkt übertragbar**: 6 von 15 F-3-Patterns (P1, P3, P6, P8, P14, P15)
- **teil-übertragbar**: 2 von 15 F-3-Patterns (P2, P5)
- **bereits-erfüllt**: 3 von 15 F-3-Patterns (P9, P11, P12)
- **nicht anwendbar**: 4 von 15 F-3-Patterns (P4, P7, P10, P13)

→ **F-3-Cross-Feature-H4-Quote ~53%** (8 von 15 als direkt + teil; 11 von 15 als „relevant inkl. bereits-erfüllt" = 73%). Im erwarteten Master-Bereich „mittel ~30-50%" leicht überschritten — höher als F-4.2's 0% (kein Cross-Feature-Bezug), niedriger als F-5.2's 86% (Schwester-Feature mit Helper-Bestand) wie im Sprint-Prompt anti­zipiert. Die F-3.1-Memory-Erwartung „Helper-Reuse-Konvergenz auf etablierte Helper" bestätigt sich: 5 von 6 direkt-übertragbaren Patterns hängen am Helper-Reuse (`showAlert`/`safeJSON`/Toast-Level/Banner-Mountpoint), nur P6 ist reine Microcopy-Welle.

---

## Separate Befunde (nummeriert, nicht in Tabelle gemischt)

Auffälligkeiten, die **über fehlende States hinaus** gehen. Stage 2 (`F6-REVIEW`) entscheidet, welche Heuristik-Findings werden und welche separate Bug-Tickets:

1. **Copy-Btn kopiert nur 200-char-Preview statt Full-Content.** `copyContent(id)` liest `card.querySelector('.line-clamp-3').textContent` ([static/js/library.js:19](static/js/library.js#L19)) — das ist exakt der Server-clipped Inhalt aus `{{ conv.content[:200] }}...` ([templates/library.html:59](templates/library.html#L59)), nicht der DB-Full-Content. **User-Erwartung „Copy" = Full-Content**; die Card-Toolbar suggeriert keine Preview-Only-Semantik. Sev: mittel (funktionaler Bug; User merkt es erst beim Paste in Notion/Editor). Disposition: Finding + Bug-Ticket.

2. **`toggleFavorite` ohne Error-Handling / Loading-State / `safeJSON`.** ([static/js/library.js:3-15](static/js/library.js#L3-L15)) Bei `PUT /api/conversions/<id>` mit 4xx/5xx oder Network-Fail bleibt der Glyph alt, kein Toast, kein Banner. Bei Session-Expired (Login-Redirect-HTML) durch fehlende `safeJSON`-Detection könnte `r.ok===true` (302 → 200 HTML) den Glyph fälschlich togglen. Identische Familie wie F-3 Befund 2 (Detail-View) / Pattern P1+P14. Sev: mittel (Datenverlust-Risiko silent). Disposition: Finding + Bug-Ticket (Konvergenz mit F-3-Sub-Batch-A-Patches).

3. **`deleteConversion` ohne Error-Handling / Loading-State / `safeJSON`.** ([static/js/library.js:27-38](static/js/library.js#L27-L38)) Bei `DELETE /api/conversions/<id>` mit 4xx/5xx bleibt Card sichtbar, kein Hinweis; bei Session-Expired analog Befund 2 — Card würde aus dem DOM entfernt ohne DB-Delete. Identische Familie wie F-3 Befund 3 / Pattern P3. Sev: mittel. Disposition: Finding + Bug-Ticket.

4. **Toast-Level für `copyContent`-Failure-Pfad ist `success`.** ([static/js/library.js:23](static/js/library.js#L23)) `showToast('Copy failed')` ohne `opts.level='danger'` → grüner Erfolgs-Toast für Fehler. Identisches Issue wie F-3 Befund 13 / Pattern P8 (Toast-Level pro Call-Site). Sev: mittel (UX-Verwirrung). Disposition: Finding + Bug-Ticket.

5. **Kein Banner-Mountpoint-Container im List-View-Template für `showAlert`.** [templates/library.html](templates/library.html) hat keinen `#library-alert-container` o.ä. — Befunde 2/3 können `showAlert` nicht aufrufen, ohne erst einen Mountpoint im Template einzuführen. Identische Familie wie F-3 Pattern P15 (struktureller Vorbedingung-Fix). Sev: niedrig (Vorbedingung-Befund). Disposition: Finding mit Pattern-Verzahnung zu Befund 2/3.

6. **Sammel-Befund EN-Strings im List-View** (Sprint-Konstitutiv-Hinweis aus Sprint-Prompt — Disposition: „F6-PATTERNS DE-Microcopy-Folge", nicht hier fixen):
   - Filter-Optionen "All Types" / "Document" / "Audio" / "Dialogue" / "Markdown" ([templates/library.html:12-16](templates/library.html#L12-L16)) — gemeinsam mit F-3 Detail-View Type-Badge.
   - `title="Filter by type"` ([templates/library.html:11](templates/library.html#L11)).
   - Search-Placeholder "Search title, content, tags..." ([templates/library.html:21](templates/library.html#L21)).
   - "Favorites"-Label ([templates/library.html:26](templates/library.html#L26)).
   - Sort-Optionen "Newest" / "Oldest" / "Title A-Z" ([templates/library.html:29-31](templates/library.html#L29-L31)).
   - `title="Sort order"` ([templates/library.html:28](templates/library.html#L28)).
   - Search-Button "Search" ([templates/library.html:33](templates/library.html#L33)).
   - Type-Badge-Texte "Document" / "Audio" / "Dialogue" / "Markdown" ([templates/library.html:44-47](templates/library.html#L44-L47)) — überlappt mit F-3 #12, also einmal fixen genügt.
   - `title="Toggle favorite"` ([templates/library.html:50](templates/library.html#L50)).
   - Card-Button "Copy" + `title="Copy content"` ([templates/library.html:75](templates/library.html#L75)).
   - Card-Button "Delete" + `title="Delete"` ([templates/library.html:76](templates/library.html#L76)).
   - Pagination "Prev" / "Next" ([templates/library.html:88,101](templates/library.html#L88)).
   - Empty-State "No saved conversions yet" + Hinweis-Text ([templates/library.html:108-109](templates/library.html#L108-L109)).
   - JS-Toast-Strings "Content copied to clipboard" + "Copy failed" ([static/js/library.js:21,23](static/js/library.js#L21)).
   - JS-confirm-Text "Delete this conversion? This cannot be undone." ([static/js/library.js:28](static/js/library.js#L28)) — überlappt mit F-3 #17 Detail-View confirm-Text, einmal fixen genügt.
   - **Bemerkung zu BACKLOG-P3-Reminder „2 EN-Strings in `library.js`"**: das stimmt für JS-Toast/confirm; die Template-EN-Strings (Filter/Sort/Card-Btn/Pagination/Empty) sind ein zusätzlicher größerer Block. Sammel-Befund deckt beide ab.

7. **Search-Input ohne Live-Search / Debouncing.** ([templates/library.html:19-22](templates/library.html#L19-L22)) Sub-Befund von Befund 8 — keine Live-Filter-Mechanik. Reader-Ersatz-Daily-Usage (Memory-Notiz) wäre mit Live-Search performant da Pagination-Cap 20 ist. Sev: niedrig–mittel. Disposition: Finding (UX-Polish).

8. **UX-Inkonsistenz: Search-Input nicht auto-submit, alle anderen Filter sind auto-submit.** `<select name="type">` ([templates/library.html:11](templates/library.html#L11)), `<input type="checkbox" name="favorites">` ([templates/library.html:24](templates/library.html#L24)), `<select name="sort">` ([templates/library.html:28](templates/library.html#L28)) haben `onchange="this.form.submit()"`. `<input type="text" name="search">` ([templates/library.html:19-22](templates/library.html#L19-L22)) hat das nicht. User-Mental-Model "Filter ändern = Liste filtert sich" kollidiert für den Search-Input. Sev: niedrig (entdeckbar via Search-Btn). Disposition: Finding (UX-Konvention).

9. **Pagination-URL mit leerem `favorites=''`-String-Artifact.** ([templates/library.html:87,92,100](templates/library.html#L87)) `favorites='1' if current_favorites else ''` produziert bei deaktivierten Favorites einen leeren Query-Param `?...&favorites=&...` in der URL. Funktional unkritisch (`request.args.get('favorites', '') == '1'` → False), aber URL ist hässlich/uneinheitlich. Sev: kosmetisch. Disposition: Finding (Code-Hygiene).

10. **Type-Filter Backend-Validation fehlt.** [app_pkg/library.py:43-44](app_pkg/library.py#L43-L44) — `query.filter_by(conversion_type=conversion_type)` akzeptiert jeden String aus dem URL-Query; bei `?type=nonsense` → leere Liste statt 400 oder Fallback auf "All". `ALLOWED_CONVERSION_TYPES`-Set ist nur im POST-Pfad genutzt ([:91](app_pkg/library.py#L91)). Sev: niedrig (kein SQL-Injection-Risk dank SQLAlchemy-Parametrisierung; nur UX). Disposition: Finding.

11. **Card-Datum `%b`-Monatskürzel Locale-abhängig.** ([templates/library.html:63](templates/library.html#L63)) `conv.created_at.strftime('%d %b %Y, %H:%M')` — bei Docker-Container-Default-Locale `C` oder `C.UTF-8` liefert `%b` englische Kürzel. Container-Image enthält wahrscheinlich keine `de_DE`-Locale-Bundles standardmäßig. Master-Walkthrough-Verifikation: was sieht der User real im Browser („May" oder „Mai")? Sev: niedrig (kosmetisch); Disposition: Finding (zusammen mit P6 DE-Microcopy via expliziter Monatsnamen-Map oder Locale-Setting im Container).

12. **Empty-State nicht filter-aware.** ([templates/library.html:106-110](templates/library.html#L106-L110)) Wenn der User mit einem aktiven Filter („Audio" + „Favorites") keine Treffer hat, sieht er "No saved conversions yet" + "Use Save to Library on any converter page" — irreführend, weil er **schon** Conversions hat, nur eben nicht im aktuellen Filter. Sev: niedrig (UX-Polish). Disposition: Finding.

13. **Per-Page-Size 20 hardcoded.** ([app_pkg/library.py:39](app_pkg/library.py#L39)) Kein UI-Toggle für „mehr pro Seite zeigen". Bei großer Library (Reader-Ersatz für Readwise, möglicherweise mehrere hundert/tausend Einträge in der Zukunft) ein Pagination-Klick-Marathon. Sev: niedrig (Skalierungs-UX). Disposition: Finding (Welle nach Reader-Ersatz-Skalierung).

14. **Keine aria-live für Card-Remove.** ([static/js/library.js:30-36](static/js/library.js#L30-L36)) Delete-Animation entfernt die Card visuell, aber Screenreader bekommt keinen Hinweis. F-3-Welle hat aria-live-Region (`#notion-target-status`) für Notion-Target-Switches eingeführt — selbes Pattern wäre für Card-Remove anwendbar (z.B. `#library-action-status`-Region mit Microcopy „Eintrag gelöscht."). Sev: niedrig (a11y); Disposition: Finding (Verzahnung mit Befund 3 / P3-Konvergenz).

15. **Card-Hover-Lift Animation kollidiert visuell mit Favorite-Klick und Card-Link-Hover.** ([static/css/style.css:242-245](static/css/style.css#L242-L245)) `.c-card:hover { translateY(-2px) }` läuft während die Maus die Card überquert; sowohl Favorite-Btn (#19) als auch Card-Link (#20) als auch Copy/Delete-Buttons (#23/#24) liegen innerhalb der hover-bewegten Card. Klick-Hit-Robustheit ist Browser-abhängig (Pointer-Capture); ggf. visuelles Flackern bei schnellem Maus-Wechsel zwischen Cards. Live-Walkthrough-Lücke. Sev: kosmetisch. Disposition: Finding.

16. **Helper-Reuse-Beobachtung (analog F-3.1 Befund 18, Pflicht für Phase-2-Konsistenz):**
  - **Verwendet** in `library.js`:
    - `fallbackCopyText` ([static/js/library.js:20](static/js/library.js#L20))
    - `showToast` ([static/js/library.js:21,23](static/js/library.js#L21))
  - **Nicht verwendet, obwohl im Helper-Set verfügbar**:
    - `showAlert` — kein Banner-Mountpoint im Template (Befund 5 / P15-Verzahnung)
    - `safeJSON` — `toggleFavorite`/`deleteConversion` parsen `r.ok` ohne Login-Redirect-HTML-Detection (Befund 2/3)
    - `formatFileSize` — kein File-Size-Display auf den Cards (n/a)
    - `formatDatetimeLocalNow` — kein `datetime-local`-Input (n/a)
    - `confirmIfLong` — semantisch nicht passend für Card-Delete (n/a)
    - `loadViewState`/`saveViewState` — URL-Persistierung statt localStorage (n/a-mit-Begründung; siehe Helper-Reuse-Spuren-Sektion)
  - `file_size`-Jinja-Filter (Server-Helper aus F3-IMPL P12): **n/a auf List-View** — keine File-Size-Anzeige auf Card.
  - Beobachtung, **kein Empfehlungs-Vorschlag** (das kommt in F6-PATTERNS).

---

## Live-Walkthrough-Lücken

Dieser Sub-Thread hatte **keinen Browser-Access** (analog F-3.1 / F-5.1 Lehre — Code-only-Inventur ist 80–90% des Werts). Folgende States/Pfade sind aus Code abgeleitet, aber nicht live verifiziert:

- **Locale des `%b`-Monatskürzels** (Befund 11) — Docker-Container-Locale-Wert ist nicht aus Code ableitbar, Master-Verifikation per Walkthrough oder `docker exec ... locale`.
- **Card-Hover-Lift Animation vs. Favorite-/Delete-Klick-Hit-Robustheit** (Befund 15) — Browser-Default-Pointer-Capture macht das vermutlich unkritisch.
- **`.c-card:hover` Translate visuell ↔ Sub-Element-Klicks** — Pointer-Up auf bewegtem Sub-Element.
- **Browser-Verhalten beim Filter-Auto-Submit** (#11/#13/#14) — Reload-Latenz, Visual-Flash, Scroll-Position-Reset.
- **Empty-State vs. Pagination-Page-2-mit-Filter-ohne-Treffer** — was rendert, wenn User auf Page 3 ist und dann „Audio"-Filter aktiviert wird der nur 1 Page Treffer hat? Backend `error_out=False` ([app_pkg/library.py:64](app_pkg/library.py#L64)) liefert `pagination.items=[]` → Empty-State greift, aber URL bleibt `?page=3` (Live-Walkthrough). Vermutlich harmlos, aber nicht live verifiziert.
- **Reihenfolge URL-Query-Param-Persistierung beim Pagination-Klick mit aktivem Filter** — funktional aus Code OK, visuelle Konsistenz bleibt zu verifizieren.
- **Toast-Auto-Dismiss-Timing** (#29) — 2.5 s, sichtbar in jedem Erfolgs-/Fehler-Pfad.
- **`confirm()`-Dialog** bei Delete (#24/#30) — Browser-Default.
- **CSRF-Token-Handling für die API-Calls** in `toggleFavorite`/`deleteConversion` — `fetch` setzt keinen `X-CSRFToken`-Header. Wenn die App globale CSRF-Protection für `/api/*` aktiviert hat, würden diese Calls 400 sein. Aus dem Code-Reading der API-Routes ([app_pkg/library.py:111-137](app_pkg/library.py#L111-L137)) ist kein explizites `@csrf.exempt` zu sehen, aber Flask-WTF-CSRF könnte global per `WTF_CSRF_CHECK_DEFAULT=False` oder via Sub-Application-Config deaktiviert sein. **Out-of-Scope für Inventur** (gehört zu Sammel-Bug-Pass falls relevant), aber als Live-Walkthrough-Lücke notiert für künftigen Sicherheits-Sweep.

Master kann ggf. Walkthrough-Nachreichung in F6-REVIEW machen — die meisten dieser Lücken sind dort auswertungs-relevant, nicht inventur-relevant.

---

## Disposition-Übersicht (Befund 1–15, vorläufig — final in F6-REVIEW)

- **Finding + Bug-Ticket-Kandidat (4):** #1 Copy-Btn-Preview-Only, #2 toggleFavorite-silent-fail, #3 deleteConversion-silent-fail, #4 Toast-Level-Failure-Pfad.
- **Nur Finding (10):** #5 Banner-Mountpoint-Vorbedingung, #7 Search-Live-Search-Polish, #8 Search-vs-Filter-Auto-Submit-Inkonsistenz, #9 favorites=''-URL-Artifact, #10 Type-Filter-Validation, #11 %b-Locale, #12 Empty-State-filter-aware, #13 Per-Page-Size-hardcoded, #14 aria-live-Card-Remove, #15 Card-Hover-vs-Click.
- **Sammel-Befund (1):** #6 EN-Strings (Sprint-Konstitutiv aus Prompt — Disposition F6-PATTERNS DE-Microcopy-Folge).

**Akut-flag-Kandidaten** (Crash-Pfad / Datenverlust-Risiko): **keine**. Befund 1 (Copy-Preview) ist funktional-irreführend, aber kein Crash. Befunde 2/3 sind silent-fail-Risiken aber identisch zur F-3-Welle gehandhabt — kein Hot-Fix-Sprint nötig, F6-IMPL-Sub-Batch-A nach Pattern-Sprint reicht.

---

## Hinweis zu library_detail-Pfaden beim Code-Reading

Konstitutiv-Erwähnung aus Sprint-Prompt: Beim Code-Reading von `app_pkg/library.py` und `static/js/library.js` sind **keine neuen library_detail-spezifischen Auffälligkeiten** aufgefallen, die F-3 nicht schon abgedeckt hat. Die API-Routes (`api_*`) sind sauber gemeinsam genutzt; der Refactor `get_owned_conversion` ([app_pkg/library.py:19-27](app_pkg/library.py#L19-L27)) aus F-010 ist bestätigt aktiv. BT7 (textarea-escape in library_detail.js) und BT8 (window.open-noopener in library_detail.js) sind nicht in den List-View-Code-Pfaden und bleiben out-of-scope per Sprint-Prompt-Disziplin.

---

**Hinweis:** Diese Stufe-1-Datei enthält bewusst **keine** Pattern- oder Microcopy-Vorschläge oder Severity-Bewertungen. Diese folgen in Stufe 2 (Heuristik-Review, `F6-REVIEW`) und Stufe 3 (Patterns + Microcopy, `F6-PATTERNS`).
