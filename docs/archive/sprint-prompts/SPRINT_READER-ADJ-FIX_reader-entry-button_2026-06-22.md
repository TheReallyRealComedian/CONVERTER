# Sprint READER-ADJ-FIX — Fixer Reader-Mode-Einstieg oben rechts (XS, 1 Phase)

> **Executor-Doc.** Eine Phase, dann **Stop + Bericht**. Pre-Flight: `pytest tests/` grün (Baseline **393**). Du committest selbst (Hash + push). Arbeitsverzeichnis `/Users/olivergluth/CODE/CONVERTER`. **Frontend-Fix** → Live-Smoke (dark+light, 0 Console-Errors) ist der Gate, nicht pytest.

## Warum

Der READER-ADJ-„Reader Mode"-Button sitzt in der `c-btn-row` oben in der Hauptspalte — bei einem schon gelesenen Doc öffnet die Seite aber an der **Resume-Scroll-Position** (R2-B) mitten im Text, und die Aktionsleiste ist dann rausgescrollt → der Einstieg ist real schwer erreichbar (Oli 2026-06-22). **Fix**: ein **fixer Einstiegs-Button oben rechts, direkt unter dem Theme-Toggle**, identische Optik.

## Verifizierte Code-Fakten (Master-gegroundet)

- **Theme-Toggle** [templates/base.html:42](templates/base.html) (`#theme-toggle.theme-toggle-btn`, global): CSS [static/css/style.css:2141](static/css/style.css) — `position:fixed; top:14px; right:16px; z-index:50; width/height:36px; border-radius:50%; background:var(--nm-bg); box-shadow:var(--nm-raised-sm); color:var(--nm-text-secondary)`; Hover `--nm-raised`+`scale(1.08)`, Active `--nm-pressed-sm`+`scale(.94)`; **`body.reader-active .theme-toggle-btn { display:none }`** (style.css:2178); plus eine `@media`-Regel (~2377) fürs responsive Verhalten.
- **Aa-Trigger** [static/css/style.css:882](static/css/style.css): identischer 36×36-Kreis, **gleiche Position `top:14/right:16`**, `z-index:1000`, `display:none` außer `body.reader-active` → erscheint nur im Reader-Mode (am selben Fleck, wo der Theme-Toggle dann weg ist). **→ kein Stacking-Konflikt mit dem neuen Button.**
- **Einstieg-JS existiert**: `toggleLibraryReader()` (window-exportiert, [static/js/library_detail.js](static/js/library_detail.js)).
- **Aktueller Einstieg** [templates/library_detail.html:39](templates/library_detail.html): `<button … onclick="toggleLibraryReader()">Reader Mode</button>` in der `c-btn-row` (Z.38, `library-reader-hide`) — **der wird ersetzt**.

## Aufgabe (1 Phase)

1. **Neuer fixer Einstiegs-Button** in [templates/library_detail.html](templates/library_detail.html) (eigenes Element in der Hauptspalte, fixed-positioniert — er lebt nur auf der Detailseite, also keine Page-Gating nötig): `<button type="button" class="library-reader-enter" onclick="toggleLibraryReader()" title="Reader Mode" aria-label="Reader Mode öffnen">…</button>` mit einem **reader-passenden Icon** (z.B. Feather `book-open` / Text-Linien — 18×18, `viewBox 0 0 24 24`, `stroke="currentColor"` `stroke-width="2"`, **gleicher Stil wie die `theme-icon`-SVGs**).
2. **CSS** (neben den anderen Reader-Regeln, token-driven, **kein Hardcode**): `.library-reader-enter` = **1:1 der `.theme-toggle-btn`-Shell** (fixed, 36×36 Kreis, `--nm-bg`/`--nm-raised-sm`/`--nm-text-secondary`, gleiche Hover/Active), nur **`top:58px`** (= 14 + 36 + 8 Gap, also direkt unter dem Theme-Toggle), `right:16px`, `z-index:50`. **Im Reader-Mode verstecken**: `body.reader-active .library-reader-enter, body.library-reader .library-reader-enter { display:none }` (drinnen reicht Aa + Esc/Exit). **Responsive-Parität**: die `@media`-Regel des Theme-Toggles (~style.css:2377) spiegeln, damit der Button auf schmalen Viewports mit-rückt und sich nicht beißt.
3. **Alten Einstieg raus**: den `Reader Mode`-Button aus der `c-btn-row` ([library_detail.html:39](templates/library_detail.html)) entfernen. Die übrigen Leisten-Buttons (Kopieren / Als .txt / Im Editor öffnen / An Kindle / Löschen) **bleiben**.
4. **Live-Smoke** (lokale Instanz, MacChrome **dark+light**, **0 Console-Errors**): Normalansicht → der neue Button steht **oben rechts direkt unter dem Theme-Toggle**, gleiche Optik, beide Themes token-korrekt; Klick → Reader-Mode an (Chrome weg, Theme-Toggle **und** der neue Button verschwinden, Aa-Trigger erscheint top-right); Esc/Exit → zurück, beide Eck-Buttons wieder da. Auf einem **langen, resume-gescrollten** Doc bestätigen, dass der Einstieg ohne Scrollen erreichbar ist. `node --check` der berührten JS (falls JS angefasst — i.d.R. nur Template/CSS).
5. **Wrap** (in derselben Phase): **STATUS.md** + **BACKLOG.md** — READER-ADJ-FIX ☑ done mit Hash (kurzer Eintrag, Muster wie die anderen Done-Items; **Bullet-Guard** `grep -nE '(- \*\*.*){2,}' BACKLOG.md`). **Keine Memory** nötig (reiner Placement-Tweak). Finaler `pytest tests/` (393, unverändert).

**Stop + Schluss-Bericht** — inkl. Deploy-Hinweis: reiner Frontend-/Template-/CSS-Touch, **keine Migration, kein Dep**; Mintbox `git pull` + `docker compose up -d --build` + Browser-Hard-Reload.

## Bewusst NICHT

- **Markdown-converter unverändert** (sein Reader-Einstieg sitzt sichtbar oben im Editor-Pane, nicht scroll-abhängig).
- **Kein** zweiter Einstieg behalten (der fixe Button **ersetzt** den Leisten-Button — kein Duplikat).
- **Kein** neues Verhalten am Reader-Mode selbst (nur der Einstiegspunkt wandert).
- **Kein** Backend-/Schema-Touch.

## Akzeptanz

- [ ] Fixer Reader-Mode-Button oben rechts **direkt unter dem Theme-Toggle**, **identische Optik** (token-driven, 36×36-Kreis), beide Themes
- [ ] Klick → `toggleLibraryReader()`; im Reader-Mode ist der Button (wie der Theme-Toggle) versteckt, Aa-Trigger erscheint
- [ ] Einstieg **ohne Scrollen** erreichbar (auf resume-gescrolltem Doc bestätigt)
- [ ] alter `c-btn-row`-„Reader Mode"-Button entfernt; übrige Leisten-Buttons unberührt
- [ ] Live-Smoke dark+light, 0 Console-Errors; `pytest` 393 unverändert
