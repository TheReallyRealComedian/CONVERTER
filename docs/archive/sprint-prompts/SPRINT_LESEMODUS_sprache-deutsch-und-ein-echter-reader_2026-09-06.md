# SPRINT LESEMODUS — Transkription spricht Deutsch, der Lesemodus wird ein echter Scope

**Größe**: M (3 Phasen) · **Datum**: 2026-09-06 · **Vorhaben**: Betrieb + UI

## Warum

Drei Items aus einer Oli-Meldung vom 2026-09-06, **nach Risiko sortiert, nicht nach Thema** (KLEINKRAM-Muster): zuerst das, was echte Aufträge und echtes Geld betrifft, dann der Defekt, dann die Aufwertung.

1. **TRANS-DE-DEFAULT** (XS) — der Transkriptions-Default steht auf Englisch. Oli diktiert deutsch.
2. **READER-SCOPE** (S/M) — im Lesemodus des Markdown→PDF-Konverters bleibt beim Umschalten ein Rand in der *anderen* Helligkeit stehen: „wenn man vom dunklen reader aufruft und hell macht, bleibt ein dunkler rand und vice versa".
3. **READER-STIL** (M) — der Lesemodus soll die PDF-Stile zulassen. Er tut es heute in **zwei** Hinsichten nicht: der Umschalter ist im Lesemodus unerreichbar, und in Dark stampft ein `!important`-Block die Farb- und Flächen-Identität jedes Stils platt.

## Gegroundeter Ist-Zustand (Master — nicht neu herleiten, alles am Code belegt 2026-09-06)

### Sprache

Der Default steht an **drei** Stellen, die miteinander übereinstimmen müssen, plus zwei Signatur-Defaults:

- [app_pkg/audio.py:261](../../../app_pkg/audio.py) — `language = request.form.get('language', 'en')` (der **Server**-Default; die Whitelist `ACCEPTED_TRANSCRIPTION_LANGUAGES = ('en', 'de')` bei :73 bleibt, Englisch bleibt wählbar).
- [static/js/audio_converter.js:65](../../../static/js/audio_converter.js) — `let selectedLanguage = 'en'` (der **Modul**-Default; er speist sowohl den Datei-Upload bei :569 als auch die **Live**-Transkription bei :244, die daraus die Deepgram-WS-URL baut).
- [templates/audio_converter.html:29–30](../../../templates/audio_converter.html) — die Klasse `lang-active` sitzt auf dem Knopf `data-lang="en"`, und Englisch steht als **erster** Knopf. Klasse und DOM-Reihenfolge sind zwei getrennte Signale.
- [services/deepgram_service.py:98](../../../services/deepgram_service.py) `load_keyterms(language='en')` und [:124](../../../services/deepgram_service.py) `transcribe_file(audio_data, language='en')` — beide Aufrufer reichen die Sprache **explizit** durch ([tasks.py:167](../../../tasks.py)).

Weiter belegt: `keyterms.json` trägt **14** deutsche und 10 englische Keyterms (`universal` ist leer) — der Umschwenk läuft nicht blind. **Kein** Test nagelt den Server-Default fest: der Helfer `_post` in [tests/test_transcriptions.py:56](../../../tests/test_transcriptions.py) übergibt `language='de'` bereits explizit; der Default ist heute ungeprüft.

⚠️ **Der Dedup-Schlüssel enthält die Sprache** (`_find_duplicate`, [app_pkg/audio.py:99](../../../app_pkg/audio.py)). Eine Datei, die schon als `en` transkribiert wurde, dedupt nach dem Wechsel **nicht** gegen einen `de`-Submit — das ist richtig so und kostet beim ersten Nachreichen einen echten Deepgram-Call. **Benennen, nicht verhindern.**

### Der Rand im Lesemodus

Der Lesemodus setzt `data-theme` auf `<html>` ([static/js/markdown_converter.js](../../../static/js/markdown_converter.js) `applyDarkMode`), das globale Theme steht als `data-global-theme` daneben ([templates/base.html:35](../../../templates/base.html)). **Der reader-eigene Scope ist heute kein Scope** — zwei belegte Ursachen, beide in [static/css/style.css](../../../static/css/style.css):

1. **`.main-container` behält sein `padding` und `background: var(--nm-bg)`** (Zeile 569–577). `--nm-bg` folgt dem **globalen** Theme. `.reader-mode .main-container::before { display: none }` (813) räumt zwar die Karte weg, aber nicht die eingefärbte Polsterung darum. Das ergibt in **beide** Richtungen einen Rand in der Fremd-Helligkeit.
2. **`[data-global-theme="dark"] .preview-pane { background: transparent }`** (2684) hat dieselbe Spezifität (0,2,0) wie `.reader-mode .preview-pane { background: var(--reader-bg, #ffffff) }` (817), steht aber **später** in der Datei und gewinnt. Bei globalem Dark + Reader-Hell wird die Fläche also durchsichtig und der dunkle App-Grund schlägt durch.

⚠️ **Das sind zwei Lesarten der Kaskade, keine Messung.** Ein dritter Kandidat liegt daneben und passt sogar besser auf Olis Wort „**zieht nach**": `.preview-iframe { background: #fff }` (690) wird von `.reader-mode .preview-iframe` (824) **nicht** überschrieben — und `renderIframe()` weist `srcdoc` bei **jedem Tastendruck** und bei jedem Treffer des `MutationObserver` auf `data-theme`/`data-global-theme` neu zu. Während des Neuaufbaus ist das Element weiß. In Dark ist das ein Weiß-Blitz, kein stehender Rand — **eine andere Mechanik, die derselbe Satz beschreibt.** Erst reproduzieren, dann zuordnen.

### Die Stile im Lesemodus

- Die Vorschau lädt den gewählten Stil bereits: `updateStyle()` holt `/static/css/pdf_styles/<theme>.css` und schiebt ihn in den iframe-`<style>`. **Es fehlt kein Lade-Weg.**
- **Der Umschalter ist im Lesemodus unerreichbar**: das `<select id="style_theme">` liegt in der `.editor-pane` ([templates/markdown_converter.html:87](../../../templates/markdown_converter.html)), und `.reader-mode .editor-pane { display: none !important }` (808) blendet sie aus. Um den Stil zu wechseln, muss man den Lesemodus verlassen.
- Die erreichbare Fläche ist das Aa-Popover, [templates/_partials/reader_aa.html](../../../templates/_partials/reader_aa.html) — es trägt bereits **konsumenten-spezifische Flags** (`reader_aa_dark`, `reader_aa_exit`), das Muster für eine markdown-only-Gruppe existiert also.
- ⚠️ **Das `<select>` bleibt Source-of-Truth**: es ist ein echtes Formularfeld, das der POST auf `/generate-pdf` mitschickt und [app_pkg/markdown.py:182](../../../app_pkg/markdown.py) mit `request.form.get('style_theme', 'default')` liest. Ein Popover-Regler muss das Feld **treiben** (und `updateStyle()` auslösen), nicht neben ihm herlaufen — sonst zeigt der Reader Bodoni und das PDF kommt als Default heraus.
- **Was Dark wirklich plattmacht** (gemessen 2026-09-06): `DARK_OVERRIDES_CSS` ([static/js/markdown_converter.js:225–252](../../../static/js/markdown_converter.js)) setzt `color`/`background`/`border-color` mit `!important` auf 18 Selektor-Gruppen unter `.pdf-page`. **`font-family`, `font-size`, `line-height`, `letter-spacing` fasst es nicht an — die Typografie aller drei Stile überlebt Dark heute.** Verloren geht **Farbe und Fläche**: `academic-latex.css` trägt 18 Farb-/Flächen-/Rahmen-Deklarationen (6 verschiedene Farben, im Kern #000/#333) · `newspaper-bodoni.css` 25 (Papierton `#f0eee8`, Linien `#bbb`/`#888`) · `default.css` **112**, davon der Löwenanteil die **GitHub-Pygments-Syntaxpalette** (#005cc5, #032f62, #d73a49, #22863a, #6f42c1 …). In Dark sind alle drei Papiere derselbe Ton, alle Linien dieselbe Deckweiß-Transparenz, und Code hat in allen drei dieselbe Farbe.

### Totes CSS in derselben Datei

`.preview-content-area` und `.preview-page` existieren **weder in `templates/` noch in `static/js/`** (grep 2026-09-06) — der READER-TABLE-Bericht hat das schon 2026-07-03 als Cleanup-Kandidat notiert (BACKLOG:112). Betroffen sind u.a. `[data-theme="dark"] .preview-content-area …` (style.css 1028–1125, ~100 Zeilen), `.reader-mode .preview-page` (841–853) und `[data-global-theme="dark"] .preview-page` (2700). Sie sind die **tote Zwillingsfassung** von `DARK_OVERRIDES_CSS` und machen jedes Lesen der Theme-Kaskade doppelt so teuer.

## Gesperrte Entscheidungen

1. **Reproduzieren vor reparieren.** Die zwei CSS-Ursachen oben sind aus der Kaskade **gelesen**, nicht gemessen. Der Beleg ist der berechnete Stil im laufenden Browser, in **beiden** Richtungen. Wer nach der Lesart repariert, ohne den Befund herzustellen, repariert vielleicht das Falsche und hat hinterher keinen Beleg.
2. ⚠️ **Der Rand wird durch Scoping behoben, nicht durch `!important`.** Genau so hat Zeile 2684 ihre Macht bekommen. Ein weiteres Gewicht auf den Stapel zu legen verschiebt das Problem auf den nächsten Sprint.
3. **Ein Item, ein Commit.** Die drei hängen nicht zusammen; ein Rückbau darf die anderen nicht mitnehmen.
4. **Keine neuen PDF-Vorlagen** (Oli, 2026-09-06 — er hat „erreichbar" und „nicht plattgestampft" gewählt, „neue Stile" ausdrücklich nicht). Vorlagen zu schreiben ist Autorenarbeit und ein eigener Zuschnitt.
5. **Dark ist Vorschau-only.** `generate_pdf` bleibt unangetastet — das erzeugte PDF ist ein helles Artefakt und bleibt es.
6. **Die Engine-Generation wird nicht angefasst** — kein Konvertierungs-Ergebnis ändert sich.

---

# Phase 1 — TRANS-DE-DEFAULT: Deutsch ist der Default

## 1.1 Die drei Defaults gemeinsam bewegen

Server-Default, Modul-Default und der optisch aktive Knopf müssen **danach wieder übereinstimmen** (heute tun sie es, auf `en`). Bewegt man nur zwei, lügt die Oberfläche darüber, was sie sendet. Die Knopf-Reihenfolge im Template gehört dazu: der Default steht vorn.

⚠️ Der **Live**-Tab hat keinen eigenen Schalter — er liest dasselbe `selectedLanguage`. **Prüfen, dass er mitzieht; keinen zweiten Umschalter bauen.**

## 1.2 Die zwei Signatur-Defaults entscheiden

`load_keyterms(language='en')` und `transcribe_file(..., language='en')` haben heute keinen Aufrufer, der sie braucht — beide reichen explizit durch. Entscheide **mit Begründung**, ob sie auf `de` wandern oder ganz verschwinden (Pflichtargument). Steuerung des Masters: ein impliziter Sprach-Default tief im Service ist genau die Klasse verhaltensbestimmender Library-Default, vor der die FSRS-`learning_steps`-Lehre in CLAUDE.md warnt — aber eine Signaturänderung ist ein Blast-Radius, also **erst die Aufrufer zählen, dann entscheiden.**

## 1.3 Der Beleg

Ein Test, der den **Server**-Default festnagelt: ein POST auf `/api/transcriptions` **ohne** `language`-Feld landet als `de`. Den gibt es heute nicht (der Test-Helfer übergibt die Sprache immer). Dazu ein Blick in den Browser: frisch geladene Seite → Deutsch ist aktiv, ein Datei-Submit trägt `de`.

**Im Bericht benennen**: dass eine bereits als `en` transkribierte Datei nach dem Wechsel **nicht** dedupt (Sprache ist Teil des Schlüssels) und der erste Neusubmit deshalb einen echten Deepgram-Call kostet.

## Stop
Test grün (Baseline **1018 + 1 Skip**). **Commit + Push** `fix(TRANS-DE-DEFAULT): Deutsch als Transkriptions-Default`. Dann warten.

---

# Phase 2 — READER-SCOPE: der Lesemodus färbt sich selbst

## 2.1 Den Befund herstellen

Vier Kombinationen, im echten Browser, am Markdown-Konverter: global hell × Reader hell/dunkel, global dunkel × Reader hell/dunkel. Halte für jede fest, **welches Element** die Fremdfarbe trägt — `getComputedStyle` auf `.main-container`, `.preview-pane`, `.preview-container`, `.preview-iframe` und `body` ist die Messung, der Screenshot ist die Illustration.

⚠️ **Trenne stehenden Rand von Blitzen.** Der `srcdoc`-Neuaufbau bei jedem Tastendruck und bei jedem Theme-Wechsel legt kurz die Element-Farbe des iframes frei (`background: #fff`, nie reader-überschrieben). Wenn Olis „zieht nach" das meint, ist die Abhilfe eine andere als beim Polsterungs-Rand — **beide prüfen, beide berichten.**

## 2.2 Den Scope echt machen

Die Eigenschaft, gegen die repariert wird, formuliert sich als Satz und ist im Browser nachmessbar: **ist der Lesemodus aktiv, wird kein Element zwischen `<body>` und dem iframe-Inhalt aus einem globalen Token gefärbt** — und umgekehrt wirkt der Reader-Scope nirgends, wo der Lesemodus aus ist.

Wie du das erreichst, ist deine Wahl (Reader-Tokens statt globaler auf den beteiligten Flächen · die kollidierende Regel scope-gerecht ziehen · die Polsterung im Lesemodus fallen lassen) — **aber nicht per `!important`** (gesperrte Entscheidung 2).

## 2.3 Die Nachbarn prüfen, bevor du sie anfasst

Der Library-Reader (`body.library-reader`, `.reader-view`) teilt sich `reader_settings.js` und den `readerPrefs`-Blob mit dieser Seite. **Miss, ob derselbe Rand dort auftritt.** Tritt er auf: benennen und **mit Begründung** entscheiden, ob er in diesen Sprint gehört. Tritt er nicht auf: sag warum — der Unterschied ist die Erklärung.

## 2.4 Totes CSS raus

`.preview-content-area` und `.preview-page` sind nach heutigem Stand tot. ⚠️ **Erst der caller-first-`grep` über das ganze Repo, dann das Delete** (Memory `reference_flow_retirement_shared_package`) — Templates, JS, Tests, Skripte, Docs. Was der grep nicht als tot belegt, bleibt stehen. Was fällt, macht die Theme-Kaskade für Phase 3 lesbar.

## 2.5 Der reproduzierbare Beleg

Bau den Smoke, den es für diese Seite noch nicht gibt — Vorbild [scripts/smoke_document_converter.py](../../../scripts/smoke_document_converter.py) (Playwright **im** Web-Container, Wegwerf-User, Laufanleitung im Docstring, Memory `reference_browser_smoke_in_app_container`). Er fährt die vier Theme-Kombinationen, schießt je einen Screenshot und **gibt die gemessenen Hintergrundfarben aus**, damit der Befund in der Konsole steht und nicht im Auge des Betrachters. Er trägt in Phase 3 die Stil-Prüfung mit.

⚠️ **`pytest` fängt kein CSS** (CLAUDE.md, *Test-Suite-Limit*) — dieser Smoke **ist** der Gate, nicht die Suite.

## Stop
Vier Kombinationen sauber, Farbwerte im Bericht, Smoke committed, `pytest tests/` grün. **Commit + Push** (Fix und Toten-CSS-Räumung getrennt). Dann warten.

---

# Phase 3 — READER-STIL: die Stile sind erreichbar und bleiben sie selbst

## 3.1 Der Umschalter wird im Lesemodus erreichbar

Ins Aa-Popover, hinter einem konsumenten-spezifischen Flag wie `reader_aa_dark` — der Library-Reader kennt keine PDF-Stile und darf die Gruppe nicht sehen. Die Stilliste kommt aus derselben Quelle wie das `<select>` (`themes` aus [app_pkg/markdown.py:145](../../../app_pkg/markdown.py), also `STYLE_DIR.glob('*.css')`); eine zweite, handgepflegte Liste im Template wäre die nächste stille Drift.

⚠️ **Das `<select>` bleibt Source-of-Truth** (Ist-Zustand oben): der Regler setzt dessen Wert und löst den bestehenden Renderweg aus. Der Beleg dafür ist nicht das Aussehen, sondern ein **erzeugtes PDF**: im Lesemodus Bodoni wählen, Lesemodus verlassen, PDF erstellen — es muss Bodoni sein.

Ob die Wahl in `readerPrefs` persistiert, entscheidest du **mit Begründung**. Steuerung des Masters: der Stil gehört zum Dokument, das entsteht, nicht zum Lesekomfort — und `readerPrefs` ist mit dem Library-Reader geteilt, der keinen Stil kennt. Microcopy-Hausregeln gelten (Buttons max 3 Wörter, keine Emojis bei Fehlern).

## 3.2 Dark stampft nicht mehr platt

Miss zuerst **je Stil**, was der Override konkret zerstört (der Ist-Zustand oben nennt die Zahlen, nicht die Stellen). Dann entscheide **mit dem Messwert** zwischen den benannten Kandidaten:

- **(a) Den Override auf das Nötige schrumpfen** — nur was Lesbarkeit rettet, alles andere sagt der Stil. Billig; die Frage ist, ob dann irgendwo Schwarz auf Dunkel steht.
- **(b) Eine Dunkel-Fassung je Stil** (`<name>.dark.css` neben der Vorlage, nur im Reader-Dark geladen). Der einzige Weg, auf dem ein Bodoni-Papier ein Bodoni-Papier bleiben kann — dafür trägt jede künftige Vorlage zwei Dateien.
- **(c) Dunkle Umgebung, helles Papier** — die Fläche bleibt Papier, weil das erzeugte PDF hell ist und eine dunkle Vorschau über das Artefakt lügt. ⚠️ Legitimes Ergebnis, **aber nur mit Olis Sign-off**: er hat ausdrücklich dunkel lesen wollen. Wenn deine Messung dorthin zeigt, **stopp und frag**, statt es zu bauen.

**Das Kriterium ist nicht verhandelbar**: nach dem Umbau sind die drei Stile in Dark **unterscheidbar, und das wird gezeigt, nicht behauptet** — drei Screenshots plus berechnete Werte für mindestens Papierfläche, eine Überschriftenfarbe, eine Tabellenlinie und (bei `default`) eine Syntaxfarbe.

## 3.3 Wrap

- **CLAUDE.md**: es gibt heute **keinen** Bullet zum Lesemodus des Markdown-Konverters (READER-ADJ steht nur im BACKLOG). Wenn dieser Sprint eine Regel setzt, die jemand sonst falsch neu herleitet — und „der Reader-Scope ist ein echter Scope, Fixes laufen nicht über `!important`" ist so eine —, gehört sie dorthin, kurz.
- **STATUS.md**, **BACKLOG.md** (⚠️ Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md`, exit 1 = sauber): die drei Items schließen, je mit Ergebnis. Was **nicht** repariert wurde, steht als Ergebnis da. „Neue PDF-Vorlagen" als benannte Möglichkeit hinterlassen, nicht als Versäumnis.
- **Memory** nur bei übertragbarer Lehre — „keine" ist ein legitimes Ergebnis. Ein Kandidat drängt sich auf: *ein Theme-Scope, der nur Tokens umdefiniert, ist kein Scope, solange eine gleich spezifische globale Regel später in der Datei steht.* Nach dem Schreiben mit `ls` prüfen, dass Datei und Index-Zeile zusammenpassen.
- **Im Bericht benennen**: die vier gemessenen Theme-Kombinationen vorher/nachher · welche der drei Rand-Ursachen sich bestätigt hat und welche nicht · was der caller-first-grep als tot belegt hat und was stehen blieb · welchen Dark-Kandidaten du gewählt hast und mit welcher Messung · dass das erzeugte PDF dem im Reader gewählten Stil entspricht.

## Nicht-Ziele

- **Keine** neuen PDF-Vorlagen (gesperrte Entscheidung 4).
- **Kein** Touch an `generate_pdf`, am PDF-Landscape-Pfad oder an `render_markdown_to_html` (letzteres speist auch EPUB/Kindle).
- **Kein** Umbau des Library-Readers über die Messung aus 2.3 hinaus.
- **Kein** JS-Test-Harness — die Stop-Bedingung `pytest tests/` bekommt keine Node-Abhängigkeit; der Browser-Smoke ist der Gate.
- **Kein** Anfassen der Engine-Generation, der Dokument-Pipeline oder des Lern-Layers.
- ⚠️ **Editiert wird nur auf dem Mac.** Die Mintbox ist Runtime — Deploy und Smoke ja, Arbeitsplatz nein, keine unversionierten Dateien zurücklassen; Wegwerf-User am Ende **strikt nach `user_id`** abräumen (die `api_token`-Tabelle trägt Olis iOS-Tokens).
