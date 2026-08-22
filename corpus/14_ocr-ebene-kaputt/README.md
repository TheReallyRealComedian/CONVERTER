# 14 — Scan mit vorhandener, **beschädigter** OCR-Textebene

**Was ist hier schwierig?** Dass es leicht aussieht — und dass der Schaden *sporadisch* ist.
Die Datei ist ein 600-dpi-Bildscan mit einer bereits eingebackenen OCR-Textebene. Für jeden
Konverter ist das eine Seite mit gesunder Textebene: CONVERTERs alter Klassifikator rechnete
Bildabdeckung 1,00 gegen Textdichte 8,20 und stufte **NATIVE** ein (Schwelle: Dichte < 0,5
für „scanned"), nahm `page.get_text()` und gab die Ebene unverändert weiter — **ohne jedes
Degradationssignal**. Der OCR-Pfad wurde nie betreten.

- Datei: `Dahlhaus – ETA Hoffmanns Beethoven-Kritik und die Ästhetik des Erhabenen.pdf`
- 15 Seiten, DE, 600 dpi; Textebene 5.684 Wörter
- Gemessen: Laplace-Varianz 8305 (scharf), Schieflage 0,0°, Bimodalität 0,91,
  Textdichte 8,20 → **NATIVE**

## ⚠️ Korrektur der Prämisse (DOC-BAKE 2026-08-08 §2.5, nachgemessen KLEINKRAM 2026-08-22)

Die erste Fassung dieser README behauptete, die Ebene sei „mit einem englischen Modell
erzeugt" und habe „jeden Umlaut verloren". **Beides ist falsch.** Nachgemessen an der Ebene
selbst (PyMuPDF `page.get_text()`, je Seite) gegen die unabhängige Bild-Lesung aus dem
Bake-off (`corpus/bakeoff/results/gemini-nativ/14/output.md`):

- **549 Umlaute/ß sind korrekt** — keine der 15 Seiten ist umlautfrei (S. 1: 3 · S. 2: 31 ·
  S. 3: 37 · S. 4: 38 · S. 5: 47 · S. 6: 43 · S. 7: 52 · S. 8: 41 · S. 9: 41 · S. 10: 29 ·
  S. 11: 37 · S. 12: 48 · S. 13: 50 · S. 14: 39 · S. 15: 13). `für` 17× korrekt, `fur` 0×;
  `über` 14× korrekt; `Ästhetik` 30× korrekt.
- Der Schaden ist **sporadisch im Kleinen und systematisch an einer Stelle**: die Ebene
  enthält genau **1** Großbuchstaben-`Ü`, das Bild trägt 12 verschiedene „Über…"-Wörter.
  Dazu kommen ~100 Einzelbrüche in 5.684 Wörtern (≈ 2 %) — genug, um Zitate zu verfälschen,
  zu wenig, um in einer Umlaut-Zählung aufzufallen.

Die Klasse bleibt gültig — **sporadischer Schaden ist unsichtbarer als flächiger** —, aber
wer gegen sie misst, misst **gegen die Anker unten**, nicht gegen eine Umlaut-Quote.

## Bruchstellen-Anker (Seite · Ebene → Druck)

**Inhaltlich folgenreich — die Zitatstelle (S. 15):**

| Seite | Ebene | Druck (am Bild geprüft) |
|---|---|---|
| 15 | `(Seifert,8.167)` | `(Seifert, S. 157)` — zwei Zeichenbrüche in einem Token, die Belegstelle ist falsch |

**Titelschnitt — die beiden sichtbaren Umlaut-Brüche:**

| Seite | Ebene | Druck |
|---|---|---|
| 2 | `Asthetik des Erhabenen` (Titelzeile, fett) | `Ästhetik des Erhabenen` |
| 14 | `Asthetik des Erhabenen` (Kolumnentitel) | `Ästhetik des Erhabenen` |
| 8 | `Bsthetik` | `Ästhetik` |

**Großbuchstaben-Ü — systematisch (13 Anker, 1 Treffer in der ganzen Ebene):**

| Seite | Ebene | Druck |
|---|---|---|
| 3 | `Uberlieferung` | `Überlieferung` |
| 4 | `Ubergänge` | `Übergänge` |
| 5 | `Obertragung` | `Übertragung` |
| 5, 7 | `Oberraschende` | `Überraschende` |
| 6 | `ffberwältigung` | `Überwältigung` |
| 7 | `Uberwältigenden` | `Überwältigenden` |
| 8 | `Uberein timmun` | `Übereinstimmung` |
| 11 | `ffberlegenheit` | `Überlegenheit` |
| 11 | `tfberltgenheit` | `Überlegenheit` |
| 12 | `Oberwältigung` | `Überwältigung` |
| 12 | `Ubertreibung` | `Übertreibung` |
| 12 | `Uberinterpretation` | `Überinterpretation` |

**Kleinbuchstaben-Umlaut — sporadisch (Beispiele):** S. 3 `Gefuhls`, `auspriigen`,
`Kilnste` · S. 5 `oberraschenden`, `tibergängeC` · S. 7 `ubergänge`, `Gefhhl` · S. 8
`ubertreibung`, `ober` (für `über`) · S. 9 `Gefihl` · S. 10 `ober` · S. 11 `uberlegenheit` ·
S. 13 `miisikalischen`.

**Sonstige Einzelbrüche (Beispiele, ~100 insgesamt):** `rn`→`m`-Klasse S. 4 `Hoffrnanns`,
S. 6 `Grundrnusters`, S. 14 `Syrnphoniesatzes`, `irn` · eingefügte Leerzeichen S. 3
`manife tiert`, S. 8 `hinwegt uschen`, S. 10 `Unendli hen` · angehängte Reste von
Anführungs-/Sonderzeichen S. 4 `Strukturu`, S. 7 `Großeu`, S. 8 `Weltu`, S. 10 `vermagU` ·
Dreher/Ersetzungen S. 2 `hin ugeben`, S. 3 `Theork`, `GeschZchte`, S. 6 `Iäßt`, `Beethoren`,
S. 8 `Schreckenfi`, S. 11 `uurde`, S. 13 `Pranz`, S. 15 `Hoffnlanns`, `Selbstgndigkeit`.

Reproduzierbar in einer Minute (Mac, PyMuPDF): Ebene je Seite extrahieren, Wörter der
Ebene gegen die Wortmenge der Gemini-Lesung prüfen, Trennungen am Zeilenende vorher
zusammenziehen (`-\s*\n\s*` → ``), Treffer mit einer Umlaut-Substitution (`U→Ü`, `O→Ü`,
`A→Ä`, `u→ü`, `ff→Ü`, `tf→Ü`, `ii→ü`, `ti→ü`, `B→Ä`) als Umlaut-Bruch zählen, den Rest als
sonstigen Bruch. Ergebnis: 18 Umlaut-Brüche (davon 13 Großbuchstaben-Ü), 113 sonstige
Tokens, die die Bild-Lesung nicht kennt (ein kleiner Teil davon sind Geminis eigene
Abweichungen, etwa Eigennamen wie `Vietor`).

## Was diese Klasse prüft

Nicht OCR-Qualität — die textebenen-freie Fassung derselben Seiten liegt dafür in
`../05_scan-sauber/`. Hier geht es um die Frage, die kein Kandidat des Bestands beantwortet:

**Erkennt der Konverter, dass eine vorhandene Textebene beschädigt ist — oder meldet er
es wenigstens?**

Ein Kandidat besteht diese Klasse, wenn er entweder (a) die beschädigte Ebene erkennt und
neu liest, oder (b) sie durchreicht, das aber **meldet**. Er fällt durch, wenn er sie
kommentarlos ausliefert. Gemessen wird an den Ankern oben — zuerst an `Seifert, S. 157`
(inhaltlich), dann an den Ü-Wörtern (systematisch), dann an `Asthetik` S. 2/14 (sichtbar).
Wer nur Umlaute zählt, sieht 549 richtige und bescheinigt der Ebene fälschlich Gesundheit.

Der Fall ist nicht konstruiert: gescannte Altbestände mit einmal drübergelaufener OCR sind
in Firmenablagen die Regel — und der Schaden ist dort typischerweise genau so: punktuell,
im Titelschnitt und bei Großbuchstaben, nicht flächig.

Das Tesseract-Gegenexperiment aus `00_KORPUS-UEBERSICHT.md` (`-l deu` → `Ästhetik` ✅,
`-l eng` → `Asthetik` ✗) belegt, dass das **Bild** gut genug ist — **nicht**, dass die
eingebackene Ebene mit einem englischen Modell entstand (dagegen sprechen 549 korrekte
Umlaute und 17× `für`).

**⚠️ Rechtehinweis:** wissenschaftlicher Aufsatz aus einem urheberrechtlich geschützten Band.
Für einen privaten OCR-Test unkritisch, aber kein gemeinfreies Werk.
