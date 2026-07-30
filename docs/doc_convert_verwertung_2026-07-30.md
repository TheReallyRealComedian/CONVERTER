# Verwertungs-Kette — wie aus der Recherche Sprints werden

**Datum**: 2026-07-30 · **Gilt für**: das Vorhaben „CONVERTER bietet Dokument-Konvertierung als API-Dienst an" · **Eingang**: [doc_convert_research_brief_2026-07-30.md](doc_convert_research_brief_2026-07-30.md) · **Grundlage**: [doc_convert_bestand_2026-07-30.md](doc_convert_bestand_2026-07-30.md)

> **Der Fehlermodus, gegen den diese Kette gebaut ist**: Ein umfangreicher Recherchebericht landet, fühlt sich wertvoll an, und niemand übersetzt ihn in Entscheidungen. Die zweite Variante ist schlimmer — er wird als Ganzes umgesetzt, weil er kompetent klingt.

---

## Grundsatz

**Der Bericht ist Daten, keine Anweisung.** Präzedenz ist vier Wochen alt: von den drei Befunden des Lern-Layer-Berichts vom 29.07. hielt einer der Prüfung stand, und der eigentliche Defekt stand nicht darin (Memory `feedback_verify_agent_diagnoses`). Ein Bericht über *fremde* Bibliotheken ist nicht vertrauenswürdiger als einer über den eigenen Code — er ist schwerer zu prüfen, weil die Gegenprobe eine Installation kostet statt eines `grep`.

Daraus folgt die Reihenfolge: **erst messen können, dann recherchieren verwerten.** Deshalb beginnt die Kette nicht mit dem Bericht.

---

## Stufe 0 — Sofort, unabhängig von der Recherche

Vier Arbeitspakete hängen an **keiner** offenen Frage. Sie laufen, während die Recherche läuft.

### 0.1 Referenz-Korpus (Oli, eine Stunde)

Zehn bis fünfzehn **echte** Dokumente, die die vier Klassen abdecken. Für jedes: warum es drin ist und was daran schwierig ist.

| # | Klasse | was es prüft |
|---|---|---|
| 1–3 | wissenschaftliches PDF | zweispaltig · Tabelle über Seitengrenze · Formeln |
| 4–5 | DOCX | Überschriftenhierarchie · Tabelle mit verbundenen Zellen |
| 6–7 | PPTX | Spalten-Layout · SmartArt/Prozessflow |
| 8–9 | gescanntes PDF | deutscher Fließtext · Tabelle im Scan |
| 10–11 | HTML/EML | Boilerplate · Zitatkette |
| 12–13 | Grenzfälle | 200-Seiter (Durchsatz) · Mischdokument nativ+Scan |

Dazu für **drei** davon eine handgeschriebene Soll-Fassung („so soll das Markdown aussehen"). Drei reichen — mehr ist Fleißarbeit ohne Zusatzerkenntnis, weniger trägt keine Metrik.

⚠️ **Vertraulichkeit**: Der Korpus darf kein betriebliches Material enthalten, solange die Weiche unten (W-1) nicht entschieden ist. Wenn ein Dokumenttyp nur betrieblich vorliegt, braucht es ein öffentliches Äquivalent.

### 0.2 Bake-off-Harness (Sprint, M)

Ein Werkzeug, das Kandidaten gegen den Korpus fährt und vergleichbar macht: Kandidat rein, pro Dokument Markdown plus Kennzahlen raus (Laufzeit, Speicher, Modell-Calls, Tokens, geschätzte Kosten), Diff gegen die Soll-Fassung. Metriken aus der Recherche übernehmen, nicht selbst erfinden — Block 1, Frage 2 des Auftrags liefert sie.

**Das ist das teuerste und wertvollste Stück Stufe 0.** Ohne es entscheiden wir nach Herstellerprosa; mit ihm entscheiden Zahlen an Olis Dokumenten. Und es überlebt die Entscheidung: es wird die Regressionsprüfung des Dienstes.

### 0.3 Die Blutungen stoppen (Sprint, S)

Fünf Befunde aus der Bestandsaufnahme sind **unabhängig von jeder Engine-Entscheidung** falsch und heute in Betrieb:

1. `"\n\n".join(el.text)` verwirft Element-Kategorie und `metadata.text_as_html` — der Nicht-PDF-Pfad produziert kein Markdown. **Höchster Ertrag pro Zeile im ganzen Vorhaben.**
2. Der globale Code-Fence-Sweep im Post-Processing entfernt **jeden echten Codeblock** des Dokuments.
3. `_embed_links` macht `str.replace` des Linktexts über das ganze Dokument — jedes weitere Vorkommen desselben Worts wird zum Link, auch in Tabellenzellen.
4. Der PDF-Service baut seinen genai-Client selbst und umgeht `TIMEOUT_GEMINI_SECONDS` — synchron, hinter `--workers 1`. Exakt der Fehlermodus, den NARR-TIMEOUT auf der Narration-Seite schon einmal behoben hat (Memory `reference_worker_sdk_per_call_deadline`).
5. Auf einer Seite mit Konsens-Tabelle wird der Nicht-Tabellen-Text nur eingesammelt, wenn mindestens eine Tabelle erfolgreich extrahiert wurde — schlägt die Extraktion fehl, **verschwindet die ganze Seite**.

Diese Fixes sind auch dann richtig, wenn `pdf_extraction` später komplett wegfällt: bis dahin läuft es in Prod, und Punkt 1 und 5 sind stiller Datenverlust.

### 0.4 API-Kontrakt (Sprint, M — Entwurf zuerst, Master)

Welcher Endpoint, welcher Vertrag, welche Auth, synchron oder Job, welche Fehlercodes, welches Antwort-Schema. **Das hängt an keiner Engine-Frage** — die Engine ist die austauschbare Innenseite.

Gesetzt aus dem Bestand, ohne Diskussion:
- **Eigener Token für die kostenverursachende Fläche**, unabhängig revozierbar. Die Begründung des `NARRATION_TOKEN` ist wörtlich die Anforderung hier (Memory `reference_narration_token_billing`).
- **JSON-Antwort statt Datei-Download.** Der heutige `text/markdown`-Attachment kann von keinem Dienst verarbeitet werden.
- **Teilerfolg ist ein 200 mit Fehlerliste, nicht ein 500.** Das Muster aus Agentsuite2s Parser, das Einzige, das der Bestand richtig hat.
- **Degradationssignal in der Antwort** — „diese Seite ist OCR-only", „diese Tabelle ist modellgeneriert", „hier wurde verworfen". Das fehlt in allen fünf Implementierungen und ist die Voraussetzung dafür, dass ein aufrufender Dienst dem Ergebnis überhaupt trauen kann.
- **Seitentrenner `\n\n---\n\n` mit Seitennummer.** Drei Implementierungen sind unabhängig auf den Trenner gekommen; die Nummer führt nur eine mit.

Offen und von der Recherche abhängig ist allein **Frage 13** — ob das Antwort-Schema ein etabliertes Zwischenformat übernimmt oder ein eigenes ist. Deshalb: Kontrakt entwerfen, diese eine Stelle als Platzhalter, Sprint startet erst nach der Antwort.

---

## Stufe 1 — Annahme-Gate

Wenn der Bericht landet: **jeder Befund einzeln in ein Register**, keine Sammelbewertung. Der Recherche-Auftrag erzwingt bereits die passende Ausgabeform (Behauptung / Beleg / Konfidenz / nachmessbar durch / Widerspruch) — das Register übernimmt sie und ergänzt eine Spalte.

| Status | Bedeutung | Folge |
|---|---|---|
| **bestätigt** | gegen Bestand oder Korpus geprüft, hält | geht in die Entscheidung ein |
| **widerlegt** | geprüft, hält nicht | fliegt raus, mit Begründung im Register |
| **zu messen** | entscheidungsrelevant, nicht aus Papier klärbar | geht in den Bake-off |
| **irrelevant** | wahr, aber ohne Folge für dieses Vorhaben | wird nicht weiterverfolgt |

Zwei Regeln:

- **Kein Befund mit Konfidenz „hoch" wird ungeprüft übernommen, wenn er eine Löschung rechtfertigt.** „Docling schlägt das Ensemble" ist die teuerste Aussage des ganzen Berichts — sie muss am Korpus gemessen werden, nicht auf einem Benchmark geglaubt.
- **Widersprüche zwischen Quellen sind ein Messauftrag, keine Abwägung.** Wenn zwei Benchmarks unterschiedliche Sieger nennen, entscheidet der Korpus.

**Master-Arbeit, ein Durchgang, kein Sprint.**

---

## Stufe 2 — Entscheidungs-Doc

Eine Seite. Gereihte Empfehlung je Entscheidungspunkt, der Handel im Klartext, und die Weichen als **gesperrte Entscheidungen** im Sinne der Working Practice — danach wird nicht neu diskutiert.

### Die Weichen, die Oli gehören

Keine davon kann die Recherche abnehmen. Zwei sind vorgelagert und **präjudizieren den Rest**:

**W-1 — Darf betriebliches Material durch eine externe Modell-API?**
Das Muncher-Korpus besteht aus GMP-Anlagenplanung und Gremienprotokollen. Optionen: (a) ja, unter den Zusicherungen, die die Recherche zutage fördert · (b) nein — betrieblich nur lokal, privat darf zu Gemini · (c) grundsätzlich nur lokal.
**Konsequenz**: (b) heißt zwei Profile mit zwei Backends und einer Trennung, die im Datenmodell durchgehalten werden muss — kein Konfigurationsschalter. (c) macht GPU-Passthrough zur Pflicht.

**W-2 — Ziehen wirklich alle vier Dienste um?**
Agentsuite2 hängt am firmeninternen Apollo-Gateway und steht nicht unter Olis alleiniger Hoheit. image_extracter löst mit Template-Regionen und QA-Queue ein *anderes* Problem als Dokument→Markdown. **Eine Vereinheitlichung über alle vier ist womöglich gar nicht wünschenswert** — das ist eine Zuschnitt-Entscheidung, keine technische.

Die übrigen, nach der Recherche zu entscheiden:

- **W-3** Wo läuft der schwere Teil — im CONVERTER-Image, in einem eigenen Extraktions-Container, oder gar nicht lokal?
- **W-4** Zielfunktion Treue *oder* Lesbarkeit — ein Default oder zwei Profile als API-Parameter? (Muncher fordert „NEVER paraphrase", CONVERTER will lesbares Lernmaterial. Ein Dienst kann nicht beides als Default haben.)
- **W-5** Darf ein Modell deterministisch extrahierten Text **ersetzen** oder nur **ergänzen**? Der Bestand zeigt beide Extreme und ihre Kosten.
- **W-6** Kanonische Ausgabe: Markdown-String oder Blockstruktur mit Provenienz, aus der Markdown gerendert wird? *(Nur Letzteres macht Seitennummern, Bild-Referenzen und ein Degradationsflag überhaupt möglich — es ist der Punkt, an dem alle fünf Bestandsimplementierungen scheitern.)*
- **W-7** Werden Bilder wirklich extrahiert, oder bleibt es bei Textbeschreibungen?
- **W-8** Kostendeckel je Auftrag — und was beim Überschreiten passiert.
- **W-9** Greenfield oder Sanierung: wird `pdf_extraction` abgeschaltet oder repariert?

---

## Stufe 3 — Sprint-Schnitt

Nach dem Entscheidungs-Doc, in dieser Reihenfolge. Sprints mit **·** sind unabhängig voneinander parallel dispatchbar.

| # | Sprint | Größe | hängt ab von |
|---|---|---|---|
| 1 · | Blutungen stoppen (0.3) | S | — |
| 2 · | Bake-off-Harness (0.2) | M | Korpus |
| 3 | Bake-off fahren, Register füllen | M | 2 + Recherche |
| 4 | API-Kontrakt + Auth + Job-Mechanik | M | W-6 |
| 5 | Engine einsetzen | L | 3 + W-3/W-9 |
| 6 · | Formate nachziehen (DOCX/PPTX/HTML/OCR) | L | 5 |
| 7 · | Ausgaberichtung anschließen (MD→PDF/EPUB) | M | 4 |
| 8 | Migration je Konsument, ein Sprint pro Dienst | je S/M | 5, W-2 |

**Die Entkopplung ist der Punkt**: Sprint 1, 2 und 4 brauchen die Recherche nicht. Wenn der Bericht in zwei Wochen kommt, steht bis dahin die Messfähigkeit *und* die Außenseite des Dienstes — und der Bericht entscheidet nur noch die Innenseite.

Jeder Migrations-Sprint bringt seinen **eigenen Abnahmetest aus dem Korpus** mit: das Dokument, das dieser Konsument tatsächlich schickt, muss mindestens so gut herauskommen wie heute. Kein Konsument zieht auf eine Verschlechterung um.

---

## Rollenverteilung

Unverändert nach Working Practice: **Master plant, gründet Behauptungen im Code, schreibt Sprint-Prompts, prüft jede Phase adversarial gegen das Diff. Sub-Threads führen aus. Master macht keine Feature-Edits.**

Neu für dieses Vorhaben:

- **Das Befund-Register ist ein Master-Artefakt** und lebt in `docs/`. Sub-Threads lesen daraus, schreiben nicht hinein.
- **Der Bake-off wird von einem Sub-Thread gefahren, aber vom Master abgenommen** — mit derselben Härte wie ein Code-Diff. Ein Messergebnis, das eine Löschung von 1.418 Zeilen rechtfertigen soll, wird gegengerechnet, nicht geglaubt.
- **Der Korpus ist Olis Beitrag.** Er ist die einzige Stelle in der ganzen Kette, die niemand sonst liefern kann.
