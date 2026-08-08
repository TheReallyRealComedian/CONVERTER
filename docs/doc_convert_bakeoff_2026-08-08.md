# Bake-off-Ergebnisse (DOC-BAKE) — 2026-08-08

**Feld**: 13 Kandidaten × 14 Korpus-Klassen, 113 gewertete Läufe. CPU/Cloud (P2): Eigenbau, gemini-3.6-flash nativ-PDF, docling 2.118.1, unstructured 0.18.32/0.24.1, pandoc 3.10.1, markitdown 0.1.7, trafilatura 2.2.0, tesseract 5.5.1/deu, textlayer-Nullpunkt. GPU auf der A2000 12 GB (P3): mineru 3.4.4 (vlm-engine), marker-pdf 2.0.0, dots.ocr 3B (vLLM 0.26).

**Methode**: drei Messebenen — (a) Gold 01/07/08 mit den verbindlichen Bewertungsregeln aus [\_UNSICHERHEITEN.md](../corpus/gold/_UNSICHERHEITEN.md) als implementierter Metrik, (b) Strukturkennzahlen gegen deterministische Referenzen, (c) 16 vergleichende LLM-Judge-Verdikte (Claude-Familie ≠ Gemini-Kandidat) mit Handstichproben. Rohdaten: [SUMMARY.md](../corpus/bakeoff/results/SUMMARY.md), Verdikte unter `corpus/bakeoff/results/_judge/*/verdict*.json`, jede Zahl reproduzierbar aus `corpus/bakeoff/results/`.

**Kein Entscheidungs-Doc.** Zahlen, Reihungen, Handel — die Entscheidung fällt woanders.

---

## 1. Die eine Frage: schlägt der 1.418-Zeilen-Eigenbau eine Standardpipeline?

**Auf drei Klassen ja — als einziger; auf vier anderen versagt er still und flächig; dazwischen ist er die rohe Textebene. Er ist bimodal, nicht schlecht.**

Die Zahlen dazu:

- **Score-identisch zur Textebene, wo seine Detektoren nichts sehen**: Auf 01.gold ist der Eigenbau bis auf die Nachkommastelle die rohe Textebene (f1 0,9149 / CER 0,0645 / Zellen 0,0 / Regel 2 0/11 — beide Kandidaten byte-gleich gescort). Die dvips-Pixellinien-Tabellen des Papers sind für alle fünf Detektoren unsichtbar, null Gemini-Calls.
- **Stille Flächenverluste**: 02 Recall **0,4894** — die Ranking-Seiten 17–31 kommen leer, ein Kontaktblock wird mitten im Wort zu ~20 Kopien zerschreddert („| im per | s | önlichen | Gesprä |"), Seite 56 fehlt ganz. 12 Recall 0,9841, aber eine Geistertabelle ersetzt kommentarlos die Seiten 40 und 70. 05: gedruckte Seite 86 fehlt komplett. Alles bei `warnings: []`.
- **Drei Klassensiege, die kein anderer Kandidat holt**:
  - **03 (Tabelle über Seitengrenze)**: einziger Kandidat im 13er-Feld mit EINER fortlaufenden Tabelle (gemini: inhaltlich fehlerfrei, aber 20 Fragmente; docling: verliert Zeilen und erfindet rekombinierte Namen). Das ist `multi_page.py`.
  - **06 (degradierter Scan, Seite 2 kopfüber)**: liest die 180°-Seite, SGB-Zitat ziffernexakt — wo gemini 3 Checkboxen und ein falsches Zitat erfindet, docling Zeichensalat liest und tesseract auf 22 Wörter kollabiert. Das ist Seitenklassifikation + Gemini-HIGH-Routing.
  - **13 (Mischdokument)**: Scan-Abdeckung + native Verlustfreiheit zusammen schlägt das ganze P2-Feld (im Gesamtvergleich nur von mineru übertroffen). Das ist das Routing.

**Was davon einzigartig ist**: die Multi-Page-Merge-Logik und das Seitenrouting. **Was davon Ballast ist**: die fünf Tabellen-Detektoren mit Konsens-Clustering — sie sind auf 01 blind, produzieren auf 02 leere Skelette und auf 12 Geistertabellen, und auf 04 gewinnt der Eigenbau *trotz* ihnen (Judge: 634/655 Zeilen korrekt, aber 12 zeichenverdoppelte Overlay-Zeilen „MMyyVViiddeeoo").

---

## 2. Fünf Befunde, die sonst zwischen den Tabellen verschwinden

**2.1 Das Schweigen als Muster.** Der gefährlichste Fehlermodus des Felds ist nicht der niedrige Score, sondern der hohe Score mit leiser Lücke: Eigenbau 12 mit Recall 0,9841 bei `warnings: []`, während zwei komplette Seiten durch eine Geistertabelle ersetzt sind — keine Kennzahl findet das, erst der Judge. Dasselbe Muster bei docling (verlorene Einzelzeilen + erfundene Namen in 03, „8.167" statt „S. 157" in 14), marker (unmarkierte Misch-Provenienz in 14: eine Seite still neu gelesen, 14 durchgereicht) und den Durchreichern auf 14 generell (keiner der 13 Kandidaten meldet die kaputte Ebene). *„Degradationssignal in der Antwort"* ist damit keine Komfort-Anforderung, sondern die schärfste des ganzen Vorhabens.

**2.2 Generative Decoder füllen auf, Pipelines bleiben wörtlich.** Auf den beschnittenen Zeitungsspalten von Klasse 13 schreiben exakt die generativen Decoder unsichtbaren Pseudo-Volltext über die Crop-Fragmente: Gemini (via Eigenbau-Seitenpfad, P2), dots.ocr und marker@18k (P3) — teils fast wortgleich dieselbe erfundene Fortsetzung. Die Pipeline-Kandidaten (docling, tesseract, textlayer) und mineru lassen die Fragmente stehen. Und der marker-8k→18k-Vergleich zeigt die Richtung: **mehr Decoder-Kontext erhöht die Auffüll-Neigung** (8k: Fragmente „U nehmen", 18k: „Unternehmen" + neu konfabulierter „*Andrew Barker*"). Für eine Zielfunktion „Treue zuerst" ist das die wichtigste Klasseneigenschaft des Felds — sie steht quer zu jedem Qualitäts-Score.

**2.3 Der Handel bei dots.ocr.** Bestes Gold-f1 des gesamten Felds auf der Paper-Seite (0,9853 / CER 0,031 / Zellen 0,905), auf 07.gold gleichauf mit gemini-medium (f1 0,963, Zellen 0,779), einziger Kandidat im 13er-Feld, der die kaputte 14er-Ebene an allen Bruchstellen korrekt neu liest, 52/52 Fußnoten — **bei 2,5 Seiten/Minute: der 280-Seiter braucht 112 Minuten** (mineru 13, docling 8, Eigenbau 3½). Praktisch heißt das auf der A2000: eine 20-Seiten-Konvertierung dauert ~8 Minuten statt ~1; ein Enquete-Bericht ist ein Feierabend-Job, kein Klick; und die Karte ist währenddessen zu ~100 % belegt. Als Decoder füllt dots zudem auf (2.2) und versteckte auf 06 eine Kerntabelle in einem 1,67-MB-Inline-PNG statt Text.

**2.4 Die unstructured-Drift ist beantwortet** (Backlog-Item nebenbei gemessen): 0.18.32 → 0.24.1 ist auf 08/10/11 **byte-identisch**, auf 09 repariert der Bump genau mehrabsätzige Tabellenzellen (14 Zahlwerte, die der Pin verliert) und abgeschnittene PASSIVA-Header — sonst nichts, bei +50 % Laufzeit / +37 % RSS. Kein Breaking. Der 62-%-Wortverlust auf 09A ist **Aufruf-Konvention, kein Parser-Limit**: `partition(strategy="fast")` ohne `include_slide_notes` verliert alle Speaker-Notes plus Kopf-/Fußzeilen — ein Ein-Zeilen-Hebel im Prod-Aufruf.

**2.5 Korpus-Korrektur Klasse 14.** Die README-Prämisse „jeden Umlaut verloren / mit englischem Modell erzeugt" ist falsifiziert: die eingebackene Ebene trägt 549 korrekte Umlaute und ist nur **sporadisch** kaputt (2× „Asthetik", ~13 Einzelbrüche — darunter inhaltlich: „8.167" statt „S. 157" als Zitatstelle). Die Klasse bleibt gültig (sporadischer Schaden ist *unsichtbarer* als flächiger), aber Messungen gegen sie müssen die Bruchstellen-Anker nutzen, nicht Umlaut-Zählung. Judge-Mandat wurde mid-flight korrigiert; die READMEs sind noch nicht angepasst (→ Backlog).

---

## 3. Ergebnis nach Klassen (Reihung = Judge-Verdikt bzw. Gold-Score; volle Zahlen in SUMMARY.md)

| Klasse | Reihung (beste zuerst) | Tragende Zahl |
|---|---|---|
| 01 Paper zweispaltig (Gold) | **dots 0,985** > gemini-med 0,981 > gemini 0,978 > marker 0,971 > docling 0,959 > mineru 0,955 > eigenbau = textlayer 0,915 | Zellen: mineru 0,916 > dots 0,905 > gemini 0,868 >> docling 0,361 > marker 0,235 > eigenbau 0,0; Regel 2 (Stellungen): nur gemini, dots, mineru 11/11 — alle Textebenen-Kandidaten 0/11 |
| 02 Guideline | gemini > docling > textlayer > eigenbau | gemini 640/640 wertgenau (Judge-Vollabgleich); eigenbau Recall 0,489 |
| 03 Tabelle über Seitengrenze | **eigenbau** > gemini > mineru/dots (o. Judge) > docling > textlayer | einziger mit 1 fortlaufender Tabelle; docling erfindet rekombinierte Namen |
| 04 Verbundene Zellen | gemini > eigenbau > docling > textlayer | gemini 655/655 wert-exakt mit echten rowspan/colspan (nach Fairness-Nachlauf) |
| 05 Scan sauber | CPU: gemini > tesseract > eigenbau > docling · GPU: **marker** > dots > mineru | Wortlaut: tesseract, gemini, alle drei GPU = 0 Wortfehler in Prüfabsätzen — „man zahlt für nichts"; Differenz nur im Fußnotenapparat (dots 52/52) |
| 06 Scan degradiert, S. 2 kopfüber | CPU: **eigenbau** > gemini > docling > tesseract · GPU: dots > mineru > marker | 180°-Seite: eigenbau/gemini/dots lesen sie, docling liest rückwärts, tesseract 22 Wörter gesamt; gemini erfindet 3 Checkboxen + falsches SGB-Zitat |
| 07 Formular-Scan (Gold) | gemini-med 0,979 > **dots 0,963** > gemini 0,963 > eigenbau 0,920 > marker 0,867 > docling 0,854 > mineru 0,790 > tesseract 0,780 | Regel 1 (Kopfzeile): mineru als Einziger via `colspan`; Zellen: gemini-med/dots 0,779 |
| 08 DOCX Fußnoten (Gold) | u-pin/u-neu 0,964 > docling 0,955 > pandoc 0,941 | Regel 3 (Bild+Fußnote+Link): **nur pandoc 4/4**, docling/unstructured 0/4; Zellen alle 0,963 |
| 09A PPTX mehrspaltig+Notes | **markitdown** > docling > u-neu > u-pin | markitdown Recall 1,0 inkl. Notes; unstructured-Verlust = Aufruf-Konvention (2.4) |
| 09B PPTX SmartArt | **markitdown** > docling > u-neu > u-pin | SmartArt verlieren **alle vier** (Klassen-These bestätigt); Notes nur markitdown |
| 10 HTML-Artikel | **trafilatura** > u-pin = u-neu | trafilatura: Fließtext lückenlos, <2 % Boilerplate — verliert aber Titel/Autor/Datum; unstructured: alles da + 31 % Boilerplate |
| 11 EML Zitatkette | u-pin = u-neu (byte-identisch) | Treue makellos; zitierte Mails werden zu 1×1-Zellen plattgedrückt |
| 12 Großes PDF 280 S. | gemini > docling > textlayer > eigenbau | gemini: einzige echte Tiefen-Hierarchie + `[^n]`-Fußnoten; eigenbau: Geistertabelle ersetzt still S. 40+70 |
| 13 Mischdokument | Gesamt: **mineru** > eigenbau > dots > gemini > tesseract > textlayer > marker > docling | mineru: einziger mit Text auf allen Scan-Stichproben UND Crop-Ehrlichkeit; Preis: Tweet-Loop |
| 14 OCR-Ebene kaputt | gemini > **dots** > mineru > tesseract > textlayer > eigenbau > docling > marker | bestanden nur, wer neu liest; kein einziger Kandidat **meldet** etwas |

---

## 4. Reihungen pro Format, mit dem Handel im Klartext

**PDF, nativ (01–04, 12).** `gemini-nativ (medium)` ist die breiteste Spitze: wertgenaue Tabellen (02: 640/640, 04: 655/655 mit echten Spans), einzige Tiefen-Hierarchie auf 12, Regel 2 komplett — Handel: 1,5 ct/Seite, ~11 S/min, sporadische Leer-Chunks (1/40, Wiederholung heilt), erfundene Kleinigkeiten unter Druck (Dateiname auf 12), und Cloud. `vlm-dots` erreicht auf der Gold-Stichprobe dieselbe Klasse (0,985) — Handel: Faktor 8 Zeit. `mineru` ist der beste lokale Allrounder fürs Tabellenwerk (Zellen 0,916) bei ~20 S/min — Handel: Fußnoten-Lecks, HTML-only-Tabellen. `docling` solide Mitte (0,89–0,98 Recall) — Handel: stille Einzelzeilen-Verluste, Hierarchie auf h2 geplättet, S. 280 fehlt. `eigenbau`: siegt auf 03, sonst Textebene oder stiller Flächenverlust. `marker`: gut auf sauberem Satz — Handel: reproduzierbare Konfabulation an schwachen Stellen.

**Scans (05–07, 13, 14).** Zwei Ligen: Wer nur OCRt (tesseract) oder der Textebene vertraut (textlayer, eigenbau-ohne-Routing, docling auf 14), scheitert an Rotation, Degradation oder kaputten Ebenen. Wer Seiten *ansieht* (gemini, dots, mineru, eigenbau-Routing), trägt die Klassen — mit dem 2.2-Handel: die generativen Seher füllen im Zweifel auf. Sauberer Scan (05) ist die Ausnahme: dort sind fast alle wortfehlerfrei, und tesseract ist in Sekunden fertig — „wer hier zahlt, zahlt für den Apparat, nicht den Wortlaut". Auf dem Formular (07.gold) gewinnt gemini-medium (0,979) vor dots; die verbundene Kopfzeile liefert allein mineru.

**DOCX (08).** unstructured (Pin wie neu) führt im Text (0,964), pandoc trägt als Einziger die komplette Bild-Fußnoten-Link-Kette (Regel 3 4/4) bei minimal schwächerem Text (0,941, „smart"-Typografie-Abweichungen). docling dazwischen. Handel: Wer Fußnoten + Bilder braucht, hat nur pandoc; wer den CONVERTER-Serializer behalten will, verliert genau die.

**PPTX (09).** markitdown gewinnt beide Decks klar (Recall 1,0/0,99, einziger mit Notes) — Handel: keine echten Tabellenstrukturen überall (Klebe-Nähte auf Folie 22, „EVA -186" in gerenderter Ansicht verdeckt). SmartArt verliert das gesamte Feld — wer SmartArt braucht, braucht die `diagrams/data*.xml`-Extraktion, die kein Kandidat hat (die Referenz-Extraktion dieses Bake-offs hat sie).

**HTML (10).** trafilatura, wenn Artikel-Extraktion gewollt ist (Boilerplate <2 %) — Handel: der komplette Artikelkopf (Dachzeile, Titel, Autor, Datum) fehlt ersatzlos. unstructured, wenn Vollständigkeit gewollt ist — Handel: 31 % Boilerplate und der Titel als H3 unter dem Site-H1.

**EML (11).** unstructured funktional (Treue makellos, Kette erkennbar) — Handel: zitierte Mails als 1×1-Tabellen-Brei, vierzeiliger Outlook-Kopf auf einer Zeile. Kein Kandidat des Felds behandelt E-Mail wirklich als E-Mail.

---

## 5. Kosten und Durchsatz

**Cloud-Kosten gesamt: 8,37 USD / 7,61 €** von 20 € Deckel (106 Calls, Preise gegen die offizielle Preisliste verifiziert: 1,50/7,50 USD je 1M In/Out inkl. Thinking; Thinking via `thinking_level=low` gedeckelt). gemini-nativ: 7,30 USD für 492 gemessene Seiten = **1,48 ct/Seite** (medium; enthält beide Fairness-Wiederholungen). Eigenbau-Scans (Gemini HIGH je Seite): 0,89 USD.

**Durchsatz auf dem 280-Seiter** (A2000 für die lokalen, Wall-Clock):

| | S/min | Zeit 280 S. | VRAM-Peak |
|---|---|---|---|
| eigenbau (lokal, Detektoren) | 83,6 | 3,4 min | – |
| docling (CPU!) | 34,1 | 8,2 min | – |
| mineru-vlm | 21,9 | 12,8 min | 6,5 GB |
| marker2 | 19,3 | 14,5 min | 8,0 GB |
| gemini-nativ | 10,7 | 26 min | – (3,36 USD) |
| vlm-dots | 2,5 | **110 min** | 9,9 GB |

**Loop-Raten** (06/07 = dokumentierte Punktlinien-Trigger): Im gesamten Feld looped **nur mineru**, 2 seiner 12 Läufe — 88 Kästchen-Symbole in einer Zelle (06) und 13× ein Tweet-Handle (13); Klasse 07 blieb bei **allen 13 Kandidaten** loop-frei. Die befürchtete zweistellige Loop-Rate der Dokument-VLMs trat auf diesem Korpus und dieser Modellgeneration nicht auf. (Detektor dafür umgebaut: Zell-Ebene statt Zeile, Gold-validiert — der ursprüngliche Judge-Befund „30×" war mit 88 sogar untertrieben.)

---

## 6. Was nicht messbar war

- **XLSX**: kein Belegexemplar im Korpus — die markitdown/docling-XLSX-Zeile der Sprint-Tabelle bleibt ungemessen.
- **PaddleOCR-VL**: nicht gefahren; dots.ocr passte nach OOM-Tuning in die 12 GB, der benannte Fallback wurde nie nötig. Die „PaddleOCR-VL vs. dots"-Frage ist offen.
- **Die 8k-marker-Rohtexte sind gelöscht** — die Scan-Klassen-Läufe gegen den zu kleinen surya-Kontext (Harness-Artefakt) wurden vor Erkenntnis ihrer Artefakt-Natur entfernt statt als `attempt*`-Sidecar erhalten; ihre Evidenz tragen nur die Judge-Zitate und `re_run_notiz` in den Verdikten. Benannte Lücke; die Sidecar-Disziplin galt ab da (gemini-01/-04-attempts sind erhalten).
- **Klasse-06/07-Loop-Trigger für ältere VLM-Generationen**: der Korpus misst die 2026er-Modelle; die publizierten zweistelligen Loop-Raten stammen aus anderen Stacks und sind hier weder bestätigt noch widerlegt — nur: *diese* drei Kandidaten loopen auf *diesem* Korpus (fast) nicht.
- **Judge-Unabhängigkeit hat eine bekannte Grenze**: Judge und Stichproben-Prüfer sind dieselbe Modellfamilie (Claude); dokumentierte Divergenzen Judge↔eigener Eindruck gab es zweimal (06-GPU-Ranking; 14-Prämisse), beide zugunsten des gründlicheren Blicks aufgelöst.
- **Formel-Enrichment für docling** (`do_formula_enrichment=True`, Register-Punkt D): nicht gefahren — CPU-Kostenpunkt in unbekannter Höhe, und die Regel-2-Messung war über die drei stellungs-fähigen Kandidaten bereits entschieden. Bleibt offen, falls docling in die engere Wahl kommt.

## 7. Reproduzierbarkeit

Harness + alle Scores + modellgestützte Outputs versioniert (`corpus/bakeoff/`), Kalibrierung in [KALIBRIERUNG.md](../corpus/bakeoff/KALIBRIERUNG.md), Gold-Selbsttest perfekt (f1 1,0/CER 0/Zellen 1,0 auf allen drei Gold-Dateien), Negativ-Tests bestanden. Vier Fairness-Korrekturen sind als Commits + `attempt*`-Sidecars dokumentiert (Gemini-Leer-Chunk transient; 32k-Output-Deckel 04; dots-Assemblierungs-Glob; surya-8k-Kontext). GPU-Rezepte unter `corpus/bakeoff/gpu/`.
