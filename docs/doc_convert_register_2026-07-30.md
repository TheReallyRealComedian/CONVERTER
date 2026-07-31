# Befund-Register — Cowork-Report nach dem Annahme-Gate

**Datum**: 2026-07-30 · **Eingang**: `~/Downloads/doc_convert_stand_der_technik_2026-07-30.md` (62 Befunde B-01…B-62) · **Gate**: 6 adversariale Skeptiker mit Websuche auf die tragenden Behauptungen, plus zwei Selbstprüfungen des Masters gegen Prod · **Verfahren**: [doc_convert_verwertung_2026-07-30.md](doc_convert_verwertung_2026-07-30.md), Stufe 1

> **Ergebnis in einem Satz**: Der Report ist handwerklich gut und richtungsstabil, aber **sechs von sechs geprüften Kernbehauptungen kamen als „teilweise" zurück** — und die drei folgenreichsten Negativurteile drehen sich. Hätte man ihn übernommen, wären drei Werkzeuge zu Unrecht ausgeschlossen worden.

---

## A. Selbst geprüft (Master, gegen Prod)

### R-01 · `gemini-2.0-flash` ist tot — Prod-Pfad gebrochen ✅ **bestätigt**

Report B-53 behauptete die Abschaltung zum 01.06.2026. **Live gegen die Prod-Instanz verifiziert:**

```
404 NOT_FOUND — This model models/gemini-2.0-flash is no longer available.
```

[service.py:28](../services/pdf_extraction/service.py:28) pinnt `VISION_MODEL = "gemini-2.0-flash"`. Die Fehlerkette ist vollständig belegt: der 404 ist **nicht** retrybar ([service.py:446](../services/pdf_extraction/service.py:446) prüft nur `429`/`rate`/`resource`) → `raise` → in `_extract_scanned_page` ([service.py:341](../services/pdf_extraction/service.py:341)) gefangen → Fallback auf `page.get_text()` → bei gescannten Seiten leer. Einziges Signal ist eine `WARNING`-Zeile.

**Tatsächlicher Schaden bisher: minimal.** Seit dem 01.06. existiert genau **eine** Dokument-Konvertierung in der Library (id 17, 05.06., „BlueprintDSP_Tool-Kit"). Die ging nicht durch den PDF-Pfad: ein einziger Block, keine Seitentrenner, 21.790 Zeichen, **null Pipe-Tabellenzeilen** — der `el.text`-Pfad. Der Container-Log taugt nicht als Gegenprobe, er wird bei jedem Deploy neu angelegt.

**Fix** (aus R-02 präzisiert): Nachfolger ist **`gemini-3.6-flash`**. Ausdrücklich **nicht** `gemini-2.5-flash` — das wird am **16.10.2026** ebenfalls abgeschaltet und landet dann bei demselben Nachfolger. Ein Wechsel auf 2.5 wäre in elf Wochen wieder fällig.

### R-02 · CONVERTERs Bestand ist überwiegend betriebliches Material ✅ **bestätigt**

Relevant, weil Oli entschieden hat, dass betriebliches Material lokal bleibt (W-1). Titel aus der Prod-Library: *Vortragsmanuskript — Die BI-Pipeline in vier Stufen · Die Kontrollstrategie im CMC-Kontext · Vorgespräch Joachim Bär — PPM/Kompetenzabgrenzung · TFA Analytical Technologies — OPS-Synthese · S&T-Abgleich*. Mit Klarnamen von Kollegen.

**Folge**: Der lokale Pfad ist für CONVERTER nicht die Ausbaustufe, sondern der Hauptpfad — und der Report benennt selbst, dass es dafür kein Qualitätsargument gibt („die 12-GB-Klasse liegt auf Benchmarks unter der Gemini-Klasse").

**Zweiter Befund derselben Abfrage**: Die Library besteht aus 57 `markdown_input`, 36 `ai_newsletter`, 19 `audio_transcription`, 2 `audio_narration` und **1** `document_to_markdown`. Der Dokument-Konverter ist faktisch ungenutzt; Inhalte kommen als agent-geschriebenes Markdown herein und umgehen ihn.

---

## B. Adversarial geprüft — alle sechs „teilweise"

### R-03 · B-03 · Docling-Treue — **Ausschluss trägt nicht**

Die *Rangordnung* (Docling schwächer als Marker/MinerU) hält und wird sogar von Doclings **eigenen** publizierten Zahlen gestützt (OmniDocBench markdown-F1 0,44). Die *Begründung* des Reports trägt nicht:

- **50,3 % existiert ausschließlich in Datalabs eigenem Harness** — Datalab ist der Hersteller des Konkurrenten Marker. Weder Ai2 noch das IDP-Leaderboard führen Docling auf olmOCR-Bench. Keine Reproduktion durch Dritte.
- **Gemessen wurde Docling-Default, nicht Docling-sinnvoll**: `PdfPipelineOptions()` mit `do_formula_enrichment=False`, bei einem Benchmark, dessen Makro-Mittel zwei mathematiklastige Splits enthält. Dazu asymmetrische Nachbearbeitung — Markers Ausgabe läuft durch `postprocess.py` für den Literal-Match-Checker, Doclings nicht.
- **0,119 ist keine Eigenschaft, sondern eine Einzelmessung auf hart gesampeltem Material.** Dieselbe Tabellenzeile zeigt **0,999** auf der Baseline-Kategorie. Korrekt gelesen: Docling bricht auf Handschrift, Formularen und Mehrspaltigem zusammen und ist auf normalem Drucksatz praktisch fehlerfrei.
- **„IBM publiziert keine Qualitätszahlen" ist nachweisbar falsch** — `docling-eval` publiziert F1, TEDS, Layout-mAP und Reading-Order auf fünf Benchmarks.

**Entscheidungsfolge**: Für CONVERTERs Korridor — born-digital, sauberer Drucksatz — steht Docling bei **64,0** bzw. **0,999**, nicht bei 50,3 bzw. 0,119. **Ein Ausschluss von Docling darf auf diesen Zahlen nicht ruhen.**

### R-04 · B-52/B-53 · Gemini-PDF nativ und Preis — **Kostenprämisse zu optimistisch**

Alle vier Teilaussagen haben einen wörtlichen Beleg, aber:

- **(b), (c), (d) sind Gemini-3-Eigenschaften**, nicht allgemeine Gemini-Eigenschaften — die Doku setzt sie explizit unter „Gemini 3 models".
- **„Wird nicht berechnet" ≠ gratis.** Frei sind nur die Tokens des *extrahierten Textlayers*; die **280/560/1120 Bildtokens pro Seite** fallen unabhängig an und sind der dominante Posten. 1.000 Seiten sind im Default **560k Input-Tokens**. Der Textlayer belegt außerdem Kontextfenster, er wird nur nicht fakturiert.
- **(d) ist gehedged und teilweise selbst-widerlegt** — „typically saturates" für „standard documents", während derselbe Anbieter im Migrationsleitfaden für „dense document parsing" ausdrücklich `media_resolution_high` zu testen empfiehlt.
- Stolperstein in der Quelle: die Seite trägt im Einleitungsabsatz weiter die veraltete 2.x-Zahl „258 tokens" neben der 3er-Tabelle mit 560 — wer überfliegt, rechnet um Faktor 2,2 daneben.

**Der Hebel, den der Report nicht gezogen hat**: Für Textdokumente `media_resolution` **explizit auf `low`** setzen — der native Textlayer geht ungekürzt und unberechnet mit.

> **Nachtrag 2026-07-30, an einem echten Call gemessen (DOC-FIX P1)**: Der Gewinn ist **4×, nicht 2×** — `low` = **266** Bildtokens, Default = **1092** (Prompt-Kosten der Seite 483 statt 1309). Der gemessene Default liegt damit an der dokumentierten **HIGH**-Stufe (1120), nicht an MEDIUM (560). Für `gemini-3.6-flash` ist der Default also offenbar HIGH; die Doku-Angabe „Default 560" aus B-52 gilt dafür **nicht**. Zwei Folgen: der Spar-Hebel ist doppelt so groß wie angenommen, und die Kostenrechnung „1.000 Seiten = 560k Input-Tokens" ist im Default eher **1,1M** — mit `low` dagegen ~266k.

### R-05 · B-24 · Doclings Deutsch-OCR-Falle — **überholt, Schlussfolgerung kippt**

Der Kern stimmte bis Mitte Juni 2026 und ist zum Berichtsdatum **falsch**:

- **Issue #2927 ist seit 16.06.2026 geschlossen**, Fix in **v2.103.0** (17.06.). Der Report beschreibt einen Stand von Januar–Mai als Gegenwart.
- **Seit v2.109.0 (03.07.2026) sind die Defaults PP-OCRv6**, und `de` ist dort ein eigener unterstützter Sprachcode.
- Die Zuschreibung „laut PaddleOCR-Maintainer" ist falsch — die Fehlbeispiele sind Nutzerbeobachtungen aus einer im Juni 2025 geschlossenen Issue, in der ein anderer Nutzer die Kategorik mit korrekt erkannten Umlauten bei 300 dpi direkt widerlegte. Die Belegstelle für „Spaß"→„SpafS" existiert in den docling-Issues **nicht**.
- Geblieben ist nur: im Default `--ocr-engine auto` wird die Sprache **bewusst nicht durchgereicht**. Aus einem unbehobenen Bug ist eine schlechte Voreinstellung mit Ausweg geworden.

**Entscheidungsfolge**: Docling ist für deutsches Material brauchbar — **aber nur mit explizit gesetzter Engine, nie auf `auto`**. Reihenfolge: `--ocr-engine rapidocr --ocr-lang de` (Mindestversion **v2.109.0**, im Container pinnen) · `--ocr-engine tesseract --ocr-lang deu` als robuster Fallback · EasyOCR **nicht** (seit v2.56.0 deprecated).

⚠️ Das ist exakt die Klasse Wirkungslücke, die im Repo schon dokumentiert ist: *Deploy grün ≠ wirksam* (Memory `feedback_verify_feature_reachable_under_user_config`). Engine- und Sprachwahl gehören als **expliziter Aufrufparameter** in den Code.

### R-06 · B-15/B-59 · camelot und die AGPL-Kaskade — **Kriterium streichen**

Harte Fakten stimmen (v2.0.0 am 04.06.2026, MIT, pdfium als Default-Backend seit v1.0.0), aber: die Wiederbelebung fand **2024** statt, nicht 2026; „Ghostscript-Zwang" ist überzeichnet (poppler war seit v0.10.0/2021 Alternative, Ghostscript betraf ohnehin nur `flavor=lattice`); und camelot war durchgehend MIT — es gab nie eine Lizenz-Vererbung.

**Entscheidungsfolge**: **Die AGPL-Frage als Auswahlkriterium streichen.** Sie war nie ein Vererbungsproblem und ist für eine LAN-only-Single-User-App ohne Distribution gegenstandslos. Das echte Argument für pdfium ist Deployment, nicht Recht.

### R-07 · B-18 · unstructured-„Herabstufung" — **kein Migrationsgrund**

Die Zitate existieren wortgenau, aber **es gibt kein Ereignis**: dieselben Formulierungen stehen seit **Mai 2024** unverändert in den Docs. Es ist eine Vergleichsliste gegen das eigene Bezahlprodukt, inklusive reiner SaaS-Lücken wie SOC2 und ETL-Scheduling. Und jeder technisch harte Punkt (VLM, fine-tuned OCR, Bildextraktion, Hierarchie-Erkennung) betrifft den **hi_res/OCR-Pfad, den CONVERTER nicht fährt** — bei `strategy="fast"` auf docx/pptx läuft überhaupt keine Modell-Inferenz. „Kein GPU-Support" ist für diese Installation ein Nullsatz.

**Entscheidungsfolge**: Kein Grund, von unstructured wegzugehen. **Was bleibt, ist ein ganz anderes, echtes Item: Versions-Drift.** Gepinnt ist 0.18.32 (10.02.2026), aktuell ist **0.24.1** (11.07.2026) — sechs Minor-Sprünge in fünfeinhalb Monaten, Breaking Changes wahrscheinlich. Changelog lesen plus Container-Extraktions-Smoke (docx/pptx), kein Ersetzungs-Sprint.

### R-08 · B-32/B-60 · Lizenzlage der Modelle — **eine Empfehlung dreht sich**

Vier von sechs Angaben halten. Aber:

- **Der MinerU-Teilsatz ist sachlich falsch**: der von MinerU 3.1.0 ausgelieferte VLM ist **Apache-2.0**, nicht AGPL. Code = Apache-basierte MinerU-Lizenz mit Schwellen (100 Mio. MAU / 20 Mio. $ Monatsumsatz), die für eine Single-User-LAN-App unerreichbar sind; die Nennungspflicht greift nur bei Online-Diensten.
- „dots.ocr = MIT" unterschlägt einen Zusatzvertrag mit Nutzungsbeschränkungen.
- Die Datalab-Nicht-Wettbewerbs-Klausel ist **breiter** als „gegen die Datalab-API".
- Nanonets-OCR2-3B ist nicht „research-only", sondern **lizenz-los**.

**Entscheidungsfolge**: **MinerU darf nicht mehr wegen Lizenz abgewertet werden.** Umgekehrt bleibt der Datalab-Stack (marker/surya/chandra) der einzige mit echten Einschränkungen.

---

## C. Was das Gate netto verändert hat

| Report sagte | Register sagt |
|---|---|
| Docling als Treue-Engine raus (50,3 / 0,119) | Zahlen unbrauchbar; im relevanten Korridor 64,0 / 0,999 — **bleibt im Rennen** |
| Docling für Deutsch fraglich (offener Bug) | **Bug seit 16.06. behoben**; brauchbar ab v2.109.0 mit expliziter Engine |
| unstructured herabgestuft → weg damit | Zwei Jahre alter Marketingtext, falscher Codepfad — **bleibt**; echtes Item ist Versions-Drift |
| MinerU-VLM ist AGPL | **Apache-2.0** — Lizenz ist kein Abwertungsgrund |
| AGPL-Kaskade camelot→Ghostscript | Nie ein Vererbungsproblem — **Kriterium streichen** |
| Gemini-PDF: nativer Text gratis | Nur Gemini-3; **Bildtokens dominieren**, 1.000 Seiten = 560k Input-Tokens |
| — (nicht gezogen) | **`media_resolution=low`** halbiert die Kosten für Textdokumente |
| `gemini-2.5-flash` als Ausweg | **`gemini-3.6-flash`** — 2.5 stirbt am 16.10.2026 |

**Muster**: Der Report hat in jedem der sechs Fälle eine *reale* Sache gefunden und sie **zu weit formuliert** — Hersteller-Marketing als Neuigkeit gelesen, Einzelmessungen als Eigenschaften verkauft, Datumsstände nicht nachgezogen. Das ist kein schlechter Report; es ist der normale Abstand zwischen „belegt" und „entscheidungstragfähig". Genau dafür existiert diese Stufe.

---

## D. Offen — geht in den Bake-off

Unverändert gültig aus Teil 3 des Reports, ergänzt um das, was das Gate offengelassen hat:

- **Es gibt keinen deutschen Dokument→Markdown-Benchmark.** Der beste Proxy ist Französisch. Der eigene Goldstandard bleibt die einzige Entscheidungsgrundlage.
- **Keine publizierte Durchsatzmessung eines Dokument-VLM auf einer A2000.**
- **Ensemble vs. gelerntes Modell** wurde nie direkt gemessen — der Eigenbau muss als Kandidat in den eigenen Bake-off, sonst bleiben die 1.418 Zeilen unbewertbar.
- **DPI→Genauigkeits-Kurve** existiert nicht; Gegenprobe `low`/`medium`/`high` mit `usage_metadata` an einem echten Fach-PDF.
- **Docling mit sinnvoller Konfiguration** (`do_formula_enrichment=True`, TableFormer ACCURATE, explizite OCR-Engine) ist nirgends gemessen — nur der Default.
