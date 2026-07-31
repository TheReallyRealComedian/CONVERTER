# Dokument→Markdown — Stand der Technik 2026: Befund-Register

**Recherche für**: CONVERTER-Master · **Stand**: 30.07.2026 · **Auftrag**: Architektur-Entscheidung Konvertierungs-Dienst

**Methodik**: Acht parallele Recherchestränge entlang der fünf Auftragsblöcke, zusammen ~450 Web-Zugriffe auf Primärquellen (Benchmark-Repos, arXiv, PyPI/GitHub-Release-Historien, offizielle Doku-/Preis-/Vertragsseiten), alle am 30.07.2026 abgerufen. 26 entscheidungskritische Zahlen wurden in einem zweiten Durchgang gegen die zitierten Quellen nachgeprüft: 23 exakt bestätigt, 2 Zahlenkorrekturen (Gemini-3-Pro auf dem Französisch-Benchmark: 0,786 statt 0,76; Chandra 2 auf olmOCR-Bench: 85,8 statt 85,9), 1 Zitatpfad-Präzisierung — alles bereits eingearbeitet.

**Konfidenz-Konvention** (wie beauftragt): *hoch* nur bei reproduzierbarer Messung aus unabhängiger Quelle bzw. bei direkt nachlesbaren Fakten (Lizenztexte, Release-Historien, Schemas). Herstellerangaben höchstens *mittel* und als solche markiert. Zahlen auf englischem/chinesischem Material sind gekennzeichnet; Übertragung auf Deutsch ist Analogie, nicht Messung.

---

## Teil 1 — Befunde

### Block 1 · Referenzpipelines, Benchmarks und die Eigenbau-Frage

### B-01 — Auf OmniDocBench v1.6 führen kleine spezialisierte Dokument-VLMs (MinerU2.5-Pro 95,75) vor General-VLMs (Gemini 3 Pro 92,91), während die klassische Pipeline Marker mit 78,44 weit abgeschlagen ist.
**Beleg**: OmniDocBench-Repo (OpenDataLab), README-Leaderboard „v1.6_full", https://github.com/opendatalab/OmniDocBench (30.07.2026, nachverifiziert): MinerU2.5-Pro (1,2B) Overall 95,75, Text-Edit 0,036, Tabellen-TEDS 93,42; GLM-OCR (0,9B) 95,22; PaddleOCR-VL-1.5 (0,9B) 94,93; Gemini 3 Pro 92,91 (Text-Edit 0,064, TEDS 89,15, Lesereihenfolge 0,165); Gemini 3 Flash 92,62; Qwen3-VL-235B 89,78; Marker (1.8.2) 78,44 (Text-Edit 0,157, TEDS 65,77). Datensatz: 1.651 PDF-Seiten.
**Konfidenz**: mittel — Leaderboard-Betreiber und Spitzenmodell stammen von derselben Organisation (OpenDataLab/Shanghai AI Lab); die Fremdmodell-Zahlen sind aber einheitlich vom Benchmark-Betreiber gemessen.
**Gilt unter**: englisch/chinesisches Material (v1.0-Basis ≈62 % Chinesisch), GPU-Inferenz für VLMs.
**Nachmessbar durch**: 20–30 eigene deutsche Seiten mit Gold-Markdown annotieren und mit dem offenen OmniDocBench-Eval-Code (Edit-Distanz + TEDS) gegen MinerU, Marker und Gemini auswerten.
**Widerspruch**: B-06 — auf schwerem französischem Material kollabiert MinerU2.5 auf 0,222; die v1.6-Rangfolge ist nicht sprachübergreifend stabil.

### B-02 — Auf olmOCR-Bench (englisch, 1.403 PDFs, 7.010 maschinenprüfbare Unit-Tests) liegen die besten Systeme bei ~82–83 % Pass-Rate (olmOCR v0.4.0 82,4, Chandra 83,1*), Marker 1.10.1 bei 76,1, MinerU 2.5.4* bei 75,2, GPT-4o bei 68,9 — gescannte Altdokumente bleiben mit ≤50,4 % für alle ungelöst.
**Beleg**: olmOCR-Bench-README (Allen AI), https://github.com/allenai/olmocr/tree/main/olmocr/bench (30.07.2026, nachverifiziert): olmOCR v0.4.0 82,4±1,1; Marker 1.10.1 76,1±1,1 (Tables 72,9; Old Scans 33,5; Multi-Column 80,0); MinerU 2.5.4* 75,2±1,1 (Tables 84,9; Headers/Footers 96,6); PaddleOCR-VL* 80,0; DeepSeek-OCR 75,7; Mistral OCR API 72,0; bestes System bei Old Scans: Chandra 50,4. Sternchen = Herstellerangabe („reported by model authors"), Rest von AI2 selbst reproduziert. GPT-4o 68,9±1,1 und Gemini Flash 2 57,8±1,1 aus dem olmOCR-2-Paper (arXiv:2510.19817, 22.10.2025).
**Konfidenz**: hoch für die AI2-reproduzierten Zeilen (offenes Harness, unabhängig von den Herstellern); mittel für die *-Zeilen.
**Gilt unter**: ausschließlich englisches Material; Unit-Test-Metrik (Pass/Fail) statt Edit-Distanz.
**Nachmessbar durch**: Das offene olmOCR-Bench-Harness akzeptiert beliebige Konverter-Outputs; analoge Unit-Tests (Textpräsenz, Reihenfolge, Tabellenzellen) lassen sich für eigene deutsche PDFs definieren.
**Widerspruch**: keiner gefunden; Zählabweichung 7.010 (HF-Card) vs. „~8.400" (Datalab) vermutlich Benchmark-Wachstum.

### B-03 — Docling ist in beiden verfügbaren Fremdmessungen der Treue abgeschlagen: 50,3 % auf olmOCR-Bench (Datalab-Run; born-digital 64,0 % vs. Marker 83,5/MinerU 83,3) und 0,119 auf schwerem Französisch — IBM selbst publiziert keine Qualitätszahlen für die Gesamtpipeline.
**Beleg**: Datalab-Messung (via MarkTechPost, 24.07.2026, https://www.marktechpost.com/2026/07/24/datalab-marker-v2-vs-mineru-docling-and-liteparse-benchmark-breakdown/): olmOCR-Bench gesamt Marker 2 76,0 / MinerU 72,7 / Docling 50,3 / LiteParse 22,4; born-digital 83,5/83,3/64,0. Französisch-Benchmark: B-06. Docling Technical Report (arXiv:2408.09869) enthält nur Speed/Memory, keine Treue-Metriken.
**Konfidenz**: mittel — Datalab ist Marker-Anbieter (Herstellermessung, markiert); die Docling-Schwäche wird aber durch die unabhängige Französisch-Messung gestützt, und Datalabs Marker-Wert (76,0) deckt sich mit AI2s unabhängigem 76,1.
**Gilt unter**: englisches olmOCR-Bench-Material (GPU-Runs); Docling-Version im Datalab-Artikel nicht ausgewiesen.
**Nachmessbar durch**: olmOCR-Bench-Harness mit aktueller Docling-Version selbst nachfahren; 20 eigene deutsche PDFs Marker vs. Docling, Zellfehler zählen.
**Widerspruch**: ältere Einzelvergleiche (Procycons 03/2025) sahen Docling bei Tabellen vorn — Treue-Rankings hängen stark am Benchmark-Design.

### B-04 — Als End-to-End-Konverter ist Gemini messbar besser als Claude: OmniDocBench v1.5 (IDP-Leaderboard) Gemini-3-Flash 90,1 vs. Claude Sonnet 4.6 86,9, mit auffällig schwacher Texttreue bei Claude (Edit-Distanz 0,165 vs. 0,077); bei Tabellen urteilt eine unabhängige deutsche Hochschul-Studie 9,55/10 (Gemini 3 Pro) vs. 7,02/10 (Claude Sonnet 4.6).
**Beleg**: IDP-Leaderboard (Nanonets, OmniDocBench v1.5, 1.355 Seiten), https://www.idp-leaderboard.org/benchmarks/omnidocbench/ (30.07.2026, nachverifiziert): Gemini-3-Flash 90,1 (Text-Edit 0,077); Gemini-3-Pro 88,8; GPT-5.2 88,0; Claude Sonnet 4.6 86,9 (Text-Edit 0,165); Datalab Marker 85,5; Claude Haiku 4.5 79,6. Tabellen: Horn & Keuper (B-09) Gemini 3 Pro 9,55, Mathpix 8,53, Claude Sonnet 4.6 7,02.
**Konfidenz**: mittel für das Leaderboard (Drittanbieter mit Eigeninteresse); die Gemini>Claude-Ordnung wird von der unabhängigen Studie (hoch) bestätigt.
**Gilt unter**: EN/ZH- bzw. EN-Material, Prompt-basierte Konvertierung ohne Feintuning.
**Nachmessbar durch**: 10 eigene deutsche Dokumente parallel an Gemini- und Anthropic-API mit identischem Prompt; Edit-Distanz + Halluzinations-/Auslassungsprüfung von Hand.
**Widerspruch**: dasselbe Leaderboard zeigt für GLM-OCR 69,2, das offizielle OmniDocBench-README 95,22 — Harness/Prompt-Setup verschiebt Einzelwerte massiv; nur Größenordnungen belastbar.

### B-05 — Im einzigen gefundenen Tabellen-Benchmark mit deutschem Anteil (PulseBench-Tab, 6,2 % Deutsch = 113 Tabellen) schlagen Gemini 3.1 (0,816) und LlamaParse Agentic (0,798) die Cloud-Klassiker deutlich: Azure Document Intelligence 0,761, AWS Textract 0,603, Unstructured 0,360.
**Beleg**: PulseBench-Tab (Pulse AI + Georgia Tech + S&P Global), arXiv:2606.07534 (2026, abgerufen 30.07.2026): 1.820 Tabellen aus 380 Geschäfts-/Behördendokumenten, 9 Sprachen (DE 6,2 %), 48,1 % mit verbundenen Zellen; Metrik T-LAG (als TEDS-Kritik konzipiert). Pulse Ultra 2 0,935 (Anbieter selbst), Gemini 3.1 0,816, LlamaParse Agentic 0,798, Reducto Agentic 0,795 (Coverage 78,8 %), Azure DI 0,761, AWS Textract 0,603, Unstructured 0,360.
**Konfidenz**: mittel — Vendor-Co-Autorenschaft (Pulse Rang 1 = Herstellerangabe); Relativordnung der übrigen Systeme durch akademische Beteiligung und offene Methodik brauchbar.
**Gilt unter**: Tabellenextraktion aus Geschäftsdokumenten, multilingual inkl. Deutsch; „Unstructured" als API ohne Modus-Angabe (hi_res nicht verifiziert).
**Nachmessbar durch**: 50 eigene deutsche Tabellen durch Azure DI, Gemini und den Eigenbau; Zell-Adjazenz-Fehler zählen.
**Widerspruch**: Reductos eigener RD-TableBench (11/2024) behauptet die umgekehrte Ordnung (Reducto vorn, exakte Fremdzahlen nicht publiziert).

### B-06 — EN/ZH-Benchmark-Rankings übertragen sich nicht auf europäische Sprachen: Auf schweren französischen Seiten stürzt MinerU2.5 auf 0,222 und Docling auf 0,119 ab, während Gemini 3 Pro (0,786) und Gemini 3 Flash (0,755) führen — die nächstliegende Evidenz für deutsches Material.
**Beleg**: „Benchmarking VLMs for French PDF-to-Markdown Conversion" (Probayes/OpenValue — kein Parser-Anbieter), arXiv:2602.11960 (02/2026, nachverifiziert 30.07.2026): Aggregat gemini-3-pro-preview 0,786, gemini-3-flash-preview 0,755, Chandra 0,664 (bestes Open-Weight), GPT-5.2 0,602, olmOCR-2-7B-FP8 0,461, PaddleOCR-VL 0,381, MinerU2.5 0,222, Docling 0,119. Proprietäre Modelle deutlich robuster bei Handschrift (Gemini 0,60 vs. Open-Weights nahe 0,0). Fehler-Audit: 98 % echte Modellfehler (vs. 47 % bei olmOCR-Bench).
**Konfidenz**: hoch für Französisch (unabhängige Industrie-Studie mit Fehler-Audit); Übertrag auf Deutsch ist Analogie (lateinische Schrift, europäische Konventionen), nicht Messung.
**Gilt unter**: absichtlich schwere Seiten (Handschrift, Formulare, Mehrspalter, dichte Tabellen) — Absolutwerte daher niedrig; Modellstände Anfang 2026.
**Nachmessbar durch**: Die entscheidende Messung für den Dienst: 30–50 repräsentative deutsche Seiten als Mini-Benchmark; brechen MinerU/Docling dort ähnlich ein, scheiden sie trotz OmniDocBench-Spitzenwerten aus.
**Widerspruch**: B-01 (MinerU-Weltspitze auf EN/ZH) — der Widerspruch löst sich als Sprach-/Domänen-Overfitting auf.

### B-07 — Einen deutschen Dokument→Markdown-Benchmark gibt es nicht (Stand 30.07.2026); Deutsch existiert nur als Minderheitsanteil: DocLayNet 2,5 % (nur Layout-Klassen), PulseBench-Tab 6,2 % (nur Tabellen), OmniDocBench 0 %.
**Beleg**: DocLayNet-Paper (arXiv:2206.01062): „close to 95 % … English … German (2.5 %)"; OmniDocBench v1.0 (arXiv:2412.07626, Tab. S3): 290 EN / 612 ZH / 79 gemischt, 0 DE; olmOCR-Bench: englisch (HF-Datasetcard). Mehrfache gezielte Suchen (30.07.2026) ergaben als nächsten Verwandten nur den französischen Benchmark (B-06).
**Konfidenz**: hoch — Negativbefund nach systematischer Suche; Anteile aus Originalquellen.
**Gilt unter**: öffentliche, zitierfähige Benchmarks; interne Firmen-Evals können existieren, sind aber nicht publiziert.
**Nachmessbar durch**: nicht ersetzbar — ein eigener deutscher Goldstandard (30–50 Seiten inkl. Tabellen/Scans) ist die einzige Möglichkeit, Werkzeuge für den konkreten Bestand zu ranken.
**Widerspruch**: keiner gefunden.

### B-08 — OmniDocBench ist als alleinige Entscheidungsgrundlage ungeeignet: ≈62 % Chinesisch, Betreiber = MinerU-Hersteller, die eigene Metrik musste in v1.6 per „Multi-Granularity Adaptive Matching" korrigiert werden, und der Benchmark gilt seit Anfang 2026 als gesättigt.
**Beleg**: Sprachverteilung arXiv:2412.07626 (Tab. S3); OmniDocBench- und MinerU-Repo beide unter github.com/opendatalab; README v1.6 (nachverifiziert): „We propose Multi-Granularity Adaptive Matching (MGAM), which eliminates matching bias"; LlamaIndex-Blog (24.02.2026): „OmniDocBench is saturated" — Edit-Distanz/TEDS bestrafen harmlose Formatvarianten (HTML- vs. Markdown-Tabellen, Bullet-Stile), nur eine Gold-Repräsentation pro Dokument.
**Konfidenz**: hoch für die Fakten (Seitenzahlen, Organisationsidentität, MGAM-Zitat); die Sättigungs-Einschätzung stammt von einem Anbieter (LlamaIndex) → mittel.
**Gilt unter**: OmniDocBench v1.0–v1.6 und alle daraus abgeleiteten Rankings.
**Nachmessbar durch**: dieselben 10 Seiten einmal mit Markdown-, einmal mit HTML-Tabellen ausgeben lassen; TEDS/Edit-Differenz zeigt die Normalisierungs-Empfindlichkeit direkt.
**Widerspruch**: OpenDataLab hält mit v1.6/v1.7 dagegen — bestätigt damit implizit die Kritik an den Vorversionen.

### B-09 — Die Standardmetriken selbst sind schwach valide: TEDS korreliert nur mit r=0,684 und GriTS mit r=0,70 mit menschlichem Qualitätsurteil (LLM-Judge: r=0,93); in derselben unabhängigen Studie ist Gemini 3 Pro (9,55/10) das beste Tabellensystem, Mathpix 8,53, Claude Sonnet 4.6 nur 7,02.
**Beleg**: Horn & Keuper, „Beyond String Matching: Semantic Evaluation of PDF Table Extraction", HS Offenburg + Univ. Mannheim, arXiv:2603.18652 (06/2026, nachverifiziert): 451 Tabellen aus 100 aus arXiv-LaTeX neu gerenderten Seiten, 21 Parser, 1.554 Human-Ratings (α=0,77); TEDS r=0,684, GriTS-Con r=0,70, LLM-Judge (Gemma-4-31b) r=0,93 (bester Judge Claude Opus 4.6: 0,94). Scores: Gemini 3 Pro 9,55; Gemini 3 Flash 9,50; Mathpix 8,53; Qwen3-VL-235B 8,43; Claude Sonnet 4.6 7,02; olmOCR-2-7B 4,05; GROBID 2,10. „Rule-based metrics penalize harmless variation while overlooking critical errors."
**Konfidenz**: hoch — unabhängige akademische Studie mit Humanvalidierung und offener Methodik.
**Gilt unter**: englische wissenschaftliche Tabellen, born-digital gerendert; Modellstände Anfang 2026.
**Nachmessbar durch**: den LLM-Judge-Ansatz (Judge-Prompt im Paper) mit der vorhandenen Gemini-API direkt auf eigene deutsche Tabellen übertragen — laut Studie valider als TEDS.
**Widerspruch**: auf OmniDocBench v1.5 liegt Claudes Tabellen-TEDS fast gleichauf mit Gemini — die Diskrepanz illustriert genau das Metrik-Problem.

### B-10 — Die letzte direkte Messung Heuristik vs. gelerntes Modell (10/2024) zeigt: Table Transformer schlägt camelot/tabula/pdfplumber auf 4 von 6 Dokumentkategorien drastisch (Scientific F1 0,91 vs. camelot 0,34; Financial 0,79 vs. 0,10) — Heuristiken gewinnen nur bei klar linierten Behörden-/Handbuch-Tabellen (camelot-lattice 0,83 vs. 0,75; PyMuPDF 0,75 vs. 0,62).
**Beleg**: Adhikari & Agarwal, arXiv:2410.09871 (13.10.2024): Camelot 0.11.0, Tabula 2.9.3, pdfplumber 0.11.2, PyMuPDF 1.24.7 vs. TATR-v1.1 auf DocLayNet (6 Kategorien à 400 Dokumente); F1 Financial TATR 0,79 vs. Camelot 0,10/pdfplumber 0,06; Scientific 0,91 vs. 0,34; Patent 0,53 vs. 0,02; aber Tender: Camelot 0,83 vs. TATR 0,75; Manual: PyMuPDF 0,75 vs. TATR 0,62. Fazit der Autoren: gelernte Modelle für komplexe/rahmenlose Tabellen.
**Konfidenz**: mittel — unabhängig, aber unterschiedliche Schwellen je Werkzeugklasse und nur Tabellen-*Detektion*, nicht Strukturtreue.
**Gilt unter**: DocLayNet-Material (≈95 % Englisch, born-digital), Stand Ende 2024.
**Nachmessbar durch**: die fünf Bestands-Detektoren einzeln gegen TATR/TableFormer auf 50 eigenen Tabellen; Erkennungs-F1 und Zellfehler getrennt auswerten — zeigt, ob der Konsens überhaupt über dem besten Einzeldetektor liegt.
**Widerspruch**: der Tender-/Manual-Befund ist ein echter Teilwiderspruch zugunsten von camelot-lattice bei linierten Verwaltungstabellen (für deutsche Behördendokumente relevant).

### B-11 — Multi-Detektor-Ensembles aus Heuristik-Werkzeugen sind aus der Evaluationslandschaft 2025/2026 vollständig verschwunden: Kein aktueller Benchmark führt camelot, pdfplumber, tabula oder img2table noch mit; ein Vergleich „Konsens-Ensemble vs. gelerntes Modell" existiert nicht.
**Beleg**: Systematische Durchsicht (30.07.2026): OmniDocBench v1.6 — keine Heuristik-Tools; olmOCR-Bench — keine; PulseBench-Tab — explizit: „No rule-based tools (Camelot, pdfplumber, Tabula) … were evaluated"; Horn & Keuper (21 Parser) — einziges Nicht-ML-System GROBID mit 2,10/10; Französisch-Benchmark — keine. Letzte akademische Messung überhaupt: B-10 (10/2024). Gezielte Suche nach Konsens-/Ensemble-Vergleichsstudien: keine Treffer.
**Konfidenz**: hoch für den Negativbefund (mehrere unabhängige Benchmarks geprüft). Konsequenz: Der Ensemble-Ansatz ist weder bestätigt noch widerlegt — er ist unevaluiert und aus dem Stand der Technik herausgefallen; die Beweislast liegt beim Ensemble.
**Gilt unter**: öffentliche Benchmarks 2024–2026.
**Nachmessbar durch**: den Eigenbau als zusätzliche „Pipeline" in das eigene deutsche Testset einreihen und mit identischer Metrik gegen MinerU/Marker/Gemini messen — vorher ist jede Aussage über seinen Wert unbegründet.
**Widerspruch**: Teilwiderspruch B-10 (camelot-lattice bei linierten Tabellen vorn) — einzelne Komponenten hatten belegten Wert, der Fünffach-Konsens als Konstrukt nie.

### B-12 — Stärkenprofil nach Dokumentklasse (englisches Material): MinerU dominiert Tabellen (84,9) und Kopf-/Fußzeilen (96,6), Marker Mehrspalter (80,0) und arXiv-Layouts (83,8), General-VLMs alte Scans und Handschrift — alte Scans bleiben für alle Systeme die schwächste Klasse (33–50).
**Beleg**: olmOCR-Bench-Kategoriewerte (AI2-README, 30.07.2026): Tables MinerU 84,9 vs. Marker 72,9; Multi-Column Marker 80,0 vs. MinerU 78,2; Headers/Footers MinerU 96,6 vs. Marker 86,6; Old Scans Marker 33,5/MinerU 33,7/bestes System Chandra 50,4; ArXiv Marker 83,8. Handschrift: Französisch-Benchmark (B-06) — proprietäre VLMs „substantially higher robustness".
**Konfidenz**: hoch für die olmOCR-Bench-Zeilen (englisch); mittel für den Handschrift-Transfer.
**Gilt unter**: englisch, born-digital + Scans, Marker 1.10.1 / MinerU 2.5.4.
**Nachmessbar durch**: eigenes Testset nach denselben Klassen schichten und pro Klasse das beste Werkzeug bestimmen — spricht für Routing im Dienst statt eines Gesamtsiegers.
**Widerspruch**: keiner gefunden; Profile über AI2-, Datalab- und Frankreich-Messung konsistent.

### B-13 — Trainingskontamination ist bei den kanonischen Tabellen-/Layout-Benchmarks systemisch: Doclings Layout-Modell ist auf DocLayNet trainiert (IBM erstellte beide), TATR auf PubTables-1M/FinTabNet, granite-doclings FinTabNet-TEDS 0,97 ist eine In-Distribution-Herstellerangabe — solche Werte messen keine Generalisierung.
**Beleg**: Docling Technical Report (arXiv:2408.09869): Layout-Modell „re-trained on DocLayNet, our … dataset"; TATR-Repo (github.com/microsoft/table-transformer): trainiert auf PubTables-1M/FinTabNet.c, dort GriTS 0,985 (in-distribution, Modelle von 2023); granite-docling-Modellkarte: FinTabNet TEDS 0,97. DocLayNet-Inter-Annotator-Agreement für Tabellen: nur mAP 77–81 %.
**Konfidenz**: hoch für die Zitate/Zahlen (Originalquellen); die Einordnung ist Standardschluss.
**Gilt unter**: alle Rankings auf DocLayNet-mAP, PubTables-/FinTabNet-TEDS; englisches Material.
**Nachmessbar durch**: Werkzeuge nur auf Out-of-Distribution-Material (eigene deutsche Dokumente) vergleichen; Benchmark-Selbstangaben der Hersteller ignorieren.
**Widerspruch**: keiner gefunden — die Trainingsherkunft steht in den eigenen Reports.

### B-14 — docling ist das aktivste und governance-stabilste Projekt des Feldes: Release-Kadenz alle 2–4 Tage (v2.114.0 am 20.07.2026, ~48 Releases allein 2026), seit April 2025 LF-AI-&-Data-Projekt, seit 09.06.2026 mit eigener Standardisierungs-Arbeitsgruppe (DocLang; IBM, NVIDIA, Red Hat) — als einziges Projekt mit Multi-Vendor-Governance.
**Beleg**: GitHub-Releases docling-project/docling (30.07.2026): v2.114.0 (20.07.2026), allein Juli 2026 sieben Releases; PyPI: ≥42 Releases 2025; keine Breaking Changes in der 2.x-Linie ausgewiesen. LF-AI-&-Data-Projektseite: „IBM donated Docling as an Incubation-stage project … in April 2025"; LF-Pressemitteilung 09.06.2026: DocLang Specification WG (IBM, NVIDIA, Red Hat; Beiträge ABBYY, HumanSignal).
**Konfidenz**: hoch — Release-Historien und Foundation-Erstquellen.
**Gilt unter**: docling 2.x, Stand 30.07.2026. Wichtig: Governance-Stärke ist von gemessener Treue (B-03) zu trennen.
**Nachmessbar durch**: Release-Seiten und lfaidata.foundation/projects/docling abrufen.
**Widerspruch**: keiner gefunden.

### B-15 — camelot ist entgegen der verbreiteten Annahme nicht tot, sondern wiederbelebt: v2.0.0 am 04.06.2026 mit neuem Maintainer-Team, und pdfium ist seit v1.0.0 Default-Backend — der Ghostscript-Zwang (und damit dessen AGPL) ist für aktuelle Versionen entfallen.
**Beleg**: PyPI camelot-py (nachverifiziert): 2.0.0 — 04.06.2026; 1.0.0 — 30.12.2024; davor Lücke seit 02/2023. Install-Doku (nachverifiziert): „as of v1.0.0 ghostscript is replaced by pdfium as the default image conversion backend"; Ghostscript/Poppler optional. Aber: 2025er-„Aktivität" bestand aus faktisch zwei Release-Tagen (1.0.1–1.0.9 alle am 09./10.08.2025); 56 offene Issues.
**Konfidenz**: hoch — Registry-Daten und offizielle Doku, doppelt geprüft.
**Gilt unter**: camelot-py ≥1.0.0 (pdfium-Default), 2.0.0 mit Python ≥3.10.
**Nachmessbar durch**: PyPI-Historie und Install-Doku abrufen; im Deployment `backend="pdfium"` prüfen.
**Widerspruch**: keiner gefunden.

### B-16 — pdfminer.six ist offiziell ein Community-Projekt mit „limited maintainer availability" und hatte 2025 eine Remote-Code-Execution-Lücke (CVE-2025-64512, pickle-basierte CMaps) — für einen Dienst, der fremde PDFs entgegennimmt, ist Version ≥20251230 Pflicht; pdfplumber (stabil-konservativ, 0.11.x seit 03/2024) hängt architektonisch daran.
**Beleg**: pdfminer.six-CHANGELOG (30.07.2026): 20251107 „Fixed arbitrary code execution vulnerability when loading pickle cmaps"; 20251230 „Eliminated … (CVE-2025-64512) by replacing pickle CMap storage with JSON"; 20251227 entfernt Python 3.9. README: „community-maintained project with limited maintainer availability". pdfplumber: 0.11.9 (05.01.2026), 2 benannte Maintainer, MIT.
**Konfidenz**: hoch — Changelog und README direkt gelesen.
**Gilt unter**: pdfminer.six ≥20251230, pdfplumber 0.11.x.
**Nachmessbar durch**: CVE-2025-64512 in einer CVE-Datenbank; eigene Versionsstände im Image prüfen.
**Widerspruch**: keiner gefunden.

### B-17 — PyMuPDF ist das am solidesten firmengetragene Bestands-Werkzeug (Artifex, ~10 Releases 2025, zuletzt 1.27.2.3 am 24.04.2026) — der Preis ist die AGPL-/Kommerzlizenz-Dualität (Details B-58).
**Beleg**: PyPI-Historie PyMuPDF (30.07.2026): 1.27.2.3 — 24.04.2026; 2025: ~10 Releases. README: maintained by Artifex Software; AGPL v3 + kommerzielle Lizenzen; kommerzielles Add-on „PyMuPDF Pro" für Office-Formate. Breaking: 1.27.1 entfernt mupdf <1.26; Python 3.10–3.14.
**Konfidenz**: hoch — PyPI + Repo deckungsgleich.
**Gilt unter**: PyMuPDF 1.27.x, Stand 30.07.2026.
**Nachmessbar durch**: PyPI-Historie; Artifex-Lizenzseite.
**Widerspruch**: keiner gefunden; auffällig nur: seit 24.04.2026 kein Release (~3 Monate) — für Artifex' Rhythmus lang, aber kein Abbruchsignal.

### B-18 — Die unstructured-Open-Source-Bibliothek wird vom Hersteller offiziell zum Prototyping-Werkzeug herabgestuft — „not designed for production", „significantly decreased performance", kein GPU-Support, keine aktuellen VLM-/OCR-Modelle — bei gleichzeitig weiterlaufender wöchentlicher Release-Kadenz (0.24.0 am 06.07.2026).
**Beleg**: docs.unstructured.io/open-source/introduction/overview (30.07.2026), wörtlich: „designed as a starting point for quick prototyping and has limits"; „Not designed for production scenarios"; „Significantly decreased performance on document and table extraction"; „No access to Unstructured's latest vision language model (VLM) offerings"; „GPU usage is not supported". GitHub-Releases: 0.24.0 — 06.07.2026, 25 Releases 2025; 178 offene Issues.
**Konfidenz**: hoch — wörtliche Zitate aus der Hersteller-Doku (die Qualitätsaussage selbst ist Eigen-Downgrade-Marketing möglich → als Qualitätsaussage mittel).
**Gilt unter**: unstructured OSS 0.22–0.24, Stand 30.07.2026.
**Nachmessbar durch**: identisches PDF durch OSS-hi_res und die bezahlte API; Tabellenstruktur vergleichen.
**Widerspruch**: die anhaltende Release-Aktivität — strategisch degradiert heißt (noch) nicht technisch aufgegeben.

### B-19 — markitdown wird von Microsofts AutoGen-Team auf Sparflamme gepflegt: 3 Releases 2026 (zuletzt 0.1.6 am 26.05.2026), dauerhafter 0.x-Status mit dokumentierten API-Brüchen, und ein extremes Missverhältnis von 439 offenen PRs zu 313 Commits insgesamt.
**Beleg**: PyPI markitdown (30.07.2026): 0.1.6 — 26.05.2026, 0.1.5 — 20.02.2026; Release-Lücken von je ~3 Monaten. GitHub: 379 offene Issues, 439 offene PRs, 313 Commits. v0.1.0-Notes (03/2025): Breaking Changes (Stream-basierte Converter-API, optionale Dependency-Gruppen). README warnt selbst: „may not be the best option for high-fidelity document conversions for human consumption".
**Konfidenz**: hoch für Daten; mittel für die Deutung „Sparflamme".
**Gilt unter**: markitdown 0.1.6.
**Nachmessbar durch**: PyPI-Historie; Pulls-Tab (Zahl + Alter der ältesten PRs).
**Widerspruch**: keiner gefunden.

### B-20 — Governance der Modell-Pipelines: marker gehört Datalab (Kommerzialisierungsdruck: bestes Modell primär auf der Bezahlplattform, Gewichte-Lizenz mit Umsatzschwelle, B-60); MinerU gehört OpenDataLab/Shanghai-AI-Lab-Umfeld (Lizenz 2026 liberalisiert, B-60); img2table ist ein Bus-Faktor-1-Projekt (Solo-Maintainer, 2.0.0-Rewrite 05/2026).
**Beleg**: marker-README (30.07.2026): Managed Platform „runs … our latest open source model, Chandra — higher accuracy than Marker"; MinerU-README: Lizenzwechsel mit v3.1.0 (18.04.2026), Kommerzangebot mineru.net; img2table: PyPI 2.0.0 — 10.05.2026, primär Repo-Owner xavctn, 64 offene Issues.
**Konfidenz**: hoch — READMEs/Registry direkt gelesen.
**Gilt unter**: Stand 30.07.2026.
**Nachmessbar durch**: READMEs und Contributor-Graphen abrufen.
**Widerspruch**: keiner gefunden.

### Block 2 · CPU-Welt und GPU-Welt

### B-21 — docling schafft auf einer 8-vCPU-x86-Maschine im Median unter 1 s/Seite (Durchschnitt ~3,1 s); der OCR-Anteil dominiert die Laufzeit (EasyOCR dort 13 s/Seite; OCR abschalten spart ~60 %).
**Beleg**: Docling Technical Report (arXiv:2408.09869v4, Stand 12/2024, abgerufen 30.07.2026): docling 2.5.2 auf AWS g6.xlarge (AMD EPYC 7R13, 8 vCPU, 32 GB RAM): Median 0,79 s/Seite, Ø 3,1 s, 95. Perzentil 16,3 s; EasyOCR 13 s/Seite; „Disabling OCR saves ~60% runtime". Grobbestätigung: Procycons-Benchmark (24.03.2025): ~1,3 s/Seite, Hardware nicht offengelegt.
**Konfidenz**: mittel — Herstellerreport (IBM) mit sauber dokumentierter Hardware; unabhängige Bestätigung ohne CPU-Angabe.
**Gilt unter**: CPU-only x86, docling 2.5.x, überwiegend born-digital, englisch; alter OCR-Default (EasyOCR).
**Nachmessbar durch**: 20 repräsentative eigene PDFs durch docling auf der Mintbox, `time` pro Datei, einmal mit `do_ocr=false`.
**Widerspruch**: keiner gefunden für CPU; auf GPU-Benchmarks ist docling langsamer als Marker (B-03-Quelle: 2,1 vs. 2,9 S./s auf B200).

### B-22 — Der CUDA-Anteil kostet im Konverter-Image 4–7 GB: offizielles docling-serve-CPU-Image 4,4 GB vs. CUDA-12.8-Image 11,4 GB; ein dokumentierter Umbau schrumpfte ein docling-Image von 9,74 auf 1,74 GB allein durch den CPU-torch-Index — die 15,5→8,9-GB-Erfahrung des Bestands ist damit reproduziert und nicht am Boden.
**Beleg**: docling-serve-README (nachverifiziert 30.07.2026): Base amd64 8,7 GB, `docling-serve-cpu` 4,4 GB, `docling-serve-cu128` 11,4 GB. Unabhängig: shekhargulati.com (05.02.2025): docling 2.18.0 mit `--extra-index-url …/whl/cpu` + Cache-Bereinigung: 9,74 → 1,74 GB. Wheel-Größen (PyPI, 30.07.2026): torch 2.12.0 Linux-x86_64 (CUDA-gekoppelt) 532 MB; CPU-Builds derselben Version 88–123 MB (macOS/Windows); onnxruntime 1.27.0: 18,7 MB.
**Konfidenz**: hoch für die Image-Tabelle (nachverifiziert) und Wheel-Größen; mittel für die Übertragbarkeit des 1,74-GB-Umbaus.
**Gilt unter**: Linux/amd64, docling-serve v1.x; Modellgewichte im Image.
**Nachmessbar durch**: `docker pull ghcr.io/docling-project/docling-serve-cpu && docker images` auf der Mintbox.
**Widerspruch**: keiner gefunden.

### B-23 — Docling-Treue ohne torch im Image gibt es nicht: torch ist laut Maintainer auch für reinen CPU-Betrieb zwingend (Layout-/TableFormer-Modelle); nur der OCR-Teilschritt ist über RapidOCR/onnxruntime torch-frei.
**Beleg**: docling Discussion #1349 (04/2025): Maintainer dolfim-ibm — PyTorch für CPU-Verarbeitung nötig, „Docling itself doesn't strictly require CUDA packages"; Empfehlung CPU-Index. Eine ONNX-Gesamtdistribution existiert nicht.
**Konfidenz**: hoch — Maintainer-Aussage über die eigene Abhängigkeitsstruktur, per `pip` nachprüfbar.
**Gilt unter**: docling 2.x (Stand 04/2025 bis heute, keine gegenteilige Ankündigung gefunden).
**Nachmessbar durch**: Installation mit `--no-deps` + Abhängigkeitsbaum prüfen.
**Widerspruch**: keiner gefunden; die einzige komplett torch-freie Route ist ein anderes Werkzeug (B-27: Tika) mit Strukturverlust.

### B-24 — Doclings Default-OCR ist für Deutsch aktuell eine Falle: Seit ~v2.56 ist RapidOCR Default, lädt aber fest verdrahtete chinesisch/englische PP-OCR-Modelle und ignoriert `--ocr-lang` (Issue offen, Stand 30.07.2026); die PP-OCRv4/v5-Hauptmodelle können laut PaddleOCR-Maintainer keine Nicht-Englisch-Zeichen („Spaß"→„SpafS", „Zähne"→„Zahne").
**Beleg**: docling Issue #2927 (erstellt 28.01.2026, nachverifiziert offen): „Regardless of the language specified, Docling consistently loads Chinese (‚ch_PP-OCRv4') models"; „model selection logic … is hardcoded". PaddleOCR Issue #14861 (nachverifiziert): Maintainer: „Models PP-OCRv5 and v4 don't support chars outside english"; Beispiele „Spaß"→„SpafS"/„SpaR", „frühstücken"→„fruhstucken". Deutsch deckt erst das separate `latin_PP-OCRv5_mobile_rec` ab (Herstellerangabe 84,7 % Acc auf eigenem Eval-Set).
**Konfidenz**: hoch für den Umlaut-Defekt und den offenen Issue-Status (nachverifiziert); mittel für die 84,7 % (Herstellerangabe).
**Gilt unter**: docling ≥2.56 mit Default-OCR, gescannte Dokumente; irrelevant für born-digital ohne OCR. Abhilfe: `ocr_engine=tesseract, lang=deu` oder manuell eingehängte Latin-Modelle.
**Nachmessbar durch**: deutschen Testscan mit ä/ö/ü/ß durch docling-Default vs. Tesseract-deu; CER vergleichen.
**Widerspruch**: die Hersteller-Doku bewirbt „106 languages" — gilt nur für die dedizierten Multilingual-Modelle, nicht für die Default-Pipeline.

### B-25 — Auf sauberem modernem deutschem Druck ist Tesseract 5 fast auf Kommerz-Niveau (2 Fehler vs. 0 bei ABBYY/Azure); auf degradiertem deutschem Material bricht es deutlich ein (94,9 % vs. 99,7 % Wortgenauigkeit).
**Beleg**: officemanager.de-OCR-Test (20.01.2024, abgerufen 30.07.2026), deutsche Vorlagen, Tesseract 5.3.3 vs. ABBYY FineReader 16, OmniPage 19.2, Azure KI Vision u. a.: sauberer Druck — ABBYY/Azure 0 Fehler, Tesseract 2; degradierte AGB-Seite — OmniPage 99,7 %, Feld 98,6–99,3 %, Tesseract 94,9 %; Roman-Fließtext — mehrere Engines 100 %.
**Konfidenz**: mittel — unabhängiger, methodisch beschriebener Test auf echtem deutschem Material, aber nur 3 Vorlagen.
**Gilt unter**: modernes deutsches Druckmaterial (Antiqua), Tesseract 5.3.3 `deu`, CPU. Historische Drucke/Fraktur sind eine andere Welt (OCR-D/OCR-BW: `deu_latf` offiziell fehlerbehaftet, Projekt-Modell Frak2021 3,25 % CER).
**Nachmessbar durch**: je 5 saubere und schlechte eigene deutsche Scans, tesseract -l deu vs. Azure Read, Fehler von Hand zählen.
**Widerspruch**: Statworx (2020, Tesseract 4.1): auf deutschen Rechnungen mit Sonderfonts deutlich schlechter (€-Zeichen in 50 % falsch) — auf untypischen Fonts bleibt Tesseract fragil.

### B-26 — Cloud-OCR ist gegenüber Tesseract auf sauberem Material nur moderat, auf verrauschtem deutlich besser (peer-reviewed, englisch: WER ~3–4 % Google Document AI vs. ~5–6 % Tesseract 4.1; „substantially better" bei Rauschen); Preisreferenz ~1,50 $/1.000 Seiten.
**Beleg**: Journal of Computational Social Science (Springer, 2021): 322 englische Buchseiten (1853–1920), Tesseract 4.1.1 vs. Textract vs. Google Document AI. Deutsche Stützstelle: B-25 (Azure auf Deutsch 0 Fehler bzw. ~99 %).
**Konfidenz**: hoch für die englischen Zahlen (unabhängig, peer-reviewed); als Deutsch-Näherung ausdrücklich nur mittel; Tesseract-Version veraltet (4.1, nicht 5).
**Gilt unter**: älteres Buchmaterial, englisch, Cloud-Stand 2021 — als Richtungsaussage, nicht als aktuelle Messung.
**Nachmessbar durch**: identisches deutsches Testset an Azure Read + Google Document AI, CER gegen lokale Engines (Kosten im Cent-Bereich).
**Widerspruch**: B-25 — auf sauberem modernem Druck fast Gleichstand; der Cloud-Abstand ist primär ein Schlechtmaterial-Phänomen.

### B-27 — Die übrigen CPU-Kandidaten in Kürze: unstructured hi_res ~5 s/Seite (Laptop-i5, Maintainer-bestätigt „expected"); MinerU-CPU nur im pipeline-Backend mit 16–32 GB RAM-Anforderung und ohne belastbare s/Seite-Zahl; Marker v2 hat auf CPU nur den „fast, no OCR"-Modus, der von 76,0 auf 43,6 % Treue abstürzt; Apache Tika (347 MB, torch-frei, inkl. Tesseract-deu) liefert Text/XHTML statt strukturtreuem Markdown; docTR-Standardmodelle kennen kein ä/ö/ß (französisches Trainings-Vokabular).
**Beleg**: unstructured Issue #3217 (700 Seiten ≈ 1 h, i5-1135G7, Maintainer: „expected"); MinerU-Quick-Start (nachverifiziert): pipeline „Pure CPU Support ✅", VLM ❌, „Min 16GB+, Recommended 32GB+"; marker-README (nachverifiziert): balanced 76,0 (GPU) vs. „fast, no OCR (CPU)" 43,6 bei 23,7 S./s; Docker Hub apache/tika: full-Image 347 MB (komprimiert) mit Tesseract + deutschem Sprachpaket; docTR-Doku: „most of our recognition models were trained on our french vocab" (ä/ö/ß fehlen im french-Vokabular).
**Konfidenz**: mittel — Einzelmessungen mit Hardware-Angabe bzw. offizielle Doku; keine kontrollierten Benchmarks.
**Gilt unter**: CPU-only; Marker-Lizenz siehe B-60; Tika = Formatbreite ohne Struktur.
**Nachmessbar durch**: 50-Seiten-Testkorpus je Kandidat auf der Mintbox stoppen (Wall-Clock, RAM-Peak via cgroups).
**Widerspruch**: Drittanbieter-Schätzungen für MinerU-CPU (~2–6 s/Seite) wirken gegen die A10-GPU-Referenz des Maintainers (0,5–1 s/S.) optimistisch — unbelastbar.

### B-28 — Auf der 12-GB-A2000 ist die ≤3B-Klasse sicher, die 7B-Klasse nur quantisiert: unquantisierte 7–8B-Gewichte (BF16 ≈ 14–16 GB) passen nicht; olmOCRs beworbene „12 GB Minimum" wurden von AI2 nie auf einer 12-GB-Karte demonstriert (getestet: RTX 4090/L40S/A100/H100); FP8-Checkpoints laufen auf Ampere nur als Weight-only-W8A16 über vLLMs FP8-Marlin (laut Upstream FP16-vergleichbare Genauigkeit).
**Beleg**: olmOCR-README (nachverifiziert): „Recent NVIDIA GPU (tested on RTX 4090, L40S, A100, H100) with at least 12 GB of GPU RAM"; vLLM-Issue #27934 (RTX 3060 12 GB): 7B–13B-Init-Failures, 1,5–3B laufen mit 4,6–5,5 GB; vLLM PR #5975 (FP8-Marlin auf Ampere): weight-only, „maintains accuracy comparable to FP16", 2× Gewichts-Speicherreduktion.
**Konfidenz**: mittel — Arithmetik + Upstream-Angaben + Einzelberichte; genau die 12-GB-Konstellation ist unvermessen.
**Gilt unter**: vLLM auf Ampere (CC 8.6), dichte Modelle (kein MoE); A2000 zusätzlich 70-W-limitiert.
**Nachmessbar durch**: olmOCR-2-FP8 und ein 3B-Modell auf der A2000 laden; VRAM via nvidia-smi, OOM-Verhalten und KV-Cache-Größe protokollieren.
**Widerspruch**: olmOCR-Werbeaussage vs. vLLM-Praxisberichte — löst sich nur über Quantisierung.

### B-29 — Die 12-GB-Grenze ist praktisch kein Qualitätshindernis mehr: Die 0,65–3B-Klasse (Surya 2: 83,3; LightOnOCR-2-1B: 83,2; dots.mocr 3B: 83,9) liegt auf olmOCR-Bench nur 2–3 Punkte hinter dem 5B-Spitzenreiter Chandra 2 (85,8) und vor olmOCR-2-7B (82,4).
**Beleg**: Surya-README (nachverifiziert): Surya 2 (650M) 83,3; Chandra-README (nachverifiziert): Chandra 2 85,8±0,8; dots.ocr-Repo: dots.mocr 83,9; LightOn-Blog (19.01.2026): LightOnOCR-2-1B 83,2±0,9, Apache 2.0, Fokus europäische Sprachen. Alle Werte Herstellerangaben auf dem offenen olmOCR-Bench-Harness.
**Konfidenz**: mittel — Herstellerangaben, aber auf offenem, nachfahrbarem Harness; englisches Material.
**Gilt unter**: olmOCR-Bench (englisch); VRAM: 0,65–3B ≈ 1,3–8 GB Gewichte in BF16 → 12 GB unkritisch.
**Nachmessbar durch**: zwei Kandidaten (z. B. dots.ocr, PaddleOCR-VL) auf der A2000 gegen 50 deutsche Seiten; Fehlertypen zählen.
**Widerspruch**: keiner gefunden.

### B-30 — Publizierte Deutsch-Werte für Dokument-VLMs existieren fast nur aus Datalab-eigenen Benchmarks (Surya 2: 89,7 %; Chandra 2: 94,8 %); olmOCR ist laut eigenem Team faktisch englisch trainiert („most of the training data is English only"), und für dots.ocr, DeepSeek-OCR, PaddleOCR-VL, MinerU-VLM, LightOnOCR gibt es gar keinen publizierten Deutsch-Score.
**Beleg**: Surya-README (nachverifiziert): „de | German | 89.7%" (interner 91-Sprachen-Benchmark); Chandra-README (nachverifiziert): „de | … | 94.8%" (43 Sprachen); olmOCR-HF-Diskussion (02/2025, jakep-allenai): „Most of the training data is English only, but there is some basic performance in other languages"; PaddleOCR-VL: Deutsch in der 109-Sprachen-Liste, kein separater Score; granite-docling-Modellkarte: nur Englisch (+ ja/ar/zh experimentell) — Deutsch nicht genannt.
**Konfidenz**: mittel für die Datalab-Zahlen (Hersteller-Benchmark, Metrik nicht offengelegt); hoch für die Negativbefunde (explizite Herstellerangaben bzw. Abwesenheit).
**Gilt unter**: Stand 30.07.2026; die Lizenzfrage der Datalab-Gewichte (B-60) schränkt deren Nutzbarkeit zusätzlich ein.
**Nachmessbar durch**: eigener 50-Seiten-Deutsch-Test pro Kandidat — es gibt keinen Ersatz.
**Widerspruch**: Blogposts, die olmOCR pauschal als Multilingual-Lösung führen — vom Hersteller selbst nicht gedeckt.

### B-31 — Repetitions-/Halluzinations-Loops sind bei OCR-VLMs kein Randphänomen, sondern quantifiziert zweistellig möglich: DeepSeek-OCR 9,2 % „catastrophic failure" auf schweren Scans (Gegenmaßnahmen unwirksam), Chandra ~8–10 % Token-Loops, und dots.ocr dokumentiert den Trigger selbst — durchgehende Punktlinien/Unterstriche, wie sie deutsche Formulare prägen.
**Beleg**: DeepSeek-OCR Issue #151 (26.10.2025): 55/600 British-Library-Scans mit Loops (Output 3–5× GT-Länge), repetition_penalty/no_repeat_ngram wirkungslos; auf Erfolgsfällen CER 6,11 %. Chandra Issue #51 (nachverifiziert): „5-6 of them on average always have token loop hallucination" bei ~60 Dokumenten. dots.ocr-Modellkarte (nachverifiziert): „Continuous special characters, such as ellipses (`...`) and underscores (`_`), may cause the prediction output to repeat endlessly."
**Konfidenz**: hoch — unabhängige quantifizierte Messung (DeepSeek), Nutzerbericht (Chandra), Hersteller-Selbstauskunft (dots.ocr).
**Gilt unter**: schwere/degradierte Scans bzw. Formularlinien; gemessen auf EN/RU/KK — Übertragbarkeit auf deutsche Alt-Scans plausibel, nicht belegt.
**Nachmessbar durch**: 200 gemischte deutsche Seiten, Loop-Rate zählen (Detektor: Output-Länge > 2× erwartet, n-Gramm-Wiederholung).
**Widerspruch**: keiner gefunden. Konsequenz für die Zielfunktion „nichts erfunden": Ein Loop-/Längen-Detektor ist Pflicht, egal welches Modell.

### B-32 — Die VLM-Lizenzlandschaft hat sich gedreht: Frei kommerziell nutzbar und multilingual sind vor allem dots.ocr (MIT), PaddleOCR-VL (Apache, 109 Sprachen inkl. DE), LightOnOCR-2 (Apache) und Qwen3-VL (Apache); die Modelle mit den besten publizierten Deutsch-Werten (Datalab: Surya/Chandra) haben restriktive Gewichte-Lizenzen, Nanonets-OCR2-3B ist entgegen verbreiteter Darstellung research-only, und das MinerU2.5-VLM-Modell steht auf HF unter AGPL-3.0.
**Beleg**: dots.ocr-HF (nachverifiziert): MIT; PaddleOCR-VL-HF: Apache 2.0, Sprachliste mit „German"; LightOnOCR-2-1B: Apache 2.0; Qwen3-VL-8B: Apache 2.0 (9B Parameter → auf 12 GB nur quantisiert, GGUF Q4 ≈ 6,1 GB); Nanonets-OCR2-3B-Diskussion (Maintainer, 10/2025): „This one is Qwen's research license", nur 1.5B-exp ist Apache; MinerU2.5-2509-1.2B-HF-Karte: license AGPL-3.0 (obwohl der MinerU-Code 2026 zu Apache-basiert wechselte, B-60).
**Konfidenz**: hoch — Lizenzangaben direkt von Modellkarten/Diskussionen.
**Gilt unter**: Stand 30.07.2026; „frei" bezieht sich auf die Gewichte, nicht auf Qualitätsgarantien.
**Nachmessbar durch**: LICENSE-Angaben der jeweiligen HF-Modellkarten abrufen.
**Widerspruch**: Blogposts führen Nanonets-OCR2 pauschal als „open source" — die Research-Lizenz der 3B-Variante widerspricht dem.

### B-33 — MinerU passt auf die A2000 nur in der reinen VLM-Variante sicher (MinerU2.5 1,2B: ~4–5 GB VRAM, offizielle Untergrenze 8 GB+); das seit v2.7 voreingestellte Hybrid-Backend lädt mehrere Modelle parallel und lief auf 8 GB in OOM — 12 GB sind ungeklärt.
**Beleg**: MinerU-Doku (nachverifiziert): vllm-Beschleunigung „Volta … 8GB+ VRAM"; Nutzerbericht nite07.com (23.04.2026): hybrid-auto-engine OOM auf 8 GB, vlm-auto-engine 4–5 GB; Durchsatzreferenz Maintainer: 1–2 Seiten/s auf A10; Nutzer: ~5 s/Seite auf RTX A2000 8GB (Discussion #1226).
**Konfidenz**: mittel — offizielle Untergrenze + unabhängiger Einzelbericht; keine 12-GB-Referenzmessung.
**Gilt unter**: MinerU ≥2.7/3.x, vlm-Backend, BF16; Deutsch unevaluiert (B-30); Modell-Lizenz B-32.
**Nachmessbar durch**: mineru mit vlm- vs. hybrid-Backend auf der A2000; VRAM-Peak und Seiten/Minute auf deutschen PDFs.
**Widerspruch**: keiner gefunden.

### B-34 — GPU-Passthrough unter Docker ist 2026 konsolidiert, aber nicht wartungsfrei: nvidia-container-toolkit 1.19.1 mit CDI als Default (seit 1.18.0, Legacy deprecated, Docker ≥26.1 nötig) brachte eine dokumentierte Regressionswelle; der klassische „Failed to initialize NVML"-Bruch nach `systemctl daemon-reload` ist ein verstandener Legacy-Defekt mit offiziellen Workarounds; die A2000 (Ampere) ist nicht abkündigungsgefährdet (580er-Legacy trifft nur Maxwell/Pascal/Volta); Hauptbruchpunkt bleiben Kernel-Updates (DKMS/Secure-Boot), wogegen Ubuntus vorkompilierte, signierte NVIDIA-Module der dokumentierte Routine-Ausweg sind.
**Beleg**: Toolkit-Release-Notes (Seite akt. 22.07.2026): v1.19.1; v1.18.0 CDI-Default, Legacy deprecated, CDI v0.7.0 braucht Docker ≥26.1.0; Regressionen Issues #1456 (CUDA-Libs nach CDI-Umstieg nicht gefunden), #1487 (Rootless-BPF); NOTICE-Issue #48 (NVML/daemon-reload, udev-Symlink-Fix via `nvidia-ctk system create-dev-char-symlinks`); Phoronix (01.07.2025): „release 580 series will be the last to support … Maxwell, Pascal, and Volta" — Ampere nicht betroffen; Troubleshooting-Guide oneuptime (02.03.2026): „Kernel updates are among the most common triggers for NVIDIA driver failures"; NVIDIA-Doku zu precompiled/Canonical-signed drivers.
**Konfidenz**: hoch für Versionsstand/CDI/580er-Aussage (Primärquellen); mittel für die Häufigkeitseinordnung der Regressionen.
**Gilt unter**: Linux-Host (Ubuntu/Mint-Klasse), aktueller Treiberzweig, A2000.
**Nachmessbar durch**: Toolkit im CDI-Mode auf der Mintbox einrichten, `docker run --gpus all … nvidia-smi`, dann gezielt `systemctl daemon-reload` + zwei Kernel-Update-Zyklen mit vorkompilierten Modulen durchtesten.
**Widerspruch**: die Dichte der 2026er „NVIDIA nach Kernel-Update reparieren"-Artikel widerspricht der Erzählung, GPU-Hosting sei wartungsfrei — es ist Routine mit bekannten Handgriffen.

### B-35 — Für die A2000 selbst existiert keine einzige publizierte Durchsatzmessung eines Dokument-VLM-Stacks: Alle gefundenen Zahlen stammen von A100/H100/B200/RTX 3090/5090 (Beispiele: MinerU 2,12 fps auf A100; Chandra 1,44 S./s auf H100; Surya 5,35 S./s auf RTX 5090) — Seiten/Minute auf der Zielhardware sind nur durch eigene Messung zu bekommen.
**Beleg**: Negativbefund über alle in Block 2 zitierten Quellen (30.07.2026); die genannten GPU-Referenzen aus den jeweiligen READMEs/Modellkarten. Die A2000 ist eine 70-W-Ampere-Karte deutlich unter den Benchmark-GPUs.
**Konfidenz**: hoch als Negativbefund im Rahmen dieser Recherche.
**Gilt unter**: alle GPU-Kandidaten aus B-28–B-33.
**Nachmessbar durch**: 100-Seiten-Testlauf je Kandidat auf der A2000 (vLLM, Batch 1–4), Seiten/Minute und Watt protokollieren.
**Widerspruch**: keiner — die Lücke ist der Befund.

### Block 3 · Die Formate, die heute versagen

### B-36 — Für DOCX→Markdown ist pandoc 2026 der vollständigste Reader: Fußnoten UND Endnoten (still zusammengelegt), OMML-Mathe, Bilder, Überschriften per Style, Änderungsverfolgung steuerbar (`--track-changes accept|reject|all`); dokumentierte Schwächen liegen bei verbundenen Zellen im Markdown-Writer (der Reader kann Spans, die MD-Writer geben sie kaum aus) und bei SDT-Inhaltssteuerelementen/Word-Bibliografien.
**Beleg**: Quellcode-Header des offiziellen DOCX-Readers (jgm/pandoc, main, 30.07.2026): „[X] Math", „[X] Note (Footnotes and Endnotes are silently combined.)", „[-] Table (column widths and alignments not yet implemented)"; Manual (pandoc 3.10.1, 21.07.2026): `--track-changes`-Optionen; Issue #6316/#8346: Reader unterstützt rowspan/colspan seit PR #6512, „writer support is more limited"; Dritt-Matrix (docx2md-cli): pandoc „Bibliography/SDT: ❌", „vMerge im MD-Output: ❌".
**Konfidenz**: hoch für Feature-Existenz (Quellcode + Manual); mittel für die Schwächenliste (teils vom Konkurrenz-Tool-Autor).
**Gilt unter**: pandoc 3.10.1, DOCX→gfm/markdown; Verlust der Fußnote/Endnote-Unterscheidung ist dokumentiert.
**Nachmessbar durch**: DOCX mit Fuß-/Endnoten, OMML-Formel, vMerge-Tabelle und Track-Changes durch `pandoc -f docx -t gfm --track-changes=all`; prüfen, was ankommt.
**Widerspruch**: intern dokumentiert (Reader kann Spans, End-zu-End-MD verliert sie).

### B-37 — Die zwei „modernen" Parser verwerfen DOCX-Fußnoten komplett (Quellcode-belegt: weder docling noch unstructured extrahieren footnote/endnote), während docling dafür kann, was pandoc/mammoth nicht können: OMML→LaTeX, Zell-Spans ins Tabellenmodell, Textboxen, outlineLvl-Überschriften.
**Beleg**: docling `msword_backend.py` (main, 30.07.2026): `oMath2Latex`, `col_span=cell.grid_span`, Textbox-Handling, kein Vorkommen von footnote/endnote; Issue #1250 (offen): Word-Zitate/SDT verschwinden. unstructured `partition/docx.py` (main): null Treffer „footnote"; Kategorien Title/ListItem/Table/Header/Footer, Tabellen-HTML in `metadata.text_as_html`, Listentiefe nur als `category_depth`-Metadatum.
**Konfidenz**: hoch — direkter Quellcode-Review beider Backends.
**Gilt unter**: docling ≤2.101+, unstructured main (Stand 30.07.2026). Für deutsche Verträge/Verwaltungsdokumente mit Fußnoten ist das ein harter Treueverlust.
**Nachmessbar durch**: DOCX mit 3 Fußnoten durch beide Tools; grep auf die Fußnotentexte (erwartet: fehlen).
**Widerspruch**: keiner gefunden — die Doku beider Projekte bewirbt DOCX-Support, ohne den Fußnotenverlust zu nennen (Auslassung).

### B-38 — markitdown ist bei DOCX nur ein dünner mammoth→HTML→Markdown-Wrapper: mammoth deklariert Fußnoten/Endnoten als unterstützt, ignoriert aber Tabellen-Formatierung, und mammoths eigener Markdown-Pfad ist offiziell deprecated — die Qualität ist formatweise extrem ungleich (bei PPTX ist markitdown strukturell stark, B-39).
**Beleg**: markitdown `_docx_converter.py` (main): `mammoth.convert_to_html` → HTML-Converter, keine eigene Fußnoten-/Mathe-Logik; mammoth-README: „Footnotes and endnotes" unterstützt, „The formatting of the table itself … is currently ignored", „Markdown support is deprecated"; markitdown-README: „may not be the best option for high-fidelity document conversions".
**Konfidenz**: hoch für Pipeline und Feature-Existenz (Quellcode + READMEs); niedrig für die genaue Fußnoten-Darstellung im Endergebnis (nicht getestet).
**Gilt unter**: markitdown 0.1.6 + mammoth 1.11.
**Nachmessbar durch**: deutsche DOCX mit Fußnoten, 3-Ebenen-Liste, vMerge-Tabelle, Formel durch `markitdown`; Verluste zählen.
**Widerspruch**: markitdown-README verspricht Erhalt von „headings, lists, tables, links" — steht in Spannung zur eigenen High-Fidelity-Warnung.

### B-39 — Bei PPTX sind markitdown und docling die zwei ernsthaften XML-Leser mit komplementären Fehlerprofilen: markitdown sortiert Shapes nach Position (top→left), löst Gruppen rekursiv, hängt Speaker-Notes als „### Notes:" an und wandelt Charts in Datentabellen; docling trägt Notes in einen eigenen NOTES-Layer (Default-Export ins Markdown unklar), Zell-Spans ins Tabellenmodell und Bilder ins Dokumentmodell, iteriert aber in XML-Reihenfolge statt nach Position; unstructured hat Notes per Default AUS; pandoc liest PPTX erst seit 3.8.3 (12/2025) und rudimentär; SmartArt-Text verliert JEDES der vier Tools (python-pptx-Grenze).
**Beleg**: Quellcode-Reviews (alle main, 30.07.2026): markitdown `_pptx_converter.py` (Positions-Sortierung, Gruppen-Rekursion, „### Notes:", Charts→Tabelle, keine SmartArt-Logik); docling `mspowerpoint_backend.py` (notes_slide→NOTES-Layer, rowSpan/gridSpan, `for shape in slide.shapes` ohne Sortierung); unstructured `partition/pptx.py` (`include_slide_notes` Default False, Positions-Sortierung); pandoc-Releases 3.8.3 (01.12.2025): „Add `pptx` … as new input format" — im Slides-Modul weder Notes- noch Tabellen-Handling sichtbar.
**Konfidenz**: hoch für die Fähigkeiten (Quellcode); die Frage, welche Lesereihenfolge-Philosophie bei realen Decks öfter richtig liegt, ist unbelegt.
**Gilt unter**: PPTX→Markdown, Stand 07/2026; sprachneutral.
**Nachmessbar durch**: zweispaltiges Deck mit Gruppe, SmartArt, Notes und Folientabelle durch alle vier Tools; Reihenfolge, Notes-Abdeckung, SmartArt-Verlust prüfen.
**Widerspruch**: keiner gefunden.

### B-40 — Für die These „VLM über Folien-Screenshots schlägt XML-Extraktion" gibt es keinen Beleg — die einzige systematische Evaluation zeigt das Gegenteil auf Strukturebene („VLMs underperform on pixel-accurate extraction"), und Speaker-Notes sind auf Screenshots prinzipbedingt unsichtbar.
**Beleg**: VLM-SlideEval (arXiv:2510.22045, 10/2025): VLMs „underperform on pixel-accurate extraction … while performing better on single-slide content understanding; however, they do not reliably capture narrative structure across slides."
**Konfidenz**: mittel — ein Paper mit anderem Primärziel, Modellstand Ende 2025; VLM-Ansätze sind zudem sprachabhängig (deutsche Folien = Zusatzrisiko), XML-Extraktion sprachneutral.
**Gilt unter**: PPTX-Strukturextraktion aus gerenderten Folien.
**Nachmessbar durch**: 10 eigene deutsche Folien: markitdown-Output vs. VLM-Screenshot-Beschreibung; Notes-Abdeckung und Tabellenzellen-Treffer zählen.
**Widerspruch**: Marketing der VLM-Parsing-Anbieter — ohne PPTX-spezifische Messung.

### B-41 — Bei XLSX erhält kein Kandidat die Verbundzellen-Semantik: markitdown konvertiert alle Sheets („## Sheetname") über pandas mit Formeln-als-cached-Werten (leere Zellen, wenn der Formel-Cache fehlt), docling „löst" Merged Cells dokumentiert durch Plätten („flatten merged cells", vorher Crash-Quelle), unstructured ist der einzige mit Subtabellen-Erkennung (Connected Components) je Sheet; docling ist beim Formatspektrum konkurrenzlos (Legacy .doc/.xls/.ppt via LibreOffice, ODF, CSV), was aber die LibreOffice-Abhängigkeit ins Image holt.
**Beleg**: markitdown `_xlsx_converter.py` (pd.read_excel sheet_name=None → to_html, keine Merge-Logik); docling-CHANGELOG v2.64.0 (02.12.2025): „Improve Excel table bounds detection and flatten merged cells", v2.66.0: IndexError-Fix bei Merged Cells; unstructured `partition/xlsx.py`: „Connected-components … to detect contiguous groups of non-empty cells"; docling-Formatliste (offiziell): DOC/XLS/PPT „requires LibreOffice".
**Konfidenz**: hoch — Quellcode/Changelogs/offizielle Doku.
**Gilt unter**: XLSX/CSV→Markdown, Stand 07/2026; sprachneutral.
**Nachmessbar durch**: XLSX mit 2 Sheets, Formeln, verbundenem Kopfbereich durch alle drei; zusätzlich programmatisch erzeugte XLSX ohne Formel-Cache (erwartet: leere Zellen bei pandas-Pfad).
**Widerspruch**: keiner gefunden.

### B-42 — Es existiert kein methodischer öffentlicher Benchmark für Office→Markdown-Strukturerhalt: alle gefundenen „Vergleiche" sind SEO-/Marketing-Texte ohne Testkorpus; die Feature-Aussagen in B-36–B-41 stammen aus Quellcode und Doku, nicht aus gemessener Treue.
**Beleg**: Negativbefund nach ~10 einschlägigen Suchen (30.07.2026); geprüfte Kandidaten (file2markdown.ai, glukhov.org u. a.) ohne Methodik; akademisch nur VLM-SlideEval (Slide-Verständnis, nicht Konvertierung).
**Konfidenz**: mittel-hoch — Negativbefund; ein übersehener Nischen-Benchmark ist nicht ausschließbar.
**Gilt unter**: DOCX/PPTX/XLSX→Markdown, öffentliche Quellen bis 30.07.2026.
**Nachmessbar durch**: eigener Mini-Benchmark (10 repräsentative Bestandsdokumente, Checkliste: H-Ebenen, Listentiefe, Zellen, Fußnoten, Notes) — derzeit die einzige belastbare Entscheidungsbasis.
**Widerspruch**: keiner — die Abwesenheit ist der Befund.

### B-43 — Für HTML-Boilerplate-Entfernung bleibt trafilatura die Referenz (aktiv, v2.1.0 vom 07.06.2026, Apache-2.0, Markdown-Ausgabe eingebaut, offline): Auf dem deutschlastigen Eigen-Korpus F1 0,909 (deutlich vor readability-lxml 0,801, goose3 0,793, jusText 0,742), auf dem unabhängigen englischen WCXB-Benchmark 2026 F1 0,791/0,841 — dort ist resiliparse der Geschwindigkeits-Sweetspot (F1 0,797 bei 28 ms/Seite) und Readability abgeschlagen (0,674).
**Beleg**: PyPI trafilatura (nachverifiziert): 2.1.0, 07.06.2026, Apache-2.0; trafilatura-Evaluationsseite (Messung 2022, 750 überwiegend deutsche Seiten, Skripte offen): F1 0,909; WCXB (webcontentextraction.org + arXiv:2605.21097, 05/2026, 2.008 Seiten, 7 Seitentypen, englisch): Dev-Set trafilatura 0,791 (97 ms), resiliparse 0,797 (28 ms), Readability 0,674 (785 ms); Held-out: trafilatura 0,841. Caveat: WCXB-Autor ist zugleich Autor des Erstplatzierten rs-trafilatura (0,859/0,903).
**Konfidenz**: mittel — Eigen-Eval (2022, alte Version) plus unabhängiger, aber interessenbehafteter 2026er-Benchmark; die Richtung ist konsistent.
**Gilt unter**: HTML-Artikel-/News-Extraktion; DE-Zahl von 2022 (trafilatura 1.2.2); WCXB explizit ohne Deutsch.
**Nachmessbar durch**: WCXB-Harness (Datensatz CC-BY-4.0) lokal + eigenes 50-Seiten-Sample deutscher Newsletter/News annotieren.
**Widerspruch**: intern dokumentiert (Korpuszusammensetzung dreht die Readability-Relativwerte).

### B-44 — „LLM/SLM liest rohes HTML" schlägt gute Heuristiken auf unabhängigen Daten nicht und kostet 16–375× mehr Rechenzeit: MinerU-HTML (0,6B) F1 0,827 bei 1.570 ms/Seite und ReaderLM-v2 0,741 bei 10.410 ms vs. Heuristiken 0,79–0,86 bei 28–97 ms; der Hersteller-Benchmark (WebMainBench: MinerU-HTML 0,90 vs. trafilatura 0,64) widerspricht dem diametral; ReaderLM-v2 ist zudem CC-BY-NC (kommerziell gesperrt), Firecrawl ist AGPL und für reine LAN-Konvertierung Overkill.
**Beleg**: WCXB-Paper (arXiv:2605.21097): „Neural systems inherit the same article bias as heuristic systems"; MinerU-HTML-Repo (WebMainBench v1.1, 19.03.2026): ROUGE-N.f1 0,9001 vs. trafilatura 0,6402; ReaderLM-v2-HF: CC-BY-NC-4.0, 29 Sprachen inkl. Deutsch; Firecrawl-Repo: „primarily AGPL-3.0", self-hostbar.
**Konfidenz**: mittel — zwei sich widersprechende Benchmarks (unabhängig-englisch vs. Hersteller-gemischt), beide markiert.
**Gilt unter**: Boilerplate-Entfernung 2026; MinerU-HTML wäre offline möglich (Apache, CPU langsam).
**Nachmessbar durch**: beide Harnesses auf identischem deutschem 100-Seiten-Sample; WCXB-Metrik als Schiedsrichter.
**Widerspruch**: direkt dokumentiert (0,9001 vs. 0,827 für dasselbe Tool).

### B-45 — Für E-Mail existiert keine fertige Kette EML→sauberes Markdown: unstructured partitioniert Header/Multipart/Anhänge sauber, entfernt aber keine Signaturen/Zitate; markitdown kann bis heute kein EML (Issue offen, PR liegengeblieben); Mailguns talon ist seit 2016 tot und englischtrainiert; das einzige gepflegte, deutschfähige Werkzeug für Zitat-/Signatur-Ketten ist mail-parser-reply (MIT, v1.36 vom 01.12.2025, 13 Sprachen inkl. „Am … schrieb …"-Mustern) — ein Einzelmaintainer-Projekt.
**Beleg**: unstructured-Partitioning-Doku (30.07.2026): partition_email mit Header-Elementen, `process_attachments`, keine Signatur-/Zitatentfernung; markitdown Issue #89 + PR #271 (Merge-Konflikte); talon: letztes Release 04/2016, SVM auf ENRON; PyPI mail-parser-reply: v1.36, MIT, „detect headers, signatures and disclaimers", Sprachen inkl. de.
**Konfidenz**: hoch — Primärdoku und Registry-Stände direkt gelesen; für die Trefferquote von mail-parser-reply auf Deutsch existiert keine unabhängige Messung (mittel).
**Gilt unter**: EML/MSG→Markdown; Kompositionspflicht: Struktur-Parser + Reply-Splitter + HTML-Teil durch den HTML-Pfad.
**Nachmessbar durch**: 30 reale deutsche Reply-Ketten (Outlook/Gmail/Apple-Mail) durch die Kette; Zitat-/Signaturreste zählen.
**Widerspruch**: keiner gefunden.

### B-46 — Bei Formel-Erkennung sind die klassischen Spezialwerkzeuge tot oder eingefroren, und das integrierte docling-Enrichment ist unbelegt: pix2tex kollabiert außerhalb sauberer Drucke (BLEU 0,092 auf Screenshots, 0,012 Handschrift), texify ist archiviert (01/2025, Nachfolger in surya), UniMERNet ist das stärkste offene Modell (BLEU 0,616/0,921 auf denselben Kategorien) aber seit 12/2024 ohne Release; docling-CodeFormula (0,2B, MIT) hat keinerlei publizierte Qualitätszahlen und dokumentierte Praxiskosten (Laufzeit 80→240 s/PDF auf CPU, RAM-Spitzen).
**Beleg**: UniMERNet-Paper (arXiv:2404.15254, Tab. 5): SPE/CPE/SCE/HWE — UniMERNet 0,917/0,916/0,616/0,921 vs. pix2tex 0,873/0,655/0,092/0,012 vs. texify 0,906/0,690/0,420/0,341; UniMERNet-Repo: Apache-2.0, letztes Release 0.2.3 (26.12.2024); texify-Repo: „archived … January 29, 2025"; CodeFormula-HF-Karte: „Evaluation Metrics: None provided"; docling Discussion #891: 80→240 s/PDF (CPU), Memory-Spikes.
**Konfidenz**: hoch für Paper-Zahlen und Repo-Status; niedrig-mittel für die CodeFormula-Laufzeit (Einzelbericht).
**Gilt unter**: PDF→LaTeX, sprachneutral; CPU-Formelerkennung ist damit faktisch unbelegt bis unpraktikabel.
**Nachmessbar durch**: 100 Formel-Crops aus eigenen PDFs durch CodeFormula, UniMERNet-Tiny und Mathpix; CDM-Toolkit (im UniMERNet-Repo) als Metrik.
**Widerspruch**: keiner gefunden (mangels anderer Messungen zu CodeFormula).

### B-47 — Generalisten-VLMs sind auf Formeln inzwischen konkurrenzfähig bis führend, aber nicht gleichverteilt (olmOCR-Bench „Old Scans Math": Qwen3-VL-Plus 88,0, GPT-5.x 82–85, Gemini-3-Pro nur 75,1; Spezialist Nanonets OCR-3 88,9); als Metrik löst das bildbasierte CDM (CVPR 2025) BLEU/EditDistance ab; kostenseitig ist Mathpix mit ~0,001–0,002 $/Seite nicht teurer als ein Gemini-Flash-Call (~0,003–0,008 $/Seite) — das Argument für Selbst-Hosting ist Datenschutz, nicht Preis.
**Beleg**: IDP-Leaderboard olmOCR-Bench (30.07.2026): Math-Spalte Nanonets OCR-3 88,9; Qwen3-VL-Plus 88,0; GPT-5.4 82,3; Gemini-3-Pro 75,1; GPT-5-Nano 1,7. CDM-Paper arXiv:2409.03643 (CVPR 2025): BLEU „highly sensitive to the distribution of training data". Mathpix-Preisseite (30.07.2026): Convert API „From $0.002/image", Files API „From $1.00 / 1k pages". Gemini-Preise: B-53.
**Konfidenz**: mittel — unabhängiger Benchmark über Sekundär-Leaderboard abgelesen; Preise Herstellerangaben („ab"-Preise).
**Gilt unter**: gescannte/alte Mathematik (härter als born-digital); Formeln sprachneutral.
**Nachmessbar durch**: olmOCR-Bench-Math-Kategorie gegen eigenen Kandidaten; 100-Seiten-Testrechnung über beide APIs.
**Widerspruch**: die verbreiteten Annahmen „Mathpix ist teuer" und „nimm einfach Gemini für Formeln" sind beide nicht gedeckt.

### Block 4 · Architektur und Datenmodell

### B-48 — Als etabliertes Zwischenformat ist DoclingDocument konkurrenzlos: Provenienz (page_no + bbox + charspan) ist Pflichtstruktur je Inhaltselement, das Schema ist versioniert, und die Ökosystem-Traktion ist mit Abstand am größten (LF AI & Data, LangChain/LlamaIndex/Haystack-Integrationen, DocLang-Standardisierung); die Alternativen scheiden strukturell aus — Pandoc-AST kennt weder PDF-Input noch Bbox-Provenienz, MinerU-middle.json ist laut eigener Doku als Schnittstelle instabil (VLM-Output „not backward-compatible"), Marker-JSON liefert Polygone, aber Wort-Bbox+Konfidenz nur als kostenpflichtiges Add-on (0,30 $/1k Seiten).
**Beleg**: DoclingDocument-JSON-Schema (docling-core, 30.07.2026): `ProvenanceItem` mit Pflichtfeldern `page_no`, `bbox`, `charspan`; Wurzel `version`/`schema_name`. Pandoc-Manual: kein PDF in der Inputliste, kein Bbox-Konzept. MinerU-Output-Doku: middle.json trägt `_version_name`, VLM-Backend „not backward-compatible with the pipeline backend"; content_list_v2 „development version, subject to change". Datalab-API-Doku: `word_bboxes` „Billed at $0.30 per 1K pages".
**Konfidenz**: hoch — Schema-Definitionen und offizielle Doku.
**Gilt unter**: docling-core main (07/2026).
**Nachmessbar durch**: ein PDF konvertieren, JSON-Export prüfen: jedes TextItem trägt `prov[0].page_no/bbox`.
**Widerspruch**: keiner gefunden.

### B-49 — Das Degradationssignal muss der Dienst selbst bauen: docling liefert Konfidenzen (layout/ocr/parse/table-Score) nur auf Seiten-/Dokumentebene im ConversionResult — sie stehen NICHT im serialisierten DoclingDocument und gehen beim reinen Dokument-Export verloren; ein „modellgeneriert"-Flag je Block existiert in keinem der etablierten Formate; die einzige Block-Konfidenz im Feld ist unstructureds `detection_class_prob`.
**Beleg**: docling-Confidence-Doku (ab v2.34.0): Scores im „confidence field of the ConversionResult object", `table_score` „noch nicht implementiert"; Schema-Prüfung DoclingDocument.json: kein confidence-Feld auf Item-Ebene, `content_layer`-Enum (body/furniture/notes/…); unstructured-Elements-Doku: `detection_class_prob`, `coordinates.points`, `page_number`, `text_as_html`.
**Konfidenz**: hoch — Existenz/Nicht-Existenz aus Schema und offizieller Doku.
**Gilt unter**: docling ≥2.34, unstructured 0.22+.
**Nachmessbar durch**: Scan-PDF konvertieren; prüfen, dass `ConversionResult.confidence.pages` gefüllt ist, das exportierte Dokument-JSON aber keine Item-Konfidenz enthält.
**Widerspruch**: docling-serve Issue #613 fordert Provenienz-Durchreichung in Antworten — die Lücke ist dem Projekt bekannt.

### B-50 — docling-serve ist ein fertiges, passendes Serving-Layer (MIT, v1.25.0: Sync-, Async- und WebSocket-API, drei Queue-Engines, API-Key-Auth per `DOCLING_SERVE_API_KEY`/X-Api-Key-Header, Seiten-/Größen-Limits konfigurierbar, CPU-Image 4,4 GB) — mit einem realen Betriebsrisiko: einer Serie offener Memory-Leak-/Crash-Issues im Dauerbetrieb (RAM wächst pro Task bis zum Container-Limit).
**Beleg**: docling-serve-Repo + docs/usage.md + docs/configuration.md (30.07.2026, Auth+Images nachverifiziert): Endpoints `/v1/convert/source|file` (+ async-Varianten laut usage.md, Status-Polling, WebSocket), Engines local/RQ/KFP, `DOCLING_SERVE_MAX_NUM_PAGES`/`MAX_FILE_SIZE`; Image-Tabelle 4,4/8,7/11,4 GB. Issues #474 (RAM wächst bis Limit 24–32 GB → Neustart), #366, #233 (GPU-Leak), #389, #542; Upstream docling #1343 (EasyOCR-Leak).
**Konfidenz**: hoch für Features/Lizenz/Images (nachverifiziert); mittel für die Leak-Einordnung (mehrere unabhängige Berichte, Fix-Status offen).
**Gilt unter**: docling-serve ≤v1.25.0, Docker.
**Nachmessbar durch**: Soak-Test 500 PDFs seriell, RSS-Verlauf plotten; Gegenmaßnahme (Memory-Limit + Worker-Recycling) direkt mittesten.
**Widerspruch**: aktive Weiterentwicklung (60 Releases) spricht für laufende Fixes.

### B-51 — Die Serving-Alternativen decken das Anforderungsprofil schlechter ab: unstructured-api ist strategisch zweitrangig (Hersteller-Downgrade der OSS-Linie, B-18); Apache Tika Server (347 MB, Apache-2.0) maximiert Formatbreite, liefert aber XHTML/Text ohne Bboxes und ohne Markdown; MinerU bietet kein offizielles vorgebautes Image und keine dokumentierte Auth (Selbstbau auf vllm-openai-Basis, 8 GB+ VRAM); marker bringt nur einen simplen `marker_server` (FastAPI) mit Gewichte-Lizenzfrage.
**Beleg**: unstructured-api-Repo (0.1.2, 03.04.2026, Apache-2.0, README verweist auf Hosted-API); Tika-Projektseiten (3.3.0 stabil, „over a thousand different file types", Ausgabe text/html/xhtml + /rmeta-JSON); MinerU-Docker-Doku (Basis `vllm/vllm-openai`, Profile openai-server/api/gradio, keine Auth erwähnt); marker-README (`marker_server --port 8001`).
**Konfidenz**: hoch — offizielle Doku aller vier.
**Gilt unter**: Stand 30.07.2026.
**Nachmessbar durch**: Tika-full ziehen und ein PDF an /tika schicken (Strukturverlust sichtbar); MinerU-API ohne Key von zweitem Host aufrufen (keine Zugriffskontrolle).
**Widerspruch**: keiner gefunden.

### B-52 — Gemini verarbeitet PDFs nativ (bis 50 MB / 1.000 Seiten) und macht den Bestands-Ansatz „selbst gerenderte PNGs" doppelt nachteilig: PDF-Seiten laufen bei Gemini-3-Modellen als steuerbare Bild-Tokens (media_resolution LOW 280 / MEDIUM 560 / HIGH 1120 je Seite, Default 560) PLUS kostenlosem nativem Textlayer („You are not charged for tokens originating from the extracted native text in PDFs") — und Google erklärt selbst MEDIUM zum Sättigungspunkt für Dokumentverständnis („quality typically saturates at medium").
**Beleg**: Gemini-Doku document-processing (nachverifiziert): 50 MB/1.000 Seiten; Native-Text-Gratis-Zitat; ältere „258 tokens/Seite"-Angabe (2.x-Generation). media-resolution-Doku (nachverifiziert): Tabelle 280/560/1120 „+ Native Text", Default UNSPECIFIED=560; Empfehlungstabelle „PDFs → MEDIA_RESOLUTION_MEDIUM … quality typically saturates at medium".
**Konfidenz**: hoch für Limits/Token/Abrechnung (offizielle Doku, nachverifiziert); die Sättigungsaussage ist eine Herstellerangabe (mittel). Ein unabhängiger A/B-Test nativ-PDF vs. selbstgerenderte PNGs existiert nicht (Teil 3).
**Gilt unter**: Gemini API, Gemini-3-Modellfamilie; abweichend Gemini 2.5 („256 + OCR" je Seite).
**Nachmessbar durch**: `countTokens` auf dasselbe Dokument einmal als PDF, einmal als PNG-Serie; Token-/Kosten-/Qualitätsdifferenz messen.
**Widerspruch**: Tokenzahl je Seite widersprüchlich zwischen zwei offiziellen Google-Seiten (258 vs. 560) — vermutlich Generationen-Mischung; dokumentiert, per countTokens auflösbar.

### B-53 — Preisstand 30.07.2026 (Gemini API, Paid): Gemini 2.5 Flash-Lite 0,10/0,40 $ und 2.5 Flash 0,30/2,50 $ pro 1M In/Out-Tokens; aktuellste Flash-Stufen 3.5 Flash 1,50/9,00 $ und 3.6 Flash 1,50/7,50 $; eine PDF-Seite (560 Bild-Tokens + Gratis-Text) kostet damit input-seitig ~0,00017–0,00084 $ — und: das im Bestand verwendete gemini-2.0-flash ist laut Preisseite „deprecated and has been shut down June 1, 2026". Claude rechnet PDF-Seiten dagegen als Bild UND Text voll ab (1.500–3.000 Text-Tokens/Seite; Limits 32 MB, 600 Seiten bzw. 100 bei <1M-Kontext) und ist damit strukturell teurer je Seite.
**Beleg**: Gemini-Preisseite (nachverifiziert 30.07.2026, inkl. wörtlich: „Gemini 2.0 Flash is deprecated and has been shut down June 1, 2026."); Claude-PDF-Doku (nachverifiziert): 32 MB, 600/100 Seiten, „1,500–3,000 tokens per page" + Bild-Tokens nach Vision-Formel.
**Konfidenz**: hoch — offizielle Preis-/Doku-Seiten, nachverifiziert; die Pro-Seite-Rechnung ist transparente Arithmetik.
**Gilt unter**: Paid Tier, Stand 30.07.2026; Preise ändern sich quartalsweise.
**Nachmessbar durch**: usage_metadata eines realen 100-Seiten-Laufs gegen die Rechnung halten; API-Antwort des Bestands auf das tatsächlich bediente Modell prüfen.
**Widerspruch**: keiner gefunden.

### B-54 — Eine belegte optimale Screenshot-Auflösung für VLM-Dokumentlesen existiert nicht — belegt sind stattdessen harte Provider-Caps und niedrige Arbeitspunkte, die ALLE vier Bestands-DPIs (200/216/288/300) obsolet machen: Claude skaliert auf max. 1568 px lange Kante herunter (High-Res-Tier 2576 px ab Claude 4.7; Token-Formel ⌈w/28⌉×⌈h/28⌉), Gemini intern auf 768–3072 px, und die spezialisierten Open-Source-OCR-VLMs arbeiten bei 1024–1288 px Längskante (≈88–110 dpi für A4); A4@300 dpi (3.508 px) wird überall heruntergerechnet, A4@200 dpi überschreitet bereits Claudes Standardgrenze.
**Beleg**: Anthropic-Vision-Doku (nachverifiziert): Standard 1568 px/1568 Visual-Tokens, High-Res „Claude 4.7 and later" 2576 px/4784 Tokens, Patch-Formel ⌈w/28⌉×⌈h/28⌉ (die alte /750-Formel ist von der Seite verschwunden); Gemini-Doku: Skalierung auf max 3072×3072 bzw. 768er-Tiles à 258 Tokens; olmOCR-Paper (arXiv:2502.18443): Rendering „maximum dimension of 1024 pixels"; olmOCR-2-Modellkarte: 1288 px; DeepSeek-OCR-Paper (arXiv:2510.18234): Auflösungs-Moden 512–1280 px + „Gundam" nur für Zeitungs-Kleindruck; Präzision 97 % unter 10× Token-Kompression, ~60 % bei 20×.
**Konfidenz**: hoch für Caps und Arbeitspunkte (offizielle Doku/Paper, nachverifiziert); hoch für die Nicht-Existenz einer öffentlichen DPI→Genauigkeits-Kurve im Rahmen dieser Recherche.
**Gilt unter**: Claude-/Gemini-API 07/2026, olmOCR v1/v2, DeepSeek-OCR; inhaltsabhängige Ausnahme: dichter Kleindruck braucht mehr.
**Nachmessbar durch**: identisches A4-Testset bei 100/134/150/200/300 dpi rendern (134 dpi füllt Claudes 1568-px-Kante exakt), CER/Tabellen-F1 je Stufe — die eigene Kurve ersetzt die fehlende öffentliche.
**Widerspruch**: keiner — das Fehlen der Kurve ist der Befund; „mehr DPI = besser" ist durch nichts gedeckt.

### Block 5 · Vertrauen und Recht

### B-55 — Produktreife Halluzinationserkennung in VLM-Konvertaten besteht 2026 aus Pipeline-Heuristiken, nicht aus Modell-Introspektion: olmOCR setzt n-Gramm-Repetitionsdetektion (>30 wiederholte Gramme), Charset-/Sprachwechsel-Checks, Retries mit Temperatur-Variation und eine dokumentweite Fehlerquote (`--max_page_error_rate`) ein; gemessen wird Halluzination über Unit-Test-Benchmarks (olmOCR-Bench „Text Absence"-Klasse; ParseBench 04/2026 „Content Faithfulness" inkl. fabrizierter Inhalte) — OmniDocBench misst sie nicht.
**Beleg**: olmOCR-Repo/Paper (arXiv:2502.18443 App. D.2): Repetitions-/Charset-Checks, Retry-Strategien; v0.3.0-Changelog: Fix „hallucinations on blank documents"; olmOCR-Bench-README: Testklasse „Text Absence" („text does NOT appear"); ParseBench (arXiv:2604.08538): „detecting both omissions … and hallucinations (fabricated content)", ~2.000 Seiten, bestes System LlamaParse Agentic 84,88 %.
**Konfidenz**: hoch — offizielle Repos/Paper.
**Gilt unter**: olmOCR-Pipeline (Apache-2.0, nachbaubar); Benchmarks englisch.
**Nachmessbar durch**: die olmOCR-Guards (Längen-Check, n-Gramm-Detektor, Charset-Filter) als Pipeline-Stufe nachbauen und an 200 eigenen Seiten die Trigger-Rate messen.
**Widerspruch**: keiner gefunden.

### B-56 — Die kommerziellen Extraktionsdienste liefern produktreife Element-Konfidenzen, die als Degradationssignal direkt nutzbar sind: Azure Document Intelligence 0–1 je Wort/Feld/Zelle (Schwellen-Empfehlung ≥0,8, für Finanz-/Medizindaten „close to 100 %", sonst Human-Review), AWS Textract 0–100 je Block (Text UND Geometrie), Mathpix Ergebnis- plus Per-Character-Konfidenz.
**Beleg**: Microsoft-Learn accuracy-confidence (ms.date 08.04.2026): Definition + Schwellenempfehlung, ab API 2024-11-30 auch Tabellen/Zellen; AWS-Textract-API-Doku (Block.Confidence 0–100); Mathpix-API-Doku (`confidence_threshold`, `confidence_rate_threshold`).
**Konfidenz**: hoch — offizielle Doku; Konfidenzen sind Selbstauskünfte der Anbieter, keine kalibrierte Garantie.
**Gilt unter**: Azure DI ≥2024-11-30, Textract aktuell, Mathpix v3.
**Nachmessbar durch**: 20 deutsche Testseiten durch Azure DI; Korrelation der Konfidenzen mit tatsächlichen Fehlern stichprobenhaft prüfen.
**Widerspruch**: keiner gefunden.

### B-57 — Self-Consistency und Cross-Model-Agreement sind Forschung, nicht Produktpraxis — und Agreement ist nur ein schwacher Korrektheits-Proxy: „Consensus Entropy" (Multi-VLM-Divergenz) zeigt Gewinne (+8,2 % OCR bei 7,3 % Routing), aber ein Audit von 07/2026 misst nur ρ=0,20–0,59 zwischen Agreement und Korrektheit und findet konfidente Fehler, die über Anbieter hinweg (GPT und Claude) geteilt werden; Logprobs sind bei Gemini offiziell verfügbar (Vertex, seit 07/2025), aber für OCR-Halluzinationsdetektion existiert keine Kalibrierungs-Literatur.
**Beleg**: arXiv:2504.11101 (Consensus Entropy); arXiv:2607.08065 (07/2026): „Agreement is not accuracy"; Google-Developers-Blog (16.07.2025): `response_logprobs` auf Vertex; NeurIPS-2025-Paper arXiv:2506.20168 (KIE-HVQA, Refusal-Training „+22 % hallucination-free accuracy over GPT-4o") als Forschungsstand.
**Konfidenz**: mittel — Paper-Lage jung und teils widersprüchlich; der Negativbefund „nicht produktetabliert" beruht auf Abwesenheit von Belegen.
**Gilt unter**: Forschungsstand 2025–07/2026.
**Nachmessbar durch**: eigenen Zwei-Pass-Vergleich (deterministisch vs. VLM, plus zweites VLM bei Konflikt) an 100 Seiten kalibrieren — der im Auftrag genannte „Vergleich gegen deterministische Extraktion" bleibt der am besten begründete Mechanismus.
**Widerspruch**: intern dokumentiert (Consensus-Entropy-Gewinne vs. Agreement-Audit).

### B-58 — PyMuPDF bleibt dual lizenziert (AGPL-3.0 / kommerzielle Artifex-Lizenz, Preise nur auf Anfrage), und Artifex vertritt öffentlich die breite Auslegung, dass Server-Deployment ohne Offenlegung des eigenen Anwendungs-Quellcodes unter AGPL nicht zulässig ist; der AGPL-§13-Wortlaut selbst knüpft die Netzwerk-Pflicht nur an MODIFIZIERTE Versionen und nur gegenüber den remote interagierenden Nutzern — für einen internen LAN-Dienst ohne externe Nutzer ist die praktische Pflichtenlage eine Interpretationsfrage (keine Rechtsberatung).
**Beleg**: artifex.com/licensing (nachverifiziert, wörtlich): „You cannot deploy our open-source as part of a server-based application or service, without disclosing your own application's full source code under AGPL to any users interacting with it."; Modelle „OEM Distribution"/„Subscription" („per-copy cost with a quarterly minimum fee"), keine öffentlichen Preise. AGPL-3.0 §13 (opensource.org, wörtlich): „…if you modify the Program, your modified version must prominently offer all users interacting with it remotely through a computer network … an opportunity to receive the Corresponding Source…". Vorgelagerte Streitfrage: ob der eigene Dienst ein abgeleitetes Werk wird (Artifex bejaht das faktisch). FSF-FAQ war am Stichtag nicht abrufbar (HTTP 429, Teil 3).
**Konfidenz**: hoch für beide Wortlaute (nachverifiziert); jede Auslegung maximal mittel — Interpretationsfrage, keine Rechtsberatung.
**Gilt unter**: AGPL-3.0 generell → PyMuPDF, Ghostscript, Firecrawl-Kern, MinerU <3.1.
**Nachmessbar durch**: gnu.org-GPL-FAQ (Einträge zu interner Nutzung) abrufen, sobald erreichbar; bei Bedarf Artifex-Angebot einholen.
**Widerspruch**: Artifex' breite Lesart vs. engerer Lizenzwortlaut — der Rechteinhaber ist hier interessengeleitete Partei.

### B-59 — Die AGPL-Kaskade „camelot → Ghostscript" aus der Auftragsprämisse ist für aktuelle Versionen obsolet: Seit camelot v1.0.0 ist pdfium der Default-Backend („This should make the library easier to install"), Ghostscript nur noch optional; Ghostscript selbst bleibt AGPL/kommerziell dual (exklusiv über Artifex).
**Beleg**: camelot-Install-Doku (nachverifiziert): „as of v1.0.0 ghostscript is replaced by pdfium as the default image conversion backend"; ghostscript.com/licensing: „dual licensing model … Licensing is handled exclusively by Artifex".
**Konfidenz**: hoch — offizielle Doku beider Projekte, nachverifiziert.
**Gilt unter**: camelot-py ≥1.0.0; Alt-Installationen (<1.0.0) hängen weiter an Ghostscript. Lizenz von pypdfium2 separat prüfen (BSD/Apache-artig erwartet, hier unbelegt).
**Nachmessbar durch**: im Deployment konfigurierten Backend prüfen.
**Widerspruch**: keiner gefunden.

### B-60 — Die Lizenz-Gesamtlage der Kandidaten, Stand 30.07.2026: permissiv und unproblematisch für einen internen Dienst sind pdfplumber (MIT), pdfminer.six (MIT), docling (MIT), markitdown (MIT), unstructured (Apache-2.0), PaddleOCR (Apache-2.0), Tesseract (Apache-2.0), Apache Tika (Apache-2.0), trafilatura (Apache-2.0 ab v1.8.0, davor GPLv3+), dots.ocr (MIT); prüfpflichtig sind PyMuPDF/Ghostscript (AGPL, B-58), Firecrawl (Kern AGPL-3.0), marker/surya/chandra (Code Apache-2.0, aber Gewichte unter modifiziertem OpenRAIL-M: frei nur unter 5 bzw. 2 Mio. $ Funding/Umsatz plus Nicht-Wettbewerbs-Klausel gegen die Datalab-API); MinerU ist seit v3.1.0 (18.04.2026) NICHT mehr AGPL, sondern „MinerU Open Source License" (Apache-2.0-basiert; Kommerzlizenz erst ab 100 Mio. MAU oder 20 Mio. $ Monatsumsatz; Attributionspflicht bei Online-Diensten für Dritte; automatische Terminierung bei Verstoß) — das separat gehostete MinerU2.5-VLM-Modell trägt auf HF allerdings weiterhin AGPL-3.0.
**Beleg**: alle Lizenzangaben aus den offiziellen Repos/LICENSE-Dateien am 30.07.2026 gelesen, MinerU-LICENSE.md und marker-README nachverifiziert (wörtlich: „free for research, personal use, and startups under $5M funding/revenue"; MinerU: „monthly active users (MAU) exceed 100 million; or … total monthly revenue exceeds USD 20 million"); trafilatura-Repo: „Versions prior to v1.8.0 are under GPLv3+".
**Konfidenz**: hoch für die Lizenztexte (nachverifiziert); mittel für Auslegungen (z. B. ob ein interner Dienst „competitively with our API" sein kann) — Interpretationsfrage, keine Rechtsberatung.
**Gilt unter**: jeweils aktuelle Versionen, Stand 30.07.2026.
**Nachmessbar durch**: LICENSE-Dateien der eingesetzten Versionen im Build pinnen und prüfen.
**Widerspruch**: ältere Berichte (marker GPL-3.0 mit 2-M$-Schwelle; „MinerU = AGPL-Risiko") beschreiben überholte Stände.

### B-61 — Google-Vertragslage für durchlaufende Dokumente (nachverifiziert): Im KOSTENLOSEN Gemini-API-Tarif nutzt Google Inhalte und Antworten zur Produktverbesserung inklusive menschlicher Review; im BEZAHLTEN Tarif ausdrücklich nicht („Google doesn't use your prompts … or responses to improve our products"), Logging nur begrenzt zur Missbrauchserkennung; NEU 2026: Zero-Data-Retention gibt es per Projekt-Approval auch für die Developer-API (nicht mehr nur Vertex), ausgenommen Search-Grounding (30 Tage, nicht abschaltbar); Vertex AI trainiert nicht auf Kundendaten, loggt bei Verdachtsfällen bis zu 90 Tage in der Kundenregion (Ausnahme beantragbar) und bietet EU-Residenz-Optionen.
**Beleg**: ai.google.dev/gemini-api/terms (Effective 23.03.2026, nachverifiziert, wörtliche Zitate); ai.google.dev/gemini-api/docs/zdr (Stand 28.04.2026, nachverifiziert); Vertex-Abuse-Monitoring-Doku (Stand 01.06.2026): „up to 90 days in the same region … selected by the customer"; Trainingsausschluss produktnah: „customer data … is not used to train foundation models" (Google-Cloud-Doku, 10.07.2026).
**Konfidenz**: hoch — offizielle Terms/Doku mit Stand-Datum, Kernzitate nachverifiziert; niedrig-mittel nur für den aktuellen Stand der EU-ML-Processing-Zusagen (Primärquelle 2023, Teil 3).
**Gilt unter**: Gemini Developer API (AI-Studio-Key) vs. Vertex AI, Stand 07/2026. Praktisch: Der Bestand muss auf Paid-Tier stehen, sonst gilt das Trainings-Regime.
**Nachmessbar durch**: Billing-Status des eigenen Projekts prüfen; ggf. ZDR-Antrag stellen.
**Widerspruch**: verbreitete Sekundärangabe „30 Tage Abuse-Logging bei Vertex" widerspricht der aktuellen 90-Tage-Angabe.

### B-62 — Die übrigen Anbieter (nachverifiziert): Anthropic trainiert nicht auf Kundeninhalten („Anthropic may not train models on Customer Content from Services") und löscht Inputs/Outputs standardmäßig binnen 30 Tagen (bei Trust-&-Safety-Flags bis 2 Jahre; ZDR nur per Sondervereinbarung); Azure OpenAI trainiert nicht, bietet „modified abuse monitoring" (kein Speichern/Human-Review) und EU-Data-Zone-Verarbeitung; Azure Document Intelligence speichert Eingaben/Ergebnisse max. 24 h in der Ressourcen-Region (ein expliziter No-Training-Satz fehlt auf der DI-Privacy-Seite selbst); AWS Textract dagegen nutzt verarbeitete Dokumente PER DEFAULT zur Dienstverbesserung und darf sie dafür in andere Regionen kopieren — Opt-out nur über eine AWS-Organizations-Policy, Löschung nur via Support.
**Beleg**: anthropic.com/legal/commercial-terms (Effective 17.06.2025, nachverifiziert) + privacy.claude.com-Retention-Seite; Microsoft-Learn data-privacy Azure OpenAI (ms.date 05–06/2026) und Document Intelligence (ms.date 28.07.2026): „stores submitted input data and analyze results for 24 hours"; AWS-Textract-FAQ (nachverifiziert, wörtlich): „…to improve and develop the quality of Amazon Textract and other Amazon machine-learning/artificial-intelligence technologies", „may be stored in another AWS region", Opt-out „using an AWS Organizations opt-out policy".
**Konfidenz**: hoch — offizielle Terms-/Doku-Seiten mit Datum, Kernzitate nachverifiziert.
**Gilt unter**: Stand 07/2026; Azure/AWS wären für den Entwickler neue Zugänge.
**Nachmessbar durch**: die jeweiligen Terms-Seiten abrufen; bei AWS die Organizations-Opt-out-Policy konkret setzen, bevor Textract erwogen wird.
**Widerspruch**: keiner gefunden; AWS ist im Vergleich der einzige geprüfte Dienst mit Improvement-Nutzung als Default im Bezahlbetrieb.

---

## Teil 2 — Empfehlung je Entscheidungspunkt

Vorbemerkung zur Lesart: Die Reihungen optimieren die beauftragte Zielfunktion (1. Treue, 2. Formatbreite; Geschwindigkeit/Betrieb nachrangig) für überwiegend deutsches Material. Die größte einzelne Unsicherheit zieht sich durch alle PDF-Empfehlungen: Es gibt keinen deutschen Benchmark (B-07); der beste verfügbare Proxy (Französisch, B-06) begünstigt Gemini deutlich und straft MinerU/Docling ab. Jede PDF-Reihung steht deshalb unter dem Vorbehalt des eigenen 30–50-Seiten-Goldstandards — er ist mit einem Wochenend-Aufwand die höchstrentierliche Einzelmaßnahme des ganzen Vorhabens.

### Engine für PDF mit Textebene
1. **Gemini nativ-PDF (Paid-Tier, media_resolution MEDIUM) mit deterministischem Treue-Gate.** Beste belegte Treue auf europäischem Material und bei Tabellen (B-04, B-06, B-09), nativer Textlayer wird gratis mitgelesen (B-52), Kosten ~0,02–0,1 ct/Seite (B-53). Der Handel: Cloud-Abhängigkeit und Halluzinationsrisiko — deshalb ist das Gate nicht optional: Textlayer parallel deterministisch extrahieren (pdfplumber/pypdfium2), Zahlen-/Wortbestand abgleichen, Abweichungen als Degradationssignal je Block ausweisen (B-55, B-57).
2. **MinerU 3.x selbstgehostet** (VLM-Backend auf GPU, pipeline-Backend auf CPU). Bestes selbstgehostetes Gesamtpaket auf den großen Benchmarks (B-01), Lizenz seit 3.1 unkritisch (B-60). Der Handel: dokumentierter Kollaps auf europäischem Nicht-Englisch (B-06) — erst produktiv nehmen, wenn der eigene deutsche Goldstandard bestanden ist.
3. **Marker v2 (GPU)** — stark bei Mehrspaltern/wissenschaftlichen Layouts (B-12); der Handel: volle Qualität nur auf GPU, Gewichte-Lizenz mit Umsatzschwelle und Nicht-Wettbewerbs-Klausel (B-60).
**Nicht auf dem Podium**: docling als Treue-Engine für PDF (B-03, B-06) — es bleibt aber als Datenmodell/Serving gesetzt (siehe unten). **Falsch, wenn** die Dokumente die Cloud nicht verlassen dürfen (→ 2 und 3 rücken vor, Qualitätsverlust einpreisen) oder der eigene Goldstandard MinerU/Marker auf Deutsch gleichauf zeigt (→ Selbsthost vor API).

### Engine für gescanntes PDF (heute gar nicht abgedeckt)
1. **Gemini (Paid/ZDR) als OCR-VLM, mit Loop-/Absence-Guards.** Scans sind 2026 VLM-Territorium: klassische Pipelines erreichen auf alten Scans ≤50 % (B-02, B-12), proprietäre VLMs sind auf schwerem europäischem Material dokumentiert robuster als alle Open-Weights (B-06). Der Handel: gerade auf Scans ist die Erfindungsgefahr am größten — Längen-/n-Gramm-Detektor, Blank-Page-Handling und Stichproben-Review sind Pflicht (B-31, B-55).
2. **Lokal auf der A2000: PaddleOCR-VL 1.x oder dots.ocr via vLLM** (beide Apache/MIT, multilingual, ≤3B → passen in 12 GB; B-29, B-32). Der Handel: Deutsch ist für beide unevaluiert (B-30), dots.ocr hat einen dokumentierten Formularlinien-Loop-Trigger (B-31) — ohne eigenen Deutsch-Test und Loop-Detektor nicht produktiv.
3. **CPU-only: Tesseract 5 `deu` mit Eskalation.** Auf sauberem modernem Deutsch nahe Kommerz-Niveau (B-25); degradierte Scans erkennen (Konfidenz/Heuristik) und an Gemini oder Azure Read eskalieren (B-26, B-56).
**Falsch, wenn** der Scan-Anteil stark degradiertes Altmaterial ist UND Cloud verboten — dann ist ehrlich: Das Treue-Ziel ist dort 2026 mit keinem Werkzeug erreichbar (bestes System 50 %, B-02); Human-in-the-Loop statt Vollautomatik.

### Engine für DOCX
1. **pandoc** — vollständigster Strukturerhalt inkl. Fußnoten/Endnoten, OMML-Mathe und steuerbarer Änderungsverfolgung (B-36). Handel: verbundene Zellen überleben den Markdown-Writer nicht (ggf. HTML-Tabellen im MD zulassen), SDT/Word-Bibliografien lückenhaft.
2. **docling** — wenn einheitliches Zwischenformat und Zell-Spans im Modell wichtiger sind als Fußnoten; OMML→LaTeX vorhanden. Handel: Fußnoten werden ersatzlos verworfen (B-37) — für Verträge/Verwaltungstexte disqualifizierend.
3. **mammoth (direkt, →HTML→MD)** — trägt Fußnoten, ist aber ein schmalerer Pfad (Tabellenformatierung ignoriert, MD-Pfad deprecated; B-38).
**Falsch, wenn** das DOCX-Material nachweislich fußnotenfrei ist und der Dienst strikt EIN Zwischenformat fahren soll — dann docling zuerst.

### Engine für PPTX
1. **markitdown** — Positions-Lesereihenfolge, Gruppen-Rekursion, Speaker-Notes, Charts als Datentabellen (B-39); als PPTX-Spezialwerkzeug dem eigenen Ruf voraus. Handel: Projekt auf Sparflamme (B-19), SmartArt-Text geht verloren (wie überall) — SmartArt-Vorkommen als Degradationsflag melden.
2. **docling** — Notes und Zell-Spans im Modell, aber XML-Lesereihenfolge und unklarer Notes-Export (B-39); vor Einsatz Serializer-Verhalten testen.
3. **unstructured mit `include_slide_notes=True`** — solide Positions-Sortierung, aber Markdown-Rekonstruktion aus Element-Kategorien bleibt Eigenleistung.
**Falsch, wenn** die Bestands-Decks stark frei positioniert/zweispaltig sind und der Eigentest doclings XML-Reihenfolge dort besser trifft — die Lesereihenfolge-Frage ist unvermessen (B-39). Ein VLM-Screenshot-Pfad ist als Ersatz nicht belegt (B-40), als Zusatz für Diagramm-Folien legitim.

### Engine für XLSX/CSV
1. **markitdown** — alle Sheets, robuster pandas-Pfad, MIT (B-41). Handel: Formeln nur als cached Werte (programmatisch erzeugte Dateien ohne Cache → leere Zellen; abfangen), Verbund-Semantik verloren.
2. **docling** — gleichwertige Kernleistung plus Formatbreite (Legacy .xls, ODS) — um den Preis der LibreOffice-Abhängigkeit im Image (B-41).
3. **unstructured** — einziges Werkzeug mit Subtabellen-Erkennung je Sheet; wertvoll bei „mehrere Tabellen pro Blatt"-Realität (B-41).
**Falsch, wenn** Merge-Semantik entscheidend ist — dann gibt es keine Standard-Lösung; HTML-Tabellen als Ausgabeform erwägen statt Markdown-Pipes.

### Engine für HTML/EML
1. **HTML: trafilatura** (Markdown-Ausgabe, precision-Modus, offline, Apache) — einzige mit deutschsprachiger Evaluationsbasis (B-43). Handel: die DE-Zahl ist von 2022; auf modernem Nicht-Artikel-HTML streuen alle Extraktoren stark.
2. **resiliparse als Zweitextraktor/Schnellpfad** (28 ms/Seite, aktiv, Apache) — bei Konflikt der beiden Extrakte: das vollständigere behalten und Differenz als Degradationssignal melden (B-43).
3. **mozilla-readability(-Port) als Artikel-Fallback**; LLM-Extraktion nicht als Default (B-44).
**EML als Komposition, nicht als Einzelwerkzeug**: mail-parser/unstructured für Header+Multipart+Anhänge → HTML-Teil durch den HTML-Pfad → **mail-parser-reply** für deutsche Zitat-/Signatur-Ketten (B-45). Handel: Einzelmaintainer-Abhängigkeit; Trefferquote selbst messen.
**Falsch, wenn** JS-lastige Live-Webseiten in Scope kommen — dann Headless-Rendering/Firecrawl ergänzen (AGPL beachten, B-60).

### Formel-Erkennung: ja/nein, und womit
**Ja — aber im Primärpfad, nicht als eigene CPU-Stufe.**
1. **Formeln vom gewählten PDF-Konverter miterledigen lassen** (Gemini bzw. MinerU/Marker haben Formel-Fähigkeiten eingebaut) und Qualität per CDM-Stichprobe messen (B-46, B-47).
2. **Mathpix für formel-lastige wissenschaftliche PDFs** — ~0,001–0,002 $/Seite, Element-Konfidenzen, nicht teurer als ein Gemini-Call (B-47, B-56).
3. **UniMERNet lokal (GPU)** nur falls Offline-Pflicht — stärkstes offenes Modell, aber seit 12/2024 eingefroren (B-46).
**Ausdrücklich nicht**: pix2tex (kollabiert außerhalb sauberer Drucke), texify (archiviert), docling-CodeFormula als CPU-Default (unbelegte Qualität, 3× Laufzeit; B-46).
**Falsch, wenn** Formeln im Korpus selten sind — dann Auto-Erkennung aus und Formel-Seiten nur auf Anforderung nachbearbeiten.

### Zwischenformat / Datenmodell
1. **DoclingDocument als Kern + eigener dünner Envelope.** Provenienz (Seite/Bbox/Charspan) ist dort Pflichtstruktur, das Schema versioniert, die Traktion konkurrenzlos (B-48). Was fehlt, liefert der Envelope des Dienstes: Block-Konfidenz, Quelle je Block (deterministisch | OCR | VLM = das „modellgeneriert"-Flag), Degradationssignale — denn doclings Konfidenzen leben außerhalb des Dokuments und gehen sonst verloren (B-49). Kein eigenes Blockschema erfinden.
2. **unstructured-Elements** — einzige native Block-Konfidenz (`detection_class_prob`), aber an ein kommerziell driftendes Ökosystem gebunden (B-18, B-49).
3. **Markdown + Sidecar-JSON (Minimallösung)** — nur wenn der Dienst bewusst dumm bleiben soll; verliert die Provenienz-Zukunft.
**Falsch, wenn** die DocLang-Spezifikation (B-14) Provenienz+Konfidenz nativ standardisiert — dann dorthin migrieren.

### Serving-Layer: fertig übernehmen oder selbst bauen
1. **docling-serve als Konvertierungs-Backend übernehmen, plus hauchdünner eigener Gateway davor.** docling-serve liefert Async-API, Queues, API-Key-Auth und das 4,4-GB-CPU-Image fertig (B-50); der Gateway (wenige hundert Zeilen) macht das, was kein Fertigprodukt kann: Routing auf pandoc/trafilatura/Gemini je Format, Kostenbudget je Auftrag, Degradationssignal in der Antwort. Handel: dokumentierte Memory-Leaks im Dauerbetrieb → Memory-Limit + Worker-Recycling von Tag 1 (B-50).
2. **Voll-Eigenbau (FastAPI, Engines als Bibliotheken)** — maximale Kontrolle über das Antwortschema, dafür trägt man Queue/Watchdog/Health selbst; die reifste Betriebsmechanik des Bestands (Implementierung B) liefert dafür die Vorlage.
3. **Apache Tika als Format-Fallback-Sidecar** (347 MB) für exotische Formate — niemals als Struktur-Lieferant (B-51).
**Falsch, wenn** der Dienst auf absehbare Zeit nur PDF+Office über docling fährt — dann reicht docling-serve pur, der Gateway entfällt.

### CPU-only-Variante vs. GPU-Variante — beide ausformuliert
**CPU-Welt (Empfehlung als Startpunkt):** Schlankes Haupt-Image (~4–5 GB: docling-CPU als Struktur-Backbone, pandoc, trafilatura, Tesseract `deu`, mail-parser-reply) + Gemini-API als „ausgelagerte GPU" für Scans, schwere Seiten, Tabellen-Zweitmeinung und Formeln. Deckt alle vier Dokumentklassen, respektiert die 8,9-GB-Entscheidung (B-22), keine neue Betriebsabhängigkeit. Preis: Cloud-Abhängigkeit für genau die Fälle, in denen es um Treue geht — kontrolliert durch das deterministische Gate und den Paid-Tier-/ZDR-Vertragsrahmen (B-61). Sekunden/Seite auf CPU sind für Single-User-Betrieb unkritisch (B-21).
**GPU-Welt (Ausbaustufe, nicht Ersatz):** Separater CUDA-Container (vLLM + ein kleines Apache/MIT-Dokument-VLM: PaddleOCR-VL oder dots.ocr; B-29, B-32) neben dem schlanken Haupt-Image — die Containertrennung löst die Image-Kollision auf, das 8,9-GB-Image bleibt unberührt; genau dafür hat der Entwickler den separaten Extraktions-Container bereits als Option benannt, und die Recherche bestätigt ihn als richtige Auflösung. Preis: Passthrough-Einrichtung (CDI) plus Kernel-Update-Pflege (B-34), unvermessener Durchsatz auf der A2000 (B-35), Deutsch-Qualität der lokalen Modelle unbelegt (B-30). Nutzen: Datenhoheit für Scans ohne Cloud, Unabhängigkeit von API-Abkündigungen (B-53 hat gezeigt, wie real die sind). Ehrlich benannt: Ein QUALITÄTS-Argument für die lokale GPU gegenüber Gemini gibt es nach aktueller Beleglage nicht — die 12-GB-Klasse liegt auf Benchmarks unter der Gemini-Klasse (B-06, B-29).
**Falsch, wenn** Datenschutz Cloud kategorisch ausschließt — dann ist die GPU-Welt nicht Ausbaustufe, sondern Pflicht ab Tag 1, und die Scan-Qualitätserwartung ist entsprechend zu senken.

### Rolle des Modells: ersetzen, ergänzen, oder nur auf textlosen Seiten
1. **Differenziert (Empfehlung): ersetzen dürfen auf textlosen/gescannten Seiten, ergänzen mit Rückvalidierung auf Seiten mit Textebene.** Auf Scans ist das VLM konkurrenzlos (B-02, B-06); auf Text-Seiten liefert der Textlayer das perfekte Kontrollsignal — VLM-Output, der Zahlen/Wörter enthält, die im Layer nicht vorkommen, wird geflaggt oder verworfen. Pflicht-Guardrails unabhängig vom Modell: Loop-/Längen-Detektor, Blank-Page-Regel, Charset-Check, `max_page_error_rate`-Äquivalent, „modellgeneriert"-Flag je Block (B-31, B-55). Cross-Model-Voting ist als alleinige Absicherung nicht tragfähig (B-57).
2. **Konservativ: Modell nur auf textlosen Seiten** — maximale Erfindungssicherheit, verschenkt aber belegte Tabellen-/Struktur-Qualität der VLMs auf Text-Seiten (B-04, B-09).
3. **Voll ersetzen (VLM liest alles, keine deterministische Spur)** — nur vertretbar mit stehendem Goldstandard-Monitoring; nach Beleglage unnötig riskant für Lernmaterial.
**Falsch, wenn** die Guardrails aus Zeitgründen nicht gebaut werden — dann ist Option 2 die einzig verantwortbare.

---

## Teil 3 — Was nicht herauszufinden war

Jeder Punkt hier ist eine echte Lücke nach systematischer Suche — nicht mit Plausibilität gefüllt. Die mit **[Messplan]** markierten Punkte kann nur der Entwickler selbst schließen; sie definieren faktisch das eigene Testprogramm.

**Deutschland-Lücke (die größte):**
- Es existiert kein deutschsprachiger Dokument→Markdown- oder Dokument-OCR-Benchmark (B-07). Alle End-to-End-Rankings sind EN/ZH, der beste Proxy ist Französisch. Grund: existiert nicht. **[Messplan: eigener Goldstandard, 30–50 Seiten quer über die vier Dokumentklassen]**
- Keine unabhängige Deutsch-Evaluation irgendeines Dokument-VLMs; die einzigen Deutsch-Zahlen (Surya 89,7 %, Chandra 94,8 %) sind Datalab-interne Benchmarks ohne offengelegte Metrik (B-30). Grund: existiert nicht.
- Keine unabhängigen CER-Messungen moderner OCR-Engines (PaddleOCR-latin-v5, RapidOCR, Surya, docTR, EasyOCR) auf modernen deutschen Drucksachen — die deutsche Evaluationslandschaft (OCR-D, OCR-BW) ist fast vollständig auf historische Drucke ausgerichtet. Grund: existiert nicht. **[Messplan: 10 deutsche Scans, CER je Engine]**

**Benchmark-/Qualitätslücken:**
- Google Document AI: keine einzige belastbare 2025/26er-Vergleichszahl öffentlich (fehlt in OmniDocBench, olmOCR-Bench, PulseBench; RD-TableBench publiziert die Zahl nur in einem nicht abrufbaren interaktiven Viewer). Grund: nicht in Tabellenform publiziert.
- unstructured „hi_res" modus-genau: kein Benchmark weist den Modus aus; der einzige Wert (PulseBench 0,360) betrifft die API ohne Modus-Angabe. Grund: widersprüchlich/unterspezifiziert.
- Mathpix aktuell (Volltext/Lesereihenfolge): nur zwei Datenpunkte (OmniDocBench v1.0 ~Ende 2024; Horn & Keuper Tabellen 8,53/10). Grund: zu wenig gemessen.
- LlamaParse End-to-End-Markdown-Treue: nirgends neutral gemessen (nur Tabellenwert 0,798). Grund: existiert nicht.
- Claude auf olmOCR-Bench: keine Einträge; die Claude-Bewertung hängt an OmniDocBench-v1.5 (Nanonets) und Horn & Keuper. Grund: existiert nicht.
- GLM-OCR-Diskrepanz (95,22 offizielles README vs. 69,2 IDP-Leaderboard auf nominell demselben Benchmark): nicht auflösbar; zeigt die Harness-/Prompt-Empfindlichkeit aller Leaderboard-Zahlen. Grund: widersprüchlich.
- Ensemble-vs.-gelernt direkt: keine Studie misst Multi-Detektor-Konsens gegen gelernte Modelle (B-11). Grund: existiert nicht. **[Messplan: Eigenbau als Pipeline ins eigene Testset einreihen — die einzige Möglichkeit, die 1.418 Zeilen je zu bewerten]**

**Hardware-/Betriebslücken:**
- A2000-Durchsatz: keine publizierte Messung irgendeines Dokument-VLM-Stacks auf dieser Karte (B-35). Grund: existiert nicht. **[Messplan: 100-Seiten-Lauf je Kandidat]**
- dots.ocr auf exakt 12 GB: unbelegt (8 GB scheitert dokumentiert, 16/24 GB läuft). Grund: zu wenig berichtet. **[Messplan: vLLM-Start mit reduziertem max-model-len]**
- Halluzinations-/Loop-Raten unter Quantisierung (AWQ/GGUF-Q4): nirgends gemessen — relevant, weil die A2000 mehrere Kandidaten nur quantisiert fahren kann. Grund: existiert nicht.
- RAM-Spitzen je CPU-Kandidat: außer MinerU (16–32 GB) nirgends systematisch publiziert. Grund: existiert nicht.
- Exakte Größe des Linux-x86_64-torch-„+cpu"-Wheels: download.pytorch.org publiziert keine Dateigrößen; nur über Windows/macOS-Builds (88–123 MB) und Image-Deltas belegt. Grund: Quelle publiziert nicht.
- Ampere-Treiber-Support-Horizont: NVIDIA publiziert kein EOL-Datum; belegbar ist nur die Negativaussage (nicht in der 580er-Abkündigung). Grund: existiert nicht.

**Architektur-/Preislücken:**
- Nativer Gemini-PDF-Input vs. selbst gerenderte PNGs: kein unabhängiger, methodisch sauberer A/B-Qualitätsvergleich. Grund: existiert nicht. **[Messplan: identisches Testset beide Wege, countTokens + Fehlerrate]**
- Gemini-Tokenzahl je PDF-Seite: zwei offizielle Google-Seiten nennen 258 bzw. 560 (B-52); vermutlich Generationen-Mischung. Grund: widersprüchlich. **[Messplan: countTokens]**
- DPI→Genauigkeits-Kurve für VLM-Dokumentlesen: keine kontrollierte, modellübergreifende Kurve publiziert (B-54). Grund: existiert nicht. **[Messplan: 100/134/150/200/300 dpi gegen CER/Tabellen-F1]**
- CodeFormula (docling-Formel-Enrichment): keinerlei publizierte Qualitätszahlen; CPU-Sekunden/Formel nur anekdotisch. Grund: nicht publiziert.
- Datalab- und Artifex-Preise: datalab.to/pricing ist JS-gerendert (technisch nicht abrufbar), Artifex nennt nur Modelle ohne Beträge („Preis auf Anfrage"). Grund: Paywall/technisch.
- Frontier-LLM direkt auf rohem HTML (Boilerplate-Entfernung): keine belastbare F1-Messung gegen trafilatura gefunden; WCXB testete nur kleine Spezialmodelle. Grund: existiert nicht.
- Logprob-Kalibrierung für OCR-Halluzination: keine Literatur 2024–2026. Grund: zu neu/existiert nicht.

**Rechtslücken:**
- FSF-Auslegung zu AGPL §13 bei interner Nutzung: gnu.org war am Stichtag durchgehend rate-limitiert (HTTP 429); die einschlägigen FAQ-Einträge konnten nicht wörtlich belegt werden. Grund: technisch. Nachholbar.
- Aktuelle EU-ML-Processing-Zusagen für Vertex (2026): offizielle Data-Governance-Seiten waren nur als Navigationsgerüst extrahierbar; belegt sind At-Rest-Residenz (2023) und die 90-Tage-Abuse-Log-Regel (2026). Grund: technisch (JS-Rendering).
- Azure-OpenAI-Abuse-Monitoring-Speicherfrist: konkrete Frist auf der aktuellen Seite nicht (mehr) extrahierbar (früher verbreitet „bis zu 30 Tage"). Grund: möglicherweise entfernt.
- MinerU-Custom-Lizenz: Kernklauseln verifiziert (B-60), aber keine vollständige juristische Textprüfung aller Zusatzklauseln. Grund: Umfang; vor Produktivsetzung lesen.

---

## Teil 4 — Was mich überrascht hat

1. **Eine Auftragsprämisse ist akut kaputt: gemini-2.0-flash ist seit dem 01.06.2026 abgeschaltet** („deprecated and has been shut down", offizielle Preisseite; B-53). Der Bestand nutzt es laut Auftrag heute für die PDF-Extraktion — entweder läuft dort längst ein stiller Fallback, oder der Pfad ist tot. Das ist unabhängig von jeder Architektur-Entscheidung sofort prüfenswert.
2. **Docling ist Ökosystem-Sieger und Treue-Verlierer zugleich** (B-03, B-14): Foundation-Governance, DocLang-Standardisierung, bestes Datenmodell — aber 50,3 % auf olmOCR-Bench und 0,119 auf schwerem Französisch. Die richtige Rolle ist Datenmodell/Serving-Rückgrat, nicht Treue-Engine. Die verbreitete Pauschal-Empfehlung „nimm docling" beruht auf Governance, nicht auf Messung.
3. **MinerU ist gleichzeitig Weltspitze und Totalausfall** (95,75 auf dem hauseigenen EN/ZH-Benchmark, 0,222 auf Französisch; B-01/B-06) — die OmniDocBench-Krone ist für deutsches Material fast beweislos. Sprach-Overfitting ist 2026 der wichtigste versteckte Confounder der gesamten Benchmark-Landschaft.
4. **Gemini schlägt Claude als Dokument-Konverter deutlich** (Tabellen 9,55 vs. 7,02 in unabhängiger deutscher Hochschul-Studie; Texttreue 0,077 vs. 0,165; B-04) — und rechnet PDFs strukturell günstiger ab (B-53). Für einen Anthropic-affinen Entwickler kontraintuitiv, aber die Messlage ist einseitig.
5. **Zur Kernfrage: Der Heuristik-Ensemble-Ansatz ist nicht widerlegt — er ist verschwunden** (B-11). Seit Oktober 2024 führt keine ernsthafte Evaluation camelot/pdfplumber/tabula/img2table mehr mit. Die letzte Messung zeigt gelernte Modelle 2–8× besser auf komplexen Tabellen, aber camelot-lattice vorn bei linierten Behördentabellen — genau eine der fünf Eigenbau-Komponenten hatte je belegten Mehrwert, der Fünffach-Konsens als Konstrukt nie.
6. **Zwei weitere Auftragsprämissen sind überholt**: camelot ist nicht tot (2.0.0 vom Juni 2026) und zieht kein Ghostscript mehr nach (pdfium-Default → die AGPL-Kaskade entfällt; B-15/B-59); MinerU ist nicht mehr AGPL (B-60). Die Lizenzlage ist 2026 freundlicher als der Auftrag annimmt — mit der Ausnahme PyMuPDF, die bleibt (B-58).
7. **Alle vier Bestands-DPIs (200/216/288/300) liegen oberhalb der Provider-Caps** — 300-dpi-Renderings werden bei Claude und Gemini nachweislich heruntergerechnet, olmOCR arbeitet mit ~110-dpi-Äquivalent, und Google erklärt selbst MEDIUM (560 Tokens) zum Sättigungspunkt; wer PDFs selbst zu PNGs rendert, verliert bei Gemini zusätzlich den kostenlosen nativen Textlayer (B-52, B-54). Der Screenshot-Pfad des Bestands ist doppelt unbegründet.
8. **Doclings bequemster OCR-Default ist ausgerechnet eine Deutsch-Falle** (hardcodete chinesisch/englische PP-OCR-Modelle, `--ocr-lang` ignoriert, „Spaß"→„SpafS"; B-24) — und umgekehrt ist Tesseract auf sauberem modernem Deutsch fast auf Kommerz-Niveau (B-25). Die Intuition „neu = besser für OCR" gilt für Deutsch derzeit nicht.
9. **Fußnoten trennen die Werkzeug-Generationen genau falsch herum**: Die modernen Parser (docling, unstructured) verwerfen DOCX-Fußnoten komplett, die alten (pandoc, mammoth) tragen sie (B-36/B-37). Wer DOCX-Treue will, kommt an pandoc nicht vorbei.
10. **Repetition-Loops sind quantifiziert zweistellig** (9,2 % DeepSeek-OCR, 8–10 % Chandra auf schweren Scans) und der dokumentierte Trigger — durchgehende Punkt-/Unterstrichlinien — ist das Standardmuster deutscher Formulare (B-31). Ein Loop-Detektor ist keine Kür, sondern Teil der Treue-Definition.
11. **Die Standardmetriken selbst sind das Problem**: TEDS korreliert nur mit r=0,684, GriTS mit r=0,70 mit menschlichem Urteil, während ein LLM-Judge 0,93 erreicht (B-09) — die Zahlen, mit denen man Pipelines vergleicht, messen Tabellenqualität nur mäßig valide. Für den eigenen Goldstandard heißt das: LLM-Judge + Stichproben-Handprüfung statt blindem TEDS.
12. **AWS Textract ist der einzige geprüfte Großanbieter, der Kundendokumente per Default zur Dienstverbesserung nutzt** — inklusive Cross-Region-Kopie, Opt-out nur über AWS Organizations (B-62). Gleichzeitig hat Google die ZDR-Option in die Developer-API gebracht (B-61) — die Datenschutz-Rangfolge der Anbieter ist 2026 anders, als die Reflexe vermuten.

---

*Ende des Befund-Registers. Quellen: sämtlich inline in den Beleg-Feldern, Abrufdatum durchgehend 30.07.2026.*
