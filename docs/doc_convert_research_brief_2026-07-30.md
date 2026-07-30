# Recherche-Auftrag — Dokument→Markdown, Stand der Technik 2026

**Für**: Claude Cowork (Deep Research) · **Datum**: 2026-07-30 · **Auftraggeber-Kontext**: CONVERTER-Master

> **Bedienungshinweis für Oli**: Alles ab `═══ AUFTRAG ═══` ist der copy-paste-fähige Prompt. Er ist **selbsttragend** — Cowork hat keinen Zugriff auf deine Repos, deshalb trägt der Prompt die Bestandsaufnahme in sich. Die Langfassung des Bestands liegt in [doc_convert_bestand_2026-07-30.md](doc_convert_bestand_2026-07-30.md), die Verwertung der Ergebnisse in [doc_convert_verwertung_2026-07-30.md](doc_convert_verwertung_2026-07-30.md).

---

═══ AUFTRAG ═══

# Recherche: Dokument→Markdown-Konvertierung, Stand der Technik 2026

Du recherchierst für eine Architektur-Entscheidung, nicht für einen Überblicksartikel. Am Ende muss ein Entwickler anhand deiner Ausgabe entscheiden können, welche Konvertierungs-Pipeline er baut — und *warum nicht die anderen*. Vermeide Vollständigkeit um ihrer selbst willen; jede Zeile soll eine Entscheidung stützen.

## Das Vorhaben

Ein Entwickler hat über mehrere Jahre **fünf voneinander unabhängige Dokument-Konvertierungen** in vier Projekten gebaut. Sie sollen durch **einen Dienst** ersetzt werden, der Dokument→Markdown per HTTP-API für alle übrigen Dienste anbietet. Die Ausgangsdiagnose lautet wörtlich: *„das ist ineffizient und es ist unwahrscheinlich, dass ich immer zufällig die beste Version gecodet habe."*

Eine Bestandsaufnahme des eigenen Codes ist bereits erfolgt (unten zusammengefasst). Deine Aufgabe ist ausschließlich das, was der eigene Code **nicht** beantworten kann: wie sich der Bestand zum Stand der Technik verhält und was man stattdessen nehmen sollte.

## Zielfunktion — in dieser Reihenfolge

1. **Treue zum Original.** Struktur, Tabellen, Lesereihenfolge; nichts erfunden, nichts verloren. Das Ergebnis ist Lern- und Lesematerial — ein halluziniertes Zahlenpaar ist schlimmer als eine fehlende Tabelle.
2. **Formatbreite.** Möglichst viele Eingangsformate sauber, damit wirklich jeder Dienst denselben Endpoint nutzen kann.

**Ausdrücklich nachrangig**: Geschwindigkeit und Betriebs-Einfachheit. Empfiehl nicht das Schlankere, wenn das Schwerere treuer ist — benenne den Preis und lass ihn den Entwickler zahlen.

## Randbedingungen (verbindlich, nicht verhandelbar)

- **Hardware**: ein Linux-Host („Mintbox") mit **NVIDIA RTX A2000, 12 GB VRAM**. GPU-Passthrough in Docker ist heute *nicht* eingerichtet, **darf aber eingerichtet werden** — der Entwickler hat das freigegeben. Behandle CPU-only und GPU-Pfad als zwei getrennt zu bewertende Welten.
- **Docker-Image-Historie**: Das bestehende Image wurde vor vier Wochen bewusst von 15,5 GB auf 8,9 GB verkleinert, indem CUDA-Torch entfernt wurde (torch kam nur transitiv über `unstructured[all-docs]` und wurde nie für Inferenz benutzt). Jede Empfehlung, die schwere Modelle zurückbringt, **kollidiert damit** — benenne das explizit und bewerte, ob ein separater Extraktions-Container die richtige Auflösung ist.
- **Vorhandene Zugänge**: Google Gemini API (heute `gemini-2.0-flash` für PDF-Extraktion) und Google-Cloud-Credentials. Anthropic-API in einem der anderen Projekte. Ein firmeninternes OpenAI-kompatibles Gateway in einem weiteren — für diesen Dienst aber vermutlich außer Reichweite.
- **Sprache**: Das Material ist **überwiegend deutsch**. Alle Qualitätsaussagen zu OCR und Modellen musst du auf deutschsprachiges Material beziehen, wo immer Daten existieren — englische Benchmark-Zahlen sind nur ein Näherungswert und als solcher zu kennzeichnen.
- **Betriebsform**: heute Single-User im LAN, künftig interner Dienst für mehrere aufrufende Dienste. Keine Multi-Tenant-Anforderung, aber Service-zu-Service-Authentifizierung.
- **Kosten**: Bezahl-APIs sind **nicht** ausgeschlossen. Bewerte selbst-gehostet und bezahlt gleichrangig und stelle Qualität pro Euro gegenüber.

## Die Dokumentklassen, an denen gemessen wird

Alle vier sind gleichrangig gefordert:

1. **Wissenschaftliche/technische PDFs** — zweispaltig, Tabellen, Formeln, Abbildungen, Fußnoten.
2. **Office-Dokumente** — DOCX und PPTX. (Heute der schwächste Punkt: siehe unten.)
3. **Web und E-Mail** — HTML, EML; Kernproblem ist Boilerplate-Entfernung, nicht Layout.
4. **Gescanntes** — Bild-PDFs ohne Textebene. Heute gar nicht abgedeckt.

## Der Bestand, gegen den du recherchierst

Damit du weißt, was der Vergleichspunkt ist. Alles hier ist am Quelltext verifiziert.

**Implementierung A — die aufwendigste.** Für PDFs ein Eigenbau von 1.418 Zeilen: fünf Tabellen-Detektoren in drei Eskalationsstufen (PyMuPDF `find_tables`, pdfplumber, camelot `lattice`, camelot `stream`, img2table, dazu eine pdfminer-Layout-Heuristik), Bbox-IoU-Clustering bei 0,3, Konsens ab zwei Detektoren, danach ein Extraktor-Wettbewerb mit Scoring, Gemini Vision als letzter Ausweg mit Rückvalidierung, plus seitenübergreifender Tabellen-Merge. **Null Tests.** Behandelt *nicht*: Überschriften auf tabellenlosen Seiten, mehrspaltiges Layout (Sortierung stumpf nach y-Koordinate), Mathematik, Bilder, OCR, Metadaten.

**Implementierung A, Nicht-PDF-Pfad.** Zwei Zeilen: `unstructured` mit `strategy="fast"`, dann `"\n\n".join(el.text)`. Element-Kategorien (`Title`, `ListItem`, `Table`) und `metadata.text_as_html` werden verworfen. Für DOCX/PPTX/HTML/EML entsteht faktisch Fließtext in einer `.md`-Datei.

**Implementierung B — die modernste.** Nutzt **docling** (`DocumentConverter()` ohne Optionen) und **markitdown**, darüber drei VLM-Pässe über Gemini. Nur PDF und PPTX. Reifste Betriebsmechanik des Bestands (Job-Queue, Watchdog, Heartbeats, Circuit Breaker, Dead-Letter). Erklärtes Ziel ist Wortgetreue.

**Implementierung C — die layoutbewussteste.** Für PPTX ein python-pptx-Shape-Tree mit geometrischem Spalten-Clustering, Heading-Level aus Fontgrößen, Connector-Graph. Daneben, unverbunden, ein zweiter Pfad mit bloßem `pypdf.extract_text()`.

**Implementierung D — die einzige mit echtem OCR.** PaddleOCR 3.x mit PPStructureV3, GPU-Zweig, LLM-Validierung, Human-in-the-Loop-Queue. Zweck ist Feldextraktion, nicht Fließtext.

**Was in allen fünf fehlt**: Mathematik-Erkennung, Bild-Extraktion, Dokument-Metadaten, Lesereihenfolge bei Mehrspaltigkeit, ein Degradationssignal in der Antwort, ein Kostenbudget je Auftrag, Tests des Konvertierungspfads.

**Der zentrale ungemessene Punkt**: Der 1.418-Zeilen-Eigenbau wurde **nie** gegen docling, gegen Gemini-direkt oder gegen irgendetwas anderes gehalten. Die implizite Behauptung „mein Ensemble ist besser als eine Standardpipeline" ist völlig unbelegt.

---

## Was du herausfinden sollst

Die Fragen sind nach der Entscheidung gruppiert, die sie freischalten. Wenn du eine Frage nicht belastbar beantworten kannst, **sage das** — eine ehrliche Lücke ist wertvoller als eine plausible Erfindung.

### Block 1 — Ist der Eigenbau noch zu rechtfertigen? *(die Kernfrage)*

1. Wie schneiden die 2026er-Referenzpipelines auf öffentlichen Benchmarks gegeneinander ab — **docling, marker, MinerU, unstructured `hi_res`, Azure Document Intelligence, AWS Textract, Google Document AI, LlamaParse, Mathpix** sowie **Gemini und Claude als End-to-End-Konverter**? Interessant sind besonders **Tabellenstruktur** (TEDS, GriTS), **Lesereihenfolge** und **Textgenauigkeit**.
2. Welche Benchmarks sind dafür 2026 die maßgeblichen (OmniDocBench, DocLayNet, PubTables-1M, FinTabNet, olmOCR-Bench …)? Welche Metriken misst jeder, und wo sind sie irreführend?
3. Gibt es belastbare Vergleiche zwischen **Multi-Detektor-Ensembles** und modernen Layout-Modellen für Tabellenextraktion? Ist der Ensemble-Ansatz 2026 noch Stand der Technik oder überholt?
4. Wartungsstand von **camelot-py, img2table, pdfminer.six, pdfplumber, PyMuPDF, unstructured, markitdown, docling**: Release-Kadenz 2025/2026, offene Issues, Breaking Changes, Anzeichen von Aufgabe.

### Block 2 — Was darf ins Image, was muss auf die GPU?

5. Was leisten die Kandidaten konkret **auf CPU** — Sekunden pro Seite, RAM-Spitze, Image-Zuwachs in GB? Welche haben eine dokumentierte **CPU-only- oder onnxruntime-Distribution ohne torch**?
6. Welche **OCR-/Dokument-VLMs laufen 2026 produktiv auf 12 GB VRAM** (olmOCR, dots.ocr, Nanonets-OCR, MinerU-VLM, Qwen-VL-Klasse in Quantisierungen, Surya)? Durchsatz, Qualität, VRAM-Bedarf, **deutsche Sprachqualität**.
7. Wie stabil ist **GPU-Passthrough für eine RTX A2000 unter Docker** (nvidia-container-toolkit-Stand, Treiber-Kompatibilität, Verhalten bei Kernel-Updates)? Ist das eine betrieblich fragile Abhängigkeit?
8. Welche **OCR-Engine liefert auf deutschem Material die beste Qualität pro CPU-Sekunde** — Tesseract 5, PaddleOCR, RapidOCR, docTR, Surya, Cloud-OCR? Wie gut sind die deutschen Modelle jeweils?

### Block 3 — Die Formate, die heute versagen

9. Welches Werkzeug erhält 2026 **DOCX-Struktur** (Überschriften, Listen, Tabellen, Fußnoten) am besten in Markdown — pandoc, mammoth, markitdown, docling, `unstructured` mit ausgewerteter Element-Kategorie? Gibt es einen belegten Vergleich?
10. Dasselbe für **XLSX/CSV** und für **PPTX**.
11. Wie entfernt man **Web-Boilerplate** beim HTML→Markdown-Schritt zuverlässig (trafilatura, readability-Ports, Firecrawl, jina-reader-artige Dienste)? Welche sind offline/LAN-tauglich?
12. Stand der **Formel-Erkennung PDF→LaTeX** (docling-Formel-Enrichment, texify/marker, pix2tex, Mathpix, VLM direkt): Qualität, CPU-Tauglichkeit, Kosten.

### Block 4 — Architektur und Datenmodell

13. Gibt es ein **etabliertes Zwischenformat** für konvertierte Dokumente, an das man sich hängen kann, statt ein eigenes Blockschema zu erfinden — DoclingDocument, unstructured-Elements, MinerU-middle-json, Pandoc-AST? Welches trägt Provenienz (Seite, Bbox), Konfidenz je Block und ein „modellgeneriert"-Flag?
14. Existiert ein **fertiges Serving-Layer**, das man statt eines Eigenbaus nehmen kann — docling-serve, unstructured-api, Apache Tika Server, Marker-API? Reife, Auth-Fläche, Image-Fußabdruck, Lizenz.
15. Kann **Gemini PDFs nativ als Dokument-Input** verarbeiten statt seitenweise gerenderter PNGs? Wie verhält sich das preislich und qualitativ gegenüber dem Screenshot-Pfad (Seitenobergrenzen, Bild-Token-Abrechnung)?
16. Welche **Screenshot-Auflösung** ist für VLM-Dokumentlesen belegt optimal? Der Bestand hat vier verschiedene Werte (216, 288, 200, 300 dpi) und keine einzige Begründung. Gibt es eine Kosten-/Qualitätskurve?

### Block 5 — Vertrauen und Recht

17. **Halluzinationserkennung in VLM-Konvertaten**: Gibt es 2026 etablierte Verfahren jenseits von „gegen die deterministische Extraktion vergleichen"? Wie misst man, ob ein Modell gelesen oder erfunden hat?
18. **Lizenzlage**, sobald die Konvertierung ein über Netz erreichbarer Dienst wird: **PyMuPDF ist AGPL-3.0** und trägt heute den gesamten PDF-Pfad; Ghostscript (AGPL) trägt camelot. Wie greift die AGPL-Netzwerk-Klausel bei einem internen, nicht-öffentlichen Dienst? Was gilt für die übrigen Kandidaten?
19. **Vertragslage der Modell-Anbieter** für die durchlaufenden Dokumente: Trainings-Ausschluss, Retention, Datenresidenz — und der Unterschied zwischen **Google AI Studio und Vertex AI**. Gleiches für Anthropic und die Azure-/AWS-Dienste.

---

## Ausgabeformat — verbindlich

Die Ergebnisse werden in ein Befund-Register überführt und einzeln nachgemessen. Liefere deshalb **keine Erzählung**, sondern folgende Struktur.

### Teil 1 — Befunde

Jeder Befund einzeln, durchnummeriert, in genau dieser Form:

```
### B-07 — <Behauptung in einem Satz, entscheidungsrelevant>
**Beleg**: <Quelle mit Datum; was genau dort steht; bei Benchmarks die Zahl>
**Konfidenz**: hoch | mittel | niedrig — <warum genau diese Stufe>
**Gilt unter**: <Randbedingung: CPU/GPU, Sprache, Dokumentklasse, Version>
**Nachmessbar durch**: <ein konkreter Test, den der Entwickler an eigenen Dokumenten fahren kann>
**Widerspruch**: <andere Quelle, die etwas anderes sagt — oder "keiner gefunden">
```

Regeln dazu:
- **Konfidenz „hoch" nur bei reproduzierbarer Messung** aus unabhängiger Quelle. Herstellerangaben sind höchstens „mittel", und das musst du dazuschreiben.
- **Benchmark-Zahlen immer mit Datum und Version.** Eine Zahl ohne beides ist wertlos, weil sich diese Werkzeuge quartalsweise ändern.
- **Kennzeichne, wenn eine Zahl auf englischem Material erhoben wurde** und auf deutsches übertragen wird.

### Teil 2 — Empfehlung je Entscheidungspunkt

Für jeden der folgenden Punkte eine **gereihte** Empfehlung (Platz 1–3) mit dem Handel im Klartext:

- Engine für PDF mit Textebene
- Engine für gescanntes PDF
- Engine für DOCX · für PPTX · für XLSX/CSV · für HTML/EML
- Formel-Erkennung: ja/nein, und womit
- Zwischenformat/Datenmodell
- Serving-Layer: fertig übernehmen oder selbst bauen
- CPU-only-Variante vs. GPU-Variante — **beide ausformuliert**, nicht eine als Sieger
- Rolle des Modells: ersetzen dürfen, nur ergänzen dürfen, oder nur auf textlosen Seiten

Jede Empfehlung endet mit einem Satz: **„Falsch, wenn …"** — die Bedingung, unter der die Empfehlung kippt.

### Teil 3 — Was du nicht herausfinden konntest

Pflichtabschnitt. Liste jede Frage, zu der du **keine belastbare Quelle** gefunden hast, und sage warum (gibt es nicht / hinter Paywall / widersprüchlich / zu neu). **Fülle Lücken nicht mit Plausibilität.** Dieser Abschnitt entscheidet mit, was der Entwickler selbst messen muss — er ist so wertvoll wie Teil 1.

### Teil 4 — Was mich überrascht hat

Kurz. Befunde, die der Fragestellung widersprechen oder eine Annahme des Auftrags kippen. Wenn der Auftrag eine falsche Prämisse enthält, ist das hier der Ort, es zu sagen.

---

## Ausdrücklich nicht recherchieren

- **Keine Tutorials, keine Code-Beispiele, keine Installationsanleitungen.** Nur Entscheidungsgrundlagen.
- **Keine Bewertung der beschriebenen Bestandsimplementierungen.** Die ist erfolgt; du kennst nur die Zusammenfassung und würdest raten.
- **Keine allgemeine RAG-, Chunking- oder Embedding-Recherche.** Der Dienst endet beim Markdown.
- **Keine Frontend-, UI- oder UX-Fragen.**
- **Keine Werkzeuge ohne aktive Wartung**, außer als ausdrückliche Warnung („X wird oft empfohlen, ist aber seit … tot").

═══ ENDE AUFTRAG ═══
