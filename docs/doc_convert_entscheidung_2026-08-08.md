# Entscheidungs-Doc — welche Engine der Dokument-Dienst bekommt

**Datum**: 2026-08-08 · **Grundlage**: [doc_convert_bakeoff_2026-08-08.md](doc_convert_bakeoff_2026-08-08.md) (13 Kandidaten × 14 Klassen, 113 Läufe, 16 Judge-Verdikte, 7,61 € Cloud-Kosten) · **Verfahren**: [doc_convert_verwertung_2026-07-30.md](doc_convert_verwertung_2026-07-30.md), Stufe 2

> **Für wen das geschrieben ist**: Oli, der entscheiden muss und die Bake-off-Details nicht gelesen hat. Jeder Befund steht hier mit seiner Zahl, aber die Frage lautet nicht „welche Zahl ist größer", sondern „was kostet mich welche Wahl".

---

## Die Kurzfassung in fünf Sätzen

Die Frage „welche Engine" hat **keine Antwort**, weil kein Kandidat mehr als die Hälfte der Formate gewinnt — die Gewinner sind pro Format verschieden und die Abstände sind groß. Der Eigenbau ist **nicht schlecht, sondern zweigeteilt**: er gewinnt drei Klassen als Einziger und ist sonst entweder wirkungslos oder still zerstörerisch. Zwei seiner Mechanismen kann **niemand sonst**, fünf seiner Detektoren leisten auf dem Korpus **nichts**. Die gefährlichsten Fehler des ganzen Felds sind nicht die falschen Ergebnisse, sondern die **stillen** — hoher Score, kaputter Inhalt, keine Meldung. Und quer durch beide Felder gilt: **wer Text generiert, erfindet unter Druck; wer Text extrahiert, schweigt lieber.**

---

## 1. Warum „eine Engine" die falsche Frage war

Gewinner pro Format, mit dem Abstand zum Zweiten:

| Format | Gewinner | Abstand |
|---|---|---|
| PDF nativ, Tabellen | **gemini-nativ** (medium) | 640/640 und 655/655 wertgenau — der Zweite verliert still Zeilen |
| PDF, Tabelle über Seitengrenze | **Eigenbau** | einziger mit *einer* fortlaufenden Tabelle; alle anderen fragmentieren |
| Scans | **wer die Seite ansieht** | tesseract liefert auf der degradierten Seite 22 Wörter *insgesamt* |
| DOCX Fußnoten + Bilder | **pandoc** | 4/4 gegen **0/4** bei allen anderen |
| PPTX | **markitdown** | Recall 1,0 und als Einziger die Sprechernotizen |
| HTML | **trafilatura** | <2 % Boilerplate gegen 31 % |
| EML | unstructured | konkurrenzlos, aber niemand behandelt E-Mail als E-Mail |

Das ist kein knappes Feld mit einem Gesamtsieger. Das sind sieben verschiedene Probleme mit sieben verschiedenen besten Antworten — und die Anforderungs-Union hat genau das vorhergesagt: *„Router mit fünf Backends oder Pipeline mit einem"*.

**Empfehlung 1: Der Dienst wird ein Router.** Format rein, passendes Backend raus. Das ist keine Bequemlichkeitslösung, sondern das, was die Messung erzwingt — jede Ein-Engine-Wahl verschenkt mindestens zwei Formate vollständig.

---

## 2. Was mit den 1.418 Zeilen passiert

Die Antwort ist chirurgisch, nicht „behalten" oder „wegwerfen".

**Was der Eigenbau als Einziger kann:**

- **Seitenübergreifende Tabellen zusammenführen** (`multi_page.py`). Klasse 03 gewinnt er damit gegen das gesamte Feld — alle anderen liefern 20 Fragmente statt einer Tabelle.
- **Seiten-Routing** (`nativ` / `mixed` / `scanned`). Klasse 13 gewinnt er damit; auf 06 liest er als einziger CPU-Kandidat die um 180° gedrehte Seite und trifft das Rechtszitat ziffernexakt, wo Gemini es verfälscht.

**Was die fünf Detektoren leisten:** auf `01.gold` ist die Ausgabe des Eigenbaus **bis auf die vierte Nachkommastelle identisch mit roher Textextraktion** — Wort-F1 0,9149, CER 0,0645, **Tabellenzellen 0,0**, Tief-/Hochstellungen 0 von 11. Auf einer Seite mit 37 Tabellen-Datenzeilen. Auf Klasse 02 verliert er **die Hälfte des Wortbestands** (Recall 0,489), weil Falschdetektion Fließtext zerhackt: „`| im per | s | önlichen | Gesprä |`".

**Empfehlung 2: Die fünf Detektoren fallen, Routing und Merge werden zum Kern des Routers.** Sie sind genau das, was ein Router ohnehin braucht — eine Entscheidung, welche Seite an welches Backend geht, und eine Klammer über Seitengrenzen hinweg. Der Merge muss dabei neu auf die Ausgabe des jeweiligen Backends aufgesetzt werden; die Idee trägt, die Implementierung hängt an den alten Detektoren.

---

## 3. Der Befund, der die Architektur bestimmt

Das ist der wichtigste Absatz dieses Dokuments.

**Auf Klasse 12 erreicht der Eigenbau Recall 0,9841 — und ersetzt dabei die Seiten 40 und 70 kommentarlos durch eine erfundene Tabelle. `warnings: []`.** Der hohe Score *verdeckt* den Defekt, weil der erfundene Text als Wörter zählt. Keine automatische Kennzahl des Harness hat das gefunden; nur ein Mensch, der hinsah.

Dasselbe Muster, andere Kandidaten: docling verliert still einzelne Tabellenzeilen und erfindet rekombinierte Namen („Nürtinger Zeitung Westfälische"). marker reicht 14 von 15 Seiten durch und OCRt eine still neu — unmarkierte Mischherkunft. Auf Klasse 14 meldet **kein einziger der 13 Kandidaten**, dass die vorhandene Textebene kaputt ist.

**Und quer durch beide Felder**: auf beschnittenen Zeitungsspalten füllen **exakt die generativen Decoder** unsichtbar auf — Gemini, dots und marker schreiben dort flüssigen Pseudo-Volltext, teils fast wortgleich, während die Pipeline-Kandidaten wörtlich bleiben. Mehr Decoder-Kontext verstärkt es: derselbe marker mit 8k ließ Fragmente stehen, mit 18k konfabulierte er „Andrew Barker".

**Empfehlung 3: Drei Leitplanken sind ab jetzt Teil der Definition von „fertig", nicht Kür.**

1. **Herkunft je Block** — deterministisch extrahiert, OCR, oder modellgeneriert. Ohne das ist „hoher Score" wertlos, wie Klasse 12 zeigt.
2. **Ein Modell darf ergänzen, nicht ersetzen**, wo eine deterministische Spur existiert. Genau diese Regel hätte alle vier stillen Katastrophen verhindert.
3. **Degradationssignal in der Antwort**, nicht im Log. Ein aufrufender Dienst muss erkennen können, dass er gerade Modelltext bekommt.

---

## 4. Die Wahl je Format — mein Vorschlag

Zwei Spalten, weil du beide Pfade gebaut haben willst (W-1: betriebliches Material bleibt lokal).

| Format | Cloud-Pfad | Lokaler Pfad | Warum |
|---|---|---|---|
| **PDF nativ** | gemini-nativ **medium** | **mineru-vlm** | gemini ist die breiteste Spitze; mineru hat die besten Tabellenzellen (0,916) bei 21,9 S/min |
| **PDF Scan** | gemini-nativ medium | mineru-vlm | wer die Seite ansieht, trägt die Klasse; mineru ist der einzige lokale Allrounder mit Tempo |
| **DOCX** | **pandoc** | pandoc | 4/4 gegen 0/4 bei Bild-Fußnoten-Link; identisch in beiden Pfaden, weil lokal |
| **PPTX** | **markitdown** | markitdown | Recall 1,0, einziger mit Notizen |
| **HTML** | **trafilatura** | trafilatura | <2 % Boilerplate — ⚠️ verliert Titel/Autor/Datum, siehe unten |
| **EML** | unstructured | unstructured | funktional, konkurrenzlos, mit bekannter Schwäche |
| **XLSX** | — | — | **ungemessen**, kein Belegexemplar im Korpus |

**`media_resolution` = medium**, nicht low. Meine frühere Empfehlung ist widerrufen: low fällt bei Bewertungsregel 1 auf 0/3 und erfand 67 statt 43 Ausfülllinien; medium schlägt auch high.

**dots ist bewusst nicht gesetzt**, obwohl es das beste Gold-F1 des ganzen Felds hat (0,9853). Der Preis ist Faktor 8: **110 Minuten für 280 Seiten**, Karte währenddessen voll belegt. Als Standardpfad eines Dienstes ist das keine Option — als *Qualitätsstufe auf Anforderung* für ein einzelnes wichtiges Dokument ist es eine, und die Einstellung dafür hast du ohnehin schon beschlossen.

---

## 5. Was dich das kostet

**Cloud**: 1,48 ct pro Seite (gemini medium, gemessen über 492 Seiten inklusive zweier Fairness-Wiederholungen). Ein 280-Seiten-Dokument kostet **4,14 €** und braucht 26 Minuten.

**Lokal**: 0 € laufend, ~13 Minuten für dieselben 280 Seiten auf der A2000 (mineru, 6,5 GB VRAM). Der Preis ist Qualität: mineru liegt auf der Paper-Goldseite bei 0,9551 gegen 0,9809 bei gemini — und verliert systematisch etwa fünf Fußnoten pro Dokument.

**Der Handel in einem Satz**: Cloud kauft Breite und Tabellengenauigkeit für 1,5 ct/Seite; lokal kauft Datenhoheit und Nulltarif für spürbar mehr stille Lücken.

---

## 6. Drei Dinge, die ich dir nicht abnehmen kann

**(a) HTML: Vollständigkeit oder Sauberkeit?** trafilatura liefert Fließtext mit <2 % Boilerplate — und **verliert Dachzeile, Titel, Autor und Datum ersatzlos**. CONVERTER leitet den Library-Titel aus der ersten Überschrift ab (TITLE-FIX); mit trafilatura ist die weg. unstructured hat alles, aber 31 % Boilerplate. Entweder du nimmst trafilatura und holst die Metadaten separat, oder du nimmst Boilerplate in Kauf.

**(b) Wie viel darf ein Auftrag kosten?** 1,48 ct/Seite ist der Einzelpreis. Ohne Deckel kann ein fehlkonfigurierter Aufrufer beliebig viel erzeugen, sobald der Dienst mehrere Konsumenten hat. Deckel pro Auftrag mit Abbruch, oder mit Degradation auf den lokalen Pfad?

**(c) Wer entscheidet Cloud oder lokal?** Du hast „umschaltbar per Einstellung" gesetzt. Offen ist, ob die Einstellung *global* gilt oder *pro Auftrag* mitgegeben wird. Global ist einfacher; pro Auftrag ist das, was ein Dienst mit mehreren Konsumenten braucht.

---

## 7. Was als Nächstes gebaut wird

In dieser Reihenfolge, wenn du den Empfehlungen folgst:

1. **API-Fläche** — Endpoint, Token, Job-Modell, Antwortform inklusive Herkunft und Degradationssignal. Hängt an keiner Engine, kann sofort starten.
2. **Router mit zwei Backends** — der schnellste Weg zu etwas Nutzbarem: PDF über einen Pfad, Office über pandoc/markitdown. Deckt den Großteil ab.
3. **Die restlichen Backends** und die Rettung von Routing und Merge aus dem Eigenbau.
4. **Der zweite Pfad** (lokal gegen Cloud) samt Umschalter.

Was **nicht** gebaut wird, bis es einen Anlass gibt: XLSX (kein Belegexemplar, kein bekannter Bedarf), SmartArt-Extraktion (verliert das gesamte Feld — machbar über `diagrams/data*.xml`, aber ein eigenes Vorhaben), und eine Human-in-the-Loop-Korrekturfläche.
