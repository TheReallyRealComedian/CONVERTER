# OCR-Baseline (Tesseract, lokal)

Gemessen am 31.07.2026, direkt nach Installation von `tesseract-lang` (163 Sprachen, `deu`
vorhanden). Seiten mit 300 dpi aus den Korpus-PDFs gerendert, dann `tesseract … tsv`;
Konfidenz ist der Mittelwert über alle erkannten Wörter.

Das ist **kein** Bake-off-Ergebnis, sondern der Nullpunkt: was der lokale Pfad kann, bevor
docling/MinerU/Gemini überhaupt antreten.

## Zahlen

| Seite | Sprache | Wörter | Ø-Konfidenz | Wörter < 60 |
|---|---|---|---|---|
| #05 sauberer Scan, S. 4 | `deu` | 443 | **92,5** | 3 % |
| #05 sauberer Scan, S. 4 | `eng` | 447 | 89,9 | 4 % |
| #07 Formular, S. 2 | `deu` | 402 | **87,4** | 9 % |
| #07 Formular, S. 2 | `eng` | 415 | 80,6 | 15 % |
| #13 Scan-Seite (S. 12) | `deu` | 643 | **87,3** | 9 % |
| #13 Scan-Seite (S. 12) | `eng` | 650 | 83,7 | 12 % |
| #13 native Folienseite (S. 6) | `deu` | 142 | 60,4 | 47 % |

## Drei Beobachtungen, die den Bake-off beeinflussen

**1. Der Sprachschalter kostet keine 3 Konfidenzpunkte, sondern jeden Umlaut.**
Auf allen drei deutschen Seiten liest `eng` **null** Umlaute und kein ß — bei nur 2–7 Punkten
schlechterer Konfidenz. Tesseract meldet also hohe Sicherheit für falschen Text. Was
tatsächlich herauskommt:

| Seite | `deu` liest | `eng` liest |
|---|---|---|
| #05 | Blüten, Früchte, glücklich, heißt, darüber | `Bliiten`, `Friichte`, `gliicklich`, `heiBt`, `dariiber` |
| #13 | für, künftig, Verfügbarkeit, Geschäftsführer | `fiir`, `kiinftig`, `Verfiigbarkeit`, `Geschaftsfiihrer` |

Konsequenz für die Metrik: **Wortkonfidenz allein taugt nicht als Qualitätsmaß.** Ein
Zeichenfehler-Maß (CER) gegen die Gold-Fassungen ist Pflicht, sonst gewinnt hier der
falsche Kandidat.

**2. #05 bestätigt seine Rolle.** 92,5 Konfidenz, nur 3 % schwache Wörter, alle 39 Umlaute
sauber — der Sweet-Spot ist real. Wenn ein Cloud-Kandidat *hier* nennenswert besser ist,
zahlt man für nichts.

**3. Die Seitenklassifikation aus #13 rechnet sich sichtbar.** Dieselbe Datei, zwei
Seitensorten: Die Scan-Seite kommt auf 87,3 — die native Folienseite, durch OCR gezwungen,
auf 60,4 bei 47 % schwachen Wörtern. (Die Seite ist eine grafiklastige Folie mit
Displayschrift, das erklärt einen Teil davon.) Der native Textextrakt derselben Seite ist
verlustfrei und praktisch gratis. Genau diese Differenz ist das Argument für den
Routing-Schritt.

## Reproduzieren

```bash
tesseract seite.png ausgabe -l deu tsv
```
