# 05 — Scan, sauberer moderner deutscher Druck

**Was ist hier schwierig?** Eigentlich nichts — und genau das ist der Punkt. Scharfer
Buchdruck, hoher Kontrast, nicht schief. Das ist Tesseracts Sweet-Spot; wenn ein lokaler
Pfad *hier* nicht nahe an Kommerz-Niveau kommt, ist die Messlage falsch. Die echten Hürden
sind der Fußnotenapparat am Seitenfuß, der vom Fließtext getrennt bleiben muss, und die
deutschen Umlaute.

- Datei: `dahlhaus_beethoven-kritik_gerastert-300dpi.pdf`
- 15 Seiten, DE, **reines Bild ohne jede Textebene**
- Gemessen nach dem Rastern: Textebene **0 Zeichen**, Bildabdeckung **1,00**, Textdichte
  **0,00** → CONVERTERs Klassifikator sagt **SCANNED**, der OCR-/VLM-Pfad wird betreten

## Warum diese Datei gerastert wurde

Das Original (600 dpi) trägt bereits eine **OCR-Textebene**, und zwar eine mit englischem
Modell erzeugte: Seite 2 liest sich dort `Asthetik des Erhabenen`. Bei einer Textdichte von
8,20 gegen den Schwellwert 0,5 hätte CONVERTER die Seite als **NATIVE** eingestuft, die
kaputte Ebene unverändert zurückgegeben und **keine** Degradation gemeldet — der OCR-Pfad
wäre nie betreten worden.

Deshalb liegt hier die textebenen-freie Fassung (300 dpi, bilevel, Schwelle 150, aus dem
600-dpi-Original gerendert). Das Original liegt unverändert in `../14_ocr-ebene-kaputt/` und
prüft dort einen anderen, ebenso realen Fall.

## Nullpunkt (gemessen 2026-07-31, Seite 2, Zeile 2)

| Lauf | Ergebnis |
|---|---|
| Raster + `tesseract -l deu` | `Ästhetik des Erhabenen` ✅ |
| Raster + `tesseract -l eng` | `Asthetik des Erhabenen` ✗ |

Das ist zugleich der Beleg, dass der Raster gut genug ist: Deutsch-OCR trifft den Umlaut auf
demselben Bild, auf dem Englisch-OCR ihn verliert. Ein Kandidat, der hier `Asthetik`
liefert, scheitert am Sprachmodell, nicht an der Bildqualität.

**⚠️ Rechtehinweis:** wissenschaftlicher Aufsatz aus einem urheberrechtlich geschützten Band.
Für einen privaten OCR-Test unkritisch, aber kein gemeinfreies Werk — falls das für den
Cloud-Upload stört, siehe Alternativen in `../LUECKEN.md`.
