# 14 — Scan mit vorhandener, fehlerhafter OCR-Textebene

**Was ist hier schwierig?** Dass es leicht aussieht. Die Datei ist ein 600-dpi-Bildscan mit
einer bereits eingebackenen OCR-Textebene — nur wurde die mit einem **englischen** Modell
erzeugt und hat jeden Umlaut verloren. Seite 2 liest sich `Asthetik des Erhabenen`.

Für jeden Konverter ist das eine Seite mit gesunder Textebene. CONVERTERs Klassifikator
rechnet Bildabdeckung 1,00 gegen Textdichte 8,20 und stuft **NATIVE** ein (Schwelle: Dichte
< 0,5 für „scanned"), nimmt `page.get_text()` und gibt den Fehler unverändert weiter —
**ohne jedes Degradationssignal**. Der OCR-Pfad wird nie betreten, obwohl er die Rettung
wäre.

- Datei: `Dahlhaus – ETA Hoffmanns Beethoven-Kritik und die Ästhetik des Erhabenen.pdf`
- 15 Seiten, DE, 600 dpi
- Gemessen: Laplace-Varianz 8305 (scharf), Schieflage 0,0°, Bimodalität 0,91,
  Textdichte 8,20 → **NATIVE**

## Was diese Klasse prüft

Nicht OCR-Qualität — die textebenen-freie Fassung derselben Seiten liegt dafür in
`../05_scan-sauber/`. Hier geht es um die Frage, die kein Kandidat des Bestands beantwortet:

**Erkennt der Konverter, dass eine vorhandene Textebene unbrauchbar ist?**

Ein Kandidat besteht diese Klasse, wenn er entweder (a) die kaputte Ebene erkennt und neu
OCR't, oder (b) sie durchreicht, das aber **meldet**. Er fällt durch, wenn er sie
kommentarlos ausliefert — das ist heute CONVERTERs Verhalten.

Der Fall ist nicht konstruiert: gescannte Altbestände mit einmal drübergelaufener OCR sind
in Firmenablagen die Regel, und die Sprache war dabei selten richtig gesetzt.

**⚠️ Rechtehinweis:** wissenschaftlicher Aufsatz aus einem urheberrechtlich geschützten Band.
Für einen privaten OCR-Test unkritisch, aber kein gemeinfreies Werk.
