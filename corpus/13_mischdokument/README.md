# 13 — Mischdokument: native Seiten *und* Scan-Seiten in einer Datei

**Was ist hier schwierig?** Die Seitenklassifikation — der Punkt, an dem der Eigenbau routet.
Wer die Datei pauschal als „nativ" einstuft, liefert für die Scan-Seiten leere Ausgabe; wer
sie pauschal durch OCR schickt, verschenkt auf den nativen Seiten Qualität und Geld.

Gemessene Seitenfolge über alle 32 Seiten
(`N` = nativer Text, `S` = reiner Scan, `O` = Scan mit OCR-Lage, `.` = wenig Inhalt):

```
. S O N N N N N S . S S S S S S S S . . . S S . . . . S . S N .
1                 10                  20                  30
```

Also ein zusammenhängender nativer Block (S. 4–8) und ein zusammenhängender Scan-Block
(S. 10–17) in **derselben** Datei — genau der „eingescannte Anhang an ein digitales Dokument".

- Datei: `13 - Bock Andreas - Telekom Deutschland.pdf`
- 32 Seiten, DE; Scan-Anteil 0.44, nativer Anteil 0.31
- Konferenzvortrag, öffentlich gehalten → cloud-fähig
