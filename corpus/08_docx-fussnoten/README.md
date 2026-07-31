# 08 — DOCX: Überschriftenhierarchie + Fußnoten + Tabelle

**Was ist hier schwierig?** Die pandoc-gegen-docling-Bruchkante. Drei Überschriftenebenen,
drei echte Fußnotenreferenzen im Fließtext mit zugehörigem `footnotes.xml`, zwei Tabellen
mit verbundenen Zellen. docling verwirft Fußnoten ersatzlos — hier wird das sichtbar,
weil die Referenz im Text steht und der Text ohne sie unvollständig ist.

- Datei: `Leitfaden - Businessplan 1.1.docx`
- DE, ~14.300 Zeichen; `footnote_refs=3`, `footnotes=4`, Überschriften 1/2/3,
  2 Tabellen, 10 verbundene Zellen
- Eigenes Dokument (Uni-Hausarbeit) → cloud-fähig

**⚠️ Fehlt: Änderungsverfolgung.** In der ganzen Nextcloud gibt es **kein** DOCX, das
Überschriftenhierarchie, echte Fußnoten, Tabelle *und* Änderungsverfolgung zugleich hat
(511 DOCX geprüft). Alle 15 Dateien mit Tracked Changes sind betriebliche CMC-Dokumente —
eine davon liegt in `../intern/`. Wer die vierte Eigenschaft cloud-fähig braucht: in dieser
Datei zwei Absätze mit eingeschalteter Änderungsverfolgung überarbeiten.
