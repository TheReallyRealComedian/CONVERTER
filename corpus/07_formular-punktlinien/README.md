# 07 — Deutsches Formular mit Punkt-/Unterstrichlinien

**Was ist hier schwierig?** Der dokumentierte VLM-Loop-Trigger. Seitenweise
Ausfülllinien (`seit ______________`, `Telefon-Nr. ____________________`), Ankreuzkästchen
und ein Formularraster, dessen Zellen leer sind. Ein VLM, das eine leere Linie „sinnvoll"
füllen will, erfindet hier Inhalte — und weil das Dokument blanko ist, fällt jede
Halluzination sofort auf.

- Datei: `AOK-PLUS-Fragebogen-Aufnahme-in-Familienversicherung.pdf`
- 2 Seiten, DE, nativ; 54 Punkt-/Unterstrichläufe, 6 lange Formularlinien
- **Blanko-Behördenvordruck der AOK PLUS — enthält keinerlei Personendaten** → cloud-fähig
- Seite 2 bringt zusätzlich eine Tabelle mit verbundenem Kopf (Ehegatte | Kind | Kind | Kind)

**Abweichung von der Vorgabe:** Die Liste sieht für #07 einen *Scan* vor (so auch in der
Gold-Fassung: „der Scan mit den Punktlinien"). Diese Datei ist ein natives PDF — die
Textlage ist also extrahierbar und der VLM-Pfad wird gar nicht erst betreten. Um den
Loop-Trigger echt zu testen, diese Datei einmal rastern oder ausdrucken und scannen; dann
deckt sie zugleich die Lücke aus `../06_scan-degradiert/`.
