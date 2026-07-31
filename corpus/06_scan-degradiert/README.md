# 06 — Scan, degradiert  ❌ NICHT GEFUNDEN

**Diese Klasse existiert in der Nextcloud nicht in cloud-fähiger Form.**

Gesucht: Kopie einer Kopie, Fax, leicht schief — der Fall, in dem Tesseract einbricht und nur
ein VLM noch trägt.

Was die Messung ergeben hat (904 Scan-PDFs auf Schärfe, Rauschen, Schieflage, Tonwert geprüft):
Die tatsächlich degradierten Scans sind **ausnahmslos persönliche Dokumente** —
Handyfotos von Steuererinnerungen, ein Abiturzeugnis, Vollmachten. Beispiel für die Bandbreite:
`02 Documents/Scans/2020_08_31 Erinnerung-Steuererklärung OG.pdf` (Laplace-Varianz 106 —
extrem unscharf, Schieflage 0.5°, Mittelton-Anteil 0.92, also komplett ausgewaschen). Das ist
technisch das perfekte Exemplar und darf trotzdem nicht in die Cloud.

Die neutralen Kandidaten (Broschüren, Konferenz-Decks bei 67–100 dpi) sind **keine
Papierscans**, sondern niedrig aufgelöste Digital-Exporte. Die brechen anders als ein Fax und
messen deshalb etwas anderes.

**Vorschlag:** eine öffentliche Seite (z. B. den blanko AOK-Bogen aus `../07_formular-punktlinien/`)
zweimal über einen Kopierer schicken, leicht schief einlegen und einscannen. Damit ist die
Degradation echt *und* der Inhalt cloud-fähig. Alternativ die Klasse streichen — dann fehlt
allerdings genau die Messung, die den VLM-Pfad rechtfertigen würde.
