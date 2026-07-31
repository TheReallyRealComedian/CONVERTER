# Lücken und Abweichungen

Was sich in der Nextcloud **nicht** finden ließ, warum, und was der jeweils kleinste Schritt
zum Schließen wäre. Nach Dringlichkeit sortiert.

---

## 1. Regulatorische Guideline (#02) — muss nachgeladen werden

**Befund:** In den 3.645 PDFs liegt **keine einzige** ICH-, EMA- oder FDA-Guideline. Die
Volltextsuche nach `ICH Q*`, `guideline`, `guidance`, `EMA`, `FDA`, `Annex`, `EudraLex`,
`GMP`, `Pharmacopoeia`, `Leitlinie`, `Verordnung` liefert 10 Treffer — alle sind
Versicherungsdokumente, die auf „*versicherung*" angesprungen sind. Auch die
Projektordner (`01 Projects/iManagement`, `EDP`, `GTM-Public Hub`) enthalten Assetdaten,
Spezifikationen und Protokolle, aber keine Guideline-Dokumente.

**Warum das der wichtigste Punkt ist:** Das ist die Klasse, die als Stellvertreter für dein
Arbeitsmaterial gedacht war — nummerierte Gliederung, lange Tabellen, Fußnoten, zweispaltige
Anhänge. Der eingelegte AGOF-Bericht bildet Gliederung und Tabellenlast ab, aber nicht den
Fußnotenapparat.

**Kleinster Schritt** — eines der folgenden herunterladen (alle frei, keine Registrierung):

| Dokument | Umfang | Warum |
|---|---|---|
| **ICH Q7** *Good Manufacturing Practice for APIs* | ~50 S. | Genau in der Zielspanne, tief nummeriert, viele Tabellen |
| **ICH Q3D(R2)** *Elemental Impurities* | ~80 S. | Sehr tabellenlastig, gute Ergänzung zu #03/#04 |
| **EMA** *Guideline on process validation for finished products* | ~15 S. | Kurz, mit zweispaltigem Anhang |

ICH Q7 ist der beste Einzeltreffer für diese Klasse.

---

## 2. Degradierter Scan (#06) — existiert nicht in cloud-fähiger Form

**Befund:** 904 Scan-PDFs wurden auf Schärfe, Rauschen, Schieflage und Tonwertverteilung
vermessen. Die tatsächlich degradierten Exemplare sind **ausnahmslos persönliche
Dokumente**. Zur Einordnung die drei stärksten Kandidaten:

| Datei | Messwerte | Warum ausgeschlossen |
|---|---|---|
| `02 Documents/Scans/2020_08_31 Erinnerung-Steuererklärung OG.pdf` | Laplace-Varianz **106**, Schieflage 0,5°, Mittelton 0,92 | Steuerdokument mit Personendaten |
| `02 Documents/Scans/2020_06_19 12_35 Office Lens.pdf` | 206 dpi, Schieflage 0,67°, Mittelton 0,77 | Abiturzeugnis einer dritten Person |
| `02 Documents/Scans/2020_09_08 Vollmacht OG.pdf` | 204 dpi, Laplace-Varianz 1.324 | Vollmacht mit Unterschrift |

Die neutralen Kandidaten in der gleichen Qualitätsspanne (Broschüren, Konferenz-Decks bei
67–100 dpi) sind **keine Papierscans**, sondern niedrig aufgelöste Digital-Exporte — die
brechen anders als eine Mehrfachkopie und messen deshalb etwas anderes.

**Kleinster Schritt:** Den blanko AOK-Bogen aus `07_formular-punktlinien/` ausdrucken,
zweimal über einen Kopierer schicken, leicht schief einlegen, einscannen. Damit ist die
Degradation echt, der Inhalt bleibt cloud-fähig — und die Datei deckt zugleich die
Abweichung unter Punkt 3 ab.

**Alternative:** Klasse streichen. Dann fehlt allerdings genau die Messung, die den
VLM-Pfad überhaupt rechtfertigen würde.

---

## 3. Formular als Scan (#07) — vorhanden, aber nativ

Der AOK-Bogen ist ein Volltreffer beim Inhalt (blanko Behördenvordruck, Unterstrichlinien,
Ankreuzfelder, verbundener Tabellenkopf, keine Personendaten), aber ein **natives PDF**. Die
Textlage ist extrahierbar, der VLM-Pfad wird also gar nicht erst betreten — und genau der
Loop-Trigger sollte hier gemessen werden.

**Kleinster Schritt:** rastern (`pdftoppm -r 300`) oder ausdrucken und scannen.
Kombiniert mit Punkt 2 in einem Arbeitsgang erledigt.

---

## 4. DOCX ohne Änderungsverfolgung (#08)

**Befund:** 511 DOCX geprüft. **Kein einziges** hat Überschriftenhierarchie + echte Fußnoten
+ Tabelle + Änderungsverfolgung zugleich. Die 15 Dateien mit Tracked Changes sind
ausnahmslos betriebliche CMC-Dokumente aus `01 Projects/iManagement/02_Pipeline-Intelligence`;
die reichste davon liegt als `intern/B1_CMC_CMC1-DP-BI-VQD-1204396.docx` im lokalen Satz.
Dokumente mit echten Fußnoten wiederum stammen aus dem Uni-Bestand und wurden nie im
Review-Modus bearbeitet.

**Kleinster Schritt:** `08_docx-fussnoten/Leitfaden - Businessplan 1.1.docx` öffnen,
Änderungsverfolgung einschalten, zwei Absätze überarbeiten, speichern. Zwei Minuten, und die
Klasse ist vollständig.

---

## 5. PPTX auf zwei Dateien verteilt (#09)

**Befund:** 861 PPTX geprüft, 96 davon mit SmartArt. Alle 20 Decks mit ≥4 SmartArt-Grafiken
sind betriebliches Material (Clariant/CLNX, iManagement, GTM-Public Hub, HYVE). Cloud-fähig
bleiben nur vier Decks mit ≤2 SmartArt. Mehrspaltige Textkörper (`numCol`) wiederum gibt es
nur in zehn Dateien überhaupt, und die reichste davon (6 mehrspaltige Körper) hat kein SmartArt.

**Gewählt:** beide Seiten abdecken statt einen faulen Kompromiss schließen — `A_…` bringt
Mehrspaltigkeit, Notes und Tabellen, `B_…` das SmartArt. Der volle Fall liegt in
`intern/B3_Foliensatz_iManagement_MASTER.pptx` (11 SmartArt) für den lokalen Pfad.

**Kleinster Schritt, falls eine Datei gewünscht:** In `A_…` eine SmartArt-Grafik einfügen.

---

## 6. HTML ohne Cookie-Banner (#10)

**Befund:** 358 HTML-Dateien geprüft. Gespeicherte Seiten mit Consent-Markup gibt es genau
zwei: ein eBay-Verkaufsformular (`02 Documents/eBay/2024/…`) und ein Immobilien-Exposé
(`02 Documents/Wohnung/Eigentumssuche/…`) — beide mit persönlichem Kontext, und keines ist
ein Artikel. Der eingelegte SPIEGEL-Artikel hat dafür Boilerplate im Extremfall
(80 % des Textes sind Ressortmenü und Footer), stammt aber aus der Zeit vor der
Banner-Pflicht.

**Kleinster Schritt:** Eine beliebige aktuelle deutsche Nachrichtenseite mit der
SingleFile-Browsererweiterung speichern und danebenlegen. Eine Minute.

---

## 7. EML ohne Anhang (#11)

**Befund:** Nur 6 `.eml` in der Nextcloud, alle transaktional (Vodafone, Nissan) — ohne
Zitatkette. Die 14 `.msg`-Dateien enthalten echte deutsche Outlook-Zitatketten, aber die
einzige mit Kette **und** Signatur **und** Anhang ist betrieblich (liegt als
`intern/B4_Mailkette_Re_2023-fea-0087.eml`). Das Strato-Postfach ist über den
MCP-Connector nur lesbar — ein RFC822-Export für ein zusätzliches Beispiel ist von hier aus
nicht möglich.

Zusätzlich: Alle gefundenen Ketten benutzen den Outlook-Kopfblock
(`Von: / Gesendet: / An: / Betreff:`), nicht das `Am … schrieb …` von Thunderbird/Gmail.
Beide Muster kommen im deutschen Schriftverkehr vor.

**Kleinster Schritt:** Aus dem Mailprogramm eine passende Konversation als `.eml`
exportieren — idealerweise eine mit `Am … schrieb …`, dann sind beide Zitatstile abgedeckt.

---

## 8. Seitenzahl #12: 280 statt 150–250

Der Bundestags-Schlussbericht überschreitet die Spanne um 30 Seiten. Dafür ist er das
**einzige Dokument im Korpus ganz ohne Rechtefrage** (amtliches Werk, § 5 UrhG) und
strukturell der beste Guideline-Verwandte, den die Nextcloud hergibt — nummerierte
Gliederung, zweispaltiger Satz, Tabellen mit verbundenem Kopf und Fußnotenblöcke auf
derselben Seite.

**Alternative in der Spanne:** `02 Documents/Vorträge/Intel- IoT/alt/mobilitaetsentwicklung-2050.pdf`
(181 S., DE) — strukturell deutlich ärmer, misst im Wesentlichen nur Durchsatz.

---

## Nicht aufgenommen, obwohl technisch perfekt

Vollständigkeitshalber — diese Dateien wären ideale Prüfstücke gewesen und sind aus
Datenschutz- bzw. Rechtegründen ausgeschieden:

| Datei | Wofür sie ideal gewesen wäre | Warum nicht |
|---|---|---|
| `02 Documents/Wohnung/2023_Rhein-Main/…/Wohnraummietvertrag 20230119.pdf` | #07 — 97 Punktläufe, 19 Formularlinien, mustergültiger deutscher Vordruck | Ausgefüllt: Reisepassnummer, Geburtsdaten, Anschriften |
| `00 Inbox/Scans/Scan_2026-05-14_08-23-22.pdf` | #05 — makelloser Scan eines modernen deutschen Behördenschreibens | Name, Anschrift, IBAN, Kfz-Kennzeichen |
| `02 Documents/Scans/2020_08_31 Erinnerung-Steuererklärung OG.pdf` | #06 — Laplace-Varianz 106, der perfekt kaputte Scan | Steuerdokument |
| `02 Documents/Finanzen/Elterngeld/…/Onlineantrag_Elterngeld_Original.pdf` | #07 — 309 lange Formularlinien | Ausgefüllter Elterngeldantrag |
| `01 Projects/iManagement/06_Project-Office/Präsentationen/iManagement_MASTER.pptx` | #09 — 11 SmartArt, 74 Tabellen, 2.085 Bullets | Betrieblich → liegt in `intern/` |

---

## Klassen, die bei dir gar nicht vorkommen

Keine. Alle 13 Klassen haben in der Nextcloud mindestens einen Vertreter — sechs davon
allerdings nur in Form, die aus Datenschutz-, Rechte- oder Format­gründen nicht in den
cloud-fähigen Satz darf. Das ist ein anderes Problem als „kommt bei mir nicht vor" und
sollte nicht zum Streichen einer Klasse führen: Der Bedarf ist da, nur das Belegexemplar
fehlt.
