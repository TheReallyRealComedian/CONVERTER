# 09 — PPTX: Bullets, mehrspaltige Folien, SmartArt, Speaker-Notes

**Was ist hier schwierig?** SmartArt liegt nicht als Text, sondern als `diagrams/data*.xml`
plus gerenderte Grafik vor — alle vier Werkzeuge verlieren es. Mehrspaltige Textkörper
(`numCol`) brechen die Lesereihenfolge, und echte Bullets müssen von der Titel-Heuristik
unterschieden werden, die sonst jede kurze Zeile zur Überschrift erklärt.

**Zwei Dateien, weil keine einzelne cloud-fähige alle vier Eigenschaften trägt** (861 PPTX geprüft):

- `A_mehrspaltig-notes_Praesentation_final.pptx` — 25 Folien, DE, **6 mehrspaltige
  Textkörper**, 1.780 Zeichen Speaker-Notes, 9 Tabellen. 0,2 MB. Kein SmartArt.
- `B_smartart_KI-Praesentation-vb.pptx` — 98 Folien, EN, **1 SmartArt**, 931 Zeichen Notes,
  114 Bullets, 7 Tabellen. 53 MB — taugt nebenbei als Durchsatz-Test.

**Warum der Kompromiss:** Alle 20 Decks mit ≥4 SmartArt-Grafiken sind betriebliches Material
(Clariant/CLNX, iManagement, GTM-Public Hub, HYVE) und dürfen nicht in die Cloud. Das
SmartArt-reichste interne Deck liegt in `../intern/` und deckt den Fall für den lokalen Pfad ab.
