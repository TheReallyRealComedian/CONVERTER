# 10 — HTML-Artikel mit Nav, Footer, Cookie-Banner

**Was ist hier schwierig?** Boilerplate-Entfernung im Extremfall. Von 84 KB Quelltext sind
nur 16,7 KB überhaupt Text — und davon ist der größte Teil *nicht* der Artikel, sondern das
komplette SPIEGEL-ONLINE-Ressortmenü („Home | Politik | Übersicht Deutschland Ausland
Wirtschaft | Börse Depot Fonds Derivate | Panorama | …"), Marginalspalten und Footer. Wer
einfach alles Sichtbare einsammelt, bekommt ein Navigationsverzeichnis mit einem Artikel
darin. Das ist die Stelle, an der trafilatura gegen den Rest antritt.

- Datei: `spiegel-online_0,1518,455401,00.html` + Begleitordner `…-Dateien/` (26 Dateien)
- „Korruptes Web 2.0: Verraten und verkauft", SPIEGEL ONLINE Netzwelt, DE
- Vollständig gespeicherte Seite (Text-zu-Quelltext-Verhältnis 0,199)

**Abweichung:** **Kein Cookie-Banner.** Die Seite stammt aus der Zeit vor der
Consent-Banner-Pflicht, und in der Nextcloud existiert keine gespeicherte Artikelseite mit
Banner (358 HTML-Dateien geprüft; die einzigen Treffer mit Consent-Markup sind ein
eBay-Verkaufsformular und ein Immobilien-Exposé, beide mit persönlichem Kontext). Wenn der
Banner Teil der Messung sein soll: eine beliebige aktuelle Nachrichtenseite mit
SingleFile speichern und hier daneben legen.
