# corpus/bakeoff/harness/manifest.py
"""Klassen-Manifest des Bake-offs: welche Datei ist der Input, wo ist Gold.

Eine Klasse = ein Eintrag. ``09`` hat zwei Belegexemplare (A/B), die als
eigene Laeufe gefahren und als Unterzeilen berichtet werden. ``07`` laeuft
gegen den **Scan** — das ist die Daseinsberechtigung der Klasse (Punktlinien
als VLM-Loop-Trigger, siehe README der Klasse); das native PDF dient nur als
Referenz-Textebene. Gold-Vergleiche laufen gegen **abgeleitete Inputs**
(nur die transkribierten Seiten), damit die Metrik exakt das misst, was
transkribiert wurde — kein fuzzy Seiten-Slicing im Output. Ausnahme ``08``:
eine DOCX laesst sich nicht seitenweise schneiden, dort schneidet
``score_gold`` den Kandidaten-Output anker-basiert.
"""

from pathlib import Path

CORPUS = Path(__file__).resolve().parents[2]
BAKEOFF = CORPUS / "bakeoff"
DERIVED = BAKEOFF / "derived"
RESULTS = BAKEOFF / "results"
REFERENCES = RESULTS / "_references"

CLASSES = {
    "01": {
        "dir": "01_paper-zweispaltig",
        "file": "67 - Albert-Barabasi - Statistical Mechanics of Complex Networks (2001).pdf",
        "format": "pdf",
        "title": "Paper zweispaltig (EN, 54 S.)",
        "gold": "01.md",
        "gold_input": "01_gold-seiten.pdf",  # Seiten 4+8, gebaut von refs.py
    },
    "02": {
        "dir": "02_guideline",
        "file": "berichtsband_if2010_II.pdf",
        "format": "pdf",
        "title": "Regulatorische Guideline (DE, 57 S.)",
    },
    "03": {
        "dir": "03_tabelle-seitengrenze",
        "file": "zuordnung_Angebot-Vermarkter_if2010_II.pdf",
        "format": "pdf",
        "title": "Tabelle ueber Seitengrenze (DE, 20 S.)",
        "gold": "03.md",
        "gold_input": "03_gold-seiten.pdf",  # Seiten 11+12, gebaut von refs.py
    },
    "04": {
        "dir": "04_verbundene-zellen",
        "file": "06_ranking_angebote_monat_if2010_I.pdf",
        "format": "pdf",
        "title": "Verbundene Zellen (DE, 12 S.)",
    },
    "05": {
        "dir": "05_scan-sauber",
        "file": "dahlhaus_beethoven-kritik_gerastert-300dpi.pdf",
        "format": "pdf-scan",
        "title": "Scan sauber, textebenen-frei (DE, 15 S.)",
        "no_reference": True,  # keine Textebene, kein Gold
    },
    "06": {
        "dir": "06_scan-degradiert",
        "file": "aok_kopie-3fach-schief.pdf",
        "format": "pdf-scan",
        "title": "Scan degradiert (Kopie der Kopie, schief)",
        "no_reference": True,
    },
    "07": {
        "dir": "07_formular-punktlinien",
        "file": "aok_scan-300dpi.pdf",
        "format": "pdf-scan",
        "title": "Formular Punktlinien (Scan, DE, 2 S.)",
        "gold": "07.md",
        "gold_input": "07_gold-seite2.pdf",  # Seite 2 des Scans
        # Referenz-Textebene kommt aus dem nativen Schwester-PDF (gleicher Inhalt):
        "reference_file": "AOK-PLUS-Fragebogen-Aufnahme-in-Familienversicherung.pdf",
    },
    "08": {
        "dir": "08_docx-fussnoten",
        "file": "Leitfaden - Businessplan 1.1.docx",
        "format": "docx",
        "title": "DOCX Fussnoten (DE)",
        "gold": "08.md",
        "gold_slice": True,  # Gold deckt nur einen Abschnitt; score_gold schneidet anker-basiert
    },
    "09A": {
        "dir": "09_pptx-smartart",
        "file": "A_mehrspaltig-notes_Praesentation_final.pptx",
        "format": "pptx",
        "title": "PPTX mehrspaltig + Notes (DE, 25 Folien)",
    },
    "09B": {
        "dir": "09_pptx-smartart",
        "file": "B_smartart_KI-Praesentation-vb.pptx",
        "format": "pptx",
        "title": "PPTX SmartArt (EN, 98 Folien, 53 MB)",
    },
    "10": {
        "dir": "10_html-artikel",
        "file": "spiegel-online_0,1518,455401,00.html",
        "format": "html",
        "title": "HTML Artikel (DE)",
        # Referenz enthaelt Boilerplate — niedriger Recall kann hier GEWOLLT sein
        # (Artikel-Extraktion); im Bericht entsprechend lesen.
        "reference_note": "Referenz = kompletter Seitentext inkl. Boilerplate",
    },
    "11": {
        "dir": "11_eml-zitatkette",
        "file": "polar-care_zitatkette.eml",
        "format": "eml",
        "title": "EML Zitatkette (DE)",
    },
    "12": {
        "dir": "12_grosses-pdf",
        "file": "Enquete Komission - Recht und Ethik der modernen Medizin - Abschlussbericht.pdf",
        "format": "pdf",
        "title": "Grosses PDF (DE, 280 S.) — Durchsatz/Kosten",
    },
    "13": {
        "dir": "13_mischdokument",
        "file": "13 - Bock Andreas - Telekom Deutschland.pdf",
        "format": "pdf-mixed",
        "title": "Mischdokument (6 mixed / 2 native / 2 scanned von 32 S.)",
        "reference_note": "Textebene deckt nur die nativen Seiten — Recall entsprechend lesen",
    },
    "14": {
        "dir": "14_ocr-ebene-kaputt",
        "file": "Dahlhaus – ETA Hoffmanns Beethoven-Kritik und die Ästhetik des Erhabenen.pdf",
        "format": "pdf",
        "title": "OCR-Ebene kaputt (eng-Modell, keine Umlaute; klassifiziert NATIVE)",
        # Die Textebene EXISTIERT, ist aber kaputt. word_recall≈1 gegen sie bei
        # gleichzeitig umlauts≈0 heisst: kommentarlos durchgereicht → Klasse
        # nicht bestanden (siehe Klassen-README).
        "reference_note": "Referenz = die KAPUTTE Ebene; hoher Recall + 0 Umlaute = durchgereicht",
    },
    "15": {
        "dir": "15_tabelle-ohne-kopfwiederholung",
        "file": "Tabelle_Seitenumbruch_ohne_Kopfwiederholung.pdf",
        "format": "pdf",
        "title": "Tabelle ueber Seitengrenze OHNE Kopfwiederholung (DE, 2 S.)",
        "gold": "15.md",
        # Kein gold_input: das Dokument IST zwei Seiten, der Gold-Umfang ist
        # der ganze Output (role "main", kein Slicing).
    },
}


def input_path(class_id: str, role: str = "main") -> Path:
    """Aufloest den Eingabepfad einer Klasse (main oder gold)."""
    c = CLASSES[class_id]
    if role == "gold":
        if "gold_input" not in c:
            raise ValueError(f"Klasse {class_id} hat keinen abgeleiteten Gold-Input")
        return DERIVED / c["gold_input"]
    return CORPUS / c["dir"] / c["file"]


def gold_path(class_id: str) -> Path:
    return CORPUS / "gold" / CLASSES[class_id]["gold"]


def result_dir(candidate: str, class_id: str, role: str = "main") -> Path:
    name = class_id if role == "main" else f"{class_id}.gold"
    return RESULTS / candidate / name
