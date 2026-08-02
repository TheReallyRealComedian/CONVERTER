#!/usr/bin/env python3
"""Korpus-Statuscheck — sagt pro Klasse, ob sie vollständig ist.

    python3 corpus/pruefen.py

Prüft nicht Inhalte, sondern die Eigenschaften, auf die es beim Bake-off
ankommt: liegt überhaupt eine Quelldatei da, und lösen die Scan-Klassen
CONVERTERs Klassifikator wirklich aus. Die Schwellen unten sind aus
services/pdf_extraction/service.py übernommen — läuft der Klassifikator dort
je auseinander, muss diese Datei nachgezogen werden.
"""
import os
import sys
import glob

HIER = os.path.dirname(os.path.abspath(__file__))

# Der Korpus wurde ursprünglich in der Nextcloud gebaut und von dort ins Repo
# kopiert. Beide Orte existieren weiter — und genau daran ist am 02.08. ein
# Nachtrag verloren gegangen: die Datei lag in der Nextcloud, das Repo sah leer
# aus. Deshalb prüft dieses Skript beide Seiten gegeneinander.
NEXTCLOUD = os.path.expanduser('~/Nextcloud/00 Inbox/Benchmarkfiles')

# Schwellen aus services/pdf_extraction/service.py
SCANNED = lambda cov, dens: cov > 0.7 and dens < 0.5
MIXED = lambda cov, dens: cov > 0.3 and dens < 2.0

# Ordner -> (Beschreibung, erwartete Klassifikation oder None)
KLASSEN = [
    ('01_paper-zweispaltig', 'Paper zweispaltig', None),
    ('02_guideline', 'Regulatorische Guideline', None),
    ('03_tabelle-seitengrenze', 'Tabelle über Seitengrenze', None),
    ('04_verbundene-zellen', 'Verbundene Zellen', None),
    ('05_scan-sauber', 'Scan sauber (gerastert)', 'SCANNED'),
    ('06_scan-degradiert', 'Scan degradiert', 'SCANNED'),
    ('07_formular-punktlinien', 'Formular Punktlinien', None),
    ('08_docx-fussnoten', 'DOCX Fußnoten', None),
    ('09_pptx-smartart', 'PPTX SmartArt', None),
    ('10_html-artikel', 'HTML Artikel', None),
    ('11_eml-zitatkette', 'EML Zitatkette', None),
    ('12_grosses-pdf', 'Großes PDF', None),
    # Mischdokument: die Mehrheit ist egal, es muss BEIDE Sorten enthalten.
    ('13_mischdokument', 'Mischdokument', 'HAT_SCANNED'),
    ('14_ocr-ebene-kaputt', 'OCR-Ebene kaputt', 'NATIVE'),
]

GOLD = ['01.md', '07.md', '08.md']


def klassifiziere(pfad):
    """Häufigste Klassifikation über die ersten Seiten. None = kein PDF-Reader."""
    try:
        import fitz
    except ImportError:
        return None, None
    try:
        d = fitz.open(pfad)
    except Exception:
        return None, None
    zaehler = {}
    seiten = min(len(d), 10)
    for i in range(seiten):
        p = d[i]
        t = p.get_text().strip()
        imgs = p.get_images()
        flaeche = p.rect.width * p.rect.height
        cov = sum(r.width * r.height for im in imgs for r in p.get_image_rects(im[0])) / flaeche if imgs else 0
        dens = len(t) / flaeche * 1000
        k = 'SCANNED' if SCANNED(cov, dens) else ('MIXED' if MIXED(cov, dens) else 'NATIVE')
        zaehler[k] = zaehler.get(k, 0) + 1
    d.close()
    return max(zaehler, key=zaehler.get), zaehler


def main():
    offen = []
    print(f'Korpus: {HIER}\n')
    for ordner, name, erwartet in KLASSEN:
        pfad = os.path.join(HIER, ordner)
        dateien = [f for f in glob.glob(os.path.join(pfad, '*'))
                   if os.path.isfile(f) and not f.endswith('README.md')]
        if not dateien:
            print(f'  ❌ {ordner:26} {name:28} — LEER')
            offen.append(f'{ordner}: keine Quelldatei')
            continue

        info = f'{len(dateien)} Datei(en)'
        warnung = ''
        pdfs = [f for f in dateien if f.lower().endswith('.pdf')]
        if erwartet and pdfs:
            ist, zaehler = klassifiziere(pdfs[0])
            if ist is None:
                info += ' (PyMuPDF fehlt, nicht geprüft)'
            elif erwartet == 'HAT_SCANNED':
                verteilung = ', '.join(f'{k} {v}' for k, v in sorted(zaehler.items()))
                info += f' — Seiten: {verteilung}'
                if not zaehler.get('SCANNED'):
                    warnung = '  ⚠️ keine einzige Scan-Seite'
                    offen.append(f'{ordner}: enthält keine SCANNED-Seite')
            else:
                info += f' — Klassifikator: {ist}'
                if ist != erwartet:
                    warnung = f'  ⚠️ erwartet {erwartet}'
                    offen.append(f'{ordner}: klassifiziert als {ist}, erwartet {erwartet}')
        zeichen = '⚠️' if warnung else '✅'
        print(f'  {zeichen} {ordner:26} {name:28} — {info}{warnung}')

    print()
    fehlend = [g for g in GOLD if not os.path.exists(os.path.join(HIER, 'gold', g))]
    if fehlend:
        print(f'  ❌ gold/                      Soll-Fassungen               — fehlen: {", ".join(fehlend)}')
        offen.append(f'gold/: {len(fehlend)} von {len(GOLD)} Soll-Fassungen fehlen')
    else:
        print(f'  ✅ gold/                      Soll-Fassungen               — alle {len(GOLD)} da')

    # Drift gegen die Nextcloud-Kopie: was dort liegt und hier nicht, ist
    # verlorene Arbeit — sie sieht für alles Nachgelagerte schlicht nicht aus.
    if os.path.isdir(NEXTCLOUD):
        # Vergleich über den Dateinamen, nicht den Pfad: Klassen werden
        # umsortiert (05 → 14 beim Raster-Split), und eine verschobene Datei
        # ist nicht verloren. Gemeldet wird nur, was im Repo NIRGENDWO liegt.
        hier_namen = {d for _w, _o, ds in os.walk(HIER) for d in ds}
        fehlt_hier = []
        for wurzel, ordner, dateien in os.walk(NEXTCLOUD):
            ordner[:] = [o for o in ordner if o != 'intern']
            for d in dateien:
                if d in ('.DS_Store', 'README.md') or d.startswith('.'):
                    continue
                if d not in hier_namen:
                    fehlt_hier.append(os.path.relpath(os.path.join(wurzel, d), NEXTCLOUD))
        if fehlt_hier:
            print()
            print('  ⚠️ Liegt in der Nextcloud, aber NICHT hier:')
            for rel in sorted(fehlt_hier):
                print(f'       {rel}')
            print(f'     → kopieren:  cp "{NEXTCLOUD}/<datei>" "{HIER}/<ordner>/"')
            offen.append(f'{len(fehlt_hier)} Datei(en) nur in der Nextcloud')

    print()
    if offen:
        print(f'OFFEN ({len(offen)}):')
        for o in offen:
            print(f'  - {o}')
        return 1
    print('Korpus vollständig.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
