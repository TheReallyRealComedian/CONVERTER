"""``unstructured``-Elemente → Markdown (DOC-FIX P2).

Ersetzt das ``"\\n\\n".join(el.text for el in elements)`` des Office-Pfads, das
Element-Kategorie **und** ``metadata.text_as_html`` wegwarf: DOCX/PPTX/HTML/EML
landeten als Fliesstext in einer ``.md``-Datei, ohne eine einzige Tabelle.

Reines Modul — kein Flask, kein SDK, und bewusst **kein Import von**
``unstructured`` selbst. Die Elemente werden per Duck-Typing gelesen
(``.category`` / ``.text`` / ``.metadata.{category_depth,page_number,
text_as_html}``), damit das Modul ohne die schwere Dependency importierbar und
mit synthetischen Element-Listen testbar bleibt und der spaetere
Dokument-Dienst es unveraendert wiederverwenden kann.

Leitregel: **nie mehr Struktur behaupten, als die Quelle hergibt.** Jede Regel
unten steht auf einer Messung an echten ``unstructured``-Objekten (0.18.32,
``strategy="fast"``, 2026-07-31) — inklusive eines echten 32-Folien-Decks und
eines echten DOCX —, nicht auf der Dokumentation.

Was gemessen wurde und was daraus folgt:

* ``Table.metadata.text_as_html`` ist immer eine flache ``<tr><td>``-Matrix.
  Verbundene Zellen werden von ``unstructured`` in **jede** ueberspannte Zelle
  **dupliziert**, ``rowspan``/``colspan`` tauchen nicht auf. Der Span-Zweig
  unten ist trotzdem da: taucht er je auf, sind Pipes nachweislich das falsche
  Format, und rohes HTML im Markdown ist die ehrlichere Ausgabe.
* Zell-Entities sind escaped (``a &lt; b``) → ``convert_charrefs`` dreht das
  zurueck. Zell-**Pipes** sind *nicht* escaped (``Pipe | drin``) → wir escapen
  sie, sonst zerreisst eine Zelle die Spalte.
* ``Title`` ist **keine** verlaessliche Ueberschrift, sondern das Ergebnis
  einer Laengen-/Interpunktions-Heuristik:

  - DOCX/HTML/MD: ``Title`` kommt aus echtem Heading-Markup, ``category_depth``
    ist der Heading-Level (0/1/2) → ATX-Ueberschrift.
  - TXT/EML: **jeder** Absatz wird ``Title``, ``category_depth`` ist ``None``.
    Eine Textdatei duerfte sonst nur aus ``#``-Zeilen bestehen → ohne Tiefe
    keine Ueberschrift, sondern ein Absatz.
  - PPTX: ``_iter_shape_elements`` gibt jede kurze Body-Zeile als ``Title`` mit
    ``category_depth = level + 1`` aus; nur die Titel-Form liefert Tiefe 0. Im
    echten Deck waren das **271** ``Title``-Elemente auf Tiefe 1 gegen 28 echte
    Folientitel — als Ueberschriften waere das genau der verbotene Fall
    „kurz, also ``##``". Deshalb ``source_ext``.

``source_ext`` ist die **einzige** formatabhaengige Verzweigung im Modul, und
sie *reduziert* nur Struktur-Behauptungen. Sie ist kein Schnueffeln: die Route
kennt die Endung ohnehin. Echte PPTX-Bullets bleiben unberuehrt — die kommen
als ``ListItem`` (70 im selben Deck) und werden zu Bullets.

Bewusst nicht gebaut: Absatz-Text wird **verbatim** ausgegeben, inklusive der
Zeilenumbrueche, die in ``el.text`` stecken (PowerPoint-Zeilenumbrueche kommen
als ``\\x0b`` → ``\\n``). Theoretisch koennte eine Folgezeile mit ``#`` oder
``- `` beginnen und damit Struktur erzeugen, die der Absatz nicht behauptet —
aber sie beginnt dann auch in der Quelle so, und die Umbrueche wegzuglaetten
zerstoerte die Zeilenfuehrung des Autors sicher, um einen Grenzfall
hypothetisch zu vermeiden. Blockgrenzen sind davon unberuehrt: alle Bloecke
werden mit einer Leerzeile getrennt, ein Absatz kann also nie in eine Tabelle
oder einen Trenner hineinlaufen.
"""
from collections import Counter
from html.parser import HTMLParser

# ATX kennt sechs Ebenen; tiefere Gliederungen werden gedeckelt statt erfunden.
MAX_HEADING_LEVEL = 6

# Zwei Leerzeichen pro Listenebene (CommonMark-vertraeglich).
_LIST_INDENT = '  '

# Der Seitentrenner, auf den drei unabhaengige Bestandsimplementierungen
# gekommen sind (docs/doc_convert_bestand_2026-07-30.md) und den auch der
# PDF-Pfad in services/pdf_extraction/service.py schon fuehrt.
_PAGE_SEPARATOR = '---'

# Kategorien, die ohne Struktur-Verlust ein Absatz sind — hier entsteht keine
# Warnung. Bewusst **eng** gehalten: alles, was Struktur traegt, die wir nicht
# uebersetzen (``BulletedText``, ``List``, ``Section-header``, ``Image``, …),
# faellt in den Unbekannt-Zweig und wird gemeldet, statt still zu degradieren.
# ``CodeSnippet``/``Formula`` stehen bewusst hier: nicht sonderbehandeln.
_PARAGRAPH_CATEGORIES = frozenset({
    'Abstract', 'Address', 'Caption', 'CodeSnippet', 'EmailAddress',
    'FigureCaption', 'Footer', 'Footnote', 'Formula', 'Header', 'Link',
    'NarrativeText', 'Page-footer', 'Page-header', 'PageNumber', 'Paragraph',
    'Text', 'UncategorizedText',
})

_W_TABLE_NO_HTML = "Tabelle ohne text_as_html — als Fliesstext ausgegeben"
_W_TABLE_UNPARSED = "Tabelle mit unlesbarem text_as_html — als Fliesstext ausgegeben"
_W_TABLE_SPANS = "Tabelle mit verbundenen/ungleichen Zellen — HTML statt Pipe-Tabelle behalten"
_W_PPTX_TITLE = ("PPTX: 'Title' unterhalb der Titelebene als Absatz ausgegeben "
                 "(Laengen-Heuristik, keine Ueberschrift)")


class _TableHTMLParser(HTMLParser):
    """Liest die Zell-Matrix aus ``metadata.text_as_html``.

    stdlib statt lxml/bs4, damit das Modul dependency-frei bleibt.
    ``convert_charrefs`` (Default) macht das Entity-Escaping der Quelle
    rueckgaengig.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.has_spans = False
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self._row = []
        elif tag in ('td', 'th'):
            for key, value in attrs:
                if key in ('rowspan', 'colspan') and (value or '').strip() not in ('', '1'):
                    self.has_spans = True
            self._cell = []
        elif tag == 'br' and self._cell is not None:
            # Zeilenumbruch in der Zelle — die einzige Form, die eine
            # GFM-Zelle ausdruecken kann.
            self._cell.append('<br>')

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self._cell is not None:
            if self._row is None:
                self._row = []
            self._row.append(''.join(self._cell))
            self._cell = None
        elif tag == 'tr' and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _cell_text(raw):
    """Eine Zelle GFM-tauglich machen: Whitespace glaetten, Pipes escapen."""
    return ' '.join(raw.split()).replace('|', r'\|')


def _pipe_table(rows):
    """Rechteckige Zell-Matrix → GFM-Pipe-Tabelle (erste Zeile = Kopf).

    ``unstructured`` emittiert nie ``<th>``; die erste Zeile *ist* der Kopf.
    """
    width = len(rows[0])
    lines = ['| ' + ' | '.join(rows[0]) + ' |',
             '| ' + ' | '.join(['---'] * width) + ' |']
    for row in rows[1:]:
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)


def _meta(element, name):
    """``element.metadata.<name>`` tolerant lesen (Elemente ohne Metadata ok)."""
    metadata = getattr(element, 'metadata', None)
    return getattr(metadata, name, None) if metadata is not None else None


def _category(element):
    return getattr(element, 'category', None) or type(element).__name__


def _text(element):
    return (getattr(element, 'text', None) or '').strip()


def _render_table(element, warn):
    """``Table`` → Pipe-Tabelle, sonst HTML, sonst Fliesstext."""
    html = _meta(element, 'text_as_html')
    if not html:
        warn(_W_TABLE_NO_HTML)
        return _text(element)

    parser = _TableHTMLParser()
    parser.feed(html)
    parser.close()
    rows = [[_cell_text(cell) for cell in row] for row in parser.rows if row]

    if not rows:
        warn(_W_TABLE_UNPARSED)
        return _text(element)

    # Was Pipes nicht ausdruecken koennen, bleibt HTML — eine falsche
    # Pipe-Tabelle waere die unehrlichere Ausgabe.
    if parser.has_spans or len({len(row) for row in rows}) > 1:
        warn(_W_TABLE_SPANS)
        return html.strip()

    return _pipe_table(rows)


def _render_title(element, depth, is_pptx, warn):
    """``Title`` → Ueberschrift nur dort, wo die Tiefe eine echte ist."""
    text = _text(element)
    if not isinstance(depth, int):
        # TXT/EML: jeder Absatz wird ``Title``, ohne Tiefe. Absatz.
        return text
    if is_pptx and depth > 0:
        # PPTX: Tiefe > 0 heisst Body-Zeile, nicht Gliederungsebene.
        warn(_W_PPTX_TITLE)
        return text
    return '#' * min(depth + 1, MAX_HEADING_LEVEL) + ' ' + text


def _page_separator(element):
    """``PageBreak`` → Trenner, mit Seitennummer wenn die Quelle sie liefert.

    Die Nummer am ``PageBreak`` ist die Seite, die *endet* — das Label sagt
    genau das, statt auf die Folgeseite zu rechnen.
    """
    page = _meta(element, 'page_number')
    if page is None:
        return _PAGE_SEPARATOR
    return f'{_PAGE_SEPARATOR}\n\n<!-- Seitenumbruch nach Seite {page} -->'


def _aggregate(counter):
    """Warnungen zu gezaehlten Zeilen buendeln (271 gleiche waeren Log-Muell)."""
    return [
        message if count == 1 else f'{count}× {message}'
        for message, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def elements_to_markdown(elements, source_ext):
    """``unstructured``-Elemente → ``(markdown, warnings)``.

    ``source_ext`` ist die Endung der Quelldatei (mit oder ohne Punkt, Case
    egal) und **Pflicht, ohne Default**. Ein Default waere hier die gefaehrlichere
    Signatur: ``source_ext=None`` faehrt den Nicht-PPTX-Zweig und liefert damit
    fuer genau die 271 Elemente, wegen derer der Parameter existiert,
    ``## Body-Zeile`` — eine still falsche Antwort statt einer lauten. Als
    Pflichtargument wird daraus ein ``TypeError`` an der Aufrufstelle. Die
    heutige Route gibt die Endung immer mit; der naechste Aufrufer ist der
    Dokument-Dienst. ``None`` bleibt als *expliziter* Wert erlaubt und heisst
    „Endung unbekannt".

    ``warnings`` ist die Degradations-Liste: was nicht sauber uebersetzt werden
    konnte, steht **im Rueckgabewert**, nicht nur im Log. Die Route entscheidet,
    was sie damit tut.
    """
    warnings = Counter()

    def warn(message):
        warnings[message] += 1

    ext = str(source_ext).lower().lstrip('.') if source_ext else ''
    is_pptx = ext == 'pptx'

    blocks = []
    # Offener Listen-Lauf: aufeinanderfolgende ``ListItem`` gehoeren in EINEN
    # Block, sonst reisst die Leerzeile zwischen den Bullets die Liste
    # auseinander. Gepuffert als ``(level, text)``, weil der Einzug erst
    # feststeht, wenn die kleinste Tiefe des *ganzen* Laufs bekannt ist.
    run = []

    def close_list():
        if not run:
            return
        base = min(level for level, _ in run)
        blocks.append('\n'.join(
            f'{_LIST_INDENT * (level - base)}- {text}' for level, text in run
        ))
        run.clear()

    for element in elements or []:
        category = _category(element)
        depth = _meta(element, 'category_depth')

        if category == 'PageBreak':
            close_list()
            blocks.append(_page_separator(element))
            continue

        if category == 'ListItem':
            text = _text(element)
            if not text:
                continue
            # Die Basis-Tiefe ist nicht ueberall 0 — DOCX beginnt bei 0,
            # HTML/MD bei 1. Pro Lauf auf die kleinste Tiefe normalisieren,
            # sonst bekaeme jede HTML-Liste eine Phantom-Ebene.
            run.append((depth if isinstance(depth, int) else 0, text))
            continue

        close_list()

        if category == 'Table':
            rendered = _render_table(element, warn)
        elif category == 'Title':
            rendered = _render_title(element, depth, is_pptx, warn)
        elif category in _PARAGRAPH_CATEGORIES:
            rendered = _text(element)
        else:
            warn(f"Unbekannte Element-Kategorie {category!r} — als Absatz ausgegeben")
            rendered = _text(element)

        if rendered:
            blocks.append(rendered)

    close_list()

    return '\n\n'.join(blocks), _aggregate(warnings)
