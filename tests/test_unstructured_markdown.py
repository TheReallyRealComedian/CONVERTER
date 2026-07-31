"""Serializer-Tests fuer ``services/unstructured_markdown.py`` (DOC-FIX P2).

Synthetische Element-Listen statt echter Dateien: ``unstructured`` ist eine
schwere Dependency, die auf der Dev-Maschine gar nicht installiert ist (und in
``conftest.py`` gestubbt wird). Die Stand-ins tragen genau die vier Felder, die
der Serializer liest — ``category``, ``text``, ``metadata.category_depth``,
``metadata.page_number``, ``metadata.text_as_html``.

Die Formen der Stand-ins sind **gemessen**, nicht erfunden: sie stammen aus
Probes gegen ``unstructured==0.18.32`` (``strategy="fast"``) mit einem echten
32-Folien-PPTX und einem echten DOCX, 2026-07-31.
"""
import pytest

from services.unstructured_markdown import elements_to_markdown


class _Meta:
    def __init__(self, category_depth=None, page_number=None, text_as_html=None):
        self.category_depth = category_depth
        self.page_number = page_number
        self.text_as_html = text_as_html


class El:
    """Minimaler Element-Stand-in (Duck-Typing wie der Serializer ihn liest)."""

    def __init__(self, category, text='', **meta):
        self.category = category
        self.text = text
        self.metadata = _Meta(**meta)


# --------------------------------------------------------------------------
# Tabellen
# --------------------------------------------------------------------------

def test_table_with_html_becomes_pipe_table():
    html = ('<table><tr><td>Stoff</td><td>Menge</td></tr>'
            '<tr><td>Ethanol</td><td>12,5</td></tr></table>')
    md, warnings = elements_to_markdown([El('Table', 'Stoff Menge Ethanol 12,5', text_as_html=html)])
    assert md == ('| Stoff | Menge |\n'
                  '| --- | --- |\n'
                  '| Ethanol | 12,5 |')
    assert warnings == []


def test_table_without_html_falls_back_to_text_with_warning():
    md, warnings = elements_to_markdown([El('Table', 'Stoff Menge Ethanol')])
    assert md == 'Stoff Menge Ethanol'
    assert warnings == ['Tabelle ohne text_as_html — als Fliesstext ausgegeben']


def test_table_cell_pipes_are_escaped():
    """Gemessen: ``unstructured`` escaped Entities, aber **nicht** Pipes.

    Eine rohe Pipe in der Zelle zerrisse sonst die Spalte.
    """
    html = '<table><tr><td>a</td><td>Pipe | drin</td></tr></table>'
    md, warnings = elements_to_markdown([El('Table', 'x', text_as_html=html)])
    assert md == '| a | Pipe \\| drin |\n| --- | --- |'
    assert warnings == []


def test_table_cell_entities_are_unescaped():
    html = '<table><tr><td>a &lt; b &amp; c</td></tr></table>'
    md, _ = elements_to_markdown([El('Table', 'x', text_as_html=html)])
    assert md.splitlines()[0] == '| a < b & c |'


def test_table_cell_linebreak_survives_as_br():
    html = '<table><tr><td>c<br/>d</td></tr></table>'
    md, _ = elements_to_markdown([El('Table', 'x', text_as_html=html)])
    assert md.splitlines()[0] == '| c<br>d |'


def test_table_with_spans_keeps_html():
    """Was Pipes nicht ausdruecken koennen, bleibt HTML — mit Warnung."""
    html = '<table><tr><td colspan="2">Breit</td></tr><tr><td>a</td><td>b</td></tr></table>'
    md, warnings = elements_to_markdown([El('Table', 'x', text_as_html=html)])
    assert md == html
    assert warnings == [
        'Tabelle mit verbundenen/ungleichen Zellen — HTML statt Pipe-Tabelle behalten'
    ]


def test_table_with_ragged_rows_keeps_html():
    html = '<table><tr><td>a</td><td>b</td></tr><tr><td>c</td></tr></table>'
    md, warnings = elements_to_markdown([El('Table', 'x', text_as_html=html)])
    assert md == html
    assert len(warnings) == 1


def test_table_with_unparseable_html_falls_back_to_text():
    md, warnings = elements_to_markdown([El('Table', 'Roher Text', text_as_html='<p>kaputt</p>')])
    assert md == 'Roher Text'
    assert warnings == ['Tabelle mit unlesbarem text_as_html — als Fliesstext ausgegeben']


# --------------------------------------------------------------------------
# Ueberschriften
# --------------------------------------------------------------------------

@pytest.mark.parametrize('depth,expected', [(0, '# T'), (1, '## T'), (2, '### T')])
def test_title_depth_becomes_atx_level(depth, expected):
    """DOCX/HTML/MD: ``category_depth`` IST der Heading-Level (gemessen 0/1/2)."""
    md, warnings = elements_to_markdown([El('Title', 'T', category_depth=depth)])
    assert md == expected
    assert warnings == []


def test_title_depth_is_capped_at_six():
    md, _ = elements_to_markdown([El('Title', 'T', category_depth=42)])
    assert md == '###### T'


def test_title_without_depth_is_a_paragraph():
    """TXT/EML: **jeder** Absatz wird ``Title`` ohne Tiefe.

    Als Ueberschrift gelesen bestuende eine Textdatei nur aus ``#``-Zeilen.
    """
    md, warnings = elements_to_markdown([
        El('Title', 'Eine Ueberschrift'),
        El('Title', 'Ein ganz normaler Absatz aus einer Textdatei.'),
    ])
    assert md == 'Eine Ueberschrift\n\nEin ganz normaler Absatz aus einer Textdatei.'
    assert warnings == []


def test_pptx_title_below_title_level_is_a_paragraph():
    """PPTX: kurze Body-Zeilen kommen als ``Title`` mit ``category_depth >= 1``.

    Im echten 32-Folien-Deck waren das 271 Elemente gegen 28 echte Folientitel.
    """
    elements = [
        El('Title', 'Folientitel', category_depth=0),
        El('Title', 'Eine kurze Body-Zeile', category_depth=1),
    ]
    md, warnings = elements_to_markdown(elements, source_ext='pptx')
    assert md == '# Folientitel\n\nEine kurze Body-Zeile'
    assert warnings == [
        "PPTX: 'Title' unterhalb der Titelebene als Absatz ausgegeben "
        "(Laengen-Heuristik, keine Ueberschrift)"
    ]


def test_non_pptx_title_at_depth_one_stays_a_heading():
    """Gegenprobe: dieselbe Element-Form ist in DOCX eine echte ``## ``."""
    md, warnings = elements_to_markdown([El('Title', 'Kapitel', category_depth=1)],
                                        source_ext='docx')
    assert md == '## Kapitel'
    assert warnings == []


@pytest.mark.parametrize('ext', ['pptx', '.pptx', 'PPTX', '.PPTX'])
def test_source_ext_accepts_dot_and_case(ext):
    md, _ = elements_to_markdown([El('Title', 'B', category_depth=1)], source_ext=ext)
    assert md == 'B'


# --------------------------------------------------------------------------
# Listen
# --------------------------------------------------------------------------

def test_list_items_become_one_bullet_block():
    elements = [
        El('ListItem', 'Erster', category_depth=0),
        El('ListItem', 'Zweiter', category_depth=0),
    ]
    md, warnings = elements_to_markdown(elements)
    assert md == '- Erster\n- Zweiter'
    assert warnings == []


def test_list_nesting_uses_category_depth():
    elements = [
        El('ListItem', 'A', category_depth=0),
        El('ListItem', 'A1', category_depth=1),
        El('ListItem', 'A11', category_depth=2),
    ]
    md, _ = elements_to_markdown(elements)
    assert md == '- A\n  - A1\n    - A11'


def test_list_depth_is_normalised_per_run():
    """HTML/MD beginnen bei ``category_depth == 1``, DOCX bei 0 (gemessen).

    Ohne Normalisierung bekaeme jede HTML-Liste eine Phantom-Ebene.
    """
    elements = [
        El('ListItem', 'A', category_depth=1),
        El('ListItem', 'A1', category_depth=2),
    ]
    md, _ = elements_to_markdown(elements)
    assert md == '- A\n  - A1'


def test_list_without_depth_is_flat():
    elements = [El('ListItem', 'A'), El('ListItem', 'B')]
    md, _ = elements_to_markdown(elements)
    assert md == '- A\n- B'


def test_paragraph_between_lists_splits_the_runs():
    elements = [
        El('ListItem', 'A', category_depth=1),
        El('NarrativeText', 'Dazwischen.'),
        El('ListItem', 'B', category_depth=1),
    ]
    md, _ = elements_to_markdown(elements)
    assert md == '- A\n\nDazwischen.\n\n- B'


# --------------------------------------------------------------------------
# Seitentrenner
# --------------------------------------------------------------------------

def test_page_break_carries_the_page_number():
    elements = [
        El('NarrativeText', 'Vorher.'),
        El('PageBreak', '', page_number=1),
        El('NarrativeText', 'Nachher.'),
    ]
    md, warnings = elements_to_markdown(elements)
    assert md == ('Vorher.\n\n'
                  '---\n\n'
                  '<!-- Seitenumbruch nach Seite 1 -->\n\n'
                  'Nachher.')
    assert '\n\n---\n\n' in md
    assert warnings == []


def test_page_break_without_number_is_a_bare_separator():
    elements = [El('NarrativeText', 'A'), El('PageBreak', ''), El('NarrativeText', 'B')]
    md, _ = elements_to_markdown(elements)
    assert md == 'A\n\n---\n\nB'


def test_page_break_closes_an_open_list():
    elements = [
        El('ListItem', 'A', category_depth=0),
        El('PageBreak', '', page_number=2),
        El('ListItem', 'B', category_depth=0),
    ]
    md, _ = elements_to_markdown(elements)
    assert md.startswith('- A\n\n---')
    assert md.endswith('- B')


# --------------------------------------------------------------------------
# Absaetze, unbekannte Kategorien, Degradations-Liste
# --------------------------------------------------------------------------

@pytest.mark.parametrize('category', ['NarrativeText', 'Text', 'UncategorizedText',
                                      'Header', 'Footer', 'CodeSnippet', 'Formula'])
def test_known_categories_become_paragraphs_without_warning(category):
    md, warnings = elements_to_markdown([El(category, 'Inhalt')])
    assert md == 'Inhalt'
    assert warnings == []


def test_unknown_category_becomes_paragraph_with_warning():
    md, warnings = elements_to_markdown([El('Image', 'Ein Bild')])
    assert md == 'Ein Bild'
    assert warnings == ["Unbekannte Element-Kategorie 'Image' — als Absatz ausgegeben"]


def test_warnings_are_aggregated_with_counts():
    """271 identische Warnungen waeren Log-Muell — sie werden gezaehlt."""
    elements = [El('Image', f'Bild {i}') for i in range(3)]
    _, warnings = elements_to_markdown(elements)
    assert warnings == ["3× Unbekannte Element-Kategorie 'Image' — als Absatz ausgegeben"]


def test_empty_element_list_returns_empty_string():
    assert elements_to_markdown([]) == ('', [])


def test_none_element_list_does_not_crash():
    assert elements_to_markdown(None) == ('', [])


def test_blank_elements_are_skipped():
    elements = [El('NarrativeText', '   '), El('NarrativeText', 'Da.'), El('ListItem', '')]
    md, _ = elements_to_markdown(elements)
    assert md == 'Da.'


def test_element_without_metadata_does_not_crash():
    class Bare:
        category = 'NarrativeText'
        text = 'Nackt'

    md, warnings = elements_to_markdown([Bare()])
    assert md == 'Nackt'
    assert warnings == []


def test_category_falls_back_to_class_name():
    class Table:
        text = 'a b'
        metadata = _Meta(text_as_html='<table><tr><td>a</td><td>b</td></tr></table>')

    md, _ = elements_to_markdown([Table()])
    assert md == '| a | b |\n| --- | --- |'


# --------------------------------------------------------------------------
# Zusammenspiel
# --------------------------------------------------------------------------

def test_realistic_docx_shape_round_trip():
    """Die gemessene DOCX-Elementfolge, end-to-end."""
    elements = [
        El('Title', 'Kapitel 1', category_depth=0),
        El('NarrativeText', 'Ein einleitender Absatz.'),
        El('Title', '1.1 Grundlagen', category_depth=1),
        El('ListItem', 'Erster Punkt', category_depth=0),
        El('ListItem', 'Unterpunkt A', category_depth=1),
        El('Table', 'Stoff Menge Ethanol 12,5',
           text_as_html='<table><tr><td>Stoff</td><td>Menge</td></tr>'
                        '<tr><td>Ethanol</td><td>12,5</td></tr></table>'),
        El('UncategorizedText', 'Abschliessender Absatz.'),
    ]
    md, warnings = elements_to_markdown(elements, source_ext='docx')
    assert md == (
        '# Kapitel 1\n\n'
        'Ein einleitender Absatz.\n\n'
        '## 1.1 Grundlagen\n\n'
        '- Erster Punkt\n'
        '  - Unterpunkt A\n\n'
        '| Stoff | Menge |\n'
        '| --- | --- |\n'
        '| Ethanol | 12,5 |\n\n'
        'Abschliessender Absatz.'
    )
    assert warnings == []
