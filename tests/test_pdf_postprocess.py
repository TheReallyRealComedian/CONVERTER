"""Tests fuer die drei DOC-FIX-P3-Reparaturen in ``services/pdf_extraction``.

Der Pfad hatte bis hierher **null** Tests. Abgedeckt wird genau, was P3
repariert — die Ensemble-Logik, die Detektoren und der Multi-Page-Merge sind
ausdruecklich unberuehrt und werden hier auch nicht angefasst:

1. Die verschwindende Seite: schlaegt jede Tabellenextraktion fehl, behaelt die
   Seite ihren Fliesstext.
2. Der Fence-Sweep: ein echter Codeblock ueberlebt das Post-Processing.
3. Der globale Link-Replace: es wird gar nicht mehr ersetzt, also kollidiert
   auch nichts mehr.

``page`` und ``consensus_table`` sind minimale Stand-ins — die echten Objekte
brauchen ein PDF, und die Fixes haengen nur an den paar Feldern hier.
"""
from unittest.mock import patch

import pytest

from services.pdf_extraction import service as service_module
from services.pdf_extraction.service import PDFExtractionService, _strip_wrapper_fence


class FakePage:
    """Nur ``get_text('dict')`` — das ist alles, was ``_add_non_table_text`` liest."""

    def __init__(self, blocks):
        self._blocks = blocks

    def get_text(self, kind):
        assert kind == 'dict'
        return {'blocks': self._blocks}


class FakeConsensusTable:
    def __init__(self, bbox):
        self.bbox = bbox


def text_block(y_top, text):
    return {
        'type': 0,
        'bbox': (50, y_top, 500, y_top + 20),
        'lines': [{'spans': [{'text': text}]}],
    }


@pytest.fixture
def service():
    # Ohne API-Key ist ``gemini_client`` None → der Vision-Fallback wird
    # uebersprungen, die Tabellenextraktion schlaegt also endgueltig fehl.
    return PDFExtractionService(None)


# --------------------------------------------------------------------------
# 3.1 — die verschwindende Seite
# --------------------------------------------------------------------------

def test_page_keeps_its_text_when_every_table_extraction_fails(service):
    """Der Kern-Bug: erkannte, aber nicht extrahierbare Tabelle loeschte die Seite.

    Frueher stand vor ``_add_non_table_text`` ein ``if table_bboxes:`` — blieb
    die Liste leer, verschwand der gesamte Fliesstext rueckstandslos.
    """
    page = FakePage([
        text_block(100, 'Absatz oberhalb der Tabelle.'),
        text_block(400, 'Absatz unterhalb der Tabelle.'),
    ])
    analysis = {'consensus_tables': [FakeConsensusTable((50, 200, 500, 300))], 'tables': []}

    with patch.object(service_module, 'select_best_extraction', return_value=None):
        md = service._extract_page_with_ensemble(page, 0, analysis)

    assert 'Absatz oberhalb der Tabelle.' in md
    assert 'Absatz unterhalb der Tabelle.' in md


def test_failed_table_extraction_is_logged_as_warning(service, caplog):
    page = FakePage([text_block(100, 'Text.')])
    analysis = {'consensus_tables': [FakeConsensusTable((50, 200, 500, 300))], 'tables': []}

    with patch.object(service_module, 'select_best_extraction', return_value=None):
        with caplog.at_level('WARNING'):
            service._extract_page_with_ensemble(page, 4, analysis)

    assert any('nicht extrahierbar' in r.message for r in caplog.records)
    assert any('Seite 5' in r.message for r in caplog.records)


def test_successful_extraction_still_filters_table_text(service):
    """Gegenprobe: bei gefuellter Bbox-Liste bleibt das Verhalten unveraendert.

    Der Block *innerhalb* der Tabellen-Bbox darf nicht zusaetzlich als
    Fliesstext auftauchen.
    """
    page = FakePage([
        text_block(100, 'Ueber der Tabelle.'),
        text_block(210, 'Zelleninhalt in der Bbox'),
    ])
    analysis = {'consensus_tables': [FakeConsensusTable((50, 200, 500, 300))], 'tables': []}

    with patch.object(service_module, 'select_best_extraction',
                      return_value=[['A', 'B'], ['1', '2']]):
        md = service._extract_page_with_ensemble(page, 0, analysis)

    assert 'Ueber der Tabelle.' in md
    assert 'Zelleninhalt in der Bbox' not in md
    # ``table_to_markdown`` padded die Zellen auf Spaltenbreite.
    assert '| A' in md and '| B' in md and '| --- |' in md


def test_page_without_tables_is_unaffected(service):
    page = FakePage([text_block(100, 'Nur Text.')])
    md = service._extract_page_with_ensemble(page, 0, {'consensus_tables': [], 'tables': []})
    assert md == 'Nur Text.'


# --------------------------------------------------------------------------
# 3.2 — der Fence-Sweep
# --------------------------------------------------------------------------

def test_postprocess_keeps_a_real_code_block(service):
    """Der globale Sweep hat jeden echten Codeblock des Dokuments entfernt."""
    markdown = ('# Kapitel\n\n'
                '```python\n'
                'def f():\n'
                '    return 42\n'
                '```\n\n'
                'Nachtext.')
    assert service._postprocess_markdown(markdown) == markdown


def test_postprocess_keeps_fences_without_language(service):
    markdown = 'Text\n\n```\nroher Block\n```\n\nEnde.'
    assert service._postprocess_markdown(markdown) == markdown


def test_postprocess_still_collapses_excess_blank_lines(service):
    """Die uebrigen Aufgaben des Post-Processings bleiben."""
    assert service._postprocess_markdown('a\n\n\n\n\n\nb') == 'a\n\n\nb'


def test_postprocess_still_normalises_table_rows(service):
    assert service._postprocess_markdown('|a|b|') == '| a | b |'


# --- die Wrapper-Fence an der Quelle ---

def test_strip_wrapper_fence_removes_a_full_wrap():
    assert _strip_wrapper_fence('```markdown\n# A\n\nB\n```') == '# A\n\nB'


def test_strip_wrapper_fence_removes_unknown_language_tag():
    """Die alte Variante liess bei ```md die Sprachmarke als Text stehen."""
    assert _strip_wrapper_fence('```md\n# A\n```') == '# A'
    assert _strip_wrapper_fence('```html\n# A\n```') == '# A'


def test_strip_wrapper_fence_leaves_unwrapped_text_alone():
    assert _strip_wrapper_fence('# A\n\nB') == '# A\n\nB'


def test_strip_wrapper_fence_does_not_cut_a_trailing_code_block():
    """Die alte Variante schnitt hier das ``` ab und zerriss den Block."""
    text = '# A\n\n```python\ncode\n```'
    assert _strip_wrapper_fence(text) == text


def test_strip_wrapper_fence_keeps_inner_blocks_when_ambiguous():
    """Wrap *und* innerer Block: konservativ nichts am Ende abschneiden."""
    out = _strip_wrapper_fence('```markdown\n# A\n\n```python\ncode\n```\n```')
    assert '```python\ncode\n```' in out


# --------------------------------------------------------------------------
# 3.3 — der globale Link-Replace
# --------------------------------------------------------------------------

class FakeLinkPage:
    def __init__(self, links, textbox=''):
        self._links = links
        self._textbox = textbox

    def get_links(self):
        return self._links

    def get_textbox(self, rect):
        return self._textbox


def _uri_link(uri):
    return {'kind': service_module.fitz.LINK_URI, 'uri': uri, 'from': (0, 0, 10, 10)}


def test_link_embedding_is_gone_from_the_pipeline():
    """Es gibt keinen Ersetzungs-Schritt mehr, an dem etwas kollidieren koennte."""
    assert not hasattr(PDFExtractionService, '_embed_links')
    assert not hasattr(PDFExtractionService, '_extract_links')


def test_repeated_anchor_text_is_not_turned_into_links(service):
    """Kollisionsfall 1 (Wiederholung).

    Frueher wurde **jedes** weitere Vorkommen desselben Worts zum Link, auch
    in Tabellenzellen.
    """
    doc = [FakeLinkPage([_uri_link('https://example.org/a')], textbox='Bericht')]
    service._count_uri_links(doc)

    markdown = 'Siehe Bericht.\n\n| Bericht | 12 |\n| --- | --- |\n\nNoch ein Bericht.'
    assert service._postprocess_markdown(markdown).count('](') == 0


def test_same_anchor_text_with_two_urls_does_not_collide(service, caplog):
    """Kollisionsfall 2 (gleicher Ankertext, zwei URLs).

    Die alte ``{Text: URL}``-Map ueber alle Seiten liess die zweite „hier"-URL
    die erste ueberschreiben; im Dokument bekamen dann *beide* Fundstellen die
    zweite URL. Jetzt gibt es keine Map mehr — der Zaehler sieht beide Links.
    """
    doc = [
        FakeLinkPage([_uri_link('https://example.org/eins')], textbox='hier'),
        FakeLinkPage([_uri_link('https://example.org/zwei')], textbox='hier'),
    ]
    with caplog.at_level('WARNING'):
        assert service._count_uri_links(doc) == 2

    assert any('2 Hyperlink(s)' in r.message for r in caplog.records)


def test_link_counter_ignores_non_uri_links(service):
    doc = [FakeLinkPage([{'kind': 999, 'uri': None, 'from': (0, 0, 1, 1)}])]
    assert service._count_uri_links(doc) == 0


def test_link_counter_is_quiet_without_links(service, caplog):
    with caplog.at_level('WARNING'):
        assert service._count_uri_links([FakeLinkPage([])]) == 0
    assert caplog.records == []
