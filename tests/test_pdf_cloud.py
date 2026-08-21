"""DOC-ENGINE P2 — the page-wise cloud PDF backend behind the page_fn contract.

Mocked at the genai.Client boundary (google-genai is installed on the dev
box; only the client is faked). The PDFs are REAL two-page PyMuPDF documents
with text layers. The mid-flight degradation target is the DOC-LOCAL mineru
engine — faked HERE at the engine interface (page/close/degradations),
because its container mechanics have their own suite (test_pdf_local); what
this file proves is the WIRING: the switch reaches the engine, engine
degradations reach the payload, close() always runs.
"""
from types import SimpleNamespace

import pytest

import fitz

from services import pdf_cloud
from services.pdf_cloud import (
    _strip_fence,
    cost_eur_from_usage,
    price_per_m,
    run_cloud_pdf,
)


def _two_page_pdf(tmp_path, texts=('Seite eins Text.', 'Seite zwei Text.')):
    pdf = fitz.open()
    for text in texts:
        page = pdf.new_page()
        page.insert_text((72, 100), text)
    path = tmp_path / 'doc.pdf'
    pdf.save(str(path))
    pdf.close()
    return str(path)


def _response(text='# Seite', tokens_in=1000, tokens_out=2000,
              finish='STOP', usage=True):
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=tokens_in,
            total_token_count=tokens_in + tokens_out,
            candidates_token_count=tokens_out) if usage else None,
        candidates=[SimpleNamespace(finish_reason=finish)],
    )


@pytest.fixture
def fake_genai(monkeypatch):
    """Replace genai.Client with a fake serving queued responses/exceptions;
    records every generate_content call (model + config)."""
    state = {'queue': [], 'calls': []}

    class FakeModels:
        def generate_content(self, model=None, contents=None, config=None):
            state['calls'].append({'model': model, 'config': config})
            item = state['queue'].pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    class FakeClient:
        def __init__(self, api_key=None):
            state['api_key'] = api_key
            self.models = FakeModels()

    import google.genai
    monkeypatch.setattr(google.genai, 'Client', FakeClient)
    return state


def test_full_cloud_run_books_real_costs(fake_genai, tmp_path):
    path = _two_page_pdf(tmp_path)
    fake_genai['queue'] = [
        _response('# Eins', tokens_in=1000, tokens_out=2000),
        _response('# Zwei', tokens_in=1200, tokens_out=1800),
    ]
    payload = run_cloud_pdf(path, 'key-1', budget_eur=5.0)
    assert payload['markdown'] == '# Eins\n\n# Zwei'
    assert payload['provenance_unit'] == 'page'
    assert payload['provenance'] == ['modell', 'modell']
    assert payload['degradations'] == []
    expected = (cost_eur_from_usage('gemini-3.6-flash', 1000, 2000)
                + cost_eur_from_usage('gemini-3.6-flash', 1200, 1800))
    assert payload['usage']['model_calls'] == 2
    assert payload['usage']['cost_eur'] == pytest.approx(expected, abs=1e-6)
    # The measured call shape: medium resolution + per-call deadline in ms.
    config = fake_genai['calls'][0]['config']
    assert 'MEDIUM' in str(config.media_resolution)
    assert config.http_options.timeout == pdf_cloud.TIMEOUT_GEMINI_SECONDS * 1000


@pytest.fixture
def fake_local_engine(monkeypatch):
    """Replace the DOC-LOCAL engine at pdf_cloud's seam: serves mineru-style
    pages (modell, 0 €), records construction/serving/close. Set
    ``state['fail'] = True`` BEFORE the run to simulate an engine that falls
    back to the text layer and reports its backend_fallback degradation."""
    state = {'created': [], 'fail': False}

    class FakeEngine:
        def __init__(self, source_path, page_count):
            self.source_path = source_path
            self.page_count = page_count
            self.degradations = []
            self.closed = False
            self.pages_served = []
            state['created'].append(self)

        def page(self, index):
            self.pages_served.append(index)
            if state['fail']:
                if not self.degradations:
                    self.degradations.append(
                        {'code': 'backend_fallback', 'message': 'kaputt',
                         'pages': [index + 1]})
                return {'markdown': f'T{index}',
                        'origin': 'deterministisch', 'cost_eur': 0.0}
            return {'markdown': f'M{index}', 'origin': 'modell',
                    'cost_eur': 0.0}

        def close(self):
            self.closed = True

    monkeypatch.setattr(pdf_cloud, 'LocalPdfEngine', FakeEngine)
    return state


def test_midflight_cap_switches_to_local_engine(fake_genai, fake_local_engine,
                                                tmp_path):
    """The P2 core proof at REAL costs: page 1's booked usage exhausts the
    budget, page 2 comes from the DOC-LOCAL engine with one named entry —
    no abort, no second model call. Since DOC-LOCAL the switched page keeps
    ``modell`` provenance (mineru is a VLM); the budget_exceeded entry is
    what names the switch."""
    path = _two_page_pdf(tmp_path, ('Cloud-Seite.', 'Lokaler Rest.'))
    # 100k out tokens ≈ 0.68 € > 0.5 € budget after page 1.
    fake_genai['queue'] = [_response('# Cloud', tokens_out=100_000)]
    payload = run_cloud_pdf(path, 'key-1', budget_eur=0.5)
    assert payload['provenance'] == ['modell', 'modell']
    assert payload['markdown'].endswith('M1')  # engine served page 2
    assert [d['code'] for d in payload['degradations']] == ['budget_exceeded']
    assert payload['degradations'][0]['pages'] == [2]
    assert payload['usage']['model_calls'] == 1
    assert fake_genai['queue'] == []  # nothing left → exactly one call made
    engine = fake_local_engine['created'][0]
    assert engine.pages_served == [1]  # lazy: never touched before the cap
    assert engine.closed is True


def test_engine_degradations_reach_the_payload(fake_genai, fake_local_engine,
                                               tmp_path):
    """If the local engine itself fails after the switch, its
    backend_fallback entry must arrive IN the payload — otherwise the whole
    failure path is invisible to the caller."""
    path = _two_page_pdf(tmp_path, ('Cloud-Seite.', 'Textebene zwei.'))
    fake_genai['queue'] = [_response('# Cloud', tokens_out=100_000)]
    fake_local_engine['fail'] = True
    payload = run_cloud_pdf(path, 'key-1', budget_eur=0.5)
    assert payload['provenance'] == ['modell', 'deterministisch']
    assert [d['code'] for d in payload['degradations']] == [
        'budget_exceeded', 'backend_fallback']
    assert fake_local_engine['created'][0].closed is True


def test_empty_answer_raises(fake_genai, tmp_path):
    path = _two_page_pdf(tmp_path)
    fake_genai['queue'] = [_response('   ')]
    with pytest.raises(RuntimeError, match='leere Modell-Antwort'):
        run_cloud_pdf(path, 'key-1', budget_eur=5.0)


def test_output_cap_truncation_raises(fake_genai, tmp_path):
    path = _two_page_pdf(tmp_path)
    fake_genai['queue'] = [_response('# Halb', finish='FinishReason.MAX_TOKENS')]
    with pytest.raises(RuntimeError, match='abgeschnitten'):
        run_cloud_pdf(path, 'key-1', budget_eur=5.0)


def test_thinking_config_negotiated_once(fake_genai, tmp_path):
    """First call: the first thinking config is rejected (400) and the chain
    escalates; the negotiated config is kept — page 2 needs no retry."""
    path = _two_page_pdf(tmp_path)
    fake_genai['queue'] = [
        ValueError('400 INVALID_ARGUMENT: thinking_level not supported'),
        _response('# Eins'),
        _response('# Zwei'),
    ]
    payload = run_cloud_pdf(path, 'key-1', budget_eur=5.0)
    assert payload['provenance'] == ['modell', 'modell']
    # 3 generate calls total: 1 rejected + 1 ok (page 1), 1 ok (page 2).
    assert len(fake_genai['calls']) == 3


def test_missing_usage_books_fallback_page_price(fake_genai, tmp_path):
    """No usage_metadata → the measured per-page price is booked instead of
    0, so the cap cannot be silently disarmed."""
    path = _two_page_pdf(tmp_path)
    fake_genai['queue'] = [_response('# Eins', usage=False),
                          _response('# Zwei', usage=False)]
    payload = run_cloud_pdf(path, 'key-1', budget_eur=5.0)
    from app_pkg.config import DOC_CONVERT_CLOUD_CENT_PER_PAGE
    assert payload['usage']['cost_eur'] == pytest.approx(
        2 * DOC_CONVERT_CLOUD_CENT_PER_PAGE / 100, abs=1e-9)


def test_wrapper_fence_is_stripped(fake_genai, tmp_path):
    path = _two_page_pdf(tmp_path, ('Nur eine Seite.',))
    fake_genai['queue'] = [_response('```markdown\n# Inhalt\n```')]
    payload = run_cloud_pdf(path, 'key-1', budget_eur=5.0)
    assert payload['markdown'] == '# Inhalt'


def test_pricing_table_and_conservative_default():
    assert price_per_m('gemini-3.6-flash') == {'in': 1.50, 'out': 7.50}
    assert price_per_m('unbekannt-9000') == {'in': 2.00, 'out': 10.00}
    # 1M in + 1M out at 3.6-flash prices, converted with the documented rate.
    eur = cost_eur_from_usage('gemini-3.6-flash', 1_000_000, 1_000_000)
    assert eur == pytest.approx((1.50 + 7.50) / 1.10, abs=1e-9)


# --- the wrapper fence at the source (ported from the retired
# services/pdf_extraction suite, DOC-WEB P2 — the DOC-FIX lesson must not
# fall with the package: only a WHOLE-answer wrap is removed, a page that is
# itself a single code block keeps its fence) -------------------------------

def test_strip_fence_removes_a_full_wrap():
    assert _strip_fence('```markdown\n# A\n\nB\n```') == '# A\n\nB'


def test_strip_fence_removes_unknown_language_tag():
    """The pre-DOC-FIX sweep left the language tag standing as text."""
    assert _strip_fence('```md\n# A\n```') == '# A'
    assert _strip_fence('```html\n# A\n```') == '# A'


def test_strip_fence_leaves_unwrapped_text_alone():
    assert _strip_fence('# A\n\nB') == '# A\n\nB'


def test_strip_fence_does_not_cut_a_trailing_code_block():
    """The old sweep cut the closing ``` here and tore the block."""
    text = '# A\n\n```python\ncode\n```'
    assert _strip_fence(text) == text


def test_strip_fence_keeps_inner_blocks_when_ambiguous():
    """Wrap AND inner block: conservatively nothing is cut at the end."""
    out = _strip_fence('```markdown\n# A\n\n```python\ncode\n```\n```')
    assert '```python\ncode\n```' in out
