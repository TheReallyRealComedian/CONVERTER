"""DOC-ENGINE P2 — the page-wise cloud PDF backend behind the page_fn contract.

Mocked at the genai.Client boundary (google-genai is installed on the dev
box; only the client is faked). The PDFs are REAL two-page PyMuPDF documents
with text layers, so the local page function (text layer) and the page
splitting run for real — what the mid-flight cap serves after the switch is
the actual page text, not a placeholder.
"""
from types import SimpleNamespace

import pytest

import fitz

from services import pdf_cloud
from services.pdf_cloud import (
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


def test_midflight_cap_switches_to_text_layer(fake_genai, tmp_path):
    """The P2 core proof at REAL costs: page 1's booked usage exhausts the
    budget, page 2 comes from the deterministic text layer with flipped
    provenance and one named entry — no abort, no second model call."""
    path = _two_page_pdf(tmp_path, ('Cloud-Seite.', 'Lokaler Fallback.'))
    # 100k out tokens ≈ 0.68 € > 0.5 € budget after page 1.
    fake_genai['queue'] = [_response('# Cloud', tokens_out=100_000)]
    payload = run_cloud_pdf(path, 'key-1', budget_eur=0.5)
    assert payload['provenance'] == ['modell', 'deterministisch']
    assert 'Lokaler Fallback.' in payload['markdown']  # real page text
    assert [d['code'] for d in payload['degradations']] == ['budget_exceeded']
    assert payload['degradations'][0]['pages'] == [2]
    assert payload['usage']['model_calls'] == 1
    assert fake_genai['queue'] == []  # nothing left → exactly one call made


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
