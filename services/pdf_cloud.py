"""Cloud-PDF backend (DOC-ENGINE P2): gemini native PDF input, page-wise.

Implements the ``page_fn`` contract of ``services/document_pipeline`` with
REAL calls: one 1-page PDF per request, ``media_resolution=MEDIUM``, the
bake-off prompt verbatim, costs from ``usage_metadata`` — never an estimate.

**Why page-wise** (sprint 2.1, measured on the gold sample instead of
decided): whole-doc vs page-wise on ``01.gold`` scored word-f1 0.9809 vs
0.9747 (Δ 0.006), CER 0.0405 vs 0.0571, table cells IDENTICAL (0.9356) and
the missing-token lists identical too — the losses are notation effects
(Unicode sub/superscripts vs LaTeX), not context effects. The delta is ~3×
the run-to-run variance (0.002, measured 07 vs its calibration twin) and far
below the medium-vs-low resolution gap (0.015). Per the sprint rule ("bei
kleinem Unterschied seitenweise") page-wise wins: it keeps the mid-document
budget degradation mechanic alive. Side effect, measured: page-wise spent
half the output tokens of the whole-doc call on 01 (3.8k vs 7.9k — less
thinking per call), 1.5 ct/Seite ≈ the bake-off's 1.48 ct calculation.

**Named limit of that measurement**: nothing in ``01.gold`` runs ACROSS the
page break (page 2 opens with TABLE I) — the 0.006 measures token noise, not
continuity, and the one mechanism whole-doc could win on (a structure
spanning pages) is absent from the measuring corpus. The decision stands
(the delta bound holds for non-spanning content, and cross-page merging is
DOC-LOCAL's multi-page-merge territory anyway); a corpus exemplar WITH a
page-spanning table is backlogged so the gap is measurable.

The prompt, temperature, max_output_tokens, thinking-config chain and the
429-retry come VERBATIM from the measured harness adapter
(``corpus/bakeoff/harness/adapters.py``) — the measurement only holds for
these calls. The model is env-overridable via ``PDF_VISION_MODEL`` — the
SAME switch as ``services/pdf_extraction`` (DOC-FIX): one vision model for
both PDF paths, flipped together (a hardcoded name cost two months of
silent failure once).

``local_page`` is the per-page deterministic counterpart the pipeline
switches to when the budget cap hits mid-document: the raw PyMuPDF text
layer (provably no model; empty on scans). DOC-LOCAL replaces it with the
real local engine — this module only needs it to keep the switch honest.

Pure module: no Flask, no SDK singleton; genai/fitz imports live inside
``run_cloud_pdf`` (worker-side, in-task import convention).
"""
import logging
import os
import time

from app_pkg.config import DOC_CONVERT_CLOUD_CENT_PER_PAGE, TIMEOUT_GEMINI_SECONDS
from services.document_conversions import (
    PROVENANCE_DETERMINISTIC,
    PROVENANCE_MODEL,
)
from services.document_pipeline import run_paged_conversion

logger = logging.getLogger(__name__)

# One vision model for both PDF paths (DOC-FIX switch, shared on purpose).
DEFAULT_CLOUD_PDF_MODEL = os.environ.get('PDF_VISION_MODEL') or 'gemini-3.6-flash'

# USD per 1M tokens. Verified 2026-08-07 against ai.google.dev pricing (paid
# tier); the output price EXPLICITLY includes thinking tokens, and our
# tokens_out (total - prompt) counts exactly that. Unknown models fall back
# to a deliberately conservative default so a model switch without a price
# entry OVERSTATES costs (the cap trips early) instead of understating them.
MODEL_PRICES_USD_PER_M = {
    'gemini-3.6-flash': {'in': 1.50, 'out': 7.50},
}
DEFAULT_PRICE_USD_PER_M = {'in': 2.00, 'out': 10.00}

# Fixed conversion ASSUMPTION (bake-off ledger convention, 2026-08): 1 EUR =
# 1.10 USD. Costs are overstated — and the cap trips early, the harmless
# direction — only while the real rate stays above 1.10; if EUR/USD falls
# below that, booked costs UNDERSTATE reality. A knob is not warranted for a
# ~10 % band on cent amounts; revisit the constant if the rate regime shifts.
EUR_PER_USD = 1 / 1.10

MAX_OUTPUT_TOKENS = 32768  # harness cap; a single page never came close

# The measured prompt — verbatim from the bake-off adapter. Changing a word
# here voids the 2.1 measurement.
GEMINI_PROMPT = """Convert this PDF document to faithful Markdown. Rules:

1. FIDELITY: Reproduce the exact wording of the source. Do NOT correct
   typos, spacing quirks, numbering errors or inconsistencies — they are
   part of the document. Do NOT invent or fill in anything.
2. TABLES: Use GFM pipe tables. If a table has merged cells, use a raw HTML
   <table> with colspan/rowspan instead. Every row must keep its column count.
3. TEXT: Headings via # by visual hierarchy; preserve bold/italic; footnotes
   as [^n] with definitions at the end.
4. FORMS: Keep blank fields blank. Render empty checkboxes as ☐, dotted or
   underscored fill-in lines as _____ (five underscores). Never fill them.
5. READING ORDER: Follow the visual reading order (columns top-to-bottom,
   left column before right column).
6. PAGES: Separate consecutive pages with a line containing only ---.
7. OUTPUT: Only the Markdown content. No commentary, no code fences."""


def price_per_m(model_name):
    """USD prices for a model, substring-matched; unknown → conservative."""
    for key, price in MODEL_PRICES_USD_PER_M.items():
        if key in (model_name or ''):
            return price
    return DEFAULT_PRICE_USD_PER_M


def cost_eur_from_usage(model_name, tokens_in, tokens_out):
    price = price_per_m(model_name)
    usd = tokens_in / 1e6 * price['in'] + tokens_out / 1e6 * price['out']
    return usd * EUR_PER_USD


def _strip_fence(text):
    """Remove ONE whole-answer code fence (the DOC-FIX rule: a page that IS
    a single code block keeps its fence — only a wrapper fence falls)."""
    import re
    opening = re.match(r'```[^\n]*\n', text)
    if not opening:
        return text
    body = text[opening.end():]
    if body.count('```') == 1 and body.rstrip().endswith('```'):
        body = body.rstrip()[:-3]
    return body.strip()


def _thinking_configs(types_mod):
    """Fallback chain to cap thinking spend (output tokens cost 7.50/M
    INCLUDING thinking; transcription needs none). Which config the model
    accepts is negotiated on the first call, then kept."""
    chain = []
    try:
        chain.append(('thinking_level=low',
                      types_mod.ThinkingConfig(thinking_level='low')))
    except Exception:
        pass
    try:
        chain.append(('thinking_budget=0',
                      types_mod.ThinkingConfig(thinking_budget=0)))
    except Exception:
        pass
    chain.append(('default', None))
    return chain


def run_cloud_pdf(source_path, api_key, budget_eur, model_name=None):
    """Convert a PDF page-wise through gemini, hard-capped at ``budget_eur``.

    Opens the document once, feeds ``run_paged_conversion`` a real
    ``cloud_page`` (1-page PDF per call, costs booked from usage_metadata)
    and a real ``local_page`` (PyMuPDF text layer, deterministic), returns
    the shared result payload. Raises on unreadable PDFs, on empty model
    answers after retries and on output-cap truncation — a re-submit is this
    API's retry path; silently empty or torn pages are the failure mode the
    bake-off found worst.
    """
    import fitz
    from google import genai
    from google.genai import types

    model = model_name or DEFAULT_CLOUD_PDF_MODEL
    client = genai.Client(api_key=api_key)
    resolution = types.MediaResolution.MEDIA_RESOLUTION_MEDIUM
    thinking_chain = _thinking_configs(types)
    negotiated = {'config': None, 'done': False}

    doc = fitz.open(source_path)
    try:
        page_count = doc.page_count

        def _page_pdf_bytes(index):
            sub = fitz.open()
            sub.insert_pdf(doc, from_page=index, to_page=index)
            data = sub.tobytes()
            sub.close()
            return data

        def _generate(pdf_bytes):
            """One call with the negotiated thinking config; negotiates on
            the first call by walking the chain on config rejections."""
            chain = ([negotiated['config']] if negotiated['done']
                     else list(thinking_chain))
            last_error = None
            for tc_name, tc in chain:
                config_kwargs = dict(
                    temperature=0.1,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    media_resolution=resolution,
                    # Per-call deadline (NARR-TIMEOUT/DOC-FIX doctrine):
                    # HttpOptions.timeout is milliseconds and caps the
                    # response, not just the connect.
                    http_options=types.HttpOptions(
                        timeout=TIMEOUT_GEMINI_SECONDS * 1000),
                )
                if tc is not None:
                    config_kwargs['thinking_config'] = tc
                try:
                    for attempt in range(3):
                        try:
                            response = client.models.generate_content(
                                model=model,
                                contents=[
                                    types.Part.from_bytes(
                                        data=pdf_bytes,
                                        mime_type='application/pdf'),
                                    GEMINI_PROMPT,
                                ],
                                config=types.GenerateContentConfig(
                                    **config_kwargs),
                            )
                            break
                        except Exception as e:
                            msg = str(e).lower()
                            if attempt < 2 and ('429' in msg or 'rate' in msg
                                                or 'resource' in msg):
                                time.sleep(2.0 * (2 ** attempt))
                                continue
                            raise
                except Exception as e:
                    msg = str(e).lower()
                    if (not negotiated['done'] and tc is not None
                            and ('thinking' in msg or 'invalid' in msg
                                 or '400' in msg)):
                        last_error = e
                        continue  # config rejected → next chain entry
                    raise
                negotiated['config'] = (tc_name, tc)
                negotiated['done'] = True
                return response
            raise RuntimeError(
                f'Kein Gemini-Call erfolgreich (Thinking-Kette erschöpft): '
                f'{last_error}')

        def cloud_page(index):
            response = _generate(_page_pdf_bytes(index))
            finish = ''
            try:
                finish = str(response.candidates[0].finish_reason)
            except Exception:
                pass
            if 'MAX_TOKENS' in finish:
                raise RuntimeError(
                    f'Seite {index + 1}: Modell-Antwort am Output-Deckel '
                    f'abgeschnitten ({MAX_OUTPUT_TOKENS} Tokens).')
            text = (response.text or '').strip()
            if not text:
                raise RuntimeError(
                    f'Seite {index + 1}: leere Modell-Antwort.')
            usage = getattr(response, 'usage_metadata', None)
            tokens_in = getattr(usage, 'prompt_token_count', 0) or 0
            total = getattr(usage, 'total_token_count', 0) or 0
            tokens_out = max(
                total - tokens_in,
                getattr(usage, 'candidates_token_count', 0) or 0)
            if tokens_in or tokens_out:
                cost = cost_eur_from_usage(model, tokens_in, tokens_out)
            else:
                # No usage metadata → book the measured per-page price
                # instead of 0: an unbooked page would silently disarm the
                # cap (conservative direction, like the price default).
                cost = DOC_CONVERT_CLOUD_CENT_PER_PAGE / 100
                logger.warning(
                    'cloud_page %d: no usage_metadata, booking fallback '
                    'page price', index + 1)
            return {'markdown': _strip_fence(text),
                    'origin': PROVENANCE_MODEL,
                    'cost_eur': cost}

        def local_page(index):
            return {'markdown': doc[index].get_text('text').strip(),
                    'origin': PROVENANCE_DETERMINISTIC,
                    'cost_eur': 0.0}

        return run_paged_conversion(page_count, cloud_page, local_page,
                                    budget_eur)
    finally:
        doc.close()
