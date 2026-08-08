"""Paged conversion pipeline with a hard per-job budget cap (DOC-API P2).

Pure module. This is the mechanic behind locked decisions 2 + 3: a job that
hits its cost cap **degrades to the local path and keeps going** — it never
aborts — and every page produced after the switch carries the changed
provenance, plus one degradation entry saying why. The decision doc is
explicit that the answer shape must carry mixed provenance from day one
("sonst ist sie beim ersten Deckel-Fall falsch"), and the bake-off's harshest
verdict was exactly the unmarked mid-document path switch.

There is deliberately **no second path yet** (the local engine is the
follow-up sprint), so today no production caller runs this loop — the legacy
blackbox engine cannot be driven page-wise from outside. This module exists
NOW because it *defines the backend contract the engine sprint implements*
and proves the cap mechanic against placeholder backends in the test suite —
built before it is expensive to have it wrong. The result it returns is the
same ``build_result_payload`` shape the worker task writes, so a pipeline
result flows through the existing reconcile unchanged (test-proven).

Backend contract (what the engine sprint plugs in):

    page_fn(page_index_0based) -> {
        'markdown': str,      # the page's Markdown
        'origin':   str,      # its provenance: deterministisch | ocr | modell
        'cost_eur': float,    # what converting this page actually cost
    }

Two callables per run — ``cloud_page`` and ``local_page`` — because the cap's
whole point is switching between them mid-document.
"""
from services.document_conversions import (
    DEGRADATION_BUDGET_EXCEEDED,
    UNIT_PAGE,
    build_result_payload,
    degradation,
)

# Page separator when joining per-page Markdown into the document payload.
PAGE_JOIN = '\n\n'


def run_paged_conversion(page_count, cloud_page, local_page, budget_eur):
    """Convert ``page_count`` pages, hard-capped at ``budget_eur``.

    The cap check runs **before** each page: once the accumulated spend has
    reached or exceeded the budget, this page and every later one run through
    ``local_page`` instead — so the overshoot is at most the one page that was
    already in flight when the budget ran out (a running page is never
    aborted; "hart" bounds the spend, it does not tear results).

    Every page records the origin its backend declares, so a mid-document
    switch is visible per page; exactly ONE ``budget_exceeded`` entry names
    the cap, the spend at the switch and the affected pages (1-based).

    ``usage`` counts ``model_calls`` as the number of cloud-path pages and
    sums ``cost_eur`` over both paths (a future local backend may report
    nonzero costs; today's placeholder and the planned mineru path report 0).

    Returns the shared ``build_result_payload`` dict — directly writable as
    ``result_<id>.json`` and readable by the existing reconcile.
    """
    page_markdowns = []
    provenance = []
    degradations = []
    spent = 0.0
    model_calls = 0
    switched_at = None  # 0-based index of the first degraded page

    for index in range(page_count):
        if switched_at is None and budget_eur is not None and spent >= budget_eur:
            switched_at = index
            degradations.append(degradation(
                DEGRADATION_BUDGET_EXCEEDED,
                f'Kostendeckel {budget_eur:.2f} € erreicht '
                f'(Stand {spent:.2f} € nach Seite {index}). '
                f'Ab Seite {index + 1} lokal konvertiert.',
                pages=list(range(index + 1, page_count + 1)),
            ))
        if switched_at is None:
            result = cloud_page(index)
            model_calls += 1
        else:
            result = local_page(index)
        page_markdowns.append(result['markdown'])
        provenance.append(result['origin'])
        spent += result['cost_eur']

    return build_result_payload(
        PAGE_JOIN.join(page_markdowns),
        provenance_unit=UNIT_PAGE,
        provenance=provenance,
        degradations=degradations,
        usage={'model_calls': model_calls, 'cost_eur': round(spent, 6)},
    )
