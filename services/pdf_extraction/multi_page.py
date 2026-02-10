# services/pdf_extraction/multi_page.py
"""Multi-page table detection and merging."""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .ensemble import ConsensusTable
from .utils import columns_match, rows_similar

logger = logging.getLogger(__name__)

CONTINUATION_KEYWORDS = [
    'continued', 'cont.', "cont'd", 'fortsetzung', 'fortgesetzt',
    '(continued)', '(cont.)', '(Fortsetzung)',
]


@dataclass
class TableSpan:
    """A table that spans multiple pages."""
    pages: List[int]
    tables: List[ConsensusTable]


def detect_continuation_signals(
    page_n_table: ConsensusTable,
    page_n1_table: ConsensusTable,
    page_height: float,
    text_before_table: str,
) -> Dict[str, bool]:
    """Check signals that table on page N+1 continues table on page N."""
    signals = {}

    # Signal 1: Position — table N ends near bottom, N+1 starts near top
    signals['position'] = (
        page_n_table.bbox[3] > page_height * 0.80 and
        page_n1_table.bbox[1] < page_height * 0.20
    )

    # Signal 2: Same column count
    if page_n_table.column_count and page_n1_table.column_count:
        signals['same_columns'] = (
            page_n_table.column_count == page_n1_table.column_count
        )
    else:
        signals['same_columns'] = False

    # Signal 3: Column x-coordinate alignment
    n_cols = _get_column_positions(page_n_table)
    n1_cols = _get_column_positions(page_n1_table)
    if n_cols and n1_cols:
        signals['alignment'] = columns_match(n_cols, n1_cols, tolerance=8.0)
    else:
        signals['alignment'] = False

    # Signal 4: Continuation keywords
    text_lower = text_before_table.lower().strip()
    signals['keyword'] = any(kw in text_lower for kw in CONTINUATION_KEYWORDS)

    return signals


def is_continuation(signals: Dict[str, bool]) -> bool:
    """Decide if table continues based on signals."""
    if signals.get('keyword') and signals.get('same_columns'):
        return True
    if signals.get('position') and (
        signals.get('same_columns') or signals.get('alignment')
    ):
        return True
    return False


def find_table_spans(
    page_analyses: List[Dict],
    page_heights: List[float],
    texts_before_first_table: List[str],
) -> List[TableSpan]:
    """Identify groups of pages where a table continues across page boundaries."""
    spans: List[TableSpan] = []
    current_span: Optional[TableSpan] = None

    for page_num in range(len(page_analyses)):
        tables = page_analyses[page_num].get('consensus_tables', [])
        if not tables:
            if current_span and len(current_span.pages) > 1:
                spans.append(current_span)
            current_span = None
            continue

        # Check continuation from previous page
        if current_span and page_num > 0:
            prev_tables = page_analyses[page_num - 1].get('consensus_tables', [])
            if prev_tables:
                signals = detect_continuation_signals(
                    prev_tables[-1], tables[0],
                    page_heights[page_num - 1],
                    texts_before_first_table[page_num] if page_num < len(texts_before_first_table) else '',
                )
                if is_continuation(signals):
                    current_span.pages.append(page_num)
                    current_span.tables.append(tables[0])
                    logger.info(
                        f"Multi-page table: Seite {page_num + 1} setzt fort "
                        f"(Signale: {signals})"
                    )
                    continue
                else:
                    if len(current_span.pages) > 1:
                        spans.append(current_span)
                    current_span = None

        # Start new potential span
        last_table = tables[-1]
        if last_table.bbox[3] > page_heights[page_num] * 0.80:
            current_span = TableSpan(pages=[page_num], tables=[last_table])
        else:
            if current_span and len(current_span.pages) > 1:
                spans.append(current_span)
            current_span = None

    if current_span and len(current_span.pages) > 1:
        spans.append(current_span)

    return spans


def merge_table_rows(page_extractions: List[List[List[str]]]) -> List[List[str]]:
    """Merge table rows from multiple pages. Removes repeated headers."""
    if not page_extractions or not page_extractions[0]:
        return []

    header = page_extractions[0][0]
    merged = list(page_extractions[0])

    for page_rows in page_extractions[1:]:
        if not page_rows:
            continue
        # Skip repeated header
        if rows_similar(header, page_rows[0]):
            data_rows = page_rows[1:]
        else:
            data_rows = page_rows
        merged.extend(data_rows)

    return merged


def _get_column_positions(table: ConsensusTable) -> List[float]:
    """Extract column x-positions from detection metadata."""
    best = table.best_detection
    pymupdf_table = best.metadata.get('pymupdf_table')
    if pymupdf_table and hasattr(pymupdf_table, 'cells') and pymupdf_table.cells:
        return sorted(set(cell[0] for cell in pymupdf_table.cells))

    # Fallback: divide bbox evenly
    if table.column_count and table.column_count > 0:
        x0, _, x1, _ = table.bbox
        step = (x1 - x0) / table.column_count
        return [x0 + i * step for i in range(table.column_count + 1)]

    return []
