# services/pdf_extraction/utils.py
"""Shared utility functions for PDF table extraction ensemble."""

import re
from typing import Tuple, List, Dict, Optional

# Type alias for bounding boxes: (x0, y0, x1, y1)
BBox = Tuple[float, float, float, float]


def bbox_iou(a: BBox, b: BBox) -> float:
    """Calculate Intersection over Union of two bounding boxes."""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])

    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def bbox_overlap_ratio(a: BBox, b: BBox) -> float:
    """Fraction of box A that overlaps with box B."""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])

    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])

    return intersection / area_a if area_a > 0 else 0.0


def parse_markdown_tables(markdown: str) -> List[List[List[str]]]:
    """Parse markdown text and extract all tables as [table][row][cell]."""
    tables: List[List[List[str]]] = []
    current_table: List[List[str]] = []
    for line in markdown.split('\n'):
        stripped = line.strip()
        if stripped.startswith('|') and stripped.endswith('|'):
            # Skip separator lines (---)
            if not re.match(r'^\|\s*[-:]+\s*(\|\s*[-:]+\s*)*\|$', stripped):
                cells = [c.strip() for c in stripped.split('|')[1:-1]]
                current_table.append(cells)
        else:
            if current_table:
                tables.append(current_table)
                current_table = []
    if current_table:
        tables.append(current_table)
    return tables


def table_to_markdown(rows: List[List[str]]) -> str:
    """Convert a list of rows (each a list of cell strings) to markdown table."""
    if not rows:
        return ''

    # Clean cells
    cleaned = []
    for row in rows:
        cleaned.append([
            str(cell).strip().replace('\n', ' ').replace('\r', '').replace('|', '\\|')
            if cell else ''
            for cell in row
        ])

    # Normalize column count
    max_cols = max(len(row) for row in cleaned)
    for row in cleaned:
        while len(row) < max_cols:
            row.append('')

    # Calculate column widths
    col_widths = [3] * max_cols
    for row in cleaned:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Build markdown
    lines = []
    for idx, row in enumerate(cleaned):
        padded = [cell.ljust(col_widths[i]) for i, cell in enumerate(row)]
        lines.append('| ' + ' | '.join(padded) + ' |')
        if idx == 0:
            lines.append('| ' + ' | '.join('-' * w for w in col_widths) + ' |')
    return '\n'.join(lines)


def columns_match(cols_a: List[float], cols_b: List[float],
                  tolerance: float = 5.0) -> bool:
    """Check if two lists of column x-coordinates are spatially aligned."""
    if len(cols_a) != len(cols_b):
        return False
    return all(abs(a - b) < tolerance for a, b in zip(cols_a, cols_b))


def rows_similar(row_a: List[str], row_b: List[str],
                 threshold: float = 0.7) -> bool:
    """Check if two rows are similar (for repeated header detection)."""
    if len(row_a) != len(row_b):
        return False
    if not row_a:
        return True
    matches = sum(
        1 for a, b in zip(row_a, row_b)
        if a.strip().lower() == b.strip().lower()
    )
    return matches / len(row_a) >= threshold
