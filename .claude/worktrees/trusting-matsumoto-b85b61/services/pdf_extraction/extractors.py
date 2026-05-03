# services/pdf_extraction/extractors.py
"""Table extraction backends that convert detected tables to row data."""

import logging
from typing import List, Optional, Dict

from .detectors import DetectedTable
from .ensemble import ConsensusTable, score_extraction

logger = logging.getLogger(__name__)


def extract_pymupdf(detection: DetectedTable) -> Optional[List[List[str]]]:
    """Extract table content using PyMuPDF table object."""
    table_obj = detection.metadata.get('pymupdf_table')
    if table_obj is None:
        return None
    data = table_obj.extract()
    if not data:
        return None
    return [
        [str(cell).strip() if cell else '' for cell in row]
        for row in data
    ]


def extract_pdfplumber(detection: DetectedTable) -> Optional[List[List[str]]]:
    """Extract using cached pdfplumber rows from detection phase."""
    rows = detection.metadata.get('pdfplumber_rows')
    if not rows:
        return None
    return [
        [str(cell).strip() if cell else '' for cell in row]
        for row in rows
    ]


def extract_camelot(detection: DetectedTable) -> Optional[List[List[str]]]:
    """Extract using camelot table's DataFrame."""
    table_obj = detection.metadata.get('camelot_table')
    if table_obj is None:
        return None
    df = table_obj.df
    if df.empty:
        return None
    rows = []
    for _, row in df.iterrows():
        rows.append([str(cell).strip() for cell in row])
    return rows


def extract_img2table(detection: DetectedTable) -> Optional[List[List[str]]]:
    """Extract using img2table's DataFrame."""
    df = detection.metadata.get('df')
    if df is None or df.empty:
        return None
    rows = []
    # Column headers as first row
    rows.append([str(c).strip() for c in df.columns])
    for _, row in df.iterrows():
        rows.append([str(cell).strip() for cell in row])
    return rows


EXTRACTOR_MAP = {
    'pymupdf': extract_pymupdf,
    'pdfplumber': extract_pdfplumber,
    'camelot_lattice': extract_camelot,
    'camelot_stream': extract_camelot,
    'img2table': extract_img2table,
}


def select_best_extraction(consensus: ConsensusTable) -> Optional[List[List[str]]]:
    """Try all extractors for a consensus table, pick best by score."""
    best_score = -1.0
    best_result = None

    for det in consensus.detections:
        extractor = EXTRACTOR_MAP.get(det.detector_name)
        if extractor is None:
            continue
        rows = extractor(det)
        if rows is None:
            continue
        s = score_extraction(rows, consensus)
        logger.debug(f"  Extractor '{det.detector_name}': score={s:.2f}, rows={len(rows)}")
        if s > best_score:
            best_score = s
            best_result = rows

    return best_result
