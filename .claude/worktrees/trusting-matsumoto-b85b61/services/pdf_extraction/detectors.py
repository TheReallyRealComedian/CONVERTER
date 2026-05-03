# services/pdf_extraction/detectors.py
"""Table detection backends for ensemble detection."""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .utils import BBox

logger = logging.getLogger(__name__)


@dataclass
class DetectedTable:
    """A table region detected by a detector."""
    bbox: BBox
    confidence: float
    detector_name: str
    column_count: Optional[int] = None
    row_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PyMuPDFDetector:
    """Table detection using PyMuPDF find_tables()."""

    NAME = "pymupdf"
    MIN_CELLS = 4

    def detect(self, page) -> List[DetectedTable]:
        tables = page.find_tables()
        results = []
        for t in tables.tables:
            if len(t.cells) < self.MIN_CELLS:
                continue
            data = t.extract()
            results.append(DetectedTable(
                bbox=t.bbox,
                confidence=0.8,
                detector_name=self.NAME,
                column_count=len(data[0]) if data else None,
                row_count=len(data) if data else None,
                metadata={'pymupdf_table': t},
            ))
        return results


class PdfplumberDetector:
    """Table detection using pdfplumber's line-intersection algorithm."""

    NAME = "pdfplumber"

    def detect(self, file_path: str, page_num: int) -> List[DetectedTable]:
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                if page_num >= len(pdf.pages):
                    return []
                page = pdf.pages[page_num]
                tables = page.find_tables()
                results = []
                for t in tables:
                    rows = t.extract()
                    results.append(DetectedTable(
                        bbox=t.bbox,
                        confidence=0.75,
                        detector_name=self.NAME,
                        column_count=len(rows[0]) if rows and rows[0] else None,
                        row_count=len(rows) if rows else None,
                        metadata={'pdfplumber_rows': rows},
                    ))
                return results
        except Exception as e:
            logger.debug(f"pdfplumber detection failed page {page_num + 1}: {e}")
            return []


class CamelotDetector:
    """Table detection using camelot-py (lattice + stream modes)."""

    NAME_LATTICE = "camelot_lattice"
    NAME_STREAM = "camelot_stream"

    def detect(self, file_path: str, page_num: int,
               flavor: str = "lattice") -> List[DetectedTable]:
        try:
            import camelot
            tables = camelot.read_pdf(
                file_path,
                pages=str(page_num + 1),
                flavor=flavor,
            )
            results = []
            name = self.NAME_LATTICE if flavor == "lattice" else self.NAME_STREAM
            for t in tables:
                report = t.parsing_report
                accuracy = report.get('accuracy', 0) / 100.0
                bbox = t._bbox if hasattr(t, '_bbox') else (0, 0, 0, 0)
                df = t.df
                results.append(DetectedTable(
                    bbox=bbox,
                    confidence=accuracy,
                    detector_name=name,
                    column_count=len(df.columns) if not df.empty else None,
                    row_count=len(df) if not df.empty else None,
                    metadata={
                        'camelot_table': t,
                        'accuracy': accuracy,
                    },
                ))
            return results
        except Exception as e:
            logger.debug(f"camelot ({flavor}) detection failed page {page_num + 1}: {e}")
            return []


class Img2TableDetector:
    """Table detection using img2table (OpenCV-based, no neural network)."""

    NAME = "img2table"

    def detect(self, file_path: str, page_num: int,
               use_ocr: bool = False) -> List[DetectedTable]:
        try:
            from img2table.document import PDF as Img2TablePDF

            ocr = None
            if use_ocr:
                try:
                    from img2table.ocr import TesseractOCR
                    ocr = TesseractOCR(n_threads=1, lang="eng+deu")
                except Exception:
                    pass

            pdf_doc = Img2TablePDF(src=file_path, pages=[page_num])
            extracted = pdf_doc.extract_tables(
                ocr=ocr,
                implicit_rows=True,
                implicit_columns=True,
                borderless_tables=True,
                min_confidence=50,
            )

            results = []
            page_tables = extracted.get(page_num, [])
            for t in page_tables:
                # img2table uses pixel coords at 200 DPI, convert to PDF points (72 DPI)
                scale = 72.0 / 200.0
                bbox = (
                    t.bbox.x1 * scale,
                    t.bbox.y1 * scale,
                    t.bbox.x2 * scale,
                    t.bbox.y2 * scale,
                )
                df = t.df if hasattr(t, 'df') else None
                results.append(DetectedTable(
                    bbox=bbox,
                    confidence=0.7,
                    detector_name=self.NAME,
                    column_count=len(df.columns) if df is not None and not df.empty else None,
                    row_count=len(df) if df is not None and not df.empty else None,
                    metadata={'img2table_obj': t, 'df': df},
                ))
            return results
        except Exception as e:
            logger.debug(f"img2table detection failed page {page_num + 1}: {e}")
            return []


class TextHeuristicDetector:
    """Text spacing + pdfminer columnar layout detection."""

    NAME = "text_heuristic"

    def detect(self, page, file_path: str, page_num: int) -> List[DetectedTable]:
        if not self._text_heuristic(page):
            return []
        if not self._pdfminer_columnar(file_path, page_num):
            return []

        # Detected but no precise bbox — use full page content area
        page_rect = page.rect
        return [DetectedTable(
            bbox=(page_rect.x0 + 20, page_rect.y0 + 20,
                  page_rect.x1 - 20, page_rect.y1 - 20),
            confidence=0.5,
            detector_name=self.NAME,
            metadata={'detection_method': 'heuristic+pdfminer'},
        )]

    def _text_heuristic(self, page) -> bool:
        text = page.get_text("text")
        lines = [l for l in text.split('\n') if l.strip()]
        if len(lines) < 3:
            return False
        multi_field_lines = sum(
            1 for line in lines
            if len(re.split(r'\s{2,}', line.strip())) >= 3
        )
        return multi_field_lines >= 3

    def _pdfminer_columnar(self, file_path: str, page_num: int) -> bool:
        try:
            from pdfminer.high_level import extract_pages
            from pdfminer.layout import LTTextBoxHorizontal, LAParams

            laparams = LAParams(
                line_margin=0.3, word_margin=0.2,
                char_margin=2.0, boxes_flow=None,
            )
            for i, page_layout in enumerate(extract_pages(file_path, laparams=laparams)):
                if i != page_num:
                    continue
                text_boxes = [
                    element.bbox
                    for element in page_layout
                    if isinstance(element, LTTextBoxHorizontal)
                ]
                return self._has_columnar_layout(text_boxes)
            return False
        except Exception as e:
            logger.debug(f"pdfminer analysis failed: {e}")
            return False

    def _has_columnar_layout(self, bboxes: list, y_tolerance: float = 3.0) -> bool:
        y_groups: Dict[float, int] = defaultdict(int)
        for bbox in bboxes:
            y_rounded = round(bbox[1] / y_tolerance) * y_tolerance
            y_groups[y_rounded] += 1
        columnar_rows = sum(1 for count in y_groups.values() if count >= 3)
        return columnar_rows >= 3
