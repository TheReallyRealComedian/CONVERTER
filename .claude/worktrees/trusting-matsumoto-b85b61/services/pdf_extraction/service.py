# services/pdf_extraction/service.py
"""Main PDF extraction service with ensemble table detection and multi-page merging."""

import fitz
import logging
import re
import time
from typing import List, Dict, Optional

from google import genai
from google.genai import types

from .detectors import (
    PyMuPDFDetector, PdfplumberDetector, CamelotDetector,
    Img2TableDetector, TextHeuristicDetector, DetectedTable,
)
from .ensemble import build_consensus, ConsensusTable
from .extractors import select_best_extraction
from .multi_page import find_table_spans, merge_table_rows
from .utils import table_to_markdown, parse_markdown_tables

logger = logging.getLogger(__name__)


class PDFExtractionService:
    """PDF-to-Markdown conversion with ensemble table detection."""

    VISION_MODEL = "gemini-2.0-flash"
    RENDER_ZOOM = 3.0
    RENDER_ZOOM_SCANNED = 4.0
    MAX_RETRIES = 3
    BASE_DELAY = 1.0

    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini_client = None
        if gemini_api_key:
            self.gemini_client = genai.Client(api_key=gemini_api_key)
            logger.info("PDFExtractionService: Gemini Vision aktiviert")
        else:
            logger.warning("PDFExtractionService: Kein Gemini API Key - nur lokale Extraktion")

        # Initialize detectors
        self._pymupdf_detector = PyMuPDFDetector()
        self._pdfplumber_detector = PdfplumberDetector()
        self._camelot_detector = CamelotDetector()
        self._img2table_detector = Img2TableDetector()
        self._text_heuristic_detector = TextHeuristicDetector()

    def extract_markdown(self, file_path: str) -> str:
        """Hauptmethode: PDF zu Markdown mit Ensemble-Tabellenextraktion."""
        doc = fitz.open(file_path)

        # Phase 1: Tiered Detection + Seitenklassifizierung
        page_analyses = self._analyze_pages_tiered(doc, file_path)
        link_map = self._extract_links(doc)

        table_page_count = sum(1 for a in page_analyses if a['has_tables'])
        scanned_count = sum(1 for a in page_analyses if a.get('page_type') == 'scanned')
        logger.info(
            f"Analyse: {len(page_analyses)} Seiten, {table_page_count} mit Tabellen, "
            f"{scanned_count} gescannt"
        )

        # Phase 1b: Multi-Page Table Detection
        page_heights = [doc[i].rect.height for i in range(len(doc))]
        texts_before = self._get_texts_before_first_table(doc, page_analyses)
        table_spans = find_table_spans(page_analyses, page_heights, texts_before)
        if table_spans:
            logger.info(f"{len(table_spans)} seitenuebergreifende Tabelle(n) erkannt")

        # Pages that are continuation pages (not the first page of a span)
        continuation_pages = set()
        for span in table_spans:
            for p in span.pages[1:]:
                continuation_pages.add(p)

        # Phase 2: Seitenweise Extraktion
        page_markdowns = []
        for page_num, analysis in enumerate(page_analyses):
            page = doc[page_num]
            page_type = analysis.get('page_type', 'native')

            if page_type == 'scanned':
                md = self._extract_scanned_page(page, page_num)
            elif analysis.get('consensus_tables'):
                md = self._extract_page_with_ensemble(page, page_num, analysis)
            elif analysis['has_tables']:
                # Legacy fallback fuer heuristic-only detections
                md = self._extract_page_with_gemini_fallback(page, page_num, analysis)
            else:
                md = self._extract_text_page(page, page_num)
            page_markdowns.append(md)

        doc.close()

        # Phase 2b: Multi-Page Table Merges anwenden
        page_markdowns = self._apply_multipage_merges(
            page_markdowns, table_spans, page_analyses
        )

        # Phase 3: Zusammensetzen + Post-Processing
        full_markdown = "\n\n---\n\n".join(
            md for md in page_markdowns if md.strip()
        )
        full_markdown = self._embed_links(full_markdown, link_map)
        full_markdown = self._postprocess_markdown(full_markdown)

        return full_markdown

    # -------------------------------------------------------------------------
    # Phase 1: Analysis
    # -------------------------------------------------------------------------

    def _analyze_pages_tiered(self, doc, file_path: str) -> List[Dict]:
        """Tiered detection: fast first, then corroborate, then deep scan if needed."""
        analyses = []

        for page_num, page in enumerate(doc):
            page_type = self._classify_page(page)
            all_detections: List[DetectedTable] = []

            # --- Tier 1: Fast (always) ---
            pymupdf_dets = self._pymupdf_detector.detect(page)
            all_detections.extend(pymupdf_dets)

            heuristic_dets = self._text_heuristic_detector.detect(
                page, file_path, page_num
            )
            all_detections.extend(heuristic_dets)

            tier1_found = len(all_detections) > 0

            # --- Tier 2: Corroborate (if Tier 1 found something) ---
            if tier1_found and page_type != 'scanned':
                pdfplumber_dets = self._pdfplumber_detector.detect(file_path, page_num)
                all_detections.extend(pdfplumber_dets)

                camelot_lattice_dets = self._camelot_detector.detect(
                    file_path, page_num, flavor='lattice'
                )
                all_detections.extend(camelot_lattice_dets)

            # --- Tier 3: Deep scan (if Tier 1 found but Tier 2 didn't confirm) ---
            tier2_names = {'pdfplumber', 'camelot_lattice'}
            tier2_found = any(d.detector_name in tier2_names for d in all_detections)

            if tier1_found and not tier2_found and page_type != 'scanned':
                img2table_dets = self._img2table_detector.detect(
                    file_path, page_num, use_ocr=False
                )
                all_detections.extend(img2table_dets)

                camelot_stream_dets = self._camelot_detector.detect(
                    file_path, page_num, flavor='stream'
                )
                all_detections.extend(camelot_stream_dets)

            # --- Build consensus ---
            total_detectors = len(set(d.detector_name for d in all_detections))
            consensus_tables = build_consensus(
                all_detections,
                total_detectors=max(total_detectors, 2),
                min_votes=2,
            ) if all_detections else []

            has_tables = len(consensus_tables) > 0 or len(heuristic_dets) > 0

            # Build legacy-compatible analysis dict
            pymupdf_tables = [
                ct.best_detection.metadata.get('pymupdf_table')
                for ct in consensus_tables
                if ct.best_detection.metadata.get('pymupdf_table')
            ]

            analyses.append({
                'page_num': page_num,
                'has_tables': has_tables,
                'consensus_tables': consensus_tables,
                'all_detections': all_detections,
                'page_type': page_type,
                # Legacy fields
                'table_count': len(consensus_tables),
                'tables': pymupdf_tables,
                'table_bboxes': [ct.bbox for ct in consensus_tables],
                'detected_by': 'ensemble',
            })

            if consensus_tables:
                detectors_used = set(
                    d.detector_name
                    for ct in consensus_tables
                    for d in ct.detections
                )
                logger.info(
                    f"Seite {page_num + 1}: {len(consensus_tables)} Tabelle(n) "
                    f"(Konsensus: {', '.join(sorted(detectors_used))})"
                )
            if page_type != 'native':
                logger.info(f"Seite {page_num + 1}: Seitentyp = {page_type}")

        return analyses

    def _classify_page(self, page) -> str:
        """Klassifiziert Seite als 'native', 'scanned' oder 'mixed'."""
        text = page.get_text("text").strip()
        images = page.get_images(full=True)
        page_area = page.rect.width * page.rect.height

        total_image_area = 0
        for img in images:
            xref = img[0]
            try:
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    for rect in img_rects:
                        total_image_area += rect.width * rect.height
            except Exception:
                pass

        image_coverage = total_image_area / page_area if page_area > 0 else 0
        text_density = len(text) / (page_area / 1000) if page_area > 0 else 0

        if image_coverage > 0.7 and text_density < 0.5:
            return 'scanned'
        elif image_coverage > 0.3 and text_density < 2.0:
            return 'mixed'
        return 'native'

    def _get_texts_before_first_table(self, doc, page_analyses: List[Dict]) -> List[str]:
        """Get text content above the first table on each page (for continuation keywords)."""
        texts = []
        for page_num, analysis in enumerate(page_analyses):
            page = doc[page_num]
            consensus_tables = analysis.get('consensus_tables', [])
            if consensus_tables:
                first_table_y = consensus_tables[0].bbox[1]
                blocks = page.get_text("dict")["blocks"]
                text_parts = []
                for block in blocks:
                    if block['type'] == 0 and block['bbox'][3] < first_table_y:
                        for line in block['lines']:
                            for span in line['spans']:
                                text_parts.append(span['text'])
                texts.append(' '.join(text_parts))
            else:
                texts.append('')
        return texts

    # -------------------------------------------------------------------------
    # Phase 2: Extraction
    # -------------------------------------------------------------------------

    def _extract_page_with_ensemble(self, page, page_num: int, analysis: Dict) -> str:
        """Extract page using ensemble: best extractor per consensus table."""
        content_parts = []
        table_bboxes = []

        for ct in analysis['consensus_tables']:
            # Try local extractors first
            best_rows = select_best_extraction(ct)

            if best_rows:
                md_table = table_to_markdown(best_rows)
                content_parts.append({'y_pos': ct.bbox[1], 'content': md_table})
                table_bboxes.append(ct.bbox)
                continue

            # Fallback: Gemini Vision for this page
            if self.gemini_client:
                try:
                    gemini_md = self._extract_with_gemini_vision(page, page_num, analysis)
                    # Validate if we have PyMuPDF reference
                    if analysis.get('tables'):
                        validation = self._validate_gemini_output(gemini_md, page, analysis)
                        if validation['score'] >= 0.4:
                            content_parts.append({'y_pos': ct.bbox[1], 'content': gemini_md})
                            table_bboxes.append(ct.bbox)
                            continue
                    else:
                        content_parts.append({'y_pos': ct.bbox[1], 'content': gemini_md})
                        table_bboxes.append(ct.bbox)
                        continue
                except Exception as e:
                    logger.warning(f"Gemini fallback failed page {page_num + 1}: {e}")

        # Add non-table text blocks
        if table_bboxes:
            self._add_non_table_text(page, table_bboxes, content_parts)

        content_parts.sort(key=lambda x: x['y_pos'])
        return '\n\n'.join(part['content'] for part in content_parts)

    def _extract_page_with_gemini_fallback(self, page, page_num: int, analysis: Dict) -> str:
        """Fallback for heuristic-only detections (no consensus tables)."""
        if self.gemini_client:
            try:
                gemini_result = self._extract_with_gemini_vision(page, page_num, analysis)
                if analysis.get('tables'):
                    validation = self._validate_gemini_output(gemini_result, page, analysis)
                    if validation['score'] < 0.4:
                        logger.warning(
                            f"Seite {page_num + 1}: Gemini-Validierung fehlgeschlagen "
                            f"(Score={validation['score']:.2f}). Fallback auf PyMuPDF."
                        )
                        return self._extract_with_pymupdf_tables(page, page_num, analysis)
                    if validation['warnings']:
                        logger.warning(
                            f"Seite {page_num + 1}: Gemini mit Warnungen "
                            f"(Score={validation['score']:.2f}): {validation['warnings']}"
                        )
                return gemini_result
            except Exception as e:
                logger.warning(f"Gemini fehlgeschlagen Seite {page_num + 1}: {e}")

        return self._extract_with_pymupdf_tables(page, page_num, analysis)

    def _extract_text_page(self, page, page_num: int) -> str:
        """Extrahiert reinen Text von Seiten ohne Tabellen."""
        text = page.get_text("text")
        lines = text.split('\n')
        cleaned = []
        prev_blank = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if not prev_blank:
                    cleaned.append('')
                    prev_blank = True
            else:
                cleaned.append(stripped)
                prev_blank = False
        return '\n'.join(cleaned)

    def _extract_scanned_page(self, page, page_num: int) -> str:
        """Extrahiert gescannte Seiten via Gemini Vision mit hoher Aufloesung."""
        if self.gemini_client:
            try:
                return self._extract_with_gemini_vision(
                    page, page_num, analysis=None, zoom=self.RENDER_ZOOM_SCANNED
                )
            except Exception as e:
                logger.warning(f"Gemini fuer gescannte Seite {page_num + 1} fehlgeschlagen: {e}")

        text = page.get_text("text").strip()
        if text:
            return text
        logger.warning(f"Seite {page_num + 1}: Gescannte Seite ohne Gemini - kein Text")
        return ''

    def _add_non_table_text(self, page, table_bboxes: List, content_parts: List[Dict]):
        """Add text blocks outside of table regions."""
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block['type'] == 0:
                block_rect = fitz.Rect(block['bbox'])
                in_table = any(
                    fitz.Rect(tbbox).intersects(block_rect)
                    for tbbox in table_bboxes
                )
                if not in_table:
                    text = ''
                    for line in block['lines']:
                        for span in line['spans']:
                            text += span['text']
                        text += '\n'
                    text = text.strip()
                    if text:
                        content_parts.append({
                            'y_pos': block['bbox'][1],
                            'content': text,
                        })

    # -------------------------------------------------------------------------
    # Gemini Vision
    # -------------------------------------------------------------------------

    def _build_gemini_prompt(self, analysis: Optional[Dict] = None) -> str:
        """Baut den Gemini-Prompt mit optionalen Strukturhinweisen."""
        prompt = """Convert this PDF page to well-formatted Markdown. Follow these rules precisely:

1. TABLES: Convert all tables to proper Markdown table syntax with | delimiters and --- header separators.
   - For merged cells that span multiple columns, repeat the content in each spanned column.
   - For merged cells that span multiple rows, place the content in the first row and leave subsequent rows empty for that cell.
   - Preserve ALL data values exactly as shown.
   - Every row must have the same number of columns.

2. TEXT: Convert all non-table text to standard Markdown.
   - Use # for headings based on visual hierarchy.
   - Preserve bullet points and numbered lists.
   - Preserve bold and italic emphasis where visible.

3. STRUCTURE: Maintain the reading order of the page (top to bottom, left to right).

4. DO NOT add any commentary, explanation, or wrapper text. Output ONLY the Markdown content.

5. DO NOT use code blocks (```) around the Markdown tables. Output raw Markdown."""

        if analysis and analysis.get('tables'):
            hints = "\n\nSTRUCTURAL HINTS from automated analysis (use to verify your extraction):\n"
            for i, table in enumerate(analysis['tables']):
                data = table.extract()
                if data:
                    rows = len(data)
                    cols = len(data[0]) if data[0] else 0
                    header_cells = [str(c).strip() for c in data[0] if c]
                    header_preview = ' | '.join(header_cells[:6])
                    if len(header_cells) > 6:
                        header_preview += ' | ...'
                    hints += (
                        f"- Table {i+1}: {rows} rows x {cols} columns. "
                        f"Header: {header_preview}\n"
                    )
            hints += "Column count and header names should match these hints.\n"
            prompt += hints

        return prompt

    def _extract_with_gemini_vision(self, page, page_num: int,
                                     analysis: Optional[Dict] = None,
                                     zoom: Optional[float] = None) -> str:
        """Rendert Seite als Bild und sendet an Gemini Vision."""
        render_zoom = zoom or self.RENDER_ZOOM
        mat = fitz.Matrix(render_zoom, render_zoom)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

        prompt = self._build_gemini_prompt(analysis)

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.gemini_client.models.generate_content(
                    model=self.VISION_MODEL,
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=16384,
                    ),
                )
                break
            except Exception as e:
                last_error = e
                if '429' in str(e) or 'rate' in str(e).lower() or 'resource' in str(e).lower():
                    delay = self.BASE_DELAY * (2 ** attempt)
                    logger.warning(f"Rate-Limit Seite {page_num + 1}, Retry in {delay}s")
                    time.sleep(delay)
                else:
                    raise
        else:
            raise last_error  # type: ignore[misc]

        if not response.text:
            raise ValueError(f"Gemini leere Antwort fuer Seite {page_num + 1}")

        result = response.text.strip()

        # Code-Fences entfernen
        if result.startswith("```markdown"):
            result = result[len("```markdown"):].strip()
        if result.startswith("```"):
            result = result[3:].strip()
        if result.endswith("```"):
            result = result[:-3].strip()

        logger.info(f"Seite {page_num + 1}: Gemini Vision ({len(result)} Zeichen)")
        return result

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _validate_gemini_output(self, gemini_markdown: str, page, analysis: Dict) -> Dict:
        """Validiert Gemini-Output gegen PyMuPDF-Strukturdaten."""
        warnings = []
        score = 1.0

        gemini_tables = []
        current_table: List[str] = []
        for line in gemini_markdown.split('\n'):
            stripped = line.strip()
            if stripped.startswith('|') and stripped.endswith('|'):
                current_table.append(stripped)
            else:
                if current_table:
                    gemini_tables.append(current_table)
                    current_table = []
        if current_table:
            gemini_tables.append(current_table)

        pymupdf_table_count = analysis.get('table_count', 0)

        if len(gemini_tables) < pymupdf_table_count:
            warnings.append(
                f"Gemini: {len(gemini_tables)} Tabellen, PyMuPDF: {pymupdf_table_count}"
            )
            score -= 0.3

        for i, pymupdf_table in enumerate(analysis.get('tables', [])):
            pymupdf_data = pymupdf_table.extract()
            if not pymupdf_data:
                continue
            pymupdf_cols = len(pymupdf_data[0])

            if i < len(gemini_tables):
                data_rows = [r for r in gemini_tables[i]
                             if not re.match(r'^\|\s*[-:]+\s*(\|\s*[-:]+\s*)*\|$', r)]
                if data_rows:
                    gemini_cols = len(data_rows[0].split('|')) - 2
                    if abs(gemini_cols - pymupdf_cols) > 1:
                        warnings.append(
                            f"Tabelle {i+1}: Gemini {gemini_cols} Spalten, PyMuPDF {pymupdf_cols}"
                        )
                        score -= 0.2

        overlap = self._content_overlap_score(gemini_markdown, analysis.get('tables', []))
        if overlap < 0.5:
            warnings.append(f"Niedriger Content-Overlap: {overlap:.0%}")
            score -= (0.5 - overlap)

        return {
            'valid': score >= 0.4,
            'warnings': warnings,
            'score': max(0.0, score),
            'content_overlap': overlap,
        }

    def _content_overlap_score(self, gemini_text: str, pymupdf_tables: list) -> float:
        """Berechnet Anteil der PyMuPDF-Zellwerte die im Gemini-Text vorkommen."""
        all_cells = []
        for table in pymupdf_tables:
            data = table.extract()
            if not data:
                continue
            for row in data:
                for cell in row:
                    cell_text = str(cell).strip() if cell else ''
                    if len(cell_text) >= 2:
                        all_cells.append(cell_text)

        if not all_cells:
            return 1.0

        found = sum(1 for cell in all_cells if cell in gemini_text)
        return found / len(all_cells)

    # -------------------------------------------------------------------------
    # PyMuPDF Fallback
    # -------------------------------------------------------------------------

    def _extract_with_pymupdf_tables(self, page, page_num: int, analysis: Dict) -> str:
        """Fallback: PyMuPDF Tabellenextraktion + Text."""
        table_bboxes = analysis.get('table_bboxes', [])
        tables_data = analysis.get('tables', [])

        content_parts = []

        for i, table in enumerate(tables_data):
            if i < len(table_bboxes):
                bbox = table_bboxes[i]
            else:
                continue
            data = table.extract()
            if data:
                rows = [
                    [str(cell).strip() if cell else '' for cell in row]
                    for row in data
                ]
                md_table = table_to_markdown(rows)
                if md_table:
                    content_parts.append({'y_pos': bbox[1], 'content': md_table})

        self._add_non_table_text(page, table_bboxes, content_parts)
        content_parts.sort(key=lambda x: x['y_pos'])

        return '\n\n'.join(part['content'] for part in content_parts)

    # -------------------------------------------------------------------------
    # Multi-Page Merge
    # -------------------------------------------------------------------------

    def _apply_multipage_merges(self, page_markdowns: List[str],
                                 table_spans, page_analyses: List[Dict]) -> List[str]:
        """Apply multi-page table merges to page markdowns."""
        if not table_spans:
            return page_markdowns

        result = list(page_markdowns)

        for span in table_spans:
            if len(span.pages) < 2:
                continue

            # Extract tables from each page's markdown
            page_table_rows = []
            for page_num in span.pages:
                tables = parse_markdown_tables(result[page_num])
                if tables:
                    page_table_rows.append(tables[-1])  # Last table on the page
                else:
                    page_table_rows.append([])

            if not any(page_table_rows):
                continue

            # Merge rows
            merged_rows = merge_table_rows(page_table_rows)
            merged_md = table_to_markdown(merged_rows)

            if merged_md:
                # Replace last table on first page with merged table
                first_page = span.pages[0]
                first_md = result[first_page]
                tables_in_first = parse_markdown_tables(first_md)

                if tables_in_first:
                    # Find and replace the last table
                    last_table_md = table_to_markdown(tables_in_first[-1])
                    if last_table_md in first_md:
                        result[first_page] = first_md.replace(last_table_md, merged_md)

                # Remove table from continuation pages
                for page_num in span.pages[1:]:
                    page_md = result[page_num]
                    tables_in_page = parse_markdown_tables(page_md)
                    if tables_in_page:
                        first_table_md = table_to_markdown(tables_in_page[0])
                        if first_table_md in page_md:
                            result[page_num] = page_md.replace(first_table_md, '').strip()

            logger.info(
                f"Multi-page merge: Seiten {[p+1 for p in span.pages]} zusammengefuehrt"
            )

        return result

    # -------------------------------------------------------------------------
    # Links & Post-Processing
    # -------------------------------------------------------------------------

    def _extract_links(self, doc) -> Dict[str, str]:
        """Extrahiert Hyperlinks aus dem PDF."""
        link_map = {}
        for page_num, page in enumerate(doc):
            links = page.get_links()
            for link in links:
                if link.get('kind') == fitz.LINK_URI:
                    clickable_area = link['from']
                    link_text = page.get_textbox(clickable_area).strip().replace('\n', ' ')
                    link_url = link.get('uri')
                    if link_text and link_url:
                        link_map[link_text] = link_url
        if link_map:
            logger.info(f"{len(link_map)} Links extrahiert")
        return link_map

    def _postprocess_markdown(self, markdown: str) -> str:
        """Bereinigt und normalisiert den finalen Markdown-Output."""
        markdown = re.sub(r'```\w*\n', '', markdown)
        markdown = markdown.replace('```', '')
        markdown = re.sub(r'\n{4,}', '\n\n\n', markdown)
        markdown = re.sub(r'^#+\s*Page\s+\d+\s*$', '', markdown, flags=re.MULTILINE)

        lines = markdown.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('|') and stripped.endswith('|'):
                cells = stripped.split('|')
                cells = [c.strip() for c in cells]
                line = '| ' + ' | '.join(cells[1:-1]) + ' |'
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _embed_links(self, markdown: str, link_map: Dict[str, str]) -> str:
        """Bettet Hyperlinks als Markdown-Links ein."""
        if not link_map:
            return markdown
        for link_text in sorted(link_map.keys(), key=len, reverse=True):
            link_url = link_map[link_text]
            markdown_link = f"[{link_text}]({link_url})"
            markdown = markdown.replace(link_text, markdown_link)
        return markdown
