# services/pdf_extraction_service.py
import fitz  # PyMuPDF
import logging
import time
from typing import List, Dict, Optional
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class PDFExtractionService:
    """PDF-zu-Markdown Konvertierung mit Tabellenerkennung via PyMuPDF + Gemini Vision."""

    VISION_MODEL = "gemini-2.0-flash"
    MIN_TABLE_CELLS = 4
    RATE_LIMIT_DELAY = 0.5

    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini_client = None
        if gemini_api_key:
            self.gemini_client = genai.Client(api_key=gemini_api_key)
            logger.info("PDFExtractionService: Gemini Vision aktiviert")
        else:
            logger.warning("PDFExtractionService: Kein Gemini API Key - nur PyMuPDF Fallback")

    def extract_markdown(self, file_path: str) -> str:
        """Hauptmethode: PDF zu Markdown mit Tabellenstruktur."""
        doc = fitz.open(file_path)

        # Phase 1: Alle Seiten analysieren
        page_analyses = self._analyze_pages(doc)
        link_map = self._extract_links(doc)

        table_page_count = sum(1 for a in page_analyses if a['has_tables'])
        logger.info(f"Analyse: {len(page_analyses)} Seiten, {table_page_count} mit Tabellen")

        # Phase 2: Seitenweise Extraktion
        page_markdowns = []
        gemini_call_count = 0
        for page_num, analysis in enumerate(page_analyses):
            page = doc[page_num]
            if analysis['has_tables']:
                md = self._extract_page_with_tables(page, page_num, analysis)
                if self.gemini_client:
                    gemini_call_count += 1
                    if gemini_call_count > 1:
                        time.sleep(self.RATE_LIMIT_DELAY)
            else:
                md = self._extract_text_page(page, page_num)
            page_markdowns.append(md)

        doc.close()

        # Phase 3: Zusammensetzen
        full_markdown = "\n\n---\n\n".join(
            md for md in page_markdowns if md.strip()
        )
        full_markdown = self._embed_links(full_markdown, link_map)

        return full_markdown

    def _analyze_pages(self, doc) -> List[Dict]:
        """Erkennt Tabellen auf jeder Seite via PyMuPDF find_tables()."""
        analyses = []
        for page_num, page in enumerate(doc):
            tables = page.find_tables()
            real_tables = [t for t in tables.tables if len(t.cells) >= self.MIN_TABLE_CELLS]
            analyses.append({
                'page_num': page_num,
                'has_tables': len(real_tables) > 0,
                'table_count': len(real_tables),
                'tables': real_tables,
                'table_bboxes': [t.bbox for t in real_tables],
            })
            if real_tables:
                logger.info(f"Seite {page_num + 1}: {len(real_tables)} Tabelle(n) erkannt")
        return analyses

    def _extract_links(self, doc) -> Dict[str, str]:
        """Extrahiert Hyperlinks aus dem PDF (bestehende Logik aus app.py)."""
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

    def _extract_text_page(self, page, page_num: int) -> str:
        """Extrahiert reinen Text von Seiten ohne Tabellen."""
        text = page.get_text("text")
        # Mehrfache Leerzeilen zusammenfassen
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

    def _extract_page_with_tables(self, page, page_num: int, analysis: Dict) -> str:
        """Extrahiert Seite mit Tabellen - Gemini Vision bevorzugt, PyMuPDF als Fallback."""
        if self.gemini_client:
            try:
                return self._extract_with_gemini_vision(page, page_num)
            except Exception as e:
                logger.warning(f"Gemini Vision fehlgeschlagen fuer Seite {page_num + 1}: {e}. Fallback auf PyMuPDF.")

        return self._extract_with_pymupdf_tables(page, page_num, analysis)

    def _extract_with_gemini_vision(self, page, page_num: int) -> str:
        """Rendert Seite als Bild und sendet an Gemini Vision fuer Markdown-Extraktion."""
        # Seite mit 2x Zoom rendern (~144 DPI)
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

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

        response = self.gemini_client.models.generate_content(
            model=self.VISION_MODEL,
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_text(prompt),
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=8192,
            ),
        )

        if not response.text:
            raise ValueError(f"Gemini lieferte leere Antwort fuer Seite {page_num + 1}")

        result = response.text.strip()

        # Code-Fences entfernen, falls Gemini sie hinzufuegt
        if result.startswith("```markdown"):
            result = result[len("```markdown"):].strip()
        if result.startswith("```"):
            result = result[3:].strip()
        if result.endswith("```"):
            result = result[:-3].strip()

        logger.info(f"Seite {page_num + 1}: Gemini Vision extrahiert ({len(result)} Zeichen)")
        return result

    def _extract_with_pymupdf_tables(self, page, page_num: int, analysis: Dict) -> str:
        """Fallback: PyMuPDF Tabellenextraktion + Text zwischen Tabellen."""
        table_bboxes = analysis['table_bboxes']
        tables_data = analysis['tables']

        # Inhaltsregionen sammeln (Tabellen + Text)
        content_parts = []

        # Tabellen als Markdown
        for i, table in enumerate(tables_data):
            bbox = table_bboxes[i]
            md_table = self._pymupdf_table_to_markdown(table)
            if md_table:
                content_parts.append({
                    'y_pos': bbox[1],
                    'content': md_table,
                })

        # Textbloecke ausserhalb der Tabellen-Bounding-Boxes
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block['type'] == 0:  # Textblock
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

        # Nach vertikaler Position sortieren
        content_parts.sort(key=lambda x: x['y_pos'])

        return '\n\n'.join(part['content'] for part in content_parts)

    def _pymupdf_table_to_markdown(self, table) -> str:
        """Konvertiert ein PyMuPDF Table-Objekt zu Markdown-Tabellensyntax."""
        data = table.extract()
        if not data:
            return ''

        lines = []
        for row_idx, row in enumerate(data):
            cells = [str(cell).strip() if cell else '' for cell in row]
            line = '| ' + ' | '.join(cells) + ' |'
            lines.append(line)

            # Header-Separator nach erster Zeile
            if row_idx == 0:
                separator = '| ' + ' | '.join(['---'] * len(cells)) + ' |'
                lines.append(separator)

        return '\n'.join(lines)

    def _embed_links(self, markdown: str, link_map: Dict[str, str]) -> str:
        """Bettet Hyperlinks als Markdown-Links ein."""
        if not link_map:
            return markdown
        for link_text in sorted(link_map.keys(), key=len, reverse=True):
            link_url = link_map[link_text]
            markdown_link = f"[{link_text}]({link_url})"
            markdown = markdown.replace(link_text, markdown_link)
        return markdown
