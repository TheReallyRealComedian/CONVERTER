"""Markdown→PDF converter routes."""
import base64
import functools
import os
import re
import tempfile
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path

from flask import flash, jsonify, redirect, render_template, request, send_file, url_for
from flask_login import login_required
from werkzeug.utils import secure_filename

from .markdown_render import render_markdown_to_html


STYLE_DIR = Path('/app/static/css/pdf_styles')

# MATH-RENDER: vendored KaTeX, relativ zum Paket aufgelöst (läuft unter /app im
# Docker *und* lokal beim Smoke).
_KATEX_DIR = Path(__file__).resolve().parent.parent / 'static' / 'vendor' / 'katex'
_KATEX_FONT_URL_RE = re.compile(r'url\(fonts/([A-Za-z0-9_.-]+)\.woff2\) format\("woff2"\)')
# woff/ttf-Fallbacks nach dem woff2-Eintrag (deren fonts/-Pfade lösen unter dem
# about:blank-Kontext von set_content nicht auf → entfernen, woff2 reicht).
_KATEX_FONT_FALLBACK_RE = re.compile(
    r',url\(fonts/[A-Za-z0-9_.-]+\.woff\) format\("woff"\),'
    r'url\(fonts/[A-Za-z0-9_.-]+\.ttf\) format\("truetype"\)'
)


@functools.lru_cache(maxsize=1)
def _katex_pdf_assets():
    """Self-contained KaTeX-Bundle für den Playwright-PDF-Flow.

    ``page.set_content`` lädt das HTML mit about:blank-Base — externe/relative
    Asset-URLs lösen nicht auf. Darum inlinen wir: die CSS mit den woff2-Fonts als
    data-URIs (Fallback-woff/ttf-Einträge raus) + die JS als Inline-Script. So
    rendert KaTeX *mit* Fonts, unabhängig von Origin/Base. Einmal gebaut (lru_cache).
    """
    css = (_KATEX_DIR / 'katex.min.css').read_text()

    def _inline_font(match):
        name = match.group(1)
        data = base64.b64encode((_KATEX_DIR / 'fonts' / f'{name}.woff2').read_bytes()).decode()
        return f'url(data:font/woff2;base64,{data}) format("woff2")'

    css = _KATEX_FONT_FALLBACK_RE.sub('', css)
    css = _KATEX_FONT_URL_RE.sub(_inline_font, css)
    js = (_KATEX_DIR / 'katex.min.js').read_text()
    return css, js


# Rendert alle server-geschützten Mathe-Spans (.math-inline/.math-display, rohes
# LaTeX als textContent) mit dem inline geladenen KaTeX. Synchron → keine Race
# vor page.pdf(); throwOnError:false lässt eine kaputte Formel als Quelltext stehen.
_KATEX_RENDER_JS = '''() => {
    if (typeof katex === 'undefined') return;
    document.querySelectorAll('.math-inline, .math-display').forEach(el => {
        try {
            katex.render(el.textContent, el, {
                displayMode: el.classList.contains('math-display'),
                throwOnError: false,
            });
        } catch (e) {}
    });
}'''

# Single source of truth for what /convert-markdown accepts as a file upload.
# The template reads this via the route context for both the file-input
# ``accept`` attribute and ``window.PageData.acceptedExtensions`` (frontend
# pre-submit guard).
ACCEPTED_EXTENSIONS = ('md', 'markdown')


class _TableColumnCounter(HTMLParser):
    """Counts columns in each <table> to detect wide tables."""

    def __init__(self):
        super().__init__()
        self._in_table = False
        self._in_first_row = False
        self._col_count = 0
        self._tables = []  # list of (start_offset, col_count)
        self._table_start = 0

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self._in_table = True
            self._in_first_row = True
            self._col_count = 0
            self._table_start = self.getpos()
        elif tag in ('th', 'td') and self._in_first_row:
            self._col_count += 1
        elif tag == 'tr' and self._in_table and self._col_count > 0:
            # Second <tr> means first row is done
            self._in_first_row = False

    def handle_endtag(self, tag):
        if tag == 'table' and self._in_table:
            self._tables.append((self._table_start, self._col_count))
            self._in_table = False
            self._in_first_row = False


def _wrap_wide_tables(html: str, column_threshold: int = 6) -> str:
    """Wrap tables with >= column_threshold columns in a landscape div."""
    parser = _TableColumnCounter()
    parser.feed(html)

    if not parser._tables:
        return html

    # Process in reverse so string offsets remain valid
    lines = html.split('\n')
    for (line_no, _col_offset), col_count in reversed(parser._tables):
        if col_count < column_threshold:
            continue
        # Find the <table> tag in the source and its closing </table>
        # line_no is 1-based from HTMLParser.getpos()
        table_line_idx = line_no - 1

        # Find the line with </table> starting from table_line_idx
        end_idx = table_line_idx
        for i in range(table_line_idx, len(lines)):
            if '</table>' in lines[i]:
                end_idx = i
                break

        lines[end_idx] = lines[end_idx].replace('</table>', '</table>\n</div>', 1)
        lines[table_line_idx] = '<div class="landscape-table">\n' + lines[table_line_idx]

    return '\n'.join(lines)


def register(app):
    # Late import: tests patch ``app.async_playwright`` on the top-level
    # app.py module, so look it up at call time rather than capturing the
    # import here.
    import app as _app_module

    @app.route('/')
    @login_required
    def markdown_converter():
        themes = []
        if STYLE_DIR.exists():
            for f in STYLE_DIR.glob('*.css'):
                themes.append(f.stem)
        return render_template(
            'markdown_converter.html',
            themes=sorted(themes),
            accepted_extensions=ACCEPTED_EXTENSIONS,
            accepted_extensions_accept=','.join('.' + ext for ext in ACCEPTED_EXTENSIONS),
        )

    @app.route('/convert-markdown', methods=['POST'])
    @login_required
    async def convert_markdown():
        markdown_text = request.form.get('markdown_text')
        markdown_file = request.files.get('markdown_file')

        if markdown_file and markdown_file.filename:
            ext = os.path.splitext(secure_filename(markdown_file.filename))[1].lstrip('.').lower()
            if ext not in ACCEPTED_EXTENSIONS:
                return jsonify({
                    'error': 'Dateiformat nicht unterstützt. Erlaubt: .md, .markdown.'
                }), 400
            markdown_text = markdown_file.read().decode('utf-8')
        elif not markdown_text or not markdown_text.strip():
            flash('Kein Markdown-Inhalt. Bitte Text einfügen oder eine Datei hochladen.', 'danger')
            return redirect(url_for('markdown_converter'))

        output_filename = request.form.get('output_filename') or ''
        safe_filename = secure_filename(output_filename)
        if not safe_filename:
            flash('Ungültiger Dateiname. Bitte einen anderen Namen verwenden.', 'danger')
            return redirect(url_for('markdown_converter'))

        orientation = request.form.get('orientation', 'portrait')
        is_landscape = orientation == 'landscape'

        style_theme = request.form.get('style_theme', 'default')
        style_path = STYLE_DIR / f"{secure_filename(style_theme)}.css"
        style_content = ''
        if style_theme != 'none':
            try:
                with open(style_path, 'r') as f:
                    style_content = f.read()
            except FileNotFoundError:
                flash(f'Theme „{style_theme}" nicht gefunden. Standard-Layout wird verwendet.', 'warning')
                style_content = ''

        html_content = render_markdown_to_html(markdown_text)
        html_content = _wrap_wide_tables(html_content, column_threshold=6)
        katex_css, katex_js = _katex_pdf_assets()
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>{style_content}</style>
            {'<style>@page { size: A4 landscape; }</style>' if is_landscape else ''}
            <style>{katex_css}</style>
            <script>{katex_js}</script>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        temp_pdf_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                temp_pdf_path = temp_pdf.name

            async with _app_module.async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.set_content(full_html, wait_until='networkidle')
                # MATH-RENDER: Mathe-Spans rendern, *bevor* auf Fonts gewartet wird,
                # damit die von KaTeX referenzierten Fonts mit in document.fonts.ready
                # einfließen. KaTeX rendert synchron → keine Race vor page.pdf().
                await page.evaluate(_KATEX_RENDER_JS)
                # Wait until all web fonts (Google Fonts etc.) are loaded before rendering
                await page.evaluate('document.fonts.ready')
                # Detect tables that overflow the page width (few but very wide columns)
                await page.evaluate('''() => {
                    document.querySelectorAll('table').forEach(table => {
                        if (table.scrollWidth > table.parentElement.clientWidth
                            && !table.closest('.landscape-table')) {
                            const wrapper = document.createElement('div');
                            wrapper.className = 'landscape-table';
                            table.parentNode.insertBefore(wrapper, table);
                            wrapper.appendChild(table);
                        }
                    });
                }''')
                await page.pdf(
                    path=temp_pdf_path,
                    format='A4',
                    landscape=is_landscape,
                    print_background=True,
                    margin={'top': '2cm', 'right': '2cm', 'bottom': '2cm', 'left': '2cm'}
                )
                await browser.close()

            # Read into buffer so temp file can safely be deleted
            pdf_buffer = BytesIO()
            with open(temp_pdf_path, 'rb') as f:
                pdf_buffer.write(f.read())
            pdf_buffer.seek(0)

            return send_file(
                pdf_buffer,
                as_attachment=True,
                download_name=f"{safe_filename}.pdf",
                mimetype='application/pdf'
            )

        except Exception as e:
            app.logger.error(f"PDF generation failed: {e}", exc_info=True)
            flash(
                'PDF-Erstellung fehlgeschlagen. Bitte erneut versuchen oder eine andere Vorlage wählen.',
                'danger',
            )
            return redirect(url_for('markdown_converter'))
        finally:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.unlink(temp_pdf_path)
                except Exception as e:
                    app.logger.error(f"Error cleaning up temp file {temp_pdf_path}: {e}", exc_info=True)
