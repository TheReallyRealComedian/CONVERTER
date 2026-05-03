"""Markdown→PDF converter routes."""
import os
import tempfile
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path

import nh3
from flask import flash, redirect, render_template, request, send_file, url_for
from flask_login import login_required
from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from werkzeug.utils import secure_filename


STYLE_DIR = Path('/app/static/css/pdf_styles')


def highlight_code(code, lang, _):
    try:
        lexer = get_lexer_by_name(lang, stripall=True)
    except Exception:
        lexer = get_lexer_by_name('text', stripall=True)
    formatter = HtmlFormatter(style='default', cssclass='highlight', noclasses=True)
    return highlight(code, lexer, formatter)


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


md = MarkdownIt(
    'default',
    {'breaks': True, 'html': True, 'highlight': highlight_code}
)


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
        return render_template('markdown_converter.html', themes=sorted(themes))

    @app.route('/convert-markdown', methods=['POST'])
    @login_required
    async def convert_markdown():
        markdown_text = request.form.get('markdown_text')
        markdown_file = request.files.get('markdown_file')

        if markdown_file and markdown_file.filename:
            markdown_text = markdown_file.read().decode('utf-8')
        elif not markdown_text or not markdown_text.strip():
            flash('Error: No Markdown content provided. Please paste text or upload a file.', 'danger')
            return redirect(url_for('markdown_converter'))

        output_filename = request.form.get('output_filename')
        safe_filename = secure_filename(output_filename)
        if not safe_filename:
            flash('Error: Invalid filename provided.', 'danger')
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
                flash(f'Warning: Style "{style_theme}" not found. Using no style.', 'warning')
                style_content = ''

        html_content = md.render(markdown_text)
        html_content = nh3.clean(
            html_content,
            tags={
                'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                'p', 'br', 'hr', 'blockquote', 'pre', 'code',
                'ul', 'ol', 'li', 'dl', 'dt', 'dd',
                'table', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td', 'caption', 'colgroup', 'col',
                'a', 'img', 'figure', 'figcaption',
                'strong', 'em', 'b', 'i', 'u', 's', 'del', 'ins', 'mark',
                'sub', 'sup', 'small', 'abbr', 'cite', 'q', 'kbd', 'var', 'samp',
                'details', 'summary',
                'div', 'span', 'section', 'article', 'aside', 'header', 'footer', 'nav', 'main',
            },
            attributes={
                '*': {'class', 'id', 'style'},
                'a': {'href', 'title', 'target'},
                'img': {'src', 'alt', 'title', 'width', 'height'},
                'th': {'colspan', 'rowspan', 'scope'},
                'td': {'colspan', 'rowspan'},
                'col': {'span'},
                'colgroup': {'span'},
            },
        )
        html_content = _wrap_wide_tables(html_content, column_threshold=6)
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>{style_content}</style>
            {'<style>@page { size: A4 landscape; }</style>' if is_landscape else ''}
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
            app.logger.error(f"PDF generation failed: {e}")
            flash('Error: Could not generate PDF. Please try again.', 'danger')
            return render_template('markdown_converter.html', markdown_text=markdown_text)
        finally:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.unlink(temp_pdf_path)
                except Exception as e:
                    app.logger.error(f"Error cleaning up temp file {temp_pdf_path}: {e}")
