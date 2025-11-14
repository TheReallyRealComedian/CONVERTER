import os
import asyncio
import tempfile
import logging
import sys
import json
from pathlib import Path
from io import BytesIO
from flask import Flask, render_template, request, flash, redirect, url_for, send_file
from markdown_it import MarkdownIt
from playwright.async_api import async_playwright
from werkzeug.utils import secure_filename
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from unstructured.partition.auto import partition
from asgiref.wsgi import WsgiToAsgi
import fitz
from flask import jsonify
import traceback
from services import DeepgramService, GeminiService, GoogleTTSService


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

SECRET_KEY = os.urandom(24)
DEEPGRAM_API_KEY = os.environ.get('DEEPGRAM_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GOOGLE_CREDENTIALS_PATH = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
STYLE_DIR = Path('/app/static/css/pdf_styles')

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# Initialize services
deepgram_service = DeepgramService(DEEPGRAM_API_KEY) if DEEPGRAM_API_KEY else None
gemini_service = GeminiService(GEMINI_API_KEY) if GEMINI_API_KEY else None
google_tts_service = GoogleTTSService(GOOGLE_CREDENTIALS_PATH) if GOOGLE_CREDENTIALS_PATH else None

def highlight_code(code, lang, _):
    try:
        lexer = get_lexer_by_name(lang, stripall=True)
    except:
        lexer = get_lexer_by_name('text', stripall=True)
    formatter = HtmlFormatter(style='default', cssclass='highlight', noclasses=True)
    return highlight(code, lexer, formatter)

md = MarkdownIt(
    'default',
    {'breaks': True, 'html': True, 'highlight': highlight_code}
)


@app.route('/')
def markdown_converter():
    themes = []
    if STYLE_DIR.exists():
        for f in STYLE_DIR.glob('*.css'):
            themes.append(f.stem)
    return render_template('markdown_converter.html', themes=sorted(themes))

@app.route('/convert-markdown', methods=['POST'])
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
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>{style_content}</style>
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

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.set_content(full_html)
            await page.pdf(
                path=temp_pdf_path,
                format='A4',
                print_background=True,
                margin={'top': '2cm', 'right': '2cm', 'bottom': '2cm', 'left': '2cm'}
            )
            await browser.close()

        return send_file(
            temp_pdf_path,
            as_attachment=True,
            download_name=f"{safe_filename}.pdf",
            mimetype='application/pdf'
        )

    except Exception as e:
        app.logger.error(f"PDF generation failed: {e}")
        flash(f'Error: Could not generate PDF. {e}', 'danger')
        return render_template('markdown_converter.html', markdown_text=markdown_text)
    finally:
        try:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)
        except Exception as e:
            app.logger.error(f"Error cleaning up temp file {temp_pdf_path}: {e}")


@app.route('/mermaid-converter')
def mermaid_converter():
    return render_template('mermaid_converter.html')


@app.route('/document-converter')
def document_converter():
    return render_template('document_converter.html')

@app.route('/transform-document', methods=['POST'])
def transform_document():
    if 'document_file' not in request.files:
        flash('No file part in the request.', 'danger')
        return redirect(url_for('document_converter'))

    file = request.files['document_file']
    if file.filename == '':
        flash('No file selected.', 'danger')
        return redirect(url_for('document_converter'))

    if not file:
        return redirect(url_for('document_converter'))

    original_filename = secure_filename(file.filename)
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(original_filename).suffix) as temp_f:
            file.save(temp_f.name)
            temp_file_path = temp_f.name

        link_map = {}
        try:
            app.logger.info("Starting PyMuPDF link extraction...")
            doc = fitz.open(temp_file_path)
            for page_num, page in enumerate(doc):
                links = page.get_links()
                for link in links:
                    if link.get('kind') == fitz.LINK_URI:
                        clickable_area = link['from']
                        link_text = page.get_textbox(clickable_area).strip().replace('\n', ' ')
                        link_url = link.get('uri')

                        if link_text and link_url:
                            link_map[link_text] = link_url
                            app.logger.info(f"PyMuPDF found link: '{link_text}' -> '{link_url}'")
            doc.close()
            app.logger.info(f"PyMuPDF finished. Found {len(link_map)} unique links.")
        except Exception as e:
            app.logger.error(f"PyMuPDF failed to process links: {e}", exc_info=True)
            link_map = {}

        app.logger.info("Partitioning document with unstructured (strategy='fast')...")
        elements = partition(filename=temp_file_path, strategy="fast")
        full_text = "\n\n".join([el.text for el in elements])

        app.logger.info("Merging unstructured text with PyMuPDF link map...")
        output_markdown = full_text
        if link_map:
            for link_text in sorted(link_map.keys(), key=len, reverse=True):
                link_url = link_map[link_text]
                markdown_link = f"[{link_text}]({link_url})"
                output_markdown = output_markdown.replace(link_text, markdown_link)

        output_path_obj = Path(original_filename)
        output_filename = f"{output_path_obj.stem}.md"

        buffer = BytesIO()
        buffer.write(output_markdown.encode('utf-8'))
        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name=output_filename,
            mimetype='text/markdown'
        )

    except Exception as e:
        app.logger.error(f"Unstructured processing failed: {e}", exc_info=True)
        flash(f'Error processing file: {e}', 'danger')
        return redirect(url_for('document_converter'))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.route('/audio-converter')
def audio_converter():
    return render_template('audio_converter.html', deepgram_api_key_set=bool(DEEPGRAM_API_KEY))

@app.route('/api/get-deepgram-token', methods=['GET'])
def get_deepgram_token():
    if not deepgram_service:
        app.logger.error("Deepgram service not configured")
        return jsonify({"error": "Audio transcription service is not configured."}), 503
    
    return jsonify({"deepgram_token": deepgram_service.get_api_key()})

@app.route('/transcribe-audio-file', methods=['POST'])
def transcribe_audio_file():
    if not deepgram_service:
        return jsonify({"error": "Audio transcription service is not configured."}), 503
    
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file part in the request."}), 400
    
    file = request.files['audio_file']
    language = request.form.get('language', 'en')
    
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400
    
    try:
        buffer_data = file.read()
        transcript = deepgram_service.transcribe_file(buffer_data, language)
        return jsonify({"transcript": transcript})
    
    except Exception as e:
        app.logger.error(f"Deepgram transcription failed: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred during transcription: {str(e)}"}), 500


@app.route('/generate-podcast', methods=['POST'])
def generate_podcast():
    if not google_tts_service:
        return jsonify({"error": "Google Cloud TTS is not configured."}), 503
    
    temp_audio_path = None
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        voice_name = data.get('voice_name', 'en-US-Neural2-C')
        language_code = data.get('language_code', 'en-US')
        speaking_rate = float(data.get('speaking_rate', 1.0))
        pitch = float(data.get('pitch', 0.0))
        
        temp_audio_path = google_tts_service.synthesize_speech(
            text, voice_name, language_code, speaking_rate, pitch
        )
        
        return send_file(
            temp_audio_path,
            as_attachment=True,
            download_name='podcast.mp3',
            mimetype='audio/mpeg'
        )
    
    except Exception as e:
        app.logger.error(f"Google TTS synthesis failed: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred during synthesis: {str(e)}"}), 500
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                app.logger.error(f"Error cleaning up temp file: {e}")


@app.route('/api/get-google-voices', methods=['GET'])
def get_google_voices():
    if not google_tts_service:
        return jsonify({"error": "Google Cloud TTS is not configured."}), 503
    
    try:
        voices = google_tts_service.list_voices()
        return jsonify(voices)
    except Exception as e:
        app.logger.error(f"Failed to retrieve Google TTS voices: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/generate-gemini-podcast', methods=['POST'])
def generate_gemini_podcast():
    if not gemini_service:
        return jsonify({"error": "Gemini API Key is not configured."}), 503
    
    temp_audio_path = None
    try:
        data = request.get_json()
        dialogue = data.get('dialogue', [])
        language = data.get('language', 'en')
        
        temp_audio_path = gemini_service.generate_podcast(dialogue, language)
        
        return send_file(
            temp_audio_path,
            as_attachment=True,
            download_name='gemini_podcast.wav',
            mimetype='audio/wav'
        )
    
    except Exception as e:
        app.logger.error(f"Gemini-TTS failed: {e}", exc_info=True)
        return jsonify({"error": f"Error: {str(e)}"}), 500
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                app.logger.error(f"Cleanup error: {e}")

            
@app.route('/api/get-gemini-voices', methods=['GET'])
def get_gemini_voices():
    voices = {
        "male": [
            {"name": "Kore", "description": "Firm and authoritative"},
            {"name": "Charon", "description": "Informative and clear"},
            {"name": "Fenrir", "description": "Excitable and energetic"},
            {"name": "Orus", "description": "Firm and steady"},
            {"name": "Puck", "description": "Upbeat and cheerful"},
            {"name": "Enceladus", "description": "Breathy and soft"},
            {"name": "Iapetus", "description": "Clear and precise"},
            {"name": "Algenib", "description": "Gravelly and deep"},
            {"name": "Achernar", "description": "Soft and gentle"},
            {"name": "Algieba", "description": "Smooth and polished"},
            {"name": "Gacrux", "description": "Mature and experienced"},
            {"name": "Alnilam", "description": "Firm and direct"},
            {"name": "Rasalgethi", "description": "Informative and educational"},
            {"name": "Sadaltager", "description": "Knowledgeable and wise"},
            {"name": "Zubenelgenubi", "description": "Casual and relaxed"}
        ],
        "female": [
            {"name": "Zephyr", "description": "Bright and lively"},
            {"name": "Leda", "description": "Youthful and fresh"},
            {"name": "Laomedeia", "description": "Upbeat and positive"},
            {"name": "Aoede", "description": "Breezy and light"},
            {"name": "Callirrhoe", "description": "Easy-going and friendly"},
            {"name": "Autonoe", "description": "Bright and clear"},
            {"name": "Erinome", "description": "Clear and articulate"},
            {"name": "Umbriel", "description": "Easy-going and calm"},
            {"name": "Despina", "description": "Smooth and flowing"},
            {"name": "Pulcherrima", "description": "Forward and confident"},
            {"name": "Vindemiatrix", "description": "Gentle and warm"}
        ],
        "neutral": [
            {"name": "Kore", "description": "Firm (can be male or female)"},
            {"name": "Achird", "description": "Friendly and approachable"},
            {"name": "Schedar", "description": "Even and balanced"},
            {"name": "Sadachbia", "description": "Lively and animated"},
            {"name": "Sulafat", "description": "Warm and inviting"}
        ]
    }
    
    return jsonify(voices)


@app.route('/format-dialogue-with-llm', methods=['POST'])
def format_dialogue_with_llm():
    if not gemini_service:
        return jsonify({"error": "Gemini API Key is not configured."}), 503
    
    try:
        data = request.get_json()
        
        # CRITICAL DEBUG: Log what we receive
        raw_text_received = data.get('raw_text', '')
        app.logger.info(f"=== RAW TEXT DEBUG ===")
        app.logger.info(f"Request data keys: {list(data.keys())}")
        app.logger.info(f"raw_text length: {len(raw_text_received)}")
        app.logger.info(f"raw_text first 200 chars: {raw_text_received[:200]}")
        
        if not raw_text_received or not raw_text_received.strip():
            app.logger.error("ERROR: raw_text is empty or whitespace only!")
            return jsonify({"error": "No text provided for formatting"}), 400
        
        result = gemini_service.format_dialogue_with_llm(
            raw_text=raw_text_received.strip(),
            num_speakers=int(data.get('num_speakers', 2)),
            speaker_descriptions=data.get('speaker_descriptions', []),
            language=data.get('language', 'en'),
            tone=data.get('tone', 'professional and informative'),
            script_length=data.get('script_length', 'medium'),
            custom_prompt=data.get('custom_prompt', '').strip() or None
        )
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Dialogue formatting failed: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    



asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)