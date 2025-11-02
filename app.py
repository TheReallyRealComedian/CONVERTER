import os
import asyncio
import tempfile
import logging
import sys
from google.cloud import texttospeech
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
import fitz  # PyMuPDF
from deepgram import DeepgramClient, PrerecordedOptions
from flask import jsonify
import traceback
# ===== UPDATED IMPORT FOR NEW LIBRARY =====
from google import genai
from google.genai import types
# ==========================================


# ==========================================================
# ===== LOGGING CONFIGURATION =====
# ==========================================================
# Configure logging to output to stdout, which is standard for containers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
# ==========================================================


# --- Configuration ---
SECRET_KEY = os.urandom(24)
DEEPGRAM_API_KEY = os.environ.get('DEEPGRAM_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GOOGLE_CREDENTIALS_PATH = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
STYLE_DIR = Path('/app/static/css/pdf_styles')

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# --- Markdown Parser Initialization ---
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

# ==========================================================
# ===== ROUTES FOR MARKDOWN TO PDF CONVERTER =====
# ==========================================================
@app.route('/')
def markdown_converter():
    """Renders the main page for the markdown converter."""
    themes = []
    if STYLE_DIR.exists():
        for f in STYLE_DIR.glob('*.css'):
            themes.append(f.stem)
    return render_template('markdown_converter.html', themes=sorted(themes))

@app.route('/convert-markdown', methods=['POST'])
async def convert_markdown():
    """Handles the form submission and PDF conversion."""
    markdown_text = request.form.get('markdown_text')
    markdown_file = request.files.get('markdown_file')

    # --- Input Validation ---
    if markdown_file and markdown_file.filename:
        markdown_text = markdown_file.read().decode('utf-8')
    elif not markdown_text or not markdown_text.strip():
        flash('Error: No Markdown content provided. Please paste text or upload a file.', 'danger')
        return redirect(url_for('markdown_converter'))

    # --- Filename Sanitization ---
    output_filename = request.form.get('output_filename')
    safe_filename = secure_filename(output_filename)
    if not safe_filename:
        flash('Error: Invalid filename provided.', 'danger')
        return redirect(url_for('markdown_converter'))

    # --- Style Selection ---
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

    # --- Conversion Logic ---
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
    temp_pdf_path = None # Initialize to avoid UnboundLocalError in finally
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


# ==========================================================
# ===== ROUTES FOR MERMAID DIAGRAMS =====
# ==========================================================
@app.route('/mermaid-converter')
def mermaid_converter():
    """Renders the UI for the Mermaid diagram converter."""
    return render_template('mermaid_converter.html')


# ========================================================
# ===== ROUTES FOR UNSTRUCTURED CONVERTER (NEW) =====
# ========================================================

@app.route('/document-converter')
def document_converter():
    """Renders the UI for the document converter."""
    return render_template('document_converter.html')

@app.route('/transform-document', methods=['POST'])
def transform_document():
    """Handles file upload and transformation using a robust hybrid approach."""
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

        # ==========================================================
        # ===== NEW HYBRID LOGIC: PyMuPDF + Unstructured
        # ==========================================================

        # --- Part 1: Use PyMuPDF to build a definitive link map ---
        link_map = {}
        try:
            app.logger.info("Starting PyMuPDF link extraction...")
            doc = fitz.open(temp_file_path)
            for page_num, page in enumerate(doc):
                links = page.get_links()
                for link in links:
                    if link.get('kind') == fitz.LINK_URI:
                        # The 'from' key is a Rect object defining the clickable area
                        clickable_area = link['from']
                        # Extract the text that falls inside that clickable area
                        link_text = page.get_textbox(clickable_area).strip().replace('\n', ' ')
                        link_url = link.get('uri')

                        if link_text and link_url:
                            # Store the text and URL. We use the text as a key.
                            # This handles cases where the same text links to the same URL multiple times.
                            link_map[link_text] = link_url
                            app.logger.info(f"PyMuPDF found link: '{link_text}' -> '{link_url}'")
            doc.close()
            app.logger.info(f"PyMuPDF finished. Found {len(link_map)} unique links.")
        except Exception as e:
            app.logger.error(f"PyMuPDF failed to process links: {e}", exc_info=True)
            # We can still proceed without links if this fails
            link_map = {}

        # --- Part 2: Use unstructured for the main text body ---
        app.logger.info("Partitioning document with unstructured (strategy='fast')...")
        elements = partition(filename=temp_file_path, strategy="fast")
        # Get the full plain text output from unstructured
        full_text = "\n\n".join([el.text for el in elements])

        # --- Part 3: Merge the results ---
        app.logger.info("Merging unstructured text with PyMuPDF link map...")
        output_markdown = full_text
        if link_map:
            # Sort keys by length, longest first, to avoid partial replacements
            # e.g., replace "McKinsey & Company" before "McKinsey"
            for link_text in sorted(link_map.keys(), key=len, reverse=True):
                link_url = link_map[link_text]
                markdown_link = f"[{link_text}]({link_url})"
                # Perform a simple but effective replacement on the entire text block
                output_markdown = output_markdown.replace(link_text, markdown_link)

        # ==========================================================
        # ===== END OF NEW HYBRID LOGIC
        # ==========================================================

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


# ==========================================================
# ===== ROUTES FOR AUDIO TRANSCRIPTION (DEEPGRAM) =====
# ==========================================================

@app.route('/audio-converter')
def audio_converter():
    """Renders the UI for the audio converter."""
    return render_template('audio_converter.html', deepgram_api_key_set=bool(DEEPGRAM_API_KEY))

@app.route('/api/get-deepgram-token', methods=['GET'])
def get_deepgram_token():
    """Provides the Deepgram API Key to the frontend for live transcription."""
    if not DEEPGRAM_API_KEY:
        app.logger.error("DEEPGRAM_API_KEY not configured on the server.")
        return jsonify({"error": "Audio transcription service is not configured."}), 503

    # For a production app with user accounts, you might generate a short-lived key here.
    # For this application, we provide the main key as the app is self-contained.
    return jsonify({"deepgram_token": DEEPGRAM_API_KEY})

@app.route('/transcribe-audio-file', methods=['POST'])
def transcribe_audio_file():
    """Handles audio file upload and transcription via Deepgram."""
    if not DEEPGRAM_API_KEY:
        return jsonify({"error": "Audio transcription service is not configured."}), 503

    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file part in the request."}), 400

    file = request.files['audio_file']
    language = request.form.get('language', 'en') # Default to English

    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    try:
        # Initialize Deepgram client
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)

        # Read file into a buffer
        buffer_data = file.read()
        payload = { "buffer": buffer_data }

        # Configure transcription options
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            utterances=True,
            punctuate=True,
            language=language
        )

        # Send the request to Deepgram
        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)

        # Extract the transcript
        transcript = response.results.channels[0].alternatives[0].transcript

        return jsonify({"transcript": transcript})

    except Exception as e:
        app.logger.error(f"Deepgram transcription failed: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred during transcription: {str(e)}"}), 500


# ==========================================================
# ===== ROUTES FOR PODCAST GENERATION (GOOGLE TTS) =====
# ==========================================================

@app.route('/generate-podcast', methods=['POST'])
def generate_podcast():
    """Handles text-to-speech conversion for podcast generation using Google Cloud TTS."""
    if not GOOGLE_CREDENTIALS_PATH or not os.path.exists(GOOGLE_CREDENTIALS_PATH):
        return jsonify({"error": "Google Cloud TTS is not configured. Please set up credentials."}), 503

    temp_audio_path = None
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        voice_name = data.get('voice_name', 'en-US-Neural2-C')
        language_code = data.get('language_code', 'en-US')
        speaking_rate = float(data.get('speaking_rate', 1.0))
        pitch = float(data.get('pitch', 0.0))

        if not text:
            return jsonify({"error": "No text provided for synthesis."}), 400

        # Initialize the Text-to-Speech client
        client = texttospeech.TextToSpeechClient()

        # Set the text input
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Build the voice request
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )

        # Select the type of audio file you want returned
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate,
            pitch=pitch
        )

        # Perform the text-to-speech request
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
            temp_audio.write(response.audio_content)
            temp_audio_path = temp_audio.name

        # Send the file to the user
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
        # Clean up the temporary file
        try:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
        except Exception as e:
            app.logger.error(f"Error cleaning up temp file: {e}")


@app.route('/api/get-google-voices', methods=['GET'])
def get_google_voices():
    """Returns a list of available Google Cloud TTS voices."""
    if not GOOGLE_CREDENTIALS_PATH or not os.path.exists(GOOGLE_CREDENTIALS_PATH):
        return jsonify({"error": "Google Cloud TTS is not configured."}), 503

    try:
        client = texttospeech.TextToSpeechClient()
        voices_response = client.list_voices()

        # Group voices by language
        voices_by_language = {}
        for voice in voices_response.voices:
            for language_code in voice.language_codes:
                if language_code not in voices_by_language:
                    voices_by_language[language_code] = []

                voices_by_language[language_code].append({
                    'name': voice.name,
                    'gender': texttospeech.SsmlVoiceGender(voice.ssml_gender).name,
                    'natural_sample_rate_hertz': voice.natural_sample_rate_hertz
                })

        return jsonify(voices_by_language)

    except Exception as e:
        app.logger.error(f"Failed to retrieve Google TTS voices: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ==========================================================
# ===== ROUTES FOR GEMINI-TTS (MULTI-SPEAKER PODCASTS) =====
# ==========================================================

@app.route('/generate-gemini-podcast', methods=['POST'])
def generate_gemini_podcast():
    """Handles multi-speaker podcast generation using Gemini-TTS (UPDATED VERSION)."""
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API Key is not configured."}), 503

    try:
        data = request.get_json()
        dialogue = data.get('dialogue', [])
        language = data.get('language', 'en')
        
        if not dialogue or len(dialogue) == 0:
            return jsonify({"error": "No dialogue provided."}), 400

        # ===== UPDATED: Use new library =====
        client = genai.Client(api_key=GEMINI_API_KEY)

        # Build multi-speaker configuration
        speaker_voice_configs = []
        seen_speakers = set()
        
        # Build the text with speaker labels
        dialogue_lines = []
        for turn in dialogue:
            speaker = turn.get('speaker', 'Kore')
            text = turn.get('text', '').strip()
            style = turn.get('style', '').strip()
            
            if not text:
                continue
            
            # Add to speaker configs if not seen
            if speaker not in seen_speakers:
                speaker_voice_configs.append(
                    types.SpeakerVoiceConfig(
                        speaker=speaker,
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=speaker
                            )
                        )
                    )
                )
                seen_speakers.add(speaker)
            
            # Format: "Speaker: [style] Text"
            if style:
                dialogue_lines.append(f"{speaker}: [{style}] {text}")
            else:
                dialogue_lines.append(f"{speaker}: {text}")
        
        full_dialogue = "\n".join(dialogue_lines)
        
        app.logger.info(f"Gemini-TTS dialogue:\n{full_dialogue}")
        app.logger.info(f"Speakers: {list(seen_speakers)}")

        # ===== UPDATED: Use new API structure =====
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=full_dialogue,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                        speaker_voice_configs=speaker_voice_configs
                    )
                )
            )
        )

        app.logger.info("Gemini-TTS response received")

        # ===== UPDATED: Extract audio from response =====
        if not response or not response.candidates:
            app.logger.error(f"Invalid response: {response}")
            return jsonify({"error": "Invalid response from Gemini-TTS."}), 500
        
        # Access inline_data correctly with new library
        audio_data = response.candidates[0].content.parts[0].inline_data.data
        mime_type = response.candidates[0].content.parts[0].inline_data.mime_type
        
        if not audio_data:
            app.logger.error(f"No audio found in response")
            return jsonify({"error": "No audio data in response."}), 500

        app.logger.info(f"Found audio: {len(audio_data)} bytes, type: {mime_type}")

        # Convert PCM to WAV
        import wave
        
        temp_audio_path = None
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            
            # Create WAV file
            with wave.open(temp_audio_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(24000)  # 24kHz
                wav_file.writeframes(audio_data)
        
        app.logger.info(f"Audio converted and saved to: {temp_audio_path}")
        
        # Send to user
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
        # Cleanup
        try:
            if 'temp_audio_path' in locals() and temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
        except Exception as e:
            app.logger.error(f"Cleanup error: {e}")
            
@app.route('/api/get-gemini-voices', methods=['GET'])
def get_gemini_voices():
    """Returns the list of available Gemini-TTS voice personas."""
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
    """Uses an LLM to format raw text into structured dialogue for Gemini-TTS."""
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API Key is not configured."}), 503

    try:
        data = request.get_json()
        raw_text = data.get('raw_text', '').strip()
        num_speakers = int(data.get('num_speakers', 2))
        speaker_descriptions = data.get('speaker_descriptions', [])
        language = data.get('language', 'en')
        tone = data.get('tone', 'professional and informative')
        
        if not raw_text:
            return jsonify({"error": "No text provided."}), 400
        
        if num_speakers < 1 or num_speakers > 4:
            return jsonify({"error": "Number of speakers must be between 1 and 4."}), 400

        # Build the prompt for the LLM
        speakers_info = ""
        for i, desc in enumerate(speaker_descriptions[:num_speakers], 1):
            name = desc.get('name', f'Speaker{i}')
            voice = desc.get('voice', 'Kore')
            personality = desc.get('personality', 'neutral')
            speakers_info += f"- {name} (Voice: {voice}, Personality: {personality})\n"
        
        language_name = {
            'en': 'English',
            'de': 'German',
            'es': 'Spanish',
            'fr': 'French'
        }.get(language, 'English')

        prompt = f"""You are a podcast script formatter. Convert the following raw text into a structured dialogue script.

**Context:**
- Number of speakers: {num_speakers}
- Language: {language_name}
- Overall tone: {tone}
- Speakers:
{speakers_info}

**Raw Text to Convert:**
{raw_text}

**Instructions:**
1. Split the text naturally into dialogue turns between the {num_speakers} speaker(s)
2. Each turn should be 1-3 sentences max for natural flow
3. Add appropriate style hints in square brackets (e.g., enthusiastically, calmly, thoughtfully)
4. Make it conversational and engaging
5. Use the exact speaker names provided above

**Output Format (one line per dialogue turn):**
SpeakerName [style]: Text of what they say

**Example:**
Anna [enthusiastically]: Welcome to our show!
Max [professionally]: Today we discuss artificial intelligence.
Anna [curiously]: What makes it so revolutionary?

Now convert the raw text above into this format. Output ONLY the formatted dialogue, nothing else."""

        # ===== UPDATED: Use new library =====
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Generate the formatted dialogue
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )
        
        if not response.text:
            return jsonify({"error": "LLM did not generate any output."}), 500
        
        formatted_dialogue = response.text.strip()
        
        app.logger.info(f"LLM formatted dialogue:\n{formatted_dialogue}")
        
        # Parse the formatted dialogue into structured format
        dialogue_lines = []
        for line in formatted_dialogue.split('\n'):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('**'):
                continue
                
            # Parse format: "Name [style]: Text" or "Name: Text"
            if ':' in line:
                parts = line.split(':', 1)
                speaker_part = parts[0].strip()
                text_part = parts[1].strip() if len(parts) > 1 else ""
                
                # Extract style if present
                style = ""
                speaker = speaker_part
                if '[' in speaker_part and ']' in speaker_part:
                    speaker = speaker_part.split('[')[0].strip()
                    style = speaker_part.split('[')[1].split(']')[0].strip()
                
                if text_part:
                    dialogue_lines.append({
                        'speaker': speaker,
                        'style': style,
                        'text': text_part
                    })
        
        if not dialogue_lines:
            return jsonify({"error": "Could not parse dialogue from LLM output."}), 500
        
        return jsonify({
            'dialogue': dialogue_lines,
            'raw_formatted_text': formatted_dialogue
        })

    except Exception as e:
        app.logger.error(f"Dialogue formatting failed: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# --- ASGI Wrapper ---
asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)