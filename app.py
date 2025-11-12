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
import fitz
from deepgram import DeepgramClient, PrerecordedOptions
from flask import jsonify
import traceback
from google import genai
from google.genai import types


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

# ===== KEYTERMS LOADER =====
def load_keyterms(language='en'):
    """
    Load domain-specific keyterms for improved transcription accuracy.
    Returns a list of keyterms for the specified language.
    """
    try:
        keyterms_path = Path('/app/keyterms.json')
        if not keyterms_path.exists():
            logging.warning("keyterms.json not found, using empty keyterms list")
            return []
        
        with open(keyterms_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Combine language-specific and universal terms
        terms = data.get(language, []) + data.get('universal', [])
        
        # Limit to 100 terms as per Deepgram recommendation
        terms = terms[:100]
        
        logging.info(f"Loaded {len(terms)} keyterms for language '{language}'")
        return terms
    except Exception as e:
        logging.error(f"Error loading keyterms: {e}")
        return []

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

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
    if not DEEPGRAM_API_KEY:
        app.logger.error("DEEPGRAM_API_KEY not configured on the server.")
        return jsonify({"error": "Audio transcription service is not configured."}), 503

    return jsonify({"deepgram_token": DEEPGRAM_API_KEY})

@app.route('/transcribe-audio-file', methods=['POST'])
def transcribe_audio_file():
    """
    OPTIMIZED: Now uses Nova-3 model with keyterm prompting and improved parameters
    """
    if not DEEPGRAM_API_KEY:
        return jsonify({"error": "Audio transcription service is not configured."}), 503

    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file part in the request."}), 400

    file = request.files['audio_file']
    language = request.form.get('language', 'en')

    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    try:
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        buffer_data = file.read()
        payload = {"buffer": buffer_data}

        # Load keyterms for the selected language
        keyterms = load_keyterms(language)

        # OPTIMIZED OPTIONS for Nova-3
        options = PrerecordedOptions(
            model="nova-3",              # 🔴 UPGRADED from nova-2
            smart_format=True,            # Already optimal
            utterances=True,              # Semantic segmentation
            punctuate=True,               # Included in smart_format, but explicit
            language=language,            # Explicit is better than auto-detection
            
            # NEW OPTIMIZED PARAMETERS
            keyterms=keyterms,            # 🔴 Domain-specific terms for 90% better accuracy
            numerals=True,                # Better number formatting
            paragraphs=True,              # For longer texts
            # diarize=False,              # User doesn't need speaker separation
        )

        app.logger.info(f"Transcribing with Nova-3, language={language}, keyterms={len(keyterms)}")
        
        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
        transcript = response.results.channels[0].alternatives[0].transcript

        return jsonify({"transcript": transcript})

    except Exception as e:
        app.logger.error(f"Deepgram transcription failed: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred during transcription: {str(e)}"}), 500


@app.route('/generate-podcast', methods=['POST'])
def generate_podcast():
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

        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate,
            pitch=pitch
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
            temp_audio.write(response.audio_content)
            temp_audio_path = temp_audio.name

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
        try:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
        except Exception as e:
            app.logger.error(f"Error cleaning up temp file: {e}")


@app.route('/api/get-google-voices', methods=['GET'])
def get_google_voices():
    if not GOOGLE_CREDENTIALS_PATH or not os.path.exists(GOOGLE_CREDENTIALS_PATH):
        return jsonify({"error": "Google Cloud TTS is not configured."}), 503

    try:
        client = texttospeech.TextToSpeechClient()
        voices_response = client.list_voices()

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


@app.route('/generate-gemini-podcast', methods=['POST'])
def generate_gemini_podcast():
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API Key is not configured."}), 503

    try:
        data = request.get_json()
        dialogue = data.get('dialogue', [])
        language = data.get('language', 'en')
        
        if not dialogue or len(dialogue) == 0:
            return jsonify({"error": "No dialogue provided."}), 400

        client = genai.Client(api_key=GEMINI_API_KEY)

        speaker_voice_configs = []
        seen_speakers = set()
        
        dialogue_lines = []
        for turn in dialogue:
            speaker = turn.get('speaker', 'Kore')
            text = turn.get('text', '').strip()
            style = turn.get('style', '').strip()
            
            if not text:
                continue
            
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
            
            if style:
                dialogue_lines.append(f"{speaker}: [{style}] {text}")
            else:
                dialogue_lines.append(f"{speaker}: {text}")
        
        full_dialogue = "\n".join(dialogue_lines)
        
        unique_speakers = list(seen_speakers)
        
        if len(unique_speakers) < 1:
            app.logger.error(f"No speakers found in dialogue.")
            return jsonify({
                "error": "No speakers found. Please configure at least 1 speaker."
            }), 400
        elif len(unique_speakers) > 4:
            app.logger.error(f"Too many speakers: {len(unique_speakers)}. Gemini TTS supports maximum 4 speakers.")
            return jsonify({
                "error": "Gemini TTS supports maximum 4 speakers. Please reduce to 1-4 speakers."
            }), 400
        
        app.logger.info(f"Gemini-TTS dialogue:\n{full_dialogue}")
        app.logger.info(f"Speakers: {list(seen_speakers)}")

        if len(unique_speakers) == 1:
            app.logger.info("Using single-speaker mode")
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=full_dialogue,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=speaker_voice_configs[0].speaker
                            )
                        )
                    )
                )
            )
        else:
            app.logger.info("Using multi-speaker mode")
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

        if not response or not response.candidates:
            app.logger.error(f"Invalid response: {response}")
            return jsonify({"error": "Invalid response from Gemini-TTS."}), 500
        
        audio_data = response.candidates[0].content.parts[0].inline_data.data
        mime_type = response.candidates[0].content.parts[0].inline_data.mime_type
        
        if not audio_data:
            app.logger.error(f"No audio found in response")
            return jsonify({"error": "No audio data in response."}), 500

        app.logger.info(f"Found audio: {len(audio_data)} bytes, type: {mime_type}")

        import wave
        
        temp_audio_path = None
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            
            with wave.open(temp_audio_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes(audio_data)
        
        app.logger.info(f"Audio converted and saved to: {temp_audio_path}")
        
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
        try:
            if 'temp_audio_path' in locals() and temp_audio_path and os.path.exists(temp_audio_path):
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
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API Key is not configured."}), 503

    try:
        data = request.get_json()
        raw_text = data.get('raw_text', '').strip()
        num_speakers = int(data.get('num_speakers', 2))
        speaker_descriptions = data.get('speaker_descriptions', [])
        language = data.get('language', 'en')
        tone = data.get('tone', 'professional and informative')
        script_length = data.get('script_length', 'medium')
        custom_prompt = data.get('custom_prompt', '').strip()
        
        if not raw_text:
            return jsonify({"error": "No text provided."}), 400
        
        if num_speakers < 1 or num_speakers > 4:
            return jsonify({"error": "Number of speakers must be between 1 and 4."}), 400

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
        
        length_info = {
            'short': ('300-500 words', 'short (2-3 minute)'),
            'medium': ('800-1200 words', 'medium-length (5-7 minute)'),
            'long': ('1500-2500 words', 'long (10-15 minute)'),
            'very-long': ('3000-5000 words', 'very long (20-30 minute)')
        }
        target_length, target_length_desc = length_info.get(script_length, length_info['medium'])

        if custom_prompt:
            prompt = custom_prompt
        else:
            if num_speakers == 1:
                prompt = f"""You are a narrator/audiobook reader. Convert the following raw text into a natural, engaging narration.

**Context:**
- Single speaker (monologue)
- Language: {language_name}
- Overall tone: {tone}
- Target length: {target_length}
- Speaker:
{speakers_info}

**Raw Text to Convert:**
{raw_text}

**Instructions:**
1. Create a natural, engaging {target_length_desc} narration from the raw text
2. Format the text naturally for speaking, with appropriate pauses and emphasis
3. Add style hints in square brackets where appropriate (e.g., [thoughtfully], [enthusiastically], [calmly])
4. Maintain the {tone} tone throughout
5. Use the exact speaker name provided above
6. IMPORTANT: Generate enough content to match the target length of {target_length}. Expand on topics naturally with details and examples.
7. Make it engaging and natural to listen to

**Output Format (one line per section):**
SpeakerName [optional-style]: Text to be spoken

**Example:**
Narrator [warmly]: Welcome to today's story about artificial intelligence and how it's transforming our world.
Narrator [thoughtfully]: Let me start by explaining what AI really means...

Now convert the raw text above into this format for a single narrator. Remember to generate {target_length_desc} content!"""
            else:
                prompt = f"""You are a podcast script formatter. Convert the following raw text into a structured dialogue script.

**Context:**
- Number of speakers: {num_speakers}
- Language: {language_name}
- Overall tone: {tone}
- Target length: {target_length}
- Speakers:
{speakers_info}

**Raw Text to Convert:**
{raw_text}

**Instructions:**
1. Create a natural, engaging {target_length_desc} podcast script from the raw text
2. Split the content naturally into dialogue turns between the {num_speakers} speaker(s)
3. Each turn should be 1-4 sentences for natural conversational flow
4. Add appropriate style hints in square brackets (e.g., enthusiastically, calmly, thoughtfully)
5. Make it conversational, engaging, and maintain the {tone} tone
6. Use the exact speaker names provided above
7. IMPORTANT: Generate enough content to match the target length of {target_length}. Don't be brief - expand on topics naturally with details, examples, and explanations.
8. Include transitions, questions, clarifications, and natural back-and-forth between speakers
9. Add depth by exploring different angles, asking follow-up questions, and providing context

**Output Format (one line per dialogue turn):**
SpeakerName [style]: Text of what they say

**Example:**
Anna [enthusiastically]: Welcome to our show!
Max [professionally]: Today we discuss artificial intelligence and how it's transforming every aspect of our daily lives.
Anna [curiously]: That's fascinating! Can you break down what exactly makes AI so revolutionary for our listeners who might not be familiar with the technical details?
Max [thoughtfully]: Great question. Let me start with the basics and then we'll dive deeper into some specific examples...

Now convert the raw text above into this format. Remember to generate {target_length_desc} content - don't rush, take time to explore the topic thoroughly!"""
        
        prompt = prompt.replace('{num_speakers}', str(num_speakers))
        prompt = prompt.replace('{language_name}', language_name)
        prompt = prompt.replace('{tone}', tone)
        prompt = prompt.replace('{target_length}', target_length)
        prompt = prompt.replace('{target_length_desc}', target_length_desc)
        prompt = prompt.replace('{speakers_info}', speakers_info)
        prompt = prompt.replace('{raw_text}', raw_text)

        app.logger.info(f"Generating dialogue with script_length={script_length}, target={target_length}, num_speakers={num_speakers}")
        app.logger.info(f"Prompt preview: {prompt[:500]}...")

        client = genai.Client(api_key=GEMINI_API_KEY)
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=8192
            )
        )
        
        if not response.text:
            return jsonify({"error": "LLM did not generate any output."}), 500
        
        formatted_dialogue = response.text.strip()
        
        app.logger.info(f"LLM formatted dialogue length: {len(formatted_dialogue)} characters")
        app.logger.info(f"LLM formatted dialogue preview:\n{formatted_dialogue[:1000]}...")
        
        dialogue_lines = []
        for line in formatted_dialogue.split('\n'):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('**'):
                continue
                
            if ':' in line:
                parts = line.split(':', 1)
                speaker_part = parts[0].strip()
                text_part = parts[1].strip() if len(parts) > 1 else ""
                
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
        
        app.logger.info(f"Parsed {len(dialogue_lines)} dialogue lines")
        
        return jsonify({
            'dialogue': dialogue_lines,
            'raw_formatted_text': formatted_dialogue
        })

    except Exception as e:
        app.logger.error(f"Dialogue formatting failed: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)