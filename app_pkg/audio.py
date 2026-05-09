"""Audio transcription routes (Deepgram-backed)."""
import os

from flask import jsonify, render_template, request
from flask_login import login_required
from werkzeug.utils import secure_filename


# Single source of truth for what /transcribe-audio-file accepts. The template
# reads this via the route context for the file-input ``accept`` attribute and
# for ``window.PageData.acceptedAudioExtensions`` (frontend pre-submit check).
ACCEPTED_AUDIO_EXTENSIONS = ('mp3', 'wav', 'm4a', 'ogg', 'flac', 'webm')
MAX_AUDIO_FILE_SIZE_MB = 500

# F-013: enumerated languages that the audio-tab UI offers. Values outside
# this set used to flow through to Deepgram and surface as a 500 from the SDK;
# they now get a clean 400 + DE-JSON.
ACCEPTED_TRANSCRIPTION_LANGUAGES = ('en', 'de')


def register(app):
    # Late import: tests patch ``app.deepgram_service`` and
    # ``app.DEEPGRAM_API_KEY`` on the top-level app.py module, so look
    # them up at call time rather than capturing imports here.
    import app as _app_module

    @app.route('/audio-converter')
    @login_required
    def audio_converter():
        return render_template(
            'audio_converter.html',
            deepgram_api_key_set=bool(_app_module.DEEPGRAM_API_KEY),
            gemini_api_key_set=bool(_app_module.GEMINI_API_KEY),
            accepted_audio_extensions=ACCEPTED_AUDIO_EXTENSIONS,
            accepted_audio_extensions_accept=','.join('.' + ext for ext in ACCEPTED_AUDIO_EXTENSIONS),
            max_audio_file_size_mb=MAX_AUDIO_FILE_SIZE_MB,
        )

    @app.route('/api/get-deepgram-token', methods=['GET'])
    @login_required
    def get_deepgram_token():
        if not _app_module.deepgram_service:
            app.logger.error("Deepgram service not configured")
            return jsonify({"error": "Audio transcription service is not configured."}), 503

        try:
            temp_key = _app_module.deepgram_service.create_temporary_key(ttl_seconds=60)
            return jsonify({"deepgram_token": temp_key})
        except Exception as e:
            app.logger.error(f"Failed to create temporary Deepgram key: {e}")
            return jsonify({"error": "Failed to create transcription token."}), 500

    @app.route('/transcribe-audio-file', methods=['POST'])
    @login_required
    def transcribe_audio_file():
        if not _app_module.deepgram_service:
            return jsonify({"error": "Audio transcription service is not configured."}), 503

        if 'audio_file' not in request.files:
            return jsonify({"error": "No audio file part in the request."}), 400

        file = request.files['audio_file']
        language = request.form.get('language', 'en')

        if file.filename == '':
            return jsonify({"error": "No file selected."}), 400

        if language not in ACCEPTED_TRANSCRIPTION_LANGUAGES:
            return jsonify({
                "error": "Ungültige Sprache. Erlaubt: "
                         + ", ".join(ACCEPTED_TRANSCRIPTION_LANGUAGES) + "."
            }), 400

        original_filename = secure_filename(file.filename)
        ext = os.path.splitext(original_filename)[1].lstrip('.').lower()
        if ext not in ACCEPTED_AUDIO_EXTENSIONS:
            return jsonify({
                "error": "Dieses Dateiformat wird nicht unterstützt. "
                         "Erlaubt: MP3, WAV, M4A, OGG, FLAC, WEBM."
            }), 400

        try:
            buffer_data = file.read()
            file_size_mb = len(buffer_data) / (1024 * 1024)

            app.logger.info(f"Received audio file: {file.filename} ({file_size_mb:.1f} MB)")

            # transcribe_file handhabt automatisch Splitting wenn nötig
            transcript = _app_module.deepgram_service.transcribe_file(buffer_data, language)

            return jsonify({
                "transcript": transcript,
                "metadata": {
                    "file_size_mb": round(file_size_mb, 2),
                    "transcript_length": len(transcript),
                    "language": language
                }
            })

        except RuntimeError as e:
            # Chunk-spezifischer Fehler
            app.logger.error(f"Chunked transcription failed: {e}", exc_info=True)
            return jsonify({
                "error": "Transcription of long audio failed. Please try a shorter file."
            }), 500

        except Exception as e:
            app.logger.error(f"Deepgram transcription failed: {e}", exc_info=True)
            return jsonify({"error": "An error occurred during transcription. Please try again."}), 500
