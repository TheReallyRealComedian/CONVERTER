"""Podcast generation routes (Google TTS + Gemini async via RQ)."""
import os
from io import BytesIO

from flask import jsonify, request, send_file
from flask_login import current_user, login_required
from rq.exceptions import NoSuchJobError

from app_pkg.config import OUTPUT_DIR
from tasks import generate_podcast_task


_GEMINI_VOICES = {
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
        {"name": "Zubenelgenubi", "description": "Casual and relaxed"},
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
        {"name": "Vindemiatrix", "description": "Gentle and warm"},
    ],
    "neutral": [
        {"name": "Kore", "description": "Firm (can be male or female)"},
        {"name": "Achird", "description": "Friendly and approachable"},
        {"name": "Schedar", "description": "Even and balanced"},
        {"name": "Sadachbia", "description": "Lively and animated"},
        {"name": "Sulafat", "description": "Warm and inviting"},
    ],
}


def register(app):
    # Late import: tests patch the singletons (``task_queue``, ``Job``,
    # ``gemini_service``, ``google_tts_service``, ``GEMINI_API_KEY``,
    # ``redis_conn``, ``OUTPUT_DIR``) on the top-level app.py module, so
    # look them up at call time rather than capturing imports here.
    import app as _app_module

    @app.route('/generate-podcast', methods=['POST'])
    @login_required
    def generate_podcast():
        if not _app_module.google_tts_service:
            return jsonify({"error": "Google Cloud TTS is not configured."}), 503

        temp_audio_path = None
        try:
            data = request.get_json()
            text = data.get('text', '').strip()
            voice_name = data.get('voice_name', 'en-US-Neural2-C')
            language_code = data.get('language_code', 'en-US')
            speaking_rate = float(data.get('speaking_rate', 1.0))
            pitch = float(data.get('pitch', 0.0))

            temp_audio_path = _app_module.google_tts_service.synthesize_speech(
                text, voice_name, language_code, speaking_rate, pitch
            )

            audio_buffer = BytesIO()
            with open(temp_audio_path, 'rb') as f:
                audio_buffer.write(f.read())
            audio_buffer.seek(0)

            return send_file(
                audio_buffer,
                as_attachment=True,
                download_name='podcast.mp3',
                mimetype='audio/mpeg'
            )

        except Exception as e:
            app.logger.error(f"Google TTS synthesis failed: {e}", exc_info=True)
            return jsonify({"error": "An error occurred during synthesis. Please try again."}), 500
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except Exception as e:
                    app.logger.error(f"Error cleaning up temp file: {e}")

    @app.route('/api/get-google-voices', methods=['GET'])
    @login_required
    def get_google_voices():
        if not _app_module.google_tts_service:
            return jsonify({"error": "Google Cloud TTS is not configured."}), 503

        try:
            voices = _app_module.google_tts_service.list_voices()
            return jsonify(voices)
        except Exception as e:
            app.logger.error(f"Failed to retrieve Google TTS voices: {e}", exc_info=True)
            return jsonify({"error": "Failed to retrieve voices."}), 500

    @app.route('/generate-gemini-podcast', methods=['POST'])
    @login_required
    def generate_gemini_podcast():
        """Queue a podcast generation job to Redis."""
        if not _app_module.GEMINI_API_KEY:
            return jsonify({"error": "Gemini API Key is not configured."}), 503

        try:
            data = request.get_json()
            dialogue = data.get('dialogue', [])
            language = data.get('language', 'en')
            tts_model = data.get('tts_model', None)

            # Enqueue job to Redis (worker will pick it up)
            job = _app_module.task_queue.enqueue(
                generate_podcast_task,
                args=(dialogue, language, tts_model),
                job_timeout=600,  # 10 minutes max
                meta={'user_id': current_user.id}
            )

            app.logger.info(f"Job {job.get_id()} queued for podcast generation")
            return jsonify({"job_id": job.get_id(), "status": "queued"})

        except Exception as e:
            app.logger.error(f"Failed to queue podcast job: {e}", exc_info=True)
            return jsonify({"error": "Failed to queue podcast job. Please try again."}), 500

    @app.route('/podcast-status/<job_id>', methods=['GET'])
    @login_required
    def podcast_status(job_id):
        """Check the status of a podcast generation job in Redis."""
        try:
            job = _app_module.Job.fetch(job_id, connection=_app_module.redis_conn)
        except NoSuchJobError:
            return jsonify({"error": "Job not found"}), 404
        except Exception as e:
            app.logger.error(f"Failed to fetch RQ job {job_id}: {e}", exc_info=True)
            return jsonify({"error": "Job lookup failed"}), 500

        if job.meta.get('user_id') != current_user.id:
            return jsonify({"error": "Job not found"}), 404

        status = job.get_status()

        if status == 'finished':
            return jsonify({"status": "completed", "result": job.result})
        elif status == 'failed':
            error_msg = str(job.exc_info) if job.exc_info else "Unknown error"
            return jsonify({"status": "failed", "error": error_msg})
        elif status in ['queued', 'started']:
            return jsonify({"status": "processing"})
        else:
            return jsonify({"status": status})

    @app.route('/podcast-download/<job_id>', methods=['GET'])
    @login_required
    def podcast_download(job_id):
        """Download the generated podcast file."""
        try:
            job = _app_module.Job.fetch(job_id, connection=_app_module.redis_conn)
        except NoSuchJobError:
            return jsonify({"error": "Job not found"}), 404
        except Exception as e:
            app.logger.error(f"Failed to fetch RQ job {job_id}: {e}", exc_info=True)
            return jsonify({"error": "Job lookup failed"}), 500

        if job.meta.get('user_id') != current_user.id:
            return jsonify({"error": "Job not found"}), 404

        if not job.is_finished:
            return jsonify({"error": "Job not ready"}), 400

        file_path = job.result

        if not file_path or not os.path.exists(file_path):
            return jsonify({"error": "File not found on server"}), 404

        # Prevent path traversal — ensure file is within allowed output directory
        real_path = os.path.realpath(file_path)
        if not real_path.startswith(os.path.realpath(OUTPUT_DIR)):
            app.logger.warning(f"Path traversal attempt blocked: {file_path}")
            return jsonify({"error": "Invalid file path"}), 403

        # Read into buffer and delete file to prevent unbounded disk growth
        podcast_buffer = BytesIO()
        with open(real_path, 'rb') as f:
            podcast_buffer.write(f.read())
        podcast_buffer.seek(0)

        try:
            os.unlink(real_path)
        except Exception as e:
            app.logger.warning(f"Failed to clean up podcast file {real_path}: {e}")

        return send_file(
            podcast_buffer,
            as_attachment=True,
            download_name='gemini_podcast.wav',
            mimetype='audio/wav'
        )

    @app.route('/api/get-gemini-voices', methods=['GET'])
    @login_required
    def get_gemini_voices():
        return jsonify(_GEMINI_VOICES)

    @app.route('/format-dialogue-with-llm', methods=['POST'])
    @login_required
    def format_dialogue_with_llm():
        if not _app_module.gemini_service:
            return jsonify({"error": "Gemini API Key is not configured."}), 503

        try:
            data = request.get_json()

            raw_text_received = data.get('raw_text', '')

            if not raw_text_received or not raw_text_received.strip():
                return jsonify({"error": "No text provided for formatting"}), 400

            result = _app_module.gemini_service.format_dialogue_with_llm(
                raw_text=raw_text_received.strip(),
                num_speakers=int(data.get('num_speakers', 2)),
                speaker_descriptions=data.get('speaker_descriptions', []),
                language=data.get('language', 'en'),
                narration_style=data.get('narration_style', 'conversational'),
                script_length=data.get('script_length', 'medium'),
                custom_prompt=(data.get('custom_prompt') or '').strip() or None
            )

            return jsonify(result)

        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            app.logger.error(f"Dialogue formatting failed: {e}", exc_info=True)
            return jsonify({"error": "An error occurred during dialogue formatting. Please try again."}), 500
