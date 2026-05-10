"""Podcast generation routes (Google TTS + Gemini async via RQ)."""
import os
from io import BytesIO
from pathlib import Path

from flask import jsonify, request, send_file
from flask_login import current_user, login_required
from rq.command import send_stop_job_command
from rq.exceptions import InvalidJobOperation, NoSuchJobError

from app_pkg.config import OUTPUT_DIR, TIMEOUT_RQ_JOB_SECONDS
from app_pkg.decorators import require_service
from services.gemini.prompts import STYLE_DIRECTIVES
from services.gemini.script import LANGUAGE_NAMES, LENGTH_INFO
from tasks import generate_podcast_task


# F-013: Google Cloud TTS-supported parameter ranges.
# https://cloud.google.com/text-to-speech/docs/reference/rest/v1/text/synthesize
_GOOGLE_TTS_RATE_RANGE = (0.25, 4.0)
_GOOGLE_TTS_PITCH_RANGE = (-20.0, 20.0)
_PODCAST_NUM_SPEAKERS_RANGE = (1, 4)


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
    @require_service('google_tts')
    def generate_podcast():
        temp_audio_path = None
        try:
            data = request.get_json(silent=True)
            if not isinstance(data, dict):
                return jsonify({"error": "Ungültiger Request-Body. JSON-Objekt erwartet."}), 400
            text = data.get('text', '').strip()
            voice_name = data.get('voice_name', 'en-US-Neural2-C')
            language_code = data.get('language_code', 'en-US')
            try:
                speaking_rate = float(data.get('speaking_rate', 1.0))
                pitch = float(data.get('pitch', 0.0))
            except (TypeError, ValueError):
                return jsonify({"error": "Sprechgeschwindigkeit und Tonhöhe müssen Zahlen sein."}), 400

            rate_min, rate_max = _GOOGLE_TTS_RATE_RANGE
            if not rate_min <= speaking_rate <= rate_max:
                return jsonify({
                    "error": f"Sprechgeschwindigkeit außerhalb des erlaubten Bereichs ({rate_min}–{rate_max})."
                }), 400
            pitch_min, pitch_max = _GOOGLE_TTS_PITCH_RANGE
            if not pitch_min <= pitch <= pitch_max:
                return jsonify({
                    "error": f"Tonhöhe außerhalb des erlaubten Bereichs ({pitch_min}–{pitch_max})."
                }), 400

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
                    app.logger.error(f"Error cleaning up temp file: {e}", exc_info=True)

    @app.route('/api/get-google-voices', methods=['GET'])
    @login_required
    @require_service('google_tts')
    def get_google_voices():
        try:
            voices = _app_module.google_tts_service.list_voices()
            return jsonify(voices)
        except Exception as e:
            app.logger.error(f"Failed to retrieve Google TTS voices: {e}", exc_info=True)
            return jsonify({"error": "Failed to retrieve voices."}), 500

    @app.route('/generate-gemini-podcast', methods=['POST'])
    @login_required
    @require_service('gemini')
    def generate_gemini_podcast():
        """Queue a podcast generation job to Redis."""
        try:
            data = request.get_json(silent=True)
            if not isinstance(data, dict):
                return jsonify({"error": "Ungültiger Request-Body. JSON-Objekt erwartet."}), 400
            dialogue = data.get('dialogue', [])
            language = data.get('language', 'en')
            tts_model = data.get('tts_model', None)

            # Enqueue job to Redis (worker will pick it up)
            job = _app_module.task_queue.enqueue(
                generate_podcast_task,
                args=(dialogue, language, tts_model),
                job_timeout=TIMEOUT_RQ_JOB_SECONDS,
                meta={'user_id': current_user.id}
            )

            app.logger.info(f"Job {job.id} queued for podcast generation")
            return jsonify({"job_id": job.id, "status": "queued"})

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
        cancelled_by_user = bool(job.meta.get('cancelled_by_user'))

        # SIGKILL via send_stop_job_command surfaces as ``stopped`` in rq
        # 2.x; ``canceled`` covers Job.cancel() of queued jobs. Either path
        # plus the meta flag means the user pressed Cancel — collapse to a
        # single user-visible "cancelled" terminal state.
        if status in ('stopped', 'canceled') or (cancelled_by_user and status == 'failed'):
            return jsonify({"status": "cancelled"})

        if status == 'finished':
            return jsonify({"status": "completed", "result": job.result})
        elif status == 'failed':
            error_msg = str(job.exc_info) if job.exc_info else "Unknown error"
            return jsonify({"status": "failed", "error": error_msg})
        elif status in ['queued', 'started']:
            return jsonify({"status": "processing"})
        else:
            return jsonify({"status": status})

    @app.route('/podcast-cancel/<job_id>', methods=['POST'])
    @login_required
    def podcast_cancel(job_id):
        """Cancel a queued or running podcast job.

        For queued jobs, ``Job.cancel()`` removes it from the queue. For
        started jobs, ``send_stop_job_command`` publishes a stop command;
        the worker then SIGKILLs the work-horse subprocess (rq 2.x
        cooperative-cancel = real mid-execution stop, not just a status
        flip — verified against rq.command source).

        Best-effort cleanup of the deterministic ``{job_id}.wav`` output
        file follows, with the same Path.is_relative_to guard as the
        download path. Marks ``meta['cancelled_by_user']`` so /podcast-status
        can collapse the resulting terminal state to a clean "cancelled"
        for the frontend.
        """
        try:
            job = _app_module.Job.fetch(job_id, connection=_app_module.redis_conn)
        except NoSuchJobError:
            return jsonify({"error": "Job not found"}), 404
        except Exception as e:
            app.logger.error(f"Failed to fetch RQ job {job_id} for cancel: {e}", exc_info=True)
            return jsonify({"error": "Job lookup failed"}), 500

        if job.meta.get('user_id') != current_user.id:
            return jsonify({"error": "Job not found"}), 404

        status = job.get_status()

        if status == 'finished':
            return jsonify({
                "status": "already_finished",
                "message": "Generierung wurde noch fertig. Datei wird verworfen."
            }), 200

        if status in ('failed', 'stopped', 'canceled'):
            return jsonify({"status": "already_terminal"}), 200

        try:
            if status == 'started':
                send_stop_job_command(_app_module.redis_conn, job_id)
            else:
                # queued / deferred / scheduled
                job.cancel()
        except InvalidJobOperation:
            # Race: status changed between fetch and stop. Treat as already
            # terminal — frontend polling will see the real state.
            app.logger.info(f"Cancel race for job {job_id}: status changed mid-flight")
        except Exception as e:
            app.logger.error(f"Failed to stop RQ job {job_id}: {e}", exc_info=True)
            return jsonify({"error": "Cancel-Befehl fehlgeschlagen."}), 500

        try:
            job.meta['cancelled_by_user'] = True
            job.save_meta()
        except Exception as e:
            app.logger.warning(f"Failed to persist cancel meta for {job_id}: {e}")

        # Best-effort cleanup of the deterministic output file. The worker
        # may have already moved a freshly-written WAV into OUTPUT_DIR
        # before SIGKILL landed — same path-traversal guard as download.
        candidate = os.path.join(OUTPUT_DIR, f"{job_id}.wav")
        try:
            real_path = os.path.realpath(candidate)
            if Path(real_path).is_relative_to(Path(os.path.realpath(OUTPUT_DIR))):
                if os.path.exists(real_path):
                    os.unlink(real_path)
                    app.logger.info(f"Cleaned up orphaned podcast file {real_path}")
        except OSError as e:
            app.logger.warning(f"Failed to clean up orphaned podcast file {candidate}: {e}")

        return jsonify({"status": "cancelling"}), 202

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

        # Prevent path traversal — ensure file is within allowed output directory.
        # Path.is_relative_to (Py 3.9+) avoids the prefix-collision bug of
        # str.startswith (e.g. "/app/output_podcasts2/x.wav" matching "/app/output_podcasts").
        real_path = os.path.realpath(file_path)
        if not Path(real_path).is_relative_to(Path(os.path.realpath(OUTPUT_DIR))):
            app.logger.warning(f"Path traversal attempt blocked: {file_path}")
            return jsonify({"error": "Datei außerhalb des Podcast-Verzeichnisses."}), 403

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
    @require_service('gemini')
    def format_dialogue_with_llm():
        try:
            data = request.get_json(silent=True)
            if not isinstance(data, dict):
                return jsonify({"error": "Ungültiger Request-Body. JSON-Objekt erwartet."}), 400

            raw_text_received = data.get('raw_text', '')

            if not raw_text_received or not raw_text_received.strip():
                return jsonify({"error": "No text provided for formatting"}), 400

            language = data.get('language', 'en')
            narration_style = data.get('narration_style', 'conversational')
            script_length = data.get('script_length', 'medium')

            if language not in LANGUAGE_NAMES:
                return jsonify({
                    "error": "Ungültige Sprache. Erlaubt: " + ", ".join(LANGUAGE_NAMES) + "."
                }), 400
            if narration_style not in STYLE_DIRECTIVES:
                return jsonify({
                    "error": "Ungültiger Erzählstil. Erlaubt: " + ", ".join(STYLE_DIRECTIVES) + "."
                }), 400
            if script_length not in LENGTH_INFO:
                return jsonify({
                    "error": "Ungültige Skriptlänge. Erlaubt: " + ", ".join(LENGTH_INFO) + "."
                }), 400

            try:
                num_speakers = int(data.get('num_speakers', 2))
            except (TypeError, ValueError):
                return jsonify({"error": "Anzahl der Sprecher muss eine Zahl sein."}), 400
            speakers_min, speakers_max = _PODCAST_NUM_SPEAKERS_RANGE
            if not speakers_min <= num_speakers <= speakers_max:
                return jsonify({
                    "error": f"Anzahl der Sprecher muss zwischen {speakers_min} und {speakers_max} liegen."
                }), 400

            result = _app_module.gemini_service.format_dialogue_with_llm(
                raw_text=raw_text_received.strip(),
                num_speakers=num_speakers,
                speaker_descriptions=data.get('speaker_descriptions', []),
                language=language,
                narration_style=narration_style,
                script_length=script_length,
                custom_prompt=(data.get('custom_prompt') or '').strip() or None
            )

            return jsonify(result)

        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            app.logger.error(f"Dialogue formatting failed: {e}", exc_info=True)
            return jsonify({"error": "An error occurred during dialogue formatting. Please try again."}), 500
