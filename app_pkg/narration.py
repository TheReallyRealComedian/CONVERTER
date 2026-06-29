"""Faithful-narration library routes (NARR-2).

Serves the persisted audio of an ``audio_narration`` Conversion. The WAV lives
at the deterministic, **id-derived** path ``OUTPUT_DIR/narration_<id>.wav`` on
the shared ``podcast_data`` volume (written by the NARR-3 worker). Because the
path is resolved from the id — never from ``metadata.audio_filename`` — there is
no filename-injection surface; a traversal guard against ``OUTPUT_DIR`` is kept
as belt-and-suspenders.

Session-authed (``@login_required``, owner-404) and **persistent**: unlike the
alt-podcast ``podcast_download``, the file is **never deleted on serve**. NARR-3
will add the token POST that creates narrations to this same module.
"""
import os
from pathlib import Path

from flask import jsonify, send_file
from flask_login import login_required

from app_pkg.config import OUTPUT_DIR
from app_pkg.library import get_owned_conversion
from services.narration_library import narration_audio_path, narration_status


def register(app):
    @app.route('/api/narrations/<int:conversion_id>/audio', methods=['GET'])
    @login_required
    def narration_audio(conversion_id):
        """Stream a ready narration's WAV.

        404 on: not owned / missing (``get_owned_conversion``), wrong type
        (no type leak), status ≠ ready (pending/failed have no file), or file
        absent. 403 on a path that escapes ``OUTPUT_DIR``. Never unlinks
        (persistent library element).
        """
        conversion = get_owned_conversion(conversion_id)  # owner-404

        if conversion.conversion_type != 'audio_narration':
            return jsonify({'error': 'Vertonung nicht gefunden.'}), 404

        if narration_status(conversion) != 'ready':
            # pending/failed carry no audio file yet; NARR-5 surfaces the state.
            return jsonify({'error': 'Audio nicht verfügbar.'}), 404

        # id-derived path (no user input); existence-check before the guard.
        file_path = narration_audio_path(conversion_id)
        if not os.path.exists(file_path):
            return jsonify({'error': 'Audio nicht verfügbar.'}), 404

        # Traversal guard like podcast_download — Path.is_relative_to avoids the
        # str.startswith prefix-collision bug.
        real_path = os.path.realpath(file_path)
        if not Path(real_path).is_relative_to(Path(os.path.realpath(OUTPUT_DIR))):
            app.logger.warning(f"Narration path traversal blocked: {file_path}")
            return jsonify({'error': 'Datei außerhalb des Ausgabe-Verzeichnisses.'}), 403

        # Persistent: NO os.unlink (unlike podcast_download).
        return send_file(
            real_path,
            mimetype='audio/wav',
            download_name=f'narration_{conversion_id}.wav',
        )
