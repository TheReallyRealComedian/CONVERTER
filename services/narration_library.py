"""Persistence helpers for the faithful-narration library element (NARR-2).

Pure module — no Flask context, no SDK. Turns a rendered narration into a
first-class library citizen by reusing the existing ``Conversion`` schema
(``conversion_type='audio_narration'``) instead of a new table:

* ``content`` is the speaker-labelled transcript Markdown (NOT NULL → never
  empty), so the narration is Reader-/search-/tag-/highlight-capable and
  ``derive_title`` / ``list_conversions`` work for free.
* The audio-specific fields the schema lacks (status, audio-file pointer,
  duration, model, voice map, structured transcript, error) live in
  ``metadata_json`` — the Text/JSON escape-hatch, **no schema touch, no
  migration**.
* The audio file itself lives at the deterministic, persistent path
  ``OUTPUT_DIR/narration_<id>.wav`` on the shared ``podcast_data`` volume
  (written by the NARR-3 worker, served by ``app_pkg/narration.py``).

This sprint builds only the persistence shell — nothing here generates audio.
"""
import json
import os
from pathlib import Path

from app_pkg.config import OUTPUT_DIR

# --- metadata_json contract ---------------------------------------------------
#
# An audio_narration Conversion stores its audio-specific state under
# ``metadata_json`` (parsed via ``Conversion.to_dict()['metadata']``). The
# documented v1 schema:
#
#   {
#     "narration_status": "pending" | "ready" | "failed",
#     "audio_filename": "narration_<id>.wav",   # mirror of narration_audio_path
#     "audio_mimetype": "audio/wav",
#     "duration_seconds": int | null,
#     "tts_model": "gemini-2.5-flash-tts",       # services.narration_render model
#     "speakers": {"Anna": "Kore", "Ben": "Puck"},  # label -> Gemini voice id
#     "transcript": [{"speaker": "...", "text": "..."}],
#     "mode": "single_speaker" | "two_speaker",  # render input (NARR-5 retry)
#     "style_prompt": null | "...",              # render input (NARR-5 retry)
#     "language_code": "de-DE",                  # render input (NARR-5 retry)
#     "error": null | "..."                      # set when status == 'failed'
#   }
#
# ``mode`` / ``style_prompt`` / ``language_code`` are the render-only inputs the
# audio doesn't otherwise capture; NARR-5 persists them so a failed narration can
# be re-enqueued faithfully (POST /api/narrations/<id>/retry) without a new agent
# call. (Pre-NARR-5 rows lack them → retry falls back: mode from speaker count,
# no style, default language.)
#
# Only ``status == 'ready'`` means an audio file exists and is servable; the
# serve endpoint gates on exactly that.
NARRATION_STATUS_PENDING = 'pending'
NARRATION_STATUS_READY = 'ready'
NARRATION_STATUS_FAILED = 'failed'
NARRATION_STATUSES = (
    NARRATION_STATUS_PENDING,
    NARRATION_STATUS_READY,
    NARRATION_STATUS_FAILED,
)

DEFAULT_AUDIO_MIMETYPE = 'audio/wav'


def _audio_basename(conversion_id):
    """The deterministic audio filename for a narration: ``narration_<id>.wav``.

    Derived purely from the id, so it never carries user-controlled input — the
    serve endpoint resolves the path from the id, not from stored metadata.
    """
    return f'narration_{conversion_id}.wav'


def narration_audio_path(conversion_id):
    """Deterministic, persistent absolute path for a narration's audio file.

    ``OUTPUT_DIR/narration_<id>.wav`` on the shared ``podcast_data`` volume,
    mounted in both the web and worker containers. The ``narration_<id>`` name
    namespace never collides with the alt-podcast ``<job_id>.wav`` files.
    """
    return os.path.join(OUTPUT_DIR, _audio_basename(conversion_id))


def narration_to_markdown(turns):
    """Render a narration turn list as readable, speaker-labelled Markdown.

    Becomes the Conversion ``content`` (NOT NULL). The turn ``text`` is emitted
    **verbatim** — no Markdown/HTML escaping or mangling, since it is read
    content that flows through the shared renderer downstream.

    Shape:
      * Two-speaker → each turn is a labelled paragraph (``**Anna:** …`` blocks,
        blank-line separated).
      * Single-speaker → unlabelled paragraphs (the label is noise with one
        voice).

    Defensive: non-dict turns and blank-text turns are skipped, ``None`` text
    coerces to empty. Callers (the NARR-3 worker) pass already-validated,
    non-empty turns, so in practice the output is always non-empty.

    Args:
        turns: list of ``{'speaker': label, 'text': str}``.

    Returns:
        str: the transcript as Markdown (``''`` only for degenerate input).
    """
    if not isinstance(turns, list):
        return ''

    cleaned = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        text = turn.get('text')
        text = '' if text is None else str(text)
        if not text.strip():
            continue
        speaker = turn.get('speaker')
        speaker = speaker.strip() if isinstance(speaker, str) else ''
        cleaned.append((speaker, text))

    if not cleaned:
        return ''

    distinct = list(dict.fromkeys(speaker for speaker, _ in cleaned if speaker))
    if len(distinct) <= 1:
        # Single (or unlabelled) speaker → unlabelled paragraphs.
        return '\n\n'.join(text for _, text in cleaned)

    blocks = [f'**{speaker or "Sprecher"}:** {text}' for speaker, text in cleaned]
    return '\n\n'.join(blocks)


def build_narration_metadata(conversion_id, *, status=NARRATION_STATUS_PENDING,
                             tts_model=None, speakers=None, transcript=None,
                             duration_seconds=None, error=None,
                             mode=None, style_prompt=None, language_code=None,
                             audio_mimetype=DEFAULT_AUDIO_MIMETYPE):
    """Build the ``metadata_json`` dict for an audio_narration Conversion.

    Mirrors the documented contract above. ``audio_filename`` is derived
    deterministically from the id so it always matches ``narration_audio_path``.
    The NARR-3 worker fills ``tts_model`` / ``speakers`` / ``transcript`` and
    flips ``status`` to ``ready`` (or ``failed`` + ``error``) once the render
    completes. ``mode`` / ``style_prompt`` / ``language_code`` are the
    render-only inputs stored so NARR-5's retry can re-enqueue faithfully.
    """
    return {
        'narration_status': status,
        'audio_filename': _audio_basename(conversion_id),
        'audio_mimetype': audio_mimetype,
        'duration_seconds': duration_seconds,
        'tts_model': tts_model,
        'speakers': dict(speakers or {}),
        'transcript': list(transcript or []),
        'mode': mode,
        'style_prompt': style_prompt,
        'language_code': language_code,
        'error': error,
    }


def narration_metadata(conversion):
    """Parse a Conversion's ``metadata_json`` into a dict, robust to absence/corruption.

    Returns ``{}`` when ``metadata_json`` is missing, empty, not valid JSON, or
    not a JSON object. The serve endpoint and the read-helpers build on this.
    """
    raw = getattr(conversion, 'metadata_json', None)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def narration_status(conversion):
    """The ``narration_status`` from a Conversion's metadata, ``''`` if unset/broken.

    The serve endpoint gates on ``== 'ready'``, so an unknown/missing status is
    safely non-servable.
    """
    return narration_metadata(conversion).get('narration_status') or ''


def narration_audio_filename(conversion):
    """The stored ``audio_filename`` from a Conversion's metadata, ``''`` if absent."""
    return narration_metadata(conversion).get('audio_filename') or ''


def delete_narration_audio(conversion_id):
    """Best-effort, traversal-guarded unlink of a narration's audio file.

    Resolves the path from the id (``narration_audio_path``, never user input),
    confirms it stays within ``OUTPUT_DIR``, and unlinks it if present. Returns
    ``True`` iff a file was actually removed.

    Intended to be called **post-commit** by the delete route: the Conversion
    row is already gone, so a missing / locked / out-of-tree file must never
    raise — a leftover audio file is harmless, a thrown delete is not.
    """
    file_path = narration_audio_path(conversion_id)
    try:
        real_path = os.path.realpath(file_path)
        output_real = os.path.realpath(OUTPUT_DIR)
        if not Path(real_path).is_relative_to(Path(output_real)):
            return False
        if os.path.exists(real_path):
            os.unlink(real_path)
            return True
    except Exception:
        pass
    return False
