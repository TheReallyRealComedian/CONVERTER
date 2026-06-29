"""NARR-2 pure persistence-helper tests for ``services.narration_library``.

No Flask context, no SDK — these exercise the pure helpers that turn a rendered
narration into a first-class library element: the transcript→Markdown content,
the deterministic audio path, the ``metadata_json`` contract builder, and the
robust read-helpers. (Serve + delete-cleanup are integration-tested separately
in ``test_narration_serve.py``.)
"""
from types import SimpleNamespace

import pytest

from app_pkg.config import OUTPUT_DIR
from services.narration_library import (
    DEFAULT_AUDIO_MIMETYPE,
    NARRATION_STATUS_FAILED,
    NARRATION_STATUS_PENDING,
    NARRATION_STATUS_READY,
    build_narration_metadata,
    narration_audio_filename,
    narration_audio_path,
    narration_metadata,
    narration_status,
    narration_to_markdown,
)


# --- narration_to_markdown ----------------------------------------------------

def test_to_markdown_two_speakers_labels_both_and_keeps_all_text():
    turns = [
        {'speaker': 'Anna', 'text': 'Erstens das hier.'},
        {'speaker': 'Ben', 'text': 'Und zweitens das.'},
        {'speaker': 'Anna', 'text': 'Genau, abschließend noch.'},
    ]
    md = narration_to_markdown(turns)

    assert md  # non-empty (content is NOT NULL)
    assert '**Anna:**' in md
    assert '**Ben:**' in md
    # All turn text survives verbatim.
    assert 'Erstens das hier.' in md
    assert 'Und zweitens das.' in md
    assert 'Genau, abschließend noch.' in md
    # Blank-line separated blocks.
    assert md == (
        '**Anna:** Erstens das hier.\n\n'
        '**Ben:** Und zweitens das.\n\n'
        '**Anna:** Genau, abschließend noch.'
    )


def test_to_markdown_single_speaker_drops_labels():
    turns = [
        {'speaker': 'Anna', 'text': 'Absatz eins.'},
        {'speaker': 'Anna', 'text': 'Absatz zwei.'},
    ]
    md = narration_to_markdown(turns)

    assert md
    assert '**Anna:**' not in md  # one voice → no label noise
    assert md == 'Absatz eins.\n\nAbsatz zwei.'


def test_to_markdown_preserves_markdown_and_html_specials_verbatim():
    # Read content flows through the shared renderer downstream — no mangling here.
    turns = [
        {'speaker': 'Anna', 'text': 'Preis *war* 5$ & <b>fett</b> #hash _x_'},
        {'speaker': 'Ben', 'text': 'Antwort: a < b > c'},
    ]
    md = narration_to_markdown(turns)

    assert 'Preis *war* 5$ & <b>fett</b> #hash _x_' in md
    assert 'Antwort: a < b > c' in md


def test_to_markdown_skips_blank_and_non_dict_turns():
    turns = [
        {'speaker': 'Anna', 'text': 'Bleibt.'},
        {'speaker': 'Ben', 'text': '   '},      # blank text → skipped
        {'speaker': 'Ben', 'text': None},        # None text → skipped
        'not-a-dict',                            # ignored
        {'speaker': 'Ben', 'text': 'Auch da.'},
    ]
    md = narration_to_markdown(turns)

    assert md == '**Anna:** Bleibt.\n\n**Ben:** Auch da.'


@pytest.mark.parametrize('bad', [[], None, 'x', [{'speaker': 'A', 'text': '  '}]])
def test_to_markdown_degenerate_input_is_empty_string(bad):
    assert narration_to_markdown(bad) == ''


# --- narration_audio_path -----------------------------------------------------

def test_audio_path_is_deterministic_and_under_output_dir():
    p1 = narration_audio_path(42)
    p2 = narration_audio_path(42)

    assert p1 == p2  # deterministic
    assert p1 == f'{OUTPUT_DIR}/narration_42.wav'
    assert p1.startswith(OUTPUT_DIR)
    assert p1.endswith('narration_42.wav')


def test_audio_path_differs_per_id():
    assert narration_audio_path(1) != narration_audio_path(2)


# --- build_narration_metadata -------------------------------------------------

def test_build_metadata_default_shape_is_pending():
    meta = build_narration_metadata(7)

    assert meta['narration_status'] == NARRATION_STATUS_PENDING
    assert meta['audio_filename'] == 'narration_7.wav'  # matches narration_audio_path
    assert meta['audio_mimetype'] == DEFAULT_AUDIO_MIMETYPE
    assert meta['duration_seconds'] is None
    assert meta['tts_model'] is None
    assert meta['speakers'] == {}
    assert meta['transcript'] == []
    assert meta['error'] is None


def test_build_metadata_ready_carries_audio_fields():
    meta = build_narration_metadata(
        9,
        status=NARRATION_STATUS_READY,
        tts_model='gemini-2.5-flash-tts',
        speakers={'Anna': 'Kore', 'Ben': 'Puck'},
        transcript=[{'speaker': 'Anna', 'text': 'Hi'}],
        duration_seconds=123,
    )

    assert meta['narration_status'] == NARRATION_STATUS_READY
    assert meta['audio_filename'] == 'narration_9.wav'
    assert meta['tts_model'] == 'gemini-2.5-flash-tts'
    assert meta['speakers'] == {'Anna': 'Kore', 'Ben': 'Puck'}
    assert meta['transcript'] == [{'speaker': 'Anna', 'text': 'Hi'}]
    assert meta['duration_seconds'] == 123
    assert meta['error'] is None


def test_build_metadata_audio_filename_matches_audio_path():
    meta = build_narration_metadata(55)
    assert narration_audio_path(55).endswith(meta['audio_filename'])


# --- read-helpers (robust metadata_json parsing) ------------------------------

def _conv(metadata_json):
    """A minimal Conversion stand-in carrying just ``metadata_json``."""
    return SimpleNamespace(metadata_json=metadata_json)


@pytest.mark.parametrize('status', [
    NARRATION_STATUS_READY,
    NARRATION_STATUS_PENDING,
    NARRATION_STATUS_FAILED,
])
def test_narration_status_reads_each_state(status):
    import json
    conv = _conv(json.dumps({'narration_status': status}))
    assert narration_status(conv) == status


def test_narration_status_broken_json_is_empty():
    assert narration_status(_conv('{not valid json')) == ''


def test_narration_status_missing_metadata_is_empty():
    assert narration_status(_conv(None)) == ''
    assert narration_status(_conv('')) == ''


def test_narration_status_non_object_json_is_empty():
    # A JSON array is valid JSON but not the expected object → defaults.
    assert narration_status(_conv('[1, 2, 3]')) == ''


def test_narration_audio_filename_reads_value_and_defaults():
    import json
    conv = _conv(json.dumps({'audio_filename': 'narration_3.wav'}))
    assert narration_audio_filename(conv) == 'narration_3.wav'
    assert narration_audio_filename(_conv('broken')) == ''
    assert narration_audio_filename(_conv(None)) == ''


def test_narration_metadata_returns_full_dict_or_empty():
    import json
    payload = {'narration_status': 'ready', 'duration_seconds': 12}
    assert narration_metadata(_conv(json.dumps(payload))) == payload
    assert narration_metadata(_conv('nope')) == {}
    assert narration_metadata(_conv(None)) == {}
