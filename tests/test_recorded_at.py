"""MCP1 Phase 2 — recorded_at capture.

Two halves: pure unit tests for parse_recorded_at_from_filename (the
conservative recorder-filename date parser), and integration tests that the
POST /api/conversions handler writes metadata.recorded_at + recorded_at_source
additively, with the client value taking precedence over the filename.
"""
import pytest

from app_pkg.library import parse_recorded_at_from_filename


# --- Parser: positive patterns (the six sprint variants + a letter-prefix) ---

@pytest.mark.parametrize('name,expected', [
    ('20260612.mp3', '2026-06-12T00:00:00+02:00'),
    ('2026-06-12.mp3', '2026-06-12T00:00:00+02:00'),
    ('REC_20260612_1430.m4a', '2026-06-12T14:30:00+02:00'),
    ('20260612-143005.wav', '2026-06-12T14:30:05+02:00'),
    ('2026-06-12 14.30.mp3', '2026-06-12T14:30:00+02:00'),
    ('2026_06_12T14_30.m4a', '2026-06-12T14:30:00+02:00'),
    ('VOICE-2026-06-12.mp3', '2026-06-12T00:00:00+02:00'),  # letter prefix ignored
])
def test_parser_positive(name, expected):
    dt = parse_recorded_at_from_filename(name)
    assert dt is not None
    assert dt.isoformat() == expected  # tz-aware, offset carried


def test_parser_dst_offset_flips_in_winter():
    """Same parser, January → CET (+01:00), not CEST (+02:00)."""
    dt = parse_recorded_at_from_filename('20260115.mp3')
    assert dt.isoformat() == '2026-01-15T00:00:00+01:00'


# --- Parser: the negative examples from the sprint must all be None ---

@pytest.mark.parametrize('name', [
    'Besprechung.mp3',
    'audio (1).mp3',
    '12345678.mp3',        # 1234 not a year, 56 not a month
    'New Recording 7.m4a',
])
def test_parser_negative_sprint_examples(name):
    assert parse_recorded_at_from_filename(name) is None


# --- Parser: no false positives on out-of-range / impossible / blob dates ---

@pytest.mark.parametrize('name', [
    '20261301.mp3',        # month 13
    '20260640.mp3',        # day 40
    '20260230.mp3',        # Feb 30 — impossible calendar date
    '19991231.mp3',        # year < 2000
    '21010101.mp3',        # year > 2100
    '2026061234.mp3',      # 10-digit blob — anchors reject embedded date
])
def test_parser_rejects_invalid_dates(name):
    assert parse_recorded_at_from_filename(name) is None


def test_parser_ambiguous_two_distinct_dates_returns_none():
    """Two different valid dates in one name → ambiguous → None (conservative)."""
    assert parse_recorded_at_from_filename('2026-06-12_backup_2025-01-01.mp3') is None


def test_parser_non_string_returns_none():
    assert parse_recorded_at_from_filename(None) is None
    assert parse_recorded_at_from_filename(20260612) is None  # int, not str


# --- Integration: POST /api/conversions recorded_at capture ---

def _post(client, **extra):
    payload = {'conversion_type': 'audio_transcription', 'title': 'X', 'content': 'body'}
    payload.update(extra)
    return client.post('/api/conversions', json=payload)


def test_create_captures_client_iso_recorded_at(authenticated_client):
    resp = _post(authenticated_client, recorded_at='2026-06-12T14:30:00+02:00')
    assert resp.status_code == 201
    md = resp.get_json()['metadata']
    assert md['recorded_at'] == '2026-06-12T12:30:00+00:00'  # normalized to UTC
    assert md['recorded_at_source'] == 'client'


def test_create_captures_client_epoch_ms_recorded_at(authenticated_client):
    resp = _post(authenticated_client, recorded_at=1700000000000)  # 2023-11-14T22:13:20Z
    assert resp.status_code == 201
    md = resp.get_json()['metadata']
    assert md['recorded_at'] == '2023-11-14T22:13:20+00:00'
    assert md['recorded_at_source'] == 'client'


def test_create_captures_recorded_at_from_filename(authenticated_client):
    resp = _post(authenticated_client, source_filename='REC_20260612_1430.m4a')
    assert resp.status_code == 201
    md = resp.get_json()['metadata']
    assert md['recorded_at'] == '2026-06-12T14:30:00+02:00'  # Berlin offset kept
    assert md['recorded_at_source'] == 'filename'


def test_create_client_recorded_at_beats_filename(authenticated_client):
    resp = _post(
        authenticated_client,
        recorded_at='2026-06-12T14:30:00+02:00',
        source_filename='REC_20200101_0000.m4a',  # different date — must lose
    )
    md = resp.get_json()['metadata']
    assert md['recorded_at'] == '2026-06-12T12:30:00+00:00'
    assert md['recorded_at_source'] == 'client'


def test_create_without_any_recorded_at_source(authenticated_client):
    resp = _post(authenticated_client, source_filename='Besprechung.mp3')
    assert resp.status_code == 201  # no crash
    md = resp.get_json()['metadata']
    assert 'recorded_at' not in md
    assert 'recorded_at_source' not in md


def test_create_respects_client_preset_metadata_recorded_at(authenticated_client):
    """A recorded_at already in the metadata bag is left untouched and gets no
    auto source tag — the filename value must NOT override it."""
    resp = _post(
        authenticated_client,
        metadata={'recorded_at': '2099-01-01T00:00:00+00:00', 'foo': 'bar'},
        source_filename='REC_20260612_1430.m4a',
    )
    md = resp.get_json()['metadata']
    assert md['recorded_at'] == '2099-01-01T00:00:00+00:00'
    assert 'recorded_at_source' not in md
    assert md['foo'] == 'bar'


def test_create_unparseable_client_recorded_at_falls_back_to_filename(authenticated_client):
    """Garbage client recorded_at is ignored (no 400, additive) and the
    filename parser still gets a shot."""
    resp = _post(
        authenticated_client,
        recorded_at='not-a-date',
        source_filename='REC_20260612_1430.m4a',
    )
    assert resp.status_code == 201
    md = resp.get_json()['metadata']
    assert md['recorded_at'] == '2026-06-12T14:30:00+02:00'
    assert md['recorded_at_source'] == 'filename'


def test_create_preserves_existing_metadata_fields(authenticated_client):
    resp = _post(
        authenticated_client,
        metadata={'src': 'deepgram', 'lang': 'de'},
        recorded_at=1700000000000,
    )
    md = resp.get_json()['metadata']
    assert md['src'] == 'deepgram'
    assert md['lang'] == 'de'
    assert md['recorded_at'] == '2023-11-14T22:13:20+00:00'
    assert md['recorded_at_source'] == 'client'


def test_create_non_dict_metadata_is_coerced_to_empty(authenticated_client):
    resp = _post(
        authenticated_client,
        metadata=['not', 'a', 'dict'],
        recorded_at=1700000000000,
    )
    assert resp.status_code == 201  # no crash on the bad metadata
    md = resp.get_json()['metadata']
    assert md['recorded_at'] == '2023-11-14T22:13:20+00:00'
