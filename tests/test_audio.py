"""Audio transcription characterization tests.

Locks in: a successful POST with an audio file returns a JSON transcript
payload, an upstream Deepgram exception is logged + returned as 500, and a
POST with no file part returns 400.

The Deepgram SDK is mocked at the module-level ``app.deepgram_service``
singleton (see the ``mock_deepgram`` fixture).
"""
from io import BytesIO


def test_transcribe_audio_returns_transcript(authenticated_client, fixtures_dir, mock_deepgram):
    mock_deepgram.transcribe_file.return_value = 'hello world this is a test transcript'
    with open(fixtures_dir / 'sample.mp3', 'rb') as fh:
        resp = authenticated_client.post(
            '/transcribe-audio-file',
            data={
                'audio_file': (BytesIO(fh.read()), 'sample.mp3'),
                'language': 'en',
            },
            content_type='multipart/form-data',
        )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['transcript'] == 'hello world this is a test transcript'
    assert body['metadata']['language'] == 'en'
    assert body['metadata']['transcript_length'] == len('hello world this is a test transcript')
    mock_deepgram.transcribe_file.assert_called_once()


def test_transcribe_audio_handles_sdk_exception_as_500(authenticated_client, fixtures_dir, mock_deepgram):
    mock_deepgram.transcribe_file.side_effect = RuntimeError('upstream timeout')
    with open(fixtures_dir / 'sample.mp3', 'rb') as fh:
        resp = authenticated_client.post(
            '/transcribe-audio-file',
            data={
                'audio_file': (BytesIO(fh.read()), 'sample.mp3'),
                'language': 'en',
            },
            content_type='multipart/form-data',
        )
    # The route catches RuntimeError separately ("chunked transcription failed");
    # both that branch and the generic Exception branch return 500 + JSON error.
    assert resp.status_code == 500
    assert 'error' in resp.get_json()


def test_transcribe_audio_missing_file_returns_400(authenticated_client, mock_deepgram):
    resp = authenticated_client.post(
        '/transcribe-audio-file',
        data={'language': 'en'},
        content_type='multipart/form-data',
    )
    assert resp.status_code == 400
    assert resp.get_json()['error'].lower().startswith('no audio file')


def test_transcribe_audio_503_when_service_not_configured(authenticated_client):
    """If ``deepgram_service`` is None (no API key), the route returns 503."""
    import app as app_module
    original = app_module.deepgram_service
    app_module.deepgram_service = None
    try:
        resp = authenticated_client.post(
            '/transcribe-audio-file',
            data={'language': 'en'},
            content_type='multipart/form-data',
        )
    finally:
        app_module.deepgram_service = original
    assert resp.status_code == 503
