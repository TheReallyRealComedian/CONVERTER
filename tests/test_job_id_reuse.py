"""JOB-ID-REUSE — every job file carries the job mark in its name.

``Conversion.id`` is a plain ``INTEGER PRIMARY KEY`` (no AUTOINCREMENT):
SQLite hands the id of a deleted highest row straight to the next insert.
Until this sprint every job file on the shared volume was named after that
id alone, and all three reconciles matched a result to a row by it — so the
most natural correction path of the app (wrong file uploaded → pending row
deleted → right file uploaded) delivered the OLD job's result to the NEW row:
``ready``, right title, foreign content, no error.

These tests replay that path with real SQLite id reuse (every test asserts
the id really repeated — otherwise the case would test nothing) and the
tasks in-process. A task is run by replaying its ``task_queue.enqueue`` call
(``_run``): exactly what RQ would call, with exactly the arguments the web
side enqueued.

The invariant they pin: the worker reads and writes only under names that
carry the job mark; the reconcile looks only for the name of its own job.
"""
import inspect
import io
import json
import os
import sys
import threading
import uuid
import wave
from types import SimpleNamespace

import pytest
import sqlalchemy
from rq.exceptions import NoSuchJobError

import app_pkg.audio as audio_module
import tasks
from models import Conversion, db
from services import document_conversions as doc_lib
from services import narration_library as narr_lib
from services import transcription_jobs as tj

DOC_URL = '/api/document-conversions'
TRANS_URL = '/api/transcriptions'
NARR_URL = '/api/narrations'
NARRATION_TOKEN = 'narr-test-token-7c3f'

ENQUEUE_503 = 'Auftrag konnte nicht eingereiht werden. Bitte erneut versuchen.'


# --- shared plumbing ---------------------------------------------------------

@pytest.fixture
def volume(tmp_path, monkeypatch):
    """The shared volume as a tmp dir: all three namespaces under one root."""
    root = tmp_path / 'volume'
    root.mkdir()
    monkeypatch.setattr(narr_lib, 'OUTPUT_DIR', str(root))
    monkeypatch.setattr('app_pkg.narration.OUTPUT_DIR', str(root))
    monkeypatch.setattr(doc_lib, 'DOC_CONVERT_DIR', str(root / 'doc_conversions'))
    monkeypatch.setattr(tj, 'TRANSCRIPTION_DIR', str(root / 'transcriptions'))
    # ffprobe is not guaranteed on the dev box — the duration is a test input.
    monkeypatch.setattr(audio_module, 'probe_duration_seconds', lambda path: 60.0)
    return root


def _volume_files(root):
    """Every file under the volume root, as sorted relative paths."""
    found = []
    for base, _dirs, files in os.walk(root):
        for name in files:
            found.append(os.path.relpath(os.path.join(base, name), root))
    return sorted(found)


def _run(call):
    """Run an enqueued task in-process — exactly what RQ would call."""
    func, *args = call.args
    return func(*args)


def _calls(mock_redis_queue):
    return mock_redis_queue['queue'].enqueue.call_args_list


def _job(is_failed=False, is_finished=False, exc_info=None):
    return SimpleNamespace(is_failed=is_failed, is_finished=is_finished,
                           exc_info=exc_info)


def _meta(app, conversion_id):
    with app.app_context():
        return json.loads(db.session.get(Conversion, conversion_id).metadata_json)


def _delete(client, conversion_id):
    assert client.delete(f'/api/conversions/{conversion_id}').status_code == 200


def _elsewhere(fn):
    """Run ``fn`` in another thread: its own app context, its own DB session
    and connection — a second request handled at the same time."""
    box = {}

    def target():
        try:
            box['value'] = fn()
        except BaseException as exc:  # surface it in the test thread
            box['error'] = exc

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if 'error' in box:
        raise box['error']
    return box['value']


class _MidTask:
    """One-shot hook the fake backends call while a task is running — after
    it has read its source, before it writes its result. Lets a test delete
    the row of a job that is already under way."""

    hook = None

    @classmethod
    def fire(cls):
        hook, cls.hook = cls.hook, None
        if hook is not None:
            hook()


@pytest.fixture(autouse=True)
def _no_leftover_hook():
    _MidTask.hook = None
    yield
    _MidTask.hook = None


@pytest.fixture(autouse=True)
def _one_connection_pool_afterwards(app):
    """The tests here that run a second request in another thread leave more
    than one connection in the engine's pool. The rest of the suite runs on
    ONE: the migration tests (``test_lifecycle``) change the schema on one
    connection and ALTER on the next one the pool hands out — a second pooled
    connection still holds the old schema in its cache and answers
    ``duplicate column name``. Hand back a fresh pool."""
    yield
    with app.app_context():
        db.session.remove()
        db.engine.dispose()


def _row_count(app, conversion_type):
    with app.app_context():
        return Conversion.query.filter_by(conversion_type=conversion_type).count()


# --- document plumbing -------------------------------------------------------

@pytest.fixture
def doc_backend(monkeypatch):
    """EML rides the unstructured branch: the stub partition returns the
    file's own text, so a result names the source it was converted from."""
    def fake_partition(filename=None, strategy=None, paragraph_grouper=None):
        with open(filename, encoding='utf-8') as f:
            text = f.read()
        _MidTask.fire()
        return [SimpleNamespace(
            category='NarrativeText', text=text,
            metadata=SimpleNamespace(category_depth=None, page_number=None,
                                     text_as_html=None))]

    monkeypatch.setattr(sys.modules['unstructured.partition.auto'],
                        'partition', fake_partition)


def _doc_submit(client, text, filename='mail.eml'):
    return client.post(
        DOC_URL, data={'file': (io.BytesIO(text.encode('utf-8')), filename)},
        content_type='multipart/form-data')


def _doc_poll(client, conversion_id):
    return client.get(f'{DOC_URL}/{conversion_id}').get_json()


# --- transcription plumbing --------------------------------------------------

class _EchoDeepgram:
    """A Deepgram stand-in whose transcript names the audio bytes it got."""

    def __init__(self, api_key):
        pass

    def transcribe_file(self, audio_data, language):
        _MidTask.fire()
        return 'Transkript von ' + audio_data.decode('utf-8')


@pytest.fixture
def trans_backend(monkeypatch, mock_deepgram):
    monkeypatch.setenv('DEEPGRAM_API_KEY', 'worker-key')
    monkeypatch.setattr(tasks, 'DeepgramService', _EchoDeepgram)


def _trans_submit(client, text, filename='aufnahme.wav'):
    return client.post(
        TRANS_URL,
        data={'audio_file': (io.BytesIO(text.encode('utf-8')), filename),
              'language': 'de'},
        content_type='multipart/form-data')


def _trans_poll(client, conversion_id):
    return client.get(f'{TRANS_URL}/{conversion_id}').get_json()


# --- narration plumbing ------------------------------------------------------

def _wav(path, seconds, rate=8000):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b'\x00\x00' * (rate * seconds))


class _FakeTTS:
    """Renders ``len(turns)`` seconds of silence into a temp WAV OUTSIDE the
    volume — like the real renderer, which writes to the worker's /tmp."""

    scratch = None

    def __init__(self, creds):
        pass

    def synthesize_narration(self, turns, voices, **kwargs):
        path = os.path.join(_FakeTTS.scratch, f'render-{uuid.uuid4().hex}.wav')
        _wav(path, seconds=len(turns))
        return path


@pytest.fixture
def narr_backend(monkeypatch, tmp_path):
    monkeypatch.setenv('NARRATION_TOKEN', NARRATION_TOKEN)
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', '/fake/creds.json')
    scratch = tmp_path / 'worker-tmp'
    scratch.mkdir()
    monkeypatch.setattr(_FakeTTS, 'scratch', str(scratch))
    monkeypatch.setattr(tasks, 'GoogleTTSService', _FakeTTS)


def _narr_submit(client, n_turns):
    return client.post(
        NARR_URL, headers={'Authorization': f'Bearer {NARRATION_TOKEN}'},
        json={'title': f'Vertonung mit {n_turns} Turns', 'mode': 'single_speaker',
              'voices': {'Anna': 'Kore'},
              'turns': [{'speaker': 'Anna', 'text': f'Satz {i}.'}
                        for i in range(n_turns)]})


def _narr_poll(client, conversion_id):
    return client.get(f'{NARR_URL}/{conversion_id}').get_json()['metadata']


def _wav_seconds(path):
    with wave.open(str(path), 'rb') as w:
        return round(w.getnframes() / w.getframerate())


# =============================================================================
# (a) the old job's result arrives after its row was deleted
# =============================================================================

def test_a_document_result_of_a_job_deleted_mid_run_never_reaches_the_new_row(
        app, authenticated_client, mock_redis_queue, volume, doc_backend):
    """Job A is already running (source read) when its pending row is
    deleted; it finishes and writes its result. The row that takes the id
    must not get it."""
    client = authenticated_client
    mock_redis_queue['fetch'].return_value = _job()  # queued/started

    submitted = _doc_submit(client, 'Inhalt von A').get_json()
    id_a, mark_a = submitted['id'], submitted['job_id']
    _MidTask.hook = lambda: _delete(client, id_a)
    _run(_calls(mock_redis_queue)[0])    # deleted mid-run, result written after
    assert _row_count(app, 'document_conversion') == 0

    id_b = _doc_submit(client, 'Inhalt von B').get_json()['id']
    assert id_b == id_a, 'SQLite did not reuse the id — the case is not tested'

    polled = _doc_poll(client, id_b)
    assert polled['status'] == 'pending'
    assert polled['markdown'] is None

    _run(_calls(mock_redis_queue)[1])    # job B
    polled = _doc_poll(client, id_b)
    assert polled['status'] == 'ready'
    assert polled['markdown'] == 'Inhalt von B'

    # The named property: a job that was already running when its row was
    # deleted leaves ONE orphan — its result, under a mark nobody looks for.
    assert _volume_files(volume) == [f'doc_conversions/result_{mark_a}.json']


def test_a_transcription_result_of_a_job_deleted_mid_run_never_reaches_the_new_row(
        app, authenticated_client, mock_redis_queue, volume, trans_backend):
    client = authenticated_client
    mock_redis_queue['fetch'].return_value = _job()

    id_a = _trans_submit(client, 'Audio A').get_json()['id']
    _MidTask.hook = lambda: _delete(client, id_a)
    _run(_calls(mock_redis_queue)[0])
    assert _row_count(app, 'audio_transcription') == 0

    id_b = _trans_submit(client, 'Audio B').get_json()['id']
    assert id_b == id_a, 'SQLite did not reuse the id — the case is not tested'

    polled = _trans_poll(client, id_b)
    assert polled['status'] == 'pending'
    assert polled['transcript'] is None

    _run(_calls(mock_redis_queue)[1])
    polled = _trans_poll(client, id_b)
    assert polled['status'] == 'ready'
    assert polled['transcript'] == 'Transkript von Audio B'


def _run_job_of_a_deleted_row(call):
    """A job that was still QUEUED when its row was deleted. The delete took
    its source, so today it dies at its first read; before, it converted the
    file and left a result. Either way: what it does must not reach row B."""
    try:
        _run(call)
    except OSError:
        pass


def test_a_document_job_still_queued_at_delete_never_reaches_the_new_row(
        app, authenticated_client, mock_redis_queue, volume, doc_backend):
    client = authenticated_client
    mock_redis_queue['fetch'].return_value = _job()

    id_a = _doc_submit(client, 'Inhalt von A').get_json()['id']
    _delete(client, id_a)
    _run_job_of_a_deleted_row(_calls(mock_redis_queue)[0])

    id_b = _doc_submit(client, 'Inhalt von B').get_json()['id']
    assert id_b == id_a, 'SQLite did not reuse the id — the case is not tested'
    assert _doc_poll(client, id_b)['status'] == 'pending'

    _run(_calls(mock_redis_queue)[1])
    polled = _doc_poll(client, id_b)
    assert polled['status'] == 'ready'
    assert polled['markdown'] == 'Inhalt von B'


def test_a_transcription_job_still_queued_at_delete_never_reaches_the_new_row(
        app, authenticated_client, mock_redis_queue, volume, trans_backend):
    client = authenticated_client
    mock_redis_queue['fetch'].return_value = _job()

    id_a = _trans_submit(client, 'Audio A').get_json()['id']
    _delete(client, id_a)
    _run_job_of_a_deleted_row(_calls(mock_redis_queue)[0])

    id_b = _trans_submit(client, 'Audio B').get_json()['id']
    assert id_b == id_a, 'SQLite did not reuse the id — the case is not tested'
    assert _trans_poll(client, id_b)['status'] == 'pending'

    _run(_calls(mock_redis_queue)[1])
    polled = _trans_poll(client, id_b)
    assert polled['status'] == 'ready'
    assert polled['transcript'] == 'Transkript von Audio B'


# =============================================================================
# (b) the result was already there, never polled
# =============================================================================

def test_b_document_unpolled_result_of_a_deleted_row_is_not_served(
        app, authenticated_client, mock_redis_queue, volume, doc_backend):
    client = authenticated_client
    mock_redis_queue['fetch'].return_value = _job()

    id_a = _doc_submit(client, 'Inhalt von A').get_json()['id']
    _run(_calls(mock_redis_queue)[0])    # finished, never polled
    _delete(client, id_a)

    id_b = _doc_submit(client, 'Inhalt von B').get_json()['id']
    assert id_b == id_a, 'SQLite did not reuse the id — the case is not tested'

    polled = _doc_poll(client, id_b)
    assert polled['status'] == 'pending'
    assert polled['markdown'] is None
    with app.app_context():
        assert db.session.get(Conversion, id_b).content == ''


def test_b_transcription_unpolled_result_of_a_deleted_row_is_not_served(
        app, authenticated_client, mock_redis_queue, volume, trans_backend):
    client = authenticated_client
    mock_redis_queue['fetch'].return_value = _job()

    id_a = _trans_submit(client, 'Audio A').get_json()['id']
    _run(_calls(mock_redis_queue)[0])
    _delete(client, id_a)

    id_b = _trans_submit(client, 'Audio B').get_json()['id']
    assert id_b == id_a, 'SQLite did not reuse the id — the case is not tested'

    polled = _trans_poll(client, id_b)
    assert polled['status'] == 'pending'
    assert polled['transcript'] is None


# =============================================================================
# (c) B is submitted while A is still running — the case a label inside the
#     result would not catch: same id, same extension, A's ``finally`` runs
#     after B's source is on the volume.
# =============================================================================

def test_c_document_job_a_finishing_never_takes_job_bs_source(
        app, authenticated_client, mock_redis_queue, volume, doc_backend):
    client = authenticated_client
    mock_redis_queue['fetch'].return_value = _job()

    id_a = _doc_submit(client, 'Inhalt von A', filename='falsch.eml').get_json()['id']
    _delete(client, id_a)
    id_b = _doc_submit(client, 'Inhalt von B', filename='richtig.eml').get_json()['id']
    assert id_b == id_a, 'SQLite did not reuse the id — the case is not tested'

    # Job A runs to its end, ``finally`` included. Its row is gone; whether it
    # still finds its source is not the point — what it does to B's is.
    _run_job_of_a_deleted_row(_calls(mock_redis_queue)[0])

    _run(_calls(mock_redis_queue)[1])    # job B finds ITS source
    polled = _doc_poll(client, id_b)
    assert polled['status'] == 'ready'
    assert polled['markdown'] == 'Inhalt von B'


def test_c_transcription_job_a_finishing_never_takes_job_bs_source(
        app, authenticated_client, mock_redis_queue, volume, trans_backend):
    client = authenticated_client
    mock_redis_queue['fetch'].return_value = _job()

    id_a = _trans_submit(client, 'Audio A', filename='falsch.wav').get_json()['id']
    _delete(client, id_a)
    id_b = _trans_submit(client, 'Audio B', filename='richtig.wav').get_json()['id']
    assert id_b == id_a, 'SQLite did not reuse the id — the case is not tested'

    _run_job_of_a_deleted_row(_calls(mock_redis_queue)[0])

    _run(_calls(mock_redis_queue)[1])
    polled = _trans_poll(client, id_b)
    assert polled['status'] == 'ready'
    assert polled['transcript'] == 'Transkript von Audio B'


# =============================================================================
# (d) narration: A still renders when B takes its id
# =============================================================================

def test_d_narration_old_render_never_becomes_the_new_rows_audio(
        app, authenticated_client, mock_redis_queue, volume, narr_backend):
    client = authenticated_client
    mock_redis_queue['fetch'].return_value = _job()

    submitted = _narr_submit(client, n_turns=1).get_json()
    id_a, mark_a = submitted['narration_id'], submitted['job_id']
    _delete(client, id_a)
    id_b = _narr_submit(client, n_turns=3).get_json()['narration_id']
    assert id_b == id_a, 'SQLite did not reuse the id — the case is not tested'

    _run(_calls(mock_redis_queue)[0])    # A's render lands (1 s of audio)
    meta = _narr_poll(client, id_b)
    assert meta['narration_status'] == 'pending'
    assert not (volume / f'narration_{id_b}.wav').exists()

    _run(_calls(mock_redis_queue)[1])    # B's render (3 s)
    meta = _narr_poll(client, id_b)
    assert meta['narration_status'] == 'ready'
    assert meta['duration_seconds'] == 3
    assert _wav_seconds(volume / f'narration_{id_b}.wav') == 3

    served = client.get(f'{NARR_URL}/{id_b}/audio')
    assert served.status_code == 200
    assert served.data == (volume / f'narration_{id_b}.wav').read_bytes()

    # A's render is the one orphan: under A's mark, never adopted.
    assert _volume_files(volume) == [f'narration_{id_b}.wav',
                                     f'narration_jobs/render_{mark_a}.wav']


def test_d_narration_stays_pending_while_only_a_part_file_exists(
        app, authenticated_client, mock_redis_queue, volume, narr_backend,
        monkeypatch):
    """The worker copies the render onto the volume (another filesystem than
    its /tmp — a copy, not a rename). A reconcile inside that window must not
    see a servable file."""
    client = authenticated_client
    mock_redis_queue['fetch'].return_value = _job()
    nid = _narr_submit(client, n_turns=2).get_json()['narration_id']

    seen = {}
    real_replace = os.replace

    def replace_spy(src, dst):
        # The task's final rename: everything the copy wrote is on the volume
        # now, the finished name is not. Poll from inside this window.
        if not seen:
            seen['files'] = _volume_files(volume)
            seen['during'] = _narr_poll(client, nid)['narration_status']
        return real_replace(src, dst)

    monkeypatch.setattr(os, 'replace', replace_spy)
    _run(_calls(mock_redis_queue)[0])
    monkeypatch.setattr(os, 'replace', real_replace)

    assert seen, 'the task never renamed a finished file into place'
    assert seen['files'] and all(name.endswith('.part') for name in seen['files'])
    assert seen['during'] == 'pending'
    assert _narr_poll(client, nid)['narration_status'] == 'ready'


def test_d_narration_second_reconcile_after_adoption_is_a_noop(
        app, authenticated_client, mock_redis_queue, volume, narr_backend):
    client = authenticated_client
    nid = _narr_submit(client, n_turns=2).get_json()['narration_id']
    _run(_calls(mock_redis_queue)[0])

    assert _narr_poll(client, nid)['narration_status'] == 'ready'
    audio = (volume / f'narration_{nid}.wav').read_bytes()
    # Only the artifact is left — the job's own file was adopted, not copied.
    assert _volume_files(volume) == [f'narration_{nid}.wav']

    # The job has long left Redis (result TTL 500 s); a later reconcile must
    # neither ask for it nor flip the adopted narration.
    mock_redis_queue['fetch'].reset_mock()
    mock_redis_queue['fetch'].side_effect = NoSuchJobError('gone')
    meta = _narr_poll(client, nid)
    assert meta['narration_status'] == 'ready'
    assert meta['duration_seconds'] == 2
    mock_redis_queue['fetch'].assert_not_called()
    assert (volume / f'narration_{nid}.wav').read_bytes() == audio


def test_d_narration_stale_reconcile_does_not_fail_an_adopted_row(
        app, authenticated_client, mock_redis_queue, volume, narr_backend):
    """Two reconciles at once: the second one loaded the row while it was
    still ``pending``; by the time it looks, the first has adopted the WAV,
    committed ``ready``, and the job has left Redis. It must not write
    ``failed`` over the adopted narration."""
    from app_pkg.narration import reconcile_narration

    client = authenticated_client
    nid = _narr_submit(client, n_turns=2).get_json()['narration_id']
    _run(_calls(mock_redis_queue)[0])
    mock_redis_queue['fetch'].side_effect = NoSuchJobError('gone')

    with app.app_context():
        stale = db.session.get(Conversion, nid)
        assert json.loads(stale.metadata_json)['narration_status'] == 'pending'
        # The other reconcile, on its own connection: adopt + commit.
        other = _elsewhere(lambda: _narr_poll(client, nid))
        assert other['narration_status'] == 'ready'
        reconcile_narration(stale)

    meta = _meta(app, nid)
    assert meta['narration_status'] == 'ready'
    assert meta['error'] is None


def test_d_narration_reconciles_at_the_same_moment_all_end_ready(
        app, test_user, authenticated_client, mock_redis_queue, volume,
        narr_backend):
    """Six polls released together on a pending row whose render is finished
    and whose job has already left Redis: whoever loses the race for the
    render must not write ``failed``. The adoption is repeatable (hard link,
    the job name stays until ``ready`` is committed) — every poll answers
    ``ready``, the artifact carries the render."""
    nid = _narr_submit(authenticated_client, n_turns=2).get_json()['narration_id']
    _run(_calls(mock_redis_queue)[0])
    (render,) = _volume_files(volume)
    audio = (volume / render).read_bytes()
    mock_redis_queue['fetch'].side_effect = NoSuchJobError('gone')

    clients = []
    for _ in range(6):
        other = app.test_client()
        assert other.post('/login', data={
            'username': test_user['username'],
            'password': test_user['password']}).status_code == 302
        clients.append(other)

    barrier = threading.Barrier(len(clients))
    answers, errors = [], []

    def poll(client):
        try:
            barrier.wait(timeout=10)
            answers.append(_narr_poll(client, nid)['narration_status'])
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=poll, args=(c,)) for c in clients]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    assert answers == ['ready'] * len(clients)
    assert _meta(app, nid)['narration_status'] == 'ready'
    assert (volume / f'narration_{nid}.wav').read_bytes() == audio
    assert _volume_files(volume) == [f'narration_{nid}.wav']


# =============================================================================
# (e) enqueue raises: 503, no row, no source on the volume
# =============================================================================

def test_e_document_enqueue_failure_is_503_without_row_or_source(
        app, authenticated_client, mock_redis_queue, volume):
    mock_redis_queue['queue'].enqueue.side_effect = ConnectionError('redis down')
    resp = _doc_submit(authenticated_client, 'Inhalt')
    assert resp.status_code == 503
    assert resp.get_json() == {'error': ENQUEUE_503}
    assert _row_count(app, 'document_conversion') == 0
    assert _volume_files(volume) == []


def test_e_transcription_enqueue_failure_is_503_without_row_or_source(
        app, authenticated_client, mock_redis_queue, volume, trans_backend):
    mock_redis_queue['queue'].enqueue.side_effect = ConnectionError('redis down')
    resp = _trans_submit(authenticated_client, 'Audio')
    assert resp.status_code == 503
    assert resp.get_json() == {'error': ENQUEUE_503}
    assert _row_count(app, 'audio_transcription') == 0
    assert _volume_files(volume) == []


def test_e_narration_enqueue_failure_is_503_without_row(
        app, authenticated_client, mock_redis_queue, volume, narr_backend):
    mock_redis_queue['queue'].enqueue.side_effect = ConnectionError('redis down')
    resp = _narr_submit(authenticated_client, n_turns=1)
    assert resp.status_code == 503
    assert resp.get_json() == {'error': ENQUEUE_503}
    assert _row_count(app, 'audio_narration') == 0
    assert _volume_files(volume) == []


def test_e_narration_retry_enqueue_failure_leaves_the_row_failed(
        app, authenticated_client, test_user, mock_redis_queue, volume):
    stored = {'narration_status': 'failed', 'error': 'Vertonung fehlgeschlagen.',
              'job_id': 'old-job-1', 'duration_seconds': None,
              'tts_model': 'gemini-2.5-flash-tts', 'speakers': {'Anna': 'Kore'},
              'transcript': [{'speaker': 'Anna', 'text': 'Hallo.'}],
              'mode': 'single_speaker', 'style_prompt': None,
              'language_code': 'de-DE'}
    with app.app_context():
        conv = Conversion(user_id=test_user['id'], conversion_type='audio_narration',
                          title='Vertonung', content='Hallo.',
                          metadata_json=json.dumps(stored))
        db.session.add(conv)
        db.session.commit()
        nid = conv.id

    mock_redis_queue['queue'].enqueue.side_effect = ConnectionError('redis down')
    resp = authenticated_client.post(f'{NARR_URL}/{nid}/retry')
    assert resp.status_code == 503
    assert resp.get_json() == {'error': ENQUEUE_503}
    assert _meta(app, nid) == stored   # still failed, old job, old error


# =============================================================================
# (f) delete clears the job's own files
# =============================================================================

def test_f_delete_pending_document_removes_its_source(
        app, authenticated_client, mock_redis_queue, volume, doc_backend):
    client = authenticated_client
    cid = _doc_submit(client, 'Inhalt').get_json()['id']
    assert len(_volume_files(volume)) == 1   # the source
    _delete(client, cid)
    assert _volume_files(volume) == []


def test_f_delete_pending_document_removes_its_unpolled_result(
        app, authenticated_client, mock_redis_queue, volume, doc_backend):
    client = authenticated_client
    cid = _doc_submit(client, 'Inhalt').get_json()['id']
    _run(_calls(mock_redis_queue)[0])        # result written, source consumed
    assert len(_volume_files(volume)) == 1   # the result
    _delete(client, cid)
    assert _volume_files(volume) == []


def test_f_delete_pending_transcription_removes_source_and_result(
        app, authenticated_client, mock_redis_queue, volume, trans_backend):
    client = authenticated_client
    cid = _trans_submit(client, 'Audio').get_json()['id']
    assert len(_volume_files(volume)) == 1
    _delete(client, cid)
    assert _volume_files(volume) == []

    cid = _trans_submit(client, 'Audio zwei').get_json()['id']
    _run(_calls(mock_redis_queue)[1])
    assert len(_volume_files(volume)) == 1
    _delete(client, cid)
    assert _volume_files(volume) == []


def test_f_delete_pending_narration_removes_its_unadopted_render(
        app, authenticated_client, mock_redis_queue, volume, narr_backend):
    client = authenticated_client
    nid = _narr_submit(client, n_turns=1).get_json()['narration_id']
    _run(_calls(mock_redis_queue)[0])        # rendered, never polled
    assert len(_volume_files(volume)) == 1
    _delete(client, nid)
    assert _volume_files(volume) == []


def test_f_delete_ready_narration_removes_the_artifact(
        app, authenticated_client, mock_redis_queue, volume, narr_backend):
    client = authenticated_client
    nid = _narr_submit(client, n_turns=1).get_json()['narration_id']
    _run(_calls(mock_redis_queue)[0])
    assert _narr_poll(client, nid)['narration_status'] == 'ready'
    assert (volume / f'narration_{nid}.wav').exists()
    _delete(client, nid)
    assert _volume_files(volume) == []


def test_f_delete_never_follows_a_crafted_job_mark(
        app, authenticated_client, test_user, volume):
    """``POST /api/conversions`` takes a client-written metadata bag, so
    ``job_id`` / ``source_format`` of a row are not trustworthy: the cleanup
    must never turn them into a path outside the job's namespace."""
    client = authenticated_client
    (volume / 'doc_conversions').mkdir()
    victim = volume / 'opfer.json'
    victim.write_text('{}', encoding='utf-8')
    other = volume / 'narration_7.wav'
    other.write_bytes(b'RIFF')

    for conversion_type in ('document_conversion', 'audio_transcription',
                            'audio_narration'):
        for job_id, ext in (('../opfer', 'json'), ('x/../../opfer', 'json'),
                            ('ok', '/../../opfer.json'), ('', ''), (None, None),
                            (7, 'wav'), ('../narration_7', 'wav')):
            with app.app_context():
                conv = Conversion(
                    user_id=test_user['id'], conversion_type=conversion_type,
                    title='x', content='y',
                    metadata_json=json.dumps({'job_id': job_id,
                                              'source_format': ext}))
                db.session.add(conv)
                db.session.commit()
                cid = conv.id
            _delete(client, cid)
    assert victim.exists()
    assert other.exists()


# =============================================================================
# reconcile: own job finished, own file missing
# =============================================================================

def test_document_finished_job_without_result_fails_instead_of_pending_forever(
        app, authenticated_client, mock_redis_queue, volume, doc_backend):
    client = authenticated_client
    cid = _doc_submit(client, 'Inhalt').get_json()['id']
    mock_redis_queue['fetch'].return_value = _job(is_finished=True)
    polled = _doc_poll(client, cid)
    assert polled['status'] == 'failed'
    assert polled['error'] == 'Ergebnis nicht auffindbar.'


def test_transcription_finished_job_without_result_fails(
        app, authenticated_client, mock_redis_queue, volume, trans_backend):
    client = authenticated_client
    cid = _trans_submit(client, 'Audio').get_json()['id']
    mock_redis_queue['fetch'].return_value = _job(is_finished=True)
    polled = _trans_poll(client, cid)
    assert polled['status'] == 'failed'
    assert polled['error'] == 'Ergebnis nicht auffindbar.'


def test_narration_finished_job_without_audio_fails(
        app, authenticated_client, mock_redis_queue, volume, narr_backend):
    client = authenticated_client
    nid = _narr_submit(client, n_turns=1).get_json()['narration_id']
    mock_redis_queue['fetch'].return_value = _job(is_finished=True)
    meta = _narr_poll(client, nid)
    assert meta['narration_status'] == 'failed'
    assert meta['error'] == 'Ergebnis nicht auffindbar.'


def test_document_result_written_between_file_check_and_job_fetch_is_ready(
        app, authenticated_client, mock_redis_queue, volume, doc_backend):
    """The race the "finished without a file" rule must not lose: no result at
    the first look, the worker writes it and finishes, THEN the reconcile
    reads the job as finished. The worker writes before it returns, so a
    second look after the fetch settles it."""
    client = authenticated_client
    cid = _doc_submit(client, 'Inhalt spät').get_json()['id']

    def finish_during_fetch(*args, **kwargs):
        _run(_calls(mock_redis_queue)[0])
        return _job(is_finished=True)

    mock_redis_queue['fetch'].side_effect = finish_during_fetch
    polled = _doc_poll(client, cid)
    assert polled['status'] == 'ready'
    assert polled['markdown'] == 'Inhalt spät'


@pytest.mark.parametrize('job_state', ['finished', 'gone'])
def test_document_stale_reconcile_does_not_fail_a_ready_row(
        app, authenticated_client, mock_redis_queue, volume, doc_backend,
        job_state):
    """Two reconciles at once (the second loaded the row while ``pending``):
    the first read the result, committed ``ready`` and discarded the file.
    The second finds no file and a job that is finished — or long gone from
    Redis (result TTL 500 s). It must read the row again instead of writing
    ``failed`` over the result."""
    from app_pkg.document_api import reconcile_document_conversion

    client = authenticated_client
    cid = _doc_submit(client, 'Inhalt').get_json()['id']
    _run(_calls(mock_redis_queue)[0])
    if job_state == 'finished':
        mock_redis_queue['fetch'].return_value = _job(is_finished=True)
    else:
        mock_redis_queue['fetch'].side_effect = NoSuchJobError('gone')

    with app.app_context():
        stale = db.session.get(Conversion, cid)
        assert json.loads(stale.metadata_json)['doc_status'] == 'pending'
        other = _elsewhere(lambda: _doc_poll(client, cid))
        assert other['status'] == 'ready'
        reconcile_document_conversion(stale)

    assert _meta(app, cid)['doc_status'] == 'ready'
    with app.app_context():
        assert db.session.get(Conversion, cid).content == 'Inhalt'


@pytest.mark.parametrize('job_state', ['finished', 'gone'])
def test_transcription_stale_reconcile_does_not_fail_a_ready_row(
        app, authenticated_client, mock_redis_queue, volume, trans_backend,
        job_state):
    from app_pkg.audio import reconcile_transcription

    client = authenticated_client
    cid = _trans_submit(client, 'Audio').get_json()['id']
    _run(_calls(mock_redis_queue)[0])
    if job_state == 'finished':
        mock_redis_queue['fetch'].return_value = _job(is_finished=True)
    else:
        mock_redis_queue['fetch'].side_effect = NoSuchJobError('gone')

    with app.app_context():
        stale = db.session.get(Conversion, cid)
        other = _elsewhere(lambda: _trans_poll(client, cid))
        assert other['status'] == 'ready'
        reconcile_transcription(stale)

    assert _meta(app, cid)['transcription_status'] == 'ready'


# =============================================================================
# the mark is the only thing that reaches a file name — and only if it is one
# =============================================================================

@pytest.mark.parametrize('bad', ['../opfer', 'a/b', 'a.b', '', None, 7, 'x' * 65,
                                 'ä', 'a b', '..'])
def test_path_helpers_refuse_anything_that_is_not_a_mark(volume, bad):
    """``job_id`` is read back from ``metadata_json``, which a client can
    write (``POST /api/conversions`` takes a metadata bag). A value that is
    not a mark never becomes a path; the best-effort cleanups stay silent."""
    for helper in (doc_lib.doc_result_path, tj.transcription_result_path,
                   narr_lib.narration_job_audio_path):
        with pytest.raises(ValueError):
            helper(bad)
    for helper in (doc_lib.doc_source_path, tj.transcription_source_path):
        with pytest.raises(ValueError):
            helper(bad, 'pdf')
        with pytest.raises(ValueError):
            helper('ok-mark', '/../x')
    assert doc_lib.read_result_file(bad) is None
    assert tj.read_result_file(bad) is None
    doc_lib.discard_job_files(bad, source_ext='pdf')
    tj.discard_job_files(bad, source_ext='/../x')
    narr_lib.discard_narration_job_audio(bad)
    assert narr_lib.adopt_narration_audio(bad, 1) is False


# =============================================================================
# sentinels
# =============================================================================

def _is_uuid4(value):
    try:
        return str(uuid.UUID(value, version=4)) == value
    except (ValueError, TypeError, AttributeError):
        return False


def test_sentinel_enqueue_job_id_is_the_mark_in_the_metadata(
        app, authenticated_client, test_user, mock_redis_queue, volume,
        doc_backend, trans_backend, narr_backend):
    """The mark is created by the WEB side and handed to RQ as its job id —
    one value in the response, in the metadata, in ``enqueue(job_id=…)`` and
    as the task's first argument. Never the id RQ would have invented."""
    client = authenticated_client

    def check(body, conversion_id, call):
        mark = body['job_id']
        assert _is_uuid4(mark)
        assert _meta(app, conversion_id)['job_id'] == mark
        assert call.kwargs['job_id'] == mark
        assert call.args[1] == mark
        return mark

    body = _doc_submit(client, 'Inhalt').get_json()
    doc_mark = check(body, body['id'], _calls(mock_redis_queue)[0])

    body = _trans_submit(client, 'Audio').get_json()
    trans_mark = check(body, body['id'], _calls(mock_redis_queue)[1])

    body = _narr_submit(client, n_turns=1).get_json()
    nid = body['narration_id']
    narr_mark = check(body, nid, _calls(mock_redis_queue)[2])

    # Retry: a NEW mark, same mechanics.
    with app.app_context():
        conv = db.session.get(Conversion, nid)
        meta = json.loads(conv.metadata_json)
        meta['narration_status'] = 'failed'
        conv.metadata_json = json.dumps(meta)
        db.session.commit()
    body = client.post(f'{NARR_URL}/{nid}/retry').get_json()
    retry_mark = check(body, nid, _calls(mock_redis_queue)[3])

    assert len({doc_mark, trans_mark, narr_mark, retry_mark}) == 4


def test_sentinel_rq_reads_the_mark_as_its_job_id_and_the_task_still_gets_it(
        app, authenticated_client, mock_redis_queue, volume, doc_backend,
        trans_backend, narr_backend):
    """The tasks' first parameter is NAMED ``job_id`` and ``enqueue`` has a
    ``job_id=`` option of its own. Run the three captured calls through RQ's
    real argument parsing (no Redis): RQ takes the keyword as the job id, the
    positional mark stays the task's first argument, and nothing is left over
    as a task kwarg. Belongs on the pinned rq (container run)."""
    from rq import Queue

    client = authenticated_client
    _doc_submit(client, 'Inhalt')
    _trans_submit(client, 'Audio')
    _narr_submit(client, n_turns=1)

    for call in _calls(mock_redis_queue):
        mark = call.kwargs['job_id']
        parsed = Queue.parse_args(*call.args, **call.kwargs)
        task_args, task_kwargs = parsed[-2], parsed[-1]
        assert tuple(task_args) == tuple(call.args[1:])
        assert task_args[0] == mark
        assert not task_kwargs
        # exactly two places carry the mark: RQ's job id and the task args
        assert [item for item in parsed if item == mark] == [mark]
        inspect.signature(call.args[0]).bind(*task_args)   # the task accepts them


def test_sentinel_job_files_carry_the_mark_and_never_the_row_id(
        app, authenticated_client, mock_redis_queue, volume, doc_backend,
        trans_backend, narr_backend):
    """What actually lands on the volume: every job file is named after the
    mark; the only id-named file is the narration ARTIFACT after adoption."""
    client = authenticated_client

    body = _doc_submit(client, 'Inhalt').get_json()
    mark = body['job_id']
    assert _volume_files(volume) == [f'doc_conversions/source_{mark}.eml']
    _run(_calls(mock_redis_queue)[0])
    assert _volume_files(volume) == [f'doc_conversions/result_{mark}.json']
    assert _doc_poll(client, body['id'])['status'] == 'ready'
    assert _volume_files(volume) == []

    body = _trans_submit(client, 'Audio').get_json()
    mark = body['job_id']
    assert _volume_files(volume) == [f'transcriptions/source_{mark}.wav']
    _run(_calls(mock_redis_queue)[1])
    assert _volume_files(volume) == [f'transcriptions/result_{mark}.json']
    assert _trans_poll(client, body['id'])['status'] == 'ready'
    assert _volume_files(volume) == []

    body = _narr_submit(client, n_turns=1).get_json()
    mark, nid = body['job_id'], body['narration_id']
    assert _volume_files(volume) == []
    _run(_calls(mock_redis_queue)[2])
    (job_file,) = _volume_files(volume)
    assert mark in job_file
    assert os.path.dirname(job_file), 'the job WAV must not sit next to the artifacts'
    assert not os.path.basename(job_file).startswith('narration_')
    assert _narr_poll(client, nid)['narration_status'] == 'ready'
    assert _volume_files(volume) == [f'narration_{nid}.wav']


# The narration ARTIFACT is the one file that stays named after the row: it
# is what a ready row serves, written only by the web side's adoption.
_ARTIFACT_HELPERS = {'narration_audio_path', '_audio_basename',
                     'delete_narration_audio', 'build_narration_metadata',
                     'adopt_narration_audio'}


def test_sentinel_no_job_path_helper_takes_a_conversion_id():
    """No path helper of the three job modules and no task is keyed on the
    row id anymore (the narration artifact excepted, by name)."""
    offenders = []
    for module in (doc_lib, tj, narr_lib, tasks):
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if func.__module__ != module.__name__:
                continue
            if name in _ARTIFACT_HELPERS and module is narr_lib:
                continue
            if 'conversion_id' in inspect.signature(func).parameters:
                offenders.append(f'{module.__name__}.{name}')
    assert offenders == []

    for task in (tasks.convert_document_task, tasks.transcribe_audio_task,
                 tasks.generate_narration_task):
        assert next(iter(inspect.signature(task).parameters)) == 'job_id'


def test_sentinel_conversion_id_is_still_not_autoincrement(app):
    """The premise of this file. If the table ever becomes AUTOINCREMENT, ids
    stop repeating and the id-reuse tests above would fail on their own
    ``id_b == id_a`` guard — this names the reason."""
    with app.app_context():
        names = sqlalchemy.inspect(db.engine).get_table_names()
        with db.engine.connect() as conn:
            ddl = conn.execute(sqlalchemy.text(
                "SELECT sql FROM sqlite_master WHERE name = 'conversion'")).scalar()
    assert 'conversion' in names
    assert 'AUTOINCREMENT' not in ddl.upper()
