"""SEC-REDIS-AUTH (F-7) — Redis wants a password, RQ runs on ONE JSON serializer.

RQ's default serializer is pickle: whoever can write to Redis could hand the
worker a job that executes code on unpickling. ``RQ_SERIALIZER`` (JSON) is set
at exactly three touch points — the web Queue, the worker's Queue + Worker
(``build_worker``) and the web side's job read (``app.fetch_job``, used by all
three reconciles). These tests pin that, the Compose wiring, and that what
travels over the queue survives JSON.

``assert_rq_json_roundtrip`` is imported by the four enqueue-site tests
(narration create + retry, transcription, document conversion).

Measured against a real Redis 8.4 with rq 2.8.0 (sprint report): a serializer
mismatch does NOT raise on ``Job.fetch`` — RQ swallows the undecodable ``meta``
into ``{'unserialized': …}``. A Queue or Worker left on pickle fails every job
loudly with ``DeserializationError`` instead. The characterization test at the
bottom pins the no-raise half, because the reconciles would turn a raise there
into a silent eternal ``pending`` (their catch-all only logs a warning).
"""
from pathlib import Path
from unittest.mock import patch

import pytest
from redis import Redis
from rq.exceptions import DeserializationError
from rq.job import Job
from rq.serializers import DefaultSerializer, JSONSerializer

import app as app_module
import tasks
import worker
from app_pkg.config import RQ_SERIALIZER
from services.document_conversions import (PROVENANCE_DETERMINISTIC,
                                           UNIT_DOCUMENT, build_result_payload,
                                           degradation)

REPO = Path(__file__).resolve().parent.parent


def rq_json_roundtrip(obj):
    return RQ_SERIALIZER.loads(RQ_SERIALIZER.dumps(obj))


def assert_rq_json_roundtrip(call):
    """One ``task_queue.enqueue`` call survives the JSON serializer unchanged.

    RQ stores the task by name plus ``(args, kwargs)`` as job data and ``meta``
    on its own. Strict equality after the round trip catches every JSON loss:
    a tuple comes back as a list, int dict keys as strings, and bytes /
    datetime / set do not serialize at all.
    """
    func, *args = call.args
    assert f'{func.__module__}.{func.__qualname__}'.startswith('tasks.')
    assert rq_json_roundtrip(args) == args
    meta = call.kwargs['meta']
    assert rq_json_roundtrip(meta) == meta
    assert isinstance(call.kwargs['job_timeout'], int)


# --- the one serializer, the three touch points ---------------------------------

def test_rq_serializer_is_json():
    assert RQ_SERIALIZER is JSONSerializer


def test_web_queue_uses_rq_serializer():
    assert app_module.task_queue.serializer is RQ_SERIALIZER


def test_build_worker_uses_rq_serializer(monkeypatch):
    # rq's Worker talks to Redis in __init__ (CLIENT SETNAME/LIST) — capture
    # its arguments instead; the Queues are real (they don't connect).
    captured = {}
    monkeypatch.setattr(worker, 'Worker',
                        lambda queues, **kw: captured.update(queues=queues, **kw))
    conn = Redis.from_url('redis://localhost:6379')
    worker.build_worker(conn)
    assert captured['serializer'] is RQ_SERIALIZER
    assert captured['connection'] is conn
    assert [q.name for q in captured['queues']] == ['default']
    assert all(q.serializer is RQ_SERIALIZER for q in captured['queues'])


def test_fetch_job_pins_serializer():
    with patch.object(app_module.Job, 'fetch') as fetch:
        app_module.fetch_job('job-abc')
    fetch.assert_called_once_with('job-abc', connection=app_module.redis_conn,
                                  serializer=RQ_SERIALIZER)


def test_rq_plumbing_only_in_app_and_worker():
    """No other module builds a Queue/Worker or reads a job on its own."""
    sources = [REPO / 'app.py', REPO / 'worker.py', REPO / 'tasks.py',
               *(REPO / 'app_pkg').rglob('*.py'), *(REPO / 'services').rglob('*.py')]
    hits = {}
    for path in sources:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for needle in ('Job.fetch(', 'Queue(', 'Worker('):
                if needle in line:
                    hits[f'{path.relative_to(REPO)}:{lineno}'] = line.strip()
    assert sorted(hits) and {k.split(':')[0] for k in hits} == {'app.py', 'worker.py'}
    assert all('serializer=RQ_SERIALIZER' in line for line in hits.values()), hits


# --- Compose: password from .env, fail-closed everywhere ------------------------

def test_compose_redis_requires_password():
    compose_file = REPO / 'docker-compose.yml'
    if not compose_file.exists():
        pytest.skip('docker-compose.yml not shipped alongside the tests')
    yaml = pytest.importorskip('yaml')
    services = yaml.safe_load(compose_file.read_text())['services']

    redis = services['redis']
    assert redis['image'] == 'redis:8.4-alpine'
    assert 'ports' not in redis
    assert any(e.startswith('REDIS_PASSWORD=${REDIS_PASSWORD:?')
               for e in redis['environment'])
    command = ' '.join(redis['command'])
    assert '--requirepass' in command
    # Through the entrypoint, or Redis runs as root (it only drops privileges
    # when its first argument is redis-server).
    assert 'exec docker-entrypoint.sh redis-server' in command
    # redis-cli exits 0 on NOAUTH — only the grep turns the reply into a verdict.
    assert 'grep -q PONG' in ' '.join(redis['healthcheck']['test'])

    for name in ('markdown-converter', 'worker'):
        env = services[name]['environment']
        urls = [e for e in env if e.startswith('REDIS_URL=')]
        assert urls == ['REDIS_URL=redis://:${REDIS_PASSWORD:?REDIS_PASSWORD fehlt in .env}'
                        '@redis:6379/0'], name
        # The worker additionally waits for the mineru launcher (SEC-SOCKET,
        # pinned in test_mineru_launcher) — Redis is checked on its own.
        assert services[name]['depends_on']['redis'] == {
            'condition': 'service_healthy'}, name


# --- what travels besides the enqueue args --------------------------------------

def test_update_job_stage_meta_roundtrips(monkeypatch):
    class _CurrentJob:
        meta = {'user_id': 7}
        saved = None

        def save_meta(self):
            self.saved = RQ_SERIALIZER.dumps(self.meta)

    job = _CurrentJob()
    monkeypatch.setattr(tasks, 'get_current_job', lambda: job)
    tasks.update_job_stage('finalizing', chunks_done=3)
    assert RQ_SERIALIZER.loads(job.saved) == {
        'user_id': 7, 'stage': 'finalizing', 'chunks_done': 3}


def test_result_payload_and_return_value_roundtrip():
    """The tasks RETURN the result path (a str); the payload goes to the file.

    Both are checked: the path is what RQ stores, the payload is what a task
    would return if that ever changed.
    """
    payload = build_result_payload(
        '# Titel\n\nText mit Ümlaut.',
        provenance_unit=UNIT_DOCUMENT,
        provenance=[PROVENANCE_DETERMINISTIC],
        degradations=[degradation('serializer', 'Tabelle degradiert'),
                      degradation('backend_fallback', 'Seite leer', pages=[2, 3])],
        usage={'model_calls': 0, 'cost_eur': 0.0},
    )
    assert rq_json_roundtrip(payload) == payload
    path = ('/app/output_podcasts/doc_conversions/'
            'result_0b9e1c52-6f3a-4c1e-9d55-2a7f0c9b1e44.json')
    assert rq_json_roundtrip(path) == path


# --- what a serializer mismatch actually does (characterization) ---------------

class _HashOnlyRedis:
    """Just enough Redis for ``Job.fetch`` + ``is_failed``: job hashes, bytes."""

    def __init__(self):
        self.hashes = {}

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def hget(self, key, field):
        return self.hashes.get(key, {}).get(field.encode())


def test_pickle_job_read_through_fetch_job_does_not_raise(monkeypatch):
    """A pickle-written job (every job in Redis before this sprint) read with
    ``fetch_job``: no exception — ``meta`` degrades to ``{'unserialized': …}``,
    the status stays readable, only the job data is undecodable.

    Pinned on purpose: if RQ ever raised here, all three reconciles would
    swallow it (``except Exception`` → warning + return) and the conversion
    would stay ``pending`` forever. That is why the deploy flushes the old
    keys instead of relying on this.
    """
    conn = _HashOnlyRedis()
    legacy = Job.create(tasks.transcribe_audio_task, args=(42, 'wav', 'de'),
                        connection=conn, serializer=DefaultSerializer,
                        id='legacy-pickle-job',
                        meta={'user_id': 7, 'conversion_id': 42})
    mapping = legacy.to_dict()
    mapping['status'] = 'failed'
    conn.hashes[legacy.key] = {
        k.encode(): v if isinstance(v, bytes) else str(v).encode()
        for k, v in mapping.items()}
    monkeypatch.setattr(app_module, 'redis_conn', conn)

    job = app_module.fetch_job('legacy-pickle-job')

    assert job.serializer is RQ_SERIALIZER
    assert set(job.meta) == {'unserialized'}
    assert job.is_failed
    with pytest.raises(DeserializationError):
        job.args
