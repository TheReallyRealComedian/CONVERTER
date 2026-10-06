"""SEC-SOCKET — the mineru launcher, the one holder of the host's docker socket.

Driven over real HTTP: ``fake_launcher`` (conftest) serves the REAL launcher
on a loopback port and fakes only the docker CLI (``subprocess.run``). What
is pinned here:

1. Validation is fail-closed: a malformed request is refused (400/413)
   before ANY docker call — traversal, slashes, control characters, the
   wrong job shape, extra fields such as ``argv``, bad page counts.
2. The command line comes from the launcher's env; client fields surface
   only as ``--name <job>``, the job segment of the two ``-v`` sources and
   ``/in/<pdf_name>``.
3. No ``-v`` source of ANY docker call lies below the exchange root — the
   daemon dereferences bind sources, and everything below the root is the
   worker's to shape (a symlinked ``out`` mounted any host path, measured).
   The run mounts the job's volumes; only the digest-pinned busybox
   helpers mount the root, and resolve below it inside themselves.
4. One run at a time (409 while a run is active, /health still answers);
   at the deadline the container is removed (``docker rm -f``) BEFORE the
   answer; copy failures are named (422 in, 500 out); the job's volumes go
   in every path; SIGTERM removes a running container and the volumes.
5. Minimal surface: importing the launcher loads no service SDK, no Flask,
   no ``app_pkg``; no shell anywhere.

The compose side (the socket only at the launcher) is pinned in
tests/test_compose_socket.py.
"""
import ast
import http.client
import json
import logging
import signal
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from services import mineru_launcher
from services.mineru_invocation import (
    COPY_IN_MAX_MB,
    COPY_IN_TIMEOUT_SECONDS,
    COPY_OUT_TIMEOUT_SECONDS,
    HELPER_IMAGE,
    IMAGE_INSPECT_TIMEOUT_SECONDS,
    KILL_TIMEOUT_SECONDS,
    MINERU_MAX_PAGES,
    VOLUME_RM_TIMEOUT_SECONDS,
)

REPO = Path(__file__).resolve().parent.parent
JOB = 'mineru_0123456789ab'
GOOD = {'job': JOB, 'pdf_name': 'doc.pdf', 'page_count': 2}
JSON = {'Content-Type': 'application/json'}


def _request(url, method, path, body=None, headers=None):
    parts = urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=10)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read()
        if resp.getheader('Content-Type', '').startswith('application/json'):
            return resp.status, json.loads(data)
        return resp.status, None
    finally:
        conn.close()


def _post_run(state, payload=GOOD, headers=JSON):
    return _request(state['url'], 'POST', '/run',
                    json.dumps(payload).encode('utf-8'), headers)


def _names(state):
    """Container names the (fake) daemon was asked to run, in order."""
    return [c['cmd'][c['cmd'].index('--name') + 1]
            for c in state['calls'] if '--name' in c['cmd']]


def _put_input(state, job=JOB, pdf_name='doc.pdf', data=b'%PDF-1.7 fake'):
    """What the worker does before it asks: the input in its job dir."""
    in_dir = state['exchange'] / job / 'in'
    in_dir.mkdir(parents=True, exist_ok=True)
    (state['exchange'] / job / 'out').mkdir(exist_ok=True)
    (in_dir / pdf_name).write_bytes(data)


# --- 1. fail-closed validation -------------------------------------------------

@pytest.mark.parametrize('payload', [
    {**GOOD, 'pdf_name': '../x.pdf'},
    {**GOOD, 'pdf_name': '../../etc/passwd'},
    {**GOOD, 'pdf_name': 'a/b.pdf'},
    {**GOOD, 'pdf_name': '/in/doc.pdf'},
    {**GOOD, 'pdf_name': 'a\\b.pdf'},
    {**GOOD, 'pdf_name': ''},
    {**GOOD, 'pdf_name': 'doc\n.pdf'},
    {**GOOD, 'pdf_name': 'doc.pdf\n'},
    {**GOOD, 'pdf_name': 'doc\x00.pdf'},
    {**GOOD, 'pdf_name': 'doc\x1b.pdf'},
    {**GOOD, 'pdf_name': '.pdf'},
    {**GOOD, 'pdf_name': '..pdf'},
    {**GOOD, 'pdf_name': '-rf.pdf'},
    {**GOOD, 'pdf_name': 'doc .pdf'},
    {**GOOD, 'pdf_name': 'doc'},
    {**GOOD, 'pdf_name': ['doc.pdf']},
    {**GOOD, 'job': 'mineru_0123456789AB'},
    {**GOOD, 'job': 'mineru_0123'},
    {**GOOD, 'job': 'mineru_0123456789ab\n'},
    {**GOOD, 'job': '../mineru_0123456789ab'},
    {**GOOD, 'job': 'mineru_0123456789ab:/host'},
    {**GOOD, 'job': 'redis'},
    {**GOOD, 'job': 12},
    {**GOOD, 'page_count': 0},
    {**GOOD, 'page_count': -1},
    {**GOOD, 'page_count': '2'},
    {**GOOD, 'page_count': 2.0},
    {**GOOD, 'page_count': True},
    {**GOOD, 'page_count': None},
    {**GOOD, 'page_count': MINERU_MAX_PAGES + 1},
    {**GOOD, 'argv': ['docker', 'run', '-v', '/:/host', 'alpine']},
    {**GOOD, 'image': 'alpine'},
    {'job': JOB, 'pdf_name': 'doc.pdf'},
    [GOOD],
    'doc.pdf',
])
def test_invalid_requests_are_refused_before_any_docker_call(fake_launcher, payload):
    status, body = _post_run(fake_launcher, payload)
    assert status == 400
    assert body['error']
    assert fake_launcher['calls'] == []


@pytest.mark.parametrize('raw, headers, status', [
    (b'{not json', JSON, 400),
    (b'{"job": "mineru_0123456789ab", "job": "mineru_ffffffffffff", '
     b'"pdf_name": "doc.pdf", "page_count": 2}', JSON, 400),  # duplicate key
    (b'{"job": "mineru_0123456789ab", "pdf_name": "doc.pdf", '
     b'"page_count": NaN}', JSON, 400),
    (b'\xff\xfe{}', JSON, 400),
    (json.dumps(GOOD).encode(), {'Content-Type': 'text/plain'}, 400),
    (json.dumps(GOOD).encode(), {}, 400),
    (b'{"job": "' + b'a' * 5000 + b'"}', JSON, 413),
])
def test_malformed_bodies_are_refused(fake_launcher, raw, headers, status):
    got, body = _request(fake_launcher['url'], 'POST', '/run', raw, headers)
    assert got == status
    assert body['error']
    assert fake_launcher['calls'] == []


# --- 2. the command line comes from the launcher's env --------------------------

def test_argv_is_built_from_the_launchers_env(fake_launcher, monkeypatch,
                                             tmp_path):
    host = tmp_path / 'host-exchange'  # the daemon's view, set only here
    monkeypatch.setenv('MINERU_IMAGE', 'mineru:3.4.4')
    monkeypatch.setenv('MINERU_MODELS_DIR', '/srv/hf-cache')
    monkeypatch.setenv('DOC_LOCAL_EXCHANGE_HOST_DIR', f'{host}/')
    monkeypatch.setenv('EXCHANGE_OWNER', '1000:1001')
    fake_launcher['exchange'] = host
    _put_input(fake_launcher)
    status, body = _post_run(fake_launcher)
    assert status == 200
    assert body == {'returncode': 0, 'timed_out': False, 'deadline_seconds': 320,
                    'stdout_tail': '', 'stderr_tail': ''}
    copy_in, run, copy_out, volume_rm = fake_launcher['calls']
    assert copy_in['cmd'] == [
        'docker', 'run', '--name', f'{JOB}_copyin', '--rm', '--network', 'none',
        '-v', f'{host}:/x:ro', '-v', f'{JOB}_in:/in',
        HELPER_IMAGE, 'dd', f'if=/x/{JOB}/in/doc.pdf', 'of=/in/doc.pdf',
        'bs=1M', f'count={COPY_IN_MAX_MB}']
    assert copy_in['timeout'] == COPY_IN_TIMEOUT_SECONDS
    assert run['cmd'] == [
        'docker', 'run', '--name', JOB, '--rm', '--gpus', 'all',
        '--shm-size', '16g',
        '-v', f'{JOB}_in:/in:ro', '-v', f'{JOB}_out:/out',
        '-v', '/srv/hf-cache:/models',
        '-e', 'HF_HOME=/models', '-e', 'MINERU_MODEL_SOURCE=huggingface',
        'mineru:3.4.4', 'mineru', '-p', '/in/doc.pdf', '-o', '/out',
        '-b', 'vlm-engine']
    assert run['timeout'] == 320  # the launcher owns the deadline
    # Client fields surface ONLY as the job's names and the file name:
    assert [a for a in run['cmd'] if JOB in a] == [JOB, f'{JOB}_in:/in:ro',
                                                   f'{JOB}_out:/out']
    assert [a for a in run['cmd'] if 'doc.pdf' in a] == ['/in/doc.pdf']
    assert copy_out['cmd'] == [
        'docker', 'run', '--name', f'{JOB}_copyout', '--rm', '--network', 'none',
        '--user', '1000:1001',
        '-v', f'{JOB}_out:/o:ro', '-v', f'{host}:/x',
        HELPER_IMAGE, 'cp', '-r', '/o/.', f'/x/{JOB}/out']
    assert copy_out['timeout'] == COPY_OUT_TIMEOUT_SECONDS
    assert volume_rm['cmd'] == ['docker', 'volume', 'rm', f'{JOB}_in', f'{JOB}_out']
    assert volume_rm['timeout'] == VOLUME_RM_TIMEOUT_SECONDS
    # The content landed in the job dir, the volumes are gone.
    assert (host / JOB / 'out' / 'doc' / 'vlm' / 'doc_content_list.json').is_file()
    assert not any(fake_launcher['volumes'].iterdir())


def test_failed_run_answers_200_with_capped_tails(fake_launcher):
    _put_input(fake_launcher)
    fake_launcher['rc'] = 1
    fake_launcher['stderr'] = 'x' * 5000 + 'CUDA out of memory'
    status, body = _post_run(fake_launcher)
    assert status == 200
    assert body['returncode'] == 1
    assert body['timed_out'] is False
    assert len(body['stderr_tail']) == 800
    assert body['stderr_tail'].endswith('CUDA out of memory')
    # A failed run is not copied out; its volumes still go.
    assert fake_launcher['kinds']() == ['copy_in', 'run', 'volume_rm']


def test_missing_input_is_a_named_422_and_starts_no_run(fake_launcher):
    status, body = _post_run(fake_launcher)  # the worker put nothing there
    assert status == 422
    assert body['error'].startswith('Eingabe nicht übernehmbar: rc=1 dd:')
    assert fake_launcher['kinds']() == ['copy_in', 'volume_rm']


@pytest.mark.parametrize('env, named', [
    ({'DOC_LOCAL_EXCHANGE_HOST_DIR': None}, 'DOC_LOCAL_EXCHANGE_HOST_DIR'),
    ({'DOC_LOCAL_EXCHANGE_HOST_DIR': 'relativ/pfad'}, 'DOC_LOCAL_EXCHANGE_HOST_DIR'),
    ({'DOC_LOCAL_EXCHANGE_HOST_DIR': '/a:b'}, 'DOC_LOCAL_EXCHANGE_HOST_DIR'),
    ({'DOC_LOCAL_EXCHANGE_HOST_DIR': '/'}, 'DOC_LOCAL_EXCHANGE_HOST_DIR'),
    ({'EXCHANGE_OWNER': None}, 'EXCHANGE_OWNER'),
    ({'EXCHANGE_OWNER': 'root'}, 'EXCHANGE_OWNER'),
    ({'EXCHANGE_OWNER': '0:0 /'}, 'EXCHANGE_OWNER'),
    ({'MINERU_MODELS_DIR': 'models'}, 'MINERU_MODELS_DIR'),
])
def test_unconfigured_launcher_answers_503_and_starts_nothing(fake_launcher,
                                                              monkeypatch, env, named):
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key)
        else:
            monkeypatch.setenv(key, value)
    status, body = _post_run(fake_launcher)
    assert status == 503
    assert named in body['error']
    assert fake_launcher['calls'] == []
    # Liveness is independent of the config (the worker depends on it).
    assert _request(fake_launcher['url'], 'GET', '/health')[0] == 200


@pytest.mark.parametrize('raw, deadline', [
    ('5', 25), ('300', 320), ('99999', 320), ('0', 320), ('-5', 320),
    ('abc', 320), ('', 320)])
def test_deadline_base_override_can_only_shorten(fake_launcher, monkeypatch,
                                                 raw, deadline):
    """The kill probe's knob: shorter is honoured, longer or junk is not —
    a longer launcher deadline would outrun the worker's HTTP wait."""
    monkeypatch.setenv('MINERU_TIMEOUT_BASE_SECONDS', raw)
    _put_input(fake_launcher)
    status, body = _post_run(fake_launcher)  # 2 pages
    assert status == 200
    assert body['deadline_seconds'] == deadline
    assert fake_launcher['runs']()[0]['timeout'] == deadline


# --- 3. no bind source below the exchange root ---------------------------------

def test_no_bind_source_is_worker_controlled(fake_launcher, monkeypatch):
    """THE invariant behind the symlink finding: across every docker call
    of a run, a ``-v`` source is the job's volume, the exchange ROOT (fixed
    by the launcher's env) or the models dir — never a path below the root,
    where the worker could have planted a symlink for the daemon to follow."""
    monkeypatch.setenv('MINERU_MODELS_DIR', '/srv/hf-cache')
    _put_input(fake_launcher)
    assert _post_run(fake_launcher)[0] == 200
    root = str(fake_launcher['exchange'])
    sources = set()
    for call in fake_launcher['calls']:
        cmd = call['cmd']
        sources |= {cmd[i + 1].split(':')[0]
                    for i, arg in enumerate(cmd[:-1]) if arg == '-v'}
    assert sources == {f'{JOB}_in', f'{JOB}_out', root, '/srv/hf-cache'}


# --- 4. one at a time, the kill, the copy steps, SIGTERM -------------------

def test_second_run_during_an_active_run_gets_409(fake_launcher, monkeypatch):
    """A real concurrent request against a run that holds the lock: 409,
    nothing started for it — and /health still answers meanwhile (why the
    server is threaded; the lock, not the thread count, serialises runs)."""
    started, release = threading.Event(), threading.Event()
    inner = mineru_launcher.subprocess.run  # the fixture's fake

    def blocking(argv, **kwargs):
        if 'mineru' in argv:
            started.set()
            release.wait(10)
        return inner(argv, **kwargs)

    monkeypatch.setattr(mineru_launcher.subprocess, 'run', blocking)
    _put_input(fake_launcher)
    first = {}
    worker = threading.Thread(
        target=lambda: first.update(answer=_post_run(fake_launcher)))
    worker.start()
    try:
        assert started.wait(10)
        second = _post_run(fake_launcher, {**GOOD, 'job': 'mineru_ffffffffffff'})
        health = _request(fake_launcher['url'], 'GET', '/health')
    finally:
        release.set()
        worker.join(10)
    assert second == (409, {'error': 'Es läuft bereits ein mineru-Lauf.'})
    assert health == (200, {'status': 'ok', 'busy': True})
    assert first['answer'][0] == 200
    assert _names(fake_launcher) == [f'{JOB}_copyin', JOB, f'{JOB}_copyout']
    assert _request(fake_launcher['url'], 'GET', '/health')[1]['busy'] is False


def test_deadline_removes_the_container_before_answering(fake_launcher):
    _put_input(fake_launcher)
    fake_launcher['raise_timeout'] = True
    status, body = _post_run(fake_launcher)
    assert status == 200
    assert body == {'returncode': None, 'timed_out': True, 'deadline_seconds': 320,
                    'stdout_tail': '', 'stderr_tail': 'mineru lief noch'}
    assert fake_launcher['kinds']() == ['copy_in', 'run', 'kill', 'volume_rm']
    kill = fake_launcher['calls'][2]
    assert kill['cmd'] == ['docker', 'rm', '-f', JOB]
    assert kill['timeout'] == KILL_TIMEOUT_SECONDS


def test_copy_out_past_its_deadline_is_removed_and_named(fake_launcher,
                                                         monkeypatch):
    """A helper that tears its own deadline is removed like the run — and
    the answer names the step (500), so the worker falls back with it."""
    inner = mineru_launcher.subprocess.run

    def copy_out_hangs(argv, **kwargs):
        if 'cp' in argv:
            inner(argv, **kwargs)  # recorded
            raise subprocess.TimeoutExpired(argv, kwargs.get('timeout'))
        return inner(argv, **kwargs)

    monkeypatch.setattr(mineru_launcher.subprocess, 'run', copy_out_hangs)
    _put_input(fake_launcher)
    status, body = _post_run(fake_launcher)
    assert status == 500
    assert body['error'] == (f'Ausgabe nicht übernehmbar: Frist '
                             f'{COPY_OUT_TIMEOUT_SECONDS} s gerissen.')
    assert fake_launcher['kinds']() == ['copy_in', 'run', 'copy_out', 'kill',
                                        'volume_rm']
    assert fake_launcher['calls'][3]['cmd'] == ['docker', 'rm', '-f',
                                                f'{JOB}_copyout']


@pytest.mark.parametrize('stderr, loud', [
    ('Error response from daemon: get mineru_0123456789ab_out: no such volume\n', False),
    ('Error response from daemon: remove mineru_0123456789ab_in: volume is in use\n', True),
])
def test_volume_cleanup_is_loud_only_about_real_failures(fake_launcher, monkeypatch,
                                                         caplog, stderr, loud):
    """A copy-in that fails leaves no ``<job>_out`` behind (the run never
    created it) — that is not an error; a volume that stays is."""
    inner = mineru_launcher.subprocess.run

    def volume_rm_reports(argv, **kwargs):
        result = inner(argv, **kwargs)
        if argv[:3] == ['docker', 'volume', 'rm']:
            return SimpleNamespace(returncode=1, stdout='', stderr=stderr)
        return result

    monkeypatch.setattr(mineru_launcher.subprocess, 'run', volume_rm_reports)
    with caplog.at_level(logging.ERROR, logger='mineru_launcher'):
        status, _ = _post_run(fake_launcher)  # no input → 422
    assert status == 422
    logged = [r.getMessage() for r in caplog.records
              if r.getMessage().startswith('Volumes von')]
    assert bool(logged) is loud


def test_missing_docker_cli_is_a_500_not_a_dropped_connection(fake_launcher,
                                                              monkeypatch):
    def no_cli(argv, **kwargs):
        raise FileNotFoundError(2, 'No such file or directory', 'docker')

    monkeypatch.setattr(mineru_launcher.subprocess, 'run', no_cli)
    status, body = _post_run(fake_launcher)
    assert status == 500
    assert body['error'].startswith('docker-CLI nicht ausführbar')
    assert mineru_launcher._RUN_LOCK.locked() is False  # released


def test_sigterm_removes_the_running_container(fake_launcher, monkeypatch):
    monkeypatch.setattr(mineru_launcher, '_current_container', JOB)
    monkeypatch.setattr(mineru_launcher, '_current_job', JOB)
    with pytest.raises(SystemExit):
        mineru_launcher._terminate(signal.SIGTERM, None)
    assert [c['cmd'] for c in fake_launcher['calls']] == [
        ['docker', 'rm', '-f', JOB],
        ['docker', 'volume', 'rm', f'{JOB}_in', f'{JOB}_out']]
    # While stopping, no new run starts.
    status, body = _post_run(fake_launcher)
    assert status == 503
    assert len(fake_launcher['calls']) == 2


def test_sigterm_without_a_run_just_exits(fake_launcher):
    with pytest.raises(SystemExit):
        mineru_launcher._terminate(signal.SIGTERM, None)
    assert fake_launcher['calls'] == []


def test_routes(fake_launcher):
    url = fake_launcher['url']
    assert _request(url, 'GET', '/health') == (200, {'status': 'ok', 'busy': False})
    assert _request(url, 'GET', '/run')[0] == 405
    assert _request(url, 'POST', '/health', b'{}', JSON)[0] == 405
    assert _request(url, 'GET', '/')[0] == 404
    assert _request(url, 'POST', '/run?x=1', json.dumps(GOOD).encode(), JSON)[0] == 404
    assert _request(url, 'PUT', '/run', json.dumps(GOOD).encode(), JSON)[0] == 501
    assert _request(url, 'DELETE', '/run')[0] == 501
    assert fake_launcher['calls'] == []


# --- 4b. the image behind the tag (ARCH-BUILD, W-6a) --------------------------------

INSPECT_ARGV = ['docker', 'image', 'inspect', '--format', '{{.Id}} {{.Created}}']


def _inspect_answering(stdout='', stderr='', returncode=0, calls=None):
    def fake(argv, **kwargs):
        if calls is not None:
            calls.append((list(argv), kwargs.get('timeout')))
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)
    return fake


def test_startup_logs_the_identity_behind_the_image_tag(fake_launcher, monkeypatch, caplog):
    monkeypatch.setenv('MINERU_IMAGE', 'mineru:3.4.4')
    calls = []
    monkeypatch.setattr(mineru_launcher.subprocess, 'run', _inspect_answering(
        stdout='sha256:6cc9e57ff5bd0123 2026-08-15T10:11:12.123456789Z\n', calls=calls))
    with caplog.at_level(logging.INFO, logger='mineru_launcher'):
        mineru_launcher._log_config_state()
    assert calls == [(INSPECT_ARGV + ['mineru:3.4.4'], IMAGE_INSPECT_TIMEOUT_SECONDS)]
    messages = [(r.levelno, r.getMessage()) for r in caplog.records]
    assert (logging.INFO, 'Image mineru:3.4.4 = sha256:6cc9e57ff5bd0123, '
            'erstellt 2026-08-15T10:11:12.123456789Z') in messages
    assert not [m for level, m in messages if level >= logging.WARNING], messages


def test_missing_image_is_one_warning_and_the_service_keeps_answering(fake_launcher,
                                                                         monkeypatch, caplog):
    monkeypatch.setenv('MINERU_IMAGE', 'mineru:9.9.9')
    monkeypatch.setattr(mineru_launcher.subprocess, 'run', _inspect_answering(
        returncode=1, stderr='Error response from daemon: No such image: mineru:9.9.9\n'))
    with caplog.at_level(logging.INFO, logger='mineru_launcher'):
        mineru_launcher._log_config_state()
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, warnings
    assert warnings[0].startswith('Image mineru:9.9.9 nicht vorhanden') and 'No such image' in warnings[0]
    assert 'Textebene' in warnings[0]
    assert _request(fake_launcher['url'], 'GET', '/health') == (200, {'status': 'ok', 'busy': False})


def test_unreadable_identity_is_a_warning_not_a_crash(fake_launcher, monkeypatch, caplog):
    def no_cli(argv, **kwargs):
        raise FileNotFoundError(2, 'No such file or directory', 'docker')
    monkeypatch.setattr(mineru_launcher.subprocess, 'run', no_cli)
    with caplog.at_level(logging.INFO, logger='mineru_launcher'):
        mineru_launcher._log_config_state()
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and 'nicht lesbar' in warnings[0], warnings


# --- 5. minimal surface -----------------------------------------------------------

def test_launcher_import_surface_is_minimal():
    """The socket holder loads stdlib + services.mineru_invocation — no
    service SDK, no Flask, no app_pkg, nothing that parses documents
    (possible because services/__init__.py resolves its classes lazily)."""
    code = 'import sys, services.mineru_launcher; print("\\n".join(sorted(sys.modules)))'
    loaded = set(subprocess.run([sys.executable, '-c', code], cwd=REPO,
                                capture_output=True, text=True, timeout=60,
                                check=True).stdout.split())
    project = {m for m in loaded if m == 'services' or m.startswith('services.')}
    assert project == {'services', 'services.mineru_invocation',
                       'services.mineru_launcher'}
    for forbidden in ('app_pkg', 'flask', 'deepgram', 'google', 'grpc',
                      'requests', 'fitz', 'rq', 'redis', 'sqlalchemy'):
        assert forbidden not in loaded, forbidden


def test_launcher_never_uses_a_shell():
    tree = ast.parse(Path(mineru_launcher.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            assert all(kw.arg != 'shell' for kw in node.keywords)
            assert ast.unparse(node.func) not in (
                'os.system', 'os.popen', 'subprocess.Popen', 'subprocess.call',
                'subprocess.check_call', 'subprocess.check_output')
