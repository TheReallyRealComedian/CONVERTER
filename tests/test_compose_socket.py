"""SEC-SOCKET — compose: the host's docker socket lives at the launcher, and
only there (the sentinel pattern of tests/test_proxy_fix.py).

The socket is root-equivalent on the host. Since SEC-SOCKET exactly one
service mounts it — ``mineru-launcher``, which can do one thing (run the
measured mineru vector) and holds no app secrets (no ``env_file``), no
exchange mount, no port, and sits only on the internal ``launch`` network
(Docker name ``converter_launch``) together with the worker. The worker lost
the socket and the host-path envs; the web container never had them since
DOC-WEB-ASYNC. Text-level check first: a commented-out socket line would be
one ``#`` away from coming back.

SEC-NONROOT: the image runs as uid 1000 (Dockerfile ``USER``); the launcher
alone is set back to root (``user: "0:0"``) — holding the socket is
root-equivalent whatever uid carries it — and copies mineru's output back
as the worker's ids, ``EXCHANGE_OWNER=1000:1000``. The Dockerfile side is
pinned in tests/test_nonroot.py.
"""
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def test_compose_socket_only_at_the_launcher():
    compose_file = REPO / 'docker-compose.yml'
    if not compose_file.exists():
        pytest.skip('docker-compose.yml not shipped alongside the tests')
    yaml = pytest.importorskip('yaml')
    text = compose_file.read_text()
    assert [line.strip() for line in text.splitlines() if 'docker.sock' in line] == [
        '- /var/run/docker.sock:/var/run/docker.sock']
    config = yaml.safe_load(text)
    services = config['services']
    holders = [name for name, svc in services.items()
               if any('docker.sock' in v for v in svc.get('volumes', []))]
    assert holders == ['mineru-launcher']

    launcher = services['mineru-launcher']
    assert launcher['image'] == 'converter-app:latest'
    assert launcher['command'] == 'python -m services.mineru_launcher'
    assert launcher['user'] == '0:0'  # the image's USER is 1000 (SEC-NONROOT)
    assert launcher['volumes'] == ['/var/run/docker.sock:/var/run/docker.sock']
    assert launcher['networks'] == ['launch']
    assert 'ports' not in launcher
    assert 'env_file' not in launcher  # the socket holder carries no app secrets
    assert sorted(e.split('=', 1)[0] for e in launcher['environment']) == [
        'DOC_LOCAL_EXCHANGE_HOST_DIR', 'EXCHANGE_OWNER', 'MINERU_IMAGE',
        'MINERU_MODELS_DIR', 'MINERU_TIMEOUT_BASE_SECONDS']
    assert 'EXCHANGE_OWNER=1000:1000' in launcher['environment']
    assert 'MINERU_IMAGE=mineru:3.4.4' in launcher['environment']
    assert '8765/health' in ' '.join(launcher['healthcheck']['test'])

    worker = services['worker']
    assert not any('docker.sock' in v for v in worker['volumes'])
    assert './doclocal_exchange:/app/doclocal_exchange' in worker['volumes']
    worker_env = {e.split('=', 1)[0] for e in worker['environment']}
    assert not worker_env & {'DOC_LOCAL_EXCHANGE_HOST_DIR', 'MINERU_IMAGE',
                             'MINERU_MODELS_DIR', 'EXCHANGE_OWNER'}
    assert 'MINERU_LAUNCHER_URL=http://mineru-launcher:8765' in worker['environment']
    assert worker['depends_on']['mineru-launcher'] == {'condition': 'service_healthy'}
    assert sorted(worker['networks']) == ['default', 'launch']

    on_launch = sorted(name for name, svc in services.items()
                       if 'launch' in (svc.get('networks') or []))
    assert on_launch == ['mineru-launcher', 'worker']
    assert config['networks']['launch'] == {'internal': True}


def test_credentials_bind_is_read_only():
    """CREDS-RO + ARCH-NARR5: the GCP service-account key is bound at the
    WORKER only, and ``:ro`` there. The worker renders narrations; it reads
    the key, it never writes it — it runs as uid 1000, the owner of the host
    file (600), and without ``:ro`` the process that parses foreign documents
    could overwrite the key on the host. No other service carries the file:
    the web container lost it with ARCH-NARR5 (its Cloud-TTS singleton had no
    reader), the launcher never held it (IMG-CONTEXT)."""
    compose_file = REPO / 'docker-compose.yml'
    if not compose_file.exists():
        pytest.skip('docker-compose.yml not shipped alongside the tests')
    yaml = pytest.importorskip('yaml')
    services = yaml.safe_load(compose_file.read_text())['services']
    ro_bind = './google-credentials.json:/app/google-credentials.json:ro'
    assert ro_bind in services['worker']['volumes']
    carriers = {name: [v for v in svc.get('volumes', [])
                       if 'google-credentials.json' in v]
                for name, svc in services.items()}
    assert {n: v for n, v in carriers.items() if v} == {'worker': [ro_bind]}


def test_web_mounts_no_host_path_and_names_no_key():
    """ARCH-NARR5: the internet-facing container mounts named volumes only —
    no bind from the host — and its ``environment`` block no longer passes
    ``GOOGLE_APPLICATION_CREDENTIALS`` (the name still arrives via
    ``env_file``; nothing in the web process reads it, and
    ``test_import_app_needs_no_key_file_and_loads_no_genai`` holds that the
    app boots with the name set and no file behind it)."""
    compose_file = REPO / 'docker-compose.yml'
    if not compose_file.exists():
        pytest.skip('docker-compose.yml not shipped alongside the tests')
    yaml = pytest.importorskip('yaml')
    config = yaml.safe_load(compose_file.read_text())
    web = config['services']['markdown-converter']
    named = set(config['volumes'])
    sources = [v.split(':')[0] for v in web['volumes']]
    assert sources and all(src in named for src in sources), sources
    env_names = [e.split('=')[0] for e in web['environment']]
    assert 'GOOGLE_APPLICATION_CREDENTIALS' not in env_names
    # Positive control: the worker still names it and still binds host paths.
    worker = config['services']['worker']
    assert 'GOOGLE_APPLICATION_CREDENTIALS' in [
        e.split('=')[0] for e in worker['environment']]
    assert any(v.split(':')[0] not in named for v in worker['volumes'])


def test_web_runs_under_an_init_and_only_web():
    """SEC-DG-TOKEN P2b: without an init, gunicorn is PID 1 in the web
    container and adopts every orphan — and the browser smokes run Chromium
    there via ``docker exec``, whose helper processes are orphaned at
    ``browser.close()``. gunicorn's ``reap_workers`` waits on ANY child
    (``waitpid(-1)``), logs it as "Worker (pid:N) was sent SIGTERM!" and
    raises ``HaltServer`` if such a foreign process exits with code 3 or 4
    (22.0.0, read at the installed source) — a smoke could stop the server.
    ``init: true`` makes docker-init PID 1: it reaps the orphans, gunicorn
    only ever sees its own workers. Only at the web service — the measured
    case; nothing is run via ``docker exec`` under a reaping PID 1 elsewhere."""
    compose_file = REPO / 'docker-compose.yml'
    if not compose_file.exists():
        pytest.skip('docker-compose.yml not shipped alongside the tests')
    yaml = pytest.importorskip('yaml')
    services = yaml.safe_load(compose_file.read_text())['services']
    assert services['markdown-converter'].get('init') is True
    assert [name for name, svc in services.items() if 'init' in svc] == [
        'markdown-converter']
