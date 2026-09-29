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
    """CREDS-RO: web and worker read the GCP service-account key, they never
    write it — the bind is ``:ro`` at both, and no service carries it without.
    Since SEC-NONROOT both run as uid 1000, the owner of the host file (600):
    without ``:ro`` the process that parses foreign documents could overwrite
    the key on the host. The launcher holds no key at all (IMG-CONTEXT)."""
    compose_file = REPO / 'docker-compose.yml'
    if not compose_file.exists():
        pytest.skip('docker-compose.yml not shipped alongside the tests')
    yaml = pytest.importorskip('yaml')
    services = yaml.safe_load(compose_file.read_text())['services']
    ro_bind = './google-credentials.json:/app/google-credentials.json:ro'
    for name in ('markdown-converter', 'worker'):
        assert ro_bind in services[name]['volumes'], name
    carriers = {name: [v for v in svc.get('volumes', [])
                       if 'google-credentials.json' in v]
                for name, svc in services.items()}
    assert {n: v for n, v in carriers.items() if v} == {
        'markdown-converter': [ro_bind], 'worker': [ro_bind]}
