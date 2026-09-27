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
    assert launcher['volumes'] == ['/var/run/docker.sock:/var/run/docker.sock']
    assert launcher['networks'] == ['launch']
    assert 'ports' not in launcher
    assert 'env_file' not in launcher  # the socket holder carries no app secrets
    assert sorted(e.split('=', 1)[0] for e in launcher['environment']) == [
        'DOC_LOCAL_EXCHANGE_HOST_DIR', 'EXCHANGE_OWNER', 'MINERU_IMAGE',
        'MINERU_MODELS_DIR', 'MINERU_TIMEOUT_BASE_SECONDS']
    assert 'EXCHANGE_OWNER=0:0' in launcher['environment']
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
