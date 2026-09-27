"""SEC-SOCKET — the mineru invocation in its ONE place.

The vector sentinel moved here with the vector (it lived in test_pdf_local
while the worker built the command line itself): the measurement only holds
for the verbatim bake-off call — a deviating vector devalues it and must
fail loudly here, like the pandoc vector sentinel in DOC-ENGINE.
``--name <job>`` is the single addition to the vector (only a named
container can be removed when it tears its deadline); its ``-v`` SOURCES
are the job's volumes — the pairs are untouched.
"""
import ast
import re
from pathlib import Path

from services import mineru_invocation
from services.mineru_invocation import (
    COPY_IN_MAX_MB,
    COPY_IN_TIMEOUT_SECONDS,
    COPY_OUT_TIMEOUT_SECONDS,
    HELPER_IMAGE,
    KILL_TIMEOUT_SECONDS,
    LAUNCHER_REPLY_MARGIN_SECONDS,
    VOLUME_RM_TIMEOUT_SECONDS,
    build_copy_in_argv,
    build_copy_out_argv,
    build_kill_argv,
    build_run_argv,
    build_volume_rm_argv,
    is_job_name,
    launcher_reply_timeout_for,
    mineru_run_timeout_for,
    new_job_name,
    volume_names,
)

JOB = 'mineru_0123456789ab'


def _run_argv(**overrides):
    kwargs = dict(image='mineru:latest', in_source=f'{JOB}_in',
                  out_source=f'{JOB}_out', pdf_name='doc.pdf',
                  container_name=JOB)
    kwargs.update(overrides)
    return build_run_argv(**kwargs)


def test_invocation_vector_is_the_measured_one():
    """SENTINEL: verbatim bake-off invocation (mineru 3.4.4, vlm-engine)."""
    cmd = _run_argv()
    assert cmd[:2] == ['docker', 'run']
    adjacent = set(zip(cmd, cmd[1:]))
    for pair in (('--gpus', 'all'), ('--shm-size', '16g'),
                 ('-e', 'HF_HOME=/models'),
                 ('-e', 'MINERU_MODEL_SOURCE=huggingface'),
                 ('-p', '/in/doc.pdf'), ('-o', '/out'),
                 ('-b', 'vlm-engine')):
        assert pair in adjacent
    assert '--rm' in cmd
    assert f'{JOB}_in:/in:ro' in cmd  # source read-only
    assert f'{JOB}_out:/out' in cmd
    assert ('--name', JOB) in adjacent  # SEC-SOCKET: killable
    # The whole vector, exactly — any edit here changes the measured call.
    assert cmd == [
        'docker', 'run', '--name', JOB, '--rm', '--gpus', 'all',
        '--shm-size', '16g',
        '-v', f'{JOB}_in:/in:ro', '-v', f'{JOB}_out:/out',
        '-e', 'HF_HOME=/models', '-e', 'MINERU_MODEL_SOURCE=huggingface',
        'mineru:latest', 'mineru', '-p', '/in/doc.pdf', '-o', '/out',
        '-b', 'vlm-engine']


def test_models_dir_adds_the_cache_mount_in_place():
    cmd = _run_argv(models_dir_host='/srv/hf-cache')
    mounts = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == '-v']
    assert mounts == [f'{JOB}_in:/in:ro', f'{JOB}_out:/out',
                      '/srv/hf-cache:/models']
    assert not any(a.endswith(':/models') for a in _run_argv())


def test_helpers_use_the_digest_pinned_busybox():
    """busybox pinned by digest: the one that ran every chown pass on the
    Mintbox since DOC-LOCAL (BusyBox v1.38.0); the tag is only a label."""
    name, _, digest = HELPER_IMAGE.partition('@')
    assert name == 'busybox:1.38.0'
    assert re.fullmatch(r'sha256:[0-9a-f]{64}', digest)


def test_copy_helpers_mount_only_the_exchange_root():
    """The symlink rule: the only host path a helper mounts is the exchange
    ROOT — everything below it resolves inside the helper, never on the
    host. Copy-in reads the root read-only; copy-out writes AS the owner
    (the successor of the chown pass); both without network."""
    assert volume_names(JOB) == (f'{JOB}_in', f'{JOB}_out')
    assert build_copy_in_argv(exchange_host='/srv/ex', job=JOB, pdf_name='doc.pdf',
                              in_volume=f'{JOB}_in',
                              container_name=f'{JOB}_copyin') == [
        'docker', 'run', '--name', f'{JOB}_copyin', '--rm', '--network', 'none',
        '-v', '/srv/ex:/x:ro', '-v', f'{JOB}_in:/in',
        HELPER_IMAGE, 'dd', f'if=/x/{JOB}/in/doc.pdf', 'of=/in/doc.pdf',
        'bs=1M', f'count={COPY_IN_MAX_MB}']
    assert build_copy_out_argv(exchange_host='/srv/ex', job=JOB,
                               out_volume=f'{JOB}_out', owner='1000:1001',
                               container_name=f'{JOB}_copyout') == [
        'docker', 'run', '--name', f'{JOB}_copyout', '--rm', '--network', 'none',
        '--user', '1000:1001',
        '-v', f'{JOB}_out:/o:ro', '-v', '/srv/ex:/x',
        HELPER_IMAGE, 'cp', '-r', '/o/.', f'/x/{JOB}/out']
    assert build_volume_rm_argv(*volume_names(JOB)) == [
        'docker', 'volume', 'rm', f'{JOB}_in', f'{JOB}_out']
    # Twice the 100 MB upload cap: a real input never meets it.
    assert COPY_IN_MAX_MB == 200


def test_kill_argv():
    assert build_kill_argv(JOB) == ['docker', 'rm', '-f', JOB]


def test_deadline_curve_and_the_reply_margin():
    assert mineru_run_timeout_for(1) == 310
    assert mineru_run_timeout_for(280) == 300 + 10 * 280  # carries 12_grosses
    assert mineru_run_timeout_for(None) == mineru_run_timeout_for(0) == 310
    assert mineru_run_timeout_for(2, base_seconds=5) == 25  # the probe path
    # The worker waits for everything the launcher may do besides the
    # deadline: copy in, remove a container, copy out, remove the volumes.
    assert LAUNCHER_REPLY_MARGIN_SECONDS > (
        COPY_IN_TIMEOUT_SECONDS + KILL_TIMEOUT_SECONDS
        + COPY_OUT_TIMEOUT_SECONDS + VOLUME_RM_TIMEOUT_SECONDS)
    assert launcher_reply_timeout_for(12) == (mineru_run_timeout_for(12)
                                              + LAUNCHER_REPLY_MARGIN_SECONDS)


def test_job_name_contract():
    """The job name is the container name AND a directory segment of two
    ``-v`` sources — its alphabet must never carry '/', '..' or ':'."""
    for _ in range(50):
        assert is_job_name(new_job_name())
    for bad in ('mineru_0123456789AB', 'mineru_0123456789a',
                'mineru_0123456789abc', 'mineru_0123456789ab\n',
                'x_0123456789ab', '../mineru_0123456789ab',
                'mineru_0123456789ab/..', 'mineru_01234567:9ab', '',
                None, 12, b'mineru_0123456789ab'):
        assert not is_job_name(bad), repr(bad)


def test_module_is_pure():
    """No docker, no subprocess, no environment reads, no project imports —
    argv lists and numbers only."""
    tree = ast.parse(Path(mineru_invocation.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split('.')[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split('.')[0])
    assert imported == {'re', 'uuid'}
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & {'environ', 'getenv', 'run', 'Popen', 'system'}
