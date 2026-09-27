"""The mineru sibling-container invocation — its ONE place (SEC-SOCKET).

Pure module: argv lists, deadlines and the job-name contract. No Flask, no
docker, no subprocess, no environment reads. Two kinds of readers:

* ``services.mineru_launcher`` — the sidecar that alone holds the host's
  docker socket builds every command line from these functions plus its own
  environment. The worker sends it data (job name, PDF name, page count),
  never arguments.
* ``services.pdf_local`` (worker) and ``app_pkg.config`` (RQ envelope) —
  the deadline arithmetic, so the invariant chain hangs on one source::

      launcher deadline        mineru_run_timeout_for(n)
        < worker HTTP timeout    + LAUNCHER_REPLY_MARGIN_SECONDS
        < RQ envelope            + app_pkg.config margin (300 s)

**The run vector is replicated verbatim** from the measured bake-off adapter
(``corpus/bakeoff/harness/adapters.py::run_mineru_vlm``, mineru 3.4.4,
gold-f1 0.9551 on 01.gold) — lesson ``reference_measured_winner_version_gap``;
``tests/test_mineru_invocation.py::test_invocation_vector_is_the_measured_one``
pins its pairs::

    docker run --name <job> --rm --gpus all --shm-size 16g
      -v <in>:/in:ro -v <out>:/out [-v <models>:/models]
      -e HF_HOME=/models -e MINERU_MODEL_SOURCE=huggingface
      <image> mineru -p /in/<datei>.pdf -o /out -b vlm-engine

``--name <job>`` is the one addition: only a named container can be removed
when it tears its deadline (``subprocess.run(timeout=)`` kills the docker
CLI client, never the container).

**``<in>`` and ``<out>`` are per-job Docker volumes, not exchange paths**
(SEC-SOCKET, measured on the Mintbox): the host daemon DEREFERENCES a bind
source. The worker controls everything below the exchange root — had the
launcher passed ``<exchange>/<job>/out``, a compromised worker could replace
``out`` with a symlink to ``/etc`` and get it mounted read-write into a root
container, followed by ``chown -R``. A host path the worker can shape is a
pointer, not data. So the only host path any container here ever mounts is
the exchange ROOT (fixed by the launcher's env — the worker cannot replace
its own mount point), and only in two busybox helpers; symlinks below it
resolve inside the helper's mount namespace and never reach the host:

    copy-in:   exchange root :ro + <job>_in    dd  /x/<job>/in/<pdf> → /in
    run:       <job>_in :ro + <job>_out        the vector above
    copy-out:  <job>_out :ro + exchange root   cp -r /o/. /x/<job>/out
               as ``--user <owner>`` — the files land owned by the worker's
               ids (this replaces the old ``chown -R`` pass: chown, not
               chmod, because ``a+rX`` left the cleanup dying on EPERM)
    cleanup:   docker volume rm <job>_in <job>_out

The backend name is ``vlm-engine`` (the 2.x docs still say
``vlm-vllm-engine`` — verified live against ``--help`` in the bake-off).
The mineru container runs as root (``--user`` dies on missing passwd
entries, live-hit); its output never touches the exchange directly.
"""
import re
import uuid

MINERU_DEFAULT_IMAGE = 'mineru:latest'
MINERU_BACKEND = 'vlm-engine'

# Run deadline from the measured cost curve (~61 s model start + ~2.5 s per
# page, fitted 2..280 pages), with ~4x margin for GPU contention with Olis
# ComfyUI: 280 pages measured 766 s, allowed 300 + 10 × 280 = 3100 s. The
# per-call deadline doctrine (reference_worker_sdk_per_call_deadline): the RQ
# envelope alone would never interrupt a wedged run.
MINERU_TIMEOUT_BASE_SECONDS = 300
MINERU_TIMEOUT_PER_PAGE_SECONDS = 10

# Largest page range one run accepts — the launcher refuses more (400), the
# worker then serves the text layer with the reason named. Chosen inside the
# range where the chain above holds without the RQ hard cap ever biting:
# 1000 pages → deadline 10 300 s, worker timeout 10 540 s, envelope 10 600 s
# (cap 14 400 s). The largest measured document (12_grosses-pdf) has 280.
MINERU_MAX_PAGES = 1000

# The helper image, pinned by DIGEST (the tag is only a label next to it):
# exactly the busybox that ran every chown pass on the Mintbox since
# DOC-LOCAL (pulled 2026-05, BusyBox v1.38.0, multi-arch index). A tag alone
# can be re-pushed; these helpers run as root over the exchange root.
HELPER_IMAGE = ('busybox:1.38.0@sha256:'
                'dc2d74b28e4cf8984fa52af1f39bc7c3d9c73760b41a74d629f5d11b1ab28616')

# Copy-in reads at most this much: twice the document API's 100 MB upload
# cap, so a real input (or a fitz-cut sub-PDF of one) never meets it — while
# a worker pointing ``doc.pdf`` at an endless source (a FIFO times out, a
# device is capped) cannot fill the host disk through the volume.
COPY_IN_MAX_MB = 200

COPY_IN_TIMEOUT_SECONDS = 60
COPY_OUT_TIMEOUT_SECONDS = 90
KILL_TIMEOUT_SECONDS = 30          # docker rm -f of a container past its deadline
VOLUME_RM_TIMEOUT_SECONDS = 30

# Everything the launcher may do besides the deadline itself, plus slack for
# the HTTP round trip. The worker waits this much longer than the deadline,
# so the launcher always answers first.
LAUNCHER_REPLY_MARGIN_SECONDS = (COPY_IN_TIMEOUT_SECONDS + KILL_TIMEOUT_SECONDS
                                 + COPY_OUT_TIMEOUT_SECONDS
                                 + VOLUME_RM_TIMEOUT_SECONDS + 30)

LAUNCHER_PORT = 8765

# The job name names the containers, the volumes and the job directory — its
# alphabet must never carry '/', '..' or ':'.
JOB_NAME_RE = re.compile(r'mineru_[0-9a-f]{12}')


def new_job_name():
    """A fresh job name (worker side) — the launcher validates the same shape."""
    return f'mineru_{uuid.uuid4().hex[:12]}'


def is_job_name(value):
    # fullmatch, never ``match`` with ``$``: ``$`` also matches before a
    # trailing newline.
    return isinstance(value, str) and JOB_NAME_RE.fullmatch(value) is not None


def volume_names(job):
    """``(<job>_in, <job>_out)`` — Docker creates them on first use."""
    return f'{job}_in', f'{job}_out'


def mineru_run_timeout_for(page_count, base_seconds=MINERU_TIMEOUT_BASE_SECONDS):
    """Deadline in seconds for ONE mineru container run over ``page_count`` pages."""
    n = page_count if isinstance(page_count, int) and page_count > 0 else 1
    return base_seconds + MINERU_TIMEOUT_PER_PAGE_SECONDS * n


def launcher_reply_timeout_for(page_count):
    """How long the worker waits for the launcher's answer to one run."""
    return mineru_run_timeout_for(page_count) + LAUNCHER_REPLY_MARGIN_SECONDS


def build_run_argv(*, image, in_source, out_source, pdf_name, container_name,
                   models_dir_host=None):
    """The measured run vector (module docstring) plus ``--name``."""
    argv = ['docker', 'run', '--name', container_name,
            '--rm', '--gpus', 'all', '--shm-size', '16g',
            '-v', f'{in_source}:/in:ro', '-v', f'{out_source}:/out']
    if models_dir_host:
        argv += ['-v', f'{models_dir_host}:/models']
    argv += ['-e', 'HF_HOME=/models', '-e', 'MINERU_MODEL_SOURCE=huggingface',
             image,
             'mineru', '-p', f'/in/{pdf_name}', '-o', '/out', '-b', MINERU_BACKEND]
    return argv


def build_copy_in_argv(*, exchange_host, job, pdf_name, in_volume, container_name):
    """The worker's input into the run's volume — read from the exchange ROOT
    mounted read-only, so a symlink below it resolves inside the helper."""
    return ['docker', 'run', '--name', container_name, '--rm', '--network', 'none',
            '-v', f'{exchange_host}:/x:ro', '-v', f'{in_volume}:/in',
            HELPER_IMAGE, 'dd', f'if=/x/{job}/in/{pdf_name}', f'of=/in/{pdf_name}',
            'bs=1M', f'count={COPY_IN_MAX_MB}']


def build_copy_out_argv(*, exchange_host, job, out_volume, owner, container_name):
    """The run's output back into the job directory, written AS ``owner``."""
    return ['docker', 'run', '--name', container_name, '--rm', '--network', 'none',
            '--user', owner,
            '-v', f'{out_volume}:/o:ro', '-v', f'{exchange_host}:/x',
            HELPER_IMAGE, 'cp', '-r', '/o/.', f'/x/{job}/out']


def build_kill_argv(container_name):
    return ['docker', 'rm', '-f', container_name]


def build_volume_rm_argv(*volumes):
    return ['docker', 'volume', 'rm', *volumes]
