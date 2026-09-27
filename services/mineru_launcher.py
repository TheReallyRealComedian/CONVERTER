"""mineru launcher (SEC-SOCKET) — the ONE holder of the host's docker socket.

The worker parses foreign documents (pandoc, unstructured, PyMuPDF, mineru
output); code execution there must not become root over the host. Until
SEC-SOCKET the worker mounted ``/var/run/docker.sock`` to start the mineru
sibling container itself — whoever ran code in the worker could
``docker run -v /:/host`` and own every container on the box. A socket proxy
would not have closed that: it filters path and method, never the body, and
``POST /containers/create`` with ``HostConfig.Binds=["/:/host"]`` passes as
soon as ``create`` is allowed — which it must be.

So the allow-list is this service. It can do exactly ONE thing: run the
measured mineru vector over one job of the exchange (``services.
mineru_invocation`` — copy-in, run, copy-out, cleanup). It builds every
command line itself from its own environment; the worker sends data, never
arguments. And it never hands the daemon a path the worker could shape: the
daemon dereferences bind sources, so a worker-controlled ``out`` directory
turned symlink would have mounted any host path into a root container (see
the invocation module). A compromised worker can make mineru run over files
in the exchange directory, one run at a time — nothing it could not do
anyway.

Deliberately minimal, because root equivalence lives here now: stdlib plus
``services.mineru_invocation``, two routes, strict JSON, ``subprocess.run``
with LIST arguments and no shell, no import from ``app_pkg``. It parses
nothing that stems from a document and never sees the exchange (no mount).

Routes::

    GET  /health → 200 {"status": "ok", "busy": bool}
    POST /run    {"job": "mineru_<12 hex>", "pdf_name": "doc.pdf", "page_count": n}
                 → 200 {"returncode", "timed_out", "deadline_seconds",
                        "stdout_tail", "stderr_tail"}
                 → 400 invalid request · 409 a run is in progress
                 → 413 body too large · 422 input not copyable
                 → 500 output not copyable / docker CLI not runnable
                 → 503 not configured / stopping

Environment (moved here from the worker)::

    DOC_LOCAL_EXCHANGE_HOST_DIR  the exchange ROOT as the HOST daemon sees
                                 it (required)
    EXCHANGE_OWNER               <uid>:<gid> the output is written as — the
                                 WORKER's ids (required; 0:0 while the
                                 worker runs as root, SEC-NONROOT changes it)
    MINERU_IMAGE                 default mineru:latest (compose pins 3.4.4)
    MINERU_MODELS_DIR            host path of the HF cache, empty = no mount
    MINERU_TIMEOUT_BASE_SECONDS  can only SHORTEN the deadline base (the
                                 kill probe); never lengthens it

The launcher owns the deadline: ``docker run --name <job>`` under
``subprocess.run(timeout=)``, and on expiry ``docker rm -f <job>`` BEFORE it
answers — no container outlives its deadline; every helper is named and
bounded the same way. SIGTERM (compose stop / redeploy) removes a running
container and the job's volumes before the process exits. Configuration
errors answer 503 per run instead of failing the health check: the worker
depends on this service being healthy, and a mineru misconfiguration must
cost the local PDF path, not narration and transcription.
"""
import json
import logging
import os
import re
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import NamedTuple

from services.mineru_invocation import (
    COPY_IN_TIMEOUT_SECONDS,
    COPY_OUT_TIMEOUT_SECONDS,
    KILL_TIMEOUT_SECONDS,
    LAUNCHER_PORT,
    MINERU_DEFAULT_IMAGE,
    MINERU_MAX_PAGES,
    MINERU_TIMEOUT_BASE_SECONDS,
    VOLUME_RM_TIMEOUT_SECONDS,
    build_copy_in_argv,
    build_copy_out_argv,
    build_kill_argv,
    build_run_argv,
    build_volume_rm_argv,
    is_job_name,
    mineru_run_timeout_for,
    volume_names,
)

logger = logging.getLogger('mineru_launcher')

MAX_BODY_BYTES = 4096
TAIL_CHARS = 800
REQUEST_FIELDS = frozenset({'job', 'pdf_name', 'page_count'})
# One file name, never a path: the alphabet rules out '/', '..', whitespace
# and control characters by construction.
PDF_NAME_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}\.pdf')
OWNER_RE = re.compile(r'[0-9]{1,10}:[0-9]{1,10}')
# Control characters never reach the log verbatim (request lines are
# client-controlled; the stdlib escapes them only in ITS log_message).
_LOG_ESCAPE = str.maketrans({c: f'\\x{c:02x}' for c in [*range(0x20), *range(0x7f, 0xa0)]})

# At most ONE run at a time: a second POST /run gets 409 instead of queueing.
# The lock enforces it — the server is threaded only so /health and the 409
# still answer while a run holds the lock.
_RUN_LOCK = threading.Lock()
_STOPPING = threading.Event()
_current_job = None        # the job whose volumes exist right now
_current_container = None  # the container the running step holds


class RequestError(Exception):
    """A request this launcher refuses or cannot serve — status + message."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


class LauncherConfig(NamedTuple):
    exchange_host: str
    owner: str
    image: str
    models_dir_host: str | None
    timeout_base: int


# -- request + config validation (fail-closed) -------------------------------

def validate_run_request(payload):
    """``(job, pdf_name, page_count)`` or RequestError(400)."""
    if not isinstance(payload, dict):
        raise RequestError(400, 'Anfrage muss ein JSON-Objekt sein.')
    unknown = sorted(set(payload) - REQUEST_FIELDS)
    if unknown:
        names = ', '.join(repr(name) for name in unknown)
        raise RequestError(400, f'Unbekannte Felder: {names[:200]}.')
    missing = sorted(REQUEST_FIELDS - set(payload))
    if missing:
        raise RequestError(400, f'Felder fehlen: {", ".join(missing)}.')
    job = payload['job']
    if not is_job_name(job):
        raise RequestError(400, 'job ungültig (erwartet mineru_ und 12 Hex-Zeichen).')
    pdf_name = payload['pdf_name']
    if not (isinstance(pdf_name, str) and PDF_NAME_RE.fullmatch(pdf_name)):
        raise RequestError(400, 'pdf_name ungültig (ein Dateiname auf .pdf, ohne Pfad).')
    page_count = payload['page_count']
    if (not isinstance(page_count, int) or isinstance(page_count, bool)
            or not 1 <= page_count <= MINERU_MAX_PAGES):
        raise RequestError(
            400, f'page_count muss eine ganze Zahl von 1 bis {MINERU_MAX_PAGES} sein.')
    return job, pdf_name, page_count


def _timeout_base(raw):
    """The deadline base can only be SHORTENED by env (the kill probe): a
    longer launcher deadline would outrun the worker's HTTP timeout, which
    is computed from the default. Junk, <= 0 or larger → the default."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return MINERU_TIMEOUT_BASE_SECONDS
    return value if 0 < value <= MINERU_TIMEOUT_BASE_SECONDS else MINERU_TIMEOUT_BASE_SECONDS


def _is_host_path(value):
    # ':' would silently change a ``-v src:dst[:opts]`` spec.
    return value.startswith('/') and ':' not in value


def read_config(environ=None):
    """Launcher configuration from env, or RequestError(503)."""
    env = os.environ if environ is None else environ
    exchange_host = (env.get('DOC_LOCAL_EXCHANGE_HOST_DIR') or '').rstrip('/')
    if not _is_host_path(exchange_host):
        raise RequestError(503, 'Launcher nicht konfiguriert: DOC_LOCAL_EXCHANGE_HOST_DIR '
                                'fehlt oder ist kein absoluter Host-Pfad.')
    owner = env.get('EXCHANGE_OWNER') or ''
    if not OWNER_RE.fullmatch(owner):
        raise RequestError(503, 'Launcher nicht konfiguriert: EXCHANGE_OWNER fehlt '
                                '(erwartet <uid>:<gid>).')
    models = env.get('MINERU_MODELS_DIR') or None
    if models is not None and not _is_host_path(models):
        raise RequestError(503, 'Launcher nicht konfiguriert: MINERU_MODELS_DIR ist '
                                'kein absoluter Host-Pfad.')
    return LauncherConfig(
        exchange_host=exchange_host,
        owner=owner,
        image=env.get('MINERU_IMAGE') or MINERU_DEFAULT_IMAGE,
        models_dir_host=models,
        timeout_base=_timeout_base(env.get('MINERU_TIMEOUT_BASE_SECONDS')),
    )


# -- the run -----------------------------------------------------------------

def _tail(text):
    # TimeoutExpired carries BYTES even under text=True (documented).
    if text is None:
        return ''
    if isinstance(text, bytes):
        text = text.decode('utf-8', 'replace')
    return text[-TAIL_CHARS:]


def _docker(argv, timeout):
    """One bounded docker CLI call that must not raise (cleanup steps)."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        return None, str(e)
    return proc.returncode, _tail(proc.stderr)


def _remove_container(name):
    """``docker rm -f <name>`` — never raises; a failure is logged loudly."""
    returncode, stderr = _docker(build_kill_argv(name), KILL_TIMEOUT_SECONDS)
    if returncode != 0 and 'No such container' not in stderr:
        logger.error('Container %s nicht entfernbar: rc=%s %s', name, returncode, stderr)
        return False
    logger.warning('Container %s entfernt.', name)
    return True


def _remove_volumes(job):
    """Both volumes of the job — a volume that never came to exist (the
    copy-in failed before the run created ``<job>_out``) is not an error."""
    returncode, stderr = _docker(build_volume_rm_argv(*volume_names(job)),
                                 VOLUME_RM_TIMEOUT_SECONDS)
    problems = [line for line in stderr.splitlines()
                if line.strip() and 'no such volume' not in line.lower()]
    if returncode != 0 and (problems or returncode is None):
        logger.error('Volumes von %s nicht entfernbar: rc=%s %s', job, returncode, stderr)


def _run_named(argv, name, timeout):
    """Run one named container; at the deadline remove it before returning.

    ``(returncode | None, timed_out, stdout_tail, stderr_tail)``. OSError (no
    docker CLI) propagates.
    """
    global _current_container
    _current_container = name
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        # The timeout killed the docker CLI client, not the container.
        _remove_container(name)
        return None, True, _tail(e.stdout), _tail(e.stderr)
    finally:
        _current_container = None
    return proc.returncode, False, _tail(proc.stdout), _tail(proc.stderr)


def _helper(argv, name, timeout, status, what):
    """A copy step: success, or RequestError(status) naming what failed."""
    returncode, timed_out, _, stderr = _run_named(argv, name, timeout)
    if timed_out:
        raise RequestError(status, f'{what}: Frist {timeout} s gerissen.')
    if returncode != 0:
        raise RequestError(status, f'{what}: rc={returncode} {stderr.strip()}'[:400])


def execute_run(job, pdf_name, page_count, config):
    """The one thing this service does. The caller holds ``_RUN_LOCK``."""
    global _current_job
    in_volume, out_volume = volume_names(job)
    deadline = mineru_run_timeout_for(page_count, config.timeout_base)
    _current_job = job
    try:
        _helper(build_copy_in_argv(exchange_host=config.exchange_host, job=job,
                                   pdf_name=pdf_name, in_volume=in_volume,
                                   container_name=f'{job}_copyin'),
                f'{job}_copyin', COPY_IN_TIMEOUT_SECONDS, 422,
                'Eingabe nicht übernehmbar')
        argv = build_run_argv(image=config.image, in_source=in_volume,
                              out_source=out_volume, pdf_name=pdf_name,
                              container_name=job,
                              models_dir_host=config.models_dir_host)
        logger.info('Lauf %s: %d Seiten, Frist %d s, Image %s',
                    job, page_count, deadline, config.image)
        started = time.monotonic()
        returncode, timed_out, stdout, stderr = _run_named(argv, job, deadline)
        logger.info('Lauf %s: %s nach %.1f s', job,
                    'Frist gerissen, Container entfernt' if timed_out
                    else f'rc={returncode}', time.monotonic() - started)
        if returncode == 0 and not _STOPPING.is_set():
            _helper(build_copy_out_argv(exchange_host=config.exchange_host, job=job,
                                        out_volume=out_volume, owner=config.owner,
                                        container_name=f'{job}_copyout'),
                    f'{job}_copyout', COPY_OUT_TIMEOUT_SECONDS, 500,
                    'Ausgabe nicht übernehmbar')
        return {'returncode': returncode, 'timed_out': timed_out,
                'deadline_seconds': deadline,
                'stdout_tail': stdout, 'stderr_tail': stderr}
    finally:
        _remove_volumes(job)
        _current_job = None


# -- HTTP --------------------------------------------------------------------

def _no_duplicate_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('doppelter Schlüssel')
        obj[key] = value
    return obj


def _reject_constant(_name):
    raise ValueError('NaN/Infinity ist kein JSON')


class LauncherHandler(BaseHTTPRequestHandler):
    server_version = 'mineru-launcher'
    sys_version = ''
    # Socket timeout while reading a request: a stalled client cannot pin a
    # thread. It never touches a run — the socket is idle while mineru works.
    timeout = 30

    def log_message(self, format, *args):  # noqa: A002 — stdlib signature
        logger.info('%s %s', self.address_string(), (format % args).translate(_LOG_ESCAPE))

    def log_request(self, code='-', size='-'):
        if self.path != '/health':  # the healthcheck would flood the log
            super().log_request(code, size)

    def _send(self, status, body):
        data = json.dumps(body, ensure_ascii=False).encode('utf-8')
        try:
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            logger.warning('Antwort %d nicht zustellbar: der Client ist weg.', status)

    def do_GET(self):
        if self.path == '/health':
            self._send(200, {'status': 'ok', 'busy': _RUN_LOCK.locked()})
        elif self.path == '/run':
            self._send(405, {'error': 'Nur POST.'})
        else:
            self._send(404, {'error': 'Unbekannter Pfad.'})

    def _read_json(self):
        ctype = (self.headers.get('Content-Type') or '').split(';')[0].strip().lower()
        if ctype != 'application/json':
            raise RequestError(400, 'Content-Type muss application/json sein.')
        try:
            length = int(self.headers.get('Content-Length', ''))
        except ValueError:
            raise RequestError(400, 'Content-Length fehlt.') from None
        if length < 0:
            raise RequestError(400, 'Content-Length ungültig.')
        if length > MAX_BODY_BYTES:
            raise RequestError(413, 'Anfrage zu groß.')
        try:
            raw = self.rfile.read(length)
        except OSError:
            raise RequestError(400, 'Anfrage nicht lesbar.') from None
        if len(raw) != length:
            raise RequestError(400, 'Anfrage unvollständig.')
        try:
            return json.loads(raw.decode('utf-8'),
                              object_pairs_hook=_no_duplicate_keys,
                              parse_constant=_reject_constant)
        except ValueError:  # UnicodeDecodeError + JSONDecodeError included
            raise RequestError(400, 'Kein gültiges JSON.') from None

    def do_POST(self):
        if self.path != '/run':
            if self.path == '/health':
                self._send(405, {'error': 'Nur GET.'})
            else:
                self._send(404, {'error': 'Unbekannter Pfad.'})
            return
        try:
            job, pdf_name, page_count = validate_run_request(self._read_json())
            config = read_config()
        except RequestError as e:
            logger.warning('POST /run abgewiesen (%d): %s', e.status,
                           e.message.translate(_LOG_ESCAPE))
            self._send(e.status, {'error': e.message})
            return
        if _STOPPING.is_set():
            self._send(503, {'error': 'Launcher wird beendet.'})
            return
        if not _RUN_LOCK.acquire(blocking=False):
            logger.warning('POST /run abgewiesen (409): %s, ein Lauf ist aktiv.', job)
            self._send(409, {'error': 'Es läuft bereits ein mineru-Lauf.'})
            return
        try:
            status, body = 200, execute_run(job, pdf_name, page_count, config)
        except RequestError as e:
            logger.error('Lauf %s: %s', job, e.message.translate(_LOG_ESCAPE))
            status, body = e.status, {'error': e.message}
        except OSError as e:
            logger.error('Lauf %s: docker-CLI nicht ausführbar: %s', job, e)
            status, body = 500, {'error': f'docker-CLI nicht ausführbar: {e}'}
        except Exception as e:  # never a dropped connection, never a traceback
            logger.exception('Lauf %s: unerwarteter Fehler', job)
            status, body = 500, {'error': f'Unerwarteter Fehler im Launcher: {type(e).__name__}'}
        finally:
            _RUN_LOCK.release()
        self._send(status, body)


class LauncherServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(host='0.0.0.0', port=LAUNCHER_PORT):
    return LauncherServer((host, port), LauncherHandler)


def _terminate(_signum, _frame):
    """SIGTERM: remove the running container and the job's volumes, then
    exit — the orphan rule holds across compose stop and redeploy too."""
    _STOPPING.set()
    container, job = _current_container, _current_job
    if container:
        logger.warning('SIGTERM während %s — Container wird entfernt.', container)
        _remove_container(container)
    if job:
        _remove_volumes(job)
    raise SystemExit(0)


def _log_config_state():
    try:
        config = read_config()
    except RequestError as e:
        logger.error('%s Läufe werden mit 503 abgewiesen.', e.message)
        return
    logger.info('Konfiguration: Image %s, Austausch %s, Modelle %s, Eigentümer %s, '
                'Frist-Basis %d s', config.image, config.exchange_host,
                config.models_dir_host or '—', config.owner, config.timeout_base)


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    signal.signal(signal.SIGTERM, _terminate)
    server = make_server()
    _log_config_state()
    logger.info('lauscht auf :%d', server.server_address[1])
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
