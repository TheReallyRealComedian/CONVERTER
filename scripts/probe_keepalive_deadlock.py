#!/usr/bin/env python3
"""Is the asgiref keep-alive deadlock reproducible? (SYNC-FREEZE P1)

Reproduces the sporadic 500 ``RuntimeError: Single thread executor already
being used, would deadlock`` that DOC-WEB-ASYNC P2 hit in the browser smoke.
Mechanics (asgiref 3.8.1 ``sync.py`` + uvicorn ``h11_impl.py``): the WSGI
call runs on ``SyncToAsync.single_thread_executor`` with
``deadlock_context=True`` set in the request task's context; every
``send()`` from that thread goes through ``AsyncToSync`` → ``main_wrap`` →
``_restore_context`` copies the flag into the send task. When the NEXT
request of the same keep-alive connection is already buffered at that
moment, uvicorn's ``on_response_complete`` starts the next cycle from
*inside* that send task — ``create_task`` inherits the flag, and the next
``run_wsgi_app`` raises before it runs.

Two client shapes, both on ONE connection:

* ``pipelined`` — N requests written in a single ``send()`` before any
  response is read. The next request is always buffered when the previous
  response completes → deterministic trigger (the mechanism, not the
  browser's behaviour — browsers do not pipeline).
* ``tight`` — strictly sequential: request N+1 is written the instant
  response N is fully read. This is what a fast click/poll does; it hits
  the window only sometimes (the race the smoke ran into).

Prints per shape: requests sent, outcome histogram, and whether the
connection survived. Outcomes on a path that is otherwise a 200: ``500`` is
the deadlock (uvicorn logs the RuntimeError server-side); ``hang`` is a
request the server accepted (keep-alive timer cancelled, connection kept
open) but never answered within ``--timeout`` — the second face of the same
context leak, seen on the Mintbox but not on a Mac loopback; ``closed`` is
EOF without a response (the server closing after a 500).

    python3 scripts/probe_keepalive_deadlock.py --base-url http://localhost:5656 \
        --path /login --pipelined 20 --tight 200 --timeout 5
"""
import argparse
import collections
import socket
import sys
import time
from urllib.parse import urlsplit


def _read_response(sock, buf):
    """Read ONE HTTP/1.1 response from ``sock`` (Content-Length or chunked),
    starting from the bytes already in ``buf``. Returns (status, rest)."""
    while b'\r\n\r\n' not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError('connection closed before headers')
        buf += chunk
    head, body = buf.split(b'\r\n\r\n', 1)
    lines = head.decode('latin1').split('\r\n')
    status = int(lines[0].split(' ', 2)[1])
    headers = {}
    for line in lines[1:]:
        k, _, v = line.partition(':')
        headers[k.strip().lower()] = v.strip()
    if headers.get('transfer-encoding', '').lower() == 'chunked':
        # consume chunks until the zero-size chunk
        while True:
            while b'\r\n' not in body:
                body += sock.recv(65536)
            size_line, body = body.split(b'\r\n', 1)
            size = int(size_line.split(b';')[0], 16)
            while len(body) < size + 2:
                body += sock.recv(65536)
            body = body[size + 2:]
            if size == 0:
                break
        return status, body
    length = int(headers.get('content-length', 0))
    while len(body) < length:
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError('connection closed mid-body')
        body += chunk
    return status, body[length:]


def _request_bytes(host, path):
    return (f'GET {path} HTTP/1.1\r\nHost: {host}\r\n'
            f'Connection: keep-alive\r\nUser-Agent: probe-keepalive-deadlock\r\n\r\n'
            ).encode('ascii')


def _connect(base, timeout):
    sock = socket.create_connection((base.hostname, base.port or 80), timeout=timeout)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


def _classify(exc):
    """Map a read failure onto an outcome bucket."""
    if isinstance(exc, socket.timeout):
        return 'hang'
    return 'closed'


def run_pipelined(base, path, n, timeout):
    sock = _connect(base, timeout)
    outcomes = collections.Counter()
    sock.sendall(_request_bytes(base.netloc, path) * n)
    buf = b''
    got = 0
    err = None
    try:
        for _ in range(n):
            status, buf = _read_response(sock, buf)
            outcomes[status] += 1
            got += 1
    except Exception as e:  # the server may close the connection after a 500
        outcomes[_classify(e)] += 1
        err = f'{type(e).__name__}: {e}'
    finally:
        sock.close()
    return {'shape': 'pipelined', 'sent': n, 'received': got,
            'outcomes': dict(outcomes), 'error': err}


def run_tight(base, path, n, timeout):
    sock = _connect(base, timeout)
    outcomes = collections.Counter()
    buf = b''
    got = 0
    err = None
    gaps = []
    try:
        for _ in range(n):
            t0 = time.monotonic()
            sock.sendall(_request_bytes(base.netloc, path))
            status, buf = _read_response(sock, buf)
            gaps.append(time.monotonic() - t0)
            outcomes[status] += 1
            got += 1
    except Exception as e:
        outcomes[_classify(e)] += 1
        err = f'{type(e).__name__}: {e}'
    finally:
        sock.close()
    return {'shape': 'tight', 'sent': n, 'received': got,
            'outcomes': dict(outcomes), 'error': err,
            'median_latency_ms': round(sorted(gaps)[len(gaps) // 2] * 1000, 1) if gaps else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--base-url', default='http://localhost:5656')
    ap.add_argument('--path', default='/login',
                    help='a path that answers without a session (default /login)')
    ap.add_argument('--pipelined', type=int, default=20,
                    help='requests written in one go on one connection (0 = skip)')
    ap.add_argument('--tight', type=int, default=200,
                    help='strictly sequential requests on one connection (0 = skip)')
    ap.add_argument('--repeat', type=int, default=3,
                    help='repeat each shape on a fresh connection this many times')
    ap.add_argument('--timeout', type=float, default=5.0,
                    help='seconds to wait for a response before calling it a hang')
    args = ap.parse_args()
    base = urlsplit(args.base_url)

    results = []
    for i in range(args.repeat):
        if args.pipelined:
            r = run_pipelined(base, args.path, args.pipelined, args.timeout)
            r['run'] = i + 1
            results.append(r)
            print(f'pipelined #{i + 1}: sent={r["sent"]} answered={r["received"]} '
                  f'outcomes={r["outcomes"]} error={r["error"]}', flush=True)
        if args.tight:
            r = run_tight(base, args.path, args.tight, args.timeout)
            r['run'] = i + 1
            results.append(r)
            print(f'tight     #{i + 1}: sent={r["sent"]} answered={r["received"]} '
                  f'outcomes={r["outcomes"]} median={r["median_latency_ms"]} ms '
                  f'error={r["error"]}', flush=True)

    total_500 = sum(r['outcomes'].get(500, 0) for r in results)
    total_hang = sum(r['outcomes'].get('hang', 0) for r in results)
    answered = sum(r['received'] for r in results)
    sent = sum(r['sent'] for r in results)
    print(f'\n{total_500} × 500, {total_hang} × hang; {answered} of {sent} requests answered'
          + (' — leak reproduced' if (total_500 or total_hang) else ' — clean'),
          flush=True)
    return 1 if (total_500 or total_hang) else 0


if __name__ == '__main__':
    sys.exit(main())
