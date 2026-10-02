#!/usr/bin/env python3
"""Live probe for JOB-ID-REUSE: the wrong-file correction path, end to end.

``Conversion.id`` is a plain SQLite ``INTEGER PRIMARY KEY`` — the id of a
deleted highest row goes straight to the next insert. Until JOB-ID-REUSE the
job files on the shared volume were named after that id, so "wrong file →
delete the pending row → right file" handed the OLD job's result to the NEW
row. The pytest suite replays that with the tasks in-process; this is the one
check against the REAL stack (web, Redis, RQ worker, mineru launcher):

1. submit PDF A (mode ``lokal`` — a real mineru run, about a minute);
2. delete A's row while it is still ``pending`` (its job is already running);
3. submit a DIFFERENT PDF B and **print whether B got A's id** — if it did
   not (somebody else inserted a row in between), the case is NOT probed and
   the script says so and exits 2;
4. poll B until ``ready`` and check that the content is B's, not A's;
5. inventory the job directory: what job A left behind, by name.

"B's, not A's" is checked without trusting OCR text: A and B must have
different page counts (the result carries one provenance entry per page), and
if job A left its result on the volume, B's markdown must differ from it.

It runs INSIDE the web container (it reads the volume the web side mounts),
against the deployed app, as a THROWAWAY user — never Oli's account. Auth is
the per-user bearer (``POST /api/auth/login``), so no browser and no CSRF.

How to run (Mintbox, ~3 min — two real mineru runs):

    # 1. throwaway user; the password only ever travels via -e
    docker exec markdown-converter-web flask --app app create-user zz_probe --password '<random>'
    # 2. two DIFFERENT small PDFs with different page counts + the script
    #    (SEC-NONROOT: stream files in, never docker cp a foreign-owned file)
    docker exec -i markdown-converter-web sh -c 'cat > /tmp/probe_a.pdf' < a.pdf
    docker exec -i markdown-converter-web sh -c 'cat > /tmp/probe_b.pdf' < b.pdf
    docker exec -i markdown-converter-web sh -c 'cat > /tmp/probe.py' < scripts/probe_job_id_reuse.py
    # 3. run — exit 0 = probed and held, 1 = a check failed, 2 = not probed
    docker exec -e PROBE_USER=zz_probe -e PROBE_PASSWORD='<random>' markdown-converter-web python /tmp/probe.py
    # 4. clean up STRICTLY by user_id (the api_token table carries Oli's iOS
    #    tokens): the user's Conversion rows, ApiToken rows, the User row via
    #    the ORM; remove /tmp/probe* from the container; and remove the ONE
    #    orphan the script names — a job that was already running when its
    #    row was deleted writes a result nobody reads (the named property of
    #    JOB-ID-REUSE; it is never served, the mark is not reused).

Env: BASE_URL (default http://localhost:5000 — the container's own port),
PROBE_PDF_A (/tmp/probe_a.pdf), PROBE_PDF_B (/tmp/probe_b.pdf),
PROBE_VOLUME (/app/output_podcasts), PROBE_TIMEOUT_SECONDS (900).
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

BASE = os.environ.get('BASE_URL', 'http://localhost:5000')
USER = os.environ.get('PROBE_USER') or sys.exit('PROBE_USER missing')
PASSWORD = os.environ.get('PROBE_PASSWORD') or sys.exit('PROBE_PASSWORD missing')
PDF_A = os.environ.get('PROBE_PDF_A', '/tmp/probe_a.pdf')
PDF_B = os.environ.get('PROBE_PDF_B', '/tmp/probe_b.pdf')
VOLUME = os.environ.get('PROBE_VOLUME', '/app/output_podcasts')
TIMEOUT = int(os.environ.get('PROBE_TIMEOUT_SECONDS', '900'))
DOC_DIR = os.path.join(VOLUME, 'doc_conversions')

failures = []


def check(ok, what):
    print(('  PASS ' if ok else '  FAIL ') + what, flush=True)
    if not ok:
        failures.append(what)


def call(method, path, token=None, body=None, content_type=None):
    """One request → ``(status, parsed JSON or None)``. Never raises on 4xx/5xx."""
    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    if content_type:
        headers['Content-Type'] = content_type
    request = urllib.request.Request(BASE + path, data=body, method=method,
                                     headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, None


def submit(token, path):
    """Multipart submit of one PDF, mode ``lokal`` → ``(status, body)``."""
    boundary = uuid.uuid4().hex
    with open(path, 'rb') as f:
        data = f.read()
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="mode"\r\n\r\nlokal\r\n'.encode(),
        (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
         f'filename="{os.path.basename(path)}"\r\n'
         'Content-Type: application/pdf\r\n\r\n').encode() + data + b'\r\n',
        f'--{boundary}--\r\n'.encode(),
    ]
    return call('POST', '/api/document-conversions', token=token,
                body=b''.join(parts),
                content_type=f'multipart/form-data; boundary={boundary}')


def job_files():
    try:
        return sorted(os.listdir(DOC_DIR))
    except OSError:
        return []


def main():
    status, body = call('POST', '/api/auth/login',
                        body=json.dumps({'username': USER, 'password': PASSWORD}).encode(),
                        content_type='application/json')
    if status != 200 or not body or not body.get('token'):
        sys.exit(f'login failed: {status}')
    token = body['token']
    print(f'logged in as {USER}; job files before: {job_files()}', flush=True)

    # 1. submit A — the WRONG file
    status, a = submit(token, PDF_A)
    if status != 202:
        sys.exit(f'submit A: {status} {a} — a deduped or refused submit probes nothing')
    id_a, mark_a = a['id'], a['job_id']
    print(f'[A] submitted: id={id_a} job={mark_a} status={a["status"]} mode={a["mode"]}', flush=True)
    check(f'source_{mark_a}.pdf' in job_files(), "A's source lies under A's job mark")

    # 2. let the worker take the job, then delete the row while pending
    time.sleep(8)
    status, polled = call('GET', f'/api/document-conversions/{id_a}', token=token)
    print(f'[A] after 8 s: status={polled.get("status") if polled else status}', flush=True)
    if not polled or polled.get('status') != 'pending':
        sys.exit('A is no longer pending — choose a PDF that runs longer; nothing probed')
    status, _ = call('DELETE', f'/api/conversions/{id_a}', token=token)
    check(status == 200, f"A's pending row deleted ({status})")
    status, _ = call('GET', f'/api/document-conversions/{id_a}', token=token)
    check(status == 404, f'A is gone ({status})')
    check(f'source_{mark_a}.pdf' not in job_files(), "the delete removed A's source")

    # 3. submit B — the RIGHT file — and say whether it took A's id
    status, b = submit(token, PDF_B)
    if status != 202:
        sys.exit(f'submit B: {status} {b}')
    id_b, mark_b = b['id'], b['job_id']
    same_id = id_b == id_a
    print(f'[B] submitted: id={id_b} job={mark_b}', flush=True)
    print(f'[B] SAME ID AS A: {same_id} (A id={id_a}, B id={id_b})', flush=True)
    check(mark_b != mark_a, 'B runs under its own job mark')
    check(f'source_{mark_b}.pdf' in job_files(), "B's source lies under B's job mark")

    # 4. poll B until it is terminal; a `ready` before its own job ran would
    #    be A's result
    t0 = time.monotonic()
    seen = []
    final = None
    while time.monotonic() - t0 < TIMEOUT:
        status, polled = call('GET', f'/api/document-conversions/{id_b}', token=token)
        state = polled.get('status') if polled else f'http {status}'
        if not seen or seen[-1][1] != state:
            seen.append((round(time.monotonic() - t0, 1), state))
            print(f'[B] t={seen[-1][0]}s status={state} job files={job_files()}', flush=True)
        if state in ('ready', 'failed'):
            final = polled
            break
        time.sleep(3)
    if final is None:
        sys.exit(f'B did not finish within {TIMEOUT} s')
    check(final['status'] == 'ready', f'B ended {final["status"]} (error: {final.get("error")})')

    pages_b = final['source']['page_count']
    provenance = final.get('provenance') or []
    markdown_b = final.get('markdown') or ''
    print(f'[B] {len(markdown_b)} chars, {len(provenance)} provenance entries, '
          f'source page_count={pages_b}, degradations='
          f'{[d.get("code") for d in final.get("degradations") or []]}', flush=True)
    print(f'[B] starts: {markdown_b[:90]!r}', flush=True)
    check(final['source']['filename'] == os.path.basename(PDF_B), "B's row names B's file")
    check(len(provenance) == pages_b,
          f"B's result has one provenance entry per page of B ({len(provenance)} of {pages_b})")

    # 5. what job A left behind
    left = job_files()
    print(f'job files after: {left}', flush=True)
    orphan = f'result_{mark_a}.json'
    if orphan in left:
        with open(os.path.join(DOC_DIR, orphan), encoding='utf-8') as f:
            a_result = json.load(f)
        markdown_a = a_result.get('markdown') or ''
        print(f'[A] left ONE orphan: doc_conversions/{orphan} — {len(markdown_a)} chars, '
              f'{len(a_result.get("provenance") or [])} provenance entries', flush=True)
        print(f'[A] starts: {markdown_a[:90]!r}', flush=True)
        check(markdown_b != markdown_a, "B's markdown is not A's")
        check(len(a_result.get('provenance') or []) != len(provenance),
              'A and B differ in page count (the inputs must — otherwise this proves less)')
    else:
        print('[A] left no result (its job died without its source, or had not '
              'written yet) — B is compared by page count only', flush=True)
    check([name for name in left if mark_b in name] == [], "B's job files are discarded")
    check([name for name in left if name not in (orphan,)] == [],
          'nothing else lies in the job directory')

    if not same_id:
        print('\nNOT PROBED: B did not get the id of A — the id-reuse case was not '
              'exercised by this run.', flush=True)
        return 2
    print(f'\n{len(failures)} failure(s)' if failures
          else '\nprobed: B took the id of A and got its OWN result', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
