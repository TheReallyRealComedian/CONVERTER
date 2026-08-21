#!/usr/bin/env python3
"""Does a long synchronous conversion block the app? (DOC-WEB-ASYNC P1)

Measures, through the real HTTP surface, whether other requests are served
WHILE ``POST /transform-document`` runs a long PDF conversion on the single
gunicorn worker. Three probes (``/login`` unauthenticated, ``/library`` and
``/api/collections`` with a session) are fired on a fixed timer — each in
its own thread, so a blocked probe never delays the next one — first at
idle (baseline), then during one conversion, then during ``--concurrent``
conversions started at the same instant.

Run it on the host next to the app (``http://localhost:5656``) with a
throwaway user (the web path needs a session):

    python3 scripts/measure_sync_blocking.py \
        --base-url http://localhost:5656 --user zz_smoke --password ... \
        --pdf corpus/05_scan-sauber/dahlhaus_beethoven-kritik_gerastert-300dpi.pdf \
        --out /tmp/measure.json

Needs only ``requests``. Prints a summary table and writes every sample to
``--out`` (JSON) so the numbers in the sprint report are reproducible.
"""
import argparse
import json
import re
import statistics
import sys
import threading
import time

import requests

PROBE_TIMEOUT_SECONDS = 600  # a blocked probe must show its TRUE latency
CSRF_INPUT_RE = re.compile(r'name="csrf_token"\s+value="([^"]+)"')


# --- session plumbing ---------------------------------------------------------

def login(base_url, username, password):
    """Form login (CSRF hidden field) → authenticated ``requests.Session``."""
    session = requests.Session()
    page = session.get(f'{base_url}/login', timeout=60)
    page.raise_for_status()
    match = CSRF_INPUT_RE.search(page.text)
    if not match:
        sys.exit('login page carries no csrf_token field')
    resp = session.post(f'{base_url}/login', data={
        'username': username, 'password': password,
        'csrf_token': match.group(1),
    }, timeout=60, allow_redirects=False)
    if resp.status_code != 302 or '/login' in resp.headers.get('Location', ''):
        sys.exit(f'login failed: {resp.status_code} {resp.headers.get("Location")}')
    check = session.get(f'{base_url}/api/collections', timeout=60)
    if check.status_code != 200:
        sys.exit(f'session check failed: {check.status_code}')
    return session


def clone_session(session):
    """Own Session object per thread (cookie jar updates are not thread-safe)."""
    clone = requests.Session()
    clone.cookies.update(session.cookies)
    clone.headers.update(session.headers)
    return clone


def csrf_token(session, base_url):
    resp = session.get(f'{base_url}/api/csrf-token', timeout=60)
    resp.raise_for_status()
    return resp.json()['csrf_token']


# --- probes -------------------------------------------------------------------

def probe_once(base_url, name, session):
    """One request, timed. Returns a sample dict."""
    t0 = time.monotonic()
    sent_at = time.time()
    try:
        if name == 'login':
            # Fresh unauthenticated client, no redirect following.
            resp = requests.get(f'{base_url}/login', timeout=PROBE_TIMEOUT_SECONDS,
                                allow_redirects=False)
        elif name == 'library':
            resp = session.get(f'{base_url}/library', timeout=PROBE_TIMEOUT_SECONDS,
                               allow_redirects=False)
        elif name == 'api':
            resp = session.get(f'{base_url}/api/collections',
                               timeout=PROBE_TIMEOUT_SECONDS, allow_redirects=False)
        else:
            raise ValueError(name)
        status = resp.status_code
    except Exception as e:  # timeout / connection error is a data point too
        status = f'ERR {type(e).__name__}'
    return {'probe': name, 'sent_at': sent_at,
            'latency_s': round(time.monotonic() - t0, 3), 'status': status}


class ProbeScheduler:
    """Fires all three probes every ``interval`` seconds, each in its own
    thread, until stopped; collects samples."""

    PROBES = ('login', 'library', 'api')

    def __init__(self, base_url, session, interval):
        self.base_url = base_url
        self.session = session
        self.interval = interval
        self.samples = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = []
        self._runner = None

    def _fire(self, name):
        sample = probe_once(self.base_url, name, clone_session(self.session))
        with self._lock:
            self.samples.append(sample)

    def _loop(self):
        while not self._stop.is_set():
            for name in self.PROBES:
                t = threading.Thread(target=self._fire, args=(name,), daemon=True)
                t.start()
                self._threads.append(t)
            self._stop.wait(self.interval)

    def start(self):
        self._runner = threading.Thread(target=self._loop, daemon=True)
        self._runner.start()

    def stop(self, drain_seconds=PROBE_TIMEOUT_SECONDS):
        self._stop.set()
        self._runner.join()
        for t in self._threads:
            t.join(drain_seconds)
        with self._lock:
            return list(self.samples)


# --- conversions --------------------------------------------------------------

def run_conversion(base_url, session, token, pdf_path, label, results):
    """POST /transform-document, timed; appends a result dict."""
    started_at = time.time()
    t0 = time.monotonic()
    entry = {'label': label, 'started_at': started_at}
    try:
        with open(pdf_path, 'rb') as f:
            resp = session.post(
                f'{base_url}/transform-document',
                files={'document_file': (pdf_path.rsplit('/', 1)[-1], f, 'application/pdf')},
                headers={'X-CSRFToken': token},
                timeout=PROBE_TIMEOUT_SECONDS * 2)
        entry['status'] = resp.status_code
        try:
            body = resp.json()
        except ValueError:
            body = {}
        entry['markdown_chars'] = len(body.get('markdown') or '')
        entry['degradations'] = [d.get('code') for d in body.get('degradations') or []]
        entry['error'] = body.get('error')
    except Exception as e:
        entry['status'] = f'ERR {type(e).__name__}'
    entry['duration_s'] = round(time.monotonic() - t0, 1)
    entry['ended_at'] = time.time()
    results.append(entry)


# --- reporting ----------------------------------------------------------------

def summarize(samples, t_ref=None):
    """Per-probe stats. With ``t_ref`` (conversion start) the samples are
    annotated by their send offset so the report can show WHEN they hung."""
    out = {}
    for name in ProbeScheduler.PROBES:
        lat = [s['latency_s'] for s in samples if s['probe'] == name]
        if not lat:
            continue
        statuses = sorted({str(s['status']) for s in samples if s['probe'] == name})
        out[name] = {
            'n': len(lat),
            'median_s': round(statistics.median(lat), 3),
            'p90_s': round(sorted(lat)[max(0, int(len(lat) * 0.9) - 1)], 3),
            'max_s': round(max(lat), 3),
            'statuses': statuses,
        }
    if t_ref is not None:
        for s in samples:
            s['offset_s'] = round(s['sent_at'] - t_ref, 1)
    return out


def print_table(title, stats):
    print(f'\n{title}')
    print(f'  {"probe":8} {"n":>3} {"median":>9} {"p90":>9} {"max":>9}  statuses')
    for name, st in stats.items():
        print(f'  {name:8} {st["n"]:>3} {st["median_s"]:>8.3f}s {st["p90_s"]:>8.3f}s '
              f'{st["max_s"]:>8.3f}s  {",".join(st["statuses"])}')


def print_timeline(samples, width=None):
    """Compact per-sample view during a conversion: offset → latency."""
    print('  offset  probe    latency  status')
    for s in sorted(samples, key=lambda x: (x['sent_at'], x['probe'])):
        print(f'  {s.get("offset_s", 0):>6.1f}s {s["probe"]:8} {s["latency_s"]:>7.3f}s  {s["status"]}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--base-url', default='http://localhost:5656')
    ap.add_argument('--user', required=True)
    ap.add_argument('--password', required=True)
    ap.add_argument('--pdf', required=True, help='a PDF that converts for ~1 min or more')
    ap.add_argument('--probe-interval', type=float, default=5.0)
    ap.add_argument('--baseline-seconds', type=float, default=20.0)
    ap.add_argument('--concurrent', type=int, default=2,
                    help='conversions started at the same instant in the last phase (0 = skip)')
    ap.add_argument('--out', default=None, help='write all samples as JSON')
    args = ap.parse_args()

    session = login(args.base_url, args.user, args.password)
    token = csrf_token(session, args.base_url)
    settings = session.get(f'{args.base_url}/api/document-conversions/settings',
                           timeout=60).json()
    print(f'logged in as {args.user}; default_mode={settings.get("default_mode")}')

    report = {'base_url': args.base_url, 'pdf': args.pdf, 'settings': settings,
              'probe_interval_s': args.probe_interval}

    # Phase A — baseline at idle.
    sched = ProbeScheduler(args.base_url, session, args.probe_interval)
    sched.start()
    time.sleep(args.baseline_seconds)
    baseline = sched.stop()
    report['baseline'] = {'samples': baseline, 'stats': summarize(baseline)}
    print_table(f'A. Baseline (idle, {args.baseline_seconds:.0f} s)', report['baseline']['stats'])

    # Phase B — one conversion, probes on the timer meanwhile.
    results = []
    sched = ProbeScheduler(args.base_url, session, args.probe_interval)
    sched.start()
    time.sleep(args.probe_interval)  # one probe round before the upload
    worker = threading.Thread(target=run_conversion, args=(
        args.base_url, clone_session(session), token, args.pdf, 'single', results))
    worker.start()
    worker.join()
    time.sleep(args.probe_interval * 2)  # a couple of rounds after it ended
    during = sched.stop()
    conv = results[0]
    stats = summarize(during, t_ref=conv['started_at'])
    report['single'] = {'conversion': conv, 'samples': during, 'stats': stats}
    print(f'\nB. One conversion: status={conv["status"]} duration={conv["duration_s"]} s '
          f'markdown_chars={conv.get("markdown_chars")} degradations={conv.get("degradations")}')
    if conv['status'] != 200:
        # A rejected upload (413 above MAX_SYNC_PDF_PAGES, 400, ...) never ran
        # an engine — the probe table below would read as "no blocking"
        # while nothing was measured. Fail loudly instead.
        sys.exit(f'conversion did not run (status {conv["status"]}: '
                 f'{conv.get("error")}) — pick a PDF the web path accepts')
    print_table('   probes during/around it', stats)
    print_timeline(during)

    # Phase C — N conversions started at the same instant.
    if args.concurrent > 0:
        results = []
        sched = ProbeScheduler(args.base_url, session, args.probe_interval)
        sched.start()
        time.sleep(args.probe_interval)
        threads = [threading.Thread(target=run_conversion, args=(
            args.base_url, clone_session(session), token, args.pdf, f'c{i + 1}', results))
            for i in range(args.concurrent)]
        t_launch = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        time.sleep(args.probe_interval * 2)
        during_c = sched.stop()
        stats_c = summarize(during_c, t_ref=t_launch)
        results.sort(key=lambda r: r['started_at'])
        wall = max(r['ended_at'] for r in results) - t_launch
        total = sum(r['duration_s'] for r in results)
        report['concurrent'] = {'launched_at': t_launch, 'conversions': results,
                                'wall_s': round(wall, 1), 'sum_durations_s': round(total, 1),
                                'samples': during_c, 'stats': stats_c}
        print(f'\nC. {args.concurrent} conversions launched together: wall={wall:.1f} s, '
              f'sum of durations={total:.1f} s')
        for r in results:
            print(f'   {r["label"]}: start+{r["started_at"] - t_launch:.1f}s '
                  f'end+{r["ended_at"] - t_launch:.1f}s duration={r["duration_s"]} s '
                  f'status={r["status"]} chars={r.get("markdown_chars")} '
                  f'degradations={r.get("degradations")}')
        print_table('   probes during/around them', stats_c)
        print_timeline(during_c)

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(report, f, indent=1)
        print(f'\nsamples written to {args.out}')


if __name__ == '__main__':
    main()
