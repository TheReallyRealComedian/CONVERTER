#!/usr/bin/env python3
"""Does the app stay correct under REAL concurrency? (SYNC-FREEZE P2)

``app_pkg/asgi.py`` runs sync views on a thread pool per gunicorn process.
Flask is built for that; the risk sits in the singletons and the database
under true simultaneity. This script drives the three risks at once through
the real HTTP surface, with a throwaway user, and checks the RESULTS, not
just the status codes:

* ``--audio``    one long Deepgram transcription (the longest sync path),
* ``--markdown`` TWO concurrent Markdown→PDF renders (``/convert-markdown``:
                 async view → Playwright → a Chromium per request),
* ``--raters``   N threads rating the user's cards round-robin — every
                 request on a FRESH connection, so the writes come from
                 several processes AND several threads per process (WAL +
                 busy_timeout on trial: the realistic "rating burst" case),
* probes every second (``/login``, ``/library``, ``/api/collections``) in
  their own threads, so parking shows up as latency.

Afterwards every card is re-read: ``rating_history`` must have grown by
EXACTLY the number of ratings that were answered 200 — no lost, no doubled
write. Against one process (``--workers 1``) this proves the same-process
thread concurrency; against N processes the cross-process one.

    python3 scripts/verify_concurrency.py --base-url http://localhost:5656 \
        --user zz_smoke --password ... --audio ~/narration_99.wav \
        --markdown ~/long.md --raters 8 --rounds 10 --out /tmp/verify.json

Needs ``requests`` and ``measure_sync_blocking.py`` next to it (login,
probes, the audio vector are shared).
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_sync_blocking as m  # noqa: E402


def fresh(session):
    return m.clone_session(session)


# --- the three loads ----------------------------------------------------------

def run_pdf(base, session, token, md_path, label, results):
    t0 = time.monotonic()
    entry = {'label': label, 'started_at': time.time()}
    try:
        with open(md_path, 'rb') as f:
            resp = fresh(session).post(
                f'{base}/convert-markdown',
                files={'markdown_file': ('long.md', f, 'text/markdown')},
                data={'output_filename': f'sync-freeze-{label}',
                      'orientation': 'portrait', 'style_theme': 'default'},
                headers={'X-CSRFToken': token}, timeout=600, allow_redirects=False)
        entry['status'] = resp.status_code
        entry['content_type'] = resp.headers.get('Content-Type')
        entry['bytes'] = len(resp.content)
        entry['is_pdf'] = resp.content[:5] == b'%PDF-'
        if resp.status_code != 200:
            # the route flashes + redirects on failure
            entry['location'] = resp.headers.get('Location')
    except Exception as e:
        entry['status'] = f'ERR {type(e).__name__}'
    entry['duration_s'] = round(time.monotonic() - t0, 1)
    results.append(entry)


def rater(base, session, token, card_ids, rounds, stats, lock):
    for _ in range(rounds):
        for cid in card_ids:
            t0 = time.monotonic()
            detail = ''
            try:
                resp = fresh(session).post(
                    f'{base}/api/cards/{cid}/review', json={'rating': 'good'},
                    headers={'X-CSRFToken': token}, timeout=120)
                status = resp.status_code
                if status != 200:
                    detail = resp.text[:160]
            except Exception as e:
                status = f'ERR {type(e).__name__}'
            latency = time.monotonic() - t0
            with lock:
                stats['latencies'].append(latency)
                stats['statuses'][str(status)] = stats['statuses'].get(str(status), 0) + 1
                if status == 200:
                    stats['ok_per_card'][cid] = stats['ok_per_card'].get(cid, 0) + 1
                elif detail and len(stats['errors']) < 10:
                    stats['errors'].append(f'{status}: {detail}')


# --- card state ---------------------------------------------------------------

def card_state(base, session, cid):
    card = fresh(session).get(f'{base}/api/cards/{cid}', timeout=60).json()
    review = card.get('review') or {}
    history = review.get('rating_history')
    if isinstance(history, str):
        try:
            history = json.loads(history)
        except ValueError:
            history = []
    return {'history_len': len(history or []), 'reps': review.get('reps') or 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--base-url', default='http://localhost:5656')
    ap.add_argument('--user', required=True)
    ap.add_argument('--password', required=True)
    ap.add_argument('--audio', help='long transcription running alongside (optional)')
    ap.add_argument('--language', default='de')
    ap.add_argument('--markdown', help='markdown file for the two concurrent PDF renders')
    ap.add_argument('--pdfs', type=int, default=2)
    ap.add_argument('--raters', type=int, default=8)
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--cards', type=int, default=40, help='max cards to rate')
    ap.add_argument('--disjoint', action='store_true',
                    help='give every rater its own cards (no two writers on one '
                         'card) instead of all raters over all cards')
    ap.add_argument('--probe-interval', type=float, default=1.0)
    ap.add_argument('--out')
    args = ap.parse_args()

    session = m.login(args.base_url, args.user, args.password)
    token = m.csrf_token(fresh(session), args.base_url)  # token from another connection

    cards = fresh(session).get(f'{args.base_url}/api/cards?limit=200', timeout=60).json()
    card_ids = [c['id'] for c in cards][:args.cards]
    if args.raters and not card_ids:
        sys.exit('the user has no cards to rate')
    before = {cid: card_state(args.base_url, session, cid) for cid in card_ids}
    writes = (args.rounds * len(card_ids) if args.disjoint
              else args.raters * args.rounds * len(card_ids))
    print(f'logged in as {args.user}; {len(card_ids)} cards; '
          f'{args.raters} raters × {args.rounds} rounds '
          f'({"disjoint cards" if args.disjoint else "all raters on all cards"}) = '
          f'{writes} rating writes')

    sched = m.ProbeScheduler(args.base_url, session, args.probe_interval)
    sched.start()
    time.sleep(args.probe_interval * 2)

    threads, audio_results, pdf_results = [], [], []
    stats = {'latencies': [], 'statuses': {}, 'ok_per_card': {}, 'errors': []}
    lock = threading.Lock()
    t_launch = time.time()
    if args.audio:
        threads.append(threading.Thread(target=m.run_conversion, args=(
            args.base_url, fresh(session), token, args.audio, 'audio', audio_results,
            'audio', args.language)))
    if args.markdown:
        for i in range(args.pdfs):
            threads.append(threading.Thread(target=run_pdf, args=(
                args.base_url, session, token, args.markdown, f'pdf{i + 1}', pdf_results)))
    for i in range(args.raters):
        if args.disjoint:
            ids = card_ids[i::args.raters]  # every card has exactly ONE writer
        else:
            # shifted start so the raters collide on different cards at any moment
            shift = (i * len(card_ids)) // max(args.raters, 1)
            ids = card_ids[shift:] + card_ids[:shift]
        threads.append(threading.Thread(target=rater, args=(
            args.base_url, session, token, ids, args.rounds, stats, lock)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t_launch
    time.sleep(args.probe_interval * 2)
    samples = sched.stop()
    probe_stats = m.summarize(samples, t_ref=t_launch)

    # --- results ----------------------------------------------------------------
    print(f'\nall loads done after {wall:.1f} s')
    for r in audio_results:
        print(f'  audio: {m.describe(r)}')
    for r in pdf_results:
        print(f'  {r["label"]}: status={r["status"]} pdf={r.get("is_pdf")} '
              f'bytes={r.get("bytes")} duration={r["duration_s"]} s'
              + (f' redirect={r.get("location")}' if r.get('location') else ''))
    lat = sorted(stats['latencies'])
    if lat:
        print(f'  ratings: n={len(lat)} statuses={stats["statuses"]} '
              f'p50={statistics.median(lat) * 1000:.0f} ms '
              f'p95={lat[int(len(lat) * 0.95) - 1] * 1000:.0f} ms max={lat[-1] * 1000:.0f} ms')
        for err in stats['errors']:
            print(f'    error sample: {err}')

    after = {cid: card_state(args.base_url, session, cid) for cid in card_ids}
    mismatches = []
    for cid in card_ids:
        expected = stats['ok_per_card'].get(cid, 0)
        grew = after[cid]['history_len'] - before[cid]['history_len']
        reps = after[cid]['reps'] - before[cid]['reps']
        if grew != expected or reps != expected:
            mismatches.append({'card': cid, 'ok_responses': expected,
                               'history_grew_by': grew, 'reps_grew_by': reps})
    print(f'  consistency: {len(card_ids)} cards re-read — '
          + ('every rating_history grew by exactly its 200s' if not mismatches
             else f'{len(mismatches)} MISMATCHES: {mismatches[:5]}'))

    m.print_table('  probes during the loads', probe_stats)
    parked = [s for s in samples if 0 <= s.get('offset_s', -1) <= wall and s['latency_s'] > 1.0]
    during = [s for s in samples if 0 <= s.get('offset_s', -1) <= wall]
    print(f'  probes slower than 1 s while the loads ran: {len(parked)} of {len(during)}')

    ok = (all(r.get('status') == 200 for r in audio_results)
          and all(r.get('is_pdf') for r in pdf_results)
          and set(stats['statuses']) <= {'200'}
          and not mismatches)
    print('\nRESULT:', 'clean' if ok else 'NOT CLEAN')

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'wall_s': wall, 'audio': audio_results, 'pdf': pdf_results,
                       'ratings': {k: v for k, v in stats.items() if k != 'latencies'},
                       'rating_latencies': lat, 'mismatches': mismatches,
                       'probes': {'stats': probe_stats, 'samples': samples}}, f, indent=1)
        print(f'written to {args.out}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
