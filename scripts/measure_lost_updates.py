#!/usr/bin/env python3
"""Do the suspected read-modify-write paths lose writes too? (LOST-UPDATE P1)

SYNC-FREEZE P2 proved lost updates on ``api_review_card`` by counting
``rating_history`` afterwards (``scripts/verify_concurrency.py``). Three more
paths have the same shape — load a row, compute, write the whole row back —
and were SUSPECTED, not measured. This script measures each one through the
real HTTP surface, every request on a FRESH connection (so the writers land
on several gunicorn processes AND several pool threads), and judges the
RESULT, never the status code:

* ``--settings``  the SHARED ``User.settings_json`` blob. Per round ONE
                  ``PUT /api/learn/settings`` (flat keys) and ONE
                  ``PUT /api/document-conversions/settings`` (``document_api``
                  namespace) fire at the same instant with round-specific
                  values; both GETs must show them afterwards. A lost write is
                  the OTHER feature's namespace reverting to its previous
                  value — vanishing entirely when it never existed before (the
                  first-write case on a fresh user).
* ``--progress``  ``PATCH /api/conversions/<id>/progress`` (furthest-read,
                  forward-clamped). Per round W writers with distinct percents
                  fire at once; the stored mark must be the maximum sent. A
                  lost write leaves the mark short of the furthest position.
                  The probe document is created on the session user and
                  deleted at the end (``--keep`` to keep it).
* ``--section``   ``PATCH /api/conversions/<id>/section`` (MCP-DOCWRITE,
                  ``CARD_TOKEN`` read from the environment, never printed).
                  Per round S writers each replace THEIR OWN section of one
                  document at once; afterwards every section must carry its
                  writer's marker. The token surface has no read endpoint, so
                  the final state is read back through one more SEQUENTIAL
                  replace of a sentinel section — the PATCH response carries
                  the full content. The token targets ``INGEST_USER`` / the
                  first user, i.e. the REAL account: pass the id of a
                  throwaway document and delete it afterwards.

Every round resets the state sequentially, releases the writers on a
barrier, then reads back. A round's verdict is unambiguous: exactly one
writer per value, so any value that is not the one sent is a lost write.

    python3 scripts/measure_lost_updates.py --base-url http://localhost:5656 \
        --user zz_lostupdate --password ... --settings --progress \
        --rounds 200 --writers 8 --out /tmp/lost.json
    CARD_TOKEN=... python3 scripts/measure_lost_updates.py \
        --base-url http://localhost:5656 --section --conversion-id 170 \
        --rounds 100 --writers 8

Needs ``requests`` and ``measure_sync_blocking.py`` next to it (login, fresh
sessions, CSRF token).
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_sync_blocking as m  # noqa: E402

TIMEOUT = 60


# --- plumbing -----------------------------------------------------------------

def fire_together(fns):
    """Run every callable in its own thread, all released by ONE barrier, and
    return their results in order (an exception becomes ``ERR <type>``)."""
    barrier = threading.Barrier(len(fns))
    results = [None] * len(fns)

    def run(i, fn):
        barrier.wait()
        try:
            results[i] = fn()
        except Exception as e:  # a failed request is a data point, not a crash
            results[i] = f'ERR {type(e).__name__}'

    threads = [threading.Thread(target=run, args=(i, fn)) for i, fn in enumerate(fns)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def tally(statuses, status):
    statuses[str(status)] = statuses.get(str(status), 0) + 1


def timed(fn):
    """``fn()`` → ``(status, seconds)``."""
    t0 = time.monotonic()
    status = fn()
    return status, time.monotonic() - t0


def latency_line(latencies):
    if not latencies:
        return 'n=0'
    lat = sorted(latencies)
    return (f'n={len(lat)} p50={statistics.median(lat) * 1000:.0f} ms '
            f'p95={lat[int(len(lat) * 0.95) - 1] * 1000:.0f} ms max={lat[-1] * 1000:.0f} ms')


# --- 1. the shared settings blob ---------------------------------------------

def measure_settings(base, session, token, rounds):
    learn_url = f'{base}/api/learn/settings'
    doc_url = f'{base}/api/document-conversions/settings'
    headers = {'X-CSRFToken': token}
    statuses, latencies, lost = {}, [], []
    t_start = time.monotonic()
    for r in range(rounds):
        learn_val = 11 + (r % 40)                     # never the default (10), never last round's
        doc_mode = 'cloud' if r % 2 == 0 else 'lokal'  # alternates; 'lokal' is ALSO the default

        def put_learn():
            return m.clone_session(session).put(
                learn_url, json={'daily_new_limit': learn_val}, headers=headers,
                timeout=TIMEOUT).status_code

        def put_doc():
            return m.clone_session(session).put(
                doc_url, json={'default_mode': doc_mode}, headers=headers,
                timeout=TIMEOUT).status_code

        for status, secs in fire_together([lambda: timed(put_learn), lambda: timed(put_doc)]):
            tally(statuses, status)
            latencies.append(secs)
        got_learn = m.clone_session(session).get(learn_url, timeout=TIMEOUT).json()
        got_doc = m.clone_session(session).get(doc_url, timeout=TIMEOUT).json()
        read = {'daily_new_limit': got_learn.get('daily_new_limit'),
                'default_mode': got_doc.get('default_mode')}
        sent = {'daily_new_limit': learn_val, 'default_mode': doc_mode}
        if read != sent:
            lost.append({'round': r, 'sent': sent, 'read': read,
                         'lost_namespace': [k for k in sent if read[k] != sent[k]]})
    wall = time.monotonic() - t_start

    lost_learn = sum('daily_new_limit' in e['lost_namespace'] for e in lost)
    lost_doc = sum('default_mode' in e['lost_namespace'] for e in lost)
    print(f'\nSETTINGS BLOB — {rounds} rounds × (1 learn PUT + 1 document PUT at once), '
          f'{wall:.1f} s')
    print(f'  statuses={statuses}  latency {latency_line(latencies)}')
    print(f'  rounds with a lost write: {len(lost)} of {rounds} — '
          f'learn keys lost {lost_learn}×, document_api namespace lost {lost_doc}×')
    for e in lost[:5]:
        print(f'    round {e["round"]}: sent {e["sent"]} → read {e["read"]}')
    if lost and lost[0]['round'] == 0 and 'default_mode' in lost[0]['lost_namespace']:
        print('  ⚠ round 0 ran on a never-written blob: the document_api namespace '
              'did not revert, it VANISHED (the GET shows the default).')
    return {'rounds': rounds, 'statuses': statuses, 'wall_s': round(wall, 1),
            'lost_rounds': len(lost), 'lost_learn': lost_learn, 'lost_doc': lost_doc,
            'lost': lost, 'latencies': latencies}


# --- 2. furthest-read progress -----------------------------------------------

def measure_progress(base, session, token, rounds, writers, keep):
    headers = {'X-CSRFToken': token}
    resp = m.clone_session(session).post(
        f'{base}/api/conversions',
        json={'conversion_type': 'markdown_input',
              'title': 'zz_lostupdate progress probe (Messdokument, wird geloescht)',
              'content': '# LOST-UPDATE progress probe\n\nMessdokument.\n'},
        headers=headers, timeout=TIMEOUT)
    resp.raise_for_status()
    cid = resp.json()['id']
    url = f'{base}/api/conversions/{cid}/progress'
    # distinct percents, maximum 100 — the forward-clamp must end on 100
    percents = [round(100.0 * (i + 1) / writers, 2) for i in range(writers)]
    expected = max(percents)
    statuses, latencies, lost = {}, [], []
    t_start = time.monotonic()
    try:
        for r in range(rounds):
            reset = m.clone_session(session).patch(
                url, json={'reset': True}, headers=headers, timeout=TIMEOUT)
            if reset.status_code != 200:
                sys.exit(f'progress reset failed: {reset.status_code} {reset.text[:200]}')

            def writer(p):
                def go():
                    return m.clone_session(session).patch(
                        url, json={'percent': p}, headers=headers,
                        timeout=TIMEOUT).status_code
                return lambda: timed(go)

            for status, secs in fire_together([writer(p) for p in percents]):
                tally(statuses, status)
                latencies.append(secs)
            stored = m.clone_session(session).get(
                f'{base}/api/conversions/{cid}', timeout=TIMEOUT).json().get('last_read_percent')
            if stored != expected:
                lost.append({'round': r, 'stored': stored, 'expected': expected})
    finally:
        if not keep:
            m.clone_session(session).delete(
                f'{base}/api/conversions/{cid}', headers=headers, timeout=TIMEOUT)
    wall = time.monotonic() - t_start

    print(f'\nFURTHEST-READ — {rounds} rounds × {writers} writers at once '
          f'(percents {percents[0]}…{expected}), {wall:.1f} s')
    print(f'  statuses={statuses}  latency {latency_line(latencies)}')
    print(f'  rounds where the mark stopped short of {expected}: {len(lost)} of {rounds}')
    if lost:
        stored = [e['stored'] or 0 for e in lost]
        print(f'    stored instead: min {min(stored)} · median {statistics.median(stored)} '
              f'· max {max(stored)}')
    return {'rounds': rounds, 'writers': writers, 'percents': percents,
            'statuses': statuses, 'wall_s': round(wall, 1), 'lost_rounds': len(lost),
            'lost': lost, 'latencies': latencies, 'conversion_id': cid, 'kept': keep}


# --- 3. docwrite section replace ---------------------------------------------

def measure_section(base, card_token, conversion_id, rounds, writers):
    headers = {'Authorization': f'Bearer {card_token}'}
    names = [f'S{i + 1}' for i in range(writers)]
    base_doc = ('# zz_lostupdate_probe\n\nMessdokument LOST-UPDATE (Docwrite-Rennen).\n\n'
                + ''.join(f'## {n}\nbase\n\n' for n in names) + '## Z\nsentinel\n')
    content_url = f'{base}/api/conversions/{conversion_id}/content'
    section_url = f'{base}/api/conversions/{conversion_id}/section'
    statuses, latencies, lost = {}, [], []
    refused_total = 0
    t_start = time.monotonic()
    for r in range(rounds):
        reset = requests.patch(content_url, json={'content': base_doc}, headers=headers,
                               timeout=TIMEOUT)
        if reset.status_code != 200:
            sys.exit(f'content reset failed: {reset.status_code} {reset.text[:200]} '
                     f'(CARD_TOKEN set? conversion {conversion_id} owned by the token user?)')

        def writer(n):
            marker = f'round {r} writer {n}'

            def go():
                return requests.patch(
                    section_url, json={'heading': n, 'content': f'## {n}\n{marker}'},
                    headers=headers, timeout=TIMEOUT).status_code
            return lambda: timed(go)

        outcomes = fire_together([writer(n) for n in names])
        for status, secs in outcomes:
            tally(statuses, status)
            latencies.append(secs)
        # sequential read-back through the sentinel section (token surface has no GET)
        read = requests.patch(section_url, json={'heading': 'Z', 'content': f'## Z\nread {r}'},
                              headers=headers, timeout=TIMEOUT)
        content = read.json().get('content', '') if read.status_code == 200 else ''
        # A write is LOST when the server said 200 and the marker is gone. A
        # 409 (LOST-UPDATE P3: the bounded retry gave up) is an honest refusal,
        # counted separately — the agent was told, nothing vanished.
        missing = [n for (status, _s), n in zip(outcomes, names)
                   if status == 200 and f'round {r} writer {n}' not in content]
        refused_total += sum(1 for status, _s in outcomes if status == 409)
        if missing:
            lost.append({'round': r, 'missing': missing})
    wall = time.monotonic() - t_start

    total_missing = sum(len(e['missing']) for e in lost)
    print(f'\nDOCWRITE SECTION — {rounds} rounds × {writers} writers on their own '
          f'sections at once, {wall:.1f} s')
    print(f'  statuses={statuses}  latency {latency_line(latencies)}')
    print(f'  rounds with a lost section: {len(lost)} of {rounds} — '
          f'{total_missing} of {rounds * writers} section writes lost (200 but gone), '
          f'{refused_total} refused honestly (409)')
    if lost:
        per_round = [len(e['missing']) for e in lost]
        print(f'    sections lost per affected round: min {min(per_round)} · '
              f'median {statistics.median(per_round)} · max {max(per_round)}')
    return {'rounds': rounds, 'writers': writers, 'statuses': statuses,
            'wall_s': round(wall, 1), 'lost_rounds': len(lost),
            'lost_writes': total_missing, 'refused': refused_total, 'lost': lost,
            'latencies': latencies, 'conversion_id': conversion_id}


# --- main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--base-url', default='http://localhost:5656')
    ap.add_argument('--user', help='session user for --settings / --progress')
    ap.add_argument('--password')
    ap.add_argument('--settings', action='store_true')
    ap.add_argument('--progress', action='store_true')
    ap.add_argument('--section', action='store_true')
    ap.add_argument('--conversion-id', type=int,
                    help='--section: a THROWAWAY document owned by the token user')
    ap.add_argument('--rounds', type=int, default=100)
    ap.add_argument('--writers', type=int, default=8,
                    help='simultaneous writers per round (--progress / --section)')
    ap.add_argument('--keep', action='store_true', help='keep the --progress probe document')
    ap.add_argument('--out')
    args = ap.parse_args()
    if not (args.settings or args.progress or args.section):
        ap.error('pick at least one of --settings / --progress / --section')
    if (args.settings or args.progress) and not (args.user and args.password):
        ap.error('--settings / --progress need --user and --password')
    if args.section and not args.conversion_id:
        ap.error('--section needs --conversion-id')
    card_token = os.environ.get('CARD_TOKEN')
    if args.section and not card_token:
        ap.error('--section needs CARD_TOKEN in the environment')

    report = {'base_url': args.base_url, 'rounds': args.rounds, 'writers': args.writers}
    if args.settings or args.progress:
        session = m.login(args.base_url, args.user, args.password)
        token = m.csrf_token(m.clone_session(session), args.base_url)
        print(f'logged in as {args.user}')
        if args.settings:
            report['settings'] = measure_settings(args.base_url, session, token, args.rounds)
        if args.progress:
            report['progress'] = measure_progress(args.base_url, session, token, args.rounds,
                                                  args.writers, args.keep)
    if args.section:
        report['section'] = measure_section(args.base_url, card_token, args.conversion_id,
                                            args.rounds, args.writers)

    losses = {k: v['lost_rounds'] for k, v in report.items()
              if isinstance(v, dict) and 'lost_rounds' in v}
    print('\nRESULT:', 'no lost writes' if not any(losses.values())
          else 'LOST WRITES ' + json.dumps(losses))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(report, f, indent=1)
        print(f'written to {args.out}')
    return 1 if any(losses.values()) else 0


if __name__ == '__main__':
    sys.exit(main())
