#!/usr/bin/env python3
"""Live probe: a multi-chunk narration, end to end, with proof that the WAV
concatenation ran — and which branch of it (ARCH-NARR5).

The renderer sends at most ``MAX_TRANSCRIPT_BYTES`` (utf-8) per Cloud-TTS
call; a longer narration is rendered chunk by chunk and joined by
``services/wav_concat.py`` with one second of silence in between. The pytest
suite drives that with a mocked client. This is the check against the REAL
stack (web, Redis, RQ worker, Cloud TTS), built for the day the helpers moved
out of the retired ``services/gemini`` package:

1. build a two-turn text that crosses the chunk boundary — checked with the
   deployed ``chunk_turns`` before anything is submitted (one chunk = the
   concat is NOT probed, exit 2);
2. ``POST /api/narrations`` with ``NARRATION_TOKEN`` from the container env;
3. poll until ``ready``;
4. read the artifact ``narration_<id>.wav``: the stored duration matches it,
   and it carries ONE interior run of exact-zero samples with audio on both
   sides — the silence the concat inserts. A single chunk has no such run;
5. name the branch from the LENGTH of that run — **when the run allows it;
   this is information, not an exit criterion.** The two helpers insert a
   different number of zero samples (measured per run, on synthetic audio,
   with the deployed functions themselves — on the pin: pydub 23 998, because
   it resamples its 11 025 Hz silence to 24 kHz, wave exactly 24 000). A run
   shorter than the larger count can only be the smaller one's branch. But
   Cloud TTS sometimes ends or starts a chunk with exact-zero samples, and
   then the run is longer than both counts and proves neither (measured
   2026-10-02, three runs of the same text: 23 998 — pydub, proven — then
   24 010 and 25 691 — not provable). The branch the image takes is printed
   either way (``is_pydub_available()`` here; the worker runs the same image);
6. optionally ``GET /api/narrations/<id>/audio`` (argument ``audio``);
7. ``DELETE`` the probe row strictly by its id and check that row, artifact
   and job render are gone.

⚠️ **Why the branch is read off the artifact and not off the worker log:** the
worker process has no logging configuration (root logger without handler,
level WARNING — measured 2026-10-02), so every ``logger.info`` from ``tasks``
and ``services.*`` is dropped there, ``Concatenating … with PyDub`` included.

⚠️ **This probe creates a credential on the target account, and here is why.**
``NARRATION_TOKEN`` is a write-only, identity-less token: it can create the
narration and nothing else. Status, audio and delete are ``@login_required``
and owner-scoped, and the row lands on the account behind the token
(``INGEST_USER`` / the first user — in production Oli's), so a throwaway user
would get a 404. The script therefore issues an ``ApiToken`` for the ROW'S
OWNER through the app's own ``issue_token``, sets it to expire after 30
minutes, and deletes that row by its id in ``finally``. The plaintext never
leaves the process and is never printed. If the process is killed between
issue and ``finally``, the token dies on its own after 30 minutes; the dead
row carries the label ``probe-narration-concat`` (remove it by that label).
A probe narration left behind the same way is titled ``… - wird entfernt``.

It runs INSIDE the web container (it needs the DB for the token, the shared
volume for the artifact, and the env for ``NARRATION_TOKEN``), never with
``-u 0``. From the Mac, streamed over stdin — no file on the Mintbox:

    ssh mintbox 'docker exec -i markdown-converter-web python3 - long' \\
        < scripts/probe_narration_concat.py

or, in an image that carries it:

    docker exec markdown-converter-web python scripts/probe_narration_concat.py long

Arguments: ``long`` (default — two real Cloud-TTS calls, ~3.5 kB of text,
~80 s, a few cents) or ``short`` (one sentence, one call, no concat: proves
submit → render → adoption → serve only); ``audio`` adds the audio GET.
Exit 0 = probed and held, 1 = a check failed, 2 = not probed. No token, key
or cookie value is ever printed.

Env: BASE_URL (default http://127.0.0.1:5000 — the container's own port),
PROBE_VOLUME (/app/output_podcasts), PROBE_TIMEOUT_SECONDS (900).
"""
import array
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from datetime import datetime, timedelta, timezone

# ``python scripts/…`` puts scripts/ first on sys.path; over stdin it is the
# working directory (/app in the container). Either way the app root leads.
_ROOT = (os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
         if '__file__' in globals() else os.getcwd())
sys.path.insert(0, _ROOT)

BASE = os.environ.get('BASE_URL', 'http://127.0.0.1:5000')
VOLUME = os.environ.get('PROBE_VOLUME', '/app/output_podcasts')
TIMEOUT = int(os.environ.get('PROBE_TIMEOUT_SECONDS', '900'))
JOBS_DIR = os.path.join(VOLUME, 'narration_jobs')
TOKEN_LABEL = 'probe-narration-concat'
TOKEN_MINUTES = 30
RATE = 24000  # the Gemini-TTS output contract: LINEAR16, 24 kHz, mono

ARGS = sys.argv[1:]
KIND = 'short' if 'short' in ARGS else 'long'
WANT_AUDIO = 'audio' in ARGS

# Two turns, each under the cap, together over it → exactly two chunks.
TURN_A = (
    "Dies ist eine technische Probe für den Sprint ARCH-NARR5. Der Text ist absichtlich so lang, "
    "dass der Renderer ihn nicht in einem einzigen Aufruf an den Sprachdienst schicken kann. "
    "Die Grenze liegt bei dreitausendfünfhundert Bytes je Abschnitt, gemessen in UTF-8, nicht in Zeichen. "
    "Umlaute und das scharfe S zählen deshalb doppelt, und genau das macht deutsche Texte etwas teurer. "
    "Der Renderer teilt niemals mitten in einem Sprecherbeitrag, sondern sammelt ganze Beiträge, "
    "bis das Budget überschritten wäre. Erst dann beginnt er einen neuen Abschnitt. "
    "Jeder Abschnitt wird einzeln vertont und als eigene Datei im temporären Verzeichnis abgelegt. "
    "Am Ende stehen also mehrere kleine Dateien nebeneinander, die noch niemand hören soll. "
    "Sie müssen zu einer einzigen Aufnahme verbunden werden, mit einer Sekunde Stille dazwischen. "
    "Diese Verbindung erledigen zwei kleine Funktionen, die bis heute in einem Paket lagen, "
    "das eigentlich für etwas ganz anderes gedacht war. Das Paket hieß Gemini und brachte beim Import "
    "eine große Bibliothek mit, die der Renderer nie benutzt hat. "
    "Seit diesem Sprint wohnen die beiden Funktionen in einem eigenen Modul ohne fremde Abhängigkeiten. "
    "Das klingt nach einer Kleinigkeit, und für den Hörer ist es auch eine: Die Aufnahme klingt wie vorher. "
    "Der Unterschied liegt darin, was ein Fehler in der großen Bibliothek künftig noch mitreißen kann. "
    "Bisher hätte ein kaputter Import jede Vertonung verhindert, obwohl die Vertonung ihn gar nicht braucht. "
    "Jetzt hängt der Renderer nur noch an dem, was er wirklich aufruft, und das ist erfreulich wenig. "
    "Der erste Beitrag endet hier, und er ist für sich allein schon mehr als die Hälfte des Budgets. "
    "Damit ist sicher, dass der zweite Beitrag nicht mehr in denselben Abschnitt passt."
)
TURN_B = (
    "Der zweite Beitrag beginnt nach der Pause, und wer ihn hört, hört bereits das Ergebnis der Verbindung. "
    "Im Protokoll des Hintergrunddienstes sollte an dieser Stelle eine Zeile stehen, "
    "die sagt, dass zwei Dateien zusammengefügt wurden. "
    "Der Hintergrunddienst kennt die Datenbank nicht. Er rendert nur, legt die fertige Aufnahme "
    "unter dem Namen seines Auftrags auf das gemeinsame Laufwerk und meldet sich ab. "
    "Die Webanwendung findet die Datei beim nächsten Nachfragen, übernimmt sie unter dem endgültigen Namen "
    "und setzt den Eintrag auf fertig. Erst dann erscheint die Dauer in der Bibliothek. "
    "Für diese Probe ist die Dauer selbst ein Beleg: Sie muss ungefähr der Summe beider Beiträge entsprechen, "
    "zuzüglich der einen Sekunde Stille. Wäre nur ein Abschnitt angekommen, wäre die Aufnahme halb so lang. "
    "Der Schlüssel für den Sprachdienst liegt künftig nur noch beim Hintergrunddienst. "
    "Die Webanwendung, die aus dem Internet erreichbar ist, braucht ihn nicht mehr, "
    "weil sie selbst niemals vertont hat. Sie hat den Dienst nur beim Start aufgebaut und dann nie gefragt. "
    "Ein Geheimnis sollte dort liegen, wo sein letzter Leser sitzt, und nirgends sonst. "
    "Deshalb gibt es zwei Auslieferungen in fester Reihenfolge. Zuerst geht der neue Programmcode hinaus, "
    "der den Schlüssel nicht mehr anfasst. Erst wenn er läuft, wird die Datei aus dem Container genommen. "
    "Andersherum wäre die Webanwendung beim Start gestorben, weil der alte Code die Datei noch verlangt. "
    "Wer später zurück will, braucht aus demselben Grund beides zusammen: das alte Abbild und die alte Beschreibung. "
    "Nach dem Ende dieser Probe wird der Eintrag wieder entfernt, samt der Aufnahme, "
    "damit in der Bibliothek nichts zurückbleibt, was niemand bestellt hat. Ende der Probe."
)
TURN_SHORT = "Dies ist ein kurzer Probesatz für die Vertonung."

failures = []


def check(ok, what):
    print(('  PASS ' if ok else '  FAIL ') + what, flush=True)
    if not ok:
        failures.append(what)


def call(method, path, token, body=None, raw=False):
    """One request → ``(status, payload, headers)``. Never raises on 4xx/5xx."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    req.add_header('Authorization', f'Bearer {token}')
    if data is not None:
        req.add_header('Content-Type', 'application/json')
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = resp.read()
            return resp.status, (payload if raw else json.loads(payload or b'null')), resp.headers
    except urllib.error.HTTPError as e:
        payload = e.read()
        try:
            return e.code, json.loads(payload), e.headers
        except ValueError:
            return e.code, None, e.headers


def longest_zero_run(samples):
    """``(length, start)`` of the longest run of exact-zero samples."""
    best_len, best_start, run_start = 0, 0, None
    for i, sample in enumerate(samples):
        if sample == 0:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            if i - run_start > best_len:
                best_len, best_start = i - run_start, run_start
            run_start = None
    if run_start is not None and len(samples) - run_start > best_len:
        best_len, best_start = len(samples) - run_start, run_start
    return best_len, best_start


def read_samples(path):
    with wave.open(path, 'rb') as w:
        return w.getframerate(), array.array('h', w.readframes(w.getnframes()))


def inserted_zero_samples(concat):
    """How many zero samples ``concat`` puts between two chunks — measured
    with the deployed helper itself on two synthetic, zero-free WAVs."""
    paths = []
    for _ in range(2):
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.close()
        with wave.open(tmp.name, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(b'\x01\x00' * RATE)
        paths.append(tmp.name)
    out = concat(paths)  # unlinks its inputs
    try:
        return longest_zero_run(read_samples(out)[1])[0]
    finally:
        os.unlink(out)


def proven_branch(run, counts):
    """The branch a silence of ``run`` zero samples proves, or ``None``.

    ``counts`` maps branch → zero samples it inserts. Zero-valued chunk edges
    can only LENGTHEN the run, so a run below the larger count cannot come
    from the larger insert; a run at or above it could be either. With one
    branch available there is nothing to tell apart.
    """
    ordered = sorted(counts.items(), key=lambda kv: kv[1])
    low, high = ordered[0], ordered[-1]
    if len(counts) == 1:
        return low[0] if run >= low[1] else None
    if low[1] <= run < high[1]:
        return low[0]
    return None


def job_files(job):
    return sorted(name for name in os.listdir(JOBS_DIR) if job in name)


def main():
    from services.narration_render import MAX_TRANSCRIPT_BYTES, chunk_turns
    from services.wav_concat import (concatenate_with_pydub,
                                     concatenate_with_wave, is_pydub_available)

    if KIND == 'long':
        turns = [{'speaker': 'Erzähler', 'text': TURN_A},
                 {'speaker': 'Erzähler', 'text': TURN_B}]
    else:
        turns = [{'speaker': 'Erzähler', 'text': TURN_SHORT}]
    sizes = [len(t['text'].encode('utf-8')) for t in turns]
    chunks = len(chunk_turns(turns))
    print(f'[text] {KIND}: turn bytes {sizes}, sum {sum(sizes)}; '
          f'MAX_TRANSCRIPT_BYTES={MAX_TRANSCRIPT_BYTES} → {chunks} chunk(s)', flush=True)
    if KIND == 'long' and chunks < 2:
        print('\nNOT PROBED: the text does not cross the chunk boundary.', flush=True)
        return 2

    expected = None
    if KIND == 'long':
        counts = {'wave': inserted_zero_samples(concatenate_with_wave)}
        if is_pydub_available():
            counts['pydub'] = inserted_zero_samples(concatenate_with_pydub)
        expected = 'pydub' if 'pydub' in counts else 'wave'
        print(f'[concat] zero samples inserted per branch (measured here): {counts}; '
              f'this image takes the {expected} branch', flush=True)

    narration_token = os.environ.get('NARRATION_TOKEN')
    if not narration_token:
        print('\nNOT PROBED: NARRATION_TOKEN is not set in this container.', flush=True)
        return 2

    status, body, _ = call('POST', '/api/narrations', narration_token, {
        'title': f'Probe Vertonung ({KIND}) - wird entfernt',
        'mode': 'single_speaker', 'voices': {'Erzähler': 'Kore'},
        'turns': turns, 'language': 'de-DE',
    })
    check(status == 202, f'POST /api/narrations → {status}')
    if status != 202:
        print(f'  body: {body}', flush=True)
        return 1
    nid, job = body['narration_id'], body['job_id']
    print(f'[row] narration_id={nid} job mark={job}', flush=True)
    started = time.time()
    artifact = os.path.join(VOLUME, f'narration_{nid}.wav')

    from app import app
    from app_pkg.mobile_auth import _hash_token, issue_token
    from models import ApiToken, Conversion, User, db

    token_row_id = None
    try:
        with app.app_context():
            owner = db.session.get(User, db.session.get(Conversion, nid).user_id)
            bearer = issue_token(owner, label=TOKEN_LABEL)
            row = ApiToken.query.filter_by(token_hash=_hash_token(bearer)).one()
            row.expires_at = (datetime.now(timezone.utc).replace(tzinfo=None)
                              + timedelta(minutes=TOKEN_MINUTES))
            db.session.commit()
            token_row_id = row.id
            print(f'[token] temporary ApiToken row id={token_row_id} for the row owner '
                  f'(user_id={owner.id}), expires in {TOKEN_MINUTES} min', flush=True)
            db.session.remove()

        meta, last = {}, None
        while time.time() - started < TIMEOUT:
            status, body, _ = call('GET', f'/api/narrations/{nid}', bearer)
            meta = (body or {}).get('metadata') or {}
            state = meta.get('narration_status')
            if state != last:
                print(f'  t+{time.time() - started:5.1f}s HTTP {status} narration_status={state}', flush=True)
                last = state
            if state in ('ready', 'failed'):
                break
            time.sleep(3)
        check(meta.get('narration_status') == 'ready',
              f"narration is ready (status={meta.get('narration_status')}, "
              f"error tail={(meta.get('error') or '')[-300:]!r})")

        if os.path.exists(artifact):
            rate, samples = read_samples(artifact)
            seconds = len(samples) / rate
            print(f'[artifact] {os.path.basename(artifact)}: {os.path.getsize(artifact)} bytes, '
                  f'{rate} Hz, {seconds:.2f} s, {sum(len(t["text"]) for t in turns) / seconds:.1f} chars/s',
                  flush=True)
            check(abs((meta.get('duration_seconds') or 0) - seconds) <= 1,
                  f"stored duration ({meta.get('duration_seconds')} s) matches the artifact")
            if KIND == 'long':
                run, start = longest_zero_run(samples)
                before, after = start / rate, (len(samples) - start - run) / rate
                print(f'[silence] longest exact-zero run: {run} samples ({run / rate:.3f} s) '
                      f'at t={before:.2f} s — audio before {before:.2f} s, after {after:.2f} s', flush=True)
                check(run >= min(counts.values()) and before > 5 and after > 5,
                      'one interior silence with audio on both sides — the concatenation ran')
                proven = proven_branch(run, counts)
                print(f'[branch] this image takes {expected}; the run length '
                      + (f'proves {proven}' if proven else
                         'proves neither branch (chunk edges carry zero samples)'), flush=True)
                if proven and proven != expected:
                    check(False, f'the run proves the {proven} branch, the image should take {expected}')
        else:
            check(False, f'artifact {artifact} exists')

        check(job_files(job) == [], 'the job render was adopted (nothing of this mark in narration_jobs/)')

        if WANT_AUDIO:
            status, payload, headers = call('GET', f'/api/narrations/{nid}/audio', bearer, raw=True)
            size = len(payload) if isinstance(payload, bytes) else None
            check(status == 200 and isinstance(payload, bytes) and payload[:4] == b'RIFF'
                  and os.path.exists(artifact) and size == os.path.getsize(artifact),
                  f"GET …/audio → {status}, {headers.get('Content-Type')}, {size} bytes, the artifact's bytes")

        status, _, _ = call('DELETE', f'/api/conversions/{nid}', bearer)
        check(status == 200, f'DELETE /api/conversions/{nid} → {status}')
        status, _, _ = call('GET', f'/api/narrations/{nid}', bearer)
        check(status == 404 and not os.path.exists(artifact) and job_files(job) == [],
              f'row, artifact and job render are gone (GET → {status})')
    finally:
        if token_row_id is not None:
            with app.app_context():
                deleted = ApiToken.query.filter_by(id=token_row_id, label=TOKEN_LABEL).delete()
                db.session.commit()
                left = ApiToken.query.filter_by(label=TOKEN_LABEL).count()
            check(deleted == 1 and left == 0,
                  f'temporary ApiToken row id={token_row_id} deleted (probe tokens left: {left})')

    print(f'\n{len(failures)} failure(s)' if failures
          else f'\nprobed: {KIND} narration rendered, adopted and removed'
               + (' — the concatenation ran' if KIND == 'long' else ''), flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
