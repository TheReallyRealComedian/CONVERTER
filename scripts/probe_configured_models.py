#!/usr/bin/env python3
"""Spricht die KONFIGURIERTEN Modellnamen einmal real an und wird laut, wenn
einer nicht mehr antwortet (OKTOBER P3, 2026-08-22).

Warum: ``gemini-2.0-flash`` wurde am 01.06.2026 abgeschaltet, und CONVERTERs
PDF-Pfad war zwei Monate still tot — der 404 wurde gefangen, der Pfad
degradierte auf eine leere Textebene, die einzige Spur war eine WARNING-Zeile
(DOC-FIX). Eine Deprecations-Seite zu lesen beantwortet die Frage nicht: die
Daten gelten je Fläche (Gemini API ≠ Vertex AI ≠ Cloud TTS) und wandern
(OKTOBER P1: „16.10." war auf der einen Fläche gar kein Datum, auf der anderen
inzwischen der 20.10.). Ein echter Aufruf beantwortet sie in Sekunden.

Was: je Modellfläche EIN minimaler echter Aufruf über denselben Request-Pfad
wie die Produktion —

* **Narration / Cloud Text-to-Speech**: ``services.narration_render.render_turns``
  (baut ``VoiceSelectionParams(model_name=…)`` genau wie der Worker), ein Wort,
  ``single_speaker``; die Temp-WAV wird sofort wieder gelöscht.
* **PDF-Vision / Gemini API**: ``genai.Client(...).models.generate_content`` mit
  dem Modell aus ``services.pdf_cloud``, ein Satz statt einer PDF-Seite
  (``max_output_tokens=256`` — Thinking-Modelle verbrauchen einen Teil davon fürs Denken) — ein totes Modell ist ein 404 im Aufruf, dafür
  braucht es keine Seite.

Die Namen kommen aus der **Konfiguration**, nie aus einer Literalliste: aus
den Modul-Konstanten ``DEFAULT_NARRATION_MODEL`` / ``DEFAULT_CLOUD_PDF_MODEL``,
die ihrerseits Env (``NARRATION_TTS_MODEL`` / ``PDF_VISION_MODEL``) oder
Code-Default sind. Nach einem Modellwechsel prüft das Skript also automatisch
das Richtige; die Ausgabe nennt die Quelle des Namens mit.

Laufen lassen (Mintbox, im **Worker** — er hat beide Zugangsdaten; Kosten: ein
~1-s-TTS-Synth + ein winziger Gemini-Call, Bruchteile eines Cents; kein
DB-Zugriff, nichts bleibt liegen)::

    docker cp scripts/probe_configured_models.py markdown-converter-worker:/tmp/probe_models.py
    docker exec markdown-converter-worker python /tmp/probe_models.py
    docker exec markdown-converter-worker rm /tmp/probe_models.py

Oder ohne Kopie: ``docker exec -i markdown-converter-worker python - < scripts/probe_configured_models.py``
(``-i`` ist Pflicht — ohne reicht docker kein stdin durch und python tut still nichts).

Ausgabe: eine Zeile je Modell — Status, Fläche, Name, Quelle des Namens, Latenz,
Beleg bzw. Fehlertext. Exit **0** = alle antworten · **1** = mindestens ein
Modell antwortet nicht (``FAIL``; der Fehlertext steht in der Zeile — ein 404
ist das Abschalt-Signal) · **2** = eine Fläche konnte gar nicht angesprochen
werden (``SETUP``: Zugangsdaten/Import fehlen). Auch 2 ist absichtlich laut:
genau so sähe ein stiller Ausfall aus.
"""
import argparse
import os
import sys
import time

DEFAULT_TIMEOUT_SECONDS = 60.0


class ProbeSetupError(RuntimeError):
    """The surface could not be addressed at all (credentials / import missing).
    Not a verdict about the model — reported as ``SETUP``, exit 2."""


def _source(env_name):
    """Where the configured name comes from — env override or code default."""
    return f'env {env_name}' if os.environ.get(env_name) else 'Code-Default'


def configured_models():
    """The models the app is configured to call, read from the modules that
    call them (import at call time, so env/monkeypatching is honoured)."""
    from services.narration_render import DEFAULT_NARRATION_MODEL
    from services.pdf_cloud import DEFAULT_CLOUD_PDF_MODEL
    return [
        ('Narration / Cloud TTS', DEFAULT_NARRATION_MODEL, _source('NARRATION_TTS_MODEL'), probe_cloud_tts),
        ('PDF-Vision / Gemini API', DEFAULT_CLOUD_PDF_MODEL, _source('PDF_VISION_MODEL'), probe_gemini),
    ]


def probe_cloud_tts(model_name, timeout):
    """One single-speaker synth through the production request builder."""
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        raise ProbeSetupError('GOOGLE_APPLICATION_CREDENTIALS nicht gesetzt')
    try:
        from google.cloud import texttospeech
        from services.narration_render import render_turns
    except ImportError as exc:
        raise ProbeSetupError(f'Import: {exc}') from exc
    client = texttospeech.TextToSpeechClient()
    path = render_turns(
        client, [{'speaker': 'Probe', 'text': 'Probe.'}], {'Probe': 'Kore'},
        mode='single_speaker', language_code='de-DE', model_name=model_name,
        synth_timeout=timeout,
    )
    try:
        size = os.path.getsize(path)
    finally:
        os.unlink(path)
    return f'WAV {size} B'


def probe_gemini(model_name, timeout):
    """One tiny text call against the configured vision model (404 = dead)."""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ProbeSetupError('GEMINI_API_KEY nicht gesetzt')
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ProbeSetupError(f'Import: {exc}') from exc
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents='Antworte nur mit: OK',
        config=types.GenerateContentConfig(
            max_output_tokens=256, temperature=0,
            # no tools here — disabling AFC also silences the SDK's warning line
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            # per-call deadline in ms (DOC-FIX doctrine), caps the response
            http_options=types.HttpOptions(timeout=int(timeout * 1000)),
        ),
    )
    text = (getattr(response, 'text', None) or '').strip()
    usage = getattr(response, 'usage_metadata', None)
    total = getattr(usage, 'total_token_count', None)
    return f'Antwort {text!r}, {total} Tokens'


def run(probes, timeout=DEFAULT_TIMEOUT_SECONDS, out=sys.stdout):
    """Probe every (label, model_name, source, fn); print one line each.
    Returns the exit code: 0 all answer · 1 any FAIL · 2 only SETUP problems."""
    fails = setups = 0
    for label, model_name, source, fn in probes:
        t0 = time.monotonic()
        try:
            detail = fn(model_name, timeout)
            status = 'OK   '
        except ProbeSetupError as exc:
            status, detail = 'SETUP', f'Fläche nicht ansprechbar: {exc}'
            setups += 1
        except Exception as exc:  # noqa: BLE001 — every failure must be reported, none swallowed
            status, detail = 'FAIL ', f'{type(exc).__name__}: {str(exc)[:400]}'
            fails += 1
        elapsed = time.monotonic() - t0
        print(f'{status} {label:<24} {model_name:<34} [{source}] {elapsed:6.1f} s  {detail}',
              file=out, flush=True)
    total = len(probes)
    if fails:
        print(f'\n{fails} von {total} konfigurierten Modellen antwortet NICHT.', file=out, flush=True)
        return 1
    if setups:
        print(f'\n{setups} von {total} Flächen nicht ansprechbar (Zugangsdaten/Import).', file=out, flush=True)
        return 2
    print(f'\nAlle {total} konfigurierten Modelle antworten.', file=out, flush=True)
    return 0


def main(argv=None, probes=None, out=sys.stdout):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT_SECONDS,
                        help='per-call deadline in seconds (default %(default)s)')
    args = parser.parse_args(argv)
    # The app package is addressed from the repo / container root (/app), not
    # from wherever this file was copied to.
    for root in (os.getcwd(), '/app'):
        if os.path.isdir(os.path.join(root, 'services')) and root not in sys.path:
            sys.path.insert(0, root)
    return run(probes if probes is not None else configured_models(), timeout=args.timeout, out=out)


if __name__ == '__main__':
    sys.exit(main())
