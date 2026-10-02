"""Notion integration: suggestions cache, meeting candidates, send-to-Notion.

Three routes: ``/api/notion/suggestions`` (datalists of the panel),
``/api/conversions/<id>/notion-meetings`` (the candidate list for "Bestehendes
Meeting" — NOTION-MEETING-LINK) and ``/api/conversions/<id>/send-to-notion``
with its two ways: the form ("Neues Meeting anlegen", notes, inbox — the
client's fields are relayed) and ``page_id`` (a transcript onto an existing
meeting — the payload is built HERE and is exactly four fields).
"""
import json
import logging
import os
import re
import time as _time
from datetime import datetime, timedelta

import requests as http_requests
from flask import jsonify, request
from flask_login import login_required

from app_pkg.config import LOCAL_TZ
from app_pkg.integrations import notion_meetings as meetings_logic
from app_pkg.library import get_owned_conversion, write_metadata_keys
from models import db

logger = logging.getLogger(__name__)

NOTION_MCP_URL = os.environ.get('NOTION_MCP_URL', 'http://localhost:3333')
MCP_AUTH_TOKEN = os.environ.get('MCP_AUTH_TOKEN', '')
NOTION_TOKEN = os.environ.get('NOTION_TOKEN', '')
# NOTION-MEETING-LINK: the public origin for the back link written into the
# Notion field ``CONVERTER``. Compose defaults it to the one deployed origin;
# without the variable the request's own root is used (dev, tests).
PUBLIC_BASE_URL = os.environ.get('PUBLIC_BASE_URL', '').strip().rstrip('/')

# Per-call deadlines for the notion-mcp-server (sync web thread). The write
# of a long transcript is chunked on the other side — it gets more room.
MCP_QUERY_TIMEOUT_SECONDS = 20
MCP_WRITE_TIMEOUT_SECONDS = 60

MSG_INVALID_TARGET = 'Ungültiges Ziel.'
MSG_UNREACHABLE = 'Notion-Server nicht erreichbar. Später erneut versuchen.'
MSG_UPSTREAM_ERROR = 'Notion hat einen Fehler gemeldet.'
MSG_QUERY_FAILED = 'Meetings konnten nicht aus Notion geladen werden. Später erneut versuchen.'
MSG_INVALID_DAY = 'Ungültiger Tag. Erwartet wird JJJJ-MM-TT.'
MSG_INVALID_PAGE = 'Ungültige Meeting-Auswahl.'
MSG_INVALID_REPLACE = 'Ungültige Angabe zum Überschreiben.'
MSG_EMPTY_DOCUMENT = 'Dieses Dokument hat keinen Inhalt. Es wurde nichts gesendet.'
MSG_MEETING_NOT_LISTED = 'Meeting nicht gefunden. Liste neu laden.'
MSG_TOO_LONG = 'Das Transkript ist zu lang für Notion.'
# The other side's errors on the page_id way are TRANSLATED, never relayed
# (its texts name internals such as the registry refresh command). Statuses
# not listed here become a 502 with MSG_SEND_FAILED.
MSG_SEND_FAILED = 'Senden an Notion fehlgeschlagen. Später erneut versuchen.'
_TRANSLATED_UPSTREAM = {
    400: 'Notion hat die Anfrage abgelehnt. Es wurde nichts gesendet.',
    404: 'Das Meeting gibt es in Notion nicht mehr. Liste neu laden.',
    503: 'Notion konnte das Meeting nicht prüfen. Später erneut versuchen.',
}

# NOTION-TZ: the meeting form's ``datetime-local`` yields wall-clock time
# WITHOUT a zone, and Notion reads an offset-free datetime as UTC — 14:30 in
# the form became 16:30 (CEST) in Notion. The notion-mcp-server accepts
# ``{start, time_zone}`` (IANA name, only on an offset-free start), so the
# backend attaches the one server-fixed zone here.
_NAIVE_DATETIME_RE = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?', re.ASCII)


def normalize_notion_datum(value):
    """Attach ``LOCAL_TZ`` to a zone-less meeting datetime; never raises.

    ``YYYY-MM-DDTHH:MM[:SS]`` (a real calendar value) →
    ``{"start": "YYYY-MM-DDTHH:MM:SS", "time_zone": "Europe/Berlin"}``.
    Everything else — a date (all-day), a string with ``±HH:MM``/``Z`` offset,
    an object, blank or unrecognisable input — passes through unchanged; the
    server validates it and its 400 text is relayed by the route.
    """
    if not isinstance(value, str) or not _NAIVE_DATETIME_RE.fullmatch(value):
        return value
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    return {'start': parsed.strftime('%Y-%m-%dT%H:%M:%S'), 'time_zone': LOCAL_TZ.key}


# --- Notion suggestions cache ---
_notion_cache = {}
_SUGGESTION_KEYS = ('people', 'projects', 'meeting_types', 'note_types')


def _notion_api(method, path, body=None):
    headers = {'Authorization': f'Bearer {NOTION_TOKEN}', 'Notion-Version': '2022-06-28'}
    url = f'https://api.notion.com/v1{path}'
    if method == 'GET':
        return http_requests.get(url, headers=headers, timeout=15)
    return http_requests.post(url, json=body or {}, headers=headers, timeout=15)


def _notion_ok(resp, what):
    """True for a 200. Anything else is logged — status and what was asked,
    never the token (it lives in the request headers only) — and makes the
    caller's result incomplete, i.e. not cacheable."""
    if resp.status_code == 200:
        return True
    logger.warning('Notion suggestions: %s answered %s', what, resp.status_code)
    return False


def _cached(key, ttl, fetcher):
    """``fetcher`` returns ``(data, complete)``. Only a complete result is
    held for ``ttl`` seconds: a non-200 from Notion used to be cached like an
    answer and emptied the datalists for up to an hour. Returns the same pair."""
    entry = _notion_cache.get(key)
    if entry and _time.time() < entry['exp']:
        return entry['data'], True
    data, complete = fetcher()
    if complete:
        _notion_cache[key] = {'data': data, 'exp': _time.time() + ttl}
    return data, complete


def _get_notion_db_ids():
    def fetch():
        resp = _notion_api('POST', '/search', {
            'filter': {'value': 'database', 'property': 'object'}, 'page_size': 100
        })
        if not _notion_ok(resp, 'database search'):
            return {}, False
        ids = {}
        for db in resp.json().get('results', []):
            name = ''.join(t.get('plain_text', '') for t in db.get('title', [])).strip().upper()
            if name:
                ids[name] = db['id']
        return ids, True
    return _cached('db_ids', 3600, fetch)


def _query_db_titles(db_id, db_name):
    resp = _notion_api('POST', f'/databases/{db_id}/query', {'page_size': 100})
    if not _notion_ok(resp, f'query of {db_name}'):
        return [], False
    titles = []
    for page in resp.json().get('results', []):
        for prop in page.get('properties', {}).values():
            if prop.get('type') == 'title':
                t = ''.join(p.get('plain_text', '') for p in prop.get('title', []))
                if t:
                    titles.append(t)
                break
    return sorted(set(titles)), True


def _get_select_options(db_id, db_name, prop_name='Type'):
    resp = _notion_api('GET', f'/databases/{db_id}')
    if not _notion_ok(resp, f'schema of {db_name}'):
        return [], False
    for name, prop in resp.json().get('properties', {}).items():
        if name.lower() == prop_name.lower() and prop.get('type') == 'select':
            return [o['name'] for o in prop.get('select', {}).get('options', [])], True
    return [], True


# --- notion-mcp-server (meetings) ---

class NotionUnavailable(Exception):
    """The notion-mcp-server could not be asked (network, non-200, no JSON)."""


def _mcp_post(path, payload, timeout):
    return http_requests.post(
        f'{NOTION_MCP_URL}{path}', json=payload,
        headers={'Authorization': f'Bearer {MCP_AUTH_TOKEN}',
                 'Content-Type': 'application/json'},
        timeout=timeout)


def _json_or_empty(resp):
    try:
        body = resp.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _query_meetings(day):
    """The meetings whose start lies on ``day`` ± 1 → ``(rows, truncated)``.

    ± 1 because the other side filters by the calendar day of the start AS
    STORED: a meeting saved in UTC sits under the previous date for the first
    two hours of a Berlin day. Raises ``NotionUnavailable`` (logged with the
    status, never the token) instead of answering with an empty list — "no
    meetings" and "could not ask" must not look the same.
    """
    window = {'date_from': (day - timedelta(days=1)).isoformat(),
              'date_to': (day + timedelta(days=1)).isoformat()}
    try:
        resp = _mcp_post('/api/meetings/query', window, MCP_QUERY_TIMEOUT_SECONDS)
    except http_requests.RequestException as e:
        logger.warning('Notion meetings query unreachable: %s', type(e).__name__)
        raise NotionUnavailable() from e
    if resp.status_code != 200:
        logger.warning('Notion meetings query answered %s', resp.status_code)
        raise NotionUnavailable()
    body = _json_or_empty(resp)
    rows = body.get('meetings')
    if not isinstance(rows, list):
        logger.warning('Notion meetings query answered without a meetings list')
        raise NotionUnavailable()
    return rows, body.get('truncated') is True


def converter_link_for(conversion_id):
    """Absolute URL of the library page — the back link in Notion.

    ⚠️ ``Conversion.id`` is reused after the highest row is deleted
    (JOB-ID-REUSE): a link lying in Notion can then point at a different
    document. Named, not solved here (BACKLOG CONV-ID-NO-REUSE).
    """
    base = PUBLIC_BASE_URL or request.url_root.rstrip('/')
    return f'{base}/library/{conversion_id}'


def _conversion_metadata(conversion):
    """The metadata bag as a dict — lenient, it is client-writable input."""
    try:
        metadata = json.loads(conversion.metadata_json) if conversion.metadata_json else {}
    except ValueError:
        return {}
    return metadata if isinstance(metadata, dict) else {}


def _thousands(number):
    return f'{number:,}'.replace(',', '.')


def _remember_link(conversion, record):
    """Merge the ``notion_link`` namespace into the row (one json_patch
    UPDATE). The Notion write has already happened — a failure here must not
    turn the answer into an error; it is logged and reported as ``None``."""
    try:
        write_metadata_keys(conversion, {'notion_link': record})
        db.session.commit()
    except Exception:
        db.session.rollback()
        logger.error('notion_link of conversion %s could not be stored', conversion.id, exc_info=True)
        return None
    return meetings_logic.clean_notion_link(record)


def _send_to_existing_meeting(conversion, data):
    """Transcript → the field ``Transcript`` of an existing MEETINGS page.

    The payload is built here and is EXACTLY ``page_id``, ``transcript``,
    ``converter_link``, ``replace_transcript`` — whatever else the request
    carries (title, date, type, people, project, summary) would overwrite a
    page that belongs to the calendar. The transcript is the row's content,
    never the request's.
    """
    page_id = data.get('page_id')
    if data.get('target') != 'meetings' or not meetings_logic.is_page_id(page_id):
        return jsonify({'error': MSG_INVALID_PAGE}), 400
    day = meetings_logic.parse_day(data.get('day'))
    if day is None:
        return jsonify({'error': MSG_INVALID_DAY}), 400
    # An overwrite flag is read by identity, never by truthiness.
    replace = data.get('replace_transcript', False)
    if replace is not True and replace is not False:
        return jsonify({'error': MSG_INVALID_REPLACE}), 400
    transcript = conversion.content or ''
    if not transcript.strip():
        # The other side reads transcript "" as "do not send" — it would
        # write only the link and answer 200.
        return jsonify({'error': MSG_EMPTY_DOCUMENT}), 400

    # The meeting is looked up on the other side, not taken from the request:
    # title, start and calendar id for the question, the answer and the
    # remembered link come from Notion.
    try:
        rows, _truncated = _query_meetings(day)
    except NotionUnavailable:
        return jsonify({'error': MSG_QUERY_FAILED}), 502
    meeting = meetings_logic.find_meeting(rows, page_id)
    if meeting is None:
        return jsonify({'error': MSG_MEETING_NOT_LISTED}), 404

    payload = {
        'page_id': meeting['page_id'],
        'transcript': transcript,
        'converter_link': converter_link_for(conversion.id),
        'replace_transcript': replace,
    }
    try:
        resp = _mcp_post('/api/meetings', payload, MCP_WRITE_TIMEOUT_SECONDS)
    except http_requests.RequestException as e:
        logger.error('Notion server unreachable (meeting write): %s', type(e).__name__)
        return jsonify({'error': MSG_UNREACHABLE}), 502
    result = _json_or_empty(resp)

    if resp.status_code == 409 and result.get('code') == 'transcript_exists':
        title, date_text = meeting['title'], meetings_logic.meeting_date_text(meeting)
        sentence = f'‚{title}‘ ({date_text}) hat schon ein Transkript.'
        return jsonify({'error': sentence, 'code': 'transcript_exists',
                        'confirm': f'{sentence} Überschreiben?',
                        'meeting_title': title, 'meeting_date_text': date_text}), 409
    if resp.status_code == 413:
        length, limit = result.get('length'), result.get('max')
        numbers_ok = all(isinstance(n, int) and not isinstance(n, bool) for n in (length, limit))
        body = {'error': MSG_TOO_LONG, 'code': 'transcript_too_long'}
        if numbers_ok:
            body.update(length=length, max=limit,
                        error=f'Das Transkript ist zu lang für Notion: {_thousands(length)} Zeichen, '
                              f'erlaubt sind {_thousands(limit)}.')
        return jsonify(body), 413
    if resp.status_code >= 400:
        # Logged with the other side's text (it names what to fix); the
        # client gets our sentence.
        logger.warning('Notion meeting write answered %s: %s', resp.status_code,
                       str(result.get('error', ''))[:300])
        if resp.status_code in _TRANSLATED_UPSTREAM:
            return jsonify({'error': _TRANSLATED_UPSTREAM[resp.status_code]}), resp.status_code
        return jsonify({'error': MSG_SEND_FAILED}), 502

    record = meetings_logic.link_record(
        meeting['page_id'], result.get('url') or meeting['url'],
        meeting['calendar_event_id'], meeting['title'], meeting['start_raw'])
    link = _remember_link(conversion, record)
    return jsonify({'success': True, 'created': False, 'url': record['url'], 'link': link}), 200


def _new_meeting_start(datum):
    """The start of a freshly created meeting as an ISO string for the
    remembered link: the normalised form field (``{start, time_zone}``) gets
    its offset, everything else is kept as sent."""
    if isinstance(datum, dict):
        start, zone = datum.get('start'), datum.get('time_zone')
        if isinstance(start, str) and zone == LOCAL_TZ.key:
            try:
                return datetime.fromisoformat(start).replace(tzinfo=LOCAL_TZ).isoformat()
            except ValueError:
                return None
        return start if isinstance(start, str) else None
    return datum if isinstance(datum, str) else None


def register(app):
    @app.route('/api/notion/suggestions')
    @login_required
    def api_notion_suggestions():
        def fetch():
            result = {key: [] for key in _SUGGESTION_KEYS}
            if not NOTION_TOKEN:
                return result, True
            db_ids, complete = _get_notion_db_ids()
            for key, db_name, loader in (('people', 'PEOPLE', _query_db_titles),
                                         ('projects', 'PROJECT', _query_db_titles),
                                         ('meeting_types', 'MEETINGS', _get_select_options),
                                         ('note_types', 'NOTES', _get_select_options)):
                if db_name in db_ids:
                    result[key], ok = loader(db_ids[db_name], db_name)
                    complete = complete and ok
            return result, complete
        try:
            data, _complete = _cached('suggestions', 300, fetch)
            return jsonify(data)
        except Exception as e:
            logger.warning(f'Notion suggestions failed: {e}')
            return jsonify({key: [] for key in _SUGGESTION_KEYS})

    @app.route('/api/conversions/<int:conversion_id>/notion-meetings')
    @login_required
    def api_notion_meetings(conversion_id):
        """Candidate meetings for "Bestehendes Meeting" (NOTION-MEETING-LINK).

        ``?day=YYYY-MM-DD`` (strict, else 400) picks the shown day; without it
        the day of the remembered link, else the reference day (recording,
        else upload). Order, preselection and all display texts are computed
        here — see ``notion_meetings.build_candidates``.
        """
        conversion = get_owned_conversion(conversion_id)
        metadata = _conversion_metadata(conversion)
        reference = meetings_logic.reference_for(metadata, conversion.created_at)
        link = meetings_logic.clean_notion_link(metadata.get('notion_link'))

        if 'day' in request.args:
            shown_day = meetings_logic.parse_day(request.args.get('day'))
            if shown_day is None:
                return jsonify({'error': MSG_INVALID_DAY}), 400
        else:
            shown_day = (meetings_logic.parse_day(link['day']) if link and link['day'] else None) \
                or reference['day']

        try:
            rows, truncated = _query_meetings(shown_day)
        except NotionUnavailable:
            return jsonify({'error': MSG_QUERY_FAILED}), 502

        order, candidates, preselected, reason = meetings_logic.build_candidates(
            rows, reference, shown_day, converter_link_for(conversion.id), link)
        return jsonify({
            'reference': meetings_logic.reference_payload(reference),
            'day': shown_day.isoformat(),
            'day_text': meetings_logic.day_text(shown_day),
            'prev_day': (shown_day - timedelta(days=1)).isoformat(),
            'next_day': (shown_day + timedelta(days=1)).isoformat(),
            'order': order,
            'meetings': candidates,
            'preselected_page_id': preselected,
            'preselect_reason': reason,
            'truncated': truncated,
            'link': link,
        })

    @app.route('/api/conversions/<int:conversion_id>/send-to-notion', methods=['POST'])
    @login_required
    def api_send_to_notion(conversion_id):
        conversion = get_owned_conversion(conversion_id)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400
        if 'page_id' in data:
            return _send_to_existing_meeting(conversion, data)
        target = data.get('target')
        if target not in ('meetings', 'notes', 'inbox'):
            return jsonify({'error': MSG_INVALID_TARGET}), 400

        payload = {k: v for k, v in data.get('fields', {}).items() if v}
        if 'datum' in payload:
            payload['datum'] = normalize_notion_datum(payload['datum'])
        if target == 'meetings':
            # The back link travels on both meeting ways; ours, never the client's.
            payload['converter_link'] = converter_link_for(conversion.id)
        try:
            resp = _mcp_post(f'/api/{target}', payload, 30)
            result = resp.json()
            if resp.status_code >= 400:
                return jsonify({'error': result.get('error', result.get('detail', MSG_UPSTREAM_ERROR))}), resp.status_code
            if target == 'meetings' and meetings_logic.is_page_id(result.get('page_id')):
                # A freshly created meeting is remembered like a chosen one.
                record = meetings_logic.link_record(
                    result['page_id'], result.get('url'), None,
                    payload.get('title') if isinstance(payload.get('title'), str) else '',
                    _new_meeting_start(payload.get('datum')))
                result = dict(result, link=_remember_link(conversion, record))
            return jsonify(result), resp.status_code
        except http_requests.RequestException as e:
            app.logger.error(f"Failed to reach Notion server: {e}", exc_info=True)
            return jsonify({'error': MSG_UNREACHABLE}), 502
