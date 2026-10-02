"""Notion integration: suggestions cache + send-to-Notion endpoint."""
import logging
import os
import re
import time as _time
from datetime import datetime

import requests as http_requests
from flask import jsonify, request
from flask_login import login_required

from app_pkg.config import LOCAL_TZ
from app_pkg.library import get_owned_conversion

logger = logging.getLogger(__name__)

NOTION_MCP_URL = os.environ.get('NOTION_MCP_URL', 'http://localhost:3333')
MCP_AUTH_TOKEN = os.environ.get('MCP_AUTH_TOKEN', '')
NOTION_TOKEN = os.environ.get('NOTION_TOKEN', '')

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

    @app.route('/api/conversions/<int:conversion_id>/send-to-notion', methods=['POST'])
    @login_required
    def api_send_to_notion(conversion_id):
        get_owned_conversion(conversion_id)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400
        target = data.get('target')
        if target not in ('meetings', 'notes', 'inbox'):
            return jsonify({'error': 'Invalid target'}), 400

        payload = {k: v for k, v in data.get('fields', {}).items() if v}
        if 'datum' in payload:
            payload['datum'] = normalize_notion_datum(payload['datum'])
        try:
            resp = http_requests.post(
                f'{NOTION_MCP_URL}/api/{target}',
                json=payload,
                headers={'Authorization': f'Bearer {MCP_AUTH_TOKEN}',
                         'Content-Type': 'application/json'},
                timeout=30
            )
            result = resp.json()
            if resp.status_code >= 400:
                return jsonify({'error': result.get('error', result.get('detail', 'Notion API error'))}), resp.status_code
            return jsonify(result), resp.status_code
        except http_requests.RequestException as e:
            app.logger.error(f"Failed to reach Notion server: {e}", exc_info=True)
            return jsonify({'error': 'Failed to reach Notion server.'}), 502
