"""Notion integration: suggestions cache + send-to-Notion endpoint."""
import logging
import os
import time as _time

import requests as http_requests
from flask import jsonify, request
from flask_login import current_user, login_required

from models import Conversion


NOTION_MCP_URL = os.environ.get('NOTION_MCP_URL', 'http://localhost:3333')
MCP_AUTH_TOKEN = os.environ.get('MCP_AUTH_TOKEN', '')
NOTION_TOKEN = os.environ.get('NOTION_TOKEN', '')

# --- Notion suggestions cache ---
_notion_cache = {}


def _notion_api(method, path, body=None):
    headers = {'Authorization': f'Bearer {NOTION_TOKEN}', 'Notion-Version': '2022-06-28'}
    url = f'https://api.notion.com/v1{path}'
    if method == 'GET':
        return http_requests.get(url, headers=headers, timeout=15)
    return http_requests.post(url, json=body or {}, headers=headers, timeout=15)


def _cached(key, ttl, fetcher):
    entry = _notion_cache.get(key)
    if entry and _time.time() < entry['exp']:
        return entry['data']
    data = fetcher()
    _notion_cache[key] = {'data': data, 'exp': _time.time() + ttl}
    return data


def _get_notion_db_ids():
    def fetch():
        resp = _notion_api('POST', '/search', {
            'filter': {'value': 'database', 'property': 'object'}, 'page_size': 100
        })
        if resp.status_code != 200:
            return {}
        ids = {}
        for db in resp.json().get('results', []):
            name = ''.join(t.get('plain_text', '') for t in db.get('title', [])).strip().upper()
            if name:
                ids[name] = db['id']
        return ids
    return _cached('db_ids', 3600, fetch)


def _query_db_titles(db_id):
    resp = _notion_api('POST', f'/databases/{db_id}/query', {'page_size': 100})
    if resp.status_code != 200:
        return []
    titles = []
    for page in resp.json().get('results', []):
        for prop in page.get('properties', {}).values():
            if prop.get('type') == 'title':
                t = ''.join(p.get('plain_text', '') for p in prop.get('title', []))
                if t:
                    titles.append(t)
                break
    return sorted(set(titles))


def _get_select_options(db_id, prop_name):
    resp = _notion_api('GET', f'/databases/{db_id}')
    if resp.status_code != 200:
        return []
    for name, prop in resp.json().get('properties', {}).items():
        if name.lower() == prop_name.lower() and prop.get('type') == 'select':
            return [o['name'] for o in prop.get('select', {}).get('options', [])]
    return []


def register(app):
    @app.route('/api/notion/suggestions')
    @login_required
    def api_notion_suggestions():
        def fetch():
            if not NOTION_TOKEN:
                return {'people': [], 'projects': [], 'meeting_types': [], 'note_types': []}
            db_ids = _get_notion_db_ids()
            people = _query_db_titles(db_ids['PEOPLE']) if 'PEOPLE' in db_ids else []
            projects = _query_db_titles(db_ids['PROJECT']) if 'PROJECT' in db_ids else []
            meeting_types = _get_select_options(db_ids['MEETINGS'], 'Type') if 'MEETINGS' in db_ids else []
            note_types = _get_select_options(db_ids['NOTES'], 'Type') if 'NOTES' in db_ids else []
            return {'people': people, 'projects': projects, 'meeting_types': meeting_types, 'note_types': note_types}
        try:
            return jsonify(_cached('suggestions', 300, fetch))
        except Exception as e:
            logging.getLogger(__name__).warning(f'Notion suggestions failed: {e}')
            return jsonify({'people': [], 'projects': [], 'meeting_types': [], 'note_types': []})

    @app.route('/api/conversions/<int:conversion_id>/send-to-notion', methods=['POST'])
    @login_required
    def api_send_to_notion(conversion_id):
        Conversion.query.filter_by(id=conversion_id, user_id=current_user.id).first_or_404()
        data = request.get_json()
        target = data.get('target')
        if target not in ('meetings', 'notes', 'inbox'):
            return jsonify({'error': 'Invalid target'}), 400

        payload = {k: v for k, v in data.get('fields', {}).items() if v}
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
            app.logger.error(f"Failed to reach Notion server: {e}")
            return jsonify({'error': 'Failed to reach Notion server.'}), 502
