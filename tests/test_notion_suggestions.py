"""NOTION-MEETING-LINK P0 — the suggestions cache holds answers, not failures.

``/api/notion/suggestions`` feeds the datalists of the "An Notion senden"
panel from the Notion API. Until this sprint a non-200 from Notion became an
empty list WITHOUT a log line and was cached like an answer (300 s for the
suggestions, 3600 s for the database ids) — one 429 emptied the datalists for
up to an hour and left no trace. Now: a non-200 is logged (status, never the
token), the client still gets the empty fallback, and nothing incomplete is
cached — the next request asks Notion again.

``http_requests`` is replaced at the module boundary; no network.
"""
import logging

import pytest
import requests

from app_pkg.integrations import notion

TOKEN = 'test-notion-token-do-not-log'
EMPTY = {'people': [], 'projects': [], 'meeting_types': [], 'note_types': []}
FULL = {'people': ['Anna', 'Bert'], 'projects': ['Projekt A'],
        'meeting_types': ['Jour fixe', 'Workshop'], 'note_types': ['Idee']}


class _Resp:
    def __init__(self, status, body=None):
        self.status_code = status
        self._body = body or {}

    def json(self):
        return self._body


def _titles(*names):
    return {'results': [{'properties': {'Name': {'type': 'title', 'title': [{'plain_text': n}]}}}
                        for n in names]}


def _select(*options):
    return {'properties': {'Type': {'type': 'select', 'select': {'options': [{'name': o} for o in options]}}}}


class FakeNotion:
    """Stands in for the ``requests`` module inside ``notion``. ``status``
    maps a path to the status it answers with (default 200); ``raises`` makes
    every call raise a network error."""

    RequestException = requests.RequestException

    def __init__(self):
        self.status = {}
        self.raises = False
        self.calls = []

    def _answer(self, method, url, headers):
        path = url.replace('https://api.notion.com/v1', '')
        self.calls.append((method, path))
        assert headers['Authorization'] == f'Bearer {TOKEN}'
        if self.raises:
            raise requests.ConnectionError('boom')
        status = self.status.get(path, 200)
        if status != 200:
            return _Resp(status, {'message': 'rate limited'})
        if path == '/search':
            return _Resp(200, {'results': [
                {'id': f'db-{name.lower()}', 'title': [{'plain_text': name}]}
                for name in ('People', 'Project', 'Meetings', 'Notes')]})
        return _Resp(200, {
            '/databases/db-people/query': _titles('Bert', 'Anna'),
            '/databases/db-project/query': _titles('Projekt A'),
            '/databases/db-meetings': _select('Jour fixe', 'Workshop'),
            '/databases/db-notes': _select('Idee'),
        }[path])

    def get(self, url, headers=None, timeout=None):
        return self._answer('GET', url, headers)

    def post(self, url, json=None, headers=None, timeout=None):
        return self._answer('POST', url, headers)


@pytest.fixture
def fake_notion(monkeypatch):
    fake = FakeNotion()
    monkeypatch.setattr(notion, 'http_requests', fake)
    monkeypatch.setattr(notion, 'NOTION_TOKEN', TOKEN)
    notion._notion_cache.clear()
    yield fake
    notion._notion_cache.clear()


def _get(client):
    resp = client.get('/api/notion/suggestions')
    assert resp.status_code == 200
    return resp.get_json()


def test_answers_are_cached(authenticated_client, fake_notion):
    assert _get(authenticated_client) == FULL
    first = len(fake_notion.calls)
    assert first == 5                      # search + 2 queries + 2 schema reads
    assert _get(authenticated_client) == FULL
    assert len(fake_notion.calls) == first  # second request: no call to Notion


def test_failed_search_is_logged_and_not_cached(authenticated_client, fake_notion, caplog):
    fake_notion.status['/search'] = 502
    with caplog.at_level(logging.WARNING):
        assert _get(authenticated_client) == EMPTY          # the client's fallback stays
    assert any('502' in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert TOKEN not in caplog.text

    fake_notion.status.clear()                               # Notion is back
    assert _get(authenticated_client) == FULL                # … and is asked again


def test_one_failed_database_does_not_poison_the_cache(authenticated_client, fake_notion, caplog):
    fake_notion.status['/databases/db-people/query'] = 429
    with caplog.at_level(logging.WARNING):
        partial = _get(authenticated_client)
    assert partial == {**FULL, 'people': []}                 # the rest still arrives
    assert any('429' in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert TOKEN not in caplog.text

    fake_notion.status.clear()
    assert _get(authenticated_client) == FULL                # not held for 300 s


def test_failed_schema_read_is_logged_and_not_cached(authenticated_client, fake_notion, caplog):
    fake_notion.status['/databases/db-meetings'] = 500
    with caplog.at_level(logging.WARNING):
        partial = _get(authenticated_client)
    assert partial == {**FULL, 'meeting_types': []}
    assert any('500' in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)

    fake_notion.status.clear()
    assert _get(authenticated_client) == FULL


def test_database_ids_survive_a_later_failure(authenticated_client, fake_notion):
    """The id lookup answered 200 — it stays cached even though the
    suggestions built from it were incomplete."""
    fake_notion.status['/databases/db-people/query'] = 429
    _get(authenticated_client)
    fake_notion.status.clear()
    fake_notion.calls.clear()
    assert _get(authenticated_client) == FULL
    assert ('POST', '/search') not in fake_notion.calls


def test_network_error_falls_back_and_is_not_cached(authenticated_client, fake_notion, caplog):
    fake_notion.raises = True
    with caplog.at_level(logging.WARNING):
        assert _get(authenticated_client) == EMPTY
    assert TOKEN not in caplog.text
    fake_notion.raises = False
    assert _get(authenticated_client) == FULL


def test_without_a_token_nothing_is_asked(authenticated_client, fake_notion, monkeypatch):
    monkeypatch.setattr(notion, 'NOTION_TOKEN', '')
    assert _get(authenticated_client) == EMPTY
    assert fake_notion.calls == []


def test_suggestions_need_a_login(client, fake_notion):
    resp = client.get('/api/notion/suggestions')
    assert resp.status_code == 302
    assert fake_notion.calls == []
