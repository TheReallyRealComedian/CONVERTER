"""NOTION-MEETING-LINK — a transcript goes to an EXISTING Notion meeting.

Two pieces behind the dialog "An Notion senden → Meeting → Bestehendes
Meeting":

* ``GET /api/conversions/<id>/notion-meetings[?day=]`` — the candidate list.
  The SERVER orders, preselects and writes the display texts (weekday, date,
  time in ``LOCAL_TZ``); the JS only draws.
* ``POST /api/conversions/<id>/send-to-notion`` with ``page_id`` — the send.
  Its payload is built here and is EXACTLY ``page_id``, ``transcript`` (from
  the row, never from the request), ``converter_link``, ``replace_transcript``
  — a title/date/type/people/summary in the payload would overwrite a page
  that belongs to the calendar.

Measured on Oli's data before the build: 53 of 54 recordings carry only a
DATE (the dictaphone filename dialect has no time, ``recorded_at`` is 00:00) —
so a preselection exists only for a real time of day; a date-only recording
gets the day's meetings in order and a visible hint.

The link is remembered as the ``notion_link`` namespace in ``metadata_json``
through ONE merge writer (``json_patch``); read back it is INPUT (the metadata
bag is client-writable): the URL only becomes a link if it is https on a
Notion host.

The notion-mcp-server is replaced at ``notion.http_requests``; no network.
"""
import json
from datetime import datetime

import pytest
import requests
from sqlalchemy import text

from app_pkg import library
from app_pkg.integrations import notion
from models import Conversion, User, db

MCP_URL = 'http://notion-mcp.test'
MCP_TOKEN = 'test-mcp-token'
BASE = 'https://converter.test'

P_A = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
P_B = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb'
P_C = 'cccccccc-cccc-4ccc-8ccc-cccccccccccc'
P_D = 'dddddddd-dddd-4ddd-8ddd-dddddddddddd'
P_E = 'eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee'
P_NEW = '99999999-9999-4999-8999-999999999999'

TRANSCRIPT = '**Sprecher 1:** Guten Tag.\n\n**Sprecher 2:** Hallo.'


class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        if self._body is None:
            raise ValueError('no json')
        return self._body


def meeting(page_id, title, start, end=None, **kw):
    return {
        'page_id': page_id, 'url': 'https://app.notion.com/p/' + page_id.replace('-', ''),
        'title': title, 'datum': {'start': start, 'end': end, 'time_zone': None},
        'calendar_event_id': kw.get('uid', ''), 'type': kw.get('type', 'Meeting'),
        'meeting_link': None, 'has_transcript': kw.get('has_transcript', False),
        'people_count': 2, 'people_count_capped': False,
        'notnotion': kw.get('notnotion', False), 'converter_link': kw.get('converter_link'),
    }


class FakeMcp:
    """The notion-mcp-server as the brief and its README describe it: query by
    the stored calendar day of the start; a write to a page that has a
    transcript is a 409 unless ``replace_transcript``; more than 200 000
    characters is a 413; a create answers 201 with the new page id."""

    RequestException = requests.RequestException

    def __init__(self):
        self.meetings = []
        self.calls = []
        self.query_status = 200
        self.forced_write = None        # (status, body) instead of the emulation
        self.network_down = set()       # paths that raise

    def post(self, url, json=None, headers=None, timeout=None):
        assert url.startswith(MCP_URL), url
        path = url[len(MCP_URL):]
        assert headers['Authorization'] == f'Bearer {MCP_TOKEN}'
        self.calls.append((path, json))
        if path in self.network_down:
            raise requests.ConnectionError('connection refused')
        if path == '/api/meetings/query':
            if self.query_status != 200:
                return _Resp(self.query_status, {'error': 'query failed upstream'})
            lo, hi = json['date_from'], json['date_to']
            hits = sorted((m for m in self.meetings if lo <= m['datum']['start'][:10] <= hi),
                          key=lambda m: m['datum']['start'])
            return _Resp(200, {'count': len(hits), 'date_from': lo, 'date_to': hi,
                               'meetings': hits, 'only_notnotion': False, 'truncated': False})
        if self.forced_write is not None:
            return _Resp(*self.forced_write)
        if path == '/api/meetings' and 'page_id' in json:
            page = next((m for m in self.meetings if m['page_id'] == json['page_id']), None)
            if page is None:
                return _Resp(404, {'error': 'page not found'})
            if len(json.get('transcript', '')) > 200000:
                return _Resp(413, {'error': 'Transcript too long', 'code': 'transcript_too_long',
                                   'length': len(json['transcript']), 'max': 200000})
            if page['has_transcript'] and json.get('replace_transcript') is not True:
                return _Resp(409, {'error': 'Page already has a transcript',
                                   'code': 'transcript_exists', 'page_id': page['page_id']})
            page['has_transcript'] = True
            page['converter_link'] = json.get('converter_link')
            return _Resp(200, {'success': True, 'page_id': page['page_id'], 'created': False,
                               'url': page['url'], 'message': 'Meeting updated', 'warnings': []})
        return _Resp(201, {'success': True, 'page_id': P_NEW, 'created': True,
                           'url': 'https://app.notion.com/p/' + P_NEW.replace('-', ''),
                           'message': 'Meeting created', 'warnings': []})

    def paths(self):
        return [path for path, _ in self.calls]

    def writes(self):
        return [body for path, body in self.calls if path != '/api/meetings/query']

    def queries(self):
        return [body for path, body in self.calls if path == '/api/meetings/query']


@pytest.fixture
def mcp(monkeypatch):
    fake = FakeMcp()
    monkeypatch.setattr(notion, 'http_requests', fake)
    monkeypatch.setattr(notion, 'NOTION_MCP_URL', MCP_URL)
    monkeypatch.setattr(notion, 'MCP_AUTH_TOKEN', MCP_TOKEN)
    monkeypatch.setattr(notion, 'PUBLIC_BASE_URL', BASE, raising=False)
    return fake


def make_conv(app, user_id, *, recorded_at=None, source='filename', content=TRANSCRIPT,
              conversion_type='audio_transcription', created_at=None, duration=1500, meta=None):
    metadata = {'transcription_status': 'ready', 'language': 'de', 'duration_seconds': duration}
    if recorded_at is not None:
        metadata.update(recorded_at=recorded_at, recorded_at_source=source)
    metadata.update(meta or {})
    with app.app_context():
        conv = Conversion(user_id=user_id, conversion_type=conversion_type, title='260521_0176',
                          content=content, metadata_json=json.dumps(metadata),
                          created_at=created_at or datetime(2026, 10, 2, 19, 0, 0))
        db.session.add(conv)
        db.session.commit()
        return conv.id


def stored_meta(app, conv_id):
    with app.app_context():
        return json.loads(db.session.get(Conversion, conv_id).metadata_json)


def candidates(client, conv_id, day=None):
    url = f'/api/conversions/{conv_id}/notion-meetings'
    if day is not None:
        url += f'?day={day}'
    return client.get(url)


def send(client, conv_id, **body):
    return client.post(f'/api/conversions/{conv_id}/send-to-notion', json=body)


def five_meetings():
    """Three on the reference day, one each on the neighbours."""
    return [
        meeting(P_A, 'Früh', '2026-10-02T09:00:00.000+02:00', '2026-10-02T10:00:00.000+02:00'),
        meeting(P_B, 'Mittag', '2026-10-02T14:30:00.000+02:00', '2026-10-02T15:00:00.000+02:00',
                uid='uid-b', type='Jour Fixe'),
        meeting(P_C, 'Spät', '2026-10-02T16:00:00.000+02:00', '2026-10-02T17:00:00.000+02:00'),
        meeting(P_D, 'Vortag', '2026-10-01T15:00:00.000+02:00', '2026-10-01T16:00:00.000+02:00'),
        meeting(P_E, 'Folgetag', '2026-10-03T08:00:00.000+02:00', '2026-10-03T09:00:00.000+02:00'),
    ]


# --- candidates: order and preselection ----------------------------------------

def test_time_known_orders_by_distance_and_preselects(app, authenticated_client, test_user, mcp):
    mcp.meetings = five_meetings()
    conv_id = make_conv(app, test_user['id'], recorded_at='2026-10-02T14:40:00+02:00')
    resp = candidates(authenticated_client, conv_id)
    assert resp.status_code == 200
    body = resp.get_json()

    assert mcp.queries() == [{'date_from': '2026-10-01', 'date_to': '2026-10-03'}]
    assert [m['page_id'] for m in body['meetings']] == [P_B, P_C, P_A, P_E, P_D]
    assert body['order'] == 'distance'
    assert body['preselected_page_id'] == P_B and body['preselect_reason'] == 'time'
    assert body['reference'] == {
        'source': 'recorded_at', 'time_known': True, 'day': '2026-10-02',
        'text': 'Fr, 02.10.2026, 14:40', 'hint': None,
        'duration_seconds': 1500, 'duration_text': '25 min',
    }
    assert (body['day'], body['prev_day'], body['next_day']) == ('2026-10-02', '2026-10-01', '2026-10-03')
    assert body['day_text'] == 'Fr, 02.10.2026'
    first = body['meetings'][0]
    assert first == {
        'page_id': P_B, 'title': 'Mittag', 'type': 'Jour Fixe',
        'url': 'https://app.notion.com/p/' + P_B.replace('-', ''),
        'day': '2026-10-02', 'weekday': 'Fr', 'date_text': '02.10.2026',
        'time_text': '14:30–15:00', 'all_day': False,
        'length_minutes': 30, 'length_text': '30 min',
        'has_transcript': False, 'linked': False, 'linked_here': False, 'notnotion': False,
    }


def test_client_utc_time_counts_as_a_known_time(app, authenticated_client, test_user, mcp):
    mcp.meetings = five_meetings()
    # 12:40 UTC = 14:40 Berlin (CEST)
    conv_id = make_conv(app, test_user['id'], recorded_at='2026-10-02T12:40:00+00:00', source='client')
    body = candidates(authenticated_client, conv_id).get_json()
    assert body['reference']['time_known'] is True
    assert body['reference']['text'] == 'Fr, 02.10.2026, 14:40'
    assert body['preselected_page_id'] == P_B


def test_date_only_lists_the_day_in_order_without_preselection(app, authenticated_client, test_user, mcp):
    """The normal case on Oli's data: the dictaphone filename carries a date,
    ``recorded_at`` is 00:00 local. No guess — the day's meetings in order,
    then the neighbours, and a visible hint."""
    mcp.meetings = five_meetings()
    conv_id = make_conv(app, test_user['id'], recorded_at='2026-10-02T00:00:00+02:00')
    body = candidates(authenticated_client, conv_id).get_json()
    assert [m['page_id'] for m in body['meetings']] == [P_A, P_B, P_C, P_D, P_E]
    assert body['order'] == 'chronological'
    assert body['preselected_page_id'] is None and body['preselect_reason'] is None
    assert body['reference']['time_known'] is False
    assert body['reference']['source'] == 'recorded_at'
    assert body['reference']['text'] == 'Fr, 02.10.2026'
    assert body['reference']['hint'] == 'Uhrzeit der Aufnahme unbekannt.'


@pytest.mark.parametrize('recorded_at', [None, 'gestern', 20261002, '', {'start': 'x'}])
def test_without_recorded_at_the_upload_day_is_used_and_said(app, authenticated_client, test_user,
                                                           mcp, recorded_at):
    """created_at 19:00 UTC = 21:00 Berlin lies INSIDE a 20:35–21:35 meeting —
    and still nothing is preselected: an upload time is not a recording time."""
    mcp.meetings = [meeting(P_A, 'Abend', '2026-10-02T20:35:00.000+02:00', '2026-10-02T21:35:00.000+02:00')]
    conv_id = make_conv(app, test_user['id'], meta={'recorded_at': recorded_at} if recorded_at is not None else None)
    body = candidates(authenticated_client, conv_id).get_json()
    assert body['reference']['source'] == 'created_at'
    assert body['reference']['time_known'] is False
    assert body['reference']['day'] == '2026-10-02'
    assert body['reference']['hint'] == 'Aufnahmezeit unbekannt, Upload-Zeit verwendet.'
    assert body['preselected_page_id'] is None
    assert [m['page_id'] for m in body['meetings']] == [P_A]


def test_upload_day_is_the_local_day(app, authenticated_client, test_user, mcp):
    # 22:30 UTC on Oct 2 is 00:30 on Oct 3 in Berlin.
    conv_id = make_conv(app, test_user['id'], created_at=datetime(2026, 10, 2, 22, 30, 0))
    body = candidates(authenticated_client, conv_id).get_json()
    assert body['reference']['day'] == '2026-10-03'
    assert mcp.queries() == [{'date_from': '2026-10-02', 'date_to': '2026-10-04'}]


@pytest.mark.parametrize('recorded_time, expected', [
    ('13:45:00', P_A),     # exactly start − 15 min
    ('13:44:59', None),    # one second before the window
    ('14:00:00', P_A),
    ('15:00:00', P_A),     # exactly the end
    ('15:00:01', None),    # one second after
])
def test_preselection_window_boundaries(app, authenticated_client, test_user, mcp, recorded_time, expected):
    mcp.meetings = [meeting(P_A, 'Fenster', '2026-10-02T14:00:00.000+02:00', '2026-10-02T15:00:00.000+02:00')]
    conv_id = make_conv(app, test_user['id'], recorded_at=f'2026-10-02T{recorded_time}+02:00')
    body = candidates(authenticated_client, conv_id).get_json()
    assert body['preselected_page_id'] == expected
    assert body['preselect_reason'] == ('time' if expected else None)


def test_two_meetings_in_the_window_the_nearer_start_wins(app, authenticated_client, test_user, mcp):
    mcp.meetings = [
        meeting(P_A, 'Lang', '2026-10-02T14:00:00.000+02:00', '2026-10-02T15:30:00.000+02:00'),
        meeting(P_B, 'Kurz', '2026-10-02T14:30:00.000+02:00', '2026-10-02T15:00:00.000+02:00'),
    ]
    conv_id = make_conv(app, test_user['id'], recorded_at='2026-10-02T14:25:00+02:00')
    body = candidates(authenticated_client, conv_id).get_json()
    assert body['preselected_page_id'] == P_B        # 5 min to its start, 25 to the other
    assert [m['page_id'] for m in body['meetings']] == [P_B, P_A]


def test_meeting_without_end_has_no_length_and_a_point_window(app, authenticated_client, test_user, mcp):
    mcp.meetings = [meeting(P_A, 'Offen', '2026-10-02T14:00:00.000+02:00', None)]
    conv_id = make_conv(app, test_user['id'], recorded_at='2026-10-02T14:10:00+02:00')
    body = candidates(authenticated_client, conv_id).get_json()
    entry = body['meetings'][0]
    assert entry['time_text'] == '14:00' and entry['length_minutes'] is None and entry['length_text'] is None
    assert body['preselected_page_id'] is None       # after the start, and no end to cover it


def test_all_day_entry_is_selectable_has_no_length_and_is_never_preselected(app, authenticated_client,
                                                                          test_user, mcp):
    mcp.meetings = [meeting(P_A, 'Messe', '2026-10-02', None, type='Extern', notnotion=True)]
    conv_id = make_conv(app, test_user['id'], recorded_at='2026-10-02T00:05:00+02:00')
    body = candidates(authenticated_client, conv_id).get_json()
    assert body['reference']['time_known'] is True
    entry = body['meetings'][0]
    assert entry['page_id'] == P_A and entry['all_day'] is True
    assert entry['time_text'] == 'ganztägig'
    assert entry['length_minutes'] is None and entry['length_text'] is None
    assert entry['notnotion'] is True                # stays in the list, no type filter
    assert (entry['weekday'], entry['date_text'], entry['day']) == ('Fr', '02.10.2026', '2026-10-02')
    assert body['preselected_page_id'] is None


def test_utc_stored_meeting_is_shown_in_local_time(app, authenticated_client, test_user, mcp):
    # stored 22:30Z on Oct 1 = 00:30 on Oct 2 in Berlin; the server's query
    # finds it under Oct 1 — which is why the query spans day ± 1.
    mcp.meetings = [meeting(P_A, 'Nacht', '2026-10-01T22:30:00.000+00:00', '2026-10-01T23:00:00.000+00:00')]
    conv_id = make_conv(app, test_user['id'], recorded_at='2026-10-02T00:00:00+02:00')
    entry = candidates(authenticated_client, conv_id).get_json()['meetings'][0]
    assert (entry['day'], entry['time_text'], entry['length_text']) == ('2026-10-02', '00:30–01:00', '30 min')


def test_markers_transcript_linked_and_linked_here(app, authenticated_client, test_user, mcp):
    conv_id = make_conv(app, test_user['id'], recorded_at='2026-10-02T00:00:00+02:00')
    mcp.meetings = [
        meeting(P_A, 'Fremd', '2026-10-02T09:00:00.000+02:00', '2026-10-02T10:00:00.000+02:00',
                has_transcript=True, converter_link=f'{BASE}/library/99999'),
        meeting(P_B, 'Eigen', '2026-10-02T11:00:00.000+02:00', '2026-10-02T12:30:00.000+02:00',
                has_transcript=True, converter_link=f'{BASE}/library/{conv_id}'),
        meeting(P_C, 'Frei', '2026-10-02T13:00:00.000+02:00', '2026-10-02T15:05:00.000+02:00'),
    ]
    by_id = {m['page_id']: m for m in candidates(authenticated_client, conv_id).get_json()['meetings']}
    assert (by_id[P_A]['has_transcript'], by_id[P_A]['linked'], by_id[P_A]['linked_here']) == (True, True, False)
    assert (by_id[P_B]['has_transcript'], by_id[P_B]['linked'], by_id[P_B]['linked_here']) == (True, True, True)
    assert (by_id[P_C]['has_transcript'], by_id[P_C]['linked'], by_id[P_C]['linked_here']) == (False, False, False)
    assert by_id[P_B]['length_text'] == '1 h 30 min' and by_id[P_C]['length_text'] == '2 h 05 min'


@pytest.mark.parametrize('duration, text_', [(1500, '25 min'), (59, '1 min'), (3600, '1 h'),
                                             (5400.4, '1 h 30 min'), (None, None), ('lang', None),
                                             (True, None), (-5, None)])
def test_recording_duration_text(app, authenticated_client, test_user, mcp, duration, text_):
    conv_id = make_conv(app, test_user['id'], duration=duration)
    ref = candidates(authenticated_client, conv_id).get_json()['reference']
    assert ref['duration_text'] == text_


# --- candidates: the day parameter ----------------------------------------------

def test_day_parameter_moves_the_window(app, authenticated_client, test_user, mcp):
    mcp.meetings = five_meetings() + [
        meeting(P_NEW, 'Montag', '2026-10-05T10:00:00.000+02:00', '2026-10-05T11:00:00.000+02:00')]
    conv_id = make_conv(app, test_user['id'], recorded_at='2026-10-02T14:40:00+02:00')
    body = candidates(authenticated_client, conv_id, day='2026-10-05').get_json()
    assert mcp.queries() == [{'date_from': '2026-10-04', 'date_to': '2026-10-06'}]
    assert (body['day'], body['prev_day'], body['next_day']) == ('2026-10-05', '2026-10-04', '2026-10-06')
    assert body['day_text'] == 'Mo, 05.10.2026'
    assert [m['page_id'] for m in body['meetings']] == [P_NEW]
    # the reference stays the recording; away from its day nothing is preselected
    assert body['reference']['day'] == '2026-10-02'
    assert body['order'] == 'chronological' and body['preselected_page_id'] is None


def test_day_parameter_on_the_reference_day_keeps_the_time_rule(app, authenticated_client, test_user, mcp):
    mcp.meetings = five_meetings()
    conv_id = make_conv(app, test_user['id'], recorded_at='2026-10-02T14:40:00+02:00')
    body = candidates(authenticated_client, conv_id, day='2026-10-02').get_json()
    assert body['order'] == 'distance' and body['preselected_page_id'] == P_B


@pytest.mark.parametrize('day', ['gestern', '2026-13-01', '2026-10-5', '02.10.2026', '', '2026-10-02T10:00',
                                 '2026-02-30', ' 2026-10-02'])
def test_day_parameter_is_read_strictly(app, authenticated_client, test_user, mcp, day):
    conv_id = make_conv(app, test_user['id'])
    resp = candidates(authenticated_client, conv_id, day=day)
    assert resp.status_code == 400
    assert resp.get_json()['error']
    assert mcp.calls == []


def test_neighbour_days_follow_the_shown_day(app, authenticated_client, test_user, mcp):
    mcp.meetings = five_meetings()
    conv_id = make_conv(app, test_user['id'], recorded_at='2026-10-02T00:00:00+02:00')
    body = candidates(authenticated_client, conv_id, day='2026-10-01').get_json()
    # shown day first (Vortag), then the day before (none), then the day after (Oct 2, in order)
    assert [m['page_id'] for m in body['meetings']] == [P_D, P_A, P_B, P_C]


# --- candidates: failures, auth ---------------------------------------------------

@pytest.mark.parametrize('status', [400, 401, 500, 503])
def test_query_failure_is_a_german_502(app, authenticated_client, test_user, mcp, status):
    mcp.query_status = status
    conv_id = make_conv(app, test_user['id'])
    resp = candidates(authenticated_client, conv_id)
    assert resp.status_code == 502
    assert resp.get_json() == {'error': 'Meetings konnten nicht aus Notion geladen werden. Später erneut versuchen.'}


def test_query_network_error_is_a_german_502(app, authenticated_client, test_user, mcp):
    mcp.network_down.add('/api/meetings/query')
    conv_id = make_conv(app, test_user['id'])
    resp = candidates(authenticated_client, conv_id)
    assert resp.status_code == 502
    assert 'Notion' in resp.get_json()['error']


def test_candidates_owner_404_and_login(app, client, authenticated_client, test_user, mcp):
    with app.app_context():
        other = User(username='bob')
        other.set_password('hunter2hunter2')
        db.session.add(other)
        db.session.commit()
        other_id = other.id
    foreign = make_conv(app, other_id)
    assert candidates(authenticated_client, foreign).status_code == 404
    assert send(authenticated_client, foreign, target='meetings', page_id=P_A, day='2026-10-02').status_code == 404
    assert mcp.calls == []


def test_candidates_need_a_login(app, client, test_user, mcp):
    conv_id = make_conv(app, test_user['id'])
    assert candidates(client, conv_id).status_code == 302
    assert mcp.calls == []


# --- the remembered link ----------------------------------------------------------

LINK = {'page_id': P_C, 'url': 'https://app.notion.com/p/' + P_C.replace('-', ''),
        'calendar_event_id': 'uid-c', 'meeting_title': 'Montagsrunde',
        'meeting_start': '2026-10-05T10:00:00.000+02:00', 'linked_at': '2026-10-05T12:00:00+00:00'}


def test_remembered_link_opens_on_its_day_and_is_preselected(app, authenticated_client, test_user, mcp):
    mcp.meetings = five_meetings() + [
        meeting(P_NEW, 'Daneben', '2026-10-05T09:00:00.000+02:00', '2026-10-05T09:30:00.000+02:00')]
    mcp.meetings[2] = meeting(P_C, 'Montagsrunde', '2026-10-05T10:00:00.000+02:00', '2026-10-05T11:00:00.000+02:00')
    conv_id = make_conv(app, test_user['id'], recorded_at='2026-10-02T14:40:00+02:00', meta={'notion_link': LINK})
    body = candidates(authenticated_client, conv_id).get_json()
    assert body['day'] == '2026-10-05'                       # the link's day, not the recording's
    assert mcp.queries() == [{'date_from': '2026-10-04', 'date_to': '2026-10-06'}]
    assert body['preselected_page_id'] == P_C and body['preselect_reason'] == 'linked'
    assert next(m for m in body['meetings'] if m['page_id'] == P_C)['linked_here'] is True
    assert body['link'] == {'page_id': P_C, 'url': LINK['url'], 'title': 'Montagsrunde',
                            'date_text': 'Mo, 05.10.2026, 10:00', 'day': '2026-10-05'}


def test_remembered_link_beats_the_time_rule_but_not_a_chosen_day(app, authenticated_client, test_user, mcp):
    mcp.meetings = five_meetings()
    link = dict(LINK, page_id=P_A, meeting_start='2026-10-02T09:00:00.000+02:00')
    conv_id = make_conv(app, test_user['id'], recorded_at='2026-10-02T14:40:00+02:00', meta={'notion_link': link})
    body = candidates(authenticated_client, conv_id).get_json()
    assert body['preselected_page_id'] == P_A and body['preselect_reason'] == 'linked'   # not P_B by time
    away = candidates(authenticated_client, conv_id, day='2026-10-08').get_json()
    assert away['preselected_page_id'] is None


@pytest.mark.parametrize('url', [
    'https://evil.example/p/x',
    'http://app.notion.com/p/x',                    # not https
    'javascript:alert(1)',
    'https://app.notion.com.evil.example/p/x',      # host only LOOKS like Notion
    'https://app.notion.com@evil.example/p/x',      # userinfo trick
    'https://evil.example/?https://app.notion.com/',
    '//app.notion.com/p/x',
    42, None, '',
])
def test_link_with_a_non_notion_url_is_not_delivered_as_a_link(app, authenticated_client, test_user, mcp, url):
    """``metadata_json`` is client-writable (``POST /api/conversions`` takes a
    ``metadata`` bag) — the stored URL is input."""
    conv_id = make_conv(app, test_user['id'], meta={'notion_link': dict(LINK, url=url)})
    body = candidates(authenticated_client, conv_id).get_json()
    assert body['link']['url'] is None
    assert body['link']['title'] == 'Montagsrunde'           # the text still shows
    page = authenticated_client.get(f'/library/{conv_id}').get_data(as_text=True)
    assert 'evil.example' not in page and 'javascript:alert' not in page


@pytest.mark.parametrize('url', ['https://app.notion.com/p/abc', 'https://www.notion.so/abc',
                                 'https://notion.so/abc', 'https://NOTION.so/abc'])
def test_link_with_a_notion_https_url_is_delivered(app, authenticated_client, test_user, mcp, url):
    conv_id = make_conv(app, test_user['id'], meta={'notion_link': dict(LINK, url=url)})
    assert candidates(authenticated_client, conv_id).get_json()['link']['url'] == url


@pytest.mark.parametrize('raw', ['kaputt', 7, [], {'page_id': 'not-a-page-id'}, {'url': LINK['url']},
                                 {'page_id': ['x']}])
def test_malformed_link_is_no_link(app, authenticated_client, test_user, mcp, raw):
    conv_id = make_conv(app, test_user['id'], meta={'notion_link': raw})
    body = candidates(authenticated_client, conv_id).get_json()
    assert body['link'] is None
    assert body['day'] == '2026-10-02'                       # falls back to the reference day
    assert authenticated_client.get(f'/library/{conv_id}').status_code == 200


def test_link_fields_of_the_wrong_type_are_dropped_not_rendered(app, authenticated_client, test_user, mcp):
    link = {'page_id': P_C, 'url': LINK['url'], 'meeting_title': {'x': '<script>'}, 'meeting_start': ['x']}
    conv_id = make_conv(app, test_user['id'], meta={'notion_link': link})
    body = candidates(authenticated_client, conv_id).get_json()
    assert body['link'] == {'page_id': P_C, 'url': LINK['url'], 'title': '', 'date_text': None, 'day': None}


def test_detail_page_seeds_the_link_and_hides_it_from_the_metadata_card(app, authenticated_client, test_user, mcp):
    conv_id = make_conv(app, test_user['id'], meta={'notion_link': LINK})
    page = authenticated_client.get(f'/library/{conv_id}').get_data(as_text=True)
    assert 'notionLink: {' in page and '"title": "Montagsrunde"' in page
    assert 'Notion Link' not in page                         # not a row of the generic metadata card
    plain = make_conv(app, test_user['id'])
    assert 'notionLink: null' in authenticated_client.get(f'/library/{plain}').get_data(as_text=True)


# --- send to an existing meeting ---------------------------------------------------

def test_send_payload_is_exactly_four_fields_whatever_the_request_carries(app, authenticated_client,
                                                                        test_user, mcp):
    mcp.meetings = five_meetings()
    conv_id = make_conv(app, test_user['id'], recorded_at='2026-10-02T14:40:00+02:00')
    resp = send(authenticated_client, conv_id, target='meetings', page_id=P_B, day='2026-10-02',
                # everything below would overwrite the calendar's page — none of it may travel
                title='Umbenannt', datum='2020-01-01', type='Privat', people=['Eve'], project='X',
                summary='S', transcript='vom Client untergeschoben', converter_link='https://evil.example/',
                calendar_event_id='uid-evil',
                fields={'title': 'Umbenannt', 'datum': '2020-01-01T10:00', 'type': 'Privat',
                        'people': ['Eve'], 'summary': 'S', 'transcript': 'aus fields'})
    assert resp.status_code == 200
    assert mcp.paths() == ['/api/meetings/query', '/api/meetings']
    assert mcp.queries() == [{'date_from': '2026-10-01', 'date_to': '2026-10-03'}]
    assert mcp.writes() == [{
        'page_id': P_B,
        'transcript': TRANSCRIPT,                              # from the row
        'converter_link': f'{BASE}/library/{conv_id}',
        'replace_transcript': False,
    }]
    body = resp.get_json()
    assert body['success'] is True and body['created'] is False
    assert body['link'] == {'page_id': P_B, 'url': 'https://app.notion.com/p/' + P_B.replace('-', ''),
                            'title': 'Mittag', 'date_text': 'Fr, 02.10.2026, 14:30', 'day': '2026-10-02'}


def test_send_remembers_the_link_and_keeps_the_other_metadata(app, authenticated_client, test_user, mcp):
    mcp.meetings = five_meetings()
    conv_id = make_conv(app, test_user['id'], recorded_at='2026-10-02T14:40:00+02:00',
                        meta={'job_id': 'mark-1', 'source_sha256': 'abc'})
    before = stored_meta(app, conv_id)
    assert send(authenticated_client, conv_id, target='meetings', page_id=P_B, day='2026-10-02').status_code == 200
    after = stored_meta(app, conv_id)
    link = after.pop('notion_link')
    assert after == before                                     # every other key untouched
    linked_at = link.pop('linked_at')
    assert datetime.fromisoformat(linked_at).tzinfo is not None
    assert link == {'page_id': P_B, 'url': 'https://app.notion.com/p/' + P_B.replace('-', ''),
                    'calendar_event_id': 'uid-b', 'meeting_title': 'Mittag',
                    'meeting_start': '2026-10-02T14:30:00.000+02:00'}


def test_relinking_replaces_the_whole_namespace(app, authenticated_client, test_user, mcp):
    """json_patch merges objects recursively — the writer sends every key of
    the namespace (null deletes), so nothing of the old link survives."""
    mcp.meetings = five_meetings()
    conv_id = make_conv(app, test_user['id'], recorded_at='2026-10-02T14:40:00+02:00',
                        meta={'notion_link': dict(LINK, stray='alt')})
    assert send(authenticated_client, conv_id, target='meetings', page_id=P_A, day='2026-10-02').status_code == 200
    link = stored_meta(app, conv_id)['notion_link']
    assert link['page_id'] == P_A and link['meeting_title'] == 'Früh'
    assert 'calendar_event_id' not in link                    # P_A has none — the old uid is gone


def test_transcript_exists_becomes_a_409_with_the_question(app, authenticated_client, test_user, mcp):
    mcp.meetings = five_meetings()
    mcp.meetings[1]['has_transcript'] = True
    conv_id = make_conv(app, test_user['id'], recorded_at='2026-10-02T14:40:00+02:00')
    before = stored_meta(app, conv_id)
    resp = send(authenticated_client, conv_id, target='meetings', page_id=P_B, day='2026-10-02')
    assert resp.status_code == 409
    assert resp.get_json() == {
        'code': 'transcript_exists',
        'error': '‚Mittag‘ (Fr, 02.10.2026, 14:30) hat schon ein Transkript.',
        'confirm': '‚Mittag‘ (Fr, 02.10.2026, 14:30) hat schon ein Transkript. Überschreiben?',
        'meeting_title': 'Mittag', 'meeting_date_text': 'Fr, 02.10.2026, 14:30',
    }
    assert stored_meta(app, conv_id) == before                # nothing remembered
    assert mcp.writes()[0]['replace_transcript'] is False


def test_replace_transcript_true_goes_through(app, authenticated_client, test_user, mcp):
    mcp.meetings = five_meetings()
    mcp.meetings[1]['has_transcript'] = True
    conv_id = make_conv(app, test_user['id'], recorded_at='2026-10-02T14:40:00+02:00')
    resp = send(authenticated_client, conv_id, target='meetings', page_id=P_B, day='2026-10-02',
                replace_transcript=True)
    assert resp.status_code == 200
    assert mcp.writes() == [{'page_id': P_B, 'transcript': TRANSCRIPT,
                             'converter_link': f'{BASE}/library/{conv_id}', 'replace_transcript': True}]
    assert stored_meta(app, conv_id)['notion_link']['page_id'] == P_B


@pytest.mark.parametrize('flag', ['true', 1, 'ja', [True], None])
def test_replace_transcript_must_be_a_real_boolean(app, authenticated_client, test_user, mcp, flag):
    """An overwrite flag is read by identity, never by truthiness."""
    mcp.meetings = five_meetings()
    conv_id = make_conv(app, test_user['id'])
    resp = send(authenticated_client, conv_id, target='meetings', page_id=P_B, day='2026-10-02',
                replace_transcript=flag)
    assert resp.status_code == 400
    assert mcp.calls == []


def test_too_long_transcript_is_a_413_with_both_numbers(app, authenticated_client, test_user, mcp):
    mcp.meetings = five_meetings()
    conv_id = make_conv(app, test_user['id'], content='x' * 212345)
    before = stored_meta(app, conv_id)
    resp = send(authenticated_client, conv_id, target='meetings', page_id=P_B, day='2026-10-02')
    assert resp.status_code == 413
    assert resp.get_json() == {
        'code': 'transcript_too_long', 'length': 212345, 'max': 200000,
        'error': 'Das Transkript ist zu lang für Notion: 212.345 Zeichen, erlaubt sind 200.000.',
    }
    assert stored_meta(app, conv_id) == before
    assert mcp.meetings[1]['has_transcript'] is False


def test_413_without_numbers_still_speaks_german(app, authenticated_client, test_user, mcp):
    mcp.meetings = five_meetings()
    mcp.forced_write = (413, {'error': 'too long', 'code': 'transcript_too_long'})
    conv_id = make_conv(app, test_user['id'])
    resp = send(authenticated_client, conv_id, target='meetings', page_id=P_B, day='2026-10-02')
    assert resp.status_code == 413
    assert resp.get_json()['error'] == 'Das Transkript ist zu lang für Notion.'


@pytest.mark.parametrize('upstream, expected', [
    (400, 400), (404, 404), (503, 503),
    (401, 502), (500, 502), (502, 502), (409, 502),          # a 409 WITHOUT the transcript code
])
def test_upstream_errors_are_translated_not_relayed(app, authenticated_client, test_user, mcp,
                                                    upstream, expected):
    mcp.meetings = five_meetings()
    secret_text = 'registry stale: run refresh_registry.py --db MEETINGS'
    mcp.forced_write = (upstream, {'error': secret_text})
    conv_id = make_conv(app, test_user['id'])
    before = stored_meta(app, conv_id)
    resp = send(authenticated_client, conv_id, target='meetings', page_id=P_B, day='2026-10-02')
    assert resp.status_code == expected
    body = resp.get_json()
    assert set(body) == {'error'}
    assert secret_text not in body['error'] and 'registry' not in body['error']
    assert body['error'].count('.') <= 2                       # house microcopy: two sentences at most
    assert stored_meta(app, conv_id) == before


def test_upstream_non_json_answer_is_a_502(app, authenticated_client, test_user, mcp):
    mcp.meetings = five_meetings()
    mcp.forced_write = (502, None)
    conv_id = make_conv(app, test_user['id'])
    resp = send(authenticated_client, conv_id, target='meetings', page_id=P_B, day='2026-10-02')
    assert resp.status_code == 502 and set(resp.get_json()) == {'error'}


def test_network_error_on_the_write_is_a_german_502(app, authenticated_client, test_user, mcp):
    mcp.meetings = five_meetings()
    mcp.network_down.add('/api/meetings')
    conv_id = make_conv(app, test_user['id'])
    resp = send(authenticated_client, conv_id, target='meetings', page_id=P_B, day='2026-10-02')
    assert resp.status_code == 502
    assert resp.get_json() == {'error': 'Notion-Server nicht erreichbar. Später erneut versuchen.'}
    assert 'notion_link' not in stored_meta(app, conv_id)


def test_meeting_missing_from_the_query_is_a_404_and_nothing_is_written(app, authenticated_client,
                                                                       test_user, mcp):
    mcp.meetings = five_meetings()
    conv_id = make_conv(app, test_user['id'])
    resp = send(authenticated_client, conv_id, target='meetings', page_id=P_NEW, day='2026-10-02')
    assert resp.status_code == 404
    assert resp.get_json() == {'error': 'Meeting nicht gefunden. Liste neu laden.'}
    assert mcp.writes() == []


def test_query_failure_before_the_write_sends_nothing(app, authenticated_client, test_user, mcp):
    mcp.meetings = five_meetings()
    mcp.query_status = 503
    conv_id = make_conv(app, test_user['id'])
    resp = send(authenticated_client, conv_id, target='meetings', page_id=P_B, day='2026-10-02')
    assert resp.status_code == 502
    assert mcp.writes() == []


@pytest.mark.parametrize('page_id', ['', 'abc', '../../etc/passwd', P_B + '/x', 'P' * 36, 123,
                                     ['x'], P_B.replace('-', '')[:-1]])
def test_page_id_is_checked_before_it_reaches_the_other_side(app, authenticated_client, test_user, mcp, page_id):
    conv_id = make_conv(app, test_user['id'])
    resp = send(authenticated_client, conv_id, target='meetings', page_id=page_id, day='2026-10-02')
    assert resp.status_code == 400
    assert mcp.calls == []


def test_page_id_without_dashes_is_accepted_and_matched(app, authenticated_client, test_user, mcp):
    mcp.meetings = five_meetings()
    conv_id = make_conv(app, test_user['id'])
    resp = send(authenticated_client, conv_id, target='meetings', page_id=P_B.replace('-', '').upper(),
                day='2026-10-02')
    assert resp.status_code == 200
    assert mcp.writes()[0]['page_id'] == P_B                   # the id as Notion delivered it


@pytest.mark.parametrize('target', ['notes', 'inbox', 'evil', None])
def test_page_id_only_goes_with_the_meetings_target(app, authenticated_client, test_user, mcp, target):
    conv_id = make_conv(app, test_user['id'])
    resp = send(authenticated_client, conv_id, target=target, page_id=P_B, day='2026-10-02')
    assert resp.status_code == 400
    assert mcp.calls == []


@pytest.mark.parametrize('day', [None, '', 'heute', '2026-10-32', 20261002])
def test_send_needs_the_meeting_day(app, authenticated_client, test_user, mcp, day):
    conv_id = make_conv(app, test_user['id'])
    body = {'target': 'meetings', 'page_id': P_B}
    if day is not None:
        body['day'] = day
    resp = send(authenticated_client, conv_id, **body)
    assert resp.status_code == 400
    assert mcp.calls == []


@pytest.mark.parametrize('content', ['', '   \n '])
def test_empty_document_is_not_sent(app, authenticated_client, test_user, mcp, content):
    """The other side reads ``transcript: ""`` as "do not send" — it would
    write only the link and answer 200."""
    mcp.meetings = five_meetings()
    conv_id = make_conv(app, test_user['id'], content=content)
    resp = send(authenticated_client, conv_id, target='meetings', page_id=P_B, day='2026-10-02')
    assert resp.status_code == 400
    assert resp.get_json() == {'error': 'Dieses Dokument hat keinen Inhalt. Es wurde nichts gesendet.'}
    assert mcp.calls == []


def test_converter_link_falls_back_to_the_request_root(app, authenticated_client, test_user, mcp, monkeypatch):
    monkeypatch.setattr(notion, 'PUBLIC_BASE_URL', '', raising=False)
    mcp.meetings = five_meetings()
    conv_id = make_conv(app, test_user['id'])
    assert send(authenticated_client, conv_id, target='meetings', page_id=P_B, day='2026-10-02').status_code == 200
    assert mcp.writes()[0]['converter_link'] == f'http://localhost.test/library/{conv_id}'


# --- "Neues Meeting anlegen": unchanged but for the link --------------------------

def test_new_meeting_payload_is_the_fields_plus_converter_link(app, authenticated_client, test_user, mcp):
    conv_id = make_conv(app, test_user['id'])
    fields = {'title': 'Ad hoc', 'datum': '2026-10-02T14:30', 'type': 'Meeting', 'people': ['Anna'],
              'summary': 'S', 'transcript': 'vom Formular', 'leer': '',
              'converter_link': 'https://evil.example/'}
    resp = send(authenticated_client, conv_id, target='meetings', fields=fields)
    assert resp.status_code == 201
    assert mcp.paths() == ['/api/meetings']
    assert mcp.writes() == [{
        'title': 'Ad hoc', 'datum': {'start': '2026-10-02T14:30:00', 'time_zone': 'Europe/Berlin'},
        'type': 'Meeting', 'people': ['Anna'], 'summary': 'S', 'transcript': 'vom Formular',
        'converter_link': f'{BASE}/library/{conv_id}',          # ours, not the client's
    }]
    body = resp.get_json()
    assert body['success'] is True and body['page_id'] == P_NEW and body['url'].endswith(P_NEW.replace('-', ''))
    link = stored_meta(app, conv_id)['notion_link']
    assert link['page_id'] == P_NEW and link['meeting_title'] == 'Ad hoc'
    assert link['meeting_start'] == '2026-10-02T14:30:00+02:00'
    assert 'calendar_event_id' not in link
    assert body['link']['date_text'] == 'Fr, 02.10.2026, 14:30'


def test_new_meeting_without_a_page_in_the_answer_remembers_nothing(app, authenticated_client, test_user, mcp):
    mcp.forced_write = (200, {'ok': True})
    conv_id = make_conv(app, test_user['id'])
    resp = send(authenticated_client, conv_id, target='meetings', fields={'title': 'T'})
    assert resp.status_code == 200 and resp.get_json() == {'ok': True}
    assert 'notion_link' not in stored_meta(app, conv_id)


@pytest.mark.parametrize('target', ['notes', 'inbox'])
def test_notes_and_inbox_are_untouched(app, authenticated_client, test_user, mcp, target):
    conv_id = make_conv(app, test_user['id'])
    resp = send(authenticated_client, conv_id, target=target, fields={'title': 'N', 'content': 'C'})
    assert resp.status_code == 201
    assert mcp.calls == [(f'/api/{target}', {'title': 'N', 'content': 'C'})]   # no converter_link
    assert 'notion_link' not in stored_meta(app, conv_id)
    assert 'link' not in resp.get_json()


def test_new_meeting_error_is_still_relayed(app, authenticated_client, test_user, mcp):
    mcp.forced_write = (400, {'error': 'datum: invalid format'})
    conv_id = make_conv(app, test_user['id'])
    resp = send(authenticated_client, conv_id, target='meetings', fields={'title': 'T', 'datum': 'morgen'})
    assert resp.status_code == 400 and resp.get_json() == {'error': 'datum: invalid format'}


def test_the_old_english_messages_are_german(app, authenticated_client, test_user, mcp):
    conv_id = make_conv(app, test_user['id'])
    resp = send(authenticated_client, conv_id, target='evil', fields={})
    assert resp.status_code == 400 and resp.get_json() == {'error': 'Ungültiges Ziel.'}
    mcp.network_down.add('/api/notes')
    resp = send(authenticated_client, conv_id, target='notes', fields={'title': 'N'})
    assert resp.status_code == 502
    assert resp.get_json() == {'error': 'Notion-Server nicht erreichbar. Später erneut versuchen.'}
    mcp.network_down.clear()
    mcp.forced_write = (500, {})
    resp = send(authenticated_client, conv_id, target='notes', fields={'title': 'N'})
    assert resp.status_code == 500 and resp.get_json() == {'error': 'Notion hat einen Fehler gemeldet.'}


# --- the merge writer ---------------------------------------------------------------

def _second_connection_adds_key(app, conv_id, key, value):
    """Another request commits between this session's load and its write."""
    with app.app_context():
        with db.engine.connect() as conn:
            conn.execute(text('UPDATE conversion SET metadata_json = json_set(metadata_json, :path, :v) '
                              'WHERE id = :cid'), {'path': f'$.{key}', 'v': value, 'cid': conv_id})
            conn.commit()


def test_merge_writer_keeps_keys_another_writer_added_meanwhile(app, test_user):
    """Positive control first: the whole-blob write this route would
    otherwise have used DOES lose the key."""
    control = make_conv(app, test_user['id'])
    merged = make_conv(app, test_user['id'])
    link = {'page_id': P_A, 'url': None, 'calendar_event_id': None, 'meeting_title': 'T',
            'meeting_start': None, 'linked_at': 'now'}
    with app.app_context():
        conv = db.session.get(Conversion, control)
        loaded = json.loads(conv.metadata_json)                # read …
        _second_connection_adds_key(app, control, 'transcription_status', 'failed')
        loaded['notion_link'] = link
        conv.metadata_json = json.dumps(loaded)                # … modify, write the whole blob
        db.session.commit()
        after = json.loads(db.session.get(Conversion, control).metadata_json)
        assert after['transcription_status'] == 'ready'        # the other writer's 'failed' is lost
        db.session.remove()

        conv = db.session.get(Conversion, merged)
        json.loads(conv.metadata_json)
        _second_connection_adds_key(app, merged, 'transcription_status', 'failed')
        library.write_metadata_keys(conv, {'notion_link': link})
        db.session.commit()
        after = json.loads(db.session.get(Conversion, merged).metadata_json)
        assert after['transcription_status'] == 'failed'       # kept
        assert after['notion_link'] == {'page_id': P_A, 'meeting_title': 'T', 'linked_at': 'now'}
        assert after['language'] == 'de' and after['duration_seconds'] == 1500
    with app.app_context():
        db.engine.dispose()                                    # fresh pool for the schema tests


@pytest.mark.parametrize('blob', [None, '', 'kein json', '[1, 2]', '"text"'])
def test_merge_writer_starts_fresh_on_a_missing_or_broken_blob(app, test_user, blob):
    conv_id = make_conv(app, test_user['id'])
    with app.app_context():
        db.session.execute(text('UPDATE conversion SET metadata_json = :b WHERE id = :cid'),
                           {'b': blob, 'cid': conv_id})
        db.session.commit()
        conv = db.session.get(Conversion, conv_id)
        library.write_metadata_keys(conv, {'notion_link': {'page_id': P_A}})
        db.session.commit()
        assert json.loads(db.session.get(Conversion, conv_id).metadata_json) == {'notion_link': {'page_id': P_A}}


def test_merge_writer_does_not_touch_updated_at_or_content_version(app, test_user):
    conv_id = make_conv(app, test_user['id'])
    with app.app_context():
        conv = db.session.get(Conversion, conv_id)
        before = (conv.updated_at, conv.content_version, conv.content)
        library.write_metadata_keys(conv, {'notion_link': {'page_id': P_A}})
        db.session.commit()
        conv = db.session.get(Conversion, conv_id)
        assert (conv.updated_at, conv.content_version, conv.content) == before


# --- compose ------------------------------------------------------------------------

def test_compose_gives_the_web_service_a_public_base_url():
    from pathlib import Path
    compose_file = Path(__file__).resolve().parents[1] / 'docker-compose.yml'
    if not compose_file.exists():
        pytest.skip('docker-compose.yml not shipped alongside the tests')
    yaml = pytest.importorskip('yaml')
    config = yaml.safe_load(compose_file.read_text())
    web_env = config['services']['markdown-converter']['environment']
    assert 'PUBLIC_BASE_URL=${PUBLIC_BASE_URL:-https://converter.smallpieces.de}' in web_env
    # only the web process builds links
    for name in ('worker', 'mineru-launcher'):
        assert not any(e.startswith('PUBLIC_BASE_URL') for e in config['services'][name].get('environment', []))
