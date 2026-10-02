"""NOTION-TZ — zone-less meeting times reach Notion with ``Europe/Berlin``.

The meeting form's ``datetime-local`` yields wall-clock time without a zone;
Notion reads an offset-free datetime as UTC (14:30 in the form → 16:30 CEST in
Notion). ``normalize_notion_datum`` attaches the one server-fixed zone, the
send route applies it to ``datum``. Also pins the one-zone-one-place rule:
learn and library use ``config.LOCAL_TZ`` itself.
"""
from unittest.mock import MagicMock, patch

import pytest

from app_pkg import config, learn, library
from app_pkg.integrations.notion import normalize_notion_datum
from models import Conversion, db


def test_local_tz_is_one_object():
    assert config.LOCAL_TZ.key == 'Europe/Berlin'
    assert learn.LOCAL_TZ is config.LOCAL_TZ
    assert library._BERLIN_TZ is config.LOCAL_TZ


@pytest.mark.parametrize('value, expected', [
    # all-day date: unchanged
    ('2026-09-29', '2026-09-29'),
    # naive minutes (what datetime-local sends): zone attached, seconds added
    ('2026-09-29T14:30', {'start': '2026-09-29T14:30:00', 'time_zone': 'Europe/Berlin'}),
    # naive seconds
    ('2026-09-29T14:30:15', {'start': '2026-09-29T14:30:15', 'time_zone': 'Europe/Berlin'}),
    # explicit offset / UTC: unchanged (time_zone is not allowed next to an offset)
    ('2026-09-29T14:30:00+02:00', '2026-09-29T14:30:00+02:00'),
    ('2026-09-29T12:30:00Z', '2026-09-29T12:30:00Z'),
    # blank / object / garbage: unchanged — the server validates
    ('', ''),
    ({'start': '2026-09-29T14:30:00', 'time_zone': 'Europe/Berlin'},
     {'start': '2026-09-29T14:30:00', 'time_zone': 'Europe/Berlin'}),
    ('morgen um halb drei', 'morgen um halb drei'),
    ('2026-13-45T25:99', '2026-13-45T25:99'),        # shape fits, calendar doesn't
    ('2026-09-29T14:30\n', '2026-09-29T14:30\n'),    # no trailing-newline match
    (None, None),
    (42, 42),
])
def test_normalize_notion_datum(value, expected):
    assert normalize_notion_datum(value) == expected


def _make_conversion(app, user_id):
    with app.app_context():
        conv = Conversion(user_id=user_id, conversion_type='markdown_input',
                          title='Doc', content='# body')
        db.session.add(conv)
        db.session.commit()
        return conv.id


def _send(client, conv_id, target, fields, status=200, body=None):
    resp = MagicMock(status_code=status)
    resp.json.return_value = body if body is not None else {'ok': True}
    with patch('app_pkg.integrations.notion.http_requests.post', return_value=resp) as post:
        r = client.post(f'/api/conversions/{conv_id}/send-to-notion',
                        json={'target': target, 'fields': fields})
    return r, post


def test_route_attaches_zone_to_naive_meeting_time(app, authenticated_client, test_user):
    conv_id = _make_conversion(app, test_user['id'])
    fields = {'title': 'NOTION-TZ Probe', 'datum': '2026-09-29T14:30',
              'content': 'Text', 'leer': ''}
    r, post = _send(authenticated_client, conv_id, 'meetings', fields)
    assert r.status_code == 200
    post.assert_called_once()
    assert post.call_args.args[0].endswith('/api/meetings')
    assert post.call_args.kwargs['json'] == {
        'title': 'NOTION-TZ Probe',
        'datum': {'start': '2026-09-29T14:30:00', 'time_zone': 'Europe/Berlin'},
        'content': 'Text',
        # NOTION-MEETING-LINK: the back link travels with every meeting send
        'converter_link': f'http://localhost.test/library/{conv_id}',
    }


def test_route_leaves_all_day_date_unchanged(app, authenticated_client, test_user):
    conv_id = _make_conversion(app, test_user['id'])
    r, post = _send(authenticated_client, conv_id, 'meetings',
                    {'title': 'T', 'datum': '2026-09-29'})
    assert r.status_code == 200
    assert post.call_args.kwargs['json'] == {
        'title': 'T', 'datum': '2026-09-29',
        'converter_link': f'http://localhost.test/library/{conv_id}'}


def test_route_notes_carry_no_datum(app, authenticated_client, test_user):
    conv_id = _make_conversion(app, test_user['id'])
    r, post = _send(authenticated_client, conv_id, 'notes', {'title': 'N', 'type': 'Idee'})
    assert r.status_code == 200
    assert post.call_args.kwargs['json'] == {'title': 'N', 'type': 'Idee'}
    assert 'datum' not in post.call_args.kwargs['json']


def test_route_relays_server_400(app, authenticated_client, test_user):
    conv_id = _make_conversion(app, test_user['id'])
    r, _ = _send(authenticated_client, conv_id, 'meetings',
                 {'title': 'T', 'datum': 'morgen'},
                 status=400, body={'error': 'datum: invalid format'})
    assert r.status_code == 400
    assert r.get_json() == {'error': 'datum: invalid format'}
