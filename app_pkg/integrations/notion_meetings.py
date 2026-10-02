"""Pure helpers behind "An Notion senden → Bestehendes Meeting".

NOTION-MEETING-LINK. No Flask, no network: the routes in ``notion.py`` hand in
what they read (the conversion's metadata, the notion-mcp-server's query
answer) and get back what the page draws. THE SERVER COMPUTES, THE JS DRAWS
(house rule since LEARN-MORE): order, preselection, day boundaries and every
display text for weekday, date and time are made here, in ``LOCAL_TZ``.

Two inputs are treated as INPUT throughout:

* ``metadata_json`` is client-writable (``POST /api/conversions`` takes a
  ``metadata`` bag) — ``recorded_at``, ``duration_seconds`` and the remembered
  ``notion_link`` are type-checked before use, and the link's URL is only
  delivered if it is https on a Notion host.
* meeting titles come from Notion — they travel as plain strings and are
  drawn as text nodes.

The reference time, measured on Oli's data before the build: 53 of 54
recordings carry only a DATE (the dictaphone filename dialect ``YYMMDD_NNNN``
has no time; ``recorded_at`` is then 00:00 local). A preselection therefore
exists only for a real time of day — "falsch vorausgewählt ist schlimmer als
nicht vorausgewählt".
"""
import math
import re
from datetime import date, datetime, time, timedelta, timezone
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app_pkg.config import LOCAL_TZ

# Preselection window of the brief: the recording started between the
# meeting's start − 15 min and its end.
PRESELECT_LEAD = timedelta(minutes=15)

HINT_TIME_UNKNOWN = 'Uhrzeit der Aufnahme unbekannt.'
HINT_UPLOAD_TIME = 'Aufnahmezeit unbekannt, Upload-Zeit verwendet.'

_WEEKDAYS = ('Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So')
_DAY_RE = re.compile(r'\d{4}-\d{2}-\d{2}', re.ASCII)
# A Notion page id is a UUID, with or without dashes.
_PAGE_ID_RE = re.compile(
    r'[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}', re.ASCII | re.IGNORECASE)
_NOTION_HOSTS = ('notion.so', 'notion.com')
_TITLE_MAX = 300


# --- small strict readers --------------------------------------------------

def parse_day(value):
    """``YYYY-MM-DD`` → ``date``; anything else (wrong shape, impossible
    calendar date, not a string) → None. Strict on purpose: the day decides
    which window of Oli's meetings is asked for."""
    if not isinstance(value, str) or not _DAY_RE.fullmatch(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def is_page_id(value):
    return isinstance(value, str) and _PAGE_ID_RE.fullmatch(value) is not None


def page_key(value):
    """Comparison form of a page id (Notion hands out both spellings)."""
    return value.replace('-', '').lower()


def safe_notion_url(value):
    """The URL if it is https on a Notion host, else None. Stored URLs are
    input — a ``javascript:`` or foreign URL must never become a link."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parts = urlsplit(value)
        host = (parts.hostname or '').lower()
        has_userinfo = parts.username is not None or parts.password is not None
    except ValueError:
        return None
    if parts.scheme != 'https' or has_userinfo:
        return None
    if not any(host == h or host.endswith('.' + h) for h in _NOTION_HOSTS):
        return None
    return value


def _parse_instant(value):
    """ISO-8601 string → aware datetime in ``LOCAL_TZ``, or None. A value
    without an offset is read as UTC (the convention of
    ``_normalize_client_recorded_at``; it is also how Notion reads one)."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(LOCAL_TZ)


# --- display texts ------------------------------------------------------------

def day_text(day):
    return f'{_WEEKDAYS[day.weekday()]}, {day:%d.%m.%Y}'


def moment_text(moment):
    """``Fr, 02.10.2026, 14:30`` for a local datetime."""
    return f'{day_text(moment.date())}, {moment:%H:%M}'


def minutes_text(minutes):
    """``25 min`` · ``1 h`` · ``1 h 05 min``."""
    hours, rest = divmod(int(minutes), 60)
    if hours == 0:
        return f'{rest} min'
    return f'{hours} h' if rest == 0 else f'{hours} h {rest:02d} min'


def duration_text(seconds):
    """Recording length for the dialog, or None if the stored value is not a
    usable number (it is input). Rounded to minutes, at least one."""
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return minutes_text(max(1, round(seconds / 60)))


# --- the reference time ---------------------------------------------------------

def reference_for(metadata, created_at):
    """What the candidates are measured against.

    ``recorded_at`` with a time of day other than 00:00:00 (local) → the time
    is known. ``recorded_at`` at 00:00 → only the DATE is known (the normal
    case, see module docstring). No usable ``recorded_at`` → the upload day
    (``created_at``, naive UTC in the DB), said out loud. ``moment`` is only
    set when the time is known.
    """
    recorded = _parse_instant(metadata.get('recorded_at'))
    duration = metadata.get('duration_seconds')
    base = {'duration_seconds': duration if duration_text(duration) is not None else None,
            'duration_text': duration_text(duration)}
    if recorded is not None:
        time_known = recorded.time().replace(microsecond=0) != time(0, 0, 0)
        return {
            'source': 'recorded_at', 'time_known': time_known, 'day': recorded.date(),
            'moment': recorded if time_known else None,
            'text': moment_text(recorded) if time_known else day_text(recorded.date()),
            'hint': None if time_known else HINT_TIME_UNKNOWN, **base,
        }
    uploaded = (created_at or datetime.now(timezone.utc).replace(tzinfo=None))
    local = uploaded.replace(tzinfo=timezone.utc).astimezone(LOCAL_TZ)
    return {'source': 'created_at', 'time_known': False, 'day': local.date(), 'moment': None,
            'text': day_text(local.date()), 'hint': HINT_UPLOAD_TIME, **base}


def reference_payload(reference):
    """The reference as the client gets it (no datetime objects)."""
    return {
        'source': reference['source'], 'time_known': reference['time_known'],
        'day': reference['day'].isoformat(), 'text': reference['text'], 'hint': reference['hint'],
        'duration_seconds': reference['duration_seconds'], 'duration_text': reference['duration_text'],
    }


# --- meetings from the query ---------------------------------------------------

def _parse_meeting_time(value, zone_name):
    """One end of a Notion date → ``(local datetime | date, all_day)`` or
    None. A zone-less time uses the date's ``time_zone`` if Notion names one,
    else UTC."""
    if not isinstance(value, str):
        return None
    if _DAY_RE.fullmatch(value):
        try:
            return date.fromisoformat(value), True
        except ValueError:
            return None
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        zone = timezone.utc
        if isinstance(zone_name, str) and zone_name:
            try:
                zone = ZoneInfo(zone_name)
            except (ZoneInfoNotFoundError, ValueError):
                zone = timezone.utc
        parsed = parsed.replace(tzinfo=zone)
    return parsed.astimezone(LOCAL_TZ), False


def read_meeting(raw):
    """A query row → the internal form, or None if it cannot be shown or
    sent (no valid page id, no readable start)."""
    if not isinstance(raw, dict) or not is_page_id(raw.get('page_id')):
        return None
    datum = raw.get('datum') if isinstance(raw.get('datum'), dict) else {}
    start = _parse_meeting_time(datum.get('start'), datum.get('time_zone'))
    if start is None:
        return None
    start_value, all_day = start
    end_value = None
    if not all_day:
        end = _parse_meeting_time(datum.get('end'), datum.get('time_zone'))
        if end is not None and not end[1] and end[0] >= start_value:
            end_value = end[0]
    title = raw.get('title') if isinstance(raw.get('title'), str) else ''
    return {
        'page_id': raw['page_id'],
        'title': title.strip()[:_TITLE_MAX] or 'Ohne Titel',
        'type': raw.get('type') if isinstance(raw.get('type'), str) else '',
        'url': safe_notion_url(raw.get('url')),
        'all_day': all_day,
        'day': start_value if all_day else start_value.date(),
        'start': None if all_day else start_value,
        'end': end_value,
        'start_raw': datum.get('start'),
        'calendar_event_id': raw.get('calendar_event_id') if isinstance(raw.get('calendar_event_id'), str) else '',
        'has_transcript': raw.get('has_transcript') is True,
        'converter_link': raw.get('converter_link') if isinstance(raw.get('converter_link'), str) else '',
        'notnotion': raw.get('notnotion') is True,
    }


def meeting_date_text(meeting):
    """``Fr, 02.10.2026, 14:30`` — or just the day for an all-day entry."""
    return day_text(meeting['day']) if meeting['all_day'] else moment_text(meeting['start'])


def _sort_instant(meeting):
    """Where an entry sits on the time line; an all-day entry at 00:00 of its
    day (it has no time — it opens its day)."""
    if meeting['all_day']:
        return datetime.combine(meeting['day'], time(0, 0), tzinfo=LOCAL_TZ)
    return meeting['start']


def _candidate_payload(meeting, own_link, linked_page):
    if meeting['all_day']:
        time_text, length = 'ganztägig', None
    elif meeting['end'] is None:
        time_text, length = f'{meeting["start"]:%H:%M}', None
    else:
        time_text = f'{meeting["start"]:%H:%M}–{meeting["end"]:%H:%M}'
        length = round((meeting['end'] - meeting['start']).total_seconds() / 60)
    linked = bool(meeting['converter_link'].strip())
    linked_here = ((linked and meeting['converter_link'].strip() == own_link)
                   or (linked_page is not None and page_key(meeting['page_id']) == linked_page))
    return {
        'page_id': meeting['page_id'], 'title': meeting['title'], 'type': meeting['type'],
        'url': meeting['url'],
        'day': meeting['day'].isoformat(), 'weekday': _WEEKDAYS[meeting['day'].weekday()],
        'date_text': f'{meeting["day"]:%d.%m.%Y}', 'time_text': time_text, 'all_day': meeting['all_day'],
        'length_minutes': length, 'length_text': minutes_text(length) if length is not None else None,
        'has_transcript': meeting['has_transcript'], 'linked': linked, 'linked_here': linked_here,
        'notnotion': meeting['notnotion'],
    }


def build_candidates(raw_meetings, reference, shown_day, own_link, link):
    """Order, preselect and dress the query rows for the dialog.

    * Time known AND the shown day is the recording's day → ordered by the
      distance between meeting start and recording time; preselected is the
      meeting whose window [start − 15 min, end] holds the recording time (of
      several: the nearest start). A meeting without an end covers only up to
      its start; all-day entries are never preselected.
    * Otherwise (only a date, the upload day, or a day the user navigated to)
      → the shown day in chronological order, then the day before, then the
      day after; NO preselection.
    * A remembered link beats both: if its page is in the list it is the
      preselection.

    Returns ``(order, candidates, preselected_page_id, reason)``.
    """
    meetings = [m for m in (read_meeting(raw) for raw in raw_meetings) if m is not None]
    by_distance = reference['time_known'] and shown_day == reference['day']
    preselected, reason = None, None

    if by_distance:
        moment = reference['moment']
        meetings.sort(key=lambda m: (abs(_sort_instant(m) - moment), _sort_instant(m)))
        hits = [m for m in meetings if not m['all_day']
                and m['start'] - PRESELECT_LEAD <= moment <= (m['end'] or m['start'])]
        if hits:
            preselected, reason = min(hits, key=lambda m: abs(m['start'] - moment))['page_id'], 'time'
    else:
        day_rank = {shown_day: 0, shown_day - timedelta(days=1): 1, shown_day + timedelta(days=1): 2}
        meetings.sort(key=lambda m: (day_rank.get(m['day'], 3), _sort_instant(m)))

    linked_page = page_key(link['page_id']) if link else None
    if linked_page is not None:
        match = next((m for m in meetings if page_key(m['page_id']) == linked_page), None)
        if match is not None:
            preselected, reason = match['page_id'], 'linked'

    candidates = [_candidate_payload(m, own_link, linked_page) for m in meetings]
    return ('distance' if by_distance else 'chronological'), candidates, preselected, reason


def find_meeting(raw_meetings, page_id):
    """The query row for ``page_id`` in internal form, or None."""
    wanted = page_key(page_id)
    for raw in raw_meetings:
        meeting = read_meeting(raw)
        if meeting is not None and page_key(meeting['page_id']) == wanted:
            return meeting
    return None


# --- the remembered link ---------------------------------------------------------

# Every key of the namespace. The merge writer sends ALL of them on each write
# (None → json_patch deletes the key): json_patch merges objects recursively,
# and a partial write would let sub-keys of the PREVIOUS link survive.
LINK_KEYS = ('page_id', 'url', 'calendar_event_id', 'meeting_title', 'meeting_start', 'linked_at')


def link_record(page_id, url, calendar_event_id, title, start_raw):
    """The ``notion_link`` namespace as it is stored."""
    return {
        'page_id': page_id,
        'url': safe_notion_url(url),
        'calendar_event_id': calendar_event_id or None,
        'meeting_title': (title or '')[:_TITLE_MAX] or None,
        'meeting_start': start_raw if isinstance(start_raw, str) and start_raw else None,
        'linked_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }


def clean_notion_link(raw):
    """The stored ``notion_link`` as the page may show it, or None.

    It is read as INPUT: no valid page id → no link; a title that is not a
    string → empty; the URL only if https on a Notion host; the date text is
    recomputed here from ``meeting_start`` (all-day date or instant), never
    passed through.
    """
    if not isinstance(raw, dict) or not is_page_id(raw.get('page_id')):
        return None
    title = raw.get('meeting_title') if isinstance(raw.get('meeting_title'), str) else ''
    start = _parse_meeting_time(raw.get('meeting_start'), None)
    date_text, day = None, None
    if start is not None:
        value, all_day = start
        date_text = day_text(value) if all_day else moment_text(value)
        day = (value if all_day else value.date()).isoformat()
    return {'page_id': raw['page_id'], 'url': safe_notion_url(raw.get('url')),
            'title': title.strip()[:_TITLE_MAX], 'date_text': date_text, 'day': day}
