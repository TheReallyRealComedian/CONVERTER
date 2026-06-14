"""Library list/detail and the /api/conversions CRUD endpoints."""
import json
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from flask import jsonify, render_template, request
from flask_login import current_user, login_required

from models import Conversion, Tag, conversion_tags, db

from .markdown_render import render_markdown_to_html


ALLOWED_CONVERSION_TYPES = {
    'document_to_markdown',
    'audio_transcription',
    'dialogue_formatting',
    'markdown_input',
    'ai_newsletter',
}

# R2-C: lifecycle triage locations. Internal keys; DE labels (Inbox/Später/
# Archiv) live in the templates. Used by the ?status filter + the PUT handler.
LIFECYCLE_STATUSES = {'inbox', 'later', 'archive'}

ALLOWED_PER_PAGE = (20, 50, 100)
DEFAULT_PER_PAGE = 20


def pagination_args(page, conversion_type, search, favorites, sort, per_page, tag='', status='', view=''):
    """Build the **kwargs for url_for('library', …) so favorites='', the
    default per_page, an empty tag, an empty status and an empty view drop out
    of the URL entirely (F-6 P9 + P12; R2-B adds the tag filter, R2-C the status
    filter, R2-D the view mode)."""
    args = {'page': page, 'type': conversion_type, 'search': search, 'sort': sort}
    if favorites:
        args['favorites'] = '1'
    if per_page != DEFAULT_PER_PAGE:
        args['per_page'] = per_page
    if tag:
        args['tag'] = tag
    if status:
        args['status'] = status
    if view:
        args['view'] = view
    return args


def get_owned_conversion(conversion_id):
    """Look up a Conversion that belongs to the current user, or 404.

    Centralises the owner-scoped lookup that was open-coded in four
    separate routes (F-010).
    """
    return Conversion.query.filter_by(
        id=conversion_id, user_id=current_user.id,
    ).first_or_404()


def _conversion_summary(conversion):
    """Slim per-row dict for the JSON list endpoint (MCP1) — everything the
    MCP needs to triage a row without shipping the full ``content`` of every
    conversion. Deliberately *not* Conversion.to_dict() (which carries the
    full body and feeds the POST/PUT responses + frontend, so it stays
    untouched). Drops ``content`` in favour of ``content_length`` +
    ``content_preview``; ``tag_refs`` is the slim {id,name} pair, not the full
    Tag.to_dict()."""
    metadata = json.loads(conversion.metadata_json) if conversion.metadata_json else {}
    content = conversion.content or ''
    return {
        'id': conversion.id,
        'conversion_type': conversion.conversion_type,
        'title': conversion.title,
        'source_filename': conversion.source_filename,
        'source_mimetype': conversion.source_mimetype,
        'source_size_bytes': conversion.source_size_bytes,
        'lifecycle_status': conversion.lifecycle_status,
        'is_favorite': conversion.is_favorite,
        'last_read_percent': conversion.last_read_percent,
        'queue_position': conversion.queue_position,
        'created_at': conversion.created_at.isoformat() if conversion.created_at else None,
        'updated_at': conversion.updated_at.isoformat() if conversion.updated_at else None,
        'tag_refs': [{'id': t.id, 'name': t.name} for t in conversion.tag_refs],
        'metadata': metadata,
        'content_length': len(content),
        'content_preview': content[:300],
    }


_BERLIN_TZ = ZoneInfo('Europe/Berlin')

# MCP1: recording-timestamp parser for recorder filenames. YYYY first (so the
# year position is never ambiguous), optional time after a T / space / - / _
# separator. The (?<!\d)/(?!\d) anchors keep the token from being a slice of a
# longer digit run, so a 10-digit blob isn't read as a date.
_RECORDED_AT_RE = re.compile(
    r'(?<!\d)'
    r'(\d{4})[-_.]?(\d{2})[-_.]?(\d{2})'                  # date: Y M D, opt sep
    r'(?:[T _-](\d{2})[:._]?(\d{2})(?:[:._]?(\d{2}))?)?'  # opt time: H M (S)
    r'(?!\d)'
)

# MCP1-FIX: the real dictation-recorder dialect — YYMMDD_NNNN (2-digit year +
# running sequence, no time), e.g. 260521_0176.MP3 = 2026-05-21. Anchored at the
# *start* of the name so only the clear recorder form fires: a 6-digit run later
# in the name (notes_260521_…) is not a match, nor is a bare date without the
# _NNNN sequence (260521.mp3). YY → 20YY; the sequence number is never a date.
_DICTATION_RE = re.compile(r'^(\d{2})(\d{2})(\d{2})_\d+')


def parse_recorded_at_from_filename(filename):
    """Best-effort recording timestamp from a recorder filename, or None.

    Recognises clear YYYY-first date(+time) tokens — ``20260612``,
    ``2026-06-12``, ``20260612_1430``, ``20260612-143005``,
    ``2026-06-12 14.30``, ``2026_06_12T14_30`` — ignoring any leading recorder
    prefix (REC/ZOOM/VOICE/WS/MIC/…) by simply searching for the substring.
    MCP1-FIX also recognises the real dictation-recorder dialect
    ``YYMMDD_NNNN`` (2-digit year + running sequence, no time) — ``260521_0176``
    = 2026-05-21 00:00 — but only when it is *anchored at the start* of the name
    (not a 6-digit run elsewhere, not a bare date without the _NNNN sequence).
    Times are read as Europe/Berlin; the result is a tz-aware datetime whose
    ``.isoformat()`` carries the offset. No time component → 00:00 local.

    Conservative by design — a wrong recorded_at poisons later meeting matching,
    so "falsch ist schlimmer als leer": the year must be 2000–2100 and y/m/d/
    h/m/s must form a real datetime, else the candidate is dropped. Returns
    None when nothing valid is found OR when the name carries two *different*
    valid dates (ambiguous). Negative cases (`Besprechung.mp3`, `audio (1).mp3`,
    `12345678.mp3`, `New Recording 7.m4a`) all yield None.
    """
    if not isinstance(filename, str):
        return None

    found = set()
    for m in _RECORDED_AT_RE.finditer(filename):
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (2000 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31):
            continue
        hour = int(m.group(4)) if m.group(4) is not None else 0
        minute = int(m.group(5)) if m.group(5) is not None else 0
        second = int(m.group(6)) if m.group(6) is not None else 0
        if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
            continue
        try:
            dt = datetime(year, month, day, hour, minute, second, tzinfo=_BERLIN_TZ)
        except ValueError:
            continue  # impossible combo, e.g. Feb 30
        found.add(dt)

    # MCP1-FIX dictation dialect (YYMMDD_NNNN, anchored at the start). Shares the
    # `found` set with the YYYY branch so the existing len(found) == 1 ambiguity
    # gate stays in force: a (contrived) name that matched both branches with
    # different dates would still be conservatively None.
    md = _DICTATION_RE.match(filename)
    if md is not None:
        year, month, day = 2000 + int(md.group(1)), int(md.group(2)), int(md.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            try:
                found.add(datetime(year, month, day, tzinfo=_BERLIN_TZ))
            except ValueError:
                pass  # impossible calendar date, e.g. 260230 Feb-30

    if len(found) == 1:
        return next(iter(found))
    return None  # zero matches, or ambiguous (multiple distinct dates)


def _normalize_client_recorded_at(value):
    """Client-supplied recorded_at → ISO-8601 UTC string, or None.

    Accepts an ISO-8601 string OR epoch-milliseconds as a number (e.g. JS
    ``file.lastModified``). A naive ISO string is read as UTC. Unparseable
    input returns None so the caller stays additive (no 400, just log + fall
    back to the filename parser)."""
    if isinstance(value, bool):
        return None  # bool is an int subclass — never a timestamp
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()
        except (ValueError, OverflowError, OSError):
            return None
    if isinstance(value, str) and value.strip():
        try:
            dt = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    return None


def register(app):
    @app.route('/library')
    @login_required
    def library():
        conversion_type = request.args.get('type', '')
        search = request.args.get('search', '').strip()
        favorites = request.args.get('favorites', '') == '1'
        # Tags are stored lowercase+trimmed (Tag.get_or_create) — normalise the
        # incoming value so ?tag=KI matches the stored "ki".
        tag = request.args.get('tag', '').strip().lower()
        # R2-C: lifecycle triage filter. Only a known status filters; anything
        # else is treated as "no status filter" (current_status stays '').
        status = request.args.get('status', '').strip()
        current_status = status if status in LIFECYCLE_STATUSES else ''
        # R2-E "Readwise-3er" IA: 'inbox' = the untriaged pile (status inbox
        # AND not queued — queueing an item triages it away without touching
        # its status; de-queueing an item that is still 'inbox' drops it back
        # into triage); 'queue' = the manual reading-list (queued + not
        # archived, ordered by position, R2-D). Anything else — including the
        # retired R2-D 'reading' — falls through to the default library list.
        # A mode, not a filter — kept out of has_active_filter so the "Filter
        # zurücksetzen" empty-state stays scoped to type/search/tag/status.
        view = request.args.get('view', '')
        current_view = view if view in ('inbox', 'queue') else ''
        sort = request.args.get('sort', 'newest')
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', DEFAULT_PER_PAGE, type=int)
        if per_page not in ALLOWED_PER_PAGE:
            per_page = DEFAULT_PER_PAGE

        query = Conversion.query.filter_by(user_id=current_user.id)

        if conversion_type and conversion_type in ALLOWED_CONVERSION_TYPES:
            query = query.filter_by(conversion_type=conversion_type)
        if favorites:
            query = query.filter_by(is_favorite=True)
        if tag:
            # R2-B: exact-match tag filter over the conversion_tags junction —
            # the same Conversion.tag_refs.any(...) path the R2-A search branch
            # uses, with == instead of ilike (the name is already normalised).
            query = query.filter(Conversion.tag_refs.any(Tag.name == tag))
        if current_status:
            query = query.filter(Conversion.lifecycle_status == current_status)
        if search:
            # R2-A: tag-search joins against the conversion_tags junction. The
            # legacy Conversion.tags CSV branch was removed — after the
            # migration the column is dead and would never match.
            # P3-fix: .contains(autoescape=True) escapes %/_ and emits the
            # ESCAPE clause itself. The previous re.sub + ilike was a silent
            # no-op — ilike has no default ESCAPE, so the inserted backslashes
            # matched literally and broke substring search (Memory
            # reference_sqlalchemy_like_escape.md). LIKE is case-insensitive on
            # SQLite, so dropping ilike keeps the case behaviour.
            query = query.filter(
                db.or_(
                    Conversion.title.contains(search, autoescape=True),
                    Conversion.content.contains(search, autoescape=True),
                    Conversion.tag_refs.any(Tag.name.contains(search, autoescape=True)),
                )
            )

        # View-mode filters AND on top of the type/tag/status/search chips.
        if current_view == 'inbox':
            query = query.filter(
                Conversion.lifecycle_status == 'inbox',
                Conversion.queue_position.is_(None),
            )
        elif current_view == 'queue':
            query = query.filter(
                Conversion.queue_position.isnot(None),
                Conversion.lifecycle_status != 'archive',
            )

        # The queue view overrides the sort param (manual position); the inbox
        # view keeps the regular sort handling (newest first by default).
        if current_view == 'queue':
            query = query.order_by(Conversion.queue_position.asc())
        elif sort == 'oldest':
            query = query.order_by(Conversion.created_at.asc())
        elif sort == 'title':
            query = query.order_by(Conversion.title.asc())
        else:
            query = query.order_by(Conversion.created_at.desc())

        pagination = query.paginate(page=page, per_page=per_page, error_out=False)

        # R2-E: "Weiterlesen" section above the queue — in-progress reads
        # (0 < last_read_percent < 95) that are neither queued (their card
        # progress bar already shows it) nor archived. Sort carried over from
        # the retired view=reading: most-recently-touched first. Unpaginated
        # and unfiltered by the chips; in practice rarely more than a handful.
        reading_items = []
        if current_view == 'queue':
            reading_items = Conversion.query.filter(
                Conversion.user_id == current_user.id,
                Conversion.last_read_percent.isnot(None),
                Conversion.last_read_percent > 0,
                Conversion.last_read_percent < 95,
                Conversion.queue_position.is_(None),
                Conversion.lifecycle_status != 'archive',
            ).order_by(Conversion.updated_at.desc()).all()

        # R2-E: badge count for the Inbox tab — untriaged = status inbox AND
        # not queued. Global per user (independent of the active filters) and
        # computed for every view because the tab bar is always visible.
        inbox_count = Conversion.query.filter(
            Conversion.user_id == current_user.id,
            Conversion.lifecycle_status == 'inbox',
            Conversion.queue_position.is_(None),
        ).count()

        has_active_filter = bool(
            (conversion_type and conversion_type in ALLOWED_CONVERSION_TYPES)
            or search
            or favorites
            or tag
            or current_status
        )

        # Tags of this user that hang on at least one conversion — feeds the
        # tag chip row. R2-E: each row carries its usage count over the
        # conversion_tags junction, ordered count-desc with alphabetical
        # tie-break so the template can slice a top-N. Rows expose .name and
        # .count (Row objects, not Tag instances).
        tag_usage = db.func.count(conversion_tags.c.conversion_id)
        available_tags = (
            db.session.query(Tag.name, tag_usage.label('count'))
            .join(conversion_tags, conversion_tags.c.tag_id == Tag.id)
            .join(Conversion, Conversion.id == conversion_tags.c.conversion_id)
            .filter(Conversion.user_id == current_user.id)
            .group_by(Tag.id, Tag.name)
            .order_by(tag_usage.desc(), Tag.name.asc())
            .all()
        )

        return render_template('library.html',
                               conversions=pagination.items,
                               pagination=pagination,
                               reading_items=reading_items,
                               inbox_count=inbox_count,
                               current_type=conversion_type,
                               current_search=search,
                               current_favorites=favorites,
                               current_sort=sort,
                               current_per_page=per_page,
                               current_tag=tag,
                               current_status=current_status,
                               current_view=current_view,
                               available_tags=available_tags,
                               allowed_per_page=ALLOWED_PER_PAGE,
                               default_per_page=DEFAULT_PER_PAGE,
                               has_active_filter=has_active_filter,
                               pagination_args=pagination_args)

    @app.route('/library/<int:conversion_id>')
    @login_required
    def library_detail(conversion_id):
        conversion = get_owned_conversion(conversion_id)
        metadata = json.loads(conversion.metadata_json) if conversion.metadata_json else {}
        content_html = render_markdown_to_html(conversion.content)
        return render_template(
            'library_detail.html',
            conversion=conversion,
            metadata=metadata,
            content_html=content_html,
            conversion_tag_refs=[t.to_dict() for t in conversion.tag_refs],
        )

    @app.route('/api/conversions', methods=['GET'])
    @login_required
    def api_list_conversions():
        """JSON list of the current user's conversions (MCP1 read-API).

        Owner-scoped, read-only sibling of the HTML /library view. Returns slim
        per-row summaries (no full ``content`` — only ``content_length`` and a
        300-char ``content_preview``, see _conversion_summary) plus a ``total``
        count taken *before* the limit/offset window.

        Query params (all optional):
          type           — filter conversion_type; must be in
                           ALLOWED_CONVERSION_TYPES, else 400.
          status         — exact lifecycle_status filter; must be in
                           LIFECYCLE_STATUSES, else 400.
          exclude_status — exclude this lifecycle_status; must be in
                           LIFECYCLE_STATUSES, else 400. (exclude_status=archive
                           yields the "unarchived" set the MCP wants.)
          limit          — page size, default 100. Values >500 are *capped* to
                           500 (not rejected); non-int or <1 → 400.
          offset         — page offset, default 0; non-int or <0 → 400.

        Sorted created_at desc (newest first).
        """
        conversion_type = request.args.get('type', '')
        if conversion_type and conversion_type not in ALLOWED_CONVERSION_TYPES:
            return jsonify({'error': f'Invalid conversion type: {conversion_type}'}), 400

        status = request.args.get('status', '')
        if status and status not in LIFECYCLE_STATUSES:
            return jsonify({'error': 'Ungültiger Lifecycle-Status.'}), 400

        exclude_status = request.args.get('exclude_status', '')
        if exclude_status and exclude_status not in LIFECYCLE_STATUSES:
            return jsonify({'error': 'Ungültiger Lifecycle-Status.'}), 400

        # Parse limit/offset by hand (not request.args.get(type=int)) so an
        # absent param falls back to the default while a present-but-invalid one
        # is a 400 — type=int conflates the two by yielding None for both.
        limit_raw = request.args.get('limit')
        if limit_raw is None:
            limit = 100
        else:
            try:
                limit = int(limit_raw)
            except (TypeError, ValueError):
                return jsonify({'error': 'Feld "limit" muss eine ganze Zahl ≥ 1 sein.'}), 400
            if limit < 1:
                return jsonify({'error': 'Feld "limit" muss eine ganze Zahl ≥ 1 sein.'}), 400
            limit = min(limit, 500)  # cap, not reject (see docstring)

        offset_raw = request.args.get('offset')
        if offset_raw is None:
            offset = 0
        else:
            try:
                offset = int(offset_raw)
            except (TypeError, ValueError):
                return jsonify({'error': 'Feld "offset" muss eine ganze Zahl ≥ 0 sein.'}), 400
            if offset < 0:
                return jsonify({'error': 'Feld "offset" muss eine ganze Zahl ≥ 0 sein.'}), 400

        query = Conversion.query.filter_by(user_id=current_user.id)
        if conversion_type:
            query = query.filter_by(conversion_type=conversion_type)
        if status:
            query = query.filter(Conversion.lifecycle_status == status)
        if exclude_status:
            query = query.filter(Conversion.lifecycle_status != exclude_status)

        total = query.count()  # full match count, before the limit/offset window
        rows = (query.order_by(Conversion.created_at.desc())
                     .limit(limit).offset(offset).all())

        return jsonify({
            'items': [_conversion_summary(c) for c in rows],
            'total': total,
            'limit': limit,
            'offset': offset,
        })

    @app.route('/api/conversions', methods=['POST'])
    @login_required
    def api_create_conversion():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400
        if not data.get('content'):
            return jsonify({'error': 'Content is required'}), 400

        conversion_type = data.get('conversion_type', 'unknown')
        if conversion_type not in ALLOWED_CONVERSION_TYPES:
            return jsonify({'error': f'Invalid conversion type: {conversion_type}'}), 400

        title = data.get('title', 'Untitled')[:255]

        # MCP1: take the metadata bag defensively (must be a dict, else {}).
        metadata = data.get('metadata', {})
        if not isinstance(metadata, dict):
            metadata = {}

        # MCP1 recorded_at-capture — additive, no schema touch, never a 400.
        # Only fill it when the client didn't already put a recorded_at in the
        # metadata bag itself. MCP1-FIX precedence flip: the device-authoritative
        # source_filename date now beats the client recorded_at field, because the
        # client value (file.lastModified) can be the copy time rather than the
        # recording time, whereas the dictation filename carries the recording
        # date. So: explicit metadata.recorded_at (handled by the guard above) >
        # source_filename > client field. The client field is only consulted when
        # the filename yields nothing; an unparseable client value is then logged
        # and dropped (stay additive — no 400).
        if 'recorded_at' not in metadata:
            recorded_at, source = None, None
            parsed = parse_recorded_at_from_filename(data.get('source_filename'))
            if parsed is not None:
                recorded_at, source = parsed.isoformat(), 'filename'
            elif 'recorded_at' in data:
                recorded_at = _normalize_client_recorded_at(data.get('recorded_at'))
                if recorded_at is not None:
                    source = 'client'
                else:
                    app.logger.warning(
                        'recorded_at unparseable, ignored: %r', data.get('recorded_at'))
            if recorded_at is not None:
                metadata['recorded_at'] = recorded_at
                metadata['recorded_at_source'] = source

        conversion = Conversion(
            user_id=current_user.id,
            conversion_type=conversion_type,
            title=title,
            content=data['content'],
            source_filename=data.get('source_filename'),
            source_mimetype=data.get('source_mimetype'),
            source_size_bytes=data.get('source_size_bytes'),
            metadata_json=json.dumps(metadata),
            tags=data.get('tags', ''),
        )
        db.session.add(conversion)
        db.session.commit()
        return jsonify(conversion.to_dict()), 201

    @app.route('/api/conversions/<int:conversion_id>', methods=['GET'])
    @login_required
    def api_get_conversion(conversion_id):
        """Full JSON of one owned conversion (MCP1 read-API), including the
        full ``content`` + parsed ``metadata`` via Conversion.to_dict(). 404
        (via get_owned_conversion) for a missing or foreign id."""
        conversion = get_owned_conversion(conversion_id)
        return jsonify(conversion.to_dict())

    @app.route('/api/conversions/<int:conversion_id>', methods=['PUT'])
    @login_required
    def api_update_conversion(conversion_id):
        conversion = get_owned_conversion(conversion_id)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400

        # R2-A: the legacy CSV `tags` path on PUT is gone. The frontend now
        # uses /api/conversions/<id>/tags POST + DELETE for attach/detach
        # against the conversion_tags junction; the old CSV column is a
        # dead column that the migration helper has already drained.
        if 'title' in data:
            conversion.title = str(data['title'])[:255]
        if 'content' in data:
            conversion.content = data['content']
        if 'is_favorite' in data:
            conversion.is_favorite = bool(data['is_favorite'])
        if 'lifecycle_status' in data:
            # R2-C: the status toggle (Card + Detail) PUTs here, like is_favorite.
            if data['lifecycle_status'] not in LIFECYCLE_STATUSES:
                return jsonify({'error': 'Ungültiger Lifecycle-Status.'}), 400
            conversion.lifecycle_status = data['lifecycle_status']

        db.session.commit()
        return jsonify(conversion.to_dict())

    @app.route('/api/conversions/<int:conversion_id>', methods=['DELETE'])
    @login_required
    def api_delete_conversion(conversion_id):
        conversion = get_owned_conversion(conversion_id)
        db.session.delete(conversion)
        db.session.commit()
        return jsonify({'success': True})

    @app.route('/api/conversions/<int:conversion_id>/progress', methods=['PATCH'])
    @login_required
    def api_update_conversion_progress(conversion_id):
        conversion = get_owned_conversion(conversion_id)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400

        # R2-G "als ungelesen markieren": an explicit reset flag clears the mark
        # to NULL ("never read", identical to fresh-ingested — no card bar, not
        # in the Weiterlesen section) and deliberately bypasses the forward-clamp
        # below. This is the explicit-flag escape hatch the R2-F note called for.
        # reset must be a real bool (same explicit-type stance as the percent
        # bool-check below): truthy garbage like 1 / "true" is a 400, not a
        # silent reset, so the clamp can only ever be bypassed on purpose.
        if 'reset' in data:
            if not isinstance(data['reset'], bool):
                return jsonify({'error': 'Feld "reset" muss ein Boolean sein.'}), 400
            if data['reset']:
                conversion.last_read_percent = None
                db.session.commit()
                return jsonify({'success': True, 'last_read_percent': None}), 200
            # reset:false falls through to the normal forward-clamp path below.

        percent = data.get('percent')
        # Bool is an int subclass — reject it explicitly so True/False can't
        # pose as a 1/0 percent. Then require a real number: missing / None /
        # non-numeric is a 400 (atomic endpoint, no hidden no-op like R1-B-B).
        if isinstance(percent, bool) or not isinstance(percent, (int, float)):
            return jsonify({'error': 'Feld "percent" fehlt oder ist ungültig.'}), 400

        # Out-of-range is a rounding artefact of a fire-and-forget scroll
        # signal (e.g. 100.0001), not a client error — clamp instead of 400.
        percent = max(0.0, min(100.0, float(percent)))
        # R2-F forward-clamp: furthest-read only ever moves forward. The client
        # already sends only maxReached (forward-only, seeded from the stored
        # value), but this is a fire-and-forget signal over a shared endpoint —
        # clamping against the stored value guarantees a stale/re-seeded/buggy
        # client can never lower the mark. Phase 1 confirmed no active bug; this
        # is defence-in-depth. The one sanctioned way to lower it is the explicit
        # reset flag above (R2-G "als ungelesen markieren").
        stored = conversion.last_read_percent or 0.0
        percent = max(stored, percent)
        conversion.last_read_percent = percent
        db.session.commit()
        return jsonify({'success': True, 'last_read_percent': percent}), 200

    @app.route('/api/conversions/<int:conversion_id>/queue', methods=['POST'])
    @login_required
    def api_update_conversion_queue(conversion_id):
        # R2-D reading-list queue: add/remove toggle + up/down reorder. Float
        # positions so a future drag-reorder can slot between neighbours.
        conversion = get_owned_conversion(conversion_id)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400

        action = data.get('action')
        if action not in {'add', 'remove', 'up', 'down'}:
            return jsonify({'error': 'Ungültige Queue-Aktion.'}), 400

        if action == 'add':
            # Already on the list → no-op; otherwise append at the end. max()
            # ignores NULLs in SQL, so this is the largest position in use.
            if conversion.queue_position is None:
                max_pos = db.session.query(db.func.max(Conversion.queue_position)).filter(
                    Conversion.user_id == current_user.id,
                ).scalar()
                conversion.queue_position = (max_pos + 1.0) if max_pos is not None else 1.0
        elif action == 'remove':
            conversion.queue_position = None
        else:
            # up/down: swap queue_position with the direct neighbour among the
            # user's queued items (sorted asc). Two-row update, one commit
            # boundary. No-op at the top/bottom edge or when not queued at all.
            # The set MUST match the queue-view's visible set (archive excluded,
            # Decision #5 keeps archived rows queued) — else "up" swaps with an
            # invisible archived neighbour and appears to do nothing.
            if conversion.queue_position is not None:
                queued = Conversion.query.filter(
                    Conversion.user_id == current_user.id,
                    Conversion.queue_position.isnot(None),
                    Conversion.lifecycle_status != 'archive',
                ).order_by(Conversion.queue_position.asc()).all()
                idx = next((i for i, c in enumerate(queued) if c.id == conversion.id), None)
                if idx is not None:
                    neighbour_idx = (idx - 1) if action == 'up' else (idx + 1)
                    if 0 <= neighbour_idx < len(queued):
                        neighbour = queued[neighbour_idx]
                        conversion.queue_position, neighbour.queue_position = (
                            neighbour.queue_position, conversion.queue_position,
                        )

        db.session.commit()
        return jsonify(conversion.to_dict())

    @app.route('/api/conversions/<int:conversion_id>/tags', methods=['POST'])
    @login_required
    def api_attach_conversion_tag(conversion_id):
        conversion = get_owned_conversion(conversion_id)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400

        tag = Tag.get_or_create(current_user.id, data.get('name'))
        if tag is None:
            return jsonify({
                'error': f'Tag-Name fehlt oder ist zu lang (max {Tag.MAX_NAME_LEN} Zeichen).'
            }), 400

        if tag in conversion.tag_refs:
            # Idempotent attach — no-op 200 statt 409, mirrors highlight POST.
            db.session.commit()
            return jsonify(tag.to_dict()), 200

        conversion.tag_refs.append(tag)
        db.session.commit()
        return jsonify(tag.to_dict()), 201

    @app.route('/api/conversions/<int:conversion_id>/tags/<int:tag_id>', methods=['DELETE'])
    @login_required
    def api_detach_conversion_tag(conversion_id, tag_id):
        conversion = get_owned_conversion(conversion_id)
        tag = Tag.query.get_or_404(tag_id)
        if tag.user_id != current_user.id:
            return jsonify({'error': 'Nicht gefunden.'}), 404
        if tag in conversion.tag_refs:
            conversion.tag_refs.remove(tag)
            db.session.commit()
        return jsonify({'success': True})
