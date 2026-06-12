"""Conversion-Tag API + CSV-Migration characterization tests (R2-A).

Locks in:
- POST/DELETE on /api/conversions/<id>/tags wires the new conversion_tags
  junction analogous to the highlight-tag endpoints (find-or-create, lowercase
  + trim, idempotent attach, detach without deleting the tag).
- Tag.get_or_create is the single source of truth for normalisation across
  highlight-tag-POST, conversion-tag-POST, and the migration helper. Same
  string → same row regardless of which call site asks first.
- The _migrate_conversion_tags_csv_to_junction helper drains the legacy
  Conversion.tags CSV column into the junction idempotently — the empty-CSV
  marker means a second migration pass is a no-op.
- DELETE /api/tags/<id> cascades through *both* junctions.
"""
import json
import re

from app_pkg import _migrate_conversion_tags_csv_to_junction
from models import Conversion, Highlight, Tag, User, conversion_tags, db


def _make_conversion(app, user_id, **overrides):
    payload = dict(
        user_id=user_id,
        conversion_type='markdown_input',
        title='Sample Title',
        content='Sample content body.',
        tags='',
        metadata_json=json.dumps({'src': 'test'}),
    )
    payload.update(overrides)
    with app.app_context():
        c = Conversion(**payload)
        db.session.add(c)
        db.session.commit()
        return c.id


def _make_highlight(app, conversion_id, exact='Sample content'):
    with app.app_context():
        h = Highlight(conversion_id=conversion_id, exact=exact, prefix='', suffix='')
        db.session.add(h)
        db.session.commit()
        return h.id


def _make_other_user(app, username='bob'):
    with app.app_context():
        u = User(username=username)
        u.set_password('password1234')
        db.session.add(u)
        db.session.commit()
        return u.id


# --- POST /api/conversions/<id>/tags ---

def test_api_attach_conversion_tag_creates_and_attaches(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    resp = authenticated_client.post(f'/api/conversions/{cid}/tags', json={'name': 'ki'})
    assert resp.status_code == 201
    body = resp.get_json()
    assert body['name'] == 'ki'
    with app.app_context():
        tag = Tag.query.filter_by(name='ki').first()
        assert tag is not None
        assert tag.user_id == test_user['id']
        c = Conversion.query.get(cid)
        assert tag in c.tag_refs


def test_api_attach_conversion_tag_idempotent_when_already_attached(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    r1 = authenticated_client.post(f'/api/conversions/{cid}/tags', json={'name': 'dup'})
    assert r1.status_code == 201
    r2 = authenticated_client.post(f'/api/conversions/{cid}/tags', json={'name': 'dup'})
    assert r2.status_code == 200
    assert r2.get_json()['name'] == 'dup'
    with app.app_context():
        c = Conversion.query.get(cid)
        assert len([t for t in c.tag_refs if t.name == 'dup']) == 1


def test_api_attach_conversion_tag_normalizes_lowercase_and_trim(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    resp = authenticated_client.post(f'/api/conversions/{cid}/tags', json={'name': '  PRODUKT  '})
    assert resp.status_code == 201
    assert resp.get_json()['name'] == 'produkt'
    resp2 = authenticated_client.post(f'/api/conversions/{cid}/tags', json={'name': 'Produkt'})
    assert resp2.status_code == 200
    assert resp2.get_json()['id'] == resp.get_json()['id']


def test_api_attach_conversion_tag_rejects_empty_and_oversized(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    assert authenticated_client.post(f'/api/conversions/{cid}/tags', json={'name': ''}).status_code == 400
    assert authenticated_client.post(f'/api/conversions/{cid}/tags', json={'name': '   '}).status_code == 400
    assert authenticated_client.post(f'/api/conversions/{cid}/tags', json={'name': 'x' * 81}).status_code == 400
    assert authenticated_client.post(f'/api/conversions/{cid}/tags', json={'name': None}).status_code == 400


def test_api_attach_conversion_tag_rejects_non_dict_body(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    resp = authenticated_client.post(f'/api/conversions/{cid}/tags', json=['not', 'a', 'dict'])
    assert resp.status_code == 400


def test_api_attach_conversion_tag_to_foreign_conversion_returns_404(app, authenticated_client, test_user):
    other_id = _make_other_user(app, 'bob')
    cid = _make_conversion(app, other_id, title="Bob's doc")
    resp = authenticated_client.post(f'/api/conversions/{cid}/tags', json={'name': 'intruder'})
    assert resp.status_code == 404


# --- DELETE /api/conversions/<id>/tags/<tag_id> ---

def test_api_detach_conversion_tag_removes_junction_keeps_tag(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    attach = authenticated_client.post(f'/api/conversions/{cid}/tags', json={'name': 'keepme'})
    tag_id = attach.get_json()['id']

    resp = authenticated_client.delete(f'/api/conversions/{cid}/tags/{tag_id}')
    assert resp.status_code == 200
    with app.app_context():
        c = Conversion.query.get(cid)
        assert all(t.id != tag_id for t in c.tag_refs)
        assert Tag.query.get(tag_id) is not None


def test_api_detach_foreign_user_tag_returns_404(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    other_id = _make_other_user(app, 'carol')
    with app.app_context():
        foreign_tag = Tag(user_id=other_id, name='secret')
        db.session.add(foreign_tag)
        db.session.commit()
        foreign_tag_id = foreign_tag.id
    resp = authenticated_client.delete(f'/api/conversions/{cid}/tags/{foreign_tag_id}')
    assert resp.status_code == 404


def test_api_detach_conversion_tag_from_foreign_conversion_returns_404(app, authenticated_client, test_user):
    other_id = _make_other_user(app, 'bob')
    cid = _make_conversion(app, other_id, title="Bob's doc")
    with app.app_context():
        foreign_tag = Tag(user_id=other_id, name='secret')
        db.session.add(foreign_tag)
        db.session.commit()
        foreign_tag_id = foreign_tag.id
    resp = authenticated_client.delete(f'/api/conversions/{cid}/tags/{foreign_tag_id}')
    assert resp.status_code == 404


# --- Cross-domain Tag-Reuse: same Tag-row backs Highlight + Conversion ---

def test_tag_reuse_across_highlights_and_conversions(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    hid = _make_highlight(app, cid)
    h_resp = authenticated_client.post(f'/api/highlights/{hid}/tags', json={'name': 'shared'})
    h_tag_id = h_resp.get_json()['id']
    c_resp = authenticated_client.post(f'/api/conversions/{cid}/tags', json={'name': 'shared'})
    assert c_resp.get_json()['id'] == h_tag_id
    with app.app_context():
        assert Tag.query.filter_by(name='shared').count() == 1


# --- Conversion.to_dict includes tag_refs ---

def test_conversion_to_dict_includes_tag_refs(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    authenticated_client.post(f'/api/conversions/{cid}/tags', json={'name': 'meta'})
    with app.app_context():
        c = Conversion.query.get(cid)
        d = c.to_dict()
        assert 'tag_refs' in d
        assert len(d['tag_refs']) == 1
        assert d['tag_refs'][0]['name'] == 'meta'


# --- DELETE /api/tags/<id> cascades through BOTH junctions ---

def test_api_delete_tag_cascades_to_both_junctions(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    hid = _make_highlight(app, cid)
    attach = authenticated_client.post(f'/api/conversions/{cid}/tags', json={'name': 'bothsides'})
    tag_id = attach.get_json()['id']
    authenticated_client.post(f'/api/highlights/{hid}/tags', json={'name': 'bothsides'})

    resp = authenticated_client.delete(f'/api/tags/{tag_id}')
    assert resp.status_code == 200
    with app.app_context():
        assert Tag.query.get(tag_id) is None
        c = Conversion.query.get(cid)
        h = Highlight.query.get(hid)
        assert all(t.id != tag_id for t in c.tag_refs)
        assert all(t.id != tag_id for t in h.tags)


# --- Migration-Helper Idempotenz ---

def _seed_conversion_with_csv(app, user_id, csv_value):
    with app.app_context():
        c = Conversion(
            user_id=user_id,
            conversion_type='markdown_input',
            title='With CSV tags',
            content='body',
            tags=csv_value,
        )
        db.session.add(c)
        db.session.commit()
        return c.id


def test_migration_helper_drains_csv_into_junction(app, test_user):
    cid = _seed_conversion_with_csv(app, test_user['id'], 'eins, zwei, drei')
    with app.app_context():
        _migrate_conversion_tags_csv_to_junction(app)
        c = Conversion.query.get(cid)
        assert c.tags == ''
        names = sorted(t.name for t in c.tag_refs)
        assert names == ['drei', 'eins', 'zwei']


def test_migration_helper_is_idempotent_on_second_run(app, test_user):
    cid = _seed_conversion_with_csv(app, test_user['id'], 'eins, zwei')
    with app.app_context():
        _migrate_conversion_tags_csv_to_junction(app)
        first_tag_ids = sorted(t.id for t in Conversion.query.get(cid).tag_refs)
        _migrate_conversion_tags_csv_to_junction(app)
        second_tag_ids = sorted(t.id for t in Conversion.query.get(cid).tag_refs)
        assert first_tag_ids == second_tag_ids
        assert Tag.query.filter(Tag.name.in_(['eins', 'zwei'])).count() == 2


def test_migration_helper_normalizes_csv_entries(app, test_user):
    cid = _seed_conversion_with_csv(app, test_user['id'], ' KI , Produkt ')
    with app.app_context():
        _migrate_conversion_tags_csv_to_junction(app)
        c = Conversion.query.get(cid)
        names = sorted(t.name for t in c.tag_refs)
        # lowercase + trimmed by Tag.get_or_create
        assert names == ['ki', 'produkt']


def test_migration_helper_filters_empty_and_whitespace_entries(app, test_user):
    cid = _seed_conversion_with_csv(app, test_user['id'], 'eins,  , zwei,,')
    with app.app_context():
        _migrate_conversion_tags_csv_to_junction(app)
        c = Conversion.query.get(cid)
        names = sorted(t.name for t in c.tag_refs)
        assert names == ['eins', 'zwei']


def test_migration_helper_dedupes_duplicate_csv_entries(app, test_user):
    cid = _seed_conversion_with_csv(app, test_user['id'], 'eins, zwei, eins, drei')
    with app.app_context():
        _migrate_conversion_tags_csv_to_junction(app)
        c = Conversion.query.get(cid)
        names = sorted(t.name for t in c.tag_refs)
        assert names == ['drei', 'eins', 'zwei']


def test_migration_helper_reuses_existing_tag_rows(app, authenticated_client, test_user):
    # First, create a 'shared' tag via the highlight API so a Tag row exists.
    cid_existing = _make_conversion(app, test_user['id'], title='Existing')
    hid = _make_highlight(app, cid_existing)
    h_resp = authenticated_client.post(f'/api/highlights/{hid}/tags', json={'name': 'shared'})
    existing_tag_id = h_resp.get_json()['id']

    # Then seed a separate conversion with a CSV that names the same tag.
    cid_csv = _seed_conversion_with_csv(app, test_user['id'], 'shared, novel')
    with app.app_context():
        _migrate_conversion_tags_csv_to_junction(app)
        c = Conversion.query.get(cid_csv)
        shared = next(t for t in c.tag_refs if t.name == 'shared')
        assert shared.id == existing_tag_id
        # The Tag namespace remains de-duplicated.
        assert Tag.query.filter_by(name='shared').count() == 1


def test_migration_helper_noop_when_no_csv_rows(app, test_user):
    # Plain conversion with empty tags string — nothing to migrate.
    _make_conversion(app, test_user['id'])
    with app.app_context():
        before = conversion_tags.select()
        rows_before = db.session.execute(before).fetchall()
        _migrate_conversion_tags_csv_to_junction(app)
        rows_after = db.session.execute(before).fetchall()
        assert rows_before == rows_after


# --- Library-Search: tag matches via junction (R2-A patch) ---

def test_library_search_matches_conversion_via_tag_junction(app, authenticated_client, test_user):
    # Conversion whose title + content do NOT contain "produkt"; the only
    # match path is the tag attached via the junction.
    cid = _make_conversion(
        app,
        test_user['id'],
        title='Quartals-Roadmap',
        content='Inhalt ohne Begriffe.',
    )
    authenticated_client.post(f'/api/conversions/{cid}/tags', json={'name': 'produkt'})

    resp = authenticated_client.get('/library?search=produkt')
    assert resp.status_code == 200
    assert b'Quartals-Roadmap' in resp.data


def test_library_search_does_not_match_dead_csv_column(app, authenticated_client, test_user):
    # If someone (via SQL or a stale client) writes into the dead CSV column,
    # the search must NOT surface that row by the CSV string alone — only the
    # junction-attached tag-names count after R2-A.
    cid = _make_conversion(
        app,
        test_user['id'],
        title='Stale CSV holder',
        content='Other body.',
        tags='ghost',
    )
    resp = authenticated_client.get('/library?search=ghost')
    assert resp.status_code == 200
    # The title doesn't contain "ghost" and no junction row exists either.
    assert b'Stale CSV holder' not in resp.data
    # Sanity: the conversion *exists* — searching by title still finds it.
    resp_title = authenticated_client.get('/library?search=Stale')
    assert b'Stale CSV holder' in resp_title.data
    # Clean up so the test ID matches expected row state.
    _ = cid


def test_library_search_title_and_content_branches_still_work(app, authenticated_client, test_user):
    _make_conversion(app, test_user['id'], title='Title-Match', content='other body.')
    _make_conversion(app, test_user['id'], title='Different', content='content-needle here.')

    by_title = authenticated_client.get('/library?search=Title-Match')
    assert b'Title-Match' in by_title.data
    by_content = authenticated_client.get('/library?search=content-needle')
    assert b'Different' in by_content.data


# --- Tag.get_or_create unit-ish coverage ---

def test_tag_get_or_create_rejects_blank_and_oversize(app, test_user):
    with app.app_context():
        assert Tag.get_or_create(test_user['id'], '') is None
        assert Tag.get_or_create(test_user['id'], '   ') is None
        assert Tag.get_or_create(test_user['id'], 'x' * 81) is None
        assert Tag.get_or_create(test_user['id'], None) is None


def test_tag_get_or_create_returns_existing_row_on_second_call(app, test_user):
    with app.app_context():
        t1 = Tag.get_or_create(test_user['id'], 'reuse')
        db.session.commit()
        t2 = Tag.get_or_create(test_user['id'], 'REUSE')
        assert t1.id == t2.id


def test_tag_get_or_create_strips_markdown_artefacts(app, test_user):
    # R2-E: LLM-generated newsletter topics drag Markdown leftovers in —
    # normalize_name strips them centrally for every path.
    with app.app_context():
        assert Tag.get_or_create(test_user['id'], '** [anthropic').name == 'anthropic'
        assert Tag.get_or_create(test_user['id'], '  KI-Agenten  ').name == 'ki-agenten'
        assert Tag.get_or_create(test_user['id'], '[nvidia]').name == 'nvidia'
        assert Tag.get_or_create(test_user['id'], '`tooling`').name == 'tooling'
        assert Tag.get_or_create(test_user['id'], 'multi   *word*\ttag').name == 'multi word tag'


def test_tag_get_or_create_artefact_only_string_creates_nothing(app, test_user):
    # A string that is nothing but artefacts normalises to '' -> skip, and
    # crucially no Tag row is left behind.
    with app.app_context():
        before = Tag.query.count()
        assert Tag.get_or_create(test_user['id'], '**') is None
        assert Tag.get_or_create(test_user['id'], '[ ]') is None
        assert Tag.get_or_create(test_user['id'], '` ` *') is None
        assert Tag.query.count() == before


def test_tag_normalized_attach_finds_existing_clean_tag(app, test_user):
    # '** [anthropic' must not create a sibling of an existing 'anthropic'.
    with app.app_context():
        clean = Tag.get_or_create(test_user['id'], 'anthropic')
        db.session.commit()
        dirty = Tag.get_or_create(test_user['id'], '** [anthropic')
        assert dirty.id == clean.id
        assert Tag.query.filter_by(user_id=test_user['id'], name='anthropic').count() == 1


# --- Library Tag-Filter (R2-B): ?tag over the conversion_tags junction ---

def test_library_tag_filter_surfaces_only_matching(app, authenticated_client, test_user):
    cid_ki = _make_conversion(app, test_user['id'], title='KI-Doc', content='body')
    authenticated_client.post(f'/api/conversions/{cid_ki}/tags', json={'name': 'ki'})
    cid_other = _make_conversion(app, test_user['id'], title='Other-Doc', content='body')
    authenticated_client.post(f'/api/conversions/{cid_other}/tags', json={'name': 'produkt'})

    resp = authenticated_client.get('/library?tag=ki')
    assert resp.status_code == 200
    assert b'KI-Doc' in resp.data
    assert b'Other-Doc' not in resp.data


def test_library_tag_filter_empty_shows_all(app, authenticated_client, test_user):
    cid_a = _make_conversion(app, test_user['id'], title='Alpha-Doc')
    authenticated_client.post(f'/api/conversions/{cid_a}/tags', json={'name': 'ki'})
    cid_b = _make_conversion(app, test_user['id'], title='Beta-Doc')
    authenticated_client.post(f'/api/conversions/{cid_b}/tags', json={'name': 'produkt'})

    resp = authenticated_client.get('/library?tag=')
    assert resp.status_code == 200
    assert b'Alpha-Doc' in resp.data
    assert b'Beta-Doc' in resp.data


def test_library_tag_filter_normalizes_incoming_case(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'], title='Case-Doc')
    authenticated_client.post(f'/api/conversions/{cid}/tags', json={'name': 'ki'})
    # ?tag=KI must match the stored lowercase "ki" (route .strip().lower()).
    resp = authenticated_client.get('/library?tag=KI')
    assert resp.status_code == 200
    assert b'Case-Doc' in resp.data


def test_library_tag_filter_combines_with_type(app, authenticated_client, test_user):
    cid_md = _make_conversion(app, test_user['id'], title='MD-Tagged',
                              conversion_type='markdown_input')
    authenticated_client.post(f'/api/conversions/{cid_md}/tags', json={'name': 'shared'})
    cid_doc = _make_conversion(app, test_user['id'], title='Doc-Tagged',
                               conversion_type='document_to_markdown')
    authenticated_client.post(f'/api/conversions/{cid_doc}/tags', json={'name': 'shared'})

    resp = authenticated_client.get('/library?tag=shared&type=markdown_input')
    assert resp.status_code == 200
    assert b'MD-Tagged' in resp.data
    assert b'Doc-Tagged' not in resp.data


def test_library_tag_filter_combines_with_favorites(app, authenticated_client, test_user):
    cid_fav = _make_conversion(app, test_user['id'], title='Fav-Tagged', is_favorite=True)
    authenticated_client.post(f'/api/conversions/{cid_fav}/tags', json={'name': 'shared'})
    cid_plain = _make_conversion(app, test_user['id'], title='Plain-Tagged', is_favorite=False)
    authenticated_client.post(f'/api/conversions/{cid_plain}/tags', json={'name': 'shared'})

    resp = authenticated_client.get('/library?tag=shared&favorites=1')
    assert resp.status_code == 200
    assert b'Fav-Tagged' in resp.data
    assert b'Plain-Tagged' not in resp.data


def test_library_tag_filter_unknown_tag_hits_empty_state(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'], title='Has-A-Tag')
    authenticated_client.post(f'/api/conversions/{cid}/tags', json={'name': 'ki'})
    resp = authenticated_client.get('/library?tag=gibtsnicht')
    assert resp.status_code == 200
    assert b'Has-A-Tag' not in resp.data
    # tag is now in has_active_filter → the existing 0-hit empty-state renders.
    assert 'Keine Treffer mit aktuellen Filtern'.encode() in resp.data


def test_library_tag_filter_preserved_across_pagination(app, authenticated_client, test_user):
    # 21 ki-tagged conversions at the default per_page=20 → 2 pages. A page-2
    # pagination link must keep ?tag=ki so the next page stays filtered.
    for i in range(21):
        cid = _make_conversion(app, test_user['id'], title=f'KI-{i:02d}')
        authenticated_client.post(f'/api/conversions/{cid}/tags', json={'name': 'ki'})
    resp = authenticated_client.get('/library?tag=ki')
    assert resp.status_code == 200
    html = resp.data.decode()
    page2_links = re.findall(r'href="[^"]*page=2[^"]*"', html)
    assert page2_links, 'expected a page-2 pagination link'
    assert any('tag=ki' in link for link in page2_links)


def test_library_tag_filter_chip_row_lists_available_tags(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'], title='Chip-Doc')
    authenticated_client.post(f'/api/conversions/{cid}/tags', json={'name': 'produkt'})
    # available_tags feeds the chip-row even on the unfiltered list.
    resp = authenticated_client.get('/library')
    assert resp.status_code == 200
    assert b'tag=produkt' in resp.data
