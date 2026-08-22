"""KARTEN-ESCAPE (KLEINKRAM, 2026-08-22) — the card write API is byte-faithful.

Five prod cards (ids 3/4/7/8/10, all from the first-ever card batch on
2026-06-22) carried literal ``\\n`` and ``\\"`` sequences with zero real
newlines. Neither write seam transforms text: converter-mcp forwards its tool
arguments as ``httpx … json=payload`` and ``POST/PATCH /api/cards`` store what
``request.get_json()`` yields. A literal backslash-n in the column therefore
means the TOOL ARGUMENT already contained it (double-escaped at the caller),
and the repair belongs to the data, not to the API or the renderer.

This sentinel pins that property in both directions so nobody "helpfully"
unescapes on write or read later (which would also eat legitimate backslashes,
e.g. a card about ``\\n`` in C).
"""
import json

CARDS_URL = '/api/cards'
CARD_TOKEN = 'kleinkram-test-card-token'


def _auth():
    return {'Authorization': f'Bearer {CARD_TOKEN}'}


def test_card_api_stores_text_exactly_as_decoded_from_the_body(client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)

    real = "Zeile eins\n- Punkt \"zitiert\""          # proper JSON escaping on the wire
    escaped = "Zeile eins\\n- Punkt \\\"zitiert\\\""  # what a double-escaped argument yields

    resp = client.post(CARDS_URL, headers=_auth(), data=json.dumps({
        'type': 'generative', 'prompt': 'Frage?', 'back': real,
    }), content_type='application/json')
    assert resp.status_code == 201
    stored_real = resp.get_json()['back']
    assert stored_real == real
    assert stored_real.count('\n') == 1 and '\\' not in stored_real

    resp = client.post(CARDS_URL, headers=_auth(), data=json.dumps({
        'type': 'generative', 'prompt': 'Frage?', 'back': escaped,
    }), content_type='application/json')
    assert resp.status_code == 201
    stored_escaped = resp.get_json()['back']
    assert stored_escaped == escaped                  # no unescaping on write …
    assert '\n' not in stored_escaped and '\\n' in stored_escaped

    card_id = resp.get_json()['id']
    resp = client.patch(f'{CARDS_URL}/{card_id}', headers=_auth(), data=json.dumps({
        'back': real,
    }), content_type='application/json')
    assert resp.status_code == 200
    assert resp.get_json()['back'] == real            # … and PATCH is the repair path
