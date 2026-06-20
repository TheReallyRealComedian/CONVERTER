# Sprint R4-LEARN-FIX — Karten löschen (XS–S, 2 Phasen)

> **Executor-Doc.** Phasen nacheinander, nach jeder **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün (Baseline **336**). Du committest jede Phase selbst (eigener Hash + push). Arbeitsverzeichnis `/Users/olivergluth/CODE/CONVERTER`.

## Warum

R4-LEARN hat „Agent erzeugt / User bewertet" gebaut, aber **„User löscht" vergessen** — es gibt keinen Card-DELETE-Endpoint und keine Lösch-Aktion in der „Lernen"-UI (im Integrations-Smoke aufgefallen, eine echte Smoke-Karte id=1 hängt in der Review-Queue). **Hand-SQL ist verboten**: ohne FK-Pragma würde ein rohes `DELETE FROM card` die `review`- + `card_tags`-Zeilen verwaisen lassen (genau die `before_delete`-Falle) — Löschen muss über den ORM laufen.

## Phase 1 — `DELETE /api/cards/<id>` (Session) + Tests

Datei: `app_pkg/cards.py`, `tests/test_cards.py`.
1. **`DELETE /api/cards/<int:card_id>`** — `@login_required` (der User löscht **seine** Karte), owner-scoped **über `Card.user_id`** (wie `api_annotate_card`: `Card.query.filter_by(id=card_id, user_id=current_user.id).first_or_404()` → fremd/fehlend = 404, leakt keine Existenz). `db.session.delete(card)` + commit → **ORM-Cascade** nimmt die `review`-Zeile (`Card.review`-Relationship `cascade='all, delete-orphan'`, Phase 1) **und** die `card_tags`-Junction (secondary-Relationship) korrekt mit. Response `{'success': True}` (200). **Nicht** token-/CSRF-exempt — Session-Write, läuft über den globalen CSRF-fetch-Wrapper.
2. **Tests**: eigene Karte löschen → 200, **Karte + Review + card_tags-Rows weg** (DB-Probe: kein verwaister Review-/Junction-Eintrag); fremde Karte (anderer User) → 404; nicht-existente id → 404; unauth → Login-Redirect/401 wie die anderen Session-Endpoints. `pytest` grün ≥ Baseline.

**Stop + Bericht** (inkl. dem Cascade-Beleg: Review + card_tags wirklich weg, kein Orphan).

## Phase 2 — Lösch-Affordanz in der „Lernen"-UI + Live-Smoke + Wrap

Dateien: `templates/review.html`, `static/js/review.js`, ggf. CSS.
1. **„Löschen"-Button** im Karten-Footer (neben `review-deepen-btn`/`review-note-toggle`). **Bestätigung Pflicht** (Löschen ist irreversibel) — natives `confirm()` oder ein Zwei-Klick-Inline („wirklich löschen?"). DS-konform (dezent, danger-Ton; das gelandete neomorphe Muster, token-driven, kein Hardcode).
2. **`deleteCard()`** in `review.js` (spiegelt `deepen()`/`annotate`-Pfad): `DELETE /api/cards/<id>` über den globalen `base.html`-fetch-Wrapper (X-CSRFToken automatisch — DELETE ist state-changing, braucht CSRF), bei Erfolg **zur nächsten fälligen Karte** vorrücken (wie nach einem Rating; Queue-Counter aktualisieren); leere Queue → Done/Empty-State. Fehler → `showToast`.
3. **Live-Smoke** (lokale Docker-Instanz, echte Klicks, dark+light, 0 Console-Errors): eine Karte löschen → Bestätigung → verschwindet aus der Queue, nächste erscheint (oder Empty-State); kein 400 (CSRF ok). **Die echte Smoke-Karte auf der Mintbox (id=1) räumst NICHT du weg** — das ist Olis Real-Welt-Schritt nach dem Deploy (Hinweis in den Bericht).
4. **Wrap**: `STATUS.md` + `BACKLOG.md` (R4-LEARN-FIX ☑ done mit Hashes); `docs/card_api_contract.md` ergänzen — **`DELETE /api/cards/<id>` in die „NICHT wrappen (UI-only, Session)"-Liste** zu `review`/`annotate` (der User löscht in der Oberfläche; der Agent erzeugt/patcht, löscht nicht — falls je Agent-Delete gewünscht, eigener Token-DELETE-Follow-up). Kurzer `reader_architecture.md`-Halbsatz (Card-Lifecycle jetzt komplett: create/patch=Agent, review/annotate/delete=User). **Bullet-Guard**, finaler `pytest`.

**Stop + Schluss-Bericht** — inkl. Olis Schritten: **Mintbox-Deploy** (`git pull` + `up -d --build`, keine Migration), danach die **Smoke-Karte id=1 im „Lernen"-Tab löschen** (erster echter Anwendungsfall). Hinweis an den Koordinator: `DELETE` ist UI-only, **nicht** in den converter-mcp wrappen.

## Out of scope
- Agent-/Token-Delete (DELETE ist User-Session-only; Agent-Delete wäre ein eigener Token-Endpoint, falls je nötig).
- Karten-Management-Liste/Bulk-Delete (nur die per-Karte-Lösch-Affordanz im Review-Flow).
