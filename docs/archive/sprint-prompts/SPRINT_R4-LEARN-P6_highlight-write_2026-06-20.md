# Sprint R4-LEARN-P6 — Highlight-Write-API (Agent-Tags/Note) (S, 2 Phasen)

> **Executor-Doc.** Phasen nacheinander, nach jeder **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün (Baseline **340**). Du committest jede Phase selbst (eigener Hash + push). Arbeitsverzeichnis `/Users/olivergluth/CODE/CONVERTER`. Working-Practice in `CLAUDE.md` (Sektion „Working Practice").

## Warum

Der Karten-Agent (converter-mcp, Token-Auth) leitet sein Bucket-Tagging heute **pro Lauf neu** ab, weil er Tags/Notiz auf bestehenden **Highlights** nicht zurückschreiben kann. Dieser Sprint gibt ihm genau einen token-authentifizierten Schreibpfad — **exakt analog zur Card-Write-Schicht (R4-LEARN P2)** —, damit das Tagging persistent wird.

**Der Vertrag ist bereits fix**: das converter-mcp-Tool `update_highlight` ist gegen den unten stehenden Pfad gebaut. **Pfad, Auth, Body-Semantik und Status-Codes nicht abweichen.** Kein Phase 0.

## Verifizierte Code-Fakten (vom Master gegroundet — bau darauf, nicht neu recherchieren)

- **Pfad-Kollision ist real**: `app_pkg/highlights.py:74` registriert schon `PATCH /api/highlights/<int:highlight_id>` (Session, note-only). Flask kann Route+Methode nicht doppelt registrieren → der Agent-Write **muss** auf `/api/highlights/<id>/annotate` (Pfad ist frei, grep-bestätigt).
- **Platzierung = `app_pkg/cards.py`** (`register(app)`), **nicht** `highlights.py`: cards.py besitzt schon `_authorize_card_write()`, die `csrf.exempt(...)`-Calls und den Ingest-Auth-Import (`from .ingest import _bearer_token, _resolve_target_user`). View-Funktionsname **`api_annotate_highlight`** (flach, kein Blueprint; kollidiert nicht mit dem bestehenden `api_annotate_card`).
- **`_authorize_card_write()`** (cards.py:113) gibt `(target, None)` bei Erfolg bzw. `(None, (response, status))` zurück — fail-closed **503** ohne `CARD_TOKEN`, constant-time Bearer-Compare → **401** missing/wrong, Ziel-User via `INGEST_USER`/first(). **1:1 wiederverwenden.**
- **`Highlight.to_dict()` enthält bereits die aufgelösten Tags** (models.py:126 `'tags': [t.to_dict() for t in self.tags]`) → **kein `to_dict`-Edit nötig**, Response ist schlicht `jsonify(highlight.to_dict())`. (Die Acceptance-Zeile „Response enthält aufgelöste Tags" ist damit ohne Modell-Touch erfüllt.)
- **`Tag.get_or_create(user_id, name)`** (models.py:159) normalisiert via `normalize_name` (lowercase + trim + Markdown-Strip), dedupt auf `(user_id, name)`, `flush()` beim Insert, gibt `None` bei non-str/blank/oversize → **geteiltes Vokabular**, kein Parallelsystem. Genau die Funktion, die `_replace_card_tags` schon nutzt.
- **Kein Schema-Touch**: `highlight_tags`-Junction (models.py:95), `Highlight.tags` (`secondary=highlight_tags`, `lazy='joined'`, models.py:115), `Highlight.note` (nullable Text, models.py:110) und `Highlight.updated_at` (`onupdate`, models.py:112 → bumpt automatisch) existieren alle.
- **`MAX_NOTE_LEN = 2000`** in highlights.py:18; cards.py hat das identische `MAX_CARD_NOTE_LEN = 2000` (cards.py:53). Note-Längen-Cap = 2000.

## ⚠️ Der eine Stolperstein — Ownership = 404, NICHT 400

`_validate_highlight_ownership(highlight_id, user_id)` (cards.py:157) existiert, **darf hier aber NICHT wiederverwendet werden**: es gibt einen Error-**String** zurück, den `create_card`/`patch_card` auf **400** mappen (dort ist `highlight_id` ein *Body*-Feld = „ungültige Eingabe"). Hier ist die `highlight_id` ein **Pfad-Parameter** = die adressierte Ressource → ein fehlendes/fremdes Highlight ist **404 „Nicht gefunden"** (Existenz fremder Rows nicht leaken).

→ **Ownership wie `api_patch_highlight` (highlights.py:77-79) machen**, nur mit dem Token-Target statt der Session:
```python
hl = Highlight.query.filter_by(id=highlight_id).first()   # oder .get(highlight_id)
if hl is None or hl.conversion.user_id != target.id:
    return jsonify({'error': 'Nicht gefunden.'}), 404
```

## Phase 1 — `PATCH /api/highlights/<id>/annotate` (Token) + Helper + Tests

Dateien: `app_pkg/cards.py`, `tests/test_cards.py`.

### 1a — Helper `_replace_highlight_tags` (Modul-Ebene, neben `_replace_card_tags`)

1:1-Analog zu `_replace_card_tags` (cards.py:173), nur `highlight.tags` statt `card.tags`:
```python
def _replace_highlight_tags(highlight, names, user_id):
    """Replace a highlight's tags with the normalised get_or_create set
    (shared vocabulary — identische Tag-Rows wie Card-/UI-Tags)."""
    highlight.tags = []
    if not isinstance(names, list):
        return
    for name in names:
        tag = Tag.get_or_create(user_id, name)
        if tag is not None and tag not in highlight.tags:
            highlight.tags.append(tag)
```
**Anmerkung**: das `add-before-tags`-Caveat aus dem `_replace_card_tags`-Call-Site (cards.py:276-279, „Card muss erst in der Session sein, sonst Backref-Warnung") gilt hier **nicht** — das Highlight ist bereits persistiert (aus der DB geladen), also schon in der Session.

### 1b — View `api_annotate_highlight` (in `register(app)`)

- Route `@app.route('/api/highlights/<int:highlight_id>/annotate', methods=['PATCH'])`. **Kein** `@login_required` (Token-Pfad).
- **Auth**: `target, err = _authorize_card_write(); if err: return err`.
- **Ownership-404** (siehe Stolperstein oben) — gegen `target.id`.
- **Body**: `data = request.get_json(silent=True)`; non-dict → 400 `'Ungültiger Request-Body. JSON-Objekt erwartet.'`.
- **Mind. ein Key Pflicht**: `if 'tags' not in data and 'note' not in data: return … 'Nichts zu ändern (tags oder note erwartet).', 400` (spiegelt `api_annotate_card`:451-452, nur tags/note-Wortlaut).
- **`tags`** (wenn präsent): muss Liste sein, sonst 400 `"Feld 'tags' muss eine Liste sein."` (wie `api_patch_card`:321-322). Dann `_replace_highlight_tags(hl, data['tags'], target.id)`. `[]` = alle Tags weg.
- **`note`** (wenn präsent): str-oder-null, sonst 400 `'Notiz muss Text oder null sein.'`; `len > MAX_CARD_NOTE_LEN` → 400 `f'Notiz zu lang (max {MAX_CARD_NOTE_LEN} Zeichen).'`; `''` → `hl.note = None` (Lösch-Intent), sonst `hl.note = note`. (Spiegelt `api_annotate_card`:458-465 / `api_patch_highlight`:88-98. `MAX_CARD_NOTE_LEN` ist in-module = 2000; alternativ `from .highlights import MAX_NOTE_LEN` — beide 2000, deine Wahl, in-module ist die kleinere Änderung.)
- **`exact`/`prefix`/`suffix` bewusst ignorieren** — nicht lesen, nicht durchreichen (Agent annotiert nur, ändert keine Anker).
- `db.session.commit()`; `return jsonify(hl.to_dict())` (Tags sind drin, 200).

### 1c — CSRF-exempt registrieren

Neben die zwei bestehenden (cards.py:488-489):
```python
app.extensions['csrf'].exempt(api_annotate_highlight)
```

### 1d — Tests (`tests/test_cards.py`, neben den Card-Write-Tests)

Nutze die vorhandene Scaffolding: `_card_auth(token)` (Bearer-Header), `CARD_TOKEN = 'r4-test-card-token-9b2e'`, `monkeypatch.setenv('CARD_TOKEN', …)`, und die Highlight-Factory `_make_highlight(app, conversion_id, …)` + `_make_conversion` + `_make_other_user`. Mirror `test_card_create_*` / `test_card_patch_*`.

Pflicht-Fälle:
- **Token fehlt → 503** (CARD_TOKEN unset; wie `test_card_create_fail_closed_without_token`).
- **Falscher Token → 401** (`_card_auth('the-wrong-token')`).
- **Fremdes Highlight (anderer User) → 404**; **nicht-existente id → 404** (kein Existenz-Leak).
- **tags setzen / ersetzen / leeren (`[]`)** — nach jedem PATCH die persistierten Tags prüfen.
- **note setzen / leeren (`'' → NULL`)**; non-str-non-null → 400.
- **Normalisierung**: `tags: ["KI", "ki", " KI "]` → genau **eine** `Tag`-Row (`name='ki'`), und am Highlight genau ein Tag.
- **Geteiltes Vokabular**: ein zuvor (auf einer Card **oder** via `Tag.get_or_create`) angelegter Tag wird beim Setzen am Highlight **als dieselbe Row** (gleiche `id`) wiederverwendet — kein Parallelsystem.
- **Response enthält die aufgelösten Tags** (`resp.json['tags']` mit `id`+`name`).
- **Anker unveränderbar**: Body mit `exact`/`prefix`/`suffix` (+ einem Tag) → 200, aber `hl.exact`/`prefix`/`suffix` danach **unverändert**.
- **Leerer Body** (weder tags noch note) → **400**.
- **CSRF-exempt unter erzwungenem CSRF**: `monkeypatch.setitem(app.config, 'WTF_CSRF_ENABLED', True)` + Bearer-only PATCH → **nicht** 400/403 (mirror `test_card_create_csrf_exempt_under_enforced_csrf`). *(Caveat: conftest hat `WTF_CSRF_ENABLED=False` global — für diesen Beweis wieder anschalten; Memory `reference_token_auth_ingest_endpoint`.)*
- **`test_only_card_write_views_are_csrf_exempt` erweitern**: assert `'app_pkg.cards.api_annotate_highlight' in csrf._exempt_views` (der neue Write-View gehört in die Exempt-Menge; die bestehenden „NOT exempt"-Asserts für list/review_state/delete bleiben).

`pytest tests/` grün, ≥ Baseline **340** (Erwartung ~+12).

**Stop + Bericht** (Endpoint-Verhalten + Test-Delta + der 404-vs-400-Beleg).

## Phase 2 — Doc-Wrap (keine UI, kein Browser-Smoke)

Dieser Endpoint hat **keine UI** — Verifikation ist `pytest`. Der echte end-to-end-Beweis (`update_highlight` über den converter-mcp gegen Live-Prod) ist **Koordinator-Schritt nach dem Deploy**, nicht deiner.

1. **`STATUS.md`** + **`BACKLOG.md`**: R4-LEARN-P6 ☑ done mit Hashes (Muster wie der R4-LEARN-FIX-Eintrag). **Bullet-Guard** vor dem Commit (Memory `reference_markdown_bullet_delete_newline`): `grep -nE '(- \*\*.*){2,}' BACKLOG.md`.
2. **`docs/card_api_contract.md`** — neuen Eintrag in **„Writes (Bearer `CARD_TOKEN`)"** (das ist ein converter-mcp-Wrap-Target, Tool `update_highlight`): `PATCH /api/highlights/<id>/annotate` — Body `{tags?: [str], note?: str|null}` (mind. einer Pflicht, sonst 400), Voll-Ersetzung der Tags / `[]` = leer / `note ''` → NULL, **Ownership fremd|fehlend → 404**, `exact`/`prefix`/`suffix` werden ignoriert (Anker unveränderbar), Response = **volles Highlight inkl. aufgelöster Tags**. Auth-Fehler 503/401 wie die Card-Writes.
3. **`docs/card_agent_guide.md`** — die Grenze-Zeile „Kein Highlight-Schreiben/-Löschen über diesen Layer" (Z. 63) ist jetzt **teilweise überholt**:
   - Tool-Tabelle: Zeile `update_highlight` ergänzen („Tags/Notiz auf einem **bestehenden** Highlight setzen/ersetzen/leeren — für persistentes Bucket-Tagging").
   - Kurzer Absatz „Highlights annotieren": der Agent kann Tags + Notiz auf Highlights zurückschreiben (geteiltes Vokabular, Voll-Ersetzung der Tags). Use-Case: Bucket-Tagging persistent machen statt pro Lauf neu ableiten.
   - Grenze umformulieren auf das, was **weiterhin** verboten bleibt: **kein** Highlight-**Löschen**, **kein** Ändern von `exact`/`prefix`/`suffix` (Anker/Marker), **kein** Anlegen neuer Highlights über den Agent.
4. **Memory**: nur falls etwas Nicht-Offensichtliches auftaucht. Die 400-(Body-Ref)-vs-404-(Pfad-Ressource)-Achse ist erwähnenswert, aber grenzwertig dünn — deine Einschätzung; ansonsten ist das eine getreue Re-Anwendung von `reference_token_auth_ingest_endpoint`.
5. Finaler `pytest tests/` grün.

**Stop + Schluss-Bericht** — inkl. der Deploy-Kette (Olis/Koordinator-Schritte, nicht deine):
> Mac mergen/pushen → Mintbox `git pull` + `docker compose up -d --build` für CONVERTER (**keine Migration** — `highlight_tags`/`note` existieren) → dann converter-mcp deployen (`update_highlight` liegt dort schon → `up -d --build`). `CARD_TOKEN` ist in beiden `.env` gesetzt — **kein neuer Token**.

## Bewusst NICHT (Scope-Grenze)

- **Kein** Highlight-Delete über Token/MCP (User-only, UI).
- **Keine** Änderung an `exact`/`prefix`/`suffix` (Marker/Anker) — diese Keys im Body ignorieren, nicht durchreichen.
- **Kein** agent-seitiges Anlegen neuer Highlights.
- **Kein** Schema-Touch.
- **Keine** Änderung am bestehenden Session-`PATCH /api/highlights/<id>` (highlights.py) — der bleibt der UI-Notiz-Pfad.

## Akzeptanz

- [ ] `update_highlight` setzt/ersetzt/leert Tags auf bestehendem Highlight, normalisiert, on-demand, geteiltes Vokabular
- [ ] gesetzte Tags erscheinen danach in `/api/highlights/recent` (tags-Feld)
- [ ] note setzen/leeren mit PATCH-Semantik (`'' → NULL`)
- [ ] Ownership erzwungen, fremde/fehlende Highlights **404** (nicht 400, nicht 403)
- [ ] Highlight-Delete über diesen Pfad NICHT möglich; Marker/Anker (`exact`/`prefix`/`suffix`) unveränderbar
- [ ] Pfad ist exakt `PATCH /api/highlights/<id>/annotate` (das converter-mcp-Tool ist genau dagegen gebaut)
- [ ] `pytest tests/` grün ≥ 340
