# Sprint MOBILE-AUTH — Per-User-Bearer-Token für die native iOS-App (M, 4 Phasen)

> **Executor-Doc.** Nach jeder Phase **Stop + Bericht**, auf Sign-off warten. **Phase 0 ist ein Mechanik-Workshop** (zwei Wahlen, unten) — erst nach Sign-off Code. Pre-Flight: `pytest tests/` grün (Baseline aus STATUS, aktuell **633**). Arbeitsverzeichnis Mac `/Users/olivergluth/CODE/CONVERTER` = Source-of-Truth. **Sicherheits-sensibel** (neue **öffentliche Credential-Fläche** + CSRF-Bypass-Pfad) — fail-closed, constant-time, generische Fehler, Token nie geloggt, gehashte Speicherung. Deploy Mintbox `git pull` + `up -d --build`.
>
> **Warum (Kontext):** Oli baut eine native iOS-App (SwiftUI) als CONVERTER-Client (Start-Brief separat). CONVERTER hat heute **keine mobil-taugliche Auth**: alle `/api/...`-Reads hängen an der **Flask-Login-Session-Cookie** + globalem **CSRF**, es gibt **keinen JSON-Login**, und die drei Bearer-Tokens (`INGEST/CARD/NARRATION_TOKEN`) sind **write-only + server-fest single-user** (tragen keine Identität). Ziel: die App authentifiziert sich einmal per Username/Passwort über JSON, bekommt einen **per-User-Bearer-Token**, und ruft damit **alle** APIs (Reads **und** Writes) ohne Cookie/CSRF-Tanz.

## Master-gegroundete Fakten (Ist-Zustand — darauf aufsetzen)

- **Login heute = HTML-Formular** `POST /login` ([app_pkg/auth.py:16-34](app_pkg/auth.py)): `request.form` username/password → `user.check_password` → `login_user(user, remember=True)`. Kein JSON-Pfad.
- **`CSRFProtect(app)` global** ([app_pkg/__init__.py:55](app_pkg/__init__.py)); jeder Session-`POST/PUT/PATCH/DELETE` braucht `X-CSRFToken`. JSON-aware CSRF-Error für `/api/*` ([__init__.py:161-164](app_pkg/__init__.py)). `GET /api/csrf-token` existiert, ist aber selbst login-gated ([__init__.py:181-185](app_pkg/__init__.py)).
- **`login_manager`** ist konfiguriert (Flask-Login, `user_loader` vorhanden). Cookie-Config `SESSION_COOKIE_HTTPONLY/SAMESITE=Lax`, **`SESSION_COOKIE_SECURE` nicht gesetzt** ([__init__.py:46-49](app_pkg/__init__.py)).
- **Bearer-Muster als Vorlage:** `app_pkg/ingest.py` — `_bearer_token()` (Header `Authorization: Bearer …` parsen), `hmac.compare_digest`, fail-closed 503 ohne Konfig, 401 sonst, `@csrf.exempt` auf der View. **Spiegeln, nicht neu erfinden.**
- **`models.py` User** hat `check_password`. Multi-User-Schema (Rows tragen `user_id`), praktisch aber single-user (nur Oli).
- **Schema-Touch = neue Tabelle** → via CONVERTERs Boot-Auto-Migration: neue Tabellen entstehen durch `db.create_all` (kein `ALTER`, kein `_run_pending_migrations`-Eintrag nötig). Memory [[reference_inline_sqlite_migration]]. **Prod-DB vor dem ersten Boot sichern** (Deploy-Notiz).
- **Distinktion, die ins CLAUDE.md muss:** dies ist der **erste identitäts-tragende** Token (per-User, `user_id`), im Gegensatz zu den drei bestehenden write-only-single-user-Tokens. Additiv — die drei bleiben unberührt.

## Phase 0 — Mechanik-Workshop (kein Code, zwei Wahlen, Empfehlung mitliefern)

1. **Token-Modell** — Empfehlung **(a) gehashte DB-Tabelle**: neues `ApiToken`-Model (`id, user_id FK, token_hash, label, created_at, last_used_at, expires_at NULLABLE`). Token = `secrets.token_urlsafe(32)`, gespeichert wird **nur `sha256`** (Klartext genau einmal zurückgegeben), Lookup per Hash, `last_used_at` bei jedem Treffer aktualisiert, Widerruf = Row löschen. Revozierbar + auditierbar + DB-Leak exponiert keine Live-Tokens. Alternative (b) stateless `itsdangerous` (kein Schema, aber Widerruf nur über User-`token_version`-Bump) — nur falls ihr Schema-frei wollt. **Master-Empfehlung: (a).**
2. **CSRF-Bypass für Bearer-Writes** — der Kern-Mechanik-Punkt. Reads sind unkritisch (GET → kein CSRF; Identität liefert der `request_loader`, s. P1). Nur **Writes** brauchen die Ausnahme, weil der Client **keine Cookie** schickt und CSRF sonst 400t. Empfehlung: **CSRF-Inversion** — `WTF_CSRF_CHECK_DEFAULT = False` + ein `@app.before_request`, das `csrf.protect()` **explizit** für **Cookie-/Session-Mutationen** aufruft und für **Bearer-Requests überspringt**. So bleibt die Web-UI-CSRF-Haltung **byte-identisch** (jede Session-Mutation wird weiter protected), Bearer-Requests sind sauber ausgenommen. Alternative: per-`/api/*`-Exempt-Liste (invasiver, fehleranfällig). **Master-Empfehlung: Inversion**, aber ihr bestätigt sie in P0 (das ist die riskanteste Zeile des Sprints — Web-UI darf keine CSRF-Lücke bekommen).

**Stop + Bericht: gewähltes Token-Modell + CSRF-Mechanik, kurze Begründung. Sign-off vor P1.**

## Phase 1 — Token-Ausgabe + JSON-Login + Bearer-Identität (Reads laufen)

1. **`ApiToken`-Model** (falls (a)) in `models.py`, + `to_dict()` ohne den Hash. Auto-Create beim Boot verifizieren.
2. **Issue/Validate-Helfer** (pures Modul oder in einem neuen `app_pkg/mobile_auth.py`): `issue_token(user, label) -> plaintext`, `resolve_token(plaintext) -> User|None` (sha256 → Lookup → `last_used_at` bump → Ablauf prüfen). `hmac`/constant-time wo sinnvoll; **Token nie loggen**.
3. **`POST /api/auth/login`** (public, **CSRF-exempt**, JSON): Body `{username, password}` → `User` lookup → `check_password` → bei Erfolg `issue_token` → `200 {token, user:{id,username}}`. Bei Fehler **generisches `401`** (keine User-Enumeration, kein „user not found" vs „wrong password"-Leak), constant-time-Pfad. **Kein** Token im Log/Response-Log.
4. **Bearer-Identität via `login_manager.request_loader`** ([app_pkg/__init__.py](app_pkg/__init__.py)): Callback liest `Authorization: Bearer …`, `resolve_token`, gibt den User zurück → **alle `@login_required`-Views akzeptieren den Token ohne Per-View-Änderung**, `current_user` ist gesetzt. Kein Session-Cookie nötig. (Reads funktionieren damit sofort; Writes noch von CSRF geblockt — das macht P2.)
5. **Tests:** Login-Erfolg → Token + gehasht gespeichert (Klartext nicht in DB); falsches PW → generisches 401; unbekannter User → **dasselbe** 401; `GET /api/conversions` mit `Authorization: Bearer` **ohne** Cookie → 200 + korrekte User-Scoping; abgelaufener/widerrufener/kaputter Token → 401/Anonymous; **Session-Login-Pfad unverändert** (bestehende Auth-Tests grün); die drei Alt-Tokens unberührt.
6. `pytest` grün (633 + neue).

**Stop + Bericht.**

## Phase 2 — CSRF-Ausnahme für Bearer-Writes + `me`/`logout`

1. **CSRF-Mechanik aus P0 umsetzen.** Bearer-authentifizierte Writes (`POST/PUT/PATCH/DELETE` mit gültigem Token) laufen **ohne** CSRF; **Session-Writes bleiben CSRF-pflichtig, identisch zu heute**. Die drei bestehenden `@csrf.exempt`-Token-Writes bleiben exempt.
2. **`GET /api/auth/me`** (Bearer-authed): `{id, username}` — die App validiert damit einen gespeicherten Token beim Start.
3. **`POST /api/auth/logout`** (Bearer-authed): widerruft den **präsentierten** Token (Row löschen). Idempotent.
4. **Tests (die sicherheits-tragenden):** `PATCH /api/conversions/<id>/progress` mit Bearer **ohne** `X-CSRFToken` → **200** (Bypass greift); **dieselbe** Session-Mutation **ohne** CSRF-Token → weiterhin **400** (Web-UI-Schutz intakt — der Beweis, dass die Inversion nichts aufgerissen hat); `me` mit/ohne Token → 200/401; `logout` → Token danach ungültig (nächster Call 401). `POST /api/conversions` mit Bearer legt eine Conversion unter dem **richtigen** User an.
5. `pytest` grün.

**Stop + Bericht.**

## Phase 3 — Live-Verify + Wrap

1. **Live-Smoke (Mintbox nach `up -d --build`), mit `curl`:** `POST /api/auth/login` → Token; `GET /api/conversions` mit `Authorization: Bearer` → 200 JSON; ein **Write** mit Bearer ohne CSRF (z.B. `progress`) → 200; falscher Login → 401; Web-UI im Browser weiter normal einloggbar + eine Session-Mutation klappt (CSRF unverändert). **Kein Token in Server-Logs** gegenprüfen.
2. **Wrap:** BACKLOG (MOBILE-AUTH ☑ + Endpunkte); STATUS (pytest-Zahl); **CLAUDE.md** (neuer Auth-Abschnitt: `POST /api/auth/login` → per-User-Bearer, `request_loader` akzeptiert Bearer auf allen `@login_required`-APIs, CSRF-Inversion für Bearer-Writes; **erster identitäts-tragender Token**, distinct von den write-only-Single-User-Tokens); **Memory** (`reference_per_user_bearer_request_loader` — wiederverwendbar: „per-User-Token neben Session via `login_manager.request_loader` + CSRF-Inversion; gehashte Token-Tabelle; generisches 401 gegen Enumeration"). **Bullet-Guard** `grep -nE '(- \*\*.*){2,}' BACKLOG.md` vor Commit.
3. **Deploy-Notiz:** neue Tabelle `api_token` → **Prod-DB vor Boot sichern**, Auto-Create beim ersten Boot; **kein** Dep erwartet (falls (b) `itsdangerous` — schon via Flask da). Optionaler `.env`-Schalter (`MOBILE_AUTH_ENABLED`, Default an) nur falls ihr einen Kill-Switch wollt — sonst weglassen.
4. **Kontrakt-Doc** `docs/mobile_auth_contract.md` (knapp): die drei/vier Endpoints + Header-Format, für den iOS-Agenten als Phase-2-Referenz.

**Stop + Schluss-Bericht.**

## Bewusst NICHT

- **Kein** Anfassen der drei bestehenden Tokens (`INGEST/CARD/NARRATION_TOKEN`) — additiv bleiben.
- **Keine** CSRF-Schwächung für die Web-UI — Session-Writes müssen byte-identisch geschützt bleiben (der P2-Test beweist es).
- **Kein** OAuth/JWT/Refresh-Token-Overkill — ein opaker, revozierbarer per-User-Token reicht für Single-User + native App.
- **Kein** Rate-Limiting-Framework (CONVERTER hat keins); der Login ist LAN-/login-kontextuell low-risk — als dokumentierte Erwägung notieren, nicht bauen.
- **Kein** Token im Klartext in DB/Log/Response-Log.

## Akzeptanz

- [ ] **P0**: Token-Modell + CSRF-Mechanik gewählt + begründet; Sign-off.
- [ ] **P1**: `POST /api/auth/login` (generisches 401), gehashte Token, `request_loader` → Bearer-Reads laufen; Session-Login unverändert; pytest grün.
- [ ] **P2**: Bearer-Writes ohne CSRF = 200, Session-Writes ohne CSRF weiterhin 400 (Web-UI-Beweis); `me`/`logout`; pytest grün.
- [ ] **P3**: Live-Smoke Login→Bearer-Read+Write auf Mintbox; kein Token-Leak in Logs; Docs/CLAUDE/Memory/Kontrakt-Doc + Bullet-Guard; Deploy-Notiz (DB-Backup + neue Tabelle).
