# Mobile-Auth-Kontrakt — per-User-Bearer für die native iOS-App

Stand 2026-07-13 (Sprint MOBILE-AUTH). Alle Pfade relativ zur Basis-URL des
CONVERTER-Stacks. Request/Response `application/json`, sofern nicht anders
notiert.

## 1. Login — Token holen

    POST /api/auth/login
    {"username": "…", "password": "…", "label": "iphone-oli"}   # label optional

- **200** → `{"token": "<opaque>", "user": {"id": 1, "username": "…"}}`
- **401** → `{"error": "Nicht autorisiert."}` — **generisch für jeden
  Fehlerfall** (falsches Passwort, unbekannter User, kaputter Body). Keine
  Unterscheidung, absichtlich (Anti-Enumeration).
- Kein Cookie, kein CSRF-Token nötig.

Der Token kommt **genau einmal** im Klartext (der Server speichert nur den
sha256-Hash) → sicher persistieren (iOS Keychain). Kein Ablauf per Default;
Widerruf nur über Logout (oder serverseitiges Row-Delete).

## 2. Authentifizierte Calls — Header-Format

    Authorization: Bearer <token>

Gilt für **alle** `/api/...`-Endpoints, Reads **und** Writes
(GET/POST/PATCH/DELETE), ohne CSRF-Token und ohne Cookie. Ungültiger,
abgelaufener oder widerrufener Token → generisches
`401 {"error": "Nicht autorisiert."}`.

⚠️ Den Header **immer** mitschicken: Requests **ohne** Authorization-Header
auf Endpoints außerhalb `/api/auth/*` antworten mit dem Web-Redirect (302
→ `/login`), nicht mit 401 — das ist die unveränderte Web-UI-Semantik.

## 3. Token validieren (App-Start)

    GET /api/auth/me    →  200 {"id": 1, "username": "…"}  |  401

## 4. Logout — Token widerrufen

    POST /api/auth/logout    →  200 {"ok": true, "revoked": true}

Widerruft den **präsentierten** Token (Row-Delete). Danach ist er tot — jeder
weitere Call damit, auch ein zweiter Logout, ist ein 401. Andere Tokens
desselben Users (z.B. ein zweites Gerät) bleiben gültig.

## Nützliche Endpoints fürs App-MVP (Konsum + Erfassen)

- `GET /api/conversions` — Library-Liste, owner-scoped, slim Summaries
  (`?type=`, `?status=`, `?exclude_status=`, `?limit=` ≤500, `?offset=`)
- `GET /api/conversions/<id>` — Einzeldokument inkl. Content
- `POST /api/conversions` — Dokument anlegen (`content` required,
  `conversion_type` aus der Whitelist, z.B. `markdown_input`) → 201
- `PATCH /api/conversions/<id>/progress` — Lese-Fortschritt
  `{"percent": 42}` (forward-clamp) bzw. `{"reset": true}`
- `GET /api/narrations/<id>` / `GET /api/narrations/<id>/audio` —
  Narration-Status (Polling) + WAV-Stream

Vollständige Shapes: `app_pkg/library.py`, `app_pkg/narration.py`.
