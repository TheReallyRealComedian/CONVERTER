# Sprint NL1 — AI-Newsletter-Ingestion (CONVERTER-Seite)

**Datum**: 2026-06-01

**Ziel**: CONVERTER kann AI-Newsletter aus der externen `email-automation`-App aufnehmen. Zwei Teile: (1) neuer `conversion_type` **`ai_newsletter`** (UI-Label „AI-Newsletter") für Anzeige + R2-B-Filter, (2) ein **generischer, token-authentifizierter Server-zu-Server-Ingestion-Endpoint**, gegen den `email-automation` später (Sprint NL2) pusht. NL1 **definiert den Kontrakt** — NL2 implementiert ihn.

**Vorbedingung**: HEAD `354eab3`, lokal+remote synchron. Pytest **151/151 grün**. Mac-Dev-Stack mit dem HYG3-Live-Mount (`static/`+`templates/` gebunden → Template-Smoke via `docker compose restart markdown-converter`, kein Rebuild). Integrations-Richtung = **Push** (Master-Workshop 2026-06-01): `email-automation` POSTet an CONVERTER; der Notion-Write der anderen App bleibt unangetastet.

**Out-of-scope**:
- **NL2 — die `email-automation`-Push-Seite** (anderes Repo auf Mintbox, aktuell `No route to host`). Kommt als eigener Sprint gegen den hier definierten Kontrakt.
- **Pull-aus-Notion** — verworfen zugunsten Push.
- **R2-C Lifecycle** und alles andere im BACKLOG.

---

## Workshop-Entscheidungen (Master, 2026-06-01) — nicht neu diskutieren

| # | Frage | Entscheidung |
|---|-------|--------------|
| 1 | Integrations-Richtung | **Push**: `email-automation` → CONVERTER-Ingestion-Endpoint |
| 2 | Endpoint-Scope | **Generisch**: `POST /api/ingest/conversion`, akzeptiert jeden Typ aus `ALLOWED_CONVERSION_TYPES` (Token = Trust-Boundary) |
| 3 | UI-Label | **„AI-Newsletter"** (interner Typ-String `ai_newsletter`, snake_case-konsistent) |
| 4 | Auth | shared-secret **Bearer-Token** `INGEST_TOKEN` (Env), constant-time-compare, **fail-closed** wenn unset |
| 5 | User-Auflösung | `INGEST_USER` (Username-Env) → User; sonst der einzige `User.query.first()`; kein User → 503 |
| 6 | Dedup | idempotent über `source_id`, abgelegt in `metadata_json` — **kein Schema-Touch** |
| 7 | Mappings | `topics[]` → Conversion-Tags (R2-A-Junction via `Tag.get_or_create`); `report_date` → `created_at` |

**Kontrakt (Body-Schema), den NL2 erfüllen muss:**
```
POST /api/ingest/conversion
Authorization: Bearer <INGEST_TOKEN>
Content-Type: application/json
{
  "conversion_type": "ai_newsletter",   // muss in ALLOWED_CONVERSION_TYPES sein
  "title":           "2026-05-30 - AI Newsletter Analyse",
  "content":         "# … markdown …",  // Pflicht; das Original-Markdown von email-automation
  "topics":          ["KI-Agenten", "Code-Generierung"],  // optional → Tags
  "report_date":     "2026-05-30",        // optional ISO-Date → created_at
  "source_id":       "<stabile id, z.B. Notion-Page-ID>" // optional → Dedup-Key
}
→ 201 {id, …} bei Neuanlage · 200 {id, …, "deduped": true} bei bekanntem source_id
```

---

## Phase 0 — entfällt

Workshop ist gelaufen (Tabelle oben), Mechanik-Anker sind Master-grounded. **Direkt Phase 1.**

---

## Phase 1 — `ai_newsletter`-Typ (Foundation)

Pre-Flight: `pytest tests/` grün (151).

### Erwartete Files
```
app_pkg/library.py            # EDIT — 'ai_newsletter' in ALLOWED_CONVERSION_TYPES (Z.13)
templates/library.html        # EDIT — Filter-<select>-Option + Card-Typ-Badge-elif
templates/library_detail.html # EDIT — Typ-Badge-elif (Z.14-19)
static/css/style.css          # EDIT — .type-ai_newsletter light (~Z.1544) + dark (~Z.1807)
```

### Schritte
1. `ALLOWED_CONVERSION_TYPES` (`app_pkg/library.py:13`) um `'ai_newsletter'` erweitern.
2. `templates/library.html`: Filter-`<select name="type">` (Z.14-20) um `<option value="ai_newsletter">AI-Newsletter</option>`; Card-Typ-Badge (`{% elif %}`-Kette ~Z.54-60) um `{% elif conv.conversion_type == 'ai_newsletter' %}AI-Newsletter`.
3. `templates/library_detail.html:14-19`: dieselbe `{% elif conversion.conversion_type == 'ai_newsletter' %}AI-Newsletter`-Verzweigung.
4. `static/css/style.css`: `.type-ai_newsletter { … }` (light, ~Z.1544) + `[data-global-theme="dark"] .type-ai_newsletter { … }` (dark, ~Z.1807). **Distinkte Hue** wählen, die noch nicht belegt ist (bestehend: Blau/Grün/Lila/Amber) — z.B. Teal/Cyan oder Rosé, Token-Stil wie die Nachbarn.

### Verify + Commit Phase 1
- `pytest tests/` grün (151 — optional ein Mini-Test, dass `ai_newsletter` jetzt in `ALLOWED_CONVERSION_TYPES` ist).
- **Live-Smoke** über den HYG3-Mount: `docker compose restart markdown-converter` → `/library` zeigt „AI-Newsletter" im Typ-Filter-Dropdown; Badge-Rendering im Dark+Light prüfen (sobald ein `ai_newsletter`-Eintrag existiert — sonst Badge-Smoke auf Phase 2 verschieben, wenn der erste Newsletter via curl drin ist).
- **Commit** (plain-prose): `NL1-type: ai_newsletter conversion_type plus Filter/Badge/CSS`. Push direkt.

**STOP — Bericht an Master. Nicht in Phase 2 bis Sign-off.**

---

## Phase 2 — Generischer Token-Auth-Ingestion-Endpoint

Pre-Flight: `pytest tests/` grün.

### Erwartete Files
```
app_pkg/ingest.py     # NEU — register(app) mit POST /api/ingest/conversion
app.py                # EDIT — ingest-Modul importieren + register(app) (~Z.61-70)
tests/test_ingest.py  # NEU — Auth/Validation/Dedup/Mapping-Tests
```

### Endpoint `POST /api/ingest/conversion` — Anforderungen

**Routing/Modul**: neues Modul `app_pkg/ingest.py` mit `register(app)`-Pattern (kein Blueprint — flache Endpoint-Namen, wie alle Module). In `app.py` neben den anderen `*_module.register(app)`-Calls einhängen (Import oben + Call ~Z.70).

**Sicherheit (kritisch — neue Auth-Oberfläche, erster nicht-Session-Endpoint des Projekts):**
- **Kein `login_required`**, dafür **CSRF-exempt**: `app.extensions['csrf'].exempt(<view>)` (Flask-WTF legt die `CSRFProtect`-Instanz dort ab — kein `__init__.py`-Umbau nötig). **Nur dieser eine Endpoint.**
- **Auth**: `Authorization: Bearer <token>` gegen `INGEST_TOKEN` (Env) mit **`hmac.compare_digest`** (constant-time). Header fehlt/falsch → **401**.
- **Fail-closed**: `INGEST_TOKEN` unset/leer → Endpoint akzeptiert **nichts** (503 „Ingestion nicht konfiguriert."). Nie offen ohne Token.
- **Token nie loggen.** Auth-Fehlschläge loggen (für Probing-Sichtbarkeit), ohne Secret/Token-Wert.
- LAN-only-Kontext bleibt (App ist ohnehin LAN-only + sonst login-geschützt).

**User-Auflösung** (keine Session): Ziel-User = `User.query.filter_by(username=os.environ['INGEST_USER']).first()` wenn `INGEST_USER` gesetzt, sonst `User.query.first()` (Single-User-App). Kein User vorhanden → 503.

**Body-Validierung**: nicht-dict → 400 „Ungültiger Request-Body. JSON-Objekt erwartet." (Nachbar-Microcopy); `conversion_type` fehlt/nicht in `ALLOWED_CONVERSION_TYPES` → 400; `content` leer/fehlt → 400; `title` default „Untitled", `[:255]`; `topics` optional Liste (nicht-Liste → ignorieren oder 400, deine Wahl, dokumentieren); `report_date` optional ISO-parsebar (unparsebar → ignorieren, `created_at`=now); `source_id` optional String.

**Dedup (idempotent, kein Schema-Touch)**: wenn `source_id` da → vor Insert prüfen, ob für den Ziel-User schon eine Conversion mit diesem `source_id` in `metadata_json` existiert. Treffer → **200** mit der bestehenden id + `"deduped": true`, **keine** neue Zeile. Sonst anlegen und `source_id` in `metadata_json` ablegen (z.B. `{"source_id": "...", "ingested": true}`). Mechanik (LIKE auf `metadata_json` vs. Load+json-parse bei Kleinst-Volumen) ist deine Wahl — Newsletter sind ~wöchentlich, Volumen winzig; Hauptsache idempotent und getestet.

**Anlage**: `Conversion(user_id=target, conversion_type, title, content, metadata_json=…, created_at=report_date-falls-valide-sonst-default)`. Danach `topics[]` → für jeden Topic `Tag.get_or_create(target_user_id, topic)` + an `conversion.tag_refs` hängen (R2-A-Junction, derselbe DRY-Anker wie die Tag-POST-Routen). Response **201** mit `conversion.to_dict()`.

### Tests (`tests/test_ingest.py`)
- 201 valid create (mit korrektem Bearer) + persistiert.
- 401 bei fehlendem / falschem Token.
- 503 wenn `INGEST_TOKEN` unset (fail-closed).
- 400 bei nicht-dict / fehlendem content / ungültigem conversion_type.
- Dedup: zweimal derselbe `source_id` → 200 idempotent, genau **eine** Zeile.
- `topics[]` → Tags an der Conversion (R2-A-Junction), inkl. Reuse bestehender Tags.
- `report_date` → `created_at`; fehlend/unparsebar → default-now.
- User-Auflösung (INGEST_USER gesetzt vs. Single-User-Fallback).
- **CSRF-exempt verifiziert**: POST **ohne** CSRF-Token, nur mit Bearer → erfolgreich (kein 400 csrf_expired).
- Stil an `tests/test_conversion_tags.py` / `tests/test_highlights.py` (App-Context-Fixtures, Test-User-Setup).

### Verify + Commit Phase 2
- `pytest tests/` grün, Tests zählen hoch.
- **curl-Smoke** gegen den laufenden Container (Server-zu-Server, kein Browser nötig): mit gesetztem `INGEST_TOKEN` im Container-Env einen echten Newsletter-POST absetzen (Title/Markdown/topics/report_date/source_id), 201 prüfen → in `/library` erscheint der Eintrag mit Typ-Badge „AI-Newsletter" + den Topic-Tags + R2-B-Filter `?tag=…` greift; Re-POST mit gleichem `source_id` → 200 deduped, keine zweite Zeile; falscher Token → 401; ohne `INGEST_TOKEN` → 503.
- **Sicherheits-Selbstreview** des neuen Endpoints (das Projekt hat `/security-review` — nutzen oder einen fokussierten Pass): Token-Compare constant-time? Fail-closed? Token nirgends geloggt? Nur dieser Endpoint exempt? Ownership/User-Auflösung korrekt?
- **Commit** (plain-prose): `NL1-ingest: generischer token-auth Ingestion-Endpoint POST /api/ingest/conversion` + Body mit Auth/Dedup/Topics-Mapping. Push direkt.

**STOP — Bericht an Master (inkl. Sicherheits-Selbstreview-Ergebnis). Nicht in Phase 3 bis Sign-off.**

---

## Phase 3 — Kontrakt-Doc + Wrap-up

1. `pytest tests/` final grün.
2. **Kontrakt-Doc** `docs/ingest_contract.md` schreiben — das Interface, das NL2 (`email-automation`) implementiert: Endpoint-URL, Auth-Header, exaktes Body-Schema (Tabelle oben), Response-Codes (201/200-deduped/400/401/503), Dedup-Semantik (`source_id`), Mappings (topics→tags, report_date→created_at), und der Hinweis auf die nötigen Envs (`INGEST_TOKEN`, optional `INGEST_USER`). Knapp und präzise — NL2 soll daraus ohne Rückfrage bauen können.
3. **Env-Mechanik dokumentieren**: falls `.env.example` existiert, `INGEST_TOKEN=` + `INGEST_USER=` (leer/Platzhalter) ergänzen. **Kein echtes Secret im Repo** — nur die Variablennamen; Oliver füllt den Wert in `.env`.
4. **STATUS.md** + **BACKLOG.md**: NL1 ☑ done mit Commit-Hashes; **NL2 (email-automation-Push-Seite)** als neues P1 eintragen — **blockiert auf Mintbox-Erreichbarkeit** + verweist auf `docs/ingest_contract.md` als Spezifikation. R2-C bleibt im Backlog.
5. **Memory**: erwäge einen `feedback_*`/`reference_*`-Eintrag zum Token-Auth-Server-zu-Server-Pattern (CSRF-exempt via `app.extensions['csrf']`, fail-closed, constant-time-compare) — erster nicht-Session-Endpoint, wiederverwendbar für künftige externe Ingestion. Deine Einschätzung.
6. Push bestätigen.

**STOP — Schluss-Bericht an Master.**

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute. Bei „mach jetzt einfach"/Frust: einmal nachfragen, dann der Antwort folgen.

**Doc-Wrap-up-Guard** (Memory `reference_markdown_bullet_delete_newline.md`): nach Bullet-Löschungen in STATUS/BACKLOG vor dem Commit `grep -nE '(- \*\*.*){2,}' BACKLOG.md` — verklebte Bullets reparieren.

---

## Größe

**M** — neues Modul + erster nicht-Session-/CSRF-exempter Endpoint (sicherheits-sensibel), Typ-Foundation über 4 Files, Test-Welle (~10), Kontrakt-Doc. Kein Schema-Touch (Dedup über `metadata_json`). Backend-lastig → curl-Smoke statt Browser für Phase 2.

---

## Konstitutiv mit-genommen, falls berührt

- Nichts Zusätzliches — der Sprint ist eng geschnitten. Topic→Tag-Mapping nutzt die R2-A-Junction wieder (kein neuer Mechanismus).

---

## BACKLOG- und STATUS-Updates nach Abschluss

- ✓ Sprint NL1 durch (2026-06-01), zwei Code-Commits + Kontrakt-Doc.
- 📋 **NL2** email-automation-Push (P1, blockiert auf Mintbox, Spec = `docs/ingest_contract.md`).
- STATUS.md / BACKLOG.md wie Phase 3.
