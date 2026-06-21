# Sprint KINDLE — Library-Element nach Kindle senden (L, 4 Phasen)

> **Executor-Doc.** Phasen nacheinander, nach jeder **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün (Baseline **352**). Du committest jede Phase selbst (eigener Hash + push). Arbeitsverzeichnis `/Users/olivergluth/CODE/CONVERTER`. Working-Practice in `CLAUDE.md` (Sektion „Working Practice").

## Ziel & Entscheidungen (gesetzt — kein Workshop, nicht neu aufmachen)

Ein **Library-Element** (`Conversion`, Inhalt = Markdown) auf den Kindle bringen. Entscheidungen aus Master-Recherche + Olis Wahl (2026-06-21):

- **Format = EPUB** (reflowable, native Kindle-Lese-UX; PDF wäre Fixed-Layout = schlecht auf 6"). EPUB via **`ebooklib`** (pure-Python pip-Dep, **kein** pandoc-System-Binary), gebaut aus dem **vorhandenen** `render_markdown_to_html` → Kindle-Output = Reader-Output.
- **Lieferweg = Send-to-Kindle-E-Mail** (der **einzige** programmatische Pfad — es gibt keine öffentliche Send-to-Kindle-API). SMTP-Anhang an die geräteeigene `<name>@kindle.com`.
- **Konfiguration = `.env`** (Single-User-App, wie `CARD_TOKEN`). SMTP-Creds + Absender + Ziel-`@kindle.com` als Env-Vars; **fail-closed** ohne Konfiguration; Passwort **nie geloggt**.
- **Empfänger ist server-fest** (`KINDLE_TO_EMAIL` aus `.env`), **nie aus dem Request** — der Endpoint darf nicht zum offenen Mail-Relay werden.

## Verifizierte Code-Fakten (Master-gegroundet — bau darauf)

- **Markdown→HTML-Renderer existiert + ist geteilt**: `render_markdown_to_html(text) -> str` in [app_pkg/markdown_render.py](app_pkg/markdown_render.py) (markdown-it-py + **nh3-sanitisiert**, Pygments inline via `noclasses=True`). Schon genutzt vom Reader ([app_pkg/library.py:349](app_pkg/library.py)) und vom PDF-Flow ([app_pkg/markdown.py:141](app_pkg/markdown.py)). **Wiederverwenden, nicht neu bauen.**
- **PDF-Flow als Vorlage** (temp-file + binär ausliefern): [app_pkg/markdown.py](app_pkg/markdown.py) (`tempfile.NamedTemporaryFile` → `page.pdf()` → `send_file`).
- **Conversion** trägt `content` (Markdown), `title`, owner-scope. `get_owned_conversion(id)` (→ 404) liegt in [app_pkg/library.py](app_pkg/library.py); andere Module importieren ihn schon (`from .library import get_owned_conversion`, vgl. highlights.py).
- **Route-Module-Registry**: [app.py](app.py) ruft pro Modul `…_module.register(app)` (Zeilen ~63–74). Neues Modul muss dort ergänzt werden (Import + `register`).
- **`.env.example`** existiert mit dem `CARD_TOKEN`/`INGEST_*`-Muster (leere Keys + Kommentar) — dort die `KINDLE_*`-Keys nachziehen.
- **`ebooklib` ist neu** (nicht in requirements.txt).
- **Auth-Posture**: das ist ein **Session-Write** (der User schickt **sein** Element) → `@login_required`, **kein** Token, läuft über den globalen `base.html`-CSRF-fetch-Wrapper (X-CSRFToken automatisch), **nicht** CSRF-exempt.
- **Test-Suite-Grenze** (CLAUDE.md): Tests rendern keine Templates + mocken SDK-/Transport-Boundaries. Mail-Versand wird **gemockt** (kein echtes Netz); UI braucht **Live-Smoke**.

## Phase 1 — EPUB-Erzeugung (`services/epub_service.py`) + Dep + Tests

1. **`ebooklib`** zu `requirements.txt` (gepinnte Version, analog der anderen Pins).
2. **`services/epub_service.py`** — pure, netz-frei:
   - `build_epub(title: str, html_body: str, *, author: str = 'CONVERTER', language: str = 'de', identifier: str | None = None) -> bytes`
   - Mit `ebooklib.epub`: `EpubBook()`, `set_identifier` (z.B. `identifier or f'converter-{slug}'`), `set_title(title)`, `set_language(language)`, `add_author(author)`; **ein** `EpubHtml`-Kapitel mit `content = f'<html><body>{html_body}</body></html>'`; `EpubNcx` + `EpubNav` + Spine + `toc`. Schreibe via `tempfile` (Muster wie PDF-Flow) und gib die **Bytes** zurück (oder `BytesIO` — kein Temp-File-Leck).
   - Leerer/None-`html_body` → trotzdem valides (leeres) EPUB, kein Crash.
   - **XHTML-Caveat**: `render_markdown_to_html` liefert HTML5 (`<br>` statt `<br/>`). ebooklib + Amazons Konverter sind i.d.R. tolerant — **erst direkt versuchen**; falls die EPUB-Validierung/das Kindle-Rendering im Smoke (Phase 3) zickt, HTML über einen XHTML-Fixup serialisieren (lxml/regex für Void-Tags). Im Bericht vermerken, ob nötig.
3. **Tests** (`tests/test_epub_service.py`): `build_epub` gibt nicht-leere Bytes; **Round-Trip** `epub.read_epub` der Bytes → Titel matcht + ein bekannter Inhalts-Substring steckt im Kapitel; leerer Body crasht nicht.

`pytest tests/` grün ≥ 352. **Stop + Bericht.**

## Phase 2 — Kindle-Mail-Service + Endpoint + Tests

1. **`services/kindle_service.py`** (stdlib `smtplib` + `email.message.EmailMessage`):
   - Env-Config: `KINDLE_SMTP_HOST`, `KINDLE_SMTP_PORT` (Default 465 SSL bzw. 587 STARTTLS — eine Variante wählen, dokumentieren), `KINDLE_SMTP_USERNAME`, `KINDLE_SMTP_PASSWORD`, `KINDLE_FROM_EMAIL` (Approved-Sender), `KINDLE_TO_EMAIL` (Ziel-`@kindle.com`).
   - `is_configured() -> bool` (alle Pflicht-Keys gesetzt).
   - `send_to_kindle(filename: str, epub_bytes: bytes, subject: str) -> None`:
     - Empfänger = **`KINDLE_TO_EMAIL` aus Env** (NIE aus dem Request).
     - `EmailMessage`, From = `KINDLE_FROM_EMAIL`, Anhang `epub_bytes` als `maintype='application', subtype='epub+zip', filename=filename`.
     - SMTP mit **Connection-Timeout** (z.B. `timeout=20`), Login, send, quit. **Passwort nie loggen.**
   - Fehler (SMTP/Timeout) propagieren (der Endpoint mappt sie auf 502).
2. **`app_pkg/kindle.py`** mit `register(app)`:
   - `POST /api/conversions/<int:conversion_id>/send-to-kindle` — `@login_required`.
   - `conversion = get_owned_conversion(conversion_id)` (fremd/fehlend → 404).
   - **Fail-closed**: `if not kindle_service.is_configured(): return jsonify({'error': 'Kindle nicht konfiguriert.'}), 503`.
   - Leerer `conversion.content` → 400 `{'error': 'Kein Inhalt zum Senden.'}`.
   - `html = render_markdown_to_html(conversion.content)` → `epub = epub_service.build_epub(conversion.title or 'Dokument', html)` → `kindle_service.send_to_kindle(f'{safe_title}.epub', epub, subject=conversion.title or 'Dokument')`.
   - Erfolg → 200 `{'success': True}`. SMTP-Fehler → **502** `{'error': 'Versand an Kindle fehlgeschlagen.'}` (Exception loggen — **ohne** Passwort). **Nicht** CSRF-exempt.
   - In [app.py](app.py) registrieren (Import + `kindle_module.register(app)` in der Reihe ~63–74).
3. **Tests** (`tests/test_kindle.py`) — **Mock-Boundary = der SMTP-Transport** (`smtplib.SMTP`/`SMTP_SSL` patchen, kein echtes Netz):
   - unkonfiguriert (Env-Keys leer) → **503**;
   - fremde Conversion → **404**; leerer Inhalt → **400**; unauth → Login-Redirect/401;
   - Erfolg → **200**, und der gebaute `EmailMessage` ist korrekt: **To == `KINDLE_TO_EMAIL`** (nicht request-gesteuert), genau **ein** Anhang mit `.epub`/`application/epub+zip`, From == `KINDLE_FROM_EMAIL`;
   - SMTP wirft (z.B. `SMTPException`) → **502** (kein 500-Leak, Passwort nicht in der Response);
   - **Anti-Relay-Probe**: ein `to`/`recipient`-Feld im Request-Body wird **ignoriert** (Empfänger bleibt der Env-Wert).

`pytest tests/` grün. **Stop + Bericht.**

## Phase 3 — UI (Card + Detail) + Live-Smoke

1. **„An Kindle"-Aktion** auf zwei Flächen, DS-konform (token-driven, vorhandene Button-Muster — vgl. `.place-control`/`card-action`-Konventionen; **kein** Hardcode-Hex):
   - Library-**Karte** (Action-Row) — [templates/library.html](templates/library.html) + [static/js/library.js](static/js/library.js).
   - **Detail**-View — [templates/library_detail.html](templates/library_detail.html) + [static/js/library_detail.js](static/js/library_detail.js).
2. **`sendToKindle(id)`** über den globalen `base.html`-CSRF-fetch-Wrapper → `POST /api/conversions/<id>/send-to-kindle`. **Keine** Bestätigung (nicht-destruktiv). Erfolg → `showToast('An Kindle gesendet')`; Fehler → `showToast` mit der Server-Message (503 → „Kindle nicht konfiguriert.", 502 → „Versand an Kindle fehlgeschlagen."), Microcopy-Regeln (max 2 Sätze, keine Emojis bei Fehlern). Während des Requests den Button kurz disablen (Doppelklick-Schutz).
3. **Live-Smoke** (lokale Docker-Instanz, MacChrome **dark+light**, **0 Console-Errors**): Button auf Karte **und** Detail → Request feuert (CSRF ok, kein 400). **Wenn** lokal `KINDLE_*` + Approved-Sender stehen → **echter Versand an Olis Kindle** und auf dem Gerät prüfen (Lesbarkeit/Reflow; XHTML-Caveat aus Phase 1 verifizieren). **Falls nicht konfiguriert** → der 503-Pfad + Fehler-Toast wird gesmoked, der echte End-to-End-Versand ist dann **Olis Real-Welt-Schritt nach dem Deploy** (in den Bericht).

`pytest tests/` grün; `node --check` der berührten JS. **Stop + Bericht.**

## Phase 4 — Wrap

1. **`.env.example`**: `KINDLE_*`-Block ergänzen (leere Keys + Kommentar — Send-to-Kindle-Email, Approved-Sender-Hinweis), Muster wie `CARD_TOKEN`.
2. **STATUS.md** + **BACKLOG.md**: KINDLE ☑ done mit Hashes (Muster wie R4-LEARN-P6); den „Aktiv offen"-Block in BACKLOG aktualisieren (KINDLE raus/done, READER-ADJ bleibt nächstes P1). **Bullet-Guard** (`grep -nE '(- \*\*.*){2,}' BACKLOG.md`, Memory `reference_markdown_bullet_delete_newline`).
3. **Doc**: kurzes `docs/kindle.md` (oder Abschnitt) — das Liefermodell (Email-only, kein API), die **Olis-Setup-Schritte** (Amazon Approved-Personal-Document-Email-Liste + die `@kindle.com`-Adresse finden), die `.env`-Keys, EPUB-Format, der server-feste Empfänger (Anti-Relay). Ein Halbsatz in CLAUDE.md *Architecture Notes* (neue Deps `ebooklib` + SMTP-Versand; fail-closed Env wie CARD_TOKEN).
4. **Memory** (`reference_*`): wahrscheinlich **ja** — wiederverwendbares Faktum: „Send-to-Kindle = Email-only (keine API), server-fester Empfänger (Anti-Relay), EPUB via `ebooklib` über den geteilten `render_markdown_to_html`, fail-closed `.env` wie CARD_TOKEN." Plus MEMORY.md-Pointer.
5. Finaler `pytest tests/` grün.

**Stop + Schluss-Bericht** — inkl. **Olis Real-Welt-Schritte** (nicht deine):
> 1. **Amazon**: die `KINDLE_FROM_EMAIL`-Adresse auf die *Approved Personal Document E-mail List* setzen (Manage Your Content and Devices) + die eigene `@kindle.com`-Adresse heraussuchen.
> 2. **`.env`** (Mac + Mintbox): `KINDLE_SMTP_*` + `KINDLE_FROM_EMAIL` + `KINDLE_TO_EMAIL` füllen.
> 3. **Mintbox-Deploy**: `git pull` + `docker compose up -d --build` (neuer Dep `ebooklib` → Image baut neu; **keine Migration**).
> 4. **End-to-End**: ein Library-Element → „An Kindle" → auf dem Gerät ankommen + lesen.

## Bewusst NICHT (Scope-Grenze v1)

- **Kein** PDF-/DOCX-Ziel (EPUB ist die Entscheidung; PDF existiert separat als Download).
- **Kein** request-gesteuerter Empfänger (server-fest, Anti-Relay).
- **Kein** „sent-to-Kindle"-Tracking-State / Schema-Touch (YAGNI v1; ggf. späterer `kindle_sent_at`-Follow-up).
- **Kein** Batch-/Bulk-Send (per-Item v1).
- **Kein** Settings-UI für die Kindle-Adresse (`.env` v1; UI-Setting späterer Polish).
- **Kein** RQ-Job v1 (synchron mit SMTP-Timeout; falls der Versand spürbar blockiert → RQ-Follow-up, RQ existiert für Podcasts).

## Akzeptanz

- [ ] Library-Element → EPUB (via `ebooklib`, aus `render_markdown_to_html`) → per Send-to-Kindle-Email verschickt
- [ ] `POST /api/conversions/<id>/send-to-kindle` `@login_required`, owner-scoped (fremd → 404), fail-closed (unkonfiguriert → 503), leerer Inhalt → 400, SMTP-Fehler → 502
- [ ] Empfänger ist **server-fest** aus `.env` (Anti-Relay-Test grün), SMTP-Passwort nie geloggt/in Response
- [ ] Button auf Karte **und** Detail, Erfolg/Fehler-Toast, CSRF über den globalen Wrapper (kein 400)
- [ ] Mail-Versand in Tests **gemockt** (kein echtes Netz); `pytest` grün ≥ 352 + neue Tests
- [ ] `.env.example` + Doc (Olis Amazon-Setup + Keys) aktualisiert
