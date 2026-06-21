# Send-to-Kindle

Send a library element (`Conversion`, content = Markdown) to a Kindle as a
reflowable **EPUB**. The "An Kindle" button sits on the library card (hover
action row) and on the detail view; it POSTs to a `@login_required` endpoint
that renders → packages → mails the document.

This is CONVERTER's first **outbound mail** integration.

## Delivery model — email only (there is no API)

Amazon exposes **no public Send-to-Kindle API**. The only programmatic path is
**email**: an EPUB attachment mailed to the device's `<name>@kindle.com` address
from an address Amazon has approved. So the flow is:

```
Conversion.content (Markdown)
  → render_markdown_to_html()         # the shared renderer; Kindle output = reader output
  → epub_service.build_epub()         # single-chapter reflowable EPUB via ebooklib
  → kindle_service.send_to_kindle()   # SMTP attachment to KINDLE_TO_EMAIL
```

**Why EPUB, not PDF:** EPUB is reflowable → native Kindle reading UX on a 6"
screen. The existing PDF flow (`app_pkg/markdown.py`, via Playwright) is
fixed-layout and reads poorly on a Kindle, so it is *not* reused here.

## One-time setup (Oli's steps)

1. **Amazon — approve the sender.** Go to *Manage Your Content and Devices →
   Preferences → Personal Document Settings*. Add the address you'll use as
   `KINDLE_FROM_EMAIL` to the **Approved Personal Document E-mail List**. Mail
   from any other address is silently dropped by Amazon.
2. **Amazon — find the destination.** On the same page, note your device's
   **`<name>@kindle.com`** address — that's `KINDLE_TO_EMAIL`.
3. **`.env`** (on both the Mac and the Mintbox) — fill the keys below.
4. **Deploy** on the Mintbox: `git pull` + `docker compose up -d --build` (the
   new `ebooklib` dep makes the image rebuild; **no migration**).

## Configuration (`.env`, fail-closed like `CARD_TOKEN`)

| Key | Meaning |
| --- | --- |
| `KINDLE_SMTP_HOST` | SMTP server host (e.g. your mail provider). |
| `KINDLE_SMTP_PORT` | `465` → implicit SSL (default). Any other value (e.g. `587`) → STARTTLS. |
| `KINDLE_SMTP_USERNAME` | SMTP login. |
| `KINDLE_SMTP_PASSWORD` | SMTP password. **Never logged.** |
| `KINDLE_FROM_EMAIL` | Approved sender (must be on Amazon's Approved list). |
| `KINDLE_TO_EMAIL` | The device's `<name>@kindle.com`. **Server-fixed recipient.** |

If **any** required key is unset/empty the endpoint is **fail-closed**: it
returns `503` and never attempts a connection — exactly like `CARD_TOKEN`. Env
is read **per request**, so rotating a credential needs no restart.

## Server-fixed recipient (anti-relay)

The recipient is always `KINDLE_TO_EMAIL` from the environment. It is **never**
read from the request body — a `to`/`recipient` field in the POST is ignored.
This keeps the endpoint from becoming an open mail relay. (Covered by a test.)

## Endpoint

`POST /api/conversions/<id>/send-to-kindle` — `@login_required`, owner-scoped,
**not** CSRF-exempt (it's a session write; the browser sends `X-CSRFToken` via
the global `base.html` fetch wrapper).

| Situation | Status | Body |
| --- | --- | --- |
| Sent | `200` | `{"success": true}` |
| Foreign / missing conversion | `404` | — |
| Kindle not configured | `503` | `{"error": "Kindle nicht konfiguriert."}` |
| Empty content | `400` | `{"error": "Kein Inhalt zum Senden."}` |
| SMTP / timeout failure | `502` | `{"error": "Versand an Kindle fehlgeschlagen."}` |

SMTP failures are logged (traceback only — never the password) and mapped to
`502`; the client never sees a `500` leak.

## Code

- `services/epub_service.py` — `build_epub(title, html_body, …) -> bytes`. Pure,
  network-free. An empty/None body still yields a valid EPUB (a placeholder
  paragraph avoids an `ebooklib`/lxml empty-document crash in the EPUB3 nav scan).
- `services/kindle_service.py` — `is_configured()` + `send_to_kindle(...)` (stdlib
  `smtplib` + `EmailMessage`, 20 s connection timeout).
- `app_pkg/kindle.py` — the route (`register(app)`), wired in `app.py`.

## Known caveat — XHTML

`render_markdown_to_html` emits HTML5 (e.g. `<br>`, not `<br/>`). `ebooklib` and
Amazon's converter are generally tolerant, and the test round-trip is clean. If a
real Kindle ever mis-renders a document, serialize the body through an XHTML
void-tag fixup in `build_epub` (a hook comment marks the spot). Verify on the
device after the first real end-to-end send.

## Deliberately out of scope (v1)

PDF/DOCX target · request-controlled recipient · a "sent-to-Kindle" tracking
column · batch send · a settings UI for the address · an RQ job (sent
synchronously with an SMTP timeout; revisit if it blocks noticeably).
