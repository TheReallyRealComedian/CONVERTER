"""The web app's security posture in one module: the Secure-by-scheme session
cookie (SEC-AUDIT), the CSRF inversion for bearer writes (MOBILE-AUTH), the
baseline response headers, the CSRF-token endpoint and the error handlers
(413, CSRF).

Moved verbatim out of ``app_pkg/__init__.py`` in ARCH-FACTORY P1 (function
bodies byte-equal). ``create_app`` wires these in the same order as before:
``CSRFProtect`` → ``_register_csrf_inversion`` … → ``_register_error_handlers``
→ ``_register_security_headers`` → ``_register_csrf_endpoint``. Behaviour is
pinned by the cookie, CSRF-inversion, ProxyFix and header tests, none of which
changed for the move.
"""
from flask import jsonify, request, url_for
from flask.sessions import SecureCookieSessionInterface
from flask_login import login_required
from flask_wtf.csrf import CSRFError, generate_csrf


class HttpsOnlySecureSessionInterface(SecureCookieSessionInterface):
    """SEC-AUDIT: the session cookie is ``Secure`` exactly when the request
    itself arrived over HTTPS.

    Behind nginx (ProxyFix, X-Forwarded-Proto) that is every browser request —
    nginx 301s port 80 before proxying, so no browser session is ever issued
    over plain http through the public name. A blanket ``SESSION_COOKIE_SECURE``
    would break the clients that legitimately talk plain http: the
    converter-mcp container logs in with a cookie session over the Docker
    network (``http://markdown-converter-web:5000``) and httpx — like every
    RFC-following client — never sends a Secure cookie over http, so its very
    login POST would lose the session and die at CSRF. Same for Mac dev on
    ``http://localhost:5656`` and the in-container browser smokes. So: Secure
    wherever TLS is, and nowhere a flag has to be remembered. The remember
    cookie stays Secure always (``REMEMBER_COOKIE_SECURE``) — the http clients
    authenticate through the session and never need it.
    """

    def get_cookie_secure(self, app):
        return request.is_secure


def _register_csrf_inversion(app, csrf):
    """MOBILE-AUTH P2 — CSRF inversion for bearer writes.

    Flask-WTF's automatic per-request protection is turned off
    (``WTF_CSRF_CHECK_DEFAULT = False``); the ``before_request`` below
    re-applies it explicitly to every cookie-session mutation and skips it
    for bearer requests, so the iOS app can write without the cookie/CSRF
    dance while the web UI's CSRF posture stays byte-identical.

    The handler replicates the guard chain of Flask-WTF==1.2.1's automatic
    ``csrf_protect`` (flask_wtf/csrf.py::CSRFProtect.init_app: ENABLED →
    CHECK_DEFAULT → method → endpoint → exempt-blueprints → exempt-views →
    protect()), because ``csrf.protect()`` itself checks NONE of those
    guards — only the method. ``_exempt_views`` / ``_exempt_blueprints``
    are private Flask-WTF attributes: re-verify this replication against
    upstream on any Flask-WTF version bump.
    """
    app.config['WTF_CSRF_CHECK_DEFAULT'] = False

    @app.before_request
    def csrf_protect_session_writes():
        if not app.config['WTF_CSRF_ENABLED']:
            return
        if request.method not in app.config['WTF_CSRF_METHODS']:
            return
        # A cross-site browser request cannot carry an Authorization header
        # (custom headers need a CORS preflight, which fails without server
        # opt-in), so header *presence* is a CSRF-safe skip signal — the
        # same trust class as X-CSRFToken. Validity is deliberately NOT
        # checked here: an invalid bearer skips CSRF but dies at auth (401),
        # fail-closed. There is no cookie authority to ride on these
        # requests from a cross-site context.
        if request.headers.get('Authorization', '').startswith('Bearer '):
            return
        if not request.endpoint:
            return
        if app.blueprints.get(request.blueprint) in csrf._exempt_blueprints:
            return
        view = app.view_functions.get(request.endpoint)
        if view is not None:
            dest = f'{view.__module__}.{view.__name__}'
            if dest in csrf._exempt_views:
                return
        csrf.protect()


def _register_error_handlers(app):
    @app.errorhandler(413)
    def request_entity_too_large(error):
        if request.content_type and 'multipart/form-data' in request.content_type:
            return jsonify({'error': 'File too large. Maximum upload size is 500 MB.'}), 413
        return jsonify({'error': 'Request too large.'}), 413

    @app.errorhandler(CSRFError)
    def handle_csrf_error(error):
        if request.accept_mimetypes.best == 'application/json' or request.path.startswith('/api/'):
            return jsonify({'error': 'csrf_expired', 'message': str(error.description)}), 400
        reload_url = request.referrer or url_for('markdown_converter')
        html = (
            '<!DOCTYPE html><html><head><meta charset="UTF-8">'
            '<title>Session expired</title>'
            f'<meta http-equiv="refresh" content="2;url={reload_url}">'
            '<style>body{font-family:system-ui,sans-serif;max-width:520px;margin:4rem auto;'
            'padding:2rem;color:#333;text-align:center;}h1{font-size:1.2rem;margin-bottom:1rem;}'
            'p{color:#666;line-height:1.5;}</style></head><body>'
            '<h1>Session expired</h1>'
            '<p>Your security token expired. Reloading the page automatically&hellip;</p>'
            f'<p><a href="{reload_url}">Click here if nothing happens.</a></p>'
            '</body></html>'
        )
        return html, 400


# SEC-AUDIT: baseline headers on EVERY response — pages, JSON, send_file
# downloads, static files, error answers (after_request runs for all of
# them). They travel with the app rather than living in nginx, so the answer
# carries them however it is reached. setdefault: a view that sets its own
# value wins. HSTS is NOT here — it belongs to the TLS terminator (nginx).
#   * X-Frame-Options DENY — nothing frames the app: the markdown preview is
#     an srcdoc iframe (no HTTP response to deny), Dashy links with _blank.
#   * Referrer-Policy strict-origin-when-cross-origin — must keep the full
#     same-origin Referer: since ProxyFix, Flask-WTF's SSL-strict check needs
#     it on every cookie-session write behind nginx. A policy that drops
#     same-origin referrers ('no-referrer') would break every web-UI write.
#   * Permissions-Policy WITHOUT microphone — the audio converter's live
#     transcription needs getUserMedia.
SECURITY_HEADERS = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'Referrer-Policy': 'strict-origin-when-cross-origin',
    'Permissions-Policy': 'camera=(), geolocation=(), payment=(), usb=()',
}


def _register_security_headers(app):
    @app.after_request
    def set_security_headers(response):
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response


def _register_csrf_endpoint(app):
    @app.route('/api/csrf-token', methods=['GET'])
    @login_required
    def get_csrf_token():
        return jsonify({'csrf_token': generate_csrf()})
