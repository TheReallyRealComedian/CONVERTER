"""SEC-SSRF: das Egress-Tor für den server-seitigen PDF-Renderer.

Reines Stdlib-Modul (kein Flask-Import). ``fetch_public_https`` ist der EINZIGE
Weg, auf dem der Markdown→PDF-Renderer ins Netz greift. Es holt eine ``https``-URL
nur dann, wenn die URL einen öffentlichen DNS-Namen trägt, löst **alle** Adressen
dieses Hosts auf und findet **jede** global-routbar, pinnt die TLS-Verbindung an
die geprüfte Adresse (schließt DNS-Rebinding zwischen Prüfung und Verbindung),
reicht genau zwei Header weiter (``Accept``, ``User-Agent``) und keine Cookies,
deckelt den Körper und erlaubt nur Bild-/Font-/CSS-Content-Types. Weiterleitungen
werden **hier** verfolgt (jeder Schritt läuft wieder komplett durchs Tor, Variante A),
damit der Browser nie eine 3xx-Antwort bekommt, der er am Tor vorbei folgen könnte.
Alles andere wirft ``EgressBlocked(reason)``.
"""
from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
import time
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit


DEFAULT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_TOTAL_TIMEOUT = 10.0
MAX_REDIRECTS = 5

# Chromium liefert Bilder, Fonts und die @import-Stil-CSS. Nichts anderes darf
# der Renderer laden — ein text/html hinter einer Bild-URL ist ein Fehler.
_ALLOWED_CONTENT_PREFIXES = (
    'image/', 'font/', 'text/css',
    'application/font-', 'application/x-font-',
)
_REDIRECT_STATUS = {301, 302, 303, 307, 308}


class EgressBlocked(Exception):
    """Das Tor hat die Anfrage abgewiesen. ``reason`` ist ein stabiles Kürzel
    für die WARNING-Zeile; ``detail`` ist Kontext (nie ein Antwort-Body)."""

    def __init__(self, reason: str, detail: str = ''):
        super().__init__(reason if not detail else f'{reason}: {detail}')
        self.reason = reason
        self.detail = detail


@dataclass
class EgressResponse:
    status: int
    content_type: str
    body: bytes
    final_url: str


def _is_public_ip(ip_str: str) -> bool:
    """True, wenn ``ip_str`` nach IPv4-Mapped-Auspackung eine sichere öffentliche
    Adresse ist. Die Prüfung ist ``is_global`` (nicht ``not is_private``): CGNAT
    (100.64.0.0/10) ist weder privat noch global und muss gesperrt bleiben."""
    ip = ipaddress.ip_address(ip_str)
    mapped = getattr(ip, 'ipv4_mapped', None)
    if mapped is not None:
        ip = mapped
    if not ip.is_global:
        return False
    if ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return False
    return True


def _validate_target(url: str):
    """Prüft Schema/Userinfo/Port/Host-Form. Gibt ``(host, path)`` zurück oder
    wirft ``EgressBlocked``. IP-Literale in jeder Form sind gesperrt."""
    parts = urlsplit(url)
    if parts.scheme != 'https':
        raise EgressBlocked('scheme', parts.scheme or '(none)')
    if parts.username or parts.password:
        raise EgressBlocked('userinfo')
    try:
        port = parts.port
    except ValueError:
        raise EgressBlocked('port', 'unparsable')
    if port is not None and port != 443:
        raise EgressBlocked('port', str(port))
    host = parts.hostname
    if not host:
        raise EgressBlocked('no_host')
    # (1) klassische IP-Literale (dotted-quad, IPv6, gebracketed → urlsplit
    # entfernt die Klammern bereits).
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise EgressBlocked('ip_literal', host)
    # (2) numerische Kurzformen (0x7f000001, 2130706433, 0177.0.0.1), die der
    # Resolver sonst still zu einer IP macht — AI_NUMERICHOST erkennt sie ohne
    # DNS. Ein echter Name wirft hier gaierror und läuft weiter.
    try:
        socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM,
                           flags=socket.AI_NUMERICHOST)
    except socket.gaierror:
        pass
    else:
        raise EgressBlocked('ip_literal', host)
    path = parts.path or '/'
    if parts.query:
        path = f'{path}?{parts.query}'
    return host, path


def _resolve_checked(host):
    """Löst ALLE Adressen auf und prüft JEDE. Gibt ``(family, sockaddr)`` der
    ersten geprüften Adresse zurück (daran wird die Verbindung gepinnt)."""
    try:
        addrs = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise EgressBlocked('dns_error', str(exc))
    if not addrs:
        raise EgressBlocked('dns_empty')
    for _family, _type, _proto, _canon, sockaddr in addrs:
        if not _is_public_ip(sockaddr[0]):
            raise EgressBlocked('non_public_address', sockaddr[0])
    family, _type, _proto, _canon, sockaddr = addrs[0]
    return family, sockaddr


def _perform_request(host, family, sockaddr, path, *, accept, user_agent,
                     connect_timeout, read_timeout, max_bytes):
    """Verbindet an die GEPRÜFTE Adresse (keine zweite Auflösung), TLS mit
    ``server_hostname=host`` gegen die System-CAs, GET mit genau den zwei
    weitergereichten Headern. Gibt ``(status, content_type, location, body)``.
    ``body`` ist auf ``max_bytes + 1`` Bytes begrenzt gelesen."""
    raw = socket.socket(family, socket.SOCK_STREAM)
    raw.settimeout(connect_timeout)
    try:
        raw.connect(sockaddr)
    except OSError as exc:
        raw.close()
        raise EgressBlocked('connect_failed', str(exc))
    try:
        ctx = ssl.create_default_context()
        tls = ctx.wrap_socket(raw, server_hostname=host)
    except ssl.SSLError as exc:
        raw.close()
        raise EgressBlocked('tls', str(exc))
    except OSError as exc:
        raw.close()
        raise EgressBlocked('connect_failed', str(exc))
    try:
        tls.settimeout(read_timeout)
        conn = http.client.HTTPConnection(host, 443, timeout=read_timeout)
        conn.sock = tls  # TLS ist schon aufgebaut → keine zweite Auflösung/Connect
        conn.putrequest('GET', path, skip_host=True, skip_accept_encoding=True)
        conn.putheader('Host', host)
        conn.putheader('Accept', accept)
        conn.putheader('User-Agent', user_agent)
        conn.endheaders()
        resp = conn.getresponse()
        status = resp.status
        content_type = (resp.getheader('Content-Type') or '').split(';')[0].strip().lower()
        location = resp.getheader('Location')
        body = resp.read(max_bytes + 1)
        return status, content_type, location, body
    except (http.client.HTTPException, OSError) as exc:
        raise EgressBlocked('read_failed', str(exc))
    finally:
        try:
            tls.close()
        except OSError:
            pass


def fetch_public_https(url, *, accept, user_agent, max_bytes=DEFAULT_MAX_BYTES,
                       connect_timeout=DEFAULT_CONNECT_TIMEOUT,
                       total_timeout=DEFAULT_TOTAL_TIMEOUT,
                       max_redirects=MAX_REDIRECTS) -> EgressResponse:
    """Holt ``url`` durchs Tor. Folgt Weiterleitungen selbst (Variante A): jeder
    Schritt läuft wieder durch ``_validate_target`` + ``_resolve_checked``, der
    Browser bekommt nie eine 3xx. Wirft ``EgressBlocked`` bei jedem Verstoß."""
    deadline = time.monotonic() + total_timeout
    current = url
    for _hop in range(max_redirects + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise EgressBlocked('timeout')
        host, path = _validate_target(current)
        family, sockaddr = _resolve_checked(host)
        status, content_type, location, body = _perform_request(
            host, family, sockaddr, path,
            accept=accept, user_agent=user_agent,
            connect_timeout=min(connect_timeout, remaining),
            read_timeout=max(0.1, remaining),
            max_bytes=max_bytes,
        )
        if status in _REDIRECT_STATUS:
            if not location:
                raise EgressBlocked('redirect_no_location')
            current = urljoin(current, location)
            continue
        if status < 200 or status >= 300:
            raise EgressBlocked('status', str(status))
        if len(body) > max_bytes:
            raise EgressBlocked('too_large')
        if not content_type.startswith(_ALLOWED_CONTENT_PREFIXES):
            raise EgressBlocked('content_type', content_type or '(none)')
        return EgressResponse(status=status, content_type=content_type,
                              body=body, final_url=current)
    raise EgressBlocked('too_many_redirects')
