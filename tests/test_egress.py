"""SEC-SSRF: tests for the egress gate ``services.egress.fetch_public_https``.

The network layer (``_perform_request``) and the resolver
(``socket.getaddrinfo`` inside the module) are patched, so nothing touches a
real socket. The gate rules are exercised one blocked reason at a time, plus
the redirect-following (Variante A) and the too-large / content-type paths.
"""
import socket

import pytest

from services import egress
from services.egress import EgressBlocked, EgressResponse, fetch_public_https


def _addrinfo(*ips):
    """Build getaddrinfo-shaped tuples for the given IP strings."""
    out = []
    for ip in ips:
        family = socket.AF_INET6 if ':' in ip else socket.AF_INET
        sockaddr = (ip, 443, 0, 0) if family == socket.AF_INET6 else (ip, 443)
        out.append((family, socket.SOCK_STREAM, 6, '', sockaddr))
    return out


@pytest.fixture
def patched(monkeypatch):
    """Patch the resolver to map a fixed host to given IPs, and capture the
    args ``_perform_request`` is called with (returning a canned response)."""
    state = {
        'resolve': {'example.org': _addrinfo('93.184.216.34')},
        'response': (200, 'image/png', None, b'PNGDATA'),
        'calls': [],
        'numeric_hosts': set(),
    }

    def fake_getaddrinfo(host, port, *args, **kwargs):
        flags = kwargs.get('flags', 0)
        if flags & socket.AI_NUMERICHOST:
            if host in state['numeric_hosts']:
                return _addrinfo(host)
            raise socket.gaierror(socket.EAI_NONAME, 'not numeric')
        if host in state['resolve']:
            return state['resolve'][host]
        raise socket.gaierror(socket.EAI_NONAME, 'unknown host')

    def fake_perform(host, family, sockaddr, path, **kwargs):
        state['calls'].append({'host': host, 'sockaddr': sockaddr,
                               'path': path, **kwargs})
        resp = state['response']
        if callable(resp):
            resp = resp(len(state['calls']))
        return resp

    monkeypatch.setattr(egress.socket, 'getaddrinfo', fake_getaddrinfo)
    monkeypatch.setattr(egress, '_perform_request', fake_perform)
    return state


def _fetch(url):
    return fetch_public_https(url, accept='*/*', user_agent='UA')


# --- happy path ------------------------------------------------------------

def test_public_https_image_passes(patched):
    resp = _fetch('https://example.org/x.png')
    assert isinstance(resp, EgressResponse)
    assert resp.status == 200 and resp.content_type == 'image/png'
    assert resp.body == b'PNGDATA'
    # exactly the two headers forwarded, connected to the checked address
    call = patched['calls'][0]
    assert call['host'] == 'example.org'
    assert call['sockaddr'][0] == '93.184.216.34'
    assert call['accept'] == '*/*' and call['user_agent'] == 'UA'


# --- scheme / userinfo / port ---------------------------------------------

def test_http_scheme_blocked(patched):
    with pytest.raises(EgressBlocked) as e:
        _fetch('http://example.org/x.png')
    assert e.value.reason == 'scheme'


def test_userinfo_blocked(patched):
    with pytest.raises(EgressBlocked) as e:
        _fetch('https://user@example.org/x.png')
    assert e.value.reason == 'userinfo'


def test_foreign_port_blocked(patched):
    with pytest.raises(EgressBlocked) as e:
        _fetch('https://example.org:8443/x.png')
    assert e.value.reason == 'port'


# --- IP literals in every form --------------------------------------------

@pytest.mark.parametrize('url', [
    'https://10.0.0.1/x.png',
    'https://127.0.0.1/x.png',
    'https://[::1]/x.png',
    'https://[fc00::1]/x.png',
])
def test_ip_literal_blocked(patched, url):
    with pytest.raises(EgressBlocked) as e:
        _fetch(url)
    assert e.value.reason == 'ip_literal'


def test_decimal_and_hex_literals_blocked(patched):
    # These are not dotted-quad, so ipaddress.ip_address rejects them; the
    # AI_NUMERICHOST probe is what catches them.
    patched['numeric_hosts'] = {'2130706433', '0x7f000001'}
    for host in ('2130706433', '0x7f000001'):
        with pytest.raises(EgressBlocked) as e:
            _fetch(f'https://{host}/x.png')
        assert e.value.reason == 'ip_literal'


# --- non-public resolved addresses (is_global) ----------------------------

@pytest.mark.parametrize('ip', [
    '10.0.0.1',            # private
    '::ffff:10.0.0.1',     # ipv4-mapped private
    '100.64.0.1',          # CGNAT: not private, not global
    '169.254.169.254',     # link-local (cloud metadata class)
    '127.0.0.1',           # loopback
    'fc00::1',             # ULA
    '224.0.0.1',           # multicast
    '::',                  # unspecified
])
def test_resolved_non_public_blocked(patched, ip):
    patched['resolve'] = {'evil.example': _addrinfo(ip)}
    with pytest.raises(EgressBlocked) as e:
        _fetch('https://evil.example/x.png')
    assert e.value.reason == 'non_public_address'


def test_one_bad_address_among_many_blocks(patched):
    # A public A record plus a loopback A record → blocked (ALL must pass).
    patched['resolve'] = {'rebind.example': _addrinfo('93.184.216.34', '127.0.0.1')}
    with pytest.raises(EgressBlocked) as e:
        _fetch('https://rebind.example/x.png')
    assert e.value.reason == 'non_public_address'


def test_all_public_addresses_pass(patched):
    patched['resolve'] = {'multi.example': _addrinfo('93.184.216.34', '1.1.1.1')}
    resp = _fetch('https://multi.example/x.png')
    assert resp.status == 200
    # pinned to the FIRST checked address
    assert patched['calls'][0]['sockaddr'][0] == '93.184.216.34'


# --- content-type / oversize ----------------------------------------------

def test_wrong_content_type_blocked(patched):
    patched['response'] = (200, 'text/html', None, b'<html>')
    with pytest.raises(EgressBlocked) as e:
        _fetch('https://example.org/x.png')
    assert e.value.reason == 'content_type'


@pytest.mark.parametrize('ct', ['image/png', 'font/woff2', 'text/css',
                                'application/font-woff', 'application/x-font-ttf'])
def test_allowed_content_types_pass(patched, ct):
    patched['response'] = (200, ct, None, b'data')
    assert _fetch('https://example.org/x.png').content_type == ct


def test_oversize_body_blocked(patched):
    # _perform_request reads max_bytes + 1; a body over the cap trips too_large.
    big = b'x' * (egress.DEFAULT_MAX_BYTES + 1)
    patched['response'] = (200, 'image/png', None, big)
    with pytest.raises(EgressBlocked) as e:
        _fetch('https://example.org/x.png')
    assert e.value.reason == 'too_large'


# --- redirects (Variante A: followed here, each hop re-gated) --------------

def test_redirect_followed_through_gate(patched):
    patched['resolve'] = {
        'example.org': _addrinfo('93.184.216.34'),
        'cdn.example': _addrinfo('1.1.1.1'),
    }

    def responder(n):
        if n == 1:
            return (302, '', 'https://cdn.example/final.png', b'')
        return (200, 'image/png', None, b'FINAL')

    patched['response'] = responder
    resp = _fetch('https://example.org/x.png')
    assert resp.body == b'FINAL'
    assert resp.final_url == 'https://cdn.example/final.png'
    assert len(patched['calls']) == 2


def test_redirect_to_private_blocked(patched):
    patched['resolve'] = {
        'example.org': _addrinfo('93.184.216.34'),
        'internal.example': _addrinfo('127.0.0.1'),
    }

    def responder(n):
        if n == 1:
            return (302, '', 'https://internal.example/x.png', b'')
        return (200, 'image/png', None, b'nope')

    patched['response'] = responder
    with pytest.raises(EgressBlocked) as e:
        _fetch('https://example.org/x.png')
    assert e.value.reason == 'non_public_address'


def test_redirect_loop_capped(patched):
    # Every hop redirects to itself → too_many_redirects, not infinite.
    patched['response'] = (302, '', 'https://example.org/x.png', b'')
    with pytest.raises(EgressBlocked) as e:
        _fetch('https://example.org/x.png')
    assert e.value.reason == 'too_many_redirects'
    assert len(patched['calls']) == egress.MAX_REDIRECTS + 1


def test_redirect_without_location_blocked(patched):
    patched['response'] = (302, '', None, b'')
    with pytest.raises(EgressBlocked) as e:
        _fetch('https://example.org/x.png')
    assert e.value.reason == 'redirect_no_location'


def test_non_2xx_status_blocked(patched):
    patched['response'] = (404, 'text/html', None, b'nope')
    with pytest.raises(EgressBlocked) as e:
        _fetch('https://example.org/x.png')
    assert e.value.reason == 'status'


# --- dns error -------------------------------------------------------------

def test_unresolvable_host_blocked(patched):
    with pytest.raises(EgressBlocked) as e:
        _fetch('https://does-not-resolve.example/x.png')
    assert e.value.reason == 'dns_error'
