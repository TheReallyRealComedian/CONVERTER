"""SEC-SSRF: Default-deny-Interception für den PDF-Renderer.

``install_pdf_egress(page)`` hängt einen ``page.route``-Handler ein, der die
Anfrage NIE durchreicht: jede Subressource, die das headless-Chromium beim PDF-Bau
laden will, wird entweder aus ``services.egress.fetch_public_https`` (ein https-Fetch
durchs Tor) **erfüllt** oder **abgebrochen**. Ein Budget je Dokument deckelt Anzahl,
Gesamt-Bytes und Wanduhr; jenseits des Budgets wird alles abgebrochen. Zusammen mit
dem toten Ausgangs-Proxy am Browser (``PDF_BROWSER_ARGS`` in ``app_pkg/markdown``)
hat alles, was der Handler nicht erfüllt, gar keinen Netzweg.

Der Handler ruft bewusst nie ``route`` weiter — der Sentinel in
``tests/test_pdf_egress.py`` prüft, dass das entsprechende Wort im Modul fehlt.
"""
import asyncio
import logging
import time

from services.egress import EgressBlocked, fetch_public_https

log = logging.getLogger(__name__)

# Gürtel: toter Ausgangs-Proxy (im Container lauscht nichts auf :9), und
# ``<-loopback>`` nimmt Loopback aus der impliziten Bypass-Liste — sonst holte
# Chromium 127.0.0.1 direkt am Proxy vorbei (gemessen Phase 1). Was der Handler
# nicht erfüllt, kann der Browser nirgends holen.
PDF_BROWSER_ARGS = [
    '--proxy-server=http://127.0.0.1:9',
    '--proxy-bypass-list=<-loopback>',
]

_MAX_REQUESTS = 64
_MAX_TOTAL_BYTES = 20 * 1024 * 1024
_MAX_WALL_SECONDS = 20.0


class _Budget:
    """Budget je Dokument. Nur aus dem Event-Loop-Thread berührt (das Tor läuft
    in ``asyncio.to_thread``, aber ``reserve``/``record`` laufen im Loop) → kein Lock."""

    def __init__(self):
        self.requests = 0
        self.total_bytes = 0
        self._start = time.monotonic()

    def reserve(self):
        """Reserviert eine Anfrage. Gibt ``None`` frei oder ein Grund-Kürzel."""
        if time.monotonic() - self._start > _MAX_WALL_SECONDS:
            return 'time'
        if self.requests >= _MAX_REQUESTS:
            return 'requests'
        if self.total_bytes >= _MAX_TOTAL_BYTES:
            return 'bytes'
        self.requests += 1
        return None

    def record(self, n):
        self.total_bytes += n


def _make_handler(budget):
    async def handler(route):
        request = route.request
        url = request.url
        over = budget.reserve()
        if over is not None:
            log.warning('PDF egress budget_exhausted (%s) url=%s', over, url)
            await route.abort('blockedbyclient')
            return
        headers = await request.all_headers()
        accept = headers.get('accept', '*/*')
        user_agent = headers.get('user-agent', '')
        try:
            resp = await asyncio.to_thread(
                fetch_public_https, url, accept=accept, user_agent=user_agent)
        except EgressBlocked as blocked:
            log.warning('PDF egress blocked reason=%s url=%s', blocked.reason, url)
            await route.abort('blockedbyclient')
            return
        except Exception as exc:  # ein Fetch-Bug darf den Render nie festfahren
            log.warning('PDF egress error (%s) url=%s', type(exc).__name__, url)
            await route.abort('blockedbyclient')
            return
        budget.record(len(resp.body))
        await route.fulfill(
            status=resp.status,
            headers={'content-type': resp.content_type},
            body=resp.body,
        )
    return handler


async def install_pdf_egress(page):
    """Hängt den Default-deny-Handler ein. MUSS vor ``set_content`` laufen."""
    budget = _Budget()
    await page.route('**/*', _make_handler(budget))
