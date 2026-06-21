# services/kindle_service.py
"""Send a library element to a Kindle via the Send-to-Kindle email service.

The only programmatic path to a Kindle is email: an EPUB attachment mailed to
the device's ``<name>@kindle.com`` from an Amazon-approved sender. There is no
public Send-to-Kindle API.

Config is env-only (single-user app, like ``CARD_TOKEN``): fail-closed when
unset. The recipient is **server-fixed** (``KINDLE_TO_EMAIL``) — never read from
a request, so the endpoint can't become an open mail relay. Env is read per
call (rotation-friendly); the SMTP password is never logged.
"""
import os
import smtplib
from email.message import EmailMessage


# Port 465 → implicit SSL (the documented default); 587 (or anything else) →
# plaintext connect upgraded via STARTTLS.
_DEFAULT_PORT = 465
_SMTP_TIMEOUT = 20  # seconds — don't let a dead SMTP host hang the request

_REQUIRED_KEYS = (
    'KINDLE_SMTP_HOST',
    'KINDLE_SMTP_USERNAME',
    'KINDLE_SMTP_PASSWORD',
    'KINDLE_FROM_EMAIL',
    'KINDLE_TO_EMAIL',
)


def is_configured() -> bool:
    """True only when every required SMTP/address env var is set (fail-closed)."""
    return all(os.environ.get(key) for key in _REQUIRED_KEYS)


def send_to_kindle(filename: str, epub_bytes: bytes, subject: str) -> None:
    """Mail ``epub_bytes`` to the server-fixed Kindle address.

    The recipient comes from ``KINDLE_TO_EMAIL`` — never from the caller. SMTP
    and timeout errors propagate so the endpoint can map them to 502.
    """
    host = os.environ.get('KINDLE_SMTP_HOST', '')
    port = int(os.environ.get('KINDLE_SMTP_PORT') or _DEFAULT_PORT)
    username = os.environ.get('KINDLE_SMTP_USERNAME', '')
    password = os.environ.get('KINDLE_SMTP_PASSWORD', '')
    from_email = os.environ.get('KINDLE_FROM_EMAIL', '')
    to_email = os.environ.get('KINDLE_TO_EMAIL', '')

    msg = EmailMessage()
    msg['From'] = from_email
    msg['To'] = to_email  # server-fixed; never request-controlled (anti-relay)
    msg['Subject'] = subject
    msg.set_content('Von CONVERTER an deinen Kindle gesendet.')
    msg.add_attachment(
        epub_bytes,
        maintype='application',
        subtype='epub+zip',
        filename=filename,
    )

    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=_SMTP_TIMEOUT) as smtp:
            smtp.login(username, password)
            smtp.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=_SMTP_TIMEOUT) as smtp:
            smtp.starttls()
            smtp.login(username, password)
            smtp.send_message(msg)
