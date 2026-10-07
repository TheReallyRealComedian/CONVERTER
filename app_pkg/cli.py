"""Operator commands of the ``flask`` CLI: ``create-user``, ``set-password``
(SEC-SET-PASSWORD) and ``reset-collection`` (LEARN-BACK). Reached in the
container as ``docker exec -it markdown-converter-web flask <command>`` and by
the browser smokes as ``flask --app app create-user``; no endpoint and no UI,
by decision.

Moved verbatim out of ``app_pkg/__init__.py`` in ARCH-FACTORY P1 (function
bodies byte-equal). The one edit: the comment at the local import of
``app_pkg.cards._naive_utc`` now says what that import does — it keeps
``app_pkg.cards`` and the modules behind it out of this module's import; it
is not there to break a cycle.
"""
from datetime import datetime, timezone

import click

from models import ApiToken, Card, Collection, Review, User, db
from services.scheduler.base import initial_review_state


PASSWORD_MIN_LENGTH = 8


def _check_password_rule(password):
    """The ONE password rule — shared by ``create-user`` and ``set-password``
    (SEC-SET-PASSWORD) so the two can never drift. A violation ends the
    command with ``click.ClickException`` (exit code 1)."""
    if len(password) < PASSWORD_MIN_LENGTH:
        raise click.ClickException(
            f'Passwort muss mindestens {PASSWORD_MIN_LENGTH} Zeichen haben.')


def _register_cli_commands(app):
    @app.cli.command('create-user')
    @click.argument('username')
    @click.option('--password', prompt=True, hide_input=True, confirmation_prompt=True)
    def create_user_cmd(username, password):
        """Create a new user account."""
        _check_password_rule(password)
        if User.query.filter_by(username=username).first():
            click.echo(f'Error: User "{username}" already exists.')
            return
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        click.echo(f'User "{username}" created successfully.')

    @app.cli.command('set-password')
    @click.argument('username')
    @click.option('--revoke-tokens', is_flag=True,
                  help='Loescht danach ALLE API-Tokens dieses Benutzers '
                       '(iOS-App, converter-mcp) — sie muessen sich neu anmelden.')
    def set_password_cmd(username, revoke_tokens):
        """Setzt das Passwort EINES Benutzers neu (Operator-Werkzeug).

        SEC-SET-PASSWORD: der einzige Wechselpfad. Bewusst CLI und kein
        Endpoint, keine UI — wer das Kommando erreicht, hat Container-Zugriff
        (``docker exec -it markdown-converter-web flask set-password <user>``).
        Das Passwort wird abgefragt (verdeckt, mit Wiederholung), nie als
        Argument uebergeben — nichts landet in Shell-History oder ``ps``.

        Was das Kommando NICHT tut, sagt es am Ende selbst:

        Browser-Sessions bleiben gueltig. Sie sind signierte Cookies ohne
        Server-Zustand; der einzige Hebel ist die SECRET_KEY-Rotation.

        API-Tokens bleiben gueltig — eigene Credentials mit eigenem Widerruf.
        Ein stiller Massen-Widerruf traefe iOS-App und converter-mcp, ohne
        dass der Operator es sieht; deshalb werden die Tokens des Benutzers
        aufgelistet, und --revoke-tokens loescht sie nur auf Ansage — alle
        dieses Benutzers, nicht einzeln (einzeln: POST /api/auth/logout mit
        dem jeweiligen Token).

        Passwort und Token-Widerruf gehen in EINEM Commit: entweder beides
        oder nichts. (Docstring ohne RST-Markup: click rendert ihn als
        --help-Text und bricht Absaetze neu um.)
        """
        user = User.query.filter_by(username=username).first()
        if user is None:
            raise click.ClickException(f'Benutzer "{username}" nicht gefunden.')
        # Prompt AFTER the user check so a typo in the name fails before
        # anyone types a password twice. On a mismatch click re-prompts; at
        # EOF it aborts with exit code 1 — nothing is written either way.
        password = click.prompt('Neues Passwort', hide_input=True,
                                confirmation_prompt='Neues Passwort (Wiederholung)')
        _check_password_rule(password)

        tokens = user.api_tokens.order_by(ApiToken.id).all()
        user.set_password(password)
        if revoke_tokens:
            for token in tokens:
                db.session.delete(token)
        db.session.commit()

        click.echo(f'Passwort fuer "{user.username}" (id {user.id}) geaendert.')
        if revoke_tokens:
            if tokens:
                counts = {}
                for token in tokens:
                    key = token.label or '(ohne Label)'
                    counts[key] = counts.get(key, 0) + 1
                detail = ', '.join(f'{label} x{n}' for label, n in sorted(counts.items()))
                click.echo(f'{len(tokens)} API-Tokens widerrufen ({detail}) — '
                           f'iOS-App und converter-mcp brauchen neue Tokens '
                           f'(POST /api/auth/login).')
            else:
                click.echo('Keine API-Tokens vorhanden — nichts zu widerrufen.')
        elif tokens:
            click.echo(f'API-Tokens dieses Benutzers: {len(tokens)} — bleiben gueltig.')
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            width = max(len(token.label or '(ohne Label)') for token in tokens)
            for token in tokens:
                created = (token.created_at.strftime('%Y-%m-%d %H:%M UTC')
                           if token.created_at else '?')
                if token.expires_at is None:
                    expiry = 'laeuft nie ab'
                else:
                    expiry = f"laeuft ab {token.expires_at.strftime('%Y-%m-%d %H:%M UTC')}"
                    if token.expires_at <= now:
                        expiry += ' (abgelaufen)'
                label = (token.label or '(ohne Label)').ljust(width)
                click.echo(f'  id {token.id:<3} {label}  erstellt {created}  {expiry}')
            click.echo('Widerruf aller Tokens: dieses Kommando mit --revoke-tokens; '
                       'einzeln: POST /api/auth/logout mit dem jeweiligen Token.')
        else:
            click.echo('API-Tokens dieses Benutzers: keine.')
        click.echo('Browser-Sessions bleiben gueltig — sie enden nur mit einer '
                   'SECRET_KEY-Rotation.')

    @app.cli.command('reset-collection')
    @click.argument('collection')
    @click.option('--apply', 'apply_changes', is_flag=True,
                  help='Schreibt wirklich. Ohne den Flag wird nur berichtet.')
    def reset_collection_cmd(collection, apply_changes):
        """Setzt die bewerteten Karten EINER Sammlung auf "neu" zurueck.

        COLLECTION ist der Name ODER die id der Sammlung.

        LOST-UPDATE: die Review-Zeilen sind versioniert (``Review.version``).
        Landet waehrend des Laufs eine Bewertung auf einer der Karten, bricht
        ``--apply`` mit ``StaleDataError`` ab — nichts ist geschrieben (EIN
        Commit), Lauf wiederholen. Gewollt: kein Retry fuer ein einmaliges
        Werkzeug.

        LEARN-BACK: einmalige Korrektur vergifteter Scheduling-Daten, KEIN
        wiederkehrendes Werkzeug (deshalb CLI und kein Endpoint/kein Knopf).
        Alle Bewertungen vor LEARN-RATE trugen eine andere Semantik — "Schwer"
        hiess "kaum gewusst", der Scheduler bekam ``hard`` wo ``again``
        gehoerte. Die daraus gewachsene Stabilitaet ist nicht bloss zu hoch,
        sie ist erfunden; umdatieren wuerde das Phantom-Modell mitschleppen.

        Zwei gesperrte Entscheidungen:

        * Die Zeile wird feldweise auf ``initial_review_state()`` gesetzt —
          die EINE Definition von "neu" (beide Engines geben sie aus
          ``new_card_state`` zurueck). Eine nachgebaute Feldliste liefe bei
          der naechsten Scheduler-Aenderung still auseinander.
        * ``rating_history`` wird geleert: die Eintraege tragen die ungueltige
          Semantik, ``count_done_today`` klassifiziert neu-vs-Review am ERSTEN
          Eintrag (stehengelassen zaehlte die Karte gegen das Review- statt
          das Neu-Budget), und ``true_retention`` wuerde sie noch ~30 Tage
          weiterzaehlen. Preis ist der Audit-Trail; sein Wert ist durch die
          Semantik ohnehin zerstoert.

        Ausgewaehlt werden nur Karten mit ``stability IS NOT NULL`` (bereits
        bewertet) — damit ist der Lauf per Konstruktion idempotent: der zweite
        findet 0 Zeilen. Karteninhalte, Tags und Sammlungen bleiben unberuehrt.
        """
        target, error = _resolve_collection(collection)
        if error:
            raise click.ClickException(error)

        cards = (target.cards
                 .join(Card.review)
                 .filter(Review.stability.isnot(None))
                 .order_by(Card.id)
                 .all())

        mode = 'APPLY' if apply_changes else 'DRY-RUN'
        click.echo(f'Sammlung "{target.name}" (id {target.id}, user {target.user_id}) '
                   f'— {mode}')
        click.echo(f'Bewertete Karten: {len(cards)} von {target.cards.count()} '
                   f'in der Sammlung')

        # A card can sit in several collections; it is reset when the TARGET is
        # among them (the only sensible semantics), but name the overlap so a
        # surprise shows up before the --apply.
        overlaps = {}
        for card in cards:
            for other in card.collections:
                if other.id != target.id:
                    overlaps[other.name] = overlaps.get(other.name, 0) + 1
        if overlaps:
            detail = ', '.join(f'"{name}" ({count})'
                               for name, count in sorted(overlaps.items()))
            click.echo(f'Davon auch in anderen Sammlungen: {detail} '
                       f'— sie werden mit zurueckgesetzt.')

        if not cards:
            click.echo('Nichts zu tun.')
            return
        if not apply_changes:
            click.echo('Nichts geschrieben (Dry-run). Mit --apply ausfuehren.')
            return

        # `initial_review_state()` hands back AWARE UTC, the column convention
        # is naive UTC. SQLite drops the tzinfo silently and stores the wall
        # clock, which today happens to BE the naive UTC — right for the wrong
        # reason. Route `due` through the same `_naive_utc` every other write
        # path uses, so a future zone change in the scheduler cannot make this
        # one path write Berlin wall-clock as UTC. (`last_reviewed` is None;
        # `_naive_utc(None)` returns None, so the loop stays as it is.)
        # Local import on purpose — NOT because of a cycle: a top-level import
        # resolves in every production entry (ARCH-AUDIT: eight import orders,
        # suite green). It keeps app_pkg.cards and the ~95 modules behind it
        # (learn, library, the renderer) out of everything that imports THIS
        # module.
        from app_pkg.cards import _naive_utc
        fresh = initial_review_state()
        fresh['due'] = _naive_utc(fresh['due'])
        for card in cards:
            for field, value in fresh.items():
                setattr(card.review, field, value)
            card.review.rating_history = None
        db.session.commit()
        click.echo(f'{len(cards)} Karten zurueckgesetzt — neu und sofort faellig.')


def _resolve_collection(raw):
    """Resolve a collection by id OR name → ``(collection, error_message)``.

    The id is unambiguous, the name is what a human types at the prompt, so
    both are accepted. Names are per-user unique but not globally, so a name
    hitting several users is reported as ambiguous rather than guessed.
    """
    if raw.isdigit():
        found = Collection.query.filter_by(id=int(raw)).first()
        if found is not None:
            return found, None
    name = Collection.normalize_name(raw)
    matches = Collection.query.filter_by(name=name).all() if name else []
    if not matches:
        return None, f'Sammlung "{raw}" nicht gefunden (weder als id noch als Name).'
    if len(matches) > 1:
        owners = ', '.join(f'id {c.id} (user {c.user_id})' for c in matches)
        return None, (f'Sammlung "{name}" ist mehrdeutig: {owners}. '
                      f'Bitte die id angeben.')
    return matches[0], None
