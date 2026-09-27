# Mac-Dev-Setup (Apple Silicon)

Lokal-Dev des CONVERTER-Stacks auf einem Apple-Silicon-Mac via Docker Desktop.
Server-Setup (Mintbox + `converter.smallpieces.de`) ist davon unberuehrt.

## Vorbedingungen

- Apple Silicon (M1+), getestet auf M4.
- Docker Desktop installiert und gestartet (getestet mit 28.5.2).
- Repo-Clone unter `/Users/olivergluth/CODE/CONVERTER` (oder Pfade entsprechend anpassen).

## Secrets vom Server kopieren

Beide Files sind gitignored und muessen aus dem Mintbox-Setup uebernommen werden:

- `.env` → ins Repo-Root.
- `google-credentials.json` → ins Repo-Root.

Quelle kennt Oliver (1Password / Mintbox-scp).

⚠️ Seit SEC-REDIS-AUTH (2026-09-27) braucht die lokale `.env` einen `REDIS_PASSWORD` — ohne ihn bricht `docker compose` mit „REDIS_PASSWORD fehlt in .env" ab (gewollt, fail-closed). Lokal reicht ein eigener Wert: `python3 -c 'import secrets; print("REDIS_PASSWORD=" + secrets.token_urlsafe(32))' >> .env`.

## Start

```
docker compose up --build
```

`docker-compose.override.yml` wird automatisch zusaetzlich geladen. Es setzt
`platform: linux/amd64` (Playwright-Base-Image ist amd64-only, laeuft via
Rosetta-Emulation), entschaerft das external `notion-mcp-net` auf lokal-bridge
und schaltet `restart` auf `no` — damit ein crashender Container lokal nicht in
Endlosschleife restartet.

Erster Build dauert via Rosetta-Emulation ~5–10 min. App danach auf
`http://localhost:5656`.

Login über `http://localhost` braucht keinen Schalter: das Session-Cookie ist nur hinter HTTPS `Secure` (SEC-AUDIT, `HttpsOnlySecureSessionInterface` in `app_pkg/__init__.py`).

Web und Worker laufen seit SEC-NONROOT als uid 1000 (`converter`): die Volumes eines frischen Stacks gehören 1000, weil das Image die Mount-Punkte so anlegt, und `docker exec` geht nie mit `-u 0` in diese Container — was root in `/app/data` oder `/app/output_podcasts` anlegt, kann der Prozess danach nicht mehr anfassen.

## Was funktioniert / was nicht

Funktioniert: Markdown→PDF, Document→Markdown, Audio-Transkription, Library.

Nicht funktional auf Mac: Notion-Integration (kein lokaler `notion-mcp-server`,
Hostname loest nicht auf). App startet trotzdem, Notion-Routes laufen ins Leere.
