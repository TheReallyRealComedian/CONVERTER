# Sprint R2-F — Reader-Abschluss-Leiste + Fortschritts-Klarheit (S)

> **Executor-Doc.** Phasen strikt nacheinander, nach jeder Phase **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün. UI-Strings deutsch (Microcopy-Konventionen in CLAUDE.md).

## Kontext / Warum

Live-Befund (Oliver, nach Mintbox-Deploy): *„die fortschrittsanzeige ist problematisch: nach dem fertig lesen muss ich ja wieder hoch um zurück zu gehen oder so, aber das setzt den fortschritt zurück."*

Master-Code-Read vorab (2026-06-12):
- Die furthest-read-Persistierung ist clientseitig dicht: `maxReached` wird aus `PageData.lastReadPercent` geseedet, nur vorwärts erhöht, nur `maxReached` wird gesendet ([static/js/library_detail.js:1336-1449](../../static/js/library_detail.js), R2-B). Hochscrollen *sollte* den DB-Wert nicht senken.
- **Aber**: Der Server-Endpoint `PATCH /api/conversions/<id>/progress` ([app_pkg/library.py:255-275](../../app_pkg/library.py)) überschreibt **bedingungslos** — das Vorwärts-Gate existiert nur im Client und hängt am korrekten Seeding.
- Die sichtbare Progress-Bar im Reader folgt der **aktuellen Scroll-Position** — beim Hochscrollen läuft sie zurück und *wirkt* wie ein Reset, selbst wenn die DB korrekt bleibt.
- Echtes UX-Loch: Am Dokument-Ende gibt es **keine Abschluss-Affordance** — man muss hochscrollen, um zurück zu navigieren.

Workshop-Entscheidung (2026-06-12, Oliver): **Abschluss-Leiste** am Content-Ende („Zurück" + „Archivieren") + Anzeige-Klärung. Live-Beobachtung schlägt Code-Lektüre — deshalb Phase 1 zuerst als Befund.

## Phase 1 — Live-Befund auf der Mintbox (read-only, kein Code)

Klären: Geht der **DB-Wert** beim Hochscrollen wirklich zurück, oder ist es ein Anzeige-Eindruck?

1. Repro über die UI (Mintbox-Instanz im Browser): ein Dokument bis zum Ende lesen/scrollen → hochscrollen → zur Library zurück → zeigt die Karte „Gelesen" (≥95) oder einen niedrigeren Wert? Danach das Dokument erneut öffnen + sofort verlassen → Karte erneut prüfen (Seeding-Pfad).
2. DB-Gegencheck (read-only):
   ```
   ssh mintbox
   cd /home/oliver/CODE/CONVERTER   # Pfad ggf. verifizieren
   docker compose exec -T markdown-converter python -c "
   import sqlite3; con = sqlite3.connect('/app/data/converter.db')
   for r in con.execute('SELECT id, title, last_read_percent FROM conversion_history ORDER BY id DESC LIMIT 15'): print(r)
   "
   ```
   (Tabellennamen vorher in `models.py` verifizieren statt raten.)
3. Mit zwei Browser-Caveats aus dem Repo-Gedächtnis arbeiten: rAF pausiert in Hidden-Tabs, und `visibilitychange`-Flush feuert beim Tab-Wechsel — beim Repro den Tab wirklich im Vordergrund lassen bzw. Navigation (nicht nur Tab-Wechsel) testen.

**Befund-Matrix für den Bericht**: (a) DB-Wert sinkt nie → reines Anzeige-/Affordance-Problem, Phase 2 wie geplant. (b) DB-Wert sinkt reproduzierbar → exakte Repro-Schritte dokumentieren, **Stop**: Phase 2 wird dann um den echten Bug-Fix erweitert (Master entscheidet nach Bericht).

**Stop + Bericht.**

## Phase 2 — Abschluss-Leiste + Anzeige-Klärung + Server-Härtung

Dateien: `templates/library_detail.html`, `static/js/library_detail.js`, `app_pkg/library.py` (NUR `api_update_conversion_progress`), CSS. **`templates/library.html` / `static/js/library.js` nicht anfassen** (gehören R2-E, läuft als Nachbar-Sprint).

1. **Abschluss-Leiste** am Ende des Content-Bodys (nur im Reader-Kontext mit gerendertem Markdown, dort wo die Progress-Bar aktiv ist):
   - „Zurück zur Library" — `history.back()` wenn sinnvoll möglich, sonst Link auf `url_for('library')` (einfach halten: Link mit JS-Enhancement reicht).
   - „Archivieren" — nutzt den **bestehenden** Status-Mechanismus des Detail-Views (R2-C, nichts Neues bauen): Status auf `archive` setzen, kurzes Erfolgs-Feedback (Toast aus `_utils.js`), dann zurück zur Library navigieren.
   - Buttons max 3 Wörter, deutsch.
2. **Anzeige-Klärung**: Sobald der **persistierte** furthest-read-Zustand ≥ 95 ist (`maxReached`, nicht die Scroll-Position), neben/an der Progress-Bar ein dezentes, dauerhaftes „Gelesen"-Label zeigen, das beim Hochscrollen **nicht** verschwindet. Die Bar selbst bleibt Positions-Anzeige (das ist ihr Job) — das Label entkoppelt „wo bin ich" von „wie weit war ich".
3. **Server-Forward-Clamp** in `api_update_conversion_progress`: eingehender `percent` wird gegen den gespeicherten Wert geclampt — `last_read_percent = max(bestehend, neu)`; Response liefert den effektiven Wert. Damit ist furthest-read auch serverseitig garantiert (Schutz gegen jeden künftigen Seeding-/Client-Bug). Bewusste Konsequenz: ein „Fortschritt zurücksetzen"-Feature bräuchte künftig ein explizites Flag — als Notiz ins BACKLOG, nicht bauen.
4. Tests: Clamp-Verhalten (kleinerer Wert ändert nichts, Response zeigt Bestand; größerer Wert übernimmt; Grenzen 0/100 bleiben geclampt wie bisher). Bestehende Progress-Tests anpassen.
5. **Live-Smoke mit echtem Scrollen** (Memory-Caveats beachten: echte Maus/Trackpad-Interaktion, `visibilitychange`-Flush ggf. per Direkt-Invocation prüfen, Hidden-Tab-rAF-Pause): Leiste erscheint am Ende, Zurück funktioniert, Archivieren setzt Status + navigiert, „Gelesen"-Label bleibt beim Hochscrollen stehen, dark + light.

**Stop + Bericht.**

## Phase 3 — Wrap-up

1. `STATUS.md` + `BACKLOG.md`: R2-F erledigt mit Hashes; BACKLOG-Notiz „Fortschritt-Reset bräuchte explizites Flag (Server clampt vorwärts)"; Mintbox-Deploy als Olivers offener Schritt.
2. `docs/reader_architecture.md`: kurzer Absatz — Abschluss-Leiste, „Gelesen"-Label, Server-Forward-Clamp (furthest-read jetzt doppelt garantiert).
3. **Bullet-Guard** vor dem Doc-Commit: `grep -nE '(- \*\*.*){2,}' BACKLOG.md`.
4. `pytest tests/` final grün.

**Stop + Schluss-Bericht.**

## Out of scope
- `templates/library.html`, `static/js/library.js`, Library-View-Logik in `app_pkg/library.py` — gehören R2-E (Library-IA), nicht anfassen.
- Auto-Archivieren bei 100 % (Workshop-Option „Auto-Prompt" wurde verworfen).
- Fortschritt-Zurücksetzen-Feature (nur BACKLOG-Notiz).
