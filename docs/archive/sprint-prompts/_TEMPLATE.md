# Sprint <CODE> — <Kurz-Titel>

**Datum**: YYYY-MM-DD

**Ziel**: <1-2 Sätze: was wird erreicht, warum jetzt.>

**Vorbedingung**: <Stand der Welt — was ist live, von wo arbeiten wir, welche Sprints sind vorgelagert.>

**Out-of-scope**: <Was bewusst nicht angefasst wird — verhindert Scope-Creep. Andere offene Sprints, Architektur-Themen die später kommen, Refactor-Folgen.>

---

## Phase 0 — JETZT · Bestandsaufnahme (read-only) · OPTIONAL

> **CONVERTER-Default: weglassen.** Phase 0 ist die Ausnahme, nicht die Regel. Drinhalten **nur** wenn vor der Implementierung eine echte Mechanik-Wahl offen ist (z.B. „neuer Helper oder bestehenden erweitern", „Backend-Validation-Layer wo", „Schema-Touch ja/nein"). Bei klar geschnittenen Sprints (englische Strings übersetzen, einzelner Bugfix, ein Pattern-Cluster mit klarer Vorlage) **direkt mit Phase 1 starten**.

Wenn drin, dann:

1. Pre-Flight: `pytest tests/` grün?
2. **Code-Audit**: <relevante Files lesen, Service-Signaturen verstehen, Side-Effects durchspielen.>
3. **Mechanik-Konzept**: <Optionen identifizieren, eigene Default-Empfehlung pro offener Frage formulieren.>
4. STOP — Bericht an Master mit Mechanik-Fragen, jede mit Default-Empfehlung.

Kein Code-Edit in dieser Phase.

---

## Phase 1 — Implementation

Pre-Flight: `pytest tests/` muss grün sein.

Erwartete Files:

```
<pfad/file>          # NEU/EDIT — kurz was passiert
<pfad/file>          # NEU/EDIT — kurz was passiert
```

Code-Quality-Gates:

- `pytest tests/` grün lokal.
- UI-Strings deutsch (siehe CLAUDE.md Code-Stil).
- Helper aus `static/js/_utils.js` (`showAlert`, `showToast`, `formatFileSize`) wiederverwenden, nicht reimplementieren.
- Keine `alert()`-Calls in Frontend-JS — `showAlert`/`showToast` benutzen.
- Bei Template-Änderungen: Live-Smoke (Browser auf `localhost:5656`) erforderlich, da Test-Suite Templates nicht rendert.

---

## Phase 2 — Verify

1. `pytest tests/` grün.
2. <Smoke-Test des Hauptpfads — Browser oder `curl`.>
3. <Edge-Cases die in Phase 0 oder Phase 1 aufkamen.>
4. Bei Template-/Routing-Änderungen: GET der betroffenen Route(n) im laufenden Container, 200-Status verifizieren.

---

## Phase 3 — Commit

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- Ein Sprint = ein Branch, Cluster werden als separate Commits committet wenn der Sprint mehrere logische Schritte enthält.
- Push erst nach explizitem Sign-off durch Master (Default: lokal commit, Push folgt später).

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off. Master-Thread = Dispatch, Sub-Session = Execute.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**S / M / L / XL** — kurze Begründung (Schema-Touch ja/nein, UI-Touch, Migration nötig, Test-Welle).

---

## Konstitutiv mit-genommen, falls berührt

- <BACKLOG-Item das eh dranliegt und durch diesen Sprint billig wird.>
- <Defensive Hygiene-Fix der sonst übersehen wird.>

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

- ✓ Sprint <CODE> durch (Datum eintragen).
- 📋 <Follow-up-Item das Phase 1 oder 2 aufgeworfen hat> (P-Stufe, Größe).
- STATUS.md: aktueller Stand auf neuen Sprint-Output ziehen.
- BACKLOG.md: erledigte Items entfernen, neue Follow-ups einfügen.

---

## Phase-0-Entscheidungen (nur falls Phase 0 drin war)

| # | Frage | Entscheidung |
|---|-------|--------------|
| 1 | …    | …            |

Persistierung im Doc damit Re-Runs nicht neu diskutieren müssen.
