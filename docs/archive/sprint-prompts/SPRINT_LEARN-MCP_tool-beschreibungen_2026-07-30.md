# SPRINT LEARN-MCP — die Tool-Beschreibungen sagen wieder die Wahrheit

**Größe**: S (2 Phasen) · **Datum**: 2026-07-30 · **⚠️ Anderes Repo, andere Maschine**: `converter-mcp`, nicht CONVERTER

## Warum

Am 29.07. hat ein Agent über den `converter-mcp` einen ausführlichen Fehlerbericht über CONVERTERs Lern-Layer geschrieben. Zwei seiner drei Befunde waren falsch — und zwar **nicht**, weil er schlecht gearbeitet hätte, sondern weil die Tool-Beschreibung ihn belogen hat:

> `review_state` … due_cards are the full cards currently due (due <= now), **soonest-due first**

Das stimmt seit **LEARN-UP (2026-07-18)** nicht mehr. Die Queue ist FSRS-Retrievability-aufsteigend mit Zufalls-Tiebreak, neue Karten werden per Fractional-Merge eingestreut — und sie ist **tages-gedeckelt**. Die Beschreibung erwähnt weder die Caps noch die Felder, die genau davon erzählen. Der Agent hat daraufhin „die Queue liefert eine zufällige Teilmenge und `due_count` lügt" gemeldet, obwohl `remaining_today` **direkt daneben im selben Payload** stand und die vermisste Zahl exakt liefert.

Eine stale Beschreibung erzeugt falsche Fehlerberichte. Das ist der ganze Sprint.

## Arbeitsort

`converter-mcp` lebt **nur** auf der Mintbox — ein einziger Clone, **kein** git-Remote. Vom Mac aus erreichbar über den Mount:

```
/Volumes/MintHome/CODE/converter-mcp/server.py
```

Dort arbeiten und dort committen (Push gibt es nicht, es gibt kein Remote). ⚠️ Der Mount zeigt einen spurious 0-change-dirty-tree — **scoped `git add <datei>`**, niemals `git add -A` (Memory `reference_two_clone_coordination_mac_mintbox`).

**CONVERTER selbst wird in diesem Sprint nicht angefasst.** Kein Code, keine Tests, keine Doku dort.

---

# Phase 1 — Die vier stale Stellen

Alle vier sitzen in **Read**-Tools. Die Write-Tools sind aktuell — `create_card`/`update_card` kennen `front_svg`/`back_svg` bereits (CARD-SVG-Wrap gepflegt), die lässt du in Ruhe.

## 1.1 `review_state` (server.py ~881) — der wichtigste

Drei Fehler in vier Zeilen:

1. **„soonest-due first" ist falsch.** Richtig: `smart` (Default) sortiert bereits bewertete Karten nach **FSRS-Retrievability aufsteigend** (die wackligsten zuerst) mit Zufalls-Tiebreak; brandneue Karten werden gleichmäßig eingestreut, nie vorneweg. `random` ist ein Voll-Shuffle. Die Ordnung ist eine **Nutzer-Einstellung** (`ordering_mode`).
2. **Die Tages-Caps fehlen komplett.** `due_count`/`review_count`/`new_count` beschreiben die **gedeckelte** Queue, nicht die Fälligkeitsmenge. Das muss dastehen, sonst liest es jeder Agent wieder als „alles, was fällig ist".
3. **Sechs Felder fehlen im `Returns:`** — `review_count`, `new_count` (LEARN-UP) sowie `remaining_today`, `next_ahead`, `day_end` (LEARN-MORE).

Insbesondere **`remaining_today` muss erklärt werden**: wie viele jetzt fällige Karten der Cap zurückgehalten hat. Ein Satz, der die Fehldiagnose von vornherein unmöglich macht, etwa sinngemäß: *„`due_count` + `remaining_today` = die volle Fälligkeitsmenge; eine Differenz ist der Tages-Cap, kein Fehler."*

## 1.2 `list_collections` (~913)

`Returns:` listet `id, name, description, card_count, created_at` — **`due_count` fehlt**, obwohl es geliefert wird. Und es ist die **rohe** Fällig-Zahl (`due <= now`, **vor** allen Tages-Limits).

⚠️ Damit heißt `due_count` an zwei Tools **dasselbe und bedeutet Verschiedenes** — hier roh, in `review_state` gedeckelt. Genau diese Kollision hat den Bericht getragen („die Summe stimmt hinten und vorne nicht"). **Beide** Beschreibungen müssen den Unterschied benennen und aufeinander verweisen.

## 1.3 `get_card` (~869)

Die Feldliste nennt `front_svg`/`back_svg` nicht (CARD-SVG). Ergänzen, mit dem Hinweis, dass sie **server-seitig sanitisiert** ankommen und der Konsument nicht nachfiltert.

## 1.4 `list_cards` (~848)

Die Feldliste ist gegen `_card_summary` zu prüfen. Bekannt: **die Zusammenfassung führt bewusst kein `front_svg`/`back_svg`** (Listen-Response bleibt schlank) — das gehört als Satz hinein, sonst sucht ein Agent die Figuren an der falschen Stelle und schließt, es gäbe keine.

## 1.5 Die Verifikations-Regel für diese Phase

**Schreib keine Feldliste aus dem CONVERTER-Quelltext ab und keine aus meiner Aufzählung.** Ruf jedes der vier Tools **einmal live** gegen Prod auf und lies die Schlüssel aus der echten Antwort. Genau dieser Schritt fehlte, als die Beschreibungen das letzte Mal auseinanderliefen. Im Bericht pro Tool die tatsächlich beobachteten Schlüssel auflisten.

## Stop
Vier Beschreibungen korrigiert, jede gegen eine Live-Antwort verifiziert. **Commit** (lokal, kein Push — kein Remote). Dann warten.

---

# Phase 2 — Deploy + Gegenprobe

- `server.py` ist **ins Image gebacken** → auf der Mintbox `docker compose up -d --build`, **nicht** `restart`.
- Danach die vier Tools erneut aufrufen und bestätigen, dass die neuen Beschreibungen ankommen.
- **Die eigentliche Abnahme**: lies die neue `review_state`-Beschreibung so, wie ein fremder Agent sie läse. Wenn daraus noch immer die Frage „warum sind weniger Karten da als fällig?" entstehen kann, ist sie nicht fertig.
- Kurzer Eintrag in `CLAUDE.md`/`README.md` des converter-mcp: **Tool-Beschreibungen sind Vertrag** und wandern mit, wenn sich CONVERTERs Antwortform ändert.

**Rückmeldung an den CONVERTER-Master** für den dortigen Backlog-Eintrag: welche vier Beschreibungen geändert wurden und was live gegengeprüft ist.

## Nicht-Ziele

- **Kein** CONVERTER-Code, **keine** CONVERTER-Doku, **keine** CONVERTER-Tests.
- **Kein** neues Tool, keine geänderte Signatur, kein geändertes Verhalten — **nur** Beschreibungstexte.
- **Kein** Anfassen der Write-Tools (sie sind aktuell).
