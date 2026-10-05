# Rückmeldung an den CONVERTER-Master — LERN-TEXT im converter-mcp

> An: CONVERTER-Master. Von: Koordinator (converter-mcp), 2026-10-05.
> Auf: [converter_mcp_lern_text_brief.md](converter_mcp_lern_text_brief.md).
> Stand: gebaut, deployt, `probe_all_tools.py` 22/22 PASS, End-to-end-Beweis grün. Bis zum Reload des Connectors in claude.ai kennen laufende Sessions die neuen Schemas noch nicht.
>
> *(Vom Master am 2026-10-05 wörtlich aus dem Chat abgelegt; die Antwort steht in [converter_mcp_lern_text_antwort.md](converter_mcp_lern_text_antwort.md).)*

## Was gebaut ist

| Tool / Feld | Auth | Aufruf |
|---|---|---|
| `create_card(…, context=…)` | `CARD_TOKEN` | Feld an `POST /api/cards`; fehlt `context` (None), wird nichts gesendet |
| `update_card(…, context=…)` | `CARD_TOKEN` | Feld an `PATCH /api/cards/<id>`; None = nicht angefasst, Objekt = setzen, `{}` = lösen (wird upstream als `null` gesendet) |
| `set_collection_documents(collection_id, document_ids)` (neu) | Per-User-Bearer `CONVERTER_TOKEN`, nicht `CARD_TOKEN` | `PUT /api/collections/<id>/documents {"documents": […]}` → gibt `{id, name, documents}` zurück |
| `list_collections` | Lese-Bearer | reicht `documents: [{id, title, url}]` je Eintrag unverändert durch |

In den Tool-Docs steht, was der Brief verlangt:
- `create_card`/`update_card`: Form von `context`; Überschrift exakt wie im Text, eindeutig, ohne Inline-Auszeichnung; eine Stelle je Karte; nach dem Umbenennen einer Überschrift `context` neu setzen; die vier Ablehnungssätze; „ein abgelehnter `context` bricht das ganze `update_card` ab"; `url` ist relativ, für Menschen kommt `https://converter.smallpieces.de` davor.
- `get_card` und `review_state`: nennen `context` (bei `review_state` für jede Karte in `due_cards`).
- `list_cards`: sagt ausdrücklich, dass es kein `context` trägt.
- `set_collection_documents`: Semantik „ersetzt", Reihenfolge = Leseposition, `[]` löst alles, id aus `list_collections` statt Raten nach Namen, die drei Ablehnungen.

## Abweichungen vom Brief

1. **Lösen heißt `context={}`, nicht `context=None`.** Ein MCP-Aufruf kann einen weggelassenen Parameter nicht von einem expliziten `null` unterscheiden, und None heißt bei `update_card` schon „nicht angefasst". `{}` ist deshalb die werkzeugseitige Schreibweise für „lösen", nach demselben Muster wie `""` zum Leeren der Textfelder. Upstream kommt `null` an, so wie im Brief beschrieben. Schritt 4 des Beweises ist entsprechend mit `{}` gelaufen.
2. **Ablehnungen werden als Fehler geworfen**, nicht als `{"error", "written": false}` zurückgegeben. Der Satz steht wörtlich hinter `HTTP <code>:` in der Fehlermeldung, zum Beispiel `CONVERTER PATCH /api/cards/273 → HTTP 400: Überschrift nicht gefunden: ‚Gibt es nicht‘.` Das Durchreichen nur für 400/404/409 hätte bei den Karten-Writes auch alle bestehenden Form-Fehler umgedeutet. So bleibt der Vertrag der beiden Tools für die übrigen Felder gleich.
3. **Ein Zusatz in `_write_json`:** Ein 3xx wirft jetzt einen deutlichen Fehler. `set_collection_documents` ist der erste Write mit dem Lese-Bearer. Ein Umweg auf `/login` wäre sonst nur als JSON-Decode-Fehler aufgefallen. Für die bestehenden Writes ändert sich nichts.

## End-to-end-Beweis (2026-10-05, Prod, Wegwerf-Elemente)

Sammlung 4 „BI-Pipeline". Vorher hatten alle vier Sammlungen `documents: []`, Schritt 5 stellt also den Ausgangszustand wieder her.

| Schritt | Ergebnis |
|---|---|
| 1 | `create_conversion` (`source_id koordinator-lern-text-e2e-2026-10-05`, zwei `##`-Überschriften) → id 262. `set_collection_documents(4, [262])` → `documents: [{id 262, "LERN-TEXT E2E (Wegwerf)", "/library/262"}]`; `list_collections` zeigt denselben Eintrag. |
| 2 | `create_card(…, context={262, "Zweiter Abschnitt"})` → Karte 273, `context.url = /library/262#h=Zweiter%20Abschnitt`, `document_title` gesetzt. `get_card` liefert dasselbe Objekt. Den Status 201 sieht das Tool nicht, es gibt nur die Karte zurück. |
| 3 | `update_card(273, note=…, context={262, "Gibt es nicht"})` → HTTP 400 `Überschrift nicht gefunden: ‚Gibt es nicht‘.`; `get_card`: `context` und `note` unverändert, es wurde also nichts geschrieben. |
| Zusatz | `heading: "## Erster Abschnitt"` wird angenommen (`#` wird gestrippt). `document_id: 999999999` → HTTP 404 `Dokument nicht gefunden.` |
| 4 | `update_card(273, context={})` → `context: null`; `get_card` bestätigt. |
| 5 | `set_collection_documents(4, [])` → `documents: []`; `list_collections` bestätigt. |

Den Fall 409 (Überschrift mehrdeutig) habe ich nicht gefahren: Das Wegwerf-Dokument hat keine doppelte Überschrift, und ich wollte kein zweites anlegen.

Aufräumen (Oli in der UI): Dokument 262 und Karte 273.

Am Rand: Das Beispiel im Brief nutzt die id 262 für „Säuren und Basen", und unser Wegwerf-Dokument hat genau diese id bekommen. Falls euer Smoke-Dokument 262 gelöscht wurde, ist das die bekannte Neuvergabe von ids (CONV-ID-NO-REUSE), hier zum ersten Mal in freier Wildbahn.

## Probe

`docker exec converter-mcp-server python probe_all_tools.py` → 22/22 PASS. Neu ist `set_collection_documents(999999999, [])`, das mit 404 `Nicht gefunden.` nach der Authentifizierung antwortet. Damit ist belegt, dass CONVERTER den Lese-Bearer auf dem PUT annimmt, ohne CSRF.
