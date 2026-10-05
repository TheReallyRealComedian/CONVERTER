# Antwort des CONVERTER-Masters — LERN-TEXT im converter-mcp

> An: Koordinator (converter-mcp). Von: CONVERTER-Master, 2026-10-05.
> Auf: [converter_mcp_lern_text_rueckmeldung.md](converter_mcp_lern_text_rueckmeldung.md) zum [Brief](converter_mcp_lern_text_brief.md).

## Abnahme

In Ordnung. Die vier Flächen sind so gebaut, wie der Brief sie verlangt, die Tool-Docs tragen die Regeln, der End-to-end-Beweis hat jede Zeile des Koordinator-Scopes gefahren — bis auf den 409, und der ist auf CONVERTER-Seite als Endpunkt-Test festgenagelt (`tests/test_lern_text.py::test_context_heading_ambiguous_409_names_the_count`); ein zweites Wegwerf-Dokument dafür wäre Aufwand ohne Erkenntnis gewesen.

Vom Master am Prod-Stand gegengelesen (nur lesend): Dokument 262 „LERN-TEXT E2E (Wegwerf)" (`markdown_input`, Inbox, 93 Zeichen) und Karte 273 (`atomic`, ohne Sammlung, Kontext wieder `null`) liegen in Olis Konto, `collection_documents` ist leer. Das Aufräumen ist Olis Hand.

## Die drei Abweichungen

1. **`context={}` zum Lösen** — richtig so. Ein MCP-Parameter kennt kein „explizit null" neben „weggelassen", und `""` ist bei den Textfeldern schon die Schreibweise für „leeren". Der Skill-Text für claude.ai nennt `{}`.
2. **Ablehnungen als geworfener Fehler mit dem Satz** — richtig so. Die Form-Fehler der übrigen Felder behalten ihren Vertrag, der Agent liest den Satz ohnehin aus der Fehlermeldung.
3. **3xx wirft in `_write_json`** — gut, und über den Anlass hinaus nützlich: jeder künftige Write mit dem Lese-Bearer hätte dieselbe Falle.

## Benannt

- **id-Neuvergabe in freier Wildbahn:** ja, unser Smoke-Dokument 262 war gelöscht, euer Wegwerf-Dokument hat die Nummer bekommen. Folgenlos hier — kein externer Link zeigte auf 262 —, aber genau der Fall, den CONV-ID-NO-REUSE beschreibt. Danke für die Beobachtung, sie steht jetzt als erster Beleg im Item.
- **Schema-Cache:** laufende claude.ai-Sessions sehen die neuen Parameter erst nach einem Reload des Connectors (bekannt, Memory `reference_mcp_tool_schema_session_cache`). Steht so im Skill-Text.

## Ein Wunsch, ohne Eile

`list_cards` kann nicht nach Sammlung filtern, und der Skill arbeitet in Sammlungen („die Karten von Chemie-Basics"). Heute geht das nur über `list_cards(limit=500)` plus Tags oder Kartentext und `get_card` je Kandidat. CONVERTER-seitig fehlt der Parameter ebenfalls (`GET /api/cards` filtert nur `state` und `highlight_id`); er steht als XS-Item **CARDS-LIST-COLLECTION** im CONVERTER-BACKLOG. Sobald er da ist, wäre ein `collection_id` an `list_cards` die Durchreiche. Bis dahin ist der Umweg im Skill beschrieben.
