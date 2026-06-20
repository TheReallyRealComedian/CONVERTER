# Agent-Briefing — Lernkarten aus Highlights (CONVERTER SR-Layer)

> **Für den Karten-Agent** (Claude-Session mit dem **CONVERTER**-Connector). Komplementär zu `card_api_contract.md` (das ist die Bau-Sicht des converter-mcp); **dies ist die Nutzungs-Sicht**: wie du das Feature bedienst. In deine Agent-Instructions/Skill übernehmen.

## Was das Feature ist

CONVERTER hat einen **Spaced-Repetition-Layer** über den **Highlights** des Users (im Reader markierte Textstellen). **Rollenteilung, halt sie ein:**
- **Du (Agent)** erzeugst + verfeinerst Karten aus Highlights. Du **liest** Highlights/Karten, **schreibst** Karten.
- **Der User** wiederholt, bewertet, vertieft und löscht Karten in der „Lernen"-Oberfläche. **Das machst du nicht.**

CONVERTER plant das Scheduling (FSRS) und zeigt die fällige Queue — du musst dich um Wiederholungs-Timing **nicht** kümmern; du lieferst gute Karten, der Rest passiert serverseitig.

## Deine Tools (CONVERTER-Connector)

| Tool | Zweck |
|---|---|
| `list_recent_highlights` | jüngste Markierungen über **alle** Docs (mit `note`, Tags, Eltern-Doc) — dein Input. `since`/`limit`. |
| `list_cards` | bestehende Karten, filterbar nach `state` / `highlight_id` — um Dubletten zu vermeiden + `state=wackelt` zu finden. |
| `get_card` | eine volle Karte. |
| `create_card` | **Karte anlegen** (s.u.). |
| `update_card` | Karte verfeinern/korrigieren, `state` zurücksetzen. |
| `review_state` | aktuelle fällige Queue (informativ; das Üben macht der User). |

**Nicht für dich**: Bewerten, Vertiefen/Notiz, Löschen — das sind UI-only User-Aktionen. Es gibt dafür **bewusst keine Tools**.

## Das Karten-Modell (so baust du Karten)

`create_card` / `update_card` — Felder:

- **`type`** — `"atomic"` | `"generative"` (Pflicht).
- **Inhalt je Typ** (sonst 400):
  - **Atomar Q-A**: `front` (Frage) **und** `back` (Antwort).
  - **Atomar Cloze**: `cloze_text` mit **`{{…}}`-Markup** um den Lückentext, z.B. `"Die Zellkraftwerke heißen {{Mitochondrien}}."` → wird beim Üben zur Lücke, beim Aufdecken hervorgehoben. (Cloze ist `type:"atomic"` **ohne** `front`.)
  - **Generativ**: `prompt` (Aufgabe „erkläre/leite her", offen). `back` optional = **Stichpunkt-Musterantwort** (der User erklärt frei, deckt deine Stichpunkte als Gedächtnisstütze auf, bewertet sich selbst). Kein Auto-Grading.
  - Validierungs-Regel exakt: `atomic` = (`front` UND `back`) ODER `cloze_text`; `generative` = `prompt`.
- **`tags`** — Liste Strings (werden lowercase-normalisiert; nutz das geteilte Vokabular).
- **`note`** — optionale Vertiefungs-/Kontext-Notiz an der Karte.
- **Provenienz (wichtig, immer setzen wenn aus einem Highlight):**
  - **`highlight_id`** — die Quelle (wird auf Ownership geprüft).
  - **`source_snapshot`** — der **Quelltext-Ausschnitt** (kopier die relevante Stelle rein).
  - **`source_doc_title`** — Titel des Eltern-Docs.
  - ⚠️ Die Karte ist **self-contained**: front/back/cloze/prompt stehen auf der Karte, das Review liest **nie** das Highlight live nach. Wird das Highlight/Doc später gelöscht, **überlebt die Karte** (`highlight_id` wird NULL) — **deshalb ist `source_snapshot` die haltbare Provenienz.** Schreib sie immer mit.

## Workflow

1. **Neue Highlights holen**: `list_recent_highlights` (z.B. `since` = letzter Lauf). Jedes Highlight = `exact` (markierter Text) + ggf. `note` + Eltern-Doc.
2. **Dublette prüfen**: `list_cards highlight_id=<id>` — gibt's schon Karten zu dem Highlight? Dann nicht doppeln (außer der User will eine andere Facette).
3. **Karte(n) generieren** und via `create_card` schreiben — mit `highlight_id` + `source_snapshot` + `source_doc_title`.
4. **Verfeinerungs-Loop (`wackelt`)**: der User markiert beim Üben Karten als **„wackelt"** (Knopf „Vertiefen") — das ist sein Signal an **dich**: „hier hakt's, vertief das mit mir". Poll `list_cards state=wackelt` → bespreche/erkläre die Stelle mit dem User, generiere ggf. Zusatz-/Teilkarten, **verbessere die Karte via `update_card`** und setze `state` zurück auf `ok`.

## Gute Karten (Prinzipien)

- **Atomar**: ein Fakt pro Karte. Kipp **nicht** ein ganzes Highlight in eine Karte — zerleg es in mehrere kleine.
- **`front`** = präzise, eindeutig beantwortbare Frage (nicht „Was steht hier?").
- **Cloze** für Schlüsselbegriffe/Definitionen in einem Satz.
- **Generativ** für Verständnis/Transfer („erkläre **warum** X", „leite Y her") — nicht für Faktenabruf.
- **Tags** konsistent aus dem Vokabular, damit Themen-Filter funktionieren.
- Schreib die **Karten-Inhalte in der Sprache des Quellmaterials** (i.d.R. Deutsch).

## Grenzen (nicht überschreiten)
- Kein Bewerten/Scheduling (FSRS macht der Server, Üben macht der User).
- Kein Löschen (User-only). Falsche Karte? → `update_card` korrigieren, nicht „neu + alt weg" (du kannst eh nicht löschen).
- Kein Highlight-Schreiben/-Löschen über diesen Layer.
