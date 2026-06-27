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
| `update_highlight` | **Tags/Notiz auf einem bestehenden Highlight** setzen/ersetzen/leeren — für persistentes Bucket-Tagging (nicht pro Lauf neu ableiten). Voll-Ersetzung der Tags, geteiltes Vokabular. Anker/Marker bleiben unberührt. |
| `list_tags` | den **Themen-Wald** lesen (jedes Tag mit `parent_id` + `card_count`) — um konsistent unter bestehende Themen einzuordnen, bevor du `set_tag_parent` aufrufst. |
| `set_tag_parent` | ein Thema **in den Tag-Baum einordnen** (`{tag, parent\|null}`, by-name) — s. „Karten organisieren". |
| `list_collections` | die **Sammlungen** des Users lesen (jede mit `card_count`) — vor dem `collections`-Write, um konsistent einzuordnen. |
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
- **`collections`** — Liste Strings, optional: die **Sammlungen**, in die die Karte gehört (s. „Karten organisieren"). get_or_create **by-name**, **case-erhaltend** (Eigennamen), Voll-Ersetzung.
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

## Highlights annotieren (Bucket-Tagging)

Du kannst **Tags und eine Notiz auf bestehende Highlights zurückschreiben** (`update_highlight`) — gedacht für **persistentes Bucket-Tagging**: statt dein Themen-Bucketing pro Lauf neu abzuleiten, schreibst du es einmal aufs Highlight und liest es beim nächsten Lauf über `list_recent_highlights` (Feld `tags`) wieder ein. Die Tags sind **Voll-Ersetzung** (das übergebene Set ersetzt das bestehende; `[]` leert alle) aus dem **geteilten Vokabular** (dieselben Tag-Rows wie Karten-/UI-Tags). Die Notiz folgt der PATCH-Semantik (`""` löscht sie).

**Was hier bewusst NICHT geht**: den markierten Text bzw. seine Anker (`exact`/`prefix`/`suffix`) ändern, ein Highlight **löschen** oder ein **neues** Highlight anlegen — das bleibt User-/Reader-only.

## Karten organisieren (Gruppieren)

Du ordnest Karten jetzt auch **selbst** in die zwei Gruppierungs-Achsen ein — nicht nur der User. Beides läuft über deinen normalen Token.

**Achse B — Sammlungen** (flache, kuratierte Bündel: ein Horizont, ein Kurs, ein Themenpaket). Über das **`collections`-Feld** an `create_card`/`update_card`: eine Liste von Namen, **by-name get_or_create** (neue Sammlung wird angelegt, falls es sie nicht gibt), **Voll-Ersetzung** (`[]` entfernt die Karte aus allen Sammlungen). Namen sind **Eigennamen, case-erhaltend** („Boehringer-Pipeline" bleibt so) — anders als Tags, die lowercasen. Du legst+taggst+sammelst in **einem** `create_card`-Call.

**Achse A — Tag-Baum** (hierarchische Themen). Mit **`set_tag_parent`** `{tag, parent}` hängst du ein Thema unter ein anderes (`parent: null` macht es zur Wurzel). Beide Namen by-name (get_or_create, **lowercased** wie alle Tags). Ein **Zyklus-Guard** lehnt ab, wenn das Eltern-Tag im Teilbaum des Tags läge. **Lies vorher `list_tags`** (jedes Tag mit `parent_id` + `card_count`), damit du konsistent unter bestehende Themen einordnest statt Parallel-Äste zu bauen. Das ordnet nur das geteilte Tag-Vokabular — **keine Karte wird neu getaggt**; deine `create_card`-`tags` landen automatisch im Teilbaum.

**Grenze**: Sammlungen **löschen/umbenennen** und den Tag-Baum aufräumen bleibt **User-UI** (Kuratierung). Du legst an + ordnest zu; du räumst nicht auf. Sammlungen frei anzulegen ist ok (gewollte Autonomie) — sei trotzdem konsistent (erst `list_collections`/`list_tags` lesen).

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
- **Kein** Highlight-**Löschen** (User-/Reader-only), **kein** Ändern der Anker/Marker (`exact`/`prefix`/`suffix`) und **kein** Anlegen neuer Highlights über den Agent. (Tags/Notiz auf **bestehenden** Highlights zurückschreiben geht — siehe „Highlights annotieren".)
