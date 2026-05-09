# Sprint-Prompts

Ein Doc pro Sprint, abgelegt als `SPRINT_<CODE>_<NAME>_<datum>.md`.

`<CODE>` = stabiler Identifier wie `F2-CII` (Feature 2, Cluster II) oder `BUG-001`. `<NAME>` = ein-Wort-Kurzlabel. `<datum>` = `YYYY-MM-DD` Anlage-Datum.

Format: siehe [`_TEMPLATE.md`](./_TEMPLATE.md). Phase 0 ist Ausnahme, nicht Regel — bei klar geschnittenen Sprints direkt mit Phase 1 starten.

Workflow: Master schreibt Sprint-Prompt-Doc, gibt **direkt im Chat einen copy-paste-fähigen Anfangs-Prompt** aus (Format siehe CLAUDE.md, Sektion „Working Practice"). Oliver kopiert den Block, öffnet einen frischen Sub-Thread, paste-t. Sub-Thread arbeitet ab und meldet zwischen den Phasen zurück. Am Ende pflegt der Sub-Thread `STATUS.md` + `BACKLOG.md` (+ ggf. Memory).

Der Anfangs-Prompt enthält Sub-Thread-Rolle (Executor, nicht Reviewer) + Pfad zur Sprint-Doc + explizite „Phase 1 direkt starten"-Anweisung. Damit ist der Sub-Thread auf Executor-Modus gepinnt und kann nicht versehentlich in Plan-Review-Modus kippen.

Sprint-Prompt-Docs sind Archiv: einmal geschrieben, nicht mehr editiert. Wenn Inhalt überholt ist, neuer Sprint mit Verweis auf den alten.
