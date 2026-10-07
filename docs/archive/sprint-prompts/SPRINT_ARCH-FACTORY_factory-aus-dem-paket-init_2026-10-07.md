# SPRINT ARCH-FACTORY — die Factory aus dem Paket-Init lösen: erst das Gate, dann der Schnitt

**Größe**: M (4 Phasen: Boot-Test + Index-Abgleich · mechanischer Schnitt · Factory + lazy Init + Import-Sentinel · Deploy + Gates + Wrap) · **Datum**: 2026-10-07 · **Herkunft**: [ARCH-AUDIT](../audit-outputs/AUDIT_ARCHITEKTUR_2026-10-01.md) — W-7 (Z. 488), die Factory-Messung in C.0 (Z. 281 ff., Prototypen unter *C.0 Prototypen im Scratch*, Z. 1374), dazu die Befunde V-1, V-3, V-9 (Quellen `FACTORY-*`; im Befund-Doc per Suche nach `FACTORY-` finden); Rang 4 der Top 5; nach ARCH-BUILD

## Warum

[app_pkg/__init__.py](../../../app_pkg/__init__.py) ist mit **766 Zeilen** das Paket-Init **und** die Factory **und** die Migrationen **und** die CLI **und** die Security-Verdrahtung. Weil `app_pkg/config.py` im selben Paket liegt, führt jeder `from app_pkg.config import …` aus `services`, `tasks` und `worker` dieses Init aus und lädt damit Flask, Flask-WTF, SQLAlchemy, click, `models`, nh3 und fsrs mit — gemessen am Pin: **`import app_pkg.config` = +621 Module**, `worker` +622, `services.narration_library` +622, `tasks` +1207, in allen vier Fällen `flask`, `sqlalchemy` und `models` geladen. Der Worker-Elternprozess und der Launcher-Nachbar tragen eine Web-App, die sie nie bauen. Dazu W-7: eine **migrierte** DB hat nicht das Schema, das `create_all` deklariert — an der Prod-DB heute gemessen: **17 Indizes gegen 20**, es fehlen `ix_conversion_lifecycle_status`, `ix_conversion_queue_position`, `ix_tag_parent_id`. Und die Suite würde es nicht merken, wenn jemand den Migrationsaufruf, den Startup-Lock oder die CSV-Migration aus `create_app` streicht (Audit: alle drei Mutationen überleben).

Reihenfolge ist die Pointe: **erst das Gate** (ein Boot-Test, der die Migrationskette von einer echten Legacy-DB bis zum deklarierten Schema prüft), **dann der Schnitt** — ein Umzug ohne diesen Test wäre ein Umzug ohne Netz.

## Gegroundeter Ist-Zustand (Master, gemessen 2026-10-07)

**[app_pkg/__init__.py](../../../app_pkg/__init__.py), 766 Zeilen**, Verantwortungen in Dateireihenfolge: `HttpsOnlySecureSessionInterface` (Z. 35) · `_configure_logging` (Z. 57, `basicConfig` für den Web-Prozess, einziger Aufrufer `create_app` Z. 67) · `create_app` (Z. 65–177) · `_register_sqlite_pragmas` (Z. 178) · `_startup_lock_path`/`_startup_lock` (Z. 211–256, `fcntl.flock` neben der DB-Datei) · `_register_csrf_inversion` (Z. 257) · `_run_pending_migrations` (Z. 303–405: **12 `ALTER TABLE … ADD COLUMN`** — `highlight.note`, `conversion.last_read_percent`, `.lifecycle_status DEFAULT 'inbox'`, `.queue_position`, `.content_version`, `user.settings_json`, `tag.parent_id`, `card.front_svg`, `.back_svg`, `.context_conversion_id`, `.context_heading`, `review.version` — plus `CREATE INDEX IF NOT EXISTS ix_card_context_conversion_id` (LERN-TEXT) und der Aufruf der CSV-Migration) · `_migrate_conversion_tags_csv_to_junction` (Z. 407) · `_register_error_handlers` (Z. 436) · `SECURITY_HEADERS`/`_register_security_headers` (Z. 476–491) · `_register_csrf_endpoint` (Z. 492) · `DE_MONTH_ABBR`/`_register_template_filters` (Z. 499–541) · `PASSWORD_MIN_LENGTH`/`_check_password_rule` (Z. 542–553) · `_register_cli_commands` (Z. 554–746: `create-user`, `set-password`, `reset-collection`; **später Import** Z. 736 `from app_pkg.cards import _naive_utc` mit Kommentar „zirkulär" — laut Audit nicht reproduzierbar, der Import hält rund 95 Module aus dem Paket-Init) · `_resolve_collection` (Z. 747).

**`create_app`-Reihenfolge** (byte-gleich zu erhalten): `_configure_logging()` → `Flask(import_name)` → `ProxyFix` → Config-Zeilen (SECRET_KEY, MAX_CONTENT_LENGTH, Cookie-Flags, `session_interface`, DB-URI, TRACK_MODIFICATIONS) → `CSRFProtect` → `_register_csrf_inversion` → `db.init_app` → `_register_sqlite_pragmas` → `LoginManager` mit `user_loader`, `request_loader` (später Import `app_pkg.mobile_auth.resolve_token`), `unauthorized_handler` → `_register_error_handlers` → `_register_security_headers` → `_register_csrf_endpoint` → `_register_cli_commands` → `_register_template_filters` → `with app.app_context(): os.makedirs('/app/data', exist_ok=True); with _startup_lock(uri): db.create_all(); _run_pending_migrations(app)` → `return app`. Die Route-Module registriert **nicht** die Factory, sondern [app.py](../../../app.py) (Z. 28–48, 54: 19 `register(app)`-Aufrufe nach `create_app()`).

⚠️ **`os.makedirs('/app/data', exist_ok=True)`** steht hart in `create_app`; [tests/conftest.py](../../../tests/conftest.py) Z. 117–125 biegt deshalb `os.makedirs` für `/app/*`-Pfade auf No-op — ein Test-Pflaster für einen Container-Pfad in der Factory.

**Wer das Init mitlädt:** `from app_pkg.config import …` in sechs Service-Modulen (`narration_render`, `deepgram_service`, `narration_library`, `document_conversions`, `transcription_jobs`, `pdf_cloud`), dazu [tasks.py](../../../tasks.py) und [worker.py](../../../worker.py). [app_pkg/config.py](../../../app_pkg/config.py) selbst importiert nur Stdlib, `rq.serializers` und `services.mineru_invocation` (pur) — es ist das **Init**, das schwer ist, nicht `config`.

**Wer Factory-Interna importiert** (alle aus `app_pkg` direkt, keine `patch('app_pkg.…')`/`setattr`): `_run_pending_migrations` in 7 Testdateien (`test_lifecycle`, `test_cards`, `test_conversion_progress`, `test_docwrite_lost_update`, `test_lern_text`, `test_reading_list`, `test_review_lost_update`), `_startup_lock` + `_startup_lock_path` ([tests/test_db_runtime.py](../../../tests/test_db_runtime.py)), `HttpsOnlySecureSessionInterface` (`test_cookie_secure`), `_migrate_conversion_tags_csv_to_junction` (`test_conversion_tags`), `create_app` ([app.py](../../../app.py)). **Das sind die sechs Namen**, die das Paket nach dem Schnitt weiter anbieten muss (nur `create_app` reicht nicht: 9 Sammelfehler laut Audit-Prototyp).

**Muster im Haus:** [services/__init__.py](../../../services/__init__.py) ist seit SEC-SOCKET ein PEP-562-Lader (`_LAZY`-Dict, `__getattr__`, `import_module`, Cache am Modul) — die Vorlage für das neue Paket-Init. [tests/test_import_surface.py](../../../tests/test_import_surface.py) misst Import-Oberflächen **je Modul im eigenen Subprozess** (`sys.modules` nach dem Import als JSON) — die Vorlage für den neuen Sentinel. [tests/test_lifecycle.py](../../../tests/test_lifecycle.py) Z. 54–88 stellt den Legacy-Zustand her, indem es Index und Spalte **selbst droppt** (⚠️ für FK-Spalten geht das nicht — LERN-TEXT musste die Tabelle per Tausch nachbauen; ein Boot-Test braucht deshalb eine echte Legacy-DDL, s. Entscheidung 1).

**Legacy-Basis:** die erste Migration kam mit `5b33f75` (2026-05-25, R1-B-B, `highlight.note`) — `git show 5b33f75^:models.py` ist das letzte Modell **ohne** Migrationskette; alles seither (12 Spalten, Index, CSV-Junction, die Tabellen `collection`, `collection_documents`, `api_token`, `review` usw.) entsteht beim Boot aus `create_all` + Migrationen.

**Prod-DB (ro, heute):** 17 Indizes, s. oben; `lifecycle_status` wird mit `DEFAULT 'inbox'` nachgezogen (Z. 322) — ob die migrierte Spalte denselben Default trägt wie das Modell, misst Phase 0; `tag.parent_id` hat in der migrierten DB keinen Fremdschlüssel (inert ohne FK-Pragma, bleibt so).

**CLI-Aufrufer:** `flask --app app create-user` in sechs Smokes, `flask set-password` und `flask reset-collection` in CLAUDE.md (per `docker exec … flask …`). Nach dem Schnitt müssen alle drei Kommandos im Container unverändert antworten.

**Churn:** `app_pkg/__init__.py` 17 und `config.py` 16 Commits seit Juli bei 318 insgesamt.

**Baseline:** 1675 passed + 1 skipped (Mac und Pin, Container-Suite 1 Warnung). Prod `converter-app:latest` = `495bc81391aa`, Rollback-Tag `pre-arch-build` = `f9cd92dd81ac`; `docs/build/pip-freeze.txt` = das deployte Image (ARCH-BUILD-Ritual gilt: Freeze vorher gegen die Datei, nach dem Build Diff — **diesmal muss er leer sein**, es ändert sich keine Abhängigkeit).

## Gesperrte Entscheidungen

1. **Das Gate zuerst — ein Boot-Test von einer echten Legacy-DB** (W-7, Phase 0): eine eingecheckte DDL `tests/fixtures/legacy_schema_5b33f75.sql`, erzeugt aus `git show 5b33f75^:models.py` per `create_all` in einem Subprozess und `sqlite_master`-Dump (Kopf: Commit, Datum, Erzeugung; **einmal** erzeugt, danach eingefroren — die Basis bewegt sich nicht mehr). Der Test lädt die DDL in eine frische Datei, zeigt `DATABASE_URL` darauf, ruft **`create_app()`** (nicht nur `_run_pending_migrations`) und prüft danach gegen `db.metadata`: **alle Tabellen** des Modells existieren, **alle 12 migrierten Spalten** sind da, **die Indexmenge ist genau die von `create_all` auf leerer DB** (Namen), die migrierte `lifecycle_status` trägt den Modell-Default, die CSV-Junction ist befüllt (ein Legacy-Datensatz mit CSV-Tags im Fixture-Lauf). Zweiter `create_app()` auf derselben Datei ist ein No-op (Idempotenz). ⚠️ Die drei Mutationen aus dem Audit müssen den Test **rot** machen: Migrationsaufruf gestrichen, CSV-Aufruf gestrichen, Startup-Lock gestrichen — für den Lock, der am Schema nichts ändert, prüft ein eigener Test per Fake, dass `create_app` den Lock mit der DB-URI betritt, bevor `create_all` läuft (Reihenfolge am Fake belegt).
2. **Index-Abgleich am Ende der Migrationen** (W-7): nach den `ALTER`s vergleicht `_run_pending_migrations` je Modell-Tabelle die deklarierten Indizes (`table.indexes` aus `db.metadata`) mit `inspector.get_indexes` und legt **fehlende per Name** an (`Index.create(bind)` oder `CREATE INDEX IF NOT EXISTS` mit denselben Spalten — eine Mechanik, nicht zwei); nichts wird gelöscht, nichts umbenannt, keine UNIQUE-Indizes neu erfunden (das Modell hat heute keine; gäbe es einen, wäre er auf Bestandsdaten ein Korrektheitsfehler — der Abgleich **lehnt** UNIQUE ab und loggt es, statt ihn blind zu bauen). Log-Zeile je angelegtem Index. Effekt auf Prod beim Deploy: **17 → 20 Indizes**, vorher/nachher gemessen. Die Entscheidung „kein Alembic" bleibt.
3. **`/app/data` raus aus der Factory:** das Verzeichnis der SQLite-Datei wird **aus der DB-URI** abgeleitet (nur für Datei-URIs; `:memory:` und andere Backends: nichts), in `db_runtime` neben dem Startup-Lock, der dieselbe Herleitung schon kennt (`_startup_lock_path`). Prod-Verhalten identisch (URI zeigt auf `/app/data/converter.db`); das `os.makedirs`-Pflaster in `conftest.py` wird **entfernt** — bleibt die Suite grün, ist die Abhängigkeit weg (Sentinel: kein Literal `/app/data` mehr unter `app_pkg/`).
4. **Phase 1 ist ein mechanischer Umzug, kein Umbau:** `app_pkg/migrations.py` (`_run_pending_migrations`, `_migrate_conversion_tags_csv_to_junction`, Index-Abgleich), `app_pkg/cli.py` (`PASSWORD_MIN_LENGTH`, `_check_password_rule`, `_register_cli_commands`, `_resolve_collection`; der späte Import `_naive_utc` wandert mit und bekommt den **richtigen** Kommentar: er hält `app_pkg.cards` samt ~95 Modulen aus dem Init, „zirkulär" ist er nicht), `app_pkg/security.py` (`HttpsOnlySecureSessionInterface`, `_register_csrf_inversion`, `SECURITY_HEADERS`, `_register_security_headers`, `_register_csrf_endpoint`, `_register_error_handlers`), `app_pkg/db_runtime.py` (`_register_sqlite_pragmas`, `_startup_lock_path`, `_startup_lock`, das Verzeichnis aus Entscheidung 3). `DE_MONTH_ABBR`/`_register_template_filters` und `_configure_logging` bleiben bei der Factory. **Funktionskörper byte-gleich** (sha256 je Funktion vor/nach, Beleg wie ARCH-NARR5 und NOTION-MEETING-LINK; Ausnahmen nur die in Entscheidung 2–4 benannten Zeilen, einzeln aufgeführt), die `create_app`-Reihenfolge unverändert. In Phase 1 bleibt `create_app` noch im Init und re-exportiert die sechs Namen — die Tests laufen **ohne** Änderung.
5. **Phase 2: Factory nach `app_pkg/factory.py`, das Init wird ein PEP-562-Lader** (Muster `services/__init__.py`) für **genau sechs Namen**: `create_app`, `_run_pending_migrations`, `_startup_lock`, `_startup_lock_path`, `_migrate_conversion_tags_csv_to_junction`, `HttpsOnlySecureSessionInterface` — jeder Name zeigt auf sein neues Modul; `__all__` ist die Liste; ein unbekannter Name → `AttributeError` wie bei `services`. Das Init importiert **nichts** eager (kein Flask, kein `models`). Danach gilt der **Subprozess-Sentinel** in `tests/test_import_surface.py`: `import app_pkg.config`, `import worker`, `import services.narration_library` laden **weder `flask` noch `sqlalchemy` noch `models`**; dazu ein Positivfall (`import app` lädt alle drei), damit der Check nachweislich feuern kann. Modulzahlen vorher/nachher je Import am Pin in den Bericht (Erwartung aus dem Audit-Prototyp: `import worker` ≈ 250 statt 622). Der Rest-Zyklus `config → services.mineru_invocation` ist pur und bleibt; `services/__init__` bleibt lazy.
6. **`tasks.py` bleibt schwer, und das ist richtig so:** es importiert die Renderer und Dienste, die es braucht; Ziel ist nicht „tasks leicht", sondern „wer nur `config` will, bekommt nur `config`". Der Sentinel misst `tasks` **nicht** als leicht, nur als frei von `flask`/`models`, falls das nach dem Schnitt gilt — sonst benennen, nicht erzwingen.
7. **Keine Verhaltensänderung an Routen, Auth, CSRF, Cookies, Headern:** die bestehenden Sentinels (`test_csrf_inversion`, `test_cookie_secure`, `test_proxy_fix`, `test_db_runtime`, `test_security_headers`-Äquivalente) sind der Beleg; sie werden nicht angefasst, nur ihre Importe folgen den sechs Namen, falls nötig (sollten sie nicht — die Namen bleiben am Paket).
8. **Deploy nach Phase 3 mit dem ARCH-BUILD-Ritual:** Freeze vorher == Datei; Backup `pre-arch-factory-2026-10-07` (die Indizes sind eine Schema-Änderung); Tag `pre-arch-factory`; Build; Freeze nachher → **Diff leer** (kein pip-Edit; ein nicht-leerer Diff hieße, der Layer-Cache hat nicht gehalten — benennen, Gates fahren); Prod-Indizes 17 → 20 im Web-Log und per `mode=ro`; die drei CLI-Kommandos im Container (`create-user --help`, `set-password --help`, `reset-collection --help` reichen als Erreichbarkeits-Beleg; kein echter Lauf gegen Olis Konto); Container-Suite; Login 200; Olis Zeilenzahlen.
9. **Nicht in diesem Sprint:** Alembic, Umbau von `config.py` (seine Lage im Paket ist nach Phase 2 unschädlich), `services/__init__` (bleibt), Änderungen an den Route-Modulen oder an `app.py` über den Import hinaus, der `cards._naive_utc`-Helfer selbst, UNIQUE-Indizes, ARCH-LIBRARY-KLEIN, Agent-Flächen (**kein Brief**, im Wrap so sagen).

**Arbeitsweise:** inline, **kein Workflow, keine Subagenten** ohne Olis ausdrückliches Wort. Commit + Push je Phase, dann Stop + Bericht. Nichts Tragendes im Session-Scratch. Editiert wird nur auf dem Mac, gebaut nur auf der Mintbox. Tests zuerst, **gegen HEAD rot gezeigt**; seit ARCH-BUILD dürfen zwei pytest-Prozesse parallel laufen (Test-DB je Prozess). Byte-Gleichheit umgezogener Funktionen per sha256 belegen. Keine Tokens, keine Passwörter in Ausgaben.

---

# Phase 0 — Das Gate: Boot-Test + Index-Abgleich (kein Deploy)

1. Legacy-DDL erzeugen (Entscheidung 1), einchecken, Erzeugungsweg im Kopf.
2. `tests/test_boot_schema.py` (neu): der Boot-Test aus Entscheidung 1 — zuerst **rot** gegen HEAD (er fällt heute an den drei fehlenden Indizes), dazu der Lock-Reihenfolge-Test und die Idempotenz.
3. Index-Abgleich in `_run_pending_migrations` (Entscheidung 2; noch im Init — der Umzug ist Phase 1), UNIQUE-Ablehnung mit Test (ein künstliches Modell-Index-Objekt mit `unique=True` im Test → Log + nicht angelegt).
4. Die drei Audit-Mutationen je einmal fahren, rote Testnamen in den Bericht; `test_lifecycle` und die sechs anderen Migrations-Tests bleiben unverändert grün.
5. Mac-Suite.

## Stop
Bericht: DDL-Kopf · rote Zahl gegen HEAD · Mutationen → rote Tests · Default-Befund `lifecycle_status` (Modell vs. migriert) · Suite · Commit-Hash. **Phase 0 ist für sich abnehmbar** — der Master kann hier stoppen, wenn Oli es will. Dann warten.

---

# Phase 1 — Mechanischer Schnitt (kein Deploy)

1. Vier Module nach Entscheidung 4, `db_runtime` mit dem Verzeichnis aus Entscheidung 3, `conftest`-Pflaster entfernt, Literal-Sentinel.
2. sha256-Tabelle je umgezogener Funktion (vorher aus `git show HEAD:app_pkg/__init__.py`, nachher aus dem neuen Modul; Abweichungen einzeln benannt).
3. `create_app` ruft die umgezogenen Namen aus den neuen Modulen, Reihenfolge byte-gleich (Diff von `create_app` zeigt nur Import-Herkunft).
4. Mac-Suite; die Zahl bleibt 1675 + 1 + Phase 0 — kein neuer Test außer dem Literal-Sentinel.

## Stop
Bericht: sha256-Tabelle · `create_app`-Diff · conftest-Diff · Suite · Commit-Hash. Dann warten.

---

# Phase 2 — Factory + lazy Init + Import-Sentinel (kein Deploy)

1. `app_pkg/factory.py` mit `create_app`, `_configure_logging`, Template-Filtern; Init als PEP-562-Lader (Entscheidung 5), `__all__` = sechs Namen; `app.py` unverändert bis auf nichts (es importiert `create_app` aus `app_pkg` — das trägt der Lader).
2. Sentinel in `tests/test_import_surface.py` (Entscheidung 5), **rot gegen den Phase-1-Stand** gezeigt (dort lädt `app_pkg.config` noch Flask).
3. Modulzahlen am Pin: dieselbe Messung wie im Ist (Wegwerf-Container, `--network none`), vorher/nachher je Import in einer Tabelle.
4. `flask --app app --help` auf dem Mac listet die drei Kommandos (die Container-Prüfung ist Phase 3).
5. Mac-Suite + Container-Suite (altes Image, gestreamter Baum — das Image ändert sich erst in Phase 3).

## Stop
Bericht: Lader-Diff · Sentinel rot/grün · Modulzahlen-Tabelle · CLI-Liste · beide Suiten · Commit-Hash. Dann warten.

---

# Phase 3 — Deploy + Gates + Wrap

**Deploy** (Entscheidung 8, in dieser Reihenfolge): Freeze vorher == Datei → Backup nach Rezept (`-rw-------` ohne `+`, Prüfung nur `mode=ro&immutable=1`, Indexzahl der Kopie = 17) → Tag `pre-arch-factory` → `git pull --ff-only` + `docker compose up -d --build` aus dem Projektverzeichnis → Web-Log: die drei Index-Zeilen, keine Fehler → Prod-Indizes per `mode=ro` = 20, Namen = `create_all`-Menge → Freeze nachher, Diff **leer** → Container-Suite am neuen Image → die drei `flask`-Kommandos im Web-Container mit `--help` → Login 200, `/library` ohne Cookie 302 → Olis Zeilenzahlen vorher = nachher → Worker-Log: `import`-Zeile ohne Flask-Spuren ist nicht messbar, stattdessen `docker exec markdown-converter-worker python -c "import worker, sys; print('flask' in sys.modules)"` → `False`.

**Wrap:** CLAUDE.md (*Key Files*: `app_pkg/` beschreibt jetzt `factory.py`, `migrations.py`, `cli.py`, `security.py`, `db_runtime.py` und das lazy Init; *Architecture Notes*: ein ARCH-FACTORY-Bullet — Boot-Test als Gate, Index-Abgleich und warum UNIQUE abgelehnt wird, `/app/data` aus der URI, die sechs Namen, Modulzahlen vorher/nachher, Rest-Zyklus benannt; die Stellen, die `app_pkg/__init__.py` für Migrationen/CLI/Lock zitieren — mindestens LEARN-BACK, SEC-SET-PASSWORD, SYNC-FREEZE, LOST-UPDATE —, auf die neuen Module umschreiben; Baseline); BACKLOG (ARCH-FACTORY schließen; die beiden Folge-Verweise „vor CSP-BASELINE und SEC-TOKEN-EXPIRY" erfüllt; kein neues Item ohne Befund); STATUS; Memory (Kandidat: „Gate vor dem Schnitt — ein Boot-Test von einer eingefrorenen Legacy-DDL macht Migrationsketten prüfbar"); Bullet-Guard; Mintbox-Clone auf den Wrap-Stand bei unverändertem Image. Kein Brief.

## Stop
Bericht: Deploy-Belege in Reihenfolge · Index-Zahlen vorher/nachher · Freeze-Diff wörtlich (leer) · Gates · Abweichungen · Olis Hand (Tags/Backups nur auflisten) · Commit-Hashes. Sprint-Ende.

## Abnahme (Master)
- Suite grün, Mac und Pin; Boot-Test vom Master mit einer der drei Mutationen selbst rot gezeigt.
- Prod: 20 Indizes mit den Namen von `create_all`; `import worker` im Worker-Container ohne `flask`; `flask set-password --help` im Web-Container antwortet.
- `app_pkg/__init__.py` ≤ 60 Zeilen, ohne eager Import; sha256-Tabelle stichprobenartig nachgerechnet.
- Freeze-Diff leer.
