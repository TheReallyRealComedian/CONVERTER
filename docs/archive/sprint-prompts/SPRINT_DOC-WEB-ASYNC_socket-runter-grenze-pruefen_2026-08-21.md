# SPRINT DOC-WEB-ASYNC — der Socket kommt vom Web-Container, und die Seitengrenze wird überprüft

**Größe**: M (3 Phasen) · **Datum**: 2026-08-21 · **Vorhaben**: DOC-SVC

## Warum

Zwei Dinge, die zusammengehören.

**Erstens Sicherheit.** Seit DOC-WEB P2 hält der **Web**-Container den Docker-Socket — also ausgerechnet der Prozess, der aus dem Internet erreichbar ist. Der Socket ist eine Generalvollmacht: wer Container erzeugen darf, hängt das Wirtsdateisystem ein und ist root auf der Mintbox. Der Login-Zaun ist die einzige Schicht davor.

**Zweitens eine Grenze, die auf einer ungeprüften Annahme steht.** `MAX_SYNC_PDF_PAGES = 12` wurde gesetzt, weil ein Konvertierungslauf „den einzigen gunicorn-Worker blockiert". Gemessen wurde nur die **Dauer** eines Laufs, nie die **Blockade**. Wenn die App währenddessen weiter bedienbar ist, verteidigt die Grenze gegen nichts.

## Gegroundeter Ist-Zustand (Master — nicht neu herleiten)

**Die Instanz ist aus dem Internet erreichbar** (2026-08-21 verifiziert): `converter.smallpieces.de` → `94.114.240.3`, antwortet. **Login-geschützt ist sie** — `/`, `/library` und `/api/collections` springen ohne Cookie auf `/login`. ⚠️ CLAUDE.md sagte bis zum 2026-08-21 „LAN-only"; das war falsch und ist korrigiert.

**Der Socket am Web-Container** steht in [docker-compose.yml](../../../docker-compose.yml) mit Begründungs-Kommentar, zusammen mit dem Exchange-Bind und den `MINERU_*`-Envs — identisch zum Worker.

**Warum der Web-Container ihn überhaupt braucht**: mineru ist in ihm **nicht installiert** (gemessen: `which mineru` leer, keine GPU sichtbar), es ist ein eigenes Image von **29,7 GB** gegen 5,88 GB des App-Images. „Lokal konvertieren" heißt deshalb zwingend „einen mineru-Container starten", und das geht nur über den Docker-Dienst des Wirts.

**⚠️ Die Blockade-Annahme ist ungeprüft und vermutlich falsch.** [Dockerfile:99](../../../Dockerfile) startet `gunicorn --workers 1 --worker-class uvicorn.workers.UvicornWorker`, und [app.py:87](../../../app.py) ist `asgi_app = WsgiToAsgi(app)`. Alle Views sind **sync** (null `async def` in den Route-Modulen). asgirefs `WsgiToAsgi` führt sync-WSGI über `sync_to_async(thread_sensitive=False)` aus — also **je Anfrage ein eigener Thread aus einem Pool**, nicht serialisiert. Dazu wartet eine Konvertierung fast ausschließlich auf Netzwerk (Gemini) oder einen Subprozess (mineru), und beides gibt die GIL frei. **Das ist eine Code-Lesart, keine Messung — Phase 1 misst sie.**

**Die asynchrone Fläche existiert vollständig**: `POST /api/document-conversions` + `GET /api/document-conversions/<id>` mit Dual-Auth (Token **oder** Session/Bearer) und **ohne** CSRF-Exemption, weil Cookie-Sessions dort ein legitimer Schreibweg sind. Frontend-seitig liegen `window.CSRF_TOKEN` ([templates/base.html](../../../templates/base.html)) und `/api/csrf-token` bereit. Ein browserseitiges Einreichen-und-Pollen müsste also **nichts Neues erfinden**, nur benutzen.

**Job-Mechanik** (Option B seit NARR-3, seither zweimal getragen): Web legt eine `pending`-Zeile an und enqueued, der Worker arbeitet **DB-frei** aufs geteilte Volume, die Web-Seite rekonziliert file-first.

**`/transform-document` schreibt heute keine Historien-Zeile** — es liefert nur die Datei zurück. ⚠️ Der API-Weg legt dagegen eine `Conversion` an. Wer den Browser auf die API umlenkt, ändert damit das Verhalten: jede Umwandlung im Browser landet in der Library, und die Idempotenz greift (dieselbe Datei zweimal → gespeichertes Ergebnis statt neuer Lauf). Beides ist vertretbar, aber es ist eine **Entscheidung**, keine Nebensache.

## Gesperrte Entscheidungen

1. **Der Socket kommt vom Web-Container runter.** Das ist das Sprint-Ziel und steht nicht zur Disposition.
2. **Der Container-Start wandert zum Worker.** Der Web-Prozess spricht mit Redis — wohin er ohnehin spricht —, der Worker spricht mit Docker. Aus einer Generalvollmacht am exponierten Prozess wird ein einzelner, eng umrissener Auftrag.
3. **Keine zweite Queue** (Oli, 2026-08-21): Warteschlangen-Konkurrenz ist bei einem einzigen Nutzer kein Problem — er wartet höchstens auf sich selbst.
4. **Nicht-Wege, mit Begründung**: Web-PDF auf „immer Cloud" zu stellen macht den Browser-Knopf wieder kostenpflichtig und zerlegt das Ein-Schalter-Prinzip. Ein Socket-Proxy davor bringt fast nichts — die üblichen filtern nach API-Endpunkt, nicht nach Nutzlast, und wer Container erzeugen darf, darf auch `/` einhängen.

---

# Phase 1 — Blockiert die App überhaupt?

Das ist die billigste Phase und sie entscheidet die Form von Phase 2. **Nicht überspringen, nicht abkürzen.**

## 1.1 Die Messung

Starte über den **Web**-Pfad eine lange lokale PDF-Konvertierung (der 15-Seiten-Scan aus `05_scan-sauber` braucht ~86 s) und miss **währenddessen**, ob die App bedienbar bleibt: Latenz von `/login`, einer Library-Seite und einem `/api/...`-Read, mehrfach über die Laufzeit verteilt. Miss dieselben Latenzen im Leerlauf als Grundlinie.

Fahr zusätzlich **zwei** Konvertierungen gleichzeitig und sieh, ob beide laufen oder die zweite wartet.

⚠️ Für den Web-Pfad brauchst du eine Session. Nimm wieder einen Wegwerf-User und räum ihn am Sprint-Ende ab — **strikt nach `user_id`**, niemals pauschal: Olis iOS-App besitzt Tokens in derselben Tabelle.

## 1.2 Was das Ergebnis bedeutet

- **Die App bleibt bedienbar** → die Begründung für `MAX_SYNC_PDF_PAGES = 12` ist hinfällig. Dann ist server-seitiges Warten in Phase 2 völlig ausreichend, und die Grenze wird in Phase 3 auf einen Wert gesetzt, der eine **echte** Schranke benennt (Geduld des Nutzers, RQ-Umschlag, Speicher) — oder ganz entfernt.
- **Die App blockiert** → die Grenze hat einen echten Grund, und Phase 2 sollte den Browser **selbst pollen** lassen (die API-Endpunkte und das CSRF-Plumbing existieren, s. oben), weil nur das den gunicorn-Thread sofort freigibt.

**Beide Wege sind vorgesehen. Die Messung wählt, nicht die Vorliebe.** Zahlen in den Bericht.

## Stop
Messung belegt, Weg benannt. **Commit + Push** falls Code entstand (ein Messskript darf ins Repo, wenn es reproduzierbar ist), sonst nur Bericht. Dann warten.

---

# Phase 2 — Der Socket kommt runter

## 2.1 Der Umbau

Der Web-PDF-Pfad konvertiert nicht mehr selbst, sondern reiht den Auftrag ein. Welche der beiden Formen — server-seitig warten oder browserseitig pollen — steht nach Phase 1 fest.

⚠️ **Bei der server-seitigen Variante**: die Web-Anfrage wartet auf die Ergebnisdatei, so wie das Reconcile sie ohnehin liest. Für den Browser ändert sich **nichts** — gleicher Klick, gleiches Warten, gleicher Download, gleiche Hinweis-Box.

⚠️ **Bei der browserseitigen Variante**: benutze die bestehenden API-Endpunkte, erfinde keinen zweiten Weg. Und entscheide bewusst, was mit den beiden Verhaltensänderungen geschieht, die daran hängen (Historien-Zeile je Umwandlung, Idempotenz-Dedup bei derselben Datei) — **benenne sie im Bericht**, egal wie du dich entscheidest.

## 2.2 Compose

Socket und die `MINERU_*`-Verdrahtung kommen beim **Web**-Container weg, beim Worker bleiben sie. ⚠️ Der Exchange-Bind wird vermutlich weiterhin auf beiden Seiten gebraucht (der Web-Prozess legt die Quelldatei ab) — prüfen, nicht raten.

## 2.3 Der Beleg, der zählt

Nach dem Umbau: ein PDF im Browser mit `mode=lokal` konvertieren und zeigen, dass es **funktioniert**, während `docker exec markdown-converter-web ls /var/run/docker.sock` **fehlschlägt**. Das ist der eigentliche Nachweis des Sprints — Fähigkeit erhalten, Vollmacht weg.

## Stop
Socket weg, Konvertierung läuft, Beleg gezeigt. **Commit + Push** `feat(DOC-WEB-ASYNC): Container-Start zum Worker, Socket vom Web-Container (P2)`. Dann warten.

---

# Phase 3 — Die Seitengrenze und Wrap

## 3.1 Die Grenze

Setz `MAX_SYNC_PDF_PAGES` auf den Wert, den Phase 1 **begründet** — oder entferne sie, wenn sich keine Schranke mehr benennen lässt. ⚠️ **Ein Wert ohne benannte Schranke ist schlechter als keiner**: er sieht nach Sorgfalt aus und schützt vor nichts. Wenn du ihn behältst, muss der Kommentar sagen, **wogegen**.

Wenn eine echte Obergrenze bleibt (RQ-Umschlag, Speicher, Geduld), gehört sie in dieselbe Meldung wie heute — deutsch, max 2 Sätze, mit dem Verweis auf den Dienst.

## 3.2 Wrap

- **Kontrakt-Doc**: der Betriebsvoraussetzungs-Block zum root-äquivalenten Socket gilt danach nur noch für den **Worker** — nachziehen.
- **CLAUDE.md** (DOC-WEB-Bullet und die Socket-Warnungen), **STATUS.md**, **BACKLOG.md** (Bullet-Guard).
- **Engine-Generation**: nur bumpen, wenn sich am **Ergebnis** einer Datei etwas ändert. Ein reiner Transportweg-Umbau ändert nichts — dann **nicht** bumpen und das im Bericht sagen.
- **Memory**, falls übertragbar. Nach dem Schreiben mit `ls` prüfen, dass Datei und Index-Zeile zusammenpassen.
- **Wegwerf-User abräumen**, strikt nach `user_id`.
- **Im Bericht benennen**: die Latenzzahlen aus Phase 1 · welchen Weg sie gewählt haben · was mit `MAX_SYNC_PDF_PAGES` geschah und wogegen der Wert (falls er bleibt) schützt · ob der Exchange-Bind am Web-Container bleiben musste.

## Nicht-Ziele

- **Kein** zweiter Queue-/Worker-Aufbau (Entscheidung 3).
- **Kein** Umbau der API-Antwortform, des Job-Modells oder des Kontrakt-Vertrags.
- **Kein** Anfassen des Office-/Web-Format-Zweigs — der ist synchron, schnell und braucht keinen Container.
- **Kein** `mode=deterministisch`, kein Merge, keine Engine-Änderung.
- ⚠️ **Editiert wird nur auf dem Mac.** Die Mintbox ist Runtime — Deploy und Smoke ja, Arbeitsplatz nein, keine unversionierten Dateien zurücklassen.
