# SPRINT DOC-API — die Konvertierung wird ein Dienst

**Größe**: L (3 Phasen) · **Datum**: 2026-08-08 · **Vorhaben**: DOC-SVC

## Warum

CONVERTER kann Dokumente konvertieren, aber **kein anderer Dienst kommt daran**: es gibt genau einen Einstieg, `POST /transform-document`, session-authed, Multipart rein, Datei-Download raus. Kein JSON, kein Token, keine Statusverfolgung — und synchron hinter einem einzigen gunicorn-Worker.

Dieser Sprint baut die **Außenseite**. Er entscheidet nichts über Engines; die Engine-Wahl steht im [Entscheidungs-Doc](../../doc_convert_entscheidung_2026-08-08.md) und wird danach eingebaut. Die Reihenfolge ist Absicht: **die Antwortform muss stimmen, bevor ein Backend dahinter kommt** — sie später zu ändern hieße, jeden Konsumenten mitzuändern.

## Gegroundeter Ist-Zustand (Master — nicht neu herleiten)

**Auth-Muster, dreifach erprobt**: `INGEST_TOKEN`, `CARD_TOKEN`, `NARRATION_TOKEN` — je ein eigener Env-Token, fail-closed (503 ohne Konfiguration, constant-time 401), CSRF-exempt, nie geloggt. Die Begründung für den eigenen Narration-Token gilt hier **wörtlich**: eine Konvertierung kostet echtes Geld pro Aufruf, also muss sie unabhängig revozierbar sein.

**Job-Mechanik, Option B (NARR-3)**: Der Worker-Container mountet **nur** `podcast_data` und die Credentials — **kein `app_data`, also keine DB**. Deshalb: der Web-Prozess legt eine `pending`-Zeile an und enqueued, der Worker rendert **DB-frei** auf ein geteiltes Volume und returnt, und die **Web-Seite rekonziliert file-first** beim Pollen. Transienter Redis-Fehler → bleibt `pending`, Terminal-States idempotent. `rq_job_timeout_for(n)` skaliert den Envelope aus der Arbeitsmenge.

⚠️ **Diese Mechanik ist für Konvertate neu zu durchdenken, nicht zu kopieren.** Narration schreibt eine WAV-Datei — „Datei da = fertig" reicht. Ein Konvertat trägt Markdown **plus** Herkunft **plus** Warnungen; das Reconcile muss eine strukturierte Datei lesen, nicht nur ihre Existenz prüfen.

**Bestehender Pfad**: `POST /transform-document` ([app_pkg/documents.py](../../../app_pkg/documents.py)) bleibt **unangetastet** — die Web-UI hängt daran. Der neue Dienst ist additiv; ob die beiden später zusammenwachsen, ist eine spätere Frage.

**Vorhandene Konvertierungs-Fähigkeit**, die dieser Sprint als Backend benutzt: PDF über `services/pdf_extraction/`, alles andere über `services/unstructured_markdown.py`. Beides bleibt **unverändert** — es wird nur aufgerufen.

## Gesperrte Entscheidungen

Aus dem Entscheidungs-Doc, von Oli am 2026-08-08 gesetzt:

1. **Modus pro Auftrag**, mit Default aus einer CONVERTER-Einstellung. Der Aufrufer gibt mit, ob Cloud oder lokal.
2. **Kostendeckel je Auftrag mit Degradation auf den lokalen Pfad** — kein Abbruch. Der Nutzer bekommt immer ein Ergebnis.
3. **Herkunft je Seite** (nicht je Block, nicht je Dokument). Begründung: die Seite ist die Granularität, in der die Pipeline real arbeitet (Seiten-Routing), in der Degradation real passiert (Deckel greift ab Seite N), und die stabil bleibt — Zeichen-Offsets ins Markdown wären fragil.

⚠️ **Punkt 2 und 3 hängen zusammen**: die Degradation erzeugt **gemischte Herkunft innerhalb eines Dokuments**. Genau dafür wurde ein Kandidat im Bake-off am schärfsten kritisiert („14 von 15 Seiten durchgereicht, eine still neu geOCRt — unmarkierte Misch-Provenienz"). Ohne die Herkunft je Seite ist der Deckel nicht vertretbar.

---

# Phase 1 — Kontrakt, Auth, Job

## 1.1 Die Endpunkte

Mindestens: **einreichen**, **Status/Ergebnis abholen**. Ob das Ergebnis am selben Endpunkt hängt oder an einem eigenen, ist deine Wahl — begründe sie.

Muster für den Namensraum und die Form: `app_pkg/narration.py`. Route-Module exponieren `register(app)`, **kein** Blueprint (Hauspattern, Endpoint-Namen bleiben flach).

## 1.2 Auth

Eigener Env-Token, fail-closed wie die drei bestehenden. `_authorize_card_write` / `_authorize_narration_write` sind die Vorlage — **spiegeln, nicht teilen**: die bestehenden Flächen bleiben unberührt.

Zusätzlich muss der Web-Pfad funktionieren (Session bzw. per-User-Bearer über den `request_loader`), damit die App den Dienst später selbst nutzen kann.

## 1.3 Job-Mechanik

Nach dem Option-B-Muster, mit der oben benannten Erweiterung: der Worker schreibt ein **strukturiertes** Ergebnis aufs geteilte Volume, die Web-Seite rekonziliert daraus. Prüfe, ob `podcast_data` dafür der richtige Ort ist oder ob ein eigenes Volume sauberer ist — und begründe.

Der RQ-Envelope skaliert aus der Arbeitsmenge (Seitenzahl), Vorbild `rq_job_timeout_for(n)`.

## 1.4 Größenprüfung

**Vor** der vollständigen RAM-Allokation, nicht danach. Das war ein Muss-Punkt der Anforderungs-Union, den drei von vier Bestandsimplementierungen verletzt haben (sie lesen erst, messen dann).

## 1.5 Ein Backend, damit es fährt

Verdrahte die **vorhandene** Fähigkeit: PDF über `pdf_extraction`, Rest über `unstructured_markdown`. Das ist bewusst die heutige Qualität hinter der künftigen Form — der Sprint prüft den Kontrakt, nicht die Engine.

## Stop
End-to-End belegt: einreichen → pollen → Ergebnis, für ein PDF **und** ein DOCX. `pytest tests/` grün (Baseline **861**), neue Tests für Auth-Fehlpfade und Job-Zustände. **Commit + Push** `feat(DOC-API): Endpunkt, Auth und Job-Mechanik (P1)`. Dann warten.

---

# Phase 2 — Die Antwortform

Das Herzstück. Was hier festgelegt wird, tragen alle künftigen Konsumenten.

## 2.1 Was die Antwort trägt

- **Markdown** — die primäre Nutzlast, für einfache Konsumenten allein ausreichend.
- **Herkunft je Seite**: `deterministisch` | `ocr` | `modell`. Bei gemischten Dokumenten ist das eine Liste, kein Skalar.
- **Degradationen** als Liste **in der Antwort**, nicht im Log: was nicht sauber übersetzt werden konnte, was ausfiel, wo der Deckel griff. `services/unstructured_markdown.py` gibt bereits `(markdown, warnings)` zurück — dieses Muster wächst hier zur API-Fläche.
- **Quell-Metadaten**: Format, Seitenzahl, Dateiname.
- **Kosten und Verbrauch** des Auftrags, sofern Modell-Calls liefen.

**Teilerfolg ist ein 200 mit Fehlerliste, nicht ein 500.** Das ist das einzige Muster, das der Bestand richtig hatte, und es kommt aus dem Bake-off-Register.

## 2.2 Modus pro Auftrag

Der Aufrufer gibt `cloud` | `lokal` mit; fehlt es, greift der Default aus der CONVERTER-Einstellung. **Strikt lesen** — nur der exakte Wert schaltet, alles andere ist 400 (Hauspattern seit LEARN-MORE).

Die Einstellung selbst gehört in denselben `settings_json`-Blob wie die Lern-Einstellungen (`GET/PUT /api/learn/settings` ist die Vorlage für die Mechanik, aber **nicht** derselbe Schlüsselraum — such einen eigenen, sauber getrennten Ort und begründe ihn).

## 2.3 Budget mit Degradation

Harter Deckel je Auftrag. Wird er erreicht, läuft der Auftrag **lokal weiter** statt abzubrechen — und **jede so entstandene Seite trägt die geänderte Herkunft**, plus ein Degradationseintrag, der sagt warum.

⚠️ Der Deckel-**Wert** ist bewusst noch nicht gesetzt: er lässt sich erst sinnvoll wählen, wenn echte Dokumentgrößen durchlaufen. Nimm einen konservativen, gut sichtbar konfigurierbaren Startwert und benenne ihn im Bericht.

⚠️ In diesem Sprint gibt es **noch keinen zweiten Pfad**, auf den degradiert werden könnte. Baue die Mechanik so, dass sie mit einem Platzhalter-Backend nachweisbar funktioniert, und **belege den Fall im Test** — nicht erst, wenn der lokale Pfad existiert.

## 2.4 Idempotenz

Wiederholte Einreichung derselben Datei liefert das gespeicherte Ergebnis statt neuer Modell-Kosten (Content-Hash). Muss-Punkt der Union; zwei Bestandsimplementierungen hatten es, CONVERTER nicht.

## Stop
Antwortform vollständig, Degradation und Budget im Test belegt. **Commit + Push** `feat(DOC-API): Antwortform mit Herkunft und Degradation (P2)`. Dann warten.

---

# Phase 3 — Wrap

- **Kontrakt-Doc** `docs/document_api_contract.md` nach dem Muster von [mobile_auth_contract.md](../../mobile_auth_contract.md) — das ist das Dokument, das ein fremder Dienst liest. Vollständige Beispiele für Anfrage und Antwort, alle Fehlercodes, die Herkunfts- und Degradationswerte mit ihrer Bedeutung.
- **CLAUDE.md** (Architecture Notes), **STATUS.md**, **BACKLOG.md** (Bullet-Guard).
- **Memory**, falls sich eine übertragbare Lehre zeigt; nach dem Schreiben mit `ls` prüfen, dass Datei und Index-Zeile zusammenpassen.
- **Im Bericht benennen**: welchen Deckel-Startwert du gesetzt hast, wo die Modus-Einstellung liegt, und ob `podcast_data` als Ergebnis-Volume tragfähig war oder ein eigenes nötig wurde.

## Nicht-Ziele

- **Keine Engine.** Nicht gemini-nativ, nicht mineru, nicht pandoc, nicht markitdown, nicht trafilatura. Das ist der Folge-Sprint.
- **Kein Anfassen** von `services/pdf_extraction/` oder `services/unstructured_markdown.py` — sie werden aufgerufen, nicht verändert.
- **Kein Anfassen** von `POST /transform-document` und der Web-UI.
- **Kein** zweiter Pfad, kein lokales Modell, kein GPU-Container.
- **Kein** Deploy.
