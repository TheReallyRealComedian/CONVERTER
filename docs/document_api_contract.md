# Document-API-Kontrakt — Dokument-Konvertierung als Dienst

Stand 2026-08-09 (Sprints DOC-API + DOC-ENGINE). Alle Pfade relativ zur
Basis-URL des CONVERTER-Stacks. Antworten `application/json`; die Einreichung
ist `multipart/form-data`. Dieses Dokument ist der Vertrag, den ein fremder
Dienst liest — was hier steht, gilt; was die Antwort trägt, steht hier.

## 1. Ablauf in einem Absatz

Ein Dokument wird per `POST /api/document-conversions` eingereicht (202,
asynchron), der Aufrufer pollt `GET /api/document-conversions/<id>`, bis
`status` terminal ist (`ready` | `failed`). Die `ready`-Antwort trägt das
Ergebnis **in derselben Antwort**: Markdown plus Herkunft je Einheit plus
Degradationsliste plus Verbrauch — es gibt keinen separaten Ergebnis-Endpunkt.

## 2. Auth — zwei gleichberechtigte Wege

**Weg A, Service-Token** (für Dienste): eigener Env-Token `DOC_CONVERT_TOKEN`
(nicht mit `INGEST/CARD/NARRATION_TOKEN` geteilt — eine Konvertierung kann
Modell-Geld kosten und muss unabhängig revozierbar sein).

    Authorization: Bearer <DOC_CONVERT_TOKEN>

Gilt für **beide** Endpunkte — einreichen *und* pollen. Fail-closed: Token
nicht konfiguriert → `503 {"error": "Dokument-API nicht konfiguriert."}`;
fehlender/falscher Token → generisches `401 {"error": "Nicht autorisiert."}`
(constant-time-Vergleich, keine Unterscheidung). Die Zeilen gehören dem
Ingest-Zielnutzer (`INGEST_USER` bzw. Single-User).

**Weg B, eingeloggter Nutzer**: Session-Cookie (Web) oder per-User-Bearer
(iOS-App, [mobile_auth_contract.md](mobile_auth_contract.md)). Funktioniert
unabhängig davon, ob der Service-Token konfiguriert ist.

⚠️ **CSRF-Sonderfall, wichtig für Service-Caller**: Der POST ist bewusst
**nicht** CSRF-exempt (Cookie-Sessions sind hier ein legitimer Schreibweg und
bleiben geschützt; die Bearer-Präsenz überspringt die CSRF-Prüfung). Folge:
ein POST **ohne** `Authorization`-Header und ohne gültiges CSRF-Token stirbt
als **400 (csrf)**, *bevor* er die 401/503-Auth-Antworten erreicht. Ein
Service-Caller schickt den Token deshalb **immer** als Bearer-Header — dann
sieht er ausschließlich die Auth-Semantik aus Weg A.

## 3. Einreichen

    POST /api/document-conversions
    Content-Type: multipart/form-data
    Authorization: Bearer <token>

    file: <die Datei>            # Pflicht
    mode: cloud | lokal          # optional, strikt gelesen (s. §6)

- **202** → `{"id": 7, "status": "pending", "mode": "cloud", "job_id": "…"}`
  — neuer Auftrag angelegt und enqueued.
- **200** → die volle Antwortform aus §5 **plus** `"deduped": true` — die
  Datei war (mit diesem Modus) schon eingereicht; gespeicherter Stand, kein
  neuer Job, keine neuen Modell-Kosten (s. §8).
- **400** → kein `file`-Feld · leerer Dateiname · nicht unterstützte Endung ·
  leere Datei · ungültiger `mode`.
- **413** → Upload über **100 MB** (`MAX_DOCUMENT_UPLOAD_BYTES`). Geprüft am
  `Content-Length`-Header, **bevor** der Body geparst wird; ein fehlender oder
  lügender Header wird nach dem Spool auf Platte nachgeprüft.

Unterstützte Endungen: `pdf, docx, pptx, eml, html, htm, txt, md`
(dieselbe Liste wie der Web-Konverter, `app_pkg/documents.py`).

## 4. Pollen

    GET /api/document-conversions/<id>
    Authorization: Bearer <token>

- **200** → Antwortform aus §5. `pending` → weiter pollen (empfohlen: wenige
  Sekunden Abstand; ein Auftrag skaliert seinen internen Timeout aus der
  Seitenzahl).
- **404** → nicht vorhanden, fremder Besitzer **oder** falscher Typ —
  ununterscheidbar, absichtlich.

Terminal-Zustände sind idempotent: jedes weitere Pollen liefert dieselbe
Antwort aus der Datenbank. Ein transienter Redis-Ausfall lässt einen Auftrag
`pending` (kein Fehl-Fail); ein aus Redis verschwundener Job ohne Ergebnis
wird `failed` („Job nicht mehr auffindbar.").

## 5. Die Antwortform

```json
{
  "id": 7,
  "status": "ready",
  "mode": "cloud",
  "markdown": "# Kapitel Eins\n\n…",
  "provenance_unit": "page",
  "provenance": ["modell", "modell", "deterministisch", "deterministisch"],
  "degradations": [
    {
      "code": "budget_exceeded",
      "message": "Kostendeckel 1.00 € erreicht (Stand 1.00 € nach Seite 2). Ab Seite 3 lokal konvertiert.",
      "pages": [3, 4]
    }
  ],
  "usage": {"model_calls": 2, "cost_eur": 1.0},
  "budget_eur": 1.0,
  "error": null,
  "source": {
    "filename": "Bericht Q3.pdf",
    "format": "pdf",
    "size_bytes": 184122,
    "page_count": 4
  },
  "created_at": "2026-08-09T09:12:33.412000"
}
```

Feld für Feld:

| Feld | Bedeutung |
|---|---|
| `status` | `pending` · `ready` · `failed`. Nur `ready` trägt Ergebnisfelder. |
| `mode` | Der **effektive** Modus des Auftrags (nach Default-Auflösung, §6). |
| `markdown` | Die primäre Nutzlast; für einfache Konsumenten allein ausreichend. `null` außer bei `ready`. |
| `provenance_unit` | `page` \| `document` — worauf sich die `provenance`-Einträge beziehen (§5a). |
| `provenance` | **Immer eine Liste**, ein Eintrag je Einheit, Reihenfolge = Dokumentreihenfolge (§5a). |
| `degradations` | Strukturierte Liste dessen, was nicht sauber ging (§7). **Teilerfolg ist ein 200/`ready` mit Einträgen hier, nie ein 500.** |
| `usage` | `{"model_calls": int, "cost_eur": float}` — oder `null` = **ehrlich unbekannt** (§5b). Nie eine erfundene 0. |
| `budget_eur` | Der Kostendeckel, der für DIESEN Auftrag galt (am Submit eingefroren). |
| `error` | Nur bei `failed`: Fehlertext (Tail des Worker-Tracebacks — die Exception-Zeile steht am Ende). |
| `source` | Fakten zur Einreichung: Original-Dateiname, Endung, Bytes, Seitenzahl (`null` außer bei lesbaren PDFs). |

### 5a. Herkunft: Werte, Einheit, Garantien

Drei Werte, geordnet nach Vertrauensstufe:

- **`deterministisch`** — wörtlich extrahiert, garantiert kein Modell
  beteiligt. Der Text ist die Quelle.
- **`ocr`** — klassisches OCR: Zeichenerkennung ohne generativen Decoder
  (heute nicht vergeben; reserviert für den lokalen Engine-Pfad).
- **`modell`** — ein generativer Decoder hat den Text erzeugt oder erzeugt
  haben können. Solcher Text kann unter Druck **auffüllen** statt schweigen
  (Bake-off-Befund) — wer Wörtlichkeit braucht, prüft diese Einheiten nach.

**`provenance` ist immer eine Liste** — auch wenn sie nur einen Eintrag hat.
Es gibt keinen Skalar-Fall und keinen Typ-Switch beim Konsumenten. Was ein
Eintrag abdeckt, sagt `provenance_unit`:

- **`page`** — PDFs mit lesbarer Seitenzahl: ein Eintrag je Seite,
  Listenposition = Seitenreihenfolge (Eintrag 0 = Seite 1). Das ist die
  Granularität, in der die Pipeline real arbeitet und in der Degradation
  real passiert („Deckel greift ab Seite N").
- **`document`** — genau **ein** Eintrag für das ganze Dokument. Gilt für
  alle Nicht-PDF-Formate: die Office-/Web-Pipeline verarbeitet einen
  Element-Strom ohne stabile Seitengrenzen (DOCX-„Seiten" sind ein
  Renderer-Artefakt) — die ehrliche Einheit ist dort das Dokument.
  ⚠️ Gilt auch für ein **PDF, dessen Seitenzahl nicht lesbar war**: es wird
  keine Seiten-Granularität behauptet, die nicht belegt ist.

Die Einheit reist **in der Antwort selbst** — ein Konsument leitet sie nie
aus dem Format ab.

**Garantie der konservativen Aufrundung**: `deterministisch` wird nur
behauptet, wenn es **garantiert** ist. Kann eine Engine ihre Einheiten nicht
einzeln ausweisen, wird dokumentweit auf **`modell` aufgerundet** und die
Lücke als Degradation `provenance_document_only` benannt. Ein falsches
`modell` (Misstrauen, wo keins nötig war) ist der harmlose Fehler; ein
falsches `deterministisch` wäre unmarkierter Modelltext — genau der stille
Fehler, den der Bake-off zwölffach fand. *(Seit DOC-ENGINE weist der
Cloud-PDF-Pfad die Herkunft **echt je Seite** aus — die Aufrundung ist dort
Geschichte; die Garantie bleibt für jede künftige Engine, die ihre Einheiten
nicht trennen kann.)*

### 5b. `usage: null` heißt unbekannt, nicht null Euro

`usage` wird nur gemeldet, wenn der Lauf für sich buchführen kann:

- Lokale/deterministische Läufe melden sicher `{"model_calls": 0,
  "cost_eur": 0.0}`.
- Der Cloud-PDF-Pfad meldet seit DOC-ENGINE **echte Zahlen**: ein Call je
  Seite, Kosten aus `usage_metadata` des Modells (nie eine Schätzung; fehlt
  die Metadatenauskunft ausnahmsweise, wird der gemessene Seitenpreis
  gebucht statt 0 — sonst wäre der Deckel still entwaffnet).
- **`null`** bleibt Teil des Vertrags und heißt „ehrlich unbekannt": ein
  künftiges Backend ohne Selbst-Buchführung meldet `null`, nie eine
  erfundene 0. Wer Kosten bilanziert, behandelt `null` als „unbekannt,
  konservativ > 0 möglich".

## 6. Modus je Auftrag

`mode` ist ein Multipart-Formularfeld neben `file`:

- **fehlt** → es greift der Default aus der CONVERTER-Einstellung (s.u.);
  ohne gespeicherte Einstellung ist das seit **DOC-WEB `lokal`** (vorher
  `cloud`). ⚠️ Für Aufrufer, die nie ein `mode` mitgeben, hat sich damit
  das Ergebnis geändert: lokales Modell, 0 €, Herkunft `modell` — wer Cloud
  will, sagt `mode=cloud`. Grund (Oli-Entscheidung): derselbe Default
  treibt seit DOC-WEB auch den Browser-Knopf, und ein Code-Default `cloud`
  hätte den für einen Nutzer, der nie gewählt hat, still kostenpflichtig
  gemacht (~1,5 ct/Seite); mineru misst 0,9551 gegen 0,9809 Wort-F1 auf
  `01.gold` bei 0 € und ist ab ~2 Seiten schneller.
- **`cloud`** → beste verfügbare Qualität, Modell-Calls erlaubt, kostet Geld,
  unterliegt dem Kostendeckel (§7).
- **`lokal`** → keine Cloud-Calls, 0 €; seit DOC-LOCAL das lokale
  mineru-Modell (Herkunft `modell`, s. §10 — nicht mehr *beweisbar
  deterministisch*), Scans werden gelesen.
- **alles andere** → `400 {"error": "Ungültiger Modus. Erlaubt: 'cloud' oder
  'lokal'."}` — strikt gelesen, nur der exakte Wert schaltet; auch `""`,
  `"Cloud"` oder `" lokal"` sind 400.

Default-Einstellung (Session-Fläche, nichts für Service-Caller):

    GET /api/document-conversions/settings   → {"default_mode": "lokal"}
    PUT /api/document-conversions/settings   {"default_mode": "cloud"}

⚠️ **Eine Einstellung, zwei Eingänge** (DOC-WEB): derselbe Default
entscheidet auch, mit welcher Engine der Browser-Knopf
(`POST /transform-document`) ein PDF konvertiert. Es gibt bewusst keinen
zweiten Schalter.

Strict write (unbekannter Key / ungültiger Wert → 400, nichts geschrieben),
lenient read. Liegt im geteilten `User.settings_json` unter dem Namespace-Key
`document_api` — **jeder** Schreiber dieses Blobs geht über
`learn.write_settings_keys` (CLAUDE.md-Warnung), sonst löscht er fremde
Namespaces.

## 7. Kostendeckel und Degradationen

Jeder Auftrag trägt einen Kostendeckel (`budget_eur`), am Submit aus
`DOC_CONVERT_BUDGET_EUR` eingefroren (Start: **1,00 €**; eine Env-Änderung
preist keinen laufenden Auftrag um). Wird er erreicht, **läuft der Auftrag
lokal weiter statt abzubrechen** — der Aufrufer bekommt immer ein Ergebnis,
und jede lokal entstandene Seite trägt die geänderte Herkunft. Der Deckel
greift **zweistufig** (DOC-ENGINE, Semantik von Oli so entschieden):

1. **Preflight, ganz-oder-gar-nicht**: sagt schon die Schätzung
   (Seitenzahl × 1,48 ct, Bake-off-Messung,
   `DOC_CONVERT_CLOUD_CENT_PER_PAGE`), dass das Budget das Dokument nicht
   trägt, wird **kein einziger** Cloud-Call ausgegeben — das ganze Dokument
   läuft lokal (keine Mischqualitäts-Fragmente, kein Geld für ein Ergebnis,
   das ohnehin überwiegend lokal endete). Ein 280-Seiten-Dokument gegen
   1,00 € degradiert also vollständig, bevor ein Call läuft.
2. **Mid-flight, das Netz**: passiert die Schätzung den Preflight, reißen
   aber die **echten** per-Seite-Kosten (tokendichte Seiten) das Budget
   während des Laufs, schalten die verbleibenden Seiten auf den lokalen
   Pfad um — maximal eine Seite Überhang, die laufende Seite wird nie
   abgebrochen, jede Seite trägt ihre echte Herkunft.

Degradations-Einträge sind `{"code", "message", "pages"}`: `code` ist ein
stabiler snake_case-Slug für Maschinen, `message` deutsch für Menschen,
`pages` nennt betroffene Seiten (1-basiert) oder ist `null` = ganzer Auftrag.

| Code | Bedeutung |
|---|---|
| `budget_exceeded` | Der Deckel griff — Auftrag (teilweise) lokal statt cloud gelaufen. `pages` nennt die lokal entstandenen Seiten, sofern seitenweise bekannt (`null` beim Preflight-Fall = ganzer Auftrag). |
| `cloud_unavailable` | Modus `cloud` angefragt, aber kein Cloud-Backend konfiguriert (kein API-Key im Worker) — lokal konvertiert. |
| `provenance_document_only` | Eine Engine weist Herkunft nicht je Seite aus; konservativ dokumentweit als `modell` markiert (§5a). **Seit DOC-ENGINE nicht mehr vergeben** (der Cloud-PDF-Pfad attribuiert echt je Seite); bleibt im Vokabular für künftige Backends ohne Einheiten-Trennung. |
| `serializer` | Sammelcode für **alle Backend-Warnungen** des Konvertierwegs: der unstructured-Serializer musste Struktur aufgeben (deutsche Meldung, z.B. „Tabelle ohne text_as_html — als Fliesstext ausgegeben"), oder ein Werkzeug-Backend meldete etwas auf stderr. ⚠️ **Meldungsform bei Werkzeug-Warnungen**: deutscher Rahmen mit roh zitierter — meist englischer — Werkzeug-Ausgabe, z.B. `"pandoc meldete: [WARNING] Could not convert image …"` (bis 300 Zeichen Zitat; dieselbe Konvention wie das `error`-Feld, das rohe Traceback-Tails trägt). |
| `backend_fallback` | Das für das Format gewählte Backend lieferte nichts Verwertbares und der Bestands-Pfad übernahm. Zwei Fälle: trafilatura findet in einer HTML-Datei keinen Hauptinhalt → Element-Extraktion; die lokale mineru-Engine fällt aus (GPU belegt, Container-Fehler, Zeitlimit — seit DOC-LOCAL) → PyMuPDF-Textebene für die betroffenen Seiten, `pages` benennt sie, die Meldung zitiert die Werkzeug-Ausgabe. Das Ergebnis ist `ready`, der Pfadwechsel steht hier. |
| `scan_text_layer_empty` | **Seit DOC-WEB.** Begleitet einen `backend_fallback` der lokalen Engine: unter den auf die Textebene zurückgefallenen Seiten sind **Scans** — dort ist die Textebene *von Natur aus* leer, nicht durch Defekt. `pages` benennt genau diese Seiten (1-basiert), die Meldung sagt es („Seite 7 ist ein Scan, die Textebene ist dort leer."). Ein leerer Abschnitt im Markdown ist damit erklärt statt still serviert. Erkennung: Bildabdeckung > 70 % der Seitenfläche **und** Textdichte < 0,5 Zeichen/1000 pt² (`services/pdf_local.is_scanned_page`, der überlebende Rest des abgerissenen Seiten-Klassifikators). Tritt nie ohne einen `backend_fallback` auf. |

Die Liste ist offen — DOC-LOCAL kam ohne neuen Code aus
(`backend_fallback` trägt auch den Engine-Ausfall), DOC-WEB fügte
`scan_text_layer_empty` additiv hinzu; ein Konsument behandelt unbekannte
Codes als informativ, nicht als Fehler.

## 8. Idempotenz

Dedup-Schlüssel ist **(Besitzer, sha256 der Datei-Bytes, Modus,
Engine-Generation)**. Eine wiederholte Einreichung derselben Bytes im selben
Modus liefert **200** mit dem gespeicherten Stand (`deduped: true`, gleiche
`id`) statt eines neuen Jobs — auch während der Erste noch `pending` ist.
Der Dateiname ist egal.

⚠️ **Der Modus gehört zum Schlüssel**: ein `lokal`-Ergebnis beantwortet keine
`cloud`-Anfrage und umgekehrt — die beiden sind verschiedene
Qualitätszusagen (modellgestützt-ohne-Kosten vs. modellgestützt-mit-Kosten,
s. §10), und ein Dedup über die Modusgrenze würde still das falsche
Versprechen liefern.

⚠️ **Die Engine-Generation gehört zum Schlüssel** (seit DOC-LOCAL):
`DOC_CONVERT_ENGINE_GENERATION` in `services/document_conversions.py` wird
beim Submit in die Metadaten gestempelt und mitverglichen. Ohne sie wäre
Dedup **engine-blind** — live getroffen am 2026-08-16: eine `lokal`-Row aus
der Legacy-Ära (Textebene, `deterministisch×280`) beantwortete dieselbe
Datei dauerhaft, obwohl die mineru-Engine frisch deployt war, und es gab
keinen Bedienweg daran vorbei. Alte Rows tragen keine Generation → zählen
als 1 → matchen nie gegen die aktuelle; damit ist jeder Prä-DOC-LOCAL-Stand
genau einmal entwertet, ohne Migration. Die Generation ist **global** (nicht
je Format) und wird bei **jeder** Engine- oder Zusammensetzungs-Änderung
gebumpt (Kommentar an der Konstante — DOC-ROUTE eingeschlossen).

`failed`-Aufträge dedupen **nicht**: die erneute Einreichung derselben Datei
ist der Retry-Weg dieser API (es gibt bewusst keinen separaten
Retry-Endpunkt).

## 9. Fehlercodes gesammelt

| Status | Wann |
|---|---|
| 400 | Body-/Feldfehler (§3), ungültiger `mode`, Settings-Strict-Write — **oder** CSRF (POST ohne Bearer-Header und ohne CSRF-Token, §2) |
| 401 | Fehlender/falscher Bearer bei nicht eingeloggtem Aufrufer (generisch) |
| 404 | `id` unbekannt/fremd/falscher Typ (ununterscheidbar) |
| 413 | Upload > 100 MB |
| 503 | `DOC_CONVERT_TOKEN` nicht konfiguriert · kein Zielnutzer vorhanden |

## 10. Die Engine hinter der Form (Stand DOC-WEB)

Der Dienst ist ein **Router**: Format rein, gemessenes Backend raus
(Bake-off 2026-08-08, Entscheidungs-Doc). Die Form selbst — Felder, Werte,
Garantien — ist der Vertrag und bleibt bei jedem Backend-Tausch stehen.

**Ein Router, zwei Eingänge (DOC-WEB, 2026-08-21; Job-Modell seit
DOC-WEB-ASYNC, 2026-08-22):** derselbe Router (`services/document_router.py`
— `convert_non_pdf` / `convert_pdf`) treibt diesen Dienst **und** den
Browser. Für dieselbe Datei gibt es nur noch *eine* Qualität; das ist
strukturell garantiert (ein Aufruf, zwei Aufrufer, kein Nachformatieren)
und wurde je Format byte-identisch belegt. **Browser-PDFs sind seit
DOC-WEB-ASYNC Aufträge dieses Dienstes**: die Seite reicht sie
session-authentifiziert (Cookie + CSRF-Header) an
`POST /api/document-conversions` ein und pollt `GET …/<id>` — ohne
`mode`-Feld, der Default aus §6 gilt; `deduped`, `degradations` und `error`
aus der Antwort werden dort angezeigt. Eine Seitengrenze gibt es **nicht**
mehr (die frühere `MAX_SYNC_PDF_PAGES=12` schützte den einzigen
gunicorn-Worker vor einer synchronen Konvertierung; Messung 2026-08-21: er
bediente währenddessen *nichts* — nur der Auftrag gibt ihn frei). Nur die
Nicht-PDF-Formate laufen weiter synchron über `POST /transform-document`
(Sekunden, kein Container); dieser Sync-Pfad bleibt **kein** Teil dieses
Kontrakts (JSON `{markdown, filename, degradations}` für die eigene UI,
keine Herkunft, kein Job). Folge für die Library: jede Browser-PDF-
Umwandlung ist eine `document_conversion`-Zeile (Archiv, wie jeder
Dienst-Auftrag).

| Format | Backend | Herkunft in der Antwort |
|---|---|---|
| PDF, Modus `cloud` | gemini-nativ seitenweise (`services/pdf_cloud.py`, `media_resolution=medium`, Modell env-overridable via `PDF_VISION_MODEL`) | `page`, je Seite `modell` — greift der Deckel, übernimmt ab dieser Seite die lokale mineru-Engine (weiter `modell`, der `budget_exceeded`-Eintrag benennt den Umsprung) |
| PDF, Modus `lokal` | **mineru 3.4.4 VLM** im Geschwister-Container (`services/pdf_local.py`, seit DOC-LOCAL; ein memoisierter Lauf je gebrauchtem Seitenbereich, Seiten-Markdown aus der `content_list`); fällt die Engine aus → PyMuPDF-Textebene + `backend_fallback` | `page`, je Seite `modell` (0,00 €) — `deterministisch` nur im benannten Fallback |

⚠️ **Bedeutungsänderung von `mode=lokal` (DOC-LOCAL)**: bis dahin hieß
`lokal` *beweisbar deterministisch* (und Scan-Seiten kamen leer zurück);
seither heißt es **lokales Modell, kein Geld** — die Herkunft je Seite ist
ehrlich `modell`, die Kosten bleiben 0,00 €. Wer *Determinismus* braucht
(nicht bloß Kostenfreiheit), hat mit `lokal` keine Zusage mehr; ein
künftiger `mode=deterministisch` ist als Möglichkeit benannt, **nicht**
zugesagt.

⚠️ **Betriebsvoraussetzung Docker-Socket (root-äquivalent)**: der Worker
mountet `/var/run/docker.sock` und startet mineru als
**Geschwister-Container** auf dem Host-Daemon (GPU via `--gpus all`).
Der Socket ist **root-äquivalent auf dem Host** — bewusste, gesperrte
Entscheidung aus dem DOC-LOCAL-Sprint: die GPU ist nur während eines
Auftrags belegt (ein Dauer-Sidecar hielte 6,5 von 12 GB dauerhaft gegen
Olis ComfyUI/LoRA-Nutzung), und die Invokation bleibt wörtlich die
gemessene. Preis, akzeptiert: ~61 s Modell-Start je Auftrag und die
Socket-Vertrauensstellung. Das Image ist per Tag gepinnt
(`mineru:3.4.4`, Image-ID `6cc9e57ff5bd`, kein Registry-Digest — lokal
geladen); ein stiller `latest`-Rebuild trägt den Tag nicht und fällt
kontrolliert auf die Textebene statt still eine ungemessene Engine zu
fahren. ⚠️ **Nur der Worker hält den Socket** (DOC-WEB-ASYNC, 2026-08-22):
der aus dem Internet erreichbare Web-Container mountet weder Socket noch
Exchange-Bind und trägt keine `MINERU_*`/`DOC_LOCAL_*`-Verdrahtung —
Browser-PDFs erreichen die Engine ausschließlich als Auftrag über Redis.
`docker exec markdown-converter-web ls /var/run/docker.sock` muss
fehlschlagen; das ist der Abnahme-Beleg des Sprints.
| DOCX | pandoc `-f docx -t gfm --wrap=none` (Release-deb 3.10.1 im Image — die jammy-apt-Version 2.9 trug die Fußnoten-Kette nicht) | `document`, `deterministisch` |
| PPTX | markitdown 0.1.7 (einziger Kandidat mit Sprechernotizen) | `document`, `deterministisch` |
| HTML/HTM | trafilatura 2.2.0 + Metadaten-Kopf (`<title>`-Tag als `# `-Überschrift — TITLE-FIX greift; Autor/Datum aus `extract_metadata` als Kursivzeile); leere Extraktion → `backend_fallback` auf unstructured | `document`, `deterministisch` |
| EML, TXT, MD | unstructured + Serializer (Bestand; EML konkurrenzlos) | `document`, `deterministisch` |
| XLSX | **bewusst ungebaut** — 400 am Submit | — |

Alle Office-/Web-Backends sind modellfrei — `deterministisch` ist dort per
Konstruktion garantiert, `usage` ein sicheres 0/0.

Was mit DOC-WEB **weggefallen** ist (der Eigenbau `services/pdf_extraction/`
— fünf Tabellen-Detektoren, Ensemble, Multi-Page-Merge — war zuletzt nur
noch der Motor des Browser-Knopfs, gemessen wertlos: auf `01.gold`
identisch mit roher Textextraktion bei Tabellenzellen 0,0, auf Klasse 02
die halbe Wortmenge verloren): Seitentyp-Routing (`native`/`mixed`/
`scanned` entschied, ob eine Seite einen Vision-Call wert ist — beide
Engines lesen heute jeden Seitentyp), die seitentyp-abhängige Auflösung
LOW/HIGH (jetzt einheitlich MEDIUM, der Bake-off-Messwert), der
Tabellen-Merge über Seitengrenzen (geometrisch, nicht auf Markdown hebbar —
Backlog DOC-SPAN-MERGE) und eine kosmetische Markdown-Nachbereitung.

Implementierung: `app_pkg/document_api.py` · `app_pkg/documents.py` (Web) ·
`services/document_router.py` · `services/document_conversions.py` ·
`services/document_pipeline.py` · `services/office_backends.py` ·
`services/pdf_cloud.py` · `services/pdf_local.py` ·
`tasks.convert_document_task`.
Tests: `tests/test_document_api.py` · `tests/test_documents.py` ·
`tests/test_office_backends.py` · `tests/test_pdf_cloud.py` ·
`tests/test_pdf_local.py`.
