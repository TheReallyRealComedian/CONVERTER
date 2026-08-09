# Document-API-Kontrakt — Dokument-Konvertierung als Dienst

Stand 2026-08-09 (Sprint DOC-API, P1+P2). Alle Pfade relativ zur Basis-URL des
CONVERTER-Stacks. Antworten `application/json`; die Einreichung ist
`multipart/form-data`. Dieses Dokument ist der Vertrag, den ein fremder Dienst
liest — was hier steht, gilt; was die Antwort trägt, steht hier.

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
einzeln ausweisen (die heutige Übergangs-Engine im Cloud-Modus), wird
dokumentweit auf **`modell` aufgerundet** und die Lücke als Degradation
`provenance_document_only` benannt. Ein falsches `modell` (Misstrauen, wo
keins nötig war) ist der harmlose Fehler; ein falsches `deterministisch`
wäre unmarkierter Modelltext — genau der stille Fehler, den der Bake-off
zwölffach fand.

### 5b. `usage: null` heißt unbekannt, nicht null Euro

`usage` wird nur gemeldet, wenn der Lauf für sich buchführen kann:

- Lokale/deterministische Läufe melden sicher `{"model_calls": 0,
  "cost_eur": 0.0}`.
- Die heutige Übergangs-Engine im Cloud-Modus meldet **`null`**: sie weist
  weder Calls noch Kosten aus, und eine 0 wäre eine Behauptung. Wer Kosten
  bilanziert, behandelt `null` als „unbekannt, konservativ > 0 möglich".
  Der Engine-Folge-Sprint ersetzt das durch echte Zahlen.

## 6. Modus je Auftrag

`mode` ist ein Multipart-Formularfeld neben `file`:

- **fehlt** → es greift der Default aus der CONVERTER-Einstellung (s.u.).
- **`cloud`** → beste verfügbare Qualität, Modell-Calls erlaubt, kostet Geld,
  unterliegt dem Kostendeckel (§7).
- **`lokal`** → garantiert ohne Modell-Calls; Ergebnis ist beweisbar
  `deterministisch`. Preis heute: gescannte Seiten bleiben leer (kein lokales
  OCR bis zum Engine-Sprint).
- **alles andere** → `400 {"error": "Ungültiger Modus. Erlaubt: 'cloud' oder
  'lokal'."}` — strikt gelesen, nur der exakte Wert schaltet; auch `""`,
  `"Cloud"` oder `" lokal"` sind 400.

Default-Einstellung (Session-Fläche, nichts für Service-Caller):

    GET /api/document-conversions/settings   → {"default_mode": "cloud"}
    PUT /api/document-conversions/settings   {"default_mode": "lokal"}

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
und jede lokal entstandene Seite trägt die geänderte Herkunft. Heute wird der
Deckel **vor** dem Lauf geprüft (Seitenzahl × 1,48 ct, Bake-off-Messung,
`DOC_CONVERT_CLOUD_CENT_PER_PAGE`); mit der seitenweisen Engine greift er
mitten im Dokument (die Mechanik steht und ist testbelegt — maximal eine
Seite Überhang, die laufende Seite wird nie abgebrochen).

Degradations-Einträge sind `{"code", "message", "pages"}`: `code` ist ein
stabiler snake_case-Slug für Maschinen, `message` deutsch für Menschen,
`pages` nennt betroffene Seiten (1-basiert) oder ist `null` = ganzer Auftrag.

| Code | Bedeutung |
|---|---|
| `budget_exceeded` | Der Deckel griff — Auftrag (teilweise) lokal statt cloud gelaufen. `pages` nennt die lokal entstandenen Seiten, sofern seitenweise bekannt. |
| `cloud_unavailable` | Modus `cloud` angefragt, aber kein Cloud-Backend konfiguriert (kein API-Key im Worker) — lokal konvertiert. |
| `provenance_document_only` | Die Übergangs-Engine weist Herkunft nicht je Seite aus; konservativ dokumentweit als `modell` markiert (§5a). |
| `serializer` | Der Office-Serializer musste Struktur aufgeben (z.B. Tabelle ohne HTML-Repräsentation als Fließtext). `message` trägt den Befund. |

Die Liste ist offen — neue Codes kommen mit dem Engine-Sprint dazu; ein
Konsument behandelt unbekannte Codes als informativ, nicht als Fehler.

## 8. Idempotenz

Dedup-Schlüssel ist **(Besitzer, sha256 der Datei-Bytes, Modus)**. Eine
wiederholte Einreichung derselben Bytes im selben Modus liefert **200** mit
dem gespeicherten Stand (`deduped: true`, gleiche `id`) statt eines neuen
Jobs — auch während der Erste noch `pending` ist. Der Dateiname ist egal.

⚠️ **Der Modus gehört zum Schlüssel**: ein `lokal`-Ergebnis beantwortet keine
`cloud`-Anfrage und umgekehrt — die beiden sind verschiedene
Qualitätszusagen (deterministisch-mit-Lücken vs. modellgestützt-mit-Kosten),
und ein Dedup über die Modusgrenze würde still das falsche Versprechen
liefern.

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

## 10. Was dieser Kontrakt (noch) nicht ist

Die Engine hinter der Form ist die **heutige** Fähigkeit (PDF:
`pdf_extraction`; Office/Web: `unstructured`-Serializer). Der Engine-Sprint
(Router, gemini-nativ cloud / mineru lokal, echte per-Seite-Herkunft, echte
`usage`-Zahlen, Mid-Flight-Deckel) tauscht das Backend **hinter** dieser
Form; die Form selbst — Felder, Werte, Garantien — ist der Vertrag und
bleibt. Der bestehende Web-Pfad `POST /transform-document` ist unberührt und
kein Teil dieses Kontrakts.

Implementierung: `app_pkg/document_api.py` ·
`services/document_conversions.py` · `services/document_pipeline.py` ·
`tasks.convert_document_task`. Tests: `tests/test_document_api.py`.
