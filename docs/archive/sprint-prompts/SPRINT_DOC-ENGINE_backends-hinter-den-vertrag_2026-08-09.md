# SPRINT DOC-ENGINE — die echten Backends hinter den Vertrag

**Größe**: L (3 Phasen) · **Datum**: 2026-08-09 · **Vorhaben**: DOC-SVC

## Warum

Der Dienst steht: Endpunkt, zwei Auth-Wege, Job-Mechanik, Antwortform mit Herkunft und Degradation — alles live auf der Mintbox. **Dahinter liegt aber noch die Bestands-Fähigkeit**, also genau das, was der Bake-off als unzureichend gemessen hat. Dieser Sprint tauscht sie gegen die gemessenen Sieger aus.

**Er tauscht nur, was ohne GPU geht.** Der lokale PDF-Pfad (mineru im GPU-Sidecar), die Rettung von Routing und Multi-Page-Merge sowie das Abschalten der fünf Detektoren sind ein **eigener Sprint** (DOC-LOCAL) — zusammen wäre das XL, und die Splitting-Regel der Working Practice greift.

## Gegroundeter Ist-Zustand (Master — nicht neu herleiten)

**Der Backend-Vertrag steht** ([services/document_pipeline.py](../../../services/document_pipeline.py)): `page_fn(page_index_0based) → {markdown, origin, cost_eur}`, zwei Callables je Lauf (`cloud_page` / `local_page`), weil der Deckel genau dazwischen umschaltet. Die Mechanik ist mit Platzhalter testbelegt, das Payload fließt unverändert durch Reconcile bis in die GET-Antwort.

**Im Image liegt schon**: `pandoc`, `libreoffice`, `tesseract-ocr`, `poppler-utils` (Dockerfile:12–15) — **pandoc ist damit ohne neue Dependency verfügbar** und war bis heute ungenutzt. Neu wären `markitdown` und `trafilatura`.

**Gemessene Sieger je Format** ([Entscheidungs-Doc](../../doc_convert_entscheidung_2026-08-08.md)):

| Format | Wahl | tragende Zahl |
|---|---|---|
| DOCX | **pandoc** | Regel 3 **4/4** gegen **0/4** bei allen anderen |
| PPTX | **markitdown** | Recall 1,0, einziger mit Sprechernotizen |
| HTML | **trafilatura** + Metadaten-Schritt | <2 % Boilerplate gegen 31 % |
| EML | unstructured | funktional, konkurrenzlos |
| PDF Cloud | **gemini-nativ, `media_resolution=medium`** | 640/640 und 655/655 wertgenau; medium schlägt low **und** high |

**`media_resolution=medium`, nicht low.** Die frühere Empfehlung ist widerrufen: low fiel bei Bewertungsregel 1 auf 0/3 und erfand 67 statt 43 Ausfülllinien.

**Der lokale PDF-Pfad bleibt in diesem Sprint die Bestands-Engine ohne API-Key** — beweisbar deterministisch je Seite, aber auf Scans leer. Das ist bewusst: die Degradation braucht heute ein Ziel, und ein ehrlich schwaches ist besser als ein fehlendes. DOC-LOCAL ersetzt es.

## Gesperrte Entscheidungen

- **Ein Backend ist ein Adapter hinter dem bestehenden Vertrag.** Weder `document_pipeline.py` noch die Antwortform noch der Kontrakt werden angefasst — wenn doch etwas nicht passt, ist das ein **Befund** und gehört in den Bericht, nicht in einen stillen Vertragsumbau.
- **Herkunft ehrlich, konservativ aufgerundet.** `deterministisch` nur, wenn garantiert. Ein Modell-Backend liefert `modell`, auch wenn es „nur gelesen" hat.
- **Kein Anfassen von `services/pdf_extraction/`** — es bleibt der lokale Pfad, unverändert. Die Detektoren fallen in DOC-LOCAL.

---

# Phase 1 — Office und Web

Alles lokal, alles ohne Modell, alles ohne GPU. Deshalb zuerst: es ist die billigste Breite, die der Dienst kriegen kann.

## 1.1 Die vier Backends

- **DOCX → pandoc.** Schon im Image. Der Grund für die Wahl ist die Bild-Fußnoten-Link-Kette (Regel 3) — prüf am Gold-Dokument `corpus/gold/08.md`, dass sie ankommt, sonst ist die Wahl umsonst. ⚠️ Der Bake-off hat pandoc mit leicht schwächerem Text gemessen (0,941 gegen 0,964), Ursache waren „smart"-Typografie-Abweichungen; sieh nach, ob ein Schalter das ohne Nebenwirkung abstellt.
- **PPTX → markitdown.** Neue Dependency. Sprechernotizen sind das Unterscheidungsmerkmal — belegen.
- **HTML → trafilatura**, plus der von Oli entschiedene **Metadaten-Schritt** (Dachzeile, Titel, Autor, Datum), die trafilatura im Fließtext-Modus verwirft. ⚠️ **Prüf zuerst, ob trafilatura das selbst kann**, bevor du etwas danebenbaust — wenn ja, ist der „separate Schritt" ein zweiter Aufruf und keine zweite Bibliothek. Der Titel ist nicht kosmetisch: CONVERTER leitet den Library-Titel aus der ersten Überschrift ab (TITLE-FIX).
- **EML → unstructured.** Bereits vorhanden, nur verdrahten.

## 1.2 Was nicht dazugehört

**XLSX bleibt ungebaut** — kein Belegexemplar im Korpus, kein bekannter Bedarf. Ein 400 mit klarer Meldung ist die ehrliche Antwort.

## 1.3 Belege

Für jedes der vier Formate ein Lauf über das entsprechende Korpus-Exemplar, durch den **echten** Dienst (einreichen → pollen → Ergebnis), nicht nur durch die Adapterfunktion. Wo eine Gold-Fassung existiert (`08.md`), gegen sie prüfen.

## Stop
`pytest tests/` grün (Baseline **895**). **Commit + Push** `feat(DOC-ENGINE): Office- und Web-Backends (P1)`. Dann warten.

---

# Phase 2 — Der Cloud-PDF-Pfad

## 2.1 Die Spannung, die zuerst zu messen ist

Der Bake-off hat `gemini-nativ` mit **nativem PDF-Input über das ganze Dokument** gewonnen. Der Backend-Vertrag ist dagegen **seitenweise** — und das ist kein Formalismus: die mittendrin greifende Degradation, die DOC-API bewiesen hat, funktioniert nur, wenn das Backend Seiten einzeln liefern kann.

Beides zugleich geht nicht ohne Entscheidung:

- **Ganzes Dokument in einem Call**: die gemessene Qualität, aber der Deckel kann nur noch **vorher** entscheiden (der Preflight aus DOC-API), nicht mittendrin. Die Umschalt-Mechanik wäre für diesen Pfad toter Code.
- **Seitenweise oder in Blöcken**: die Umschalt-Mechanik lebt, aber weniger Kontext pro Call — und **weniger Kontext kann die Qualität kosten, die die Wahl begründet hat**.

⚠️ **Miss das, statt es zu entscheiden.** Du hast alles dafür: `corpus/gold/01.md` und `07.md`, das Harness in `corpus/bakeoff/harness/`, und die Bewertungsregeln. Fahre dieselbe Gold-Stichprobe einmal als ganzes Dokument und einmal seitenweise, und vergleiche mit derselben Metrik wie der Bake-off. **Nimm die Variante, die misst — und wenn der Unterschied klein ist, nimm die seitenweise**, weil sie die Degradation am Leben hält.

Die Messung gehört in den Bericht, mit Zahlen.

## 2.2 Das Backend

`media_resolution=medium`, nativer PDF-Input (nicht die gerenderten PNGs des Bestands-Pfads). Modell `gemini-3.6-flash`, env-overridable wie in DOC-FIX.

Jeder Call braucht seine **per-Call-Deadline** — das ist im Repo zweimal teuer gelernt worden (NARR-TIMEOUT, DOC-FIX). Kosten kommen aus `usage_metadata`, nicht aus einer Schätzung.

## 2.3 Der Deckel wird real

Bisher war er Platzhalter-getestet. Jetzt greift er an echten Kosten — und ein konkreter Fall im Korpus zeigt das sofort: **`12_grosses-pdf` sind 280 Seiten × 1,48 ct = 4,14 €** gegen einen Deckel von **1,00 €**. Das Dokument degradiert also vollständig auf lokal, bevor ein einziger Call läuft.

Das ist **korrektes Verhalten** und der beste verfügbare Beleg — fahr es und zeig die Antwort mit ihrer Degradation. Falls du dabei zu dem Schluss kommst, dass 1,00 € zu eng ist: **das ist ein Befund für den Bericht, keine Eigenmächtigkeit.** Oli setzt den Wert.

## Stop
Cloud-PDF fährt, Messung aus 2.1 belegt, Deckel-Fall gezeigt. **Commit + Push** `feat(DOC-ENGINE): Cloud-PDF-Backend (P2)`. Dann warten.

---

# Phase 3 — Wrap

- **Kontrakt-Doc** `docs/document_api_contract.md` nachziehen: welches Backend je Format, und was das für die Herkunftswerte bedeutet. Der Vertrag selbst ändert sich nicht — die Fußnote, welche Engine dahinter steht, schon.
- **CLAUDE.md**, **STATUS.md**, **BACKLOG.md** (Bullet-Guard).
- **Im Bericht benennen**: das Ergebnis der Ganz-gegen-seitenweise-Messung mit Zahlen · ob trafilatura die Metadaten selbst liefert · ob der pandoc-Typografie-Schalter existiert und was er kostet · und ob 1,00 € nach dem echten Deckel-Fall tragbar wirkt.
- **Memory**, falls sich eine übertragbare Lehre zeigt; nach dem Schreiben mit `ls` prüfen, dass Datei und Index-Zeile zusammenpassen.

## Nicht-Ziele

- **Kein mineru, keine GPU, kein Sidecar-Container, kein Compose-Touch.** Das ist DOC-LOCAL.
- **Kein Anfassen** von `services/pdf_extraction/` und **kein** Abschalten der fünf Detektoren — sie tragen bis DOC-LOCAL den lokalen Pfad.
- **Kein** Umbau von `document_pipeline.py`, der Antwortform oder des Kontrakts.
- **Kein** XLSX, **kein** SmartArt, **keine** Korrekturfläche.
- ⚠️ **Nicht auf der Mintbox arbeiten.** Sie ist Runtime, nicht Arbeitsplatz — beim Bake-off sind dort 72 unversionierte Ergebnisdateien liegengeblieben und haben Olis nächsten Deploy blockiert. Alles auf dem Mac, Deploy ist ohnehin Nicht-Ziel.
