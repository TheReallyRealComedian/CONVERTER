# Gold-Fassungen — was du nachprüfen musst

**Erstellt**: 2026-08-02 · **Verfahren**: drei Transkriptoren (Opus 5, `effort: max`), jeder gegen die Originalquelle, danach je ein unabhängiger Prüfer, der gegen dieselbe Quelle **nach Fehlern gesucht** hat statt nach Bestätigung.

## Das Wichtigste zuerst

**Kein erfundener Inhalt, in keiner der drei Dateien** — token-genau gegengeprüft:

| | Gegenprobe | Ergebnis |
|---|---|---|
| `01.md` | Wort-Diff Seite 4 | 616 Quell- gegen 612 Entwurfs-Token, **null** unerklärte Abweichungen (Rest: Ligatur-/Akzent-Normalisierung) |
| `07.md` | bidirektionaler Multiset-Vergleich | **0** Wörter im Entwurf ohne Gegenstück in der Quelle, 0 in Gegenrichtung |
| `08.md` | Wort- und Zell-Diff | 226/226 Wörter, 50/50 Zellen (Tab. 1), 13/13 Zellen (Tab. 2) — **identisch** |

Alle drei Urteile: *brauchbar mit Korrekturen*. Das heißt: der Wortlaut steht, die offenen Punkte sind **Struktur- und Konventionsfragen** — und die gehören ohnehin dir.

---

## ✅ Entschieden (Oli, 2026-08-02) — erledigt

| | Entscheidung | Folge |
|---|---|---|
| **Tabelle I und II** (`01.md`) | **beide perfekt getroffen** | Der Punkt mit dem höchsten Rate-Anteil ist zu. Keine Änderung. |
| **Fettbalken** (`07.md`) | `##` **bleibt** — „vom Font her eine Überschrift, sitzt aber in einer vollständig gemergten ersten Tabellenzeile" | Datei unverändert, aber **Bewertungsregel 1** unten schützt die Messung. |
| **Mathematik** (`01.md`) | **LaTeX zulassen** | Umgesetzt: 32 Ersetzungen, keine Plaintext-Indizes mehr. **Bewertungsregel 2**. |
| **Bindestrich** (`08.md`) | **korrekt aufgelöst** — steht am Zellenende, und in Word wird das manuell gesetzt, um Trennungen in Tabellen zu erzwingen | Keine Änderung. |
| **Fußnote am Bild** (`08.md`) | **Bildbezug explizit machen** | Umgesetzt: `![Bild 1]()[^1]`. **Bewertungsregel 3**. |

---

## Bewertungsregeln für den Bake-off

Drei Stellen der Gold-Fassungen kodieren eine **Wahl**, keine Wahrheit. Wer dort auf Zeichengleichheit prüft, misst die Zustimmung zu Olis Konvention statt die Fähigkeit des Werkzeugs — und verliert damit den Wert der übrigen Seite gleich mit.

**Regel 1 — `07.md`, die drei Fettbalken.** Die Gold-Fassung schreibt `##`. **Gleichwertig** ist eine verbundene Kopfzeile (`<th colspan="5">` als erste Zeile der jeweiligen Tabelle). Begründung: Oli hat beides bestätigt — es *ist* typografisch eine Überschrift und *ist* strukturell eine vollständig gemergte Zeile. Ein Werkzeug, das die zweite Lesart liefert, hat nicht schlechter gearbeitet.

**Regel 2 — `01.md`, Tief- und Hochstellungen.** Die Gold-Fassung schreibt LaTeX (`$\ell_{rand}$`). **Gleichwertig** ist jede Notation, die die Unterscheidung **erhält** — Unicode-Tiefstellungen, wo sie existieren, oder andere explizite Auszeichnung. **Nicht** gleichwertig ist das Einebnen (`ellrand`, `Pout`): dabei geht die Information verloren. Begründung: im PDF steht keine Auszeichnung, sondern typografische Positionierung; eine bestimmte Markup-Form zu verlangen misst Darstellungsgeschmack. Gemessen gehört, **ob die Unterscheidung überlebt**.

⚠️ Unicode allein reicht für diesen Korpus nachweislich nicht: es gibt keine Tiefstellung für b, c, d, f, g, q, w, y, z — `ℓ_rand`, `f_mech`, `Δ_Sub`, `Δ_Diss` und `Δ_G` sind damit nicht schreibbar. Das war der Grund für LaTeX.

**Regel 3 — `08.md`, die Abbildung.** Die Gold-Fassung schreibt `![Bild 1]()`. **Gleichwertig** ist jede Bild-Syntax an dieser Stelle mit beliebigem Ziel. Entscheidend ist nur zweierlei: dass die Abbildung überhaupt markiert wird, und dass die Fußnote an ihr hängt. Begründung: die Quelle bettet das Bild ohne Alternativtext ein (`docPr/@name = "Bild 1"`, `descr` fehlt); ob ein Pfad entsteht, hängt davon ab, ob das Werkzeug Assets extrahiert.

### Quell-Eigenheiten, die ein Werkzeug NICHT reparieren darf

Wer das „korrigiert", hat einen Fehler gemacht:

- `08.md`: **`Markteilnehmer`** (ein t) · **`19.Auflage`** und **`S.189.`** ohne Leerzeichen nach dem Punkt.
- `01.md`: die Bildunterschrift von TABLE II schreibt **`lreal, lrand, lpow`** mit kleinem L, die Spaltenköpfe **derselben** Tabelle **`ℓreal ℓrand ℓpow`** mit Skript-Ell. *Am Original nachgemessen* — eine Satz-Inkonsistenz der Quelle, kein Transkriptionsfehler. In der Gold-Fassung entsprechend `$l_{real}$` gegen `$\ell_{real}$`.
- `01.md`: die Kursivsetzung von *E. coli* ist im Original asymmetrisch (nur in TABLE I).

---

## Historisch — die Entscheidung, die zu Regel 1 geführt hat

**`07.md`, die drei Fettbalken: Überschriften oder verbundene Tabellenköpfe?**

Der Entwurf hat sie als `##`-Überschriften gesetzt. Der Prüfer widerspricht und hat den stärkeren Fall:

- **Dieselbe Messung, gegensätzlich ausgewertet.** Im Vektorraster laufen die Außen-Vertikalen bei `x=56,6` und `x=550,0` **durch** die Balkenbänder, und innerhalb der Bänder gibt es **keine** inneren Vertikalen. Das ist exakt die Signatur, mit der der Entwurf an anderer Stelle einen `colspan=5` begründet — hier hat er sie anders gelesen.
- **Im Scan sitzt der Balken sichtbar innerhalb des Tabellenrahmens** und teilt dessen Außenkanten.
- **Dein eigenes README ist der Kronzeuge**: `07_formular-punktlinien/README.md` kündigt für Seite 2 ausdrücklich *„eine Tabelle mit verbundenem Kopf (Ehegatte | Kind | Kind | Kind)"* an. Unter der Entwurfs-Lesart hat **keine** Tabelle der Seite einen verbundenen Kopf.

**Wenn du dem Prüfer folgst**, werden alle drei Tabellen zu rohem `<table>` mit `<tr><th colspan="5">…</th></tr>` als erster Zeile, und Konvention 1 (Überschriften) greift für die Balken gar nicht. Das ist ein größerer Eingriff in die Datei — deshalb steht er hier oben und nicht in der Liste.

Entscheide das zuerst; alles andere an `07.md` hängt davon ab.

---

## Was noch offen ist

Alle fünf Punkte, die eine Entscheidung brauchten, sind entschieden (Tabelle oben). Was unten im ungekürzten Befund steht, ist **mittleres und niedriges Risiko** — Formalien, die keine Wahl mehr verlangen.

Wer weitermacht, kann die Restliste durchgehen oder sie stehen lassen: sie dokumentiert, wo die Gold-Fassungen weniger scharf sind als anderswo, und das ist für die Auswertung des Bake-offs nützlich, auch ohne Korrektur.

**Ein Folge-Item, das nicht hierher gehört, aber aus der LaTeX-Entscheidung folgt**: Lernkarten rendern **keine** Mathematik — KaTeX wird nur in `markdown_converter.html` und `library_detail.html` geladen, nicht in `review.html`. Wird konvertierter Dokumenttext zu Kartenmaterial, steht `$\ell_{rand}$` roh auf der Karte. Notiert im BACKLOG.

---

## Zwei methodische Warnungen

**Eine von einem Modell geschriebene Gold-Fassung ist gegenüber Modell-Kandidaten nicht neutral.** Wo der Transkriptor geraten hat, hat er womöglich geraten wie Gemini raten wird — und dann misst der Bake-off an dieser Stelle gar nichts. Das ist der Grund, warum die Unsicherheitsliste kein Anhang ist: **jede Stelle, die du nicht selbst prüfst, ist ein blinder Fleck der späteren Messung.** Die Prüfer haben Fehler *gesucht*, nicht bestätigt — aber sie sind dieselbe Sorte Werkzeug.

**Quell-Eigenheiten sind kein Transkriptionsfehler.** In `08.md` steht `Markteilnehmer` (ein t), `19.Auflage` und `S.189.` ohne Leerzeichen — **so steht es in der Quelle**, und so muss es in der Gold-Fassung stehen. Ein Konverter, der das „korrigiert", hat einen Fehler gemacht, keinen Dienst. Der Prüfer hat angemerkt, dass diese Stellen als schützenswert markiert gehören, damit sie später niemand „aufräumt".

---

## Wie du korrigierst

Die drei Dateien sind normale Markdown-Dateien in diesem Ordner. Ändere sie direkt. Wenn du eine Stelle klärst, die unten als Unsicherheit steht, streich sie hier raus — dann bleibt am Ende stehen, was noch offen ist.

`⟪UNLESBAR: …⟫` markiert Stellen, an denen der Transkriptor bewusst ein Loch gelassen hat statt zu raten. Die sind Pflicht.

---

Alles Weitere ist der ungekürzte Befund, pro Datei: erst was die Gegenprüfung gefunden hat, dann die Selbstauskunft des Transkriptors nach Risiko sortiert, dann die Konventions-Entscheidungen — die sind **keine Fehler**, sondern Auslegungen, die vertretbar anders ausfallen können.

---

# 01.md
**Urteil der Gegenprüfung**: brauchbar mit korrekturen

## Gegenprüfung — Erfunden (schlimmster Fehlertyp)
- KEIN erfundener Inhalt gefunden. Token-Diff Seite 4 (616 Quell- vs. 612 Entwurfs-Token) ergibt NULL unerklärte Abweichungen — jede Differenz ist Ligatur-/Akzent-Normalisierung, die Kumar-Leerstelle oder die γ-Index-Reihenfolge, alle im Entwurf dokumentiert. Beide Tabellen habe ich unabhängig aus den 487 Linien-Rechtecken des Content-Streams (Spaltengrenzen TABLE I: 53.7/156.7/207.6/254.4/296.6/343.4/385.5/432.6/546.0/563.9; TABLE II: 53.7/142.4/200.7/246.8/280.7/317.5/354.3/379.4/403.9/425.3/546.7/563.9) rekonstruiert: alle 17 bzw. 20 Datenzeilen stimmen zellweise überein, inklusive aller LEEREN Zellen (WWW/Kumar κ leer; WWW,site nur γ_in=1.94; Sexual contacts nur γ=3.4; Citation nur γ_in=3; Protein S. cerev. κ leer). Auch die Ausreißer sind echt: 460.902 mit Punkt, 6.36 - 6.18 absteigend, Nr. 14/14/16.
- Einzige Zeichen im Entwurf ohne Gegenstück in der Quelle: die Marker `_` und `^` (z.B. `P_out(k)`, `k^−γ_out`, `ℓ_rand`, `γ_in^dom`). Im PDF stehen dort echte typografische Tief-/Hochstellungen (Fonts CMMI6/CMR6) ohne jedes Trennzeichen. Der Entwurf legt das als erste Unsicherheit mit Risiko 'hoch' offen — es ist bewusste Notation, keine Halluzination, aber es sind Bytes, die kein Werkzeug aus der Quelle ableiten kann.
- Vier eingefügte Leerzeichen (∼ k 2×, 'ℓ and', '⟨k⟩ and', 'cutoff κ') — ich habe alle vier nachgemessen und BESTÄTIGT: Lücken 3.64 / 3.49 / 3.61 pt gegen 3.6 pt Normal-Wortabstand. Die Gegenprobe des Entwurfs stimmt ebenfalls: 'P(k)' in der TABLE-II-Bildunterschrift hat 1.32/0.24 pt, dort steht im Entwurf korrekt KEIN Leerzeichen.

## Gegenprüfung — Fehlend
- Kein undokumentiert fehlender Inhalt. Seite 8 trägt außer den zwei Tabellen samt Bildunterschriften und der Seitenzahl '8' nichts (verifiziert über get_text der ganzen Seite). Seite 4 ist vollständig bis auf die Grafik-Textebene und die Seitenzahl.
- Bewusst weggelassen und offengelegt: die Achsen-/Panel-Beschriftungen der zwei Plots unten rechts auf Seite 4 (10⁰…10⁻¹⁰, 10⁻²…10⁶, Pout(k), Pin(k), k, (a), (b) — in der Textebene ~38 Zeilen/42 Spans, alle Helvetica). Das ist als Unsicherheit mit Messhinweis vermerkt, aber es bleibt der Punkt, an dem echte Werkzeuge systematisch 'falsch' aussehen werden.
- Seitenzahlen '4' und '8' weggelassen — korrekt nach Konvention 6. Kolumnentitel/Journal-Header gibt es auf diesen beiden Seiten tatsächlich nicht (verifiziert).

## Gegenprüfung — Struktur
- Keine Struktur-Fehler gefunden. Alles einzeln gegengeprüft: (a) Der spaltenübergreifende Absatz ist korrekt EIN Absatz — die erste rechte Spaltenzeile ('nodes being connected…') steht bei x=317.0, Absatzanfänge stehen bei x=327.0; die Einrückungs-Messung des Entwurfs ist reproduzierbar. (b) Absatzgrenzen 'Note that' (x=327.0), 'Despite the large number' (x=327.0), 'The directed nature' (x=327.0) und 'Albert, Jeong and Barabási' (x=64.0) sind alle echte Absatzanfänge. (c) Lesereihenfolge linke Spalte komplett (Text → Gl. 3 bei y=204 → FIG.-1-Unterschrift bei y=496 → Absatz ab y=576), dann rechte Spalte — entspricht der physischen Anordnung. (d) Keine Überschriften auf beiden Seiten — korrekt, Konvention 1 greift hier nicht. (e) Keine verbundenen Zellen: das vertikale Raster ist in jedem Band vollständig (TABLE I 10 Linien × 19 Bänder, TABLE II 12 × 21); die zweizeilige Yook/Pastor-Satorras-Zelle ist eine echte Einzelzelle (zwischen y=124.0 und y=146.3 fehlt die horizontale Linie). GFM-Pipe-Tabellen sind also nach Konvention 2 richtig, HTML wäre unnötig. (f) Spaltenzahlen konsistent: 9 bzw. 11, Kopf, Trennzeile und alle Datenzeilen. (g) Genau ein `---` zwischen den Seiten.

## Gegenprüfung — Konvention verletzt
- Vorgabe 'Mathematische Symbole als Unicode übernehmen, wie sie in der Quelle stehen … KEIN LaTeX erfinden': `k^−γ_out`, `P_out(k)`, `ℓ_rand`, `γ_in^dom` sind LaTeX-nahe Plaintext-Notation. Die Begründung des Entwurfs trägt sachlich (Unicode hat kein tiefgestelltes 'd' für 'rand' und kein 'w' für 'pow' — ich habe das geprüft: ₒᵤₜ und ᵢₙ und ᵣₑₐₗ wären setzbar, ᵣₐₙd und ₚₒw nicht), und die Vorgabe bietet keinen Ausweg. Aber es ist eine Abweichung, und der Bruch innerhalb EINER Zeile (Zahlen als 10⁻⁴ in Unicode, Buchstaben als `_rand`) muss vor der Nutzung als Maßstab entschieden werden — sonst misst man Werkzeuge gegen eine Notation, die kein Konverter je produziert.
- Vorgabe 'Abbildungen: … vermerke in den Unsicherheiten, dass die Abbildung selbst nicht im Markdown ist': Für FIG. 1 (Grafik im linken Spaltendrittel, y≈215–490) fehlt dieser Vermerk. Der Entwurf hat die Bildunterschrift korrekt als Absatz gesetzt, diskutiert in den Unsicherheiten aber ausschließlich die zweite Abbildung (die FIG.-2-Plots). Der geforderte Satz zur FIG.-1-Grafik steht nirgends.

## Gegenprüfung — Vom Autor übersehen
- FIG. 1 — die Abbildung selbst fehlt im Markdown, das ist nirgends notiert (die Vorgabe verlangt genau diesen Vermerk). Nur die zweite Abbildung wird in den Unsicherheiten behandelt. Stelle: Seite 4, linke Spalte zwischen Gleichung (3) und der FIG.-1-Unterschrift.
- Tausender-Leerzeichen ('325, 729', '52, 909'): Die Begründung sagt 'echtes Leerzeichen im Content-Stream … im Satzbild sichtbar' — beides stimmt, verschweigt aber die entscheidende Größenordnung. Ich habe es zeichengenau gemessen: das Leerzeichen nach dem Komma ist 1.64 pt breit, ein normaler Wortabstand auf derselben Seite 3.6 pt. Es ist ein DÜNNES Mathe-Komma-Leerzeichen. In Markdown wird daraus ein volles Leerzeichen, also eine Verbreiterung um Faktor 2. Wer nur die Unsicherheitszeile liest, entscheidet auf falscher Grundlage für 'beibehalten'. Betrifft ~30 Zahlen auf beiden Seiten.
- TABLE II, Zeile 'Internet, domain∗', Spalte κ ('30 − 40'): Der Entwurf behauptet 'Spannbreiten-Leerzeichen beidseitig sind echt (gemessen)'. Zeichengenau nachgemessen stimmt das nur links: dort steht ein CMSY9-Leerzeichen-Glyph (258.29→260.40, 2.11 pt). Rechts vom Minus (267.56→269.64, 2.08 pt) gibt es KEIN Glyph, nur eine Positionierungslücke. Optisch gleichwertig, kodierungsseitig nicht — und beide Lücken sind schmaler als die 3.05–3.07 pt der normalen Bindestrich-Bereiche in derselben Zeile. Ein Werkzeug, das hier '30−40' oder '30 −40' ausgibt, ist näher an der Textebene als die Gold-Fassung.
- Kursiv-Asymmetrie bei 'E. coli' ist zwar erwähnt ('2×, nur in TABLE I'), aber NICHT als Original-Inkonsistenz gekennzeichnet — und genau das ist sie. Ich habe die Fonts geprüft: TABLE I y=226.9/238.3 = CMTI9 (kursiv), TABLE II y=570.5 'Metabolic, E. coli' = CMR9 (aufrecht), ebenso 'S. cerev.' y=581.9 = CMR9. Dasselbe Lebewesen, zwei Auszeichnungen auf einer Seite. Das gehört in dieselbe Kategorie wie die korrekt gemeldete 460.902-vs-460, 902-Inkonsistenz, sonst wird ein Werkzeug, das TABLE II ebenfalls kursiviert, ohne erkennbaren Grund bestraft.
- Weitere unmarkierte Original-Inkonsistenzen zwischen den Tabellen, alle korrekt transkribiert aber nicht als Stolperstellen benannt: 'Silwood park' hat in TABLE I ℓ=3.40, in TABLE II ℓ_real=3.4 (nachgestellte Null nur einmal); 'Neurosci. coauthorship' hat ⟨k⟩=11.5 in TABLE I, 'Coauthors, neuro.∗' ⟨k⟩=11.54 in TABLE II. Beides an denselben Netzwerken.
- 'C. Elegans' (TABLE I, letzte Zeile) trägt in der Quelle ein großes E — biologisch falsch, aber so gesetzt (Font CMTI9 für 'C.' UND 'Elegans', verifiziert). Korrekt übernommen, aber nicht als Original-Eigenheit gelistet; ein Werkzeug oder ein Korrektor 'repariert' das sonst still zu 'C. elegans'.
- Hochgestelltes Minus: in '1.8 × 10⁻⁴' verwendet der Entwurf U+207B (SUPERSCRIPT MINUS), die Quelle setzt an dieser Stelle U+2212 in einem 6-pt-Font (CMSY6). Das ist eine Codepoint-Änderung, die unter den Entscheidungen nur als 'numerische Exponenten als Unicode-Hochstellung' pauschal abgedeckt ist. Betrifft die vier 10⁻⁴/10⁻⁵-Zellen in TABLE I.

## Unsicherheiten des Autors (17)

### [HOCH] Die Auftragsbeschreibung passt nicht auf die Seite. Seite 8 trägt ZWEI Tabellen: TABLE I (9 Spalten, 17 Datenzeilen) UND TABLE II (11 Spalten, 20 Datenzeilen), beide mit vollständigem Linienraster (Rahmen, alle Spalten- und Zeilentrenner). Ich habe beide transkribiert, weil eine Gold-Fassung für „Seite 8" alles auf der Seite enthalten muss.
- **Stelle**: Ganze Seite 8 — Auftragstext: „Seite 8 ist TABELLE I — eine grosse, LINIENLOSE Tabelle (~27 Zeilen × 9 Spalten)"
- **Warum**: Wenn nur TABLE I gemeint war, enthält diese Gold-Fassung ~40 % Material zuviel und jede spätere Messung bestraft Werkzeuge, die korrekt arbeiten. Umgekehrt: hätte ich TABLE II weggelassen, wäre die Gold-Fassung unvollständig. Die Fehlbeschreibung („linienlos") stammt vermutlich daher, dass die Linien als 1×1-Pixel-Inline-Bilder gezeichnet sind (dvips-Rules) — `page.get_drawings()` liefert 0 Objekte, `find_tables()` findet 0 Tabellen. Das erklärt die Diagnose, ändert aber nichts daran, dass die Linien optisch da sind.
- **Wie prüfen**: Seite 8 im PDF-Betrachter öffnen: zwei gerahmte Tabellen mit Linien. Dann entscheiden, ob die Gold-Fassung TABLE II behalten soll. Falls nein: Zeilen ab „TABLE II. The scaling exponents…" bis Dateiende löschen.

### [HOCH] Tief- und Hochstellungen mit Buchstaben sind als Plaintext-Notation `_name` / `^name` geschrieben. Das ist eine ERFUNDENE Notation — im PDF stehen dort echte typografische Tief-/Hochstellungen ohne jedes Zeichen dafür.
- **Stelle**: Durchgehend — z.B. `P_out(k)`, `γ_in`, `ℓ_rand`, `C_rand`, `k^−γ_out`, `γ_in^dom`, `l_pow`
- **Warum**: Es gibt keine verlustfreie Alternative innerhalb der Vorgaben. Unicode-Tiefstellungen sind unvollständig: für „d" (in `rand`) und „w" (in `pow`) existiert kein Subscript-Zeichen. Eine reine Unicode-Lösung wäre also innerhalb derselben Tabellenkopfzeile inkonsistent (`γₒᵤₜ` ginge, `ℓ_rand` nicht). LaTeX war verboten. Flaches Zusammenschreiben („Pout(k)", „k−γout") wäre buchstabengetreu, aber sachlich falsch — „k−γout" liest sich als Subtraktion.
- **Wie prüfen**: Entscheiden, ob `_`/`^` die gewünschte Konvention ist. Betroffene Stellen: Seite 4 alle γ/P/C-Vorkommen; Seite 8 beide Tabellenköpfe und beide Bildunterschriften. Prüfen ist rein mechanisch — grep auf `_` und `^`.

### [MITTEL] Zwischen „Kumar" und „et al." steht KEIN Leerzeichen — weder in der Textebene noch im Satzbild (gemessener Abstand ≤ 0,5 pt gegen ~3,5 pt bei normalen Wortabständen).
- **Stelle**: Seite 4, linke Spalte, Zeile 4: „…Barabási 1999, Kumar*et al.* 1999). Since the…"
- **Warum**: Das ist ein Satzfehler des Originals (fehlendes `\ ` vor `\emph`). Ich habe ihn getreu übernommen statt ihn zu reparieren. Beachte: dieselbe Quellenangabe steht 8 Zeilen später als „Kumar *et al.* (1999)" — dort MIT Leerzeichen. Die Gold-Fassung ist an dieser Stelle also absichtlich in sich uneinheitlich.
- **Wie prüfen**: Seite 4 bei 400 % zoomen, Zeile „bert, Jeong, Barabási 1999, Kumaret al. 1999). Since the". Wenn Quelltreue hier nicht gewollt ist, Leerzeichen einfügen.

### [MITTEL] Nach dem Tausender-Komma steht ein Leerzeichen. Ich habe es beibehalten.
- **Stelle**: Seite 4: „325, 729 nodes", „259, 794 sites", „153, 127 sites"; Seite 8: „1, 520, 251", „4, 941", „3, 015 - 4, 389", „460, 902" u.v.m.
- **Warum**: Es ist ein echtes Leerzeichen im Content-Stream (LaTeX-Mathmodus setzt `$325,729$` mit Komma-Abstand) und es ist im Satzbild sichtbar. Ein Mensch würde „325.729" bzw. „325,729" schreiben. Beide Lesarten sind vertretbar; ich habe zugunsten der Quelltreue entschieden, weil das Entfernen eine stille Korrektur wäre.
- **Wie prüfen**: Ein beliebiges Vorkommen im PDF zoomen — der Abstand ist gut sichtbar. Falls normalisiert werden soll: alle `, ` innerhalb von Zahlen ersetzen (ca. 30 Stellen).

### [MITTEL] Diese Abbildung hat auf Seite 4 KEINE Bildunterschrift; die Unterschrift „FIG. 2. Degree distribution of the World-Wide Web…" beginnt erst oben auf Seite 5. In der Gold-Fassung steht deshalb an dieser Stelle nichts. Zusätzlich habe ich die 42 Textelemente INNERHALB der Abbildung weggelassen (Achsenbeschriftungen 10⁰…10⁻¹⁰ und 10⁻²…10⁶, „P_out(k)", „P_in(k)", „k", „(a)", „(b)" — alle in Helvetica, alle in der Textebene vorhanden).
- **Stelle**: Seite 4, unteres rechtes Viertel — die zwei doppellogarithmischen Plots (a) und (b)
- **Warum**: Die Vorgabe sagt, statt eines Platzhalters solle die Bildunterschrift stehen — hier gibt es keine. Die Achsenbeschriftungen als Absatz auszuschreiben ergäbe sinnfreies Rauschen; sie gehören zur Grafik, die laut Vorgabe nicht im Markdown ist. Das ist aber eine bewusste Auslassung von Textebenen-Inhalt und muss beim Messen berücksichtigt werden, sonst gilt jedes Werkzeug, das die Achsenlabels ausgibt, als „falsch".
- **Wie prüfen**: Seite 4 unten rechts ansehen (kein Unterschriftstext unter den Plots) und Seite 5 oben links (dort beginnt „FIG. 2."). Entscheiden, ob die 42 Achsen-Elemente in der Gold-Fassung erscheinen sollen.

### [MITTEL] Kursivierung ist nur dort gesetzt, wo sie textlich ist. Mathematische Variablen sind im Original ebenfalls kursiv (Font CMMI), erscheinen hier aber ohne Auszeichnung.
- **Stelle**: Durchgehend — `*et al.*`, `*E. coli*`, `*C. Elegans*` sind kursiv ausgezeichnet; `P`, `k`, `C`, `γ`, `ℓ` NICHT
- **Warum**: Die sechs Konventionen sagen nichts über Hervorhebung. Mathe-Kursive ist eine Satzkonvention für Variablen, keine Betonung — `*C* = 0.1078` wäre irreführend. Die Grenzziehung ist trotzdem meine Entscheidung, nicht die der Quelle.
- **Wie prüfen**: Entscheiden, ob Kursivierung überhaupt Teil der Gold-Fassung sein soll. Die 22 `<em>`-Stellen sind: „et al." (20×), „E. coli" (2×, nur in TABLE I), „C. Elegans" (1×).

### [NIEDRIG] Im Original steht ein γ mit tiefgestelltem „in" UND hochgestelltem „dom" ÜBEREINANDER am selben x. Die lineare Reihenfolge ist meine Wahl (erst Index, dann Exponent).
- **Stelle**: Seite 4, rechte Spalte, Ende des ersten (spaltenübergreifenden) Absatzes: „…a power-law with γ_in^dom = 1.94."
- **Warum**: Der Content-Stream setzt „dom" zuerst, „in" danach — nach der Zeichenreihenfolge wäre `γ^dom_in` näher dran. Ich habe die konventionelle Leseform gewählt.
- **Wie prüfen**: Seite 4, rechte Spalte, 6. Zeile von oben zoomen. Beide Formen bezeichnen dasselbe Symbol; nur die Zeichenkette unterscheidet sich.

### [NIEDRIG] Die Zelle steht im Original auf zwei Zeilen („Yook et al. 2001a," / „Pastor-Satorras et al. 2001"). Ich habe sie mit einem Leerzeichen zu einer Zeile verbunden.
- **Stelle**: TABLE I, Zeile „Internet, domain level", Spalte „Reference"
- **Warum**: Es gibt keine verbundenen Zellen — das vertikale Linienraster ist in ALLEN Zeilen beider Tabellen vollständig (verifiziert: 10 bzw. 12 vertikale Linien in jedem Zeilenband). Der Umbruch ist reiner Zeilenumbruch innerhalb der Zelle. Alternative wäre `<br>` in der Zelle.
- **Wie prüfen**: TABLE I, zweite Datenzeile — die Zelle ist optisch zweizeilig, aber ein einziges Feld (die Nummer „2" rechts steht auf der zweiten Zeile).

### [NIEDRIG] Dasselbe Netzwerk trägt in TABLE I einen PUNKT und in TABLE II ein Komma als Tausendertrenner. Beides getreu übernommen.
- **Stelle**: TABLE I, Zeile 14 „Words, cooccurence", Spalte „Size": `460.902` — gegen TABLE II, Zeile 19 „Words, cooccurence∗", Spalte „Size": `460, 902`
- **Warum**: Original-Inkonsistenz, kein Transkriptionsfehler. Optisch verifiziert (500 dpi-Ausschnitt beider Zeilen).
- **Wie prüfen**: Beide Zeilen zoomen — der Punkt in TABLE I sitzt auf der Grundlinie und ohne folgendes Leerzeichen, das Komma in TABLE II hat den üblichen Mathmodus-Abstand.

### [NIEDRIG] Die Nummer 14 kommt zweimal vor (auch bei „Protein, S. cerev.∗"), die 15 fehlt ganz.
- **Stelle**: TABLE II, Spalte „Nr.": Zeile „Ythan estuary∗" = 14, Zeile „Silwood park∗" = 16
- **Warum**: Fehler des Originals, getreu übernommen. Ein Werkzeug, das hier „15" ausgibt, halluziniert.
- **Wie prüfen**: TABLE II, rechte Spalte, Zeilen 14–16 von oben — 14, 14, 16.

### [NIEDRIG] Ein absteigender „Bereich" — die Untergrenze ist größer als die Obergrenze.
- **Stelle**: TABLE I, Zeile „Internet, domain level", Spalte `ℓ_rand`: `6.36 - 6.18`
- **Warum**: Steht so im Original (optisch bei 500 dpi verifiziert). Sieht wie ein Zahlendreher der Autoren aus, ist aber nicht meine Sache zu korrigieren.
- **Wie prüfen**: TABLE I, zweite Datenzeile, fünfte Spalte.

### [NIEDRIG] Die Bildunterschrift benutzt ein normales kursives `l` (U+006C), die Tabellenkopfzeile das Skript-ℓ (U+2113). Beides getreu übernommen.
- **Stelle**: TABLE II, Bildunterschrift: „The columns l_real, l_rand and l_pow…" gegen TABLE II, Kopfzeile: `ℓ_real | ℓ_rand | ℓ_pow`
- **Warum**: Original-Inkonsistenz. Über Zeichenzählung abgesichert: ℓ (U+2113) kommt auf Seite 8 genau 7× vor — 4× in TABLE I (Unterschrift + Kopf), 3× im Kopf von TABLE II, 0× in der Unterschrift von TABLE II.
- **Wie prüfen**: Zeile „marked with a star, these values are identical. The columns l_real, l_rand and l_pow compare…" gegen die Kopfzeile darunter zoomen — die Schleife des ℓ ist im Kopf deutlich, in der Unterschrift nicht.

### [NIEDRIG] Hier steht ein Minuszeichen U+2212, während alle anderen Bereichsangaben auf der Seite einen normalen Bindestrich U+002D benutzen (z.B. `3, 015 - 4, 389` in derselben Zeile).
- **Stelle**: TABLE II, Zeile „Internet, domain∗", Spalte `κ`: `30 − 40`
- **Warum**: Original-Inkonsistenz (Mathmodus vs. Textmodus im LaTeX-Quelltext), getreu übernommen. Spannbreiten-Leerzeichen beidseitig sind echt (gemessen).
- **Wie prüfen**: Die Zeile zoomen — der Strich in der κ-Spalte ist sichtbar länger und höher gesetzt als die Striche links und rechts davon.

### [NIEDRIG] Das Zeichen ist U+2217 ASTERISK OPERATOR (∗), NICHT das ASCII-Sternchen `*`.
- **Stelle**: TABLE II, Spalte „Network": alle Sternchen, z.B. `Internet, domain∗`, `Movie actors∗`
- **Warum**: So steht es in der Textebene (13× auf Seite 8). Nebeneffekt: dadurch entsteht in Markdown keine ungewollte Kursivierung. Ein Werkzeug, das `*` ausgibt, weicht hier ab — das ist eine echte Abweichung, keine Toleranz.
- **Wie prüfen**: Codepoint-Prüfung der Gold-Datei: `grep -c '∗'` sollte 13 liefern.

### [NIEDRIG] PyMuPDFs `get_text()` liefert dort „∼k", „ℓand", „⟨k⟩and", „cutoffκ" — ohne Leerzeichen.
- **Stelle**: Vier Stellen, an denen ich ein Leerzeichen EINGEFÜGT habe, das die Textebene nicht enthält: Seite 4 Gleichung (3) „∼ k" (2×); Seite 8 TABLE-I-Unterschrift „path length ℓ and the clustering"; Seite 8 TABLE-II-Unterschrift „⟨k⟩ and the cutoff κ"
- **Warum**: Es sind echte Wortabstände, die nur wegen des Fontwechsels (CMMI→CMR) nicht als Leerzeichen-Glyph kodiert sind. Gemessene Lücken 2,7–3,6 pt gegen 0,5–1,3 pt bei echten Kerning-Korrekturen (z.B. `P(k)` in der TABLE-II-Unterschrift, das ich deshalb OHNE Leerzeichen gelassen habe). Zusätzlich bei 600 dpi optisch bestätigt.
- **Wie prüfen**: „length ℓ and the clustering coefficient" in der TABLE-I-Unterschrift zoomen — Abstand deutlich sichtbar. Gegenprobe „P(k)" in der TABLE-II-Unterschrift, Zeile 1: dort ist KEIN Abstand, und die Gold-Fassung hat dort auch keinen.

### [NIEDRIG] Die Gold-Fassung beginnt mitten im Satz, ohne Großbuchstabe und ohne Kontext.
- **Stelle**: Seite 4, allererstes Wort: „the WWW as a network has boomed…"
- **Warum**: Der Absatz beginnt auf Seite 3 („The interest in…"). Da nur Seite 4 transkribiert wird, ist das korrekt — kann aber wie ein abgeschnittener Anfang aussehen.
- **Wie prüfen**: Seite 3, rechte Spalte, letzte Zeile: „1999 (Lawrence and Giles 1998, 1999). The interest in".

### [NIEDRIG] Die Textebene liefert die Ligaturzeichen ﬁ/ﬀ/ﬃ (U+FB01/FB00/FB03) und zerlegte Akzente („Barab´asi", „Sol´e", „Istv´an" — Akut VOR dem Vokal). Ich habe beides normalisiert.
- **Stelle**: Ligaturen und Akzente durchgehend: „signifies", „coefficient", „different", „cutoff", „identifies", „Barabási", „Solé", „István"
- **Warum**: Beides sind Kodierungsartefakte der Type-1-Fonts, kein Inhalt. Der Akzentfall ist eindeutig kaputt (falsche Reihenfolge). Bei Ligaturen ist die Normalisierung Standard (NFKC), aber sie ändert Bytes gegenüber der Rohextraktion.
- **Wie prüfen**: Zählprobe: Seite 4 hat 8× ﬁ, 3× ﬃ, 1× ﬀ, 4× Akut; Seite 8 hat 2× ﬁ, 2× ﬃ, 1× ﬀ, 12× Akut. Alle sind in der Gold-Fassung als fi/ffi/ff bzw. á/é aufgelöst.

## Konventions-Entscheidungen (15) — keine Fehler, aber deine Wahl
- Sub-/Hochstellungen: rein NUMERISCHE Exponenten als Unicode-Hochstellung (10⁻⁴, 10⁻⁵, 10⁷, 10⁸, 10⁶) — so steht es in der Aufgabenvorgabe („⁻⁴"). ALLE anderen (buchstabenhaltigen) Stellungen als Plaintext `_name` / `^name`. Grund für den Bruch: Unicode hat kein tiefgestelltes „d" und kein tiefgestelltes „w", also lassen sich `ℓ_rand` und `ℓ_pow` gar nicht in Unicode setzen — eine reine Unicode-Lösung wäre innerhalb EINER Kopfzeile inkonsistent. Vertretbare Alternative: durchgängig flach schreiben („Pout(k)", „k−γout", „ℓrand") wie die Rohtextebene, oder durchgängig `^`/`_` auch für Zahlen („10^-4").
- Seite 8 vollständig: BEIDE Tabellen (TABLE I und TABLE II) plus beide Bildunterschriften. Vertretbare Alternative, falls die Aufgabe nur TABLE I meinte: alles ab „TABLE II. The scaling exponents…" entfernen.
- Beide Tabellen als GFM-Pipe-Tabellen, KEIN rohes HTML. Deckungsgleich mit Konvention 2, weil es nachweislich keine verbundenen Zellen gibt: das vertikale Linienraster ist in jedem einzelnen Zeilenband vollständig (TABLE I: 10 Linien × 18 Bänder, TABLE II: 12 Linien × 21 Bänder, maschinell geprüft).
- Keine Ausrichtungs-Doppelpunkte in der Trennzeile (`| --- |` statt `| :---: |`), obwohl im Original praktisch alle Zellen zentriert sind. Vertretbare Alternative: Zentrierung als `:---:` kodieren.
- Kursiv nur für textliche Kursive (`*et al.*`, `*E. coli*`, `*C. Elegans*`), nicht für mathematische Variablen (P, k, C, γ, ℓ), die im Original ebenfalls kursiv gesetzt sind. Vertretbare Alternative: gar keine Kursivierung (die sechs Konventionen fordern sie nicht).
- Zeilenend-Trennstriche aufgelöst (Konvention 5), 9 Stellen: Al-bert→Albert (2×), charac-terized, hyper-links, sub-set, obtain-ing, do-main, fol-lowed, re-moved. Echte Binde­striche in Komposita (power-law, World-Wide, Pastor-Satorras, Phone-call, scale-free) und Bereichsstriche in Tabellen bleiben unangetastet.
- Spaltenübergreifender Absatz auf Seite 4 zu EINEM Absatz verschmolzen („…domain name and two" + „nodes being connected…"). Abgesichert über die Erstzeilen-Einrückung: Absatzanfänge stehen bei x=64,0 (links) bzw. x=327,0 (rechts), Fortsetzungszeilen bei x=54,0 bzw. x=317,0 — die erste Zeile der rechten Spalte steht bei x=317,0, ist also KEIN neuer Absatz.
- Lesereihenfolge Seite 4: linke Spalte komplett (Absatz → Gleichung (3) → FIG.-1-Unterschrift → Absatz), dann rechte Spalte. Die FIG.-1-Unterschrift steht damit ZWISCHEN Gleichung (3) und dem Albert/Jeong-Absatz — das entspricht der physischen Anordnung. Vertretbare Alternative: Bildunterschriften ans Absatzende oder Seitenende verschieben.
- Gleichung (3) als normaler Absatz mit der Gleichungsnummer inline am Ende („… k^−γ_in.    (3)"), Abstände mit vier Leerzeichen angedeutet. Kein LaTeX, kein Codeblock, keine Tabelle. Vertretbare Alternative: die Nummer weglassen oder in eine eigene Zeile setzen.
- Zweizeilige Referenzzelle in TABLE I mit einem Leerzeichen zu einer Zelle verbunden statt mit `<br>`.
- Seitentrenner `---` genau einmal, zwischen Seite 4 und Seite 8 — nicht am Anfang, nicht am Ende (Konvention 4). Er steht mit Leerzeilen davor und danach und wird deshalb als thematischer Trenner geparst, nicht als Setext-Überschrift (mit CommonMark verifiziert: 1× `<hr>`, 0 Überschriften).
- Keine Überschriften: weder Seite 4 noch Seite 8 enthalten eine Abschnittsüberschrift. „TABLE I."/„TABLE II."/„FIG. 1." sind Bildunterschriften und stehen als normale Absätze da, nicht als `#`/`##`. Konvention 1 greift auf diesen beiden Seiten schlicht nicht.
- Weggelassen (Konvention 6): die Seitenzahlen „4" und „8" in der Fußzeile. Diese Ausgabe hat keine Kolumnentitel und keinen Journal-Header auf den beiden Seiten.
- Zahlformate exakt wie in der Quelle: Tausendertrenner-Leerzeichen bleiben („325, 729"), „460.902" (Punkt) und „460, 902" (Komma) bleiben verschieden, „6.36 - 6.18" bleibt absteigend, die Nummerierung 14/14/16 in TABLE II bleibt fehlerhaft. Nichts davon wurde stillschweigend repariert.
- Verifikationsmethode zum Nachvollziehen: Tabellenraster nicht geschätzt, sondern aus den 487 Linien-Rechtecken des Content-Streams rekonstruiert (Spaltengrenzen exakt in Punkt), Zellzuordnung über den x-Mittelpunkt jedes Spans. Danach Wort-für-Wort-Abgleich der fertigen Markdown-Datei gegen die PDF-Textebene mit difflib: Seite 4 611 gegen 611 Tokens, Seite 8 727 gegen 731 Tokens — jede verbleibende Abweichung ist oben in den Unsicherheiten benannt und einzeln optisch geprüft.

---

# 07.md
**Urteil der Gegenprüfung**: brauchbar mit korrekturen

## Gegenprüfung — Erfunden (schlimmster Fehlertyp)
- Nichts gefunden. Eigene bidirektionale Wortprobe (Textebene Seite 2, Trennstriche aufgelöst, <br>/HTML entfernt, Multiset-Vergleich): 0 Wörter im Entwurf, die nicht in der Quelle stehen. Auch zeichenweise stichprobenartig geprüft (alle Zeilenlabels von Tabelle 1–3, Unterschriftsbeschriftungen, Datenschutzhinweis) — identisch.
- Die drei Konventionszeichen ☐ (16×), — (5×) und _____ (43×) stehen so in keiner Quelle, sind aber deklarierte Platzhalter für real gedruckte Marken und in den Entscheidungen offengelegt. Ich zähle sie NICHT als Erfindung; ihre Vorkommenszahlen sind gegengemessen: 16 CheckBox-Widgets (12 in Tabelle 1 + 4 'Ja'), 5 kurze Querstriche (3× in 'besteht weiter bei' + je 1× Ehegatte bei Schulbesuch/Wehrdienst), 43 gedruckte Punkt-/Unterstrichläufe — alle drei Zahlen stimmen exakt.

## Gegenprüfung — Fehlend
- Nichts Substanzielles. Die Wortprobe in Gegenrichtung (Quelle → Entwurf) meldet nur die Punktläufe selbst ('.......', '.....................'), die konventionsgemäß zu _____ werden.
- Minimal: die Einrückung der beiden Aufzählungspunkte in Tabelle 1, Zeile 1. In der Textebene beginnt 'Die bisherige Versicherung' bei x=60,5, die beiden 'o'-Zeilen bei x=63,4 — sie sind Unterpunkte. Der Entwurf reiht sie mit <br> auf gleicher Ebene. Folgerichtig aus der Entscheidung, das 'o' nicht in eine Liste zu übersetzen, aber die Hierarchie geht verloren.

## Gegenprüfung — Struktur
- FOLGENREICHSTE STELLE — die drei Fettbalken als H2 statt als verbundene Kopfzeile. Meine Messung stützt die Gegenposition stärker als der Entwurf einräumt: Im Vektorraster laufen die Außen-Vertikalen bei x=56,6 und x=550,0 DURCH die Balkenbänder (20,3–45,8 / 245,8–265,7 / 535,0–554,8) und es gibt in diesen Bändern KEINE inneren Vertikalen — das ist exakt dieselbe Signatur, mit der der Entwurf den colspan=5 der Zeile 'Die folgenden Angaben …' (Band 583,6–600,2) begründet. Im gerenderten Scan sitzt der Balken sichtbar INNERHALB des Tabellenrahmens und teilt dessen Außenkanten. Zwei Messungen, dieselbe Evidenz, gegensätzlich ausgewertet.
- Zusatzbeleg gegen die H2-Lesart: corpus/07_formular-punktlinien/README.md kündigt für Seite 2 ausdrücklich 'eine Tabelle mit verbundenem Kopf (Ehegatte | Kind | Kind | Kind)' an. Unter der Entwurfs-Lesart hat KEINE Tabelle der Seite einen verbundenen KOPF — der einzige colspan sitzt mitten in Tabelle 3. Korrektur wäre: alle drei Tabellen auf <table> umstellen, erste Zeile jeweils <tr><th colspan="5">…</th></tr>, Konvention 1 greift dann für die Balken nicht.
- Unterschriftenblock als GFM-Tabelle. Im Scan ist dort KEIN Raster: drei freistehende Unterschriftslinien (Textebene: drei Unterstrich-Spans bei y=699,2, Längen 175/173/173 pt) mit den Beschriftungen darunter (y=707,7), kein einziger Vektor-Rahmen im ganzen Band. Der Entwurf behauptet damit Tabellenstruktur, die die Quelle nicht hergibt — genau gegen die eigene Leitregel. Ordnungstreu und strukturneutral wären drei Absätze/Zeilen '_____' + Beschriftung.
- Reihenfolge im Unterschriftenblock ist umgedreht (Beschriftung vor Linie statt Linie vor Beschriftung). Vom Autor selbst benannt und begründet — aber es bleibt die einzige Stelle der Datei mit veränderter Lesereihenfolge.

## Gegenprüfung — Konvention verletzt
- Konvention 2 — bedingt: sollte die Balken-Frage zugunsten des verbundenen Kopfes entschieden werden, sind Tabelle 1 und 2 als GFM-Pipes nicht mehr zulässig (verbundene Zelle ⇒ rohes HTML). Der Entwurf hängt damit vollständig an einer Auslegung, die ich für die schwächere halte.
- Konvention 1 — Ebene: der Entwurf setzt alle drei Balken auf ##, obwohl auf Seite 2 kein # existiert (der Dokumenttitel steht auf Seite 1). 'Ebene wie im Original' ist bei einem Ein-Seiten-Ausschnitt nicht entscheidbar; die Wahl ## impliziert eine nicht vorhandene Ebene darüber. Nicht diskutiert.
- Konvention 2 — Randfall: <br> ist rohes HTML innerhalb der Pipe-Tabellen. Die Konvention lizenziert HTML ausdrücklich nur dort, wo verbundene Zellen Pipes unmöglich machen. Praktisch alternativlos, aber ungeprüft gegen den Wortlaut.

## Gegenprüfung — Vom Autor übersehen
- Die Seite-1-Begründung der Balken-Entscheidung ist einseitig und dadurch irreführend für den korrigierenden Menschen. Der Entwurf schreibt, auf Seite 1 trügen 'ZWEI identisch gestaltete Balken … gar keine Tabelle unter sich'. Nachgemessen: Seite 1 hat DREI solche Balken. Der dritte ('Allgemeine Angaben zu Familienangehörigen', Band 528,8–548,8) grenzt mit demselben 0,7-pt-Spalt an eine Tabelle (nächste Zeile 549,5–559,8), verhält sich also genau wie die Balken auf Seite 2. Seite 1 belegt beide Muster, nicht das behauptete eine.
- Dass der Unterschriftenblock überhaupt zu einer Tabelle gemacht wurde, ist nicht als Unsicherheit geführt — nur die Reihenfolge-Umkehr. Die Struktur-Behauptung ist der größere Eingriff.
- Überschriftenebene (# vs. ##) ist nicht thematisiert; siehe konvention_verletzt.
- Die Ehegatte-Zelle in 'Die bisherige Versicherung besteht weiter bei' bleibt leer, die drei Kind-Zellen bekommen —. Das ist nachweislich richtig (Unterstrich-Läufe bei x=352,6 / 426,2 / 499,9 liegen in den Kind-Spalten 332,6–401,5 / 406,3–475,2 / 480–548,9; Ehegatte-Spalte 259–328 ist leer; im Scan bestätigt). Der Entwurf begründet die leere Zelle aber ausschließlich mit dem unsichtbaren AcroForm-Feld und benennt nicht, dass die semantische Asymmetrie (Ehegatte ausfüllbar, Kinder gesperrt) hier die eigentliche Aussage ist — für einen Korrektor die entscheidende Information.
- Die Einrückung der beiden 'o'-Unterpunkte (siehe fehlend) ist nicht als Unsicherheit geführt.
- Die Behauptung 'Kein ⟪UNLESBAR⟫-Loch nötig … der Scan ist an jeder Stelle lesbar' ist für eine Gold-Fassung, die AM SCAN gemessen wird, zu stark formuliert: der Wortlaut stammt vollständig aus dem nativen PDF, die Lesbarkeit des Scans wurde nur stichprobenhaft gegengesehen. Das ist methodisch korrekt (so war der Auftrag), aber die Formulierung suggeriert eine Scan-Volltextprüfung, die nicht stattgefunden hat.

## Unsicherheiten des Autors (9)

### [HOCH] Ich habe die drei fettgedruckten Balken als H2-Überschriften gesetzt — NICHT als verbundene Kopfzeile `<th colspan="5">` der jeweiligen Tabelle. Genau diese Entscheidung bestimmt, ob Tabelle 1 und 2 GFM-Pipes sein dürfen (sie sind dann merge-frei) oder ob alle drei Tabellen rohes HTML brauchen.
- **Stelle**: Die drei Zeilen `## Angaben zur letzten bisherigen …`, `## Sonstige Angaben zu Familienangehörigen`, `## Angaben zur Vergabe einer Krankenversichertennummer …`
- **Warum**: Geometrisch gehört der Balken zur Tabelle: sein Rahmen teilt sich die Aussenkanten mit dem Raster (Balken endet bei y=45.8, Kopfzeile beginnt bei y=46.6, kein Zwischenraum), und corpus/gold/README.md kündigt für 07.md ausdrücklich eine „Tabelle mit verbundenem Kopf" an. Dagegen steht: auf Seite 1 tragen ZWEI identisch gestaltete Balken („Allgemeine Angaben des Mitglieds", „Angaben zu Familienangehörigen") gar keine Tabelle unter sich — der Balken ist also das Überschriften-Element des Formulars und trifft hier nur zufällig auf ein Raster. Ich bin dieser Lesart gefolgt, sie ist aber nicht zwingend.
- **Wie prüfen**: Seite 1 des Scans ansehen: die Balken über den Ankreuz-Blöcken stehen frei. Wenn die Soll-Fassung den Balken trotzdem als verbundene Kopfzeile führen soll, müssen Tabelle 1 und 2 ebenfalls auf `<table>` umgestellt werden, jeweils mit `<tr><th colspan="5">…</th></tr>` als erster Zeile.

### [MITTEL] Den kurzen, mittig stehenden Querstrich habe ich als Geviertstrich `—` (Bedeutung: entfällt / nicht auszufüllen) gesetzt, nicht als Ausfüllfeld.
- **Stelle**: Zeile `| Die bisherige Versicherung besteht weiter bei:<br>… |  | — | — | — |` sowie die Ehegatte-Zellen von „Schulbesuch/Studium" und „Wehrdienst oder gesetzlich geregelter Freiwilligendienst"
- **Warum**: In der Textebene ist dieser Strich buchstäblich ein Unterstrich-Lauf (`______`) — formal dasselbe Zeichen wie eine Ausfülllinie. Optisch ist er aber kurz, mittig und ohne Beschriftung, während echte Ausfülllinien links bündig über die Zellenbreite laufen. Ich habe die optische Lesart gewählt; wer der Textebene folgt, müsste hier `\_\_\_\_\_` schreiben.
- **Wie prüfen**: Im Scan die drei Kind-Zellen der Zeile „Die bisherige Versicherung besteht weiter bei" und die Ehegatte-Zellen der beiden letzten Zeilen von Tabelle 2 vergleichen mit einer echten Ausfülllinie (z. B. Zeile „Sofern zuletzt eine Familienversicherung bestand"). Entscheiden, ob „entfällt" oder „Ausfüllfeld" gemeint ist.

### [MITTEL] Diese Zellen sind im Formular ausfüllbar, tragen aber KEINE gedruckte Linie. Ich habe sie leer gelassen (bzw. nur `EUR` geschrieben) statt ein Ausfüllfeld zu setzen.
- **Stelle**: Alle leeren Zellen: Tabelle 3 (`eigene Rentenversicherungsnummer (RV-Nr.)`, `Geburtsname`, `Geburtsort`, `Geburtsland`), die Ehegatte-Zelle von „Die bisherige Versicherung besteht weiter bei", und die Zellen, die nur `EUR` enthalten
- **Warum**: Im nativen PDF liegen dort unsichtbare AcroForm-Textfelder (`p.widgets()` listet u. a. `Geburtsname.0…3`, `Ort Datum`, `Text11.*`). Auf dem Scan — und der Scan ist der Massstab — druckt sich davon nichts ab. Ich habe nur transkribiert, was sichtbar ist. Wer die Ausfüllbarkeit mitnehmen will, setzt in diese Zellen `\_\_\_\_\_`; das wäre dann aber eine Information aus dem nativen PDF, die im Scan nicht existiert.
- **Wie prüfen**: Scan Seite 2, unterer Kasten (Krankenversichertennummer): die 16 Datenzellen sind blanko. Gegenprobe im nativen PDF über die Widget-Liste.

### [MITTEL] Im Original stehen die drei Unterschriftslinien OBEN und ihre Beschriftungen DARUNTER. In der Gold-Fassung ist die Reihenfolge umgedreht: Beschriftungen als Tabellenkopf, Ausfüllfelder als Datenzeile.
- **Stelle**: `| Ort, Datum | Unterschrift | ggf. Unterschrift der Familienangehörigen |` mit `| \_\_\_\_\_ | \_\_\_\_\_ | \_\_\_\_\_ |` darunter
- **Warum**: Der Block ist dreispaltig; GFM erzwingt eine Kopfzeile, und die Beschriftung ist die natürliche Kopfzeile eines Feldes. Der Preis ist eine echte Umstellung der Lesereihenfolge — die einzige Stelle der Datei, an der ich Reihenfolge verändert habe. Ordnungstreu wäre `| \_\_\_\_\_ | \_\_\_\_\_ | \_\_\_\_\_ |` als Kopfzeile und die Beschriftungen darunter; das sieht im Rohtext allerdings nach einem Fehler aus.
- **Wie prüfen**: Unterschriftenbereich im Scan (unterhalb von „Ich bestätige die Richtigkeit …"): Linie, darunter Beschriftung. Entscheiden, ob Ordnungstreue oder Lesbarkeit vorgeht.

### [NIEDRIG] Drei Ausfülllinien, obwohl die Beschriftung nur zwei Einträge nennt („o endete am:", „o bestand bei:").
- **Stelle**: Erste Datenzeile von Tabelle 1: `\_\_\_\_\_<br>\_\_\_\_\_<br>\_\_\_\_\_` (drei Felder pro Spalte)
- **Warum**: Gedruckt sind tatsächlich drei Punktläufe pro Spalte (y≈63,6 / 78,6 / 91,2; der dritte ist breiter und sitzt an der unteren Zellkante). Formularfelder gibt es dort aber nur zwei (`Text 1`, `Text 2`) — die dritte Linie ist eine Fortsetzungslinie ohne Feld dahinter. Wer nach Feldern statt nach gedruckten Linien zählt, schreibt hier nur zwei.
- **Wie prüfen**: Ehegatte-Zelle der ersten Datenzeile im Scan vergrössern und die Punktlinien zählen — es sind drei.

### [NIEDRIG] Das `o` ist als Buchstabe stehengeblieben und nicht in eine Markdown-Liste (`- endete am:`) übersetzt worden.
- **Stelle**: `Die bisherige Versicherung<br>o endete am:<br>o bestand bei: (Name der Krankenkasse)`
- **Warum**: In der Textebene steht buchstäblich der Kleinbuchstabe `o` in Courier New; im Scan wirkt er als kleiner Ring, funktional also ein Aufzählungszeichen. Es als `-` zu setzen wäre eine strukturelle Deutung, die die Zeichen der Quelle verändert.
- **Wie prüfen**: Beschriftungszelle der ersten Datenzeile von Tabelle 1 im Scan vergrössern.

### [NIEDRIG] Die Unterstriche sind mit Backslash maskiert. Gerendert steht überall `_____`, im Rohtext aber `\_\_\_\_\_` — ein Kandidat, der unmaskierte Unterstriche liefert, weicht textuell ab, obwohl er dasselbe meint.
- **Stelle**: Jedes `\_\_\_\_\_` im Rohtext (43 Vorkommen)
- **Warum**: Gemessen, nicht vermutet: `_____<br>_____` wird von python-markdown zu `<strong><em>_</em><br></strong>__` und von CommonMark (markdown-it) zu `<em><strong>…</strong></em>`. Ein unmaskierter Unterstrich-Lauf ist als Ausfüllfeld-Konvention schlicht nicht tragfähig, sobald zwei davon in einer Zelle stehen. Mit Maskierung rendern beide Parser identisch und emphasis-frei.
- **Wie prüfen**: Datei rendern: es darf genau EIN `<strong>` geben (der Absatz „Ich bestätige …") und 43 literale `_____`. Falls beim Vergleich gegen Kandidaten die Backslashes stören, vor dem Diff `\_` → `_` normalisieren.

### [NIEDRIG] Das Zeichen ☐ (U+2610) steht in KEINER Quelle als Zeichen — im nativen PDF ist das Kästchen ein gezeichnetes Rechteck bzw. ein AcroForm-CheckBox-Widget, im Scan ein gedrucktes Quadrat.
- **Stelle**: `☐ Mitgliedschaft`, `☐ Familienversicherung`, `☐ nicht gesetzlich`, `☐ Ja`
- **Warum**: Die Textebene liefert an diesen Stellen nur ein führendes Leerzeichen (` Mitgliedschaft`) — genau dort sitzt das Kästchen. Ohne ein gesetztes Zeichen verschwände das Ankreuzfeld vollständig aus der Gold-Fassung. Die Alternative wäre `[ ]`; das kollidiert in GFM optisch mit Task-Listen.
- **Wie prüfen**: Scan: 12 Kästchen in Tabelle 1 (3 je Spalte), 4 Kästchen in Tabelle 2 (Zeile „Selbstständige Tätigkeit liegt vor"). Zählprobe gegen die Gold-Fassung: 16 ☐.

### [NIEDRIG] Zwischen Ausfülllinie und `EUR` steht in der Gold-Fassung ein Leerzeichen; in der Quelle stossen Punktlauf und `EUR` direkt aneinander (`………………EUR`).
- **Stelle**: Zelle `\_\_\_\_\_ EUR` in der Zeile „Sonstige regelmäßige monatliche Einkünfte …"
- **Warum**: Reine Lesbarkeitsentscheidung, weil die Linie ohnehin auf eine feste Länge normalisiert ist. Wer strikt zeichenweise vergleicht, müsste `\_\_\_\_\_EUR` schreiben.
- **Wie prüfen**: Textebene Seite 2, Span bei y≈391,3 — dort steht `………………EUR` ohne Trennzeichen.

## Konventions-Entscheidungen (16) — keine Fehler, aber deine Wahl
- Ausfüllfeld-Konvention (EINE für alles): jede gedruckte, leere Ausfülllinie — egal ob mit Punkten (`.......`), Auslassungspunkten (`……`) oder Unterstrichen gezeichnet — wird zu genau `_____` (fünf Unterstriche), Länge normalisiert, Punktzahl nicht nachgebildet. Im Rohtext maskiert als `\_\_\_\_\_`, weil ein unmaskierter Lauf sonst als Emphasis gelesen wird (in python-markdown UND CommonMark nachgemessen). Ergibt 43 Ausfüllfelder auf der Seite.
- Ankreuzkästchen-Konvention: jedes leere Kästchen wird zu `☐` (U+2610), direkt vor dem zugehörigen Text. 16 Stück auf der Seite. Alternative wäre `[ ]` gewesen — verworfen, weil das in GFM optisch mit Task-Listen kollidiert.
- Entfällt-Konvention: der kurze, mittig gesetzte Querstrich, mit dem das Formular eine Zelle sperrt, wird zu `—` (U+2014) — bewusst UNTERSCHIEDEN vom Ausfüllfeld, obwohl beide in der Textebene Unterstrich-Läufe sind. Vertretbar anders: beide gleich behandeln.
- Nur Gedrucktes wird transkribiert. Zellen mit unsichtbarem AcroForm-Feld, aber ohne gedruckte Linie, bleiben leer — der Massstab ist der Scan, nicht die Formularebene des nativen PDF. Betrifft Tabelle 3 vollständig, die Ehegatte-Zelle in „besteht weiter bei" und die reinen `EUR`-Zellen.
- Die drei umrandeten Fettbalken sind H2-Überschriften (Konvention 1), keine verbundenen Kopfzeilen. Begründung: auf Seite 1 stehen zwei gestalterisch identische Balken ohne jede Tabelle darunter. Folge: Tabelle 1 und 2 sind merge-frei und dürfen GFM-Pipes sein. Dies ist die folgenreichste Auslegung der Datei.
- Tabellenformat pro Tabelle entschieden, nicht global (Konvention 2 wörtlich): Tabelle 1 und 2 als GFM-Pipes, Tabelle 3 als rohes `<table>`, weil dort die Zeile „Die folgenden Angaben werden nur dann benötigt …" nachweislich über alle fünf Spalten läuft (im Vektorraster keine Zwischenstriche zwischen x=57 und x=550 im Band y=583,6–600,2). Der gemischte Look ist Absicht; wer Einheitlichkeit will, macht alle drei zu HTML.
- Mehrzeilige Zellinhalte werden mit `<br>` erhalten, wo die Zeilen eigenständige Elemente sind (Ankreuz-Listen, Feld + Beschriftung wie `(Nachname)`, Label + Klammerzusatz). Reine Zeilenumbrüche im Fliesstext einer Zelle werden dagegen zu einem Fliesstext zusammengezogen. `<br>` ist die einzige Möglichkeit, Zeilenstruktur in GFM-Zellen zu halten.
- Die vier Spaltenköpfe bleiben `Ehegatte | Kind | Kind | Kind` — bewusst NICHT durchnummeriert. Die erste Kopfzelle bleibt leer, weil die Beschriftungsspalte im Original keine Überschrift trägt.
- Die drei „Kind"-Spalten sind in Tabelle 2 bei „Schulbesuch/Studium" und „Wehrdienst" gefüllt, die Ehegatte-Spalte gesperrt — diese Asymmetrie ist so im Formular und wurde nicht geglättet.
- Der Fettdruck des Absatzes „Ich bestätige die Richtigkeit der Angaben …" ist als `**…**` erhalten (einziger Fettdruck-Lauf der Seite ausserhalb von Überschriften und Spaltenköpfen). Vertretbar anders: als normaler Absatz.
- Der Absatz „Datenschutzhinweis (Artikel 13 …)" bleibt drin. Er steht zwar ganz unten, ist aber einmaliger Sachinhalt und keine Fusszeile im Sinne von Konvention 6 (Seitenzahl, Kolumnentitel). Seite 2 des Scans trägt weder Kopf- noch Fusszeile — es gab nichts wegzulassen.
- Die beiden nebeneinander gesetzten Schlussbemerkungen („Mit der Unterschrift erkläre ich …" links, „Bei getrennt lebenden Familienangehörigen …" rechts) bleiben ZWEI Absätze in Lesereihenfolge links→rechts. Sie sind kein über einen Spaltenumbruch laufender Absatz, sondern zwei eigenständige Hinweise.
- Kein Seitentrenner `---`: Konvention 4 gilt nur bei mehrseitigen Fassungen, diese Gold-Fassung umfasst allein Seite 2.
- Trennstriche am Zeilenende sind durchgängig aufgelöst (Konvention 5): Familienan-gehörigen, Familienversi-cherung, be-stand, de-ren, ab-geleitet, monat-lich, Einkommensteu-erbescheides, Beschäf-tigung, Be-triebsrente, Ren-ten, Bruttoarbeitsent-gelt, Ein-künfte, Ab-findung, Freiwilli-gendienst, Familienangehöri-gen, Familien-angehörigen. NICHT aufgelöst wurde `Schul- oder Studienbescheinigung` — dort ist der Bindestrich ein echter Ergänzungsstrich, der Umbruch liegt hinter „oder".
- Keine Abbildungen auf dieser Seite, daher kein Bildunterschriften-Fall. Keine Fussnoten, daher kein `[^1]`. Keine mathematischen Symbole. Kein `⟪UNLESBAR⟫`-Loch nötig: die Textebene des nativen PDF deckt Seite 2 vollständig ab, und der Scan ist an jeder Stelle lesbar.
- Verifikation (nicht nur gelesen, sondern gemessen): jedes alphabetische Wort der Gold-Fassung kommt in der umbruchbereinigten Textebene von Seite 2 vor, und jedes Wort der Textebene kommt in der Gold-Fassung vor — beide Richtungen leer, also weder erfunden noch verloren. Rendering in python-markdown und CommonMark identisch: 3 H2, 4 Tabellen, 1 colspan, 0 `<hr>`, genau 1 `<strong>`.

---

# 08.md
**Urteil der Gegenprüfung**: brauchbar mit korrekturen

## Gegenprüfung — Erfunden (schlimmster Fehlertyp)
- KEINE. Gegengeprueft mit vollem Wort-Diff: Absaetze Body-Index 9-21 ergeben 226 zu 226 Woerter ohne einen einzigen Unterschied. Tabelle 1: 50 zu 50 Zellen, Tabelle 2: 13 zu 13 Zellen, jeweils zeichengleich bis auf die EINE bewusst und dokumentiert aufgeloeste Worttrennung (siehe konvention_verletzt). Fussnotentext aus footnotes.xml (w:id=2) einschliesslich des Leerzeichens in '[GNU FDL ]' ist buchstabengetreu; das Linkziel rId1 = http://de.wikipedia.org/wiki/Bild:Five-forces.gif ist in word/_rels/footnotes.xml.rels bestaetigt. Auch die vier gecachten REF-Feldwerte ('1.1.1', '1.1.2', '1.1', '1.2') stehen tatsaechlich als w:t im Dokument und sind nicht erfunden.

## Gegenprüfung — Fehlend
- Fussnotendefinition: der Quelltext in footnotes.xml beginnt nach dem w:footnoteRef mit einem Run ' ' (einzelnes Leerzeichen) vor dem Hyperlink 'Quelle'. Der Entwurf laesst es ohne Erwaehnung weg. Trivial, aber es ist eine undokumentierte Abweichung vom Zeichenbestand.
- Tabelle 1, Zeilen 12 und 13, verbundene Zelle (gridSpan=3): sie enthaelt in der Quelle DREI separate w:p, jeder mit genau einem U+00A0. Der Entwurf gibt ein einzelnes leeres <td colspan="3"></td> aus. Die Zellinhalte sind semantisch leer, aber die Absatz-Vielzahl (drei Leerzeilen Hoehe) geht verloren; die Unsicherheitsliste nennt nur 'Zellen mit U+00A0', nicht diese Mehrfach-Absaetze.
- Whitespace-Runs am Absatzende: Body-Index 12 endet auf '... siehe etwa: ' und Body-Index 14 auf einem eigenen Run ' '. Beide fallen im Entwurf weg. Fuer Markdown korrekt (trailing whitespace ist bedeutungslos), aber nicht vermerkt.

## Gegenprüfung — Struktur
- KEINE. Im Einzelnen gegengeprueft und bestaetigt: (a) Abschnittsgrenzen — der Offset 2 gegen die rohen w:body-Kinder stimmt (Index 0 = w:bookmarkStart, Index 1 = w:sdt/TOC), alle vier Anker treffen exakt. (b) Ueberschriftenebenen — styles.xml: berschrift2 = 'heading 2'/outlineLvl 1, berschrift3 = 'heading 3'/outlineLvl 2; die sechs Ueberschriften im Bereich sind korrekt auf ## bzw. ### abgebildet und in Originalreihenfolge. (c) Tabelle 1 — 14 Zeilen, tblGrid 4 Spalten, gridSpan=3 exakt in Zeile 0, 12, 13, kein vMerge, jede Zeile summiert auf 4. (d) Tabelle 2 — gridSpan UND vMerge vorhanden; die HTML-Rasterarithmetik des Entwurfs (colspan 2 + rowspan 2 links oben, rowspan 2 auf 'Unternehmensinterne Faktoren') deckt sich Zelle fuer Zelle mit vMerge=restart/continue und ergibt in jeder Zeile 4 Spalten. Rohes HTML ist nach Konvention 2 fuer beide Tabellen richtig. (e) Kein w:tblHeader, keine Fettung und keine Schattierung (shd fill=auto) in den Kopfzeilen von Tabelle 1 — <td> statt <th> ist belegt, nicht bloss angenommen. (f) Listen — kein w:numPr und kein Listenabsatz-Stil in irgendeinem Absatz des Bereichs, auch nicht in den Tabellenzellen. (g) Fussnotennummer — w:footnoteReference-Scan ueber das ganze Dokument: id=2 an Body-Index 11 ist die ERSTE Referenz ueberhaupt (id 3 erst bei Index 68, id 4 bei Index 89), '[^1]' ist damit korrekt. (h) Inline-Auszeichnung — b auf S/W/O/T und auf beiden 'Faktoren'-Zellen, i auf 'kann': alle vier Stellen im XML verifiziert, keine erfunden, keine uebersehen.

## Gegenprüfung — Konvention verletzt
- Konvention 5 ueberdehnt: Tabelle 1, Zeile 13, erste Zelle. Die Quelle enthaelt EIN w:t mit dem Wortlaut 'tatsaechliche Unternehmens-ressourcen' (U+002D mitten im Text-Run). Der Entwurf gibt 'Unternehmensressourcen' aus. Konvention 5 zielt auf Trennstriche AM ZEILENENDE — in einer DOCX gibt es an dieser Stelle kein Zeilenende, der Bindestrich ist ein gespeichertes Inhaltszeichen. Folge fuer den Benchmark: KEIN XML-lesendes Werkzeug kann diese Aufloesung reproduzieren, die Gold-Fassung wird an genau einem Token unerreichbar und bestraft jeden korrekt arbeitenden Konverter. Empfehlung: auf 'tatsaechliche Unternehmens-ressourcen' zuruecksetzen und die Aufloesung stattdessen nur als Unsicherheit fuehren. (Der Entwurf hat die Stelle immerhin selbst flagged — es ist eine Abwaegung, kein blinder Fehler.)
- Begruendungsfehler bei der Unterstreichung (nicht das Ergebnis, die Herleitung): der Entwurf rechtfertigt den Verzicht auf <u> damit, 'Konvention 2 erlaubt rohes HTML nur fuer verbundene Zellen'. Konvention 2 regelt ausschliesslich Tabellen und sagt ueber Inline-HTML im Fliesstext nichts. Die Entscheidung, 'unternehmensexternen' (u=single, Body-Index 16) und die komplett unterstrichene Nieschlag-Zeile (Body-Index 13) unformatiert zu lassen, ist vertretbar — aber sie steht auf einer falschen Autoritaet. Wer die Gold-Fassung spaeter revidiert, wird an dieser Stelle mit einem Scheinargument abgewiesen.

## Gegenprüfung — Vom Autor übersehen
- Body-Index 14: die Quelle schreibt 'alle diejenigen Markteilnehmer gelten' — mit EINEM t, korrekt waere 'Marktteilnehmer'. Der Entwurf uebernimmt es richtig, listet es aber NICHT unter den 'sprachlichen Auffaelligkeiten' (Unsicherheit 8), obwohl es genau in diese Klasse gehoert. Ein Mensch beim Gegenlesen wird das reflexhaft korrigieren und die Gold-Fassung damit von der Quelle wegbewegen — dasselbe Risiko, gegen das Unsicherheit 8 gebaut wurde.
- Body-Index 13, gleiche Klasse: '19.Auflage' und 'S.189.' stehen in der Quelle OHNE Leerzeichen nach dem Punkt. Ebenfalls korrekt uebernommen, ebenfalls nicht als schutzbeduerftige Quell-Eigenheit vermerkt.
- Innere Widersprüchlichkeit der Gold-Fassung, nirgends als solche benannt: der Entwurf laesst die Stil-Nummerierung der Ueberschriften weg (Unsicherheit 4), uebernimmt aber gleichzeitig die gecachten REF-Feldwerte als Text ('... die Wettbewerbsanalyse (1.1.2) ...', '... Umweltanalyse (1.1) ... Unternehmensanalyse (1.2) ...'). Das Ergebnis verweist auf Abschnittsnummern, die in derselben Datei nirgends existieren. Beide Einzelentscheidungen sind je fuer sich vertretbar, ihre KOMBINATION ist es fraglich — genau diese Kopplung fehlt in der Unsicherheitsliste, obwohl sie die eigentliche Entscheidung ist, die ein Mensch faellen muss.
- Fussnotenmarker vs. Bildposition, praeziser als im Entwurf gefasst: im XML steht die w:footnoteReference VOR dem w:drawing im selben Absatz. Die Fussnote gehoert also inhaltlich zur Abbildung (sie nennt Quelle und Lizenz des Porter-Schaubilds), nicht zum Satz davor. Der Entwurf stellt die Alternative 'Marker allein / Marker ans Ende des Absatzes davor' zur Wahl, ohne diesen Beleg zu nennen — er spricht klar gegen das Anhaengen an den Vorabsatz.
- Tabelle 2, Zeile 0/1: die linke obere Zelle ist ueber gridSpan=2 UND vMerge verbunden, waehrend die Beschriftungsspalte in den Zeilen 2/3 nur EINE Rasterspalte breit ist. Das Raster ist damit unsymmetrisch (oben links ein 2x2-Block, unten links eine 1x2-Saeule). Der Entwurf bildet das korrekt ab, vermerkt aber nicht, dass hier die einzige Stelle liegt, an der ein Werkzeug beim Spaltenzaehlen plausibel verrutscht — das ist der Punkt, an dem die Messung spaeter Treffer und Fehltreffer trennen wird.

## Unsicherheiten des Autors (9)

### [HOCH] Ob eine Zeile, die nur aus dem Fussnoten-Marker besteht, die richtige Wiedergabe dieses Absatzes ist.
- **Stelle**: Zeile 5 der Gold-Fassung: die alleinstehende Zeile `[^1]` (Body-Index 11 / roher w:body-Index 13)
- **Warum**: Der Absatz enthält im XML exakt zwei Runs: erst die Fussnotenreferenz (w:footnoteReference w:id=2, rStyle Funotenzeichen), dann ein inline w:drawing (Bild rId8, 4248150×3236006 EMU = das Porter-Fünf-Kräfte-Schaubild). Kein einziges w:t. Konvention 3 verlangt den Marker im Text, die Vorgabe verbietet einen Bild-Platzhalter — übrig bleibt der nackte Marker. Ein Werkzeug könnte den Marker stattdessen an den vorhergehenden Absatz hängen oder ihn ganz verlieren.
- **Wie prüfen**: DOCX öffnen, den Absatz direkt unter 'ich... Michael Porter dar:' ansehen: dort steht die hochgestellte Fussnotenziffer und darunter/daneben das Porter-Schaubild. Entscheiden, ob die Gold-Fassung den Marker allein stehen lässt oder ob er ans Ende des Absatzes davor gehört.

### [MITTEL] Die Abbildung (Porters Branchenstrukturanalyse, word/media über rId8) ist nicht im Markdown enthalten, und es gibt keine Bildunterschrift, die ich stattdessen hätte schreiben können.
- **Stelle**: Body-Index 11 / roher w:body-Index 13 — das Bild selbst
- **Warum**: Das docPr trägt nur den generischen Namen 'Bild 1', kein descr/Alt-Text. Im Dokument existiert kein Beschriftungsabsatz. Die beiden folgenden Absätze ('Für weitere Informationen über Porters Wettbewerbskräfte siehe etwa:' und die Nieschlag-Literaturangabe) sind eigenständige Body-Absätze, keine Bildunterschrift — ich habe sie deshalb als normale Absätze belassen und NICHT als Caption behandelt.
- **Wie prüfen**: Im DOCX prüfen, ob die beiden Absätze optisch als Bildunterschrift unter der Grafik stehen. Falls ja, wäre eine andere Auszeichnung vertretbar. Ausserdem entscheiden, ob die Gold-Fassung eine textuelle Notiz zur fehlenden Grafik tragen soll.

### [MITTEL] Ich habe den Bindestrich als Worttrennung aufgelöst (Konvention 5).
- **Stelle**: Erste Tabelle, letzte Zeile, erste Zelle: Quelle 'tatsächliche Unternehmens-ressourcen' → Gold 'tatsächliche Unternehmensressourcen'
- **Warum**: Der Bindestrich ist ein echtes U+002D mitten im Wort mit kleingeschriebener Fortsetzung; 'Unternehmensressourcen' ist das korrekte deutsche Kompositum. Der Autor hat ihn offensichtlich von Hand gesetzt, um das Wort in der schmalen Zelle umbrechen zu lassen. ABER: in einer DOCX steht der Bindestrich als Zeichen im Inhalt, nicht an einem XML-seitigen Zeilenende — ein rein XML-lesendes Werkzeug kann das nicht wissen und würde 'Unternehmens-ressourcen' transkribieren. Dies ist die EINZIGE Stelle, an der die Gold-Fassung wortweise von der Quelle abweicht (per Diff verifiziert: 383 zu 383 Wörter, genau eine Ersetzung).
- **Wie prüfen**: Zelle im DOCX ansehen: bricht 'Unternehmens-' tatsächlich am Zellenrand um? Falls der Bindestrich mitten in einer Zeile steht, wäre er zu erhalten.

### [MITTEL] Die automatische Kapitelnummerierung fehlt in der Gold-Fassung.
- **Stelle**: Alle sechs Überschriften des Abschnitts, z.B. '### Wettbewerbsanalyse'
- **Warum**: Die Stile berschrift1/2/3 tragen numPr mit numId=3 (abstractNumId 2, lvlText '%1', '%1.%2', '%1.%2.%3'). Im Word-Layout liest der Mensch also '1.1.2 Wettbewerbsanalyse', '1.1.3 Identifikation von Chancen und Risiken', '1.2 Unternehmensanalyse', '1.2.1 Ressourcenprofil', '1.2.2 Identifikation von Kompetenzen (Stärken-Schwächen-Profil)', '1.3 SWOT-Analyse'. Bestätigt sind daraus 1.1.2 (Wettbewerbsanalyse) und 1.2 (Unternehmensanalyse) durch die gecachten REF-Feldwerte im Text; 1.1.3, 1.2.1, 1.2.2 und 1.3 sind aus der Nummerierungsdefinition abgeleitet, nicht im Dokument als Text belegt. Die Nummern stehen in KEINEM w:t.
- **Wie prüfen**: DOCX öffnen und die angezeigten Überschriftennummern ablesen. Dann entscheiden, ob die Gold-Fassung '## 1.2 Unternehmensanalyse' o.ä. tragen soll — das würde die Messung gegen Werkzeuge stark verändern, weil praktisch kein Konverter Stil-Nummerierung ausgibt.

### [MITTEL] Die Unterstreichung ist in der Gold-Fassung weggefallen; der Text steht dort unformatiert.
- **Stelle**: Zwei Unterstreichungen: (a) 'Nieschlag, R.; Dichtl, E.; Hörschgen, H.: (2002) Marketing, 19.Auflage, S.189.' (komplett unterstrichen), (b) das Wort 'unternehmensexternen' in 'die sich aus rein unternehmensexternen Umweltbedingungen ergeben.'
- **Warum**: Markdown/GFM kennt keine Unterstreichung. Fett und kursiv habe ich erhalten (**S**trength, *kann*), weil sie eine native Markdown-Form haben; für Unterstreichung wäre nur rohes <u> geblieben — Konvention 2 erlaubt rohes HTML aber ausdrücklich nur für verbundene Tabellenzellen. In (b) ist die Unterstreichung eine inhaltliche Hervorhebung und geht damit als Information verloren.
- **Wie prüfen**: Beide Stellen im DOCX ansehen und entscheiden, ob <u>unternehmensexternen</u> (und ggf. die Literaturangabe) in die Gold-Fassung soll.

### [NIEDRIG] Zellen, die in der Quelle genau ein geschütztes Leerzeichen (U+00A0) enthalten, sind als leeres <td></td> ausgegeben; für die Kopfzeilen der ersten Tabelle habe ich <td> statt <th>/<thead> gewählt.
- **Stelle**: Beide Tabellen — alle 33 leeren Zellen (erste Tabelle) sowie die Kopfzellen
- **Warum**: Die U+00A0-Zellen sind Ausfüllfelder eines Formulars, ihr Inhalt ist semantisch leer — aber ein Zeichen geht verloren. Für <th> fehlt jeder Beleg: die Zeilen tragen kein w:tblHeader in trPr, keine Fettung und keine Schattierung (shd=auto). Bei der zweiten Tabelle sind 'Unternehmensexterne Faktoren' und 'Unternehmensinterne Faktoren' im XML fett — die habe ich als <strong> erhalten, aber ebenfalls in <td>.
- **Wie prüfen**: Entscheiden, ob die Gold-Fassung semantische Kopfzellen behaupten soll, obwohl die Quelle sie nicht auszeichnet. Bei den leeren Zellen entscheiden, ob &nbsp; erhalten bleiben muss.

### [NIEDRIG] Zwei Details: der Hyperlink ist als Markdown-Link erhalten, und '[GNU FDL ]' (mit Leerzeichen vor der Klammer) steht buchstabengetreu da.
- **Stelle**: Fussnotendefinition am Ende: '[^1]: [Quelle](http://de.wikipedia.org/wiki/Bild:Five-forces.gif) [GNU FDL ]'
- **Warum**: In word/footnotes.xml ist nur das Wort 'Quelle' verlinkt (rId1 → http://de.wikipedia.org/wiki/Bild:Five-forces.gif); der Rest ist normaler Text. '[GNU FDL ]' in eckigen Klammern kann von Markdown-Renderern als unaufgelöste Link-Referenz interpretiert werden — es ist aber exakt der Quelltext, inklusive des Leerzeichens vor ']'. Ein Konverter, der Links verwirft, würde hier nur 'Quelle [GNU FDL ]' liefern.
- **Wie prüfen**: Fussnote im DOCX ansehen (unten auf der Seite mit dem Porter-Schaubild) und entscheiden, ob der Link in der Gold-Fassung stehen bleibt.

### [NIEDRIG] Das sind keine Transkriptionsfehler, sondern Eigenheiten der Quelle.
- **Stelle**: Drei sprachliche Auffälligkeiten, alle buchstabengetreu übernommen: (a) '...zu erarbeiten anstrebt - bietet es sich an,' (b) 'Ressourcenforderung der Geschäftsidee' in der ersten Tabelle, (c) der Absatz 'Um einen Ausgangspunkt...' endet ohne Satzpunkt: '...– identifiziert werden'
- **Warum**: (a) ist ein Bindestrich-Minus mit Leerzeichen als Gedankenstrich (das echte U+2013 kommt im Abschnitt nur im Absatz 'Ressourcenprofil' vor). (b) heisst in der Quelle 'Ressourcenforderung', nicht 'Ressourcenanforderung' oder 'Ressourcenförderung'. (c) hat im letzten Run kein '.'. Ich vermerke sie, damit niemand sie beim Gegenlesen 'korrigiert' und die Gold-Fassung damit von der Quelle wegbewegt.
- **Wie prüfen**: Nur bestätigen, nicht ändern — im DOCX nachsehen, dass alle drei Stellen so dastehen.

### [NIEDRIG] Die Aufgabenstellung nennt 'rund 2.966 Zeichen'; meine Zählung der w:t-Zeichen über die Body-Indizes 9–27 ergibt 3.265 (Absätze 2.159 + Tabelle 1 270 + Tabelle 2 836).
- **Stelle**: Umfang des Abschnitts insgesamt
- **Warum**: Die Differenz stammt vermutlich aus einem anderen Extraktionsweg (z.B. python-docx, das w:fldSimple-Text und/oder Tabellentext anders erfasst: 2.159 + 836 = 2.995 liegt sehr nah an 2.966). Die Abschnittsgrenzen selbst sind dagegen eindeutig belegt: alle vier in der Aufgabe genannten Anker treffen exakt — Index 9 = 'Wettbewerbsanalyse' (berschrift3), Index 11 = die einzige Fussnotenreferenz, Index 23 = die 14-zeilige Tabelle, Index 27 = die 4-zeilige SWOT-Tabelle.
- **Wie prüfen**: Keine Handlung nötig, sofern die vier Anker als Definition des Abschnitts gelten. Falls die Zeichenzahl massgeblich sein soll, müsste die Zählmethode geklärt werden.

## Konventions-Entscheidungen (17) — keine Fehler, aber deine Wahl
- Abschnittsgrenzen: Die Aufgabenzählung (nur w:p und w:tbl) ist gegen die rohen w:body-Kinder um 2 versetzt, weil w:bookmarkStart und w:sdt (Inhaltsverzeichnis) am Dokumentanfang stehen. Body-Index 9–27 = rohe Indizes 11–29. Bestätigt durch alle vier genannten Anker.
- Überschriftenebenen direkt aus w:pStyle: berschrift2 → ##, berschrift3 → ###. Im Abschnitt kommen nur diese beiden vor (kein # — 'Markt' als berschrift1 liegt davor).
- Automatische Kapitelnummerierung der Überschriftenstile (numId 3) NICHT ausgegeben — die Nummern stehen in keinem w:t. Vertretbar anders: '## 1.2 Unternehmensanalyse' usw.
- Beide Tabellen als rohes HTML statt GFM-Pipes (Konvention 2): Tabelle 1 hat w:gridSpan=3 in Zeile 0, 12 und 13; Tabelle 2 hat w:gridSpan UND w:vMerge (SWOT-Matrix mit zweistufigem Kopf). Beide Rasterarithmetiken sind geprüft — jede Zeile ergibt genau 4 Spalten.
- Tabellenzellen ohne w:tblHeader/Fettung als <td> ausgegeben, kein <thead>/<th>. Fettung in Tabelle 2 als <strong> erhalten.
- Zellen mit ausschliesslich U+00A0 als leeres <td></td> ausgegeben.
- Inline-Auszeichnung: fett → ** (die S/W/O/T-Initialen), kursiv → * (das Wort 'kann'). Unterstreichung fallengelassen, weil GFM keine native Form hat und Konvention 2 rohes HTML nur für verbundene Zellen erlaubt.
- Gecachte Werte der w:fldSimple-Querverweise als normaler Text übernommen ('Branchen- und Marktanalyse', '1.1.1', 'Wettbewerbsanalyse', '1.1.2', '1.1', '1.2') — sie stehen als echtes w:t im Dokument.
- Fussnote als [^1] nummeriert: w:id=2 ist die erste echte Fussnote des Dokuments (0/1 sind separator/continuationSeparator), wird also auch in Word als '1' angezeigt. Definition am Abschnittsende, wie von Konvention 3 verlangt.
- Fussnoten-Hyperlink erhalten: [Quelle](http://de.wikipedia.org/wiki/Bild:Five-forces.gif). Vertretbar anders wäre reiner Text 'Quelle [GNU FDL ]'.
- Fussnoten-Marker steht auf einer eigenen Zeile, weil sein Absatz ausser der Referenz nur ein Bild enthält. Vertretbar anders: Marker ans Ende des Absatzes davor.
- Abbildung ohne Platzhalter weggelassen (Vorgabe). Eine Bildunterschrift existiert im Dokument nicht — die zwei folgenden Absätze sind eigenständige Body-Absätze und wurden als solche belassen.
- Der leere Absatz zwischen dem Kompetenzen-Text und der ersten Tabelle (Body-Index 22 / roh 24) ist weggefallen.
- Kein Seitentrenner ---: eine DOCX hat keine Seiten. Die zwei w:lastRenderedPageBreak im Bereich (roh 16 und 26) sind Word-Layout-Artefakte und wurden bewusst nicht zu Trennern.
- Keine Listenabsätze im Bereich: weder Stil 'Listenabsatz' noch irgendein w:numPr kommt zwischen Body-Index 9 und 27 vor (die Listenabsätze des Dokuments beginnen erst bei Body-Index 89).
- Kopf-/Fusszeilen (word/footer1.xml) gar nicht erst gelesen — Konvention 6.
- Wortweise Verifikation: die Gold-Fassung wurde nach Entfernen aller Markdown-/HTML-Syntax gegen die Wortfolge aus document.xml + footnotes.xml diffed. 383 zu 383 Wörter, genau eine bewusste Abweichung ('Unternehmens-ressourcen' → 'Unternehmensressourcen').

---

# 03.md

**Erstellt**: 2026-08-16 · **Verfahren**: **abgeleitet, nicht abgelesen.** Die Quelle ist nativ mit sauberer Textebene; die Fassung entsteht aus `get_text('words')` der Seiten 11+12, gruppiert nach y-Koordinate. Bidirektionaler Multiset-Vergleich gegen die Quelle: **234 zu 234 Token, 0 erfunden, 0 verloren.**

⚠️ **Das ändert, was du prüfen musst.** Bei `01/07/08` war die Frage „hat das Modell etwas dazuerfunden". Hier ist sie beantwortet — maschinell, nicht durch Zusicherung. Offen sind **ausschließlich Konventionen**. Wenn du nichts davon ändern willst, ist die Datei fertig.

## Was gemessen ist (keine Vermutung)

| | Messung |
|---|---|
| Struktur | Gezeichnete Tabelle, 111 Zeichenobjekte: Rahmen `x=91…503`, Kopfband `y=114…149`, Sub-Kopfband `y=162…174` |
| Ebenen | Haupt-Kopf `y=137` · Sub-Kopf `y=163` · Daten ab `y=175` · Fuß `y=796` |
| Typografie | Haupt- **und** Sub-Kopf: Arial-BoldMT 9pt (**identisch**) · Daten: ArialMT 9pt · Fuß: Verdana 9pt |
| Spalten | **einspaltig** — ein Angebot je Zeile, kein Gitter |
| Fuß | liegt **außerhalb** des Rahmens (796 gegen Rahmenende 782) |
| Sonderzeichen | alle Bindestriche ASCII `U+002D`; einziges Nicht-ASCII außer Umlauten ist `©` im Fuß |
| Überlauf | `OMS` wiederholt sich S11→12→13→14, `Ströer Interactive` S16→17, `TOMORROW FOCUS` S17→18 |

## Unsicherheiten des Autors (4)

### [HOCH] Die Form ist eine Wahl, keine Wahrheit — sie braucht eine Bewertungsregel
Die Fassung schreibt eine **einspaltige GFM-Tabelle**, deren Kopfzeile der Haupt-Kopf ist, mit dem Sub-Kopf als **fetter Datenzeile**. Gleich gut vertretbar wäre `## Übersicht…` + `### OMS` + Liste. Wer auf Zeichengleichheit prüft, misst dann die Zustimmung zu dieser Wahl statt die Fähigkeit des Werkzeugs — derselbe Fall wie Regel 1 bei `07.md`. **Vorschlag Regel 4** (unten formuliert, deine Entscheidung).

### [HOCH] Die wiederholten Köpfe sind der Messgegenstand — nicht später „aufräumen"
Auf Seite 12 stehen Haupt-Kopf **und** `OMS` erneut, weil sie im PDF erneut stehen (Olis Entscheidung 2026-08-16: „alles rein, 1:1 Kopie"). **Ein Werkzeug, das sie dedupliziert, muss hier Punkte verlieren** — genau das ist die Frage der Klasse: räumt die Engine still auf? Wer die Gold-Fassung später „schlanker" macht, löscht den Prüfgegenstand.

### [MITTEL] Fett auf `OMS` trägt die Ebenenunterscheidung allein
Haupt- und Sub-Kopf sind typografisch **identisch** (beide Arial Bold); im Markdown trägt der Haupt-Kopf seine Hervorhebung schon durch die Tabellen-Kopfzeile, der Sub-Kopf braucht dafür `**`. Lässt ein Werkzeug das Fett weg, ist `OMS` von einem Angebotsnamen nicht mehr zu unterscheiden. Ist das ein Fehler oder gleichwertig? → gehört in Regel 4.

### [NIEDRIG] Textebene gegen Satzbild
Die Fassung erbt, was die Textebene sagt. Native PDFs weichen selten ab, aber nicht nie. **Stichprobe, falls du magst** — vier Einträge mit erhöhtem Risiko: `all-in.de - Das Allgäu online!` · `HAO - der Onlinedienst des Hellweger Anzeigers Unna` · `Maerkische Allgemeine.de` (steht wirklich `ae`, während zwei Zeilen weiter `Märkische Oderzeitung Online` mit `ä` steht — **Quell-Eigenheit, nicht reparieren**) · `NEWS 89,4`.

## Vorschlag: Bewertungsregel 4 — `03.md`, Ebenen statt Form

Die Gold-Fassung schreibt einspaltige GFM-Tabelle + fetter Sub-Kopf. **Gleichwertig** ist jede Form, die **drei Dinge erhält**: (a) die Zuordnung Angebot→Vermarkter, (b) die Unterscheidung Haupt-Kopf / Sub-Kopf / Daten, (c) **beide Wiederholungen auf Seite 12**. **Nicht** gleichwertig ist das Einebnen der Ebenen (Sub-Kopf als gewöhnliche Datenzeile) und das Weglassen der Wiederholungen — dabei geht Information verloren, und im zweiten Fall genau die, um derentwillen die Klasse existiert.
