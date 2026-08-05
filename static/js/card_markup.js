/* CARD-MD — der Renderer für Kartentext (front / back / prompt / cloze_text).

   Warum es ihn gibt: der Karten-Agent schreibt Markdown, das Review zeigte es
   als Literal ("**Drill mit Molekül-Ionen:**"). Bei Chemie-Karten, wo ohnehin
   Indizes und Pfeile im Text stehen, sind die Sternchen besonders störend.

   ⚠️ DOKTRIN (CARD-SVG, unverändert): Kartentext ist ROHE Agenten-Eingabe und
   bekommt NIEMALS innerHTML. Dieses Modul baut ausschließlich DOM-Knoten
   (createTextNode / createElement / textContent) — exakt wie der Cloze-Renderer
   es seit CARD-SVG tut, dessen Nachfolger es ist. Die zwei Figuren-Container in
   review.js bleiben die EINZIGE innerHTML-Stelle (dort ist das SVG server-
   sanitisiert). Wer hier eine innerHTML-Zuweisung einbaut, hebt genau die
   Eigenschaft auf, die den Kartentext heute sicher macht.

   Unterstützt wird GENAU: **fett** · *kursiv* · Listenzeilen (- / *) · {{cloze}}.
   NICHT unterstützt (gesperrte Entscheidung): _…_, Backticks, Überschriften,
   Links, Tabellen, Bilder, 1.-Aufzählungen. Kartentext ist kurz und kein
   Dokument; jedes zusätzliche Muster ist eine weitere Chance, eine Formel zu
   zerreißen.

   ⚠️ `_` BLEIBT LITERAL — das ist der wichtigste Einzelbefund des Sprints und
   keine Nachlässigkeit: alle Unterstriche in Olis Korpus sind Tiefstellungen in
   Formeln (μ_max · S/(K_S + S), Δ_Sub E, Z_eff, E_A, C_s, f_mech(x)), keine
   Auszeichnung. Als Kursiv gelesen zerrisse `_` genau die naturwissenschaft-
   lichen Formeln, für die die Karten da sind.

   EIN Durchlauf kennt Cloze UND Auszeichnung — nicht zwei Pässe, die sich
   gegenseitig die Offsets verschieben. Das ist keine Eleganz, sondern nötig:
   im Korpus tragen cloze_text-Felder beides gleichzeitig (Karte 195
   "Molekül-**Kation**", Karte 206 "**endotherm (+)**"). */
(function (global) {
    'use strict';

    // Ein Opener darf keinen Whitespace RECHTS von sich haben, ein Closer
    // keinen LINKS (die flanking-Regel aus CommonMark, auf das Nötige
    // eingedampft). Sie ist der Grund, warum "a * b * c" Text bleibt und nur
    // "*kursiv*" kursiv wird — in einem Korpus voller Formeln ist ein
    // freistehender Stern eher ein Malzeichen als eine Auszeichnung.
    const isSpace = (ch) => ch === undefined || ch === ' ' || ch === '\t' || ch === '\n';

    const isListLine = (line) => line === '- ' || line === '* ' ||
        ((line.startsWith('- ') || line.startsWith('* ')) && line.length > 2);

    // Eine eingerückte Zeile direkt unter einem Listenpunkt ist dessen
    // FORTSETZUNG, keine neue Einheit — der Agent schreibt so die Rechnung
    // unter die Regel (Karte 191: "- Ga³⁺ + SO₃²⁻ …" / "  Ga₂(SO₃)₃ …").
    // 15 solcher Zeilen in 6 der 31 Listen-Felder. Ohne diesen Fall zerfiele
    // die Aufzählung in lauter Ein-Punkt-Listen und die Fortsetzung stünde
    // linksbündig NEBEN dem Punkt, zu dem sie gehört.
    const isContinuationLine = (line) => /^\s+\S/.test(line);

    /* --- Inline-Ebene ---------------------------------------------------- */
    /* Scannt EINE Zeile und hängt Knoten an `target`. Rekursiv für den Inhalt
       von **…**, *…* und {{…}} — dadurch fällt Verschachtelung (**… *so* …**,
       real im Korpus) ohne Sonderfall heraus.
       clozeMode: 'off' (Cloze ist Literal) | 'hide' (Front: …-Kasten)
                  | 'reveal' (Back: hervorgehobene Antwort) */
    function appendInline(target, text, clozeMode) {
        let buf = '';
        let i = 0;

        const flush = () => {
            if (buf) { target.appendChild(document.createTextNode(buf)); buf = ''; }
        };

        while (i < text.length) {
            const two = text.slice(i, i + 2);

            // {{cloze}} — nur wenn dieses Feld überhaupt ein Cloze-Feld ist.
            // Sonst bleibt {{…}} Literal, exakt wie vor diesem Sprint: in
            // front/back/prompt lief der Text über textContent.
            if (clozeMode !== 'off' && two === '{{') {
                const end = text.indexOf('}}', i + 2);
                if (end > i + 2) {
                    flush();
                    const span = document.createElement('span');
                    if (clozeMode === 'reveal') {
                        span.className = 'review-cloze-fill';
                        appendInline(span, text.slice(i + 2, end), 'off');
                    } else {
                        span.className = 'review-cloze-blank';
                        span.textContent = '…';
                    }
                    target.appendChild(span);
                    i = end + 2;
                    continue;
                }
            }

            // **fett** — der längere Marker wird ZUERST probiert, sonst risse
            // das erste * eines ** die Auszeichnung auf.
            if (two === '**' && !isSpace(text[i + 2])) {
                const end = findCloser(text, i + 2, '**');
                // end > i+2 ⇒ nicht-leerer Inhalt. "****" bleibt Literal statt
                // ein leeres <strong> zu erzeugen.
                if (end > i + 2) {
                    flush();
                    const strong = document.createElement('strong');
                    appendInline(strong, text.slice(i + 2, end), clozeMode);
                    target.appendChild(strong);
                    i = end + 2;
                    continue;
                }
            }

            // *kursiv*
            if (text[i] === '*' && two !== '**' && !isSpace(text[i + 1])) {
                const end = findCloser(text, i + 1, '*');
                if (end > i + 1) {
                    flush();
                    const em = document.createElement('em');
                    appendInline(em, text.slice(i + 1, end), clozeMode);
                    target.appendChild(em);
                    i = end + 1;
                    continue;
                }
            }

            // Kein Marker (oder einer ohne Partner) → Literal. Ein einzelnes *
            // ist Text, kein Fehler.
            buf += text[i];
            i += 1;
        }
        flush();
    }

    // Sucht den passenden Closer ab `from`. Liefert -1, wenn keiner existiert —
    // dann bleibt der Opener Literal. Beim Suchen eines EINZELNEN * werden
    // **-Paare übersprungen, damit "*a **b** c*" nicht am ersten Stern von **
    // abbricht.
    function findCloser(text, from, marker) {
        let i = from;
        while (i < text.length) {
            if (marker === '*' && text.slice(i, i + 2) === '**') { i += 2; continue; }
            if (text.slice(i, i + marker.length) === marker && !isSpace(text[i - 1])) {
                return i;
            }
            i += 1;
        }
        return -1;
    }

    /* --- Block-Ebene ------------------------------------------------------ */
    /* Zeilen, die mit "- " oder "* " beginnen, bilden zusammenhängende
       Aufzählungen. Alles andere bleibt Fließtext; die Zeilenumbrüche trägt
       weiterhin `white-space: pre-wrap` auf den Textflächen (style.css).

       ⚠️ Die \n-Buchführung ist load-bearing: <ul>/<li> sind block-level und
       bringen ihren Umbruch selbst mit. Bliebe der \n des Textstroms daneben
       stehen, entstünde pro Listenzeile ein ZWEITER Umbruch (gemessen: +21 px
       je Zeile). Deshalb konsumiert ein Listenblock die \n, die zu ihm
       gehören — inklusive der Trennung zur Nachbarzeile. */
    function renderCardMarkup(target, text, clozeMode) {
        target.textContent = '';
        if (!text) return;
        const mode = clozeMode || 'off';

        const lines = text.split('\n');
        let i = 0;
        let prevWasList = false;

        while (i < lines.length) {
            if (isListLine(lines[i])) {
                const ul = document.createElement('ul');
                ul.className = 'review-md-list';
                while (i < lines.length && isListLine(lines[i])) {
                    const li = document.createElement('li');
                    appendInline(li, lines[i].slice(2), mode);
                    i += 1;
                    // Fortsetzungszeilen in denselben Punkt ziehen. Der
                    // führende Whitespace wird dabei VERBRAUCHT: er war die
                    // Zugehörigkeits-Angabe, und die trägt jetzt der <li>.
                    // Ihn zusätzlich stehen zu lassen, hieße ihn doppelt zu
                    // zählen (pre-wrap gilt im <li> weiter).
                    while (i < lines.length && isContinuationLine(lines[i])) {
                        li.appendChild(document.createTextNode('\n'));
                        appendInline(li, lines[i].replace(/^\s+/, ''), mode);
                        i += 1;
                    }
                    ul.appendChild(li);
                }
                target.appendChild(ul);
                prevWasList = true;
                continue;
            }
            // Trenner zur vorherigen Einheit — entfällt an einer Listengrenze,
            // weil die <ul> den Umbruch schon geleistet hat.
            if (i > 0 && !prevWasList) {
                target.appendChild(document.createTextNode('\n'));
            }
            appendInline(target, lines[i], mode);
            prevWasList = false;
            i += 1;
        }
    }

    global.renderCardMarkup = renderCardMarkup;
})(typeof window !== 'undefined' ? window : globalThis);
