/* Review ("Lernen") UI — walks the due queue from /api/review-state, reveals the
   answer, rates via POST /api/cards/<id>/review (FSRS), lets the user flag
   "wackelt" (Vertiefen) or set a note via POST /api/cards/<id>/annotate, and
   delete the card via DELETE /api/cards/<id>. Every state-changing request goes
   through the global base.html fetch wrapper, which adds X-CSRFToken (DELETE is
   covered too) — a raw fetch without it would 400. Card TEXT is rendered
   exclusively via textContent / DOM nodes (XSS-safe). The ONE exception
   (CARD-SVG): the two figure containers get innerHTML — their SVG arrives
   SERVER-sanitized (services/svg_sanitize.py, applied in Card.to_dict), never
   raw agent input.

   CARD-MD: die vier Kartenfelder (front, back, prompt, cloze_text) laufen seit
   diesem Sprint durch renderCardMarkup (static/js/card_markup.js) statt über
   textContent — der Renderer baut DOM-KNOTEN, keine innerHTML, die Doktrin
   steht also unverändert. Er hat renderCloze abgelöst und aufgenommen: EIN
   Durchlauf kennt Cloze und Auszeichnung. Jede ANDERE textContent-Stelle hier
   bleibt, wie sie ist — Badges, Zähler, Fortschritt und Fehlermeldungen sind
   keine Agenten-Eingabe und haben in einem Markdown-Renderer nichts zu suchen. */
(function () {
    'use strict';

    const REVIEW_STATE_URL = window.PageData.reviewStateUrl;
    const COLLECTIONS_URL = window.PageData.collectionsUrl;
    const LEARN_SETTINGS_URL = window.PageData.learnSettingsUrl;
    const LEARN_STATS_URL = window.PageData.learnStatsUrl;
    const LEARN_SIMULATE_URL = window.PageData.learnSimulateUrl;

    let queue = [];
    let index = 0;
    let totalDue = 0;
    // LEARN-COUNT: cap-line numbers ("N Reviews fällig · M neu verfügbar") live
    // in module state so each rating/delete can decrement them; null until the
    // server sends them in load().
    let reviewCount = null;
    let newCount = null;
    let revealed = false;
    let busy = false;
    // LEARN-UP scope: checked collection ids; empty = "Alles fällig".
    // Several checked → the union is studied (?collection=1,2,3).
    let scopeIds = [];
    let collections = [];  // cached /api/collections (pills + footer add)
    // LEARN-QUEUE orphan scope: cards in NO collection. Its pill only exists
    // while there are due orphans — the count rides on /api/review-state
    // (uncollected_count), NOT on /api/collections, whose bare array the iOS
    // app decodes as [LearnCollection].
    let scopeUncollected = false;
    let uncollectedCount = 0;
    // LEARN-MORE session gesture (page-lifetime only, NEVER persisted — a page
    // reload starts capped again, protecting the daily boundary): stage 1 lifts
    // today's caps, stage 2 borrows future Berlin days via ?ahead=<n>.
    let sessionUncapped = false;
    let sessionAhead = 0;          // 0 = off; else the ahead step to fetch with
    let remainingToday = 0;        // now-due cards the cap held back (server)
    let nextAhead = null;          // {days, count} | null — next borrowable day
    let dayEnd = null;             // today's Berlin day-end (aware, from server)
    let moreMode = null;           // 'uncapped' | 'ahead' — what the button does

    const el = (id) => document.getElementById(id);
    const loadingEl = el('review-loading');
    const emptyEl = el('review-empty');
    const doneEl = el('review-done');
    const doneTextEl = el('review-done-text');
    const cardEl = el('review-card');
    const progressEl = el('review-progress');
    const typeBadge = el('review-type-badge');
    const stateBadge = el('review-state-badge');
    const sourceEl = el('review-source');
    const questionEl = el('review-question');
    const figureFrontEl = el('review-figure-front');
    const figureBackEl = el('review-figure-back');
    const genHintEl = el('review-generative-hint');
    const revealBtn = el('review-reveal-btn');
    const answerWrap = el('review-answer-wrap');
    const answerLabel = el('review-answer-label');
    const answerEl = el('review-answer');
    const ratingEl = el('review-rating');
    const deepenBtn = el('review-deepen-btn');
    const noteToggle = el('review-note-toggle');
    const deleteBtn = el('review-delete-btn');
    const noteWrap = el('review-note-wrap');
    const noteInput = el('review-note-input');
    const noteSave = el('review-note-save');
    const orderSelect = el('review-order-select');
    const scopeList = el('review-scope-list');
    const scopeAllBox = el('review-scope-all');
    const limitNewInput = el('review-limit-new');
    const limitReviewsInput = el('review-limit-reviews');
    const capInfoEl = el('review-cap-info');
    const statsToggle = el('review-stats-toggle');
    const statsSection = el('review-stats');
    const emptyTitle = el('review-empty-title');
    const emptyText = el('review-empty-text');
    const collectionToggle = el('review-collection-toggle');
    const collectionWrap = el('review-collection-wrap');
    const collectionSelect = el('review-collection-select');
    const collectionNew = el('review-collection-new');
    const collectionAdd = el('review-collection-add');
    const moreWrap = el('review-more');
    const moreTextEl = el('review-more-text');
    const moreBtn = el('review-more-btn');

    const alertContainer = () => el('review-alert-container');
    const currentCard = () => queue[index];
    const show = (elm) => elm.classList.remove('hidden');
    const hide = (elm) => elm.classList.add('hidden');

    // The API's card timestamps are NAIVE UTC isoformat — a bare new Date(s)
    // would read them as local time (Berlin offset shift). Pin them to UTC
    // unless the string already carries a zone (day_end does).
    const parseUTC = (iso) =>
        new Date(/[zZ]$|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + 'Z');

    // --- Kartentext (CARD-MD) ------------------------------------------------
    // Ein Aufruf für alle vier Felder. renderCardMarkup baut DOM-Knoten und
    // kennt **fett** · *kursiv* · Aufzählungen · {{cloze}}. Der clozeMode ist
    // ausdrücklich nur für cloze_text 'hide'/'reveal' — in front/back/prompt
    // bleibt {{…}} Literal, exakt wie vor diesem Sprint (im Korpus kommt es
    // dort nirgends vor; der Renderer soll trotzdem nichts versprechen, wonach
    // niemand gefragt hat).
    const renderText = (target, text, clozeMode) =>
        window.renderCardMarkup(target, text || '', clozeMode || 'off');

    const isCloze = (card) => card.type === 'atomic' && !card.front && !!card.cloze_text;

    // CARD-SVG: the ONLY innerHTML sink for card data — the SVG is sanitized
    // server-side (Card.to_dict → services/svg_sanitize.py). Falsy (the API
    // sends null, never '') → clear AND hide, so no stale figure survives.
    function renderFigure(container, svg) {
        if (!svg) {
            container.innerHTML = '';
            hide(container);
        } else {
            container.innerHTML = svg;
            show(container);
        }
    }

    function renderCard(card) {
        revealed = false;

        typeBadge.textContent = card.type === 'generative' ? 'Generativ' : 'Atomar';
        typeBadge.className = 'type-badge ' + (card.type === 'generative' ? 'type-generative' : 'type-atomic');
        stateBadge.classList.toggle('hidden', card.state !== 'wackelt');

        if (card.source_doc_title) {
            sourceEl.textContent = 'aus: ' + card.source_doc_title;
            show(sourceEl);
        } else {
            hide(sourceEl);
        }

        if (card.type === 'generative') {
            renderText(questionEl, card.prompt);
            show(genHintEl);
            answerLabel.textContent = 'Musterantwort';
        } else if (isCloze(card)) {
            renderText(questionEl, card.cloze_text, 'hide');
            hide(genHintEl);
            answerLabel.textContent = 'Lösung';
        } else {
            renderText(questionEl, card.front);
            hide(genHintEl);
            answerLabel.textContent = 'Lösung';
        }
        // Front figure now, back figure explicitly RESET — without it, card
        // N+1 would briefly flash card N's back figure on reveal (stale trap).
        renderFigure(figureFrontEl, card.front_svg);
        renderFigure(figureBackEl, null);

        show(revealBtn);
        hide(answerWrap);
        hide(ratingEl);
        hide(noteWrap);
        noteToggle.classList.remove('is-active');
        resetCollectionPanel();
        noteInput.value = card.note || '';
        deepenBtn.classList.toggle('is-active', card.state === 'wackelt');

        show(cardEl);
        updateProgress();
    }

    function revealAnswer() {
        if (revealed) return;
        const card = currentCard();
        if (card.type === 'generative') {
            // Der Platzhalter ist UI-Text, keine Agenten-Eingabe → textContent.
            if (card.back) renderText(answerEl, card.back);
            else answerEl.textContent = '(keine Musterantwort hinterlegt)';
        } else if (isCloze(card)) {
            renderText(answerEl, card.cloze_text, 'reveal');
        } else {
            renderText(answerEl, card.back);
        }
        renderFigure(figureBackEl, card.back_svg);
        revealed = true;
        hide(revealBtn);
        show(answerWrap);
        show(ratingEl);
    }

    function updateProgress() {
        // Session progress bar, NOT a due counter — the denominator stays the
        // session size (no "fällig", locked decision). The shrinking stack is
        // shown by the pills + cap line, which run live via decrementPoolCounts.
        progressEl.textContent = `Karte ${Math.min(index + 1, totalDue)} von ${totalDue}`;
    }

    // LEARN-COUNT: re-render the cap line from module state. Wording unchanged
    // from the original load() text; untouched while the server never sent
    // numbers (reviewCount stays null).
    function renderCapInfo() {
        if (reviewCount === null) return;
        capInfoEl.textContent = `${reviewCount} Reviews fällig · ${newCount} neu verfügbar`;
    }

    // LEARN-COUNT: a due card left the pool — rated into the future or deleted.
    // Sink the badge of EVERY collection the card is in (a card in two
    // collections counts in each; one without collections touches none) and the
    // matching cap-line number. The new-vs-review split MUST be read from the
    // pre-rating queue object: new = stability === null (LEARN-UP convention);
    // after rating, stability is set and the distinction is gone.
    function decrementPoolCounts(card) {
        const wasNew = !card.review || card.review.stability === null;
        const cols = card.collections || [];
        cols.forEach((col) => {
            const cached = collections.find((c) => c.id === col.id);
            if (cached) cached.due_count = Math.max(0, (cached.due_count || 0) - 1);
        });
        // LEARN-QUEUE: a card in NO collection sinks the orphan badge instead —
        // same raw due bookkeeping, the pill just has no /api/collections row.
        if (!cols.length) uncollectedCount = Math.max(0, uncollectedCount - 1);
        renderScopePills();
        if (wasNew) {
            if (newCount !== null) newCount = Math.max(0, newCount - 1);
        } else if (reviewCount !== null) {
            reviewCount = Math.max(0, reviewCount - 1);
        }
        renderCapInfo();
    }

    function advance() {
        index += 1;
        if (index >= queue.length) {
            finishSession();
        } else {
            renderCard(currentCard());
        }
    }

    function finishSession() {
        hide(cardEl);
        doneTextEl.textContent =
            `Alle ${totalDue} ${totalDue === 1 ? 'fällige Karte' : 'fälligen Karten'} wiederholt.`;
        renderMoreOffer();
        progressEl.textContent = '';
        show(doneEl);
        // Authoritative resync (LEARN-COUNT safeguard): server truth replaces
        // the local decrementPoolCounts bookkeeping — should it ever drift
        // during a session, it self-heals here. Do not remove.
        loadCollections();
    }

    // LEARN-MORE: exactly ONE staged offer in the done panel. Stage 1 (today's
    // capped overhang) always wins over stage 2 (borrowing future days), and
    // stage 2 never happens without its own click — its copy names the price
    // (pulled-forward cards repeat earlier than scheduled). The numbers are
    // the session-start fetch; the click reloads and shows server truth.
    function renderMoreOffer() {
        if (remainingToday > 0) {
            moreTextEl.textContent = remainingToday === 1
                ? 'Über dem Tageslimit liegt noch 1 fällige Karte.'
                : `Über dem Tageslimit liegen noch ${remainingToday} fällige Karten.`;
            moreBtn.textContent = 'Mehr lernen';
            moreMode = 'uncapped';
            show(moreWrap);
        } else if (nextAhead) {
            const n = nextAhead.count;
            const when = nextAhead.days === 1 ? 'Bis morgen' : `In ${nextAhead.days} Tagen`;
            moreTextEl.textContent =
                `${when} ${n === 1 ? 'wird 1 Karte' : `werden ${n} Karten`} fällig. ` +
                'Vorziehen wiederholt sie früher als geplant.';
            moreBtn.textContent = nextAhead.days === 1 ? 'Morgen vorziehen' : 'Vorziehen';
            moreMode = 'ahead';
            show(moreWrap);
        } else {
            moreMode = null;
            hide(moreWrap);
        }
    }

    function onMoreClick() {
        if (moreMode === 'uncapped') {
            sessionUncapped = true;
        } else if (moreMode === 'ahead' && nextAhead) {
            sessionAhead = nextAhead.days;   // next click gets the NEXT step
        } else {
            return;
        }
        load();
    }

    function setRatingDisabled(disabled) {
        ratingEl.querySelectorAll('.review-rate-btn').forEach((b) => { b.disabled = disabled; });
    }

    async function rate(rating) {
        if (busy || !revealed) return;
        busy = true;
        setRatingDisabled(true);
        // Snapshot BEFORE rating: decrementPoolCounts needs the pre-rating
        // stability (new-vs-review) and the card's collections.
        const card = currentCard();
        try {
            const resp = await fetch(`/api/cards/${card.id}/review`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ rating }),
            });
            const updated = await safeJSON(resp);
            if (!resp.ok) {
                // LOST-UPDATE: a 409 ("gerade gleichzeitig bewertet … noch
                // einmal bewerten") is our own status with a precise server
                // sentence — carry it to the alert. Every other failure keeps
                // the generic line in the catch below.
                const err = new Error('rate failed');
                err.serverMessage = (updated && typeof updated.error === 'string') ? updated.error : '';
                throw err;
            }
            // Load-bearing (LEARN-COUNT): the response is the updated card,
            // incl. the NEW review.due. A card leaves the pool only when that
            // due lies beyond TODAY'S day end — the counters mean "what's
            // still on today", not "due this second": a 10-minute "Nochmal"
            // step stays in the pool on purpose, a multi-day step leaves it.
            // Deliberately NO timezone arithmetic here (LEARN-MORE): the
            // Berlin boundary comes from the server (day_end), the naive-UTC
            // due is pinned via parseUTC — we only compare instants. Missing
            // due or day_end → treat as still due (no decrement; the
            // finishSession resync heals any gap).
            const newDue = updated && updated.review && updated.review.due;
            const stillDue = !newDue || !dayEnd || parseUTC(newDue) <= dayEnd;
            if (!stillDue) decrementPoolCounts(card);
            advance();
        } catch (e) {
            showAlert(alertContainer(), 'danger', e.serverMessage ||
                'Bewertung fehlgeschlagen. Verbindung prüfen und erneut versuchen.');
        } finally {
            busy = false;
            setRatingDisabled(false);
        }
    }

    async function deepen() {
        const card = currentCard();
        try {
            const resp = await fetch(`/api/cards/${card.id}/annotate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ state: 'wackelt' }),
            });
            await safeJSON(resp);
            if (!resp.ok) throw new Error();
            card.state = 'wackelt';
            show(stateBadge);
            deepenBtn.classList.add('is-active');
            showToast('Als „wackelt“ markiert');
        } catch (e) {
            showAlert(alertContainer(), 'danger', 'Konnte nicht markieren. Erneut versuchen.');
        }
    }

    async function saveNote() {
        const card = currentCard();
        const note = noteInput.value;
        try {
            const resp = await fetch(`/api/cards/${card.id}/annotate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ note }),
            });
            await safeJSON(resp);
            if (!resp.ok) throw new Error();
            card.note = note || null;
            hide(noteWrap);
            noteToggle.classList.remove('is-active');
            showToast('Notiz gespeichert');
        } catch (e) {
            showAlert(alertContainer(), 'danger',
                'Notiz konnte nicht gespeichert werden. Erneut versuchen.');
        }
    }

    async function deleteCard() {
        if (busy) return;
        const card = currentCard();
        // Irreversible → confirm. Mirrors the library delete-confirm convention.
        if (!confirm('Diese Karte wirklich löschen? Das kann nicht rückgängig gemacht werden.')) return;
        busy = true;
        try {
            // DELETE rides the global fetch wrapper for X-CSRFToken (state-changing);
            // a raw fetch would 400. Owner-scoped server-side (404 on foreign/missing).
            const resp = await fetch(`/api/cards/${card.id}`, { method: 'DELETE' });
            const body = await safeJSON(resp);
            if (!resp.ok) {
                // LOST-UPDATE: the 409 ("gerade bewertet … noch einmal
                // löschen") carries its own sentence — show it, not the
                // generic toast.
                const err = new Error('delete failed');
                err.serverMessage = (body && typeof body.error === 'string') ? body.error : '';
                throw err;
            }
            // The card is GONE (not rated) — drop it from the queue and the due
            // counter, keeping `index` so the next card shifts into this slot.
            // A deleted due card left the pool too → same pill/cap decrements
            // as a future-due rating (card still holds collections + stability).
            decrementPoolCounts(card);
            queue.splice(index, 1);
            totalDue = Math.max(0, totalDue - 1);
            showToast('Karte gelöscht');
            if (index >= queue.length) {
                // Cleared the tail — re-fetch so the panel reflects the true due
                // state (Empty "Nichts fällig" when none remain), not a
                // "wiederholt" done-count a delete didn't earn.
                load();
            } else {
                renderCard(currentCard());
            }
        } catch (e) {
            showToast(e.serverMessage || 'Karte konnte nicht gelöscht werden. Erneut versuchen.',
                      { level: 'danger' });
        } finally {
            busy = false;
        }
    }

    function scopeUrl() {
        const params = [];
        if (scopeIds.length) params.push(`collection=${scopeIds.join(',')}`);
        // Unions with collection= server-side (not an alternative to it).
        if (scopeUncollected) params.push('uncollected=1');
        // LEARN-MORE gesture: ahead implies uncapped server-side, so send only
        // one of the two. Without either the fetch is the plain capped state.
        if (sessionAhead > 0) params.push(`ahead=${sessionAhead}`);
        else if (sessionUncapped) params.push('uncapped=1');
        return params.length ? `${REVIEW_STATE_URL}?${params.join('&')}` : REVIEW_STATE_URL;
    }

    // A scope or settings change redefines the session — the "mehr lernen"
    // gesture belongs to the previous one and must not leak across.
    function resetSessionMode() {
        sessionUncapped = false;
        sessionAhead = 0;
    }

    function scopeLabel() {
        const names = collections
            .filter((c) => scopeIds.includes(c.id))
            .map((c) => c.name);
        if (scopeUncollected) names.push('Ohne Sammlung');
        return names.join(', ');
    }

    function applyEmptyScope() {
        // Scope-aware empty state — "Nichts fällig in <Auswahl>" vs. the global text.
        if (!scopeIds.length && !scopeUncollected) {
            emptyTitle.textContent = 'Nichts fällig.';
            emptyText.textContent =
                'Du bist mit dem Wiederholen durch. Neue Karten erscheinen, sobald sie wieder fällig sind.';
        } else {
            emptyTitle.textContent = `Nichts fällig in „${scopeLabel()}“.`;
            emptyText.textContent =
                'In dieser Auswahl ist gerade nichts dran. Wähle oben andere Sammlungen oder „Alles fällig“.';
        }
    }

    async function load() {
        show(loadingEl); hide(emptyEl); hide(doneEl); hide(cardEl);
        try {
            const resp = await fetch(scopeUrl());
            const data = await safeJSON(resp);
            if (!resp.ok) throw new Error();
            queue = data.due_cards || [];
            totalDue = (typeof data.due_count === 'number') ? data.due_count : queue.length;
            index = 0;
            // Nach-Cap-Zähler (P3): was die heutige Session noch hergibt — ab
            // LEARN-COUNT in den Modul-State, damit die Zeile pro Karte mitläuft.
            if (typeof data.review_count === 'number') {
                reviewCount = data.review_count;
                newCount = data.new_count;
                renderCapInfo();
            }
            // LEARN-QUEUE: server truth for the orphan badge + pill visibility
            // (global, unscoped — so picking a collection never hides it).
            uncollectedCount = (typeof data.uncollected_count === 'number')
                ? data.uncollected_count : 0;
            renderScopePills();
            // LEARN-MORE state (fresh per fetch, scope-filtered server-side).
            remainingToday = (typeof data.remaining_today === 'number') ? data.remaining_today : 0;
            nextAhead = data.next_ahead || null;
            dayEnd = data.day_end ? new Date(data.day_end) : null;
            hide(loadingEl);
            // Clear any stale "Karte N von M" — matters when a delete empties the
            // queue and re-loads into this branch (finishSession clears it too).
            if (!queue.length) {
                progressEl.textContent = '';
                if (remainingToday > 0) {
                    // Empty ONLY because of the cap (today's budget is spent,
                    // due cards remain): show the done panel with the stage-1
                    // offer. Without this, a page reload — which correctly
                    // starts capped again — would bury the overhang for good.
                    // The pure stage-2 case stays on the empty panel: nothing
                    // was capped away, borrowing belongs to a finished session.
                    doneTextEl.textContent = 'Tagespensum erreicht.';
                    renderMoreOffer();
                    show(doneEl);
                    return;
                }
                applyEmptyScope(); show(emptyEl); return;
            }
            renderCard(currentCard());
        } catch (e) {
            hide(loadingEl);
            showAlert(alertContainer(), 'danger',
                'Karten konnten nicht geladen werden. Seite neu laden.');
        }
    }

    // --- Study-set launcher (LEARN-UP) ---------------------------------------
    // Collection checkbox pills with raw due badges (Review.due <= now, same
    // definition as the queue). The old tag optgroup is gone by design — tags
    // live on in the /tags manager, they are no longer a study axis.
    function makeScopePill(label, dueCount, checked) {
        const pill = document.createElement('label');
        pill.className = 'review-scope-pill';
        const box = document.createElement('input');
        box.type = 'checkbox';
        box.checked = checked;
        const name = document.createElement('span');
        name.textContent = label;
        const badge = document.createElement('span');
        badge.className = 'review-scope-pill__badge';
        badge.textContent = String(dueCount || 0);
        badge.title = 'Fällige Karten';
        pill.append(box, name, badge);
        return { pill, box };
    }

    function renderScopePills() {
        // LEARN-QUEUE invariant, enforced HERE because this is the one place
        // that knows "pill exists iff count > 0": when the last orphan is gone
        // the check falls away with the pill. Otherwise a checked orphan scope
        // would survive as a filter nobody can untick.
        if (uncollectedCount <= 0) scopeUncollected = false;
        // Rebuild the generated pills, preserving the current checks.
        scopeList
            .querySelectorAll('.review-scope-pill[data-col], .review-scope-pill[data-uncollected]')
            .forEach((p) => p.remove());
        (collections || []).forEach((c) => {
            const { pill, box } = makeScopePill(c.name, c.due_count, scopeIds.includes(c.id));
            pill.dataset.col = String(c.id);
            box.dataset.colId = String(c.id);
            scopeList.appendChild(pill);
        });
        if (uncollectedCount > 0) {
            // Last in the row — it is the remainder, not a study set.
            const { pill, box } = makeScopePill('Ohne Sammlung', uncollectedCount,
                scopeUncollected);
            pill.dataset.uncollected = '1';
            box.dataset.uncollected = '1';
            scopeList.appendChild(pill);
        }
        scopeAllBox.checked = scopeIds.length === 0 && !scopeUncollected;
    }

    async function loadCollections() {
        try {
            const resp = await fetch(COLLECTIONS_URL);
            collections = resp.ok ? (await safeJSON(resp)) || [] : [];
            renderScopePills();
            populateCollectionFooter();
        } catch (e) {
            // Non-fatal: the queue still loads, only the pills stay bare.
        }
    }

    // --- Learn settings (LEARN-UP): ordering mode + daily limits -------------
    // Server-side ordering/capping per fetch; the controls persist via the
    // settings blob and simply re-fetch the queue.
    async function loadLearnSettings() {
        try {
            const resp = await fetch(LEARN_SETTINGS_URL);
            if (!resp.ok) return;  // non-fatal: controls show the defaults
            const data = await safeJSON(resp);
            if (!data) return;
            if (data.ordering_mode) orderSelect.value = data.ordering_mode;
            if (typeof data.daily_new_limit === 'number') limitNewInput.value = data.daily_new_limit;
            if (typeof data.daily_review_limit === 'number') limitReviewsInput.value = data.daily_review_limit;
        } catch (e) { /* non-fatal */ }
        renderOrderHint();  // whatever the select ended up showing, explain it
    }

    // LEARN-HINT-WEB: the explanation of the selected ordering mode lives in
    // the template (both texts, verbatim from iOS); this only toggles which
    // one is visible. No title tooltip — touch has no hover.
    function renderOrderHint() {
        document.querySelectorAll('[data-order-hint]').forEach((el) => {
            el.classList.toggle('hidden', el.dataset.orderHint !== orderSelect.value);
        });
    }

    async function putLearnSetting(patch, errorText) {
        try {
            // PUT rides the global fetch wrapper (X-CSRFToken).
            const resp = await fetch(LEARN_SETTINGS_URL, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(patch),
            });
            await safeJSON(resp);
            if (!resp.ok) throw new Error();
            resetSessionMode();  // changed limits/ordering = a new session
            load();
        } catch (e) {
            showToast(errorText, { level: 'danger' });
            loadLearnSettings();  // revert the controls to server truth
        }
    }

    function onOrderChange() {
        renderOrderHint();
        putLearnSetting({ ordering_mode: orderSelect.value },
            'Reihenfolge konnte nicht gespeichert werden.');
    }

    function onLimitChange(e) {
        const key = e.target === limitNewInput ? 'daily_new_limit' : 'daily_review_limit';
        const value = Number.parseInt(e.target.value, 10);
        // NaN serialisiert zu null → Server-400 → Revert-Pfad greift.
        putLearnSetting({ [key]: Number.isNaN(value) ? null : value },
            'Limit konnte nicht gespeichert werden.');
    }

    function onScopeChange(e) {
        // Delegated: "Alles fällig" clears the selection; a collection box
        // toggles its id in/out. renderScopePills syncs the "Alles fällig"
        // check (on iff nothing selected), then the queue re-fetches.
        if (e.target === scopeAllBox) {
            scopeIds = [];
            scopeUncollected = false;
        } else if (e.target.dataset && e.target.dataset.uncollected) {
            scopeUncollected = e.target.checked;
        } else if (e.target.dataset && e.target.dataset.colId) {
            const id = Number(e.target.dataset.colId);
            scopeIds = e.target.checked
                ? scopeIds.concat(scopeIds.includes(id) ? [] : [id])
                : scopeIds.filter((x) => x !== id);
        } else {
            return;
        }
        renderScopePills();
        resetSessionMode();  // a different scope = a new session, capped again
        load();
    }

    // --- Karte → Sammlung (footer) -------------------------------------------
    function populateCollectionFooter() {
        // Rebuild the existing-collection options, preserving the placeholder and
        // the "+ Neue Sammlung" sentinel at the ends.
        collectionSelect.querySelectorAll('option[data-existing]').forEach((o) => o.remove());
        const newOpt = collectionSelect.querySelector('option[value="__new__"]');
        (collections || []).forEach((c) => {
            const opt = document.createElement('option');
            opt.value = String(c.id);
            opt.textContent = c.name;
            opt.dataset.existing = '1';
            collectionSelect.insertBefore(opt, newOpt);
        });
    }

    function resetCollectionPanel() {
        hide(collectionWrap);
        collectionToggle.classList.remove('is-active');
        collectionSelect.value = '';
        collectionNew.value = '';
        hide(collectionNew);
    }

    async function addToCollection() {
        if (busy) return;
        const card = currentCard();
        const sel = collectionSelect.value;
        if (!sel) { showToast('Bitte eine Sammlung wählen.', { level: 'danger' }); return; }
        busy = true;
        collectionAdd.disabled = true;
        try {
            let collectionId;
            if (sel === '__new__') {
                const name = collectionNew.value.trim();
                if (!name) { showToast('Bitte einen Namen eingeben.', { level: 'danger' }); return; }
                const cResp = await fetch(COLLECTIONS_URL, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name }),
                });
                const cData = await safeJSON(cResp);
                if (cResp.status === 409) {
                    showToast('Sammlung existiert bereits.', { level: 'danger' }); return;
                }
                if (!cResp.ok) throw new Error();
                collectionId = cData.id;
                collections.push({ id: cData.id, name: cData.name, card_count: 0, due_count: 0 });
            } else {
                collectionId = Number(sel);
            }
            const resp = await fetch(`/api/collections/${collectionId}/cards`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ card_id: card.id }),
            });
            await safeJSON(resp);
            if (!resp.ok) throw new Error();
            const cached = collections.find((c) => c.id === collectionId);
            const colName = (cached || {}).name || 'Sammlung';
            // LEARN-QUEUE: this card just stopped being an orphan — move its raw
            // due count over instead of leaving it in both pools. Without the
            // local `card.collections` update, rating it later would sink the
            // orphan badge a second time (same bookkeeping as
            // decrementPoolCounts, which reads exactly this field).
            if (!(card.collections || []).length) {
                uncollectedCount = Math.max(0, uncollectedCount - 1);
                if (cached) cached.due_count = (cached.due_count || 0) + 1;
            }
            card.collections = (card.collections || [])
                .concat([{ id: collectionId, name: colName }]);
            showToast(`Zu „${colName}“ hinzugefügt`);
            resetCollectionPanel();
            // Keep the launcher pills + footer select in sync with a freshly
            // created collection.
            renderScopePills();
            populateCollectionFooter();
        } catch (e) {
            showToast('Konnte nicht hinzufügen. Erneut versuchen.', { level: 'danger' });
        } finally {
            busy = false;
            collectionAdd.disabled = false;
        }
    }

    // --- Statistik (LEARN-UP P4) ---------------------------------------------
    // Lazy: erst beim ersten Aufklappen gefetcht; danach bei jedem Aufklappen
    // aktualisiert (billig, hält die Zahlen ehrlich).
    async function loadStats() {
        try {
            const resp = await fetch(LEARN_STATS_URL);
            const data = await safeJSON(resp);
            if (!resp.ok) throw new Error();
            renderStats(data);
        } catch (e) {
            showAlert(alertContainer(), 'danger',
                'Statistik konnte nicht geladen werden. Erneut versuchen.');
        }
    }

    function renderStats(data) {
        const t = data.today;
        el('stats-today').textContent =
            `${t.reviews_due} Reviews fällig · ${t.new_available} neu verfügbar — ` +
            `heute schon gelernt: ${t.reviews_done} Reviews, ${t.new_done} neu.`;

        const fc = el('stats-forecast');
        fc.textContent = '';
        const maxCount = Math.max(1, data.forecast.overdue,
            ...data.forecast.days.map((d) => d.count));
        if (data.forecast.overdue > 0) {
            const bar = document.createElement('div');
            bar.className = 'stats-forecast__bar stats-forecast__bar--backlog';
            bar.style.height = `${Math.max(4, Math.round(100 * data.forecast.overdue / maxCount))}%`;
            bar.title = `Rückstand: ${data.forecast.overdue} überfällig`;
            fc.appendChild(bar);
        }
        data.forecast.days.forEach((d) => {
            const bar = document.createElement('div');
            bar.className = 'stats-forecast__bar';
            bar.style.height = `${Math.max(2, Math.round(100 * d.count / maxCount))}%`;
            bar.title = `${d.date}: ${d.count}`;
            fc.appendChild(bar);
        });
        el('stats-backlog').textContent = data.forecast.overdue > 0
            ? `Rückstand: ${data.forecast.overdue} überfällig (roter Balken).`
            : 'Kein Rückstand.';

        const m = data.maturity;
        el('stats-maturity').textContent =
            `Neu ${m.neu} · Jung ${m.jung} · Reif ${m.reif} (reif = Intervall ab 21 Tagen)`;

        const r = data.retention;
        el('stats-retention').textContent = r.rate === null
            ? `Noch zu wenige Reviews im ${r.window_days}-Tage-Fenster.`
            : `Ist ${Math.round(100 * r.rate)} % (${r.pass} gewusst / ${r.fail} vergessen, ` +
              `${r.window_days} Tage) · Ziel ${Math.round(100 * r.desired)} %`;
    }

    function onStatsToggle() {
        const opening = statsSection.classList.contains('hidden');
        statsSection.classList.toggle('hidden', !opening);
        statsToggle.textContent = opening ? 'Statistik ausblenden' : 'Statistik anzeigen';
        if (opening) {
            if (!el('stats-sim-new').value) {
                el('stats-sim-new').value = limitNewInput.value || '10';
            }
            loadStats();
        }
    }

    async function runSimulation() {
        const btn = el('stats-sim-run');
        const resultEl = el('stats-sim-result');
        btn.disabled = true;
        try {
            const retention = el('stats-sim-retention').value;
            const perDay = Number.parseInt(el('stats-sim-new').value, 10);
            let url = `${LEARN_SIMULATE_URL}?retention=${retention}`;
            if (!Number.isNaN(perDay)) url += `&new_per_day=${perDay}`;
            const resp = await fetch(url);
            const data = await safeJSON(resp);
            if (!resp.ok) throw new Error();
            resultEl.textContent =
                `≈ ${data.reviews_per_day} Reviews/Tag bei ${data.new_per_day} neuen (Schätzung)`;
        } catch (e) {
            showToast('Simulation fehlgeschlagen.', { level: 'danger' });
        } finally {
            btn.disabled = false;
        }
    }

    scopeList.addEventListener('change', onScopeChange);
    orderSelect.addEventListener('change', onOrderChange);
    limitNewInput.addEventListener('change', onLimitChange);
    limitReviewsInput.addEventListener('change', onLimitChange);
    statsToggle.addEventListener('click', onStatsToggle);
    el('stats-sim-run').addEventListener('click', runSimulation);
    collectionToggle.addEventListener('click', () => {
        const opening = collectionWrap.classList.contains('hidden');
        // Mutually exclusive with the note panel — only one footer drawer open.
        hide(noteWrap); noteToggle.classList.remove('is-active');
        collectionWrap.classList.toggle('hidden', !opening);
        collectionToggle.classList.toggle('is-active', opening);
    });
    collectionSelect.addEventListener('change', () => {
        const isNew = collectionSelect.value === '__new__';
        collectionNew.classList.toggle('hidden', !isNew);
        if (isNew) collectionNew.focus();
    });
    collectionAdd.addEventListener('click', addToCollection);

    revealBtn.addEventListener('click', revealAnswer);
    ratingEl.addEventListener('click', (e) => {
        const btn = e.target.closest('.review-rate-btn');
        if (btn && !btn.disabled) rate(btn.dataset.rating);
    });
    deepenBtn.addEventListener('click', deepen);
    noteToggle.addEventListener('click', () => {
        noteWrap.classList.toggle('hidden');
        const open = !noteWrap.classList.contains('hidden');
        noteToggle.classList.toggle('is-active', open);
        // Mutually exclusive with the Sammlung drawer — only one open at a time.
        if (open) { resetCollectionPanel(); noteInput.focus(); }
    });
    noteSave.addEventListener('click', saveNote);
    deleteBtn.addEventListener('click', deleteCard);
    el('review-reload').addEventListener('click', load);
    moreBtn.addEventListener('click', onMoreClick);

    // Keyboard: Space/Enter reveals, 1–4 rate. Ignore while typing a note.
    document.addEventListener('keydown', (e) => {
        if (cardEl.classList.contains('hidden')) return;
        if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
        if (!revealed && (e.code === 'Space' || e.key === 'Enter')) {
            e.preventDefault();
            revealAnswer();
        } else if (revealed && ['1', '2', '3', '4'].includes(e.key)) {
            rate({ '1': 'again', '2': 'hard', '3': 'good', '4': 'easy' }[e.key]);
        }
    });

    loadCollections();
    loadLearnSettings();
    load();
})();
