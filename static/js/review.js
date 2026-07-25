/* Review ("Lernen") UI — walks the due queue from /api/review-state, reveals the
   answer, rates via POST /api/cards/<id>/review (FSRS), lets the user flag
   "wackelt" (Vertiefen) or set a note via POST /api/cards/<id>/annotate, and
   delete the card via DELETE /api/cards/<id>. Every state-changing request goes
   through the global base.html fetch wrapper, which adds X-CSRFToken (DELETE is
   covered too) — a raw fetch without it would 400. Card TEXT is rendered
   exclusively via textContent / DOM nodes (XSS-safe). The ONE exception
   (CARD-SVG): the two figure containers get innerHTML — their SVG arrives
   SERVER-sanitized (services/svg_sanitize.py, applied in Card.to_dict), never
   raw agent input. Keep every other card field on textContent. */
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

    const alertContainer = () => el('review-alert-container');
    const currentCard = () => queue[index];
    const show = (elm) => elm.classList.remove('hidden');
    const hide = (elm) => elm.classList.add('hidden');

    // --- cloze rendering: {{answer}} → a blank box (front) or the highlighted
    //     answer (back). Built as DOM nodes so the card text can't inject HTML. ---
    const CLOZE_RE = /\{\{(.+?)\}\}/g;

    function renderCloze(target, text, reveal) {
        target.textContent = '';
        let last = 0, m;
        CLOZE_RE.lastIndex = 0;
        while ((m = CLOZE_RE.exec(text)) !== null) {
            if (m.index > last) {
                target.appendChild(document.createTextNode(text.slice(last, m.index)));
            }
            const span = document.createElement('span');
            span.className = reveal ? 'review-cloze-fill' : 'review-cloze-blank';
            span.textContent = reveal ? m[1] : '…';
            target.appendChild(span);
            last = m.index + m[0].length;
        }
        if (last < text.length) {
            target.appendChild(document.createTextNode(text.slice(last)));
        }
    }

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
            questionEl.textContent = card.prompt || '';
            show(genHintEl);
            answerLabel.textContent = 'Musterantwort';
        } else if (isCloze(card)) {
            renderCloze(questionEl, card.cloze_text, false);
            hide(genHintEl);
            answerLabel.textContent = 'Lösung';
        } else {
            questionEl.textContent = card.front || '';
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
            answerEl.textContent = card.back || '(keine Musterantwort hinterlegt)';
        } else if (isCloze(card)) {
            renderCloze(answerEl, card.cloze_text, true);
        } else {
            answerEl.textContent = card.back || '';
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
        (card.collections || []).forEach((col) => {
            const cached = collections.find((c) => c.id === col.id);
            if (cached) cached.due_count = Math.max(0, (cached.due_count || 0) - 1);
        });
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
        progressEl.textContent = '';
        show(doneEl);
        // Authoritative resync (LEARN-COUNT safeguard): server truth replaces
        // the local decrementPoolCounts bookkeeping — should it ever drift
        // during a session, it self-heals here. Do not remove.
        loadCollections();
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
            if (!resp.ok) throw new Error('rate failed');
            // Load-bearing (LEARN-COUNT): the response is the updated card,
            // incl. the NEW review.due. A card leaves the due pool only when
            // that due lies in the future — on "Nochmal" FSRS reschedules
            // minutes ahead, the card is STILL due and nothing may be
            // decremented. A blind -1 per rating would run the counters away
            // from reality. Missing due in the response → treat as still due
            // (no decrement; the finishSession resync heals any gap).
            const stillDue = !updated || !updated.review || !updated.review.due
                || new Date(updated.review.due) <= new Date();
            if (!stillDue) decrementPoolCounts(card);
            advance();
        } catch (e) {
            showAlert(alertContainer(), 'danger',
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
            await safeJSON(resp);
            if (!resp.ok) throw new Error();
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
            showToast('Karte konnte nicht gelöscht werden. Erneut versuchen.', { level: 'danger' });
        } finally {
            busy = false;
        }
    }

    function scopeUrl() {
        if (!scopeIds.length) return REVIEW_STATE_URL;
        return `${REVIEW_STATE_URL}?collection=${scopeIds.join(',')}`;
    }

    function scopeLabel() {
        return collections
            .filter((c) => scopeIds.includes(c.id))
            .map((c) => c.name)
            .join(', ');
    }

    function applyEmptyScope() {
        // Scope-aware empty state — "Nichts fällig in <Auswahl>" vs. the global text.
        if (!scopeIds.length) {
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
            hide(loadingEl);
            // Clear any stale "Karte N von M" — matters when a delete empties the
            // queue and re-loads into this branch (finishSession clears it too).
            if (!queue.length) { progressEl.textContent = ''; applyEmptyScope(); show(emptyEl); return; }
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
    function renderScopePills() {
        // Rebuild the collection pills, preserving the current checks.
        scopeList.querySelectorAll('.review-scope-pill[data-col]').forEach((p) => p.remove());
        (collections || []).forEach((c) => {
            const pill = document.createElement('label');
            pill.className = 'review-scope-pill';
            pill.dataset.col = String(c.id);
            const box = document.createElement('input');
            box.type = 'checkbox';
            box.dataset.colId = String(c.id);
            box.checked = scopeIds.includes(c.id);
            const name = document.createElement('span');
            name.textContent = c.name;
            const badge = document.createElement('span');
            badge.className = 'review-scope-pill__badge';
            badge.textContent = String(c.due_count || 0);
            badge.title = 'Fällige Karten';
            pill.append(box, name, badge);
            scopeList.appendChild(pill);
        });
        scopeAllBox.checked = scopeIds.length === 0;
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
            load();
        } catch (e) {
            showToast(errorText, { level: 'danger' });
            loadLearnSettings();  // revert the controls to server truth
        }
    }

    function onOrderChange() {
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
        } else if (e.target.dataset && e.target.dataset.colId) {
            const id = Number(e.target.dataset.colId);
            scopeIds = e.target.checked
                ? scopeIds.concat(scopeIds.includes(id) ? [] : [id])
                : scopeIds.filter((x) => x !== id);
        } else {
            return;
        }
        renderScopePills();
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
            const colName = (collections.find((c) => c.id === collectionId) || {}).name || 'Sammlung';
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
