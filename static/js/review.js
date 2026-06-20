/* Review ("Lernen") UI — walks the due queue from /api/review-state, reveals the
   answer, rates via POST /api/cards/<id>/review (FSRS), lets the user flag
   "wackelt" (Vertiefen) or set a note via POST /api/cards/<id>/annotate, and
   delete the card via DELETE /api/cards/<id>. Every state-changing request goes
   through the global base.html fetch wrapper, which adds X-CSRFToken (DELETE is
   covered too) — a raw fetch without it would 400. User content is rendered via
   textContent / DOM nodes (XSS-safe). */
(function () {
    'use strict';

    const REVIEW_STATE_URL = window.PageData.reviewStateUrl;

    let queue = [];
    let index = 0;
    let totalDue = 0;
    let revealed = false;
    let busy = false;

    const el = (id) => document.getElementById(id);
    const loadingEl = el('review-loading');
    const emptyEl = el('review-empty');
    const doneEl = el('review-done');
    const doneTextEl = el('review-done-text');
    const cardEl = el('review-card');
    const progressEl = el('review-progress');
    const typeBadge = el('review-type-badge');
    const stateBadge = el('review-state-badge');
    const tagsEl = el('review-tags');
    const sourceEl = el('review-source');
    const questionEl = el('review-question');
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

    function renderCard(card) {
        revealed = false;

        typeBadge.textContent = card.type === 'generative' ? 'Generativ' : 'Atomar';
        typeBadge.className = 'type-badge ' + (card.type === 'generative' ? 'type-generative' : 'type-atomic');
        stateBadge.classList.toggle('hidden', card.state !== 'wackelt');

        tagsEl.textContent = '';
        (card.tags || []).forEach((t) => {
            const chip = document.createElement('span');
            chip.className = 'c-tag';
            chip.textContent = t.name;
            tagsEl.appendChild(chip);
        });

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

        show(revealBtn);
        hide(answerWrap);
        hide(ratingEl);
        hide(noteWrap);
        noteToggle.classList.remove('is-active');
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
        revealed = true;
        hide(revealBtn);
        show(answerWrap);
        show(ratingEl);
    }

    function updateProgress() {
        progressEl.textContent = `Karte ${Math.min(index + 1, totalDue)} von ${totalDue} fällig`;
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
    }

    function setRatingDisabled(disabled) {
        ratingEl.querySelectorAll('.review-rate-btn').forEach((b) => { b.disabled = disabled; });
    }

    async function rate(rating) {
        if (busy || !revealed) return;
        busy = true;
        setRatingDisabled(true);
        try {
            const resp = await fetch(`/api/cards/${currentCard().id}/review`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ rating }),
            });
            await safeJSON(resp);
            if (!resp.ok) throw new Error('rate failed');
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

    async function load() {
        show(loadingEl); hide(emptyEl); hide(doneEl); hide(cardEl);
        try {
            const resp = await fetch(REVIEW_STATE_URL);
            const data = await safeJSON(resp);
            if (!resp.ok) throw new Error();
            queue = data.due_cards || [];
            totalDue = (typeof data.due_count === 'number') ? data.due_count : queue.length;
            index = 0;
            hide(loadingEl);
            // Clear any stale "Karte N von M" — matters when a delete empties the
            // queue and re-loads into this branch (finishSession clears it too).
            if (!queue.length) { progressEl.textContent = ''; show(emptyEl); return; }
            renderCard(currentCard());
        } catch (e) {
            hide(loadingEl);
            showAlert(alertContainer(), 'danger',
                'Karten konnten nicht geladen werden. Seite neu laden.');
        }
    }

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
        if (open) noteInput.focus();
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

    load();
})();
