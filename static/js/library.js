/* Library list view: per-card actions (favorite, copy, delete). */

const FAVORITE_FAILURE_MSG = 'Favorit konnte nicht aktualisiert werden. Verbindung prüfen und erneut versuchen.';
const DELETE_FAILURE_MSG = 'Löschen fehlgeschlagen. Verbindung prüfen und erneut versuchen.';
const DELETE_RACE_MSG = 'Eintrag wurde bereits entfernt.';
const STATUS_FAILURE_MSG = 'Status konnte nicht geändert werden. Verbindung prüfen und erneut versuchen.';
const STATUS_LABELS = { inbox: 'Inbox', later: 'Später', archive: 'Archiv' };
const SESSION_EXPIRED_MSG = 'Sitzung abgelaufen. Seite neu laden und erneut anmelden.';
const QUEUE_FAILURE_MSG = 'Lese-Liste konnte nicht aktualisiert werden. Verbindung prüfen und erneut versuchen.';
const REORDER_FAILURE_MSG = 'Reihenfolge konnte nicht geändert werden. Verbindung prüfen und erneut versuchen.';

function libraryAlertContainer() { return document.getElementById('library-alert-container'); }
function libraryActionStatus() { return document.getElementById('library-action-status'); }

function withServerSuffix(msg, status) {
    if (status >= 500) return msg + ' Server-Fehler — bitte später erneut versuchen.';
    return msg;
}

function handleFetchError(error, fallbackMsg) {
    const msg = (error && /Session expired/i.test(error.message)) ? SESSION_EXPIRED_MSG : fallbackMsg;
    showAlert(libraryAlertContainer(), 'danger', msg);
}

function toggleFavorite(id, btn) {
    const isFav = btn.classList.contains('active');
    fetch(`/api/conversions/${id}`, {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({is_favorite: !isFav})
    }).then(r => {
        if (r.ok) {
            const container = libraryAlertContainer();
            if (container) container.innerHTML = '';
            btn.classList.toggle('active');
            btn.innerHTML = btn.classList.contains('active') ? '&#9733;' : '&#9734;';
            return null;
        }
        return safeJSON(r).catch(() => null).then(() => {
            showAlert(libraryAlertContainer(), 'danger', withServerSuffix(FAVORITE_FAILURE_MSG, r.status));
        });
    }).catch(err => handleFetchError(err, FAVORITE_FAILURE_MSG));
}

// R2-C: lifecycle triage toggle. PUTs lifecycle_status like toggleFavorite;
// success is silent (the badge + active segment ARE the feedback), errors use
// showToast. The card stays in place even under an active ?status filter — the
// reload after triage reflects the filtered set.
function setStatus(id, status) {
    const card = document.querySelector(`[data-id="${id}"]`);
    fetch(`/api/conversions/${id}`, {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({lifecycle_status: status})
    }).then(r => {
        if (r.ok) {
            const container = libraryAlertContainer();
            if (container) container.innerHTML = '';
            // Reload when the card leaves the view's visible set, so derived
            // display state (#rank badges, edge-disabled arrows, the inbox
            // tab badge, pagination) never goes stale: in the queue-view an
            // archived item drops off the to-read list (R2-D); in the
            // inbox-view (R2-E) any triage away from 'inbox' removes it.
            const view = window.PageData ? window.PageData.currentView : '';
            const leavesVisibleSet =
                (view === 'queue' && status === 'archive') ||
                (view === 'inbox' && status !== 'inbox');
            if (card && leavesVisibleSet) {
                window.location.reload();
                return null;
            }
            if (card) applyStatusToCard(card, status);
            return null;
        }
        return safeJSON(r).catch(() => null).then(() => {
            showToast(withServerSuffix(STATUS_FAILURE_MSG, r.status), { level: 'danger' });
        });
    }).catch(err => {
        const msg = (err && /Session expired/i.test(err.message)) ? SESSION_EXPIRED_MSG : STATUS_FAILURE_MSG;
        showToast(msg, { level: 'danger' });
    });
}

function applyStatusToCard(card, status) {
    const badge = card.querySelector('[data-status-badge]');
    if (badge) {
        badge.classList.remove('status-badge--inbox', 'status-badge--later', 'status-badge--archive');
        badge.classList.add(`status-badge--${status}`);
        badge.textContent = STATUS_LABELS[status] || status;
    }
    card.querySelectorAll('[data-status-control] .status-segmented__btn').forEach(btn => {
        const active = btn.dataset.status === status;
        btn.classList.toggle('is-active', active);
        btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
}

function copyContent(id) {
    const card = document.querySelector(`[data-id="${id}"]`);
    const text = card.dataset.content || '';
    fallbackCopyText(text).then(() => {
        showToast('Inhalt kopiert');
    }).catch(() => {
        showToast('Kopieren fehlgeschlagen', { level: 'danger' });
    });
}

function announceCardRemoval(msg) {
    const status = libraryActionStatus();
    if (!status) return;
    status.textContent = '';
    // Reset before re-setting so screen readers re-announce identical messages.
    setTimeout(() => { status.textContent = msg; }, 50);
}

function removeCard(card) {
    card.style.opacity = '0';
    card.style.transform = 'scale(0.95)';
    card.style.transition = 'opacity 0.2s, transform 0.2s';
    setTimeout(() => card.remove(), 200);
}

function deleteConversion(id, btn) {
    if (!confirm('Diesen Eintrag wirklich löschen? Das kann nicht rückgängig gemacht werden.')) return;
    fetch(`/api/conversions/${id}`, {method: 'DELETE'}).then(r => {
        if (r.ok) {
            const container = libraryAlertContainer();
            if (container) container.innerHTML = '';
            const card = btn.closest('[data-id]');
            announceCardRemoval('Eintrag gelöscht.');
            removeCard(card);
            return null;
        }
        if (r.status === 404) {
            const card = btn.closest('[data-id]');
            showAlert(libraryAlertContainer(), 'info', DELETE_RACE_MSG, { autoDismissMs: 4000 });
            announceCardRemoval('Eintrag bereits entfernt.');
            removeCard(card);
            return null;
        }
        return safeJSON(r).catch(() => null).then(() => {
            showAlert(libraryAlertContainer(), 'danger', withServerSuffix(DELETE_FAILURE_MSG, r.status));
        });
    }).catch(err => handleFetchError(err, DELETE_FAILURE_MSG));
}

// R2-D: reading-list toggle. POSTs add/remove to the queue endpoint. The flag
// icon (filled/outline) IS the feedback, so success is silent like the favorite
// star. In the queue-view a just-removed item no longer belongs to the visible
// set, so its card is pulled from the DOM instead of just flipping the icon.
function toggleQueue(id, btn) {
    const isQueued = btn.classList.contains('active');
    const action = isQueued ? 'remove' : 'add';
    fetch(`/api/conversions/${id}/queue`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ action })
    }).then(r => {
        if (r.ok) {
            const container = libraryAlertContainer();
            if (container) container.innerHTML = '';
            // Reload when the toggle moves the card across the view's visible
            // sets: in the queue-view BOTH directions move it (remove drops it
            // off the list, add promotes a Weiterlesen-section card into the
            // ranked queue, R2-E); in the inbox-view add = triage, the card
            // leaves the untriaged pile and the tab badge changes.
            const view = window.PageData ? window.PageData.currentView : '';
            if (view === 'queue' || (view === 'inbox' && action === 'add')) {
                window.location.reload();
                return null;
            }
            applyQueueToButton(btn, !isQueued);
            return null;
        }
        return safeJSON(r).catch(() => null).then(() => {
            showToast(withServerSuffix(QUEUE_FAILURE_MSG, r.status), { level: 'danger' });
        });
    }).catch(err => {
        const msg = (err && /Session expired/i.test(err.message)) ? SESSION_EXPIRED_MSG : QUEUE_FAILURE_MSG;
        showToast(msg, { level: 'danger' });
    });
}

function applyQueueToButton(btn, queued) {
    btn.classList.toggle('active', queued);
    btn.setAttribute('aria-pressed', queued ? 'true' : 'false');
    btn.innerHTML = queued ? '&#9873;' : '&#9872;';
    btn.title = queued ? 'Von der Lese-Liste nehmen' : 'Auf die Lese-Liste setzen';
}

// R2-D: reorder a queued item by one slot (swap server-side). The list is small
// and the swap is atomic, so a plain reload of the queue-view is the simplest
// robust reflection of the new order (sprint: DOM-swap is optional polish).
function moveQueue(id, dir) {
    fetch(`/api/conversions/${id}/queue`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ action: dir })
    }).then(r => {
        if (r.ok) {
            window.location.reload();
            return null;
        }
        return safeJSON(r).catch(() => null).then(() => {
            showToast(withServerSuffix(REORDER_FAILURE_MSG, r.status), { level: 'danger' });
        });
    }).catch(err => {
        const msg = (err && /Session expired/i.test(err.message)) ? SESSION_EXPIRED_MSG : REORDER_FAILURE_MSG;
        showToast(msg, { level: 'danger' });
    });
}

// R2-E Bibliothek tag row: "+N weitere" expand/collapse (no persistence) and
// typeahead auto-submit. Typing alone never navigates — only an exact
// datalist match (mouse pick or typed-out name) submits; Enter submits the
// GET form natively either way. The form's hidden inputs replicate the
// pagination_args semantics of a chip click (page resets, filters survive).
function initTagRow() {
    const toggle = document.querySelector('[data-tag-toggle]');
    if (toggle) {
        const collapsedRow = document.getElementById('tag-row-collapsed');
        const expandedRow = document.getElementById('tag-row-expanded');
        toggle.addEventListener('click', () => {
            const expand = expandedRow.hidden;
            expandedRow.hidden = !expand;
            collapsedRow.hidden = expand;
            toggle.textContent = expand ? 'Weniger anzeigen' : toggle.dataset.expandLabel;
            toggle.setAttribute('aria-expanded', expand ? 'true' : 'false');
        });
    }
    const input = document.getElementById('tag-typeahead-input');
    if (input) {
        const names = new Set(
            Array.from(document.querySelectorAll('#tag-name-list option')).map(o => o.value)
        );
        input.addEventListener('input', () => {
            if (names.has(input.value.trim().toLowerCase())) input.form.submit();
        });
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTagRow);
} else {
    initTagRow();
}

window.toggleFavorite = toggleFavorite;
window.setStatus = setStatus;
window.copyContent = copyContent;
window.deleteConversion = deleteConversion;
window.toggleQueue = toggleQueue;
window.moveQueue = moveQueue;
