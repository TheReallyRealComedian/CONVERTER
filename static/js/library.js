/* Library list view: per-card actions (place move, reorder, copy, delete). */

const DELETE_FAILURE_MSG = 'Löschen fehlgeschlagen. Verbindung prüfen und erneut versuchen.';
const DELETE_RACE_MSG = 'Eintrag wurde bereits entfernt.';
const PLACE_FAILURE_MSG = 'Ablage konnte nicht geändert werden. Verbindung prüfen und erneut versuchen.';
const SESSION_EXPIRED_MSG = 'Sitzung abgelaufen. Seite neu laden und erneut anmelden.';
const REORDER_FAILURE_MSG = 'Reihenfolge konnte nicht geändert werden. Verbindung prüfen und erneut versuchen.';
const KINDLE_FAILURE_MSG = 'Versand an Kindle fehlgeschlagen. Verbindung prüfen und erneut versuchen.';

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

// R2-H: the one flat move-action. POSTs the target place; the pressed segment IS
// the feedback, so success is silent. Reload-gate (Memory
// reference_reorder_over_filtered_set): when the item leaves the visible set its
// derived display state (#rank, edge-disabled arrows, inbox badge, pagination)
// goes stale, so reload; otherwise flip the pressed segment in-place. The item
// leaves the set on a move to a different place when browsing a place; in the
// global finder (search/tag/type spans non-archive) only a move to Archiv drops
// it off. Clicking the already-active place is a no-op.
function setPlace(id, place, btn) {
    if (btn.classList.contains('is-active')) return;
    fetch(`/api/conversions/${id}/place`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ place })
    }).then(r => {
        if (r.ok) {
            const container = libraryAlertContainer();
            if (container) container.innerHTML = '';
            const view = window.PageData ? window.PageData.currentView : '';
            const finder = window.PageData ? window.PageData.finder : false;
            const leavesVisibleSet = finder ? (place === 'archiv') : (place !== view);
            const card = document.querySelector(`[data-id="${id}"]`);
            if (card && leavesVisibleSet) {
                window.location.reload();
                return null;
            }
            if (card) applyPlaceToCard(card, place);
            return null;
        }
        return safeJSON(r).catch(() => null).then(() => {
            showToast(withServerSuffix(PLACE_FAILURE_MSG, r.status), { level: 'danger' });
        });
    }).catch(err => {
        const msg = (err && /Session expired/i.test(err.message)) ? SESSION_EXPIRED_MSG : PLACE_FAILURE_MSG;
        showToast(msg, { level: 'danger' });
    });
}

function applyPlaceToCard(card, place) {
    card.querySelectorAll('[data-place-control] .place-control__btn').forEach(btn => {
        const active = btn.dataset.place === place;
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

// Send this conversion to the Kindle (EPUB via Send-to-Kindle email). Non-
// destructive, so no confirm. The button disables for the in-flight request
// (double-click guard). On error we surface the server's message verbatim
// (503 → „Kindle nicht konfiguriert.", 502 → „Versand an Kindle fehlgeschlagen.").
function sendToKindle(id, btn) {
    if (btn) btn.disabled = true;
    fetch(`/api/conversions/${id}/send-to-kindle`, { method: 'POST' }).then(r => {
        if (r.ok) {
            showToast('An Kindle gesendet');
            return null;
        }
        return safeJSON(r).then(data => {
            showToast((data && data.error) || KINDLE_FAILURE_MSG, { level: 'danger' });
        });
    }).catch(err => {
        const msg = (err && /Session expired/i.test(err.message)) ? SESSION_EXPIRED_MSG : KINDLE_FAILURE_MSG;
        showToast(msg, { level: 'danger' });
    }).finally(() => {
        if (btn) btn.disabled = false;
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

// R2-D: reorder a queued item by one slot (swap server-side). The list is small
// and the swap is atomic, so a plain reload of the Lese-Liste view is the
// simplest robust reflection of the new order.
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
        // 'change', not 'input': a datalist mouse-pick fires change right
        // away, typing only on blur — so typing "spacex ipo" is never
        // hijacked at the "spacex" prefix (real Mintbox tag pair). Enter
        // submits the GET form natively either way.
        input.addEventListener('change', () => {
            if (names.has(input.value.trim().toLowerCase())) input.form.submit();
        });
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTagRow);
} else {
    initTagRow();
}

window.setPlace = setPlace;
window.copyContent = copyContent;
window.sendToKindle = sendToKindle;
window.deleteConversion = deleteConversion;
window.moveQueue = moveQueue;
