/* Library list view: per-card actions (favorite, copy, delete). */

const FAVORITE_FAILURE_MSG = 'Favorit konnte nicht aktualisiert werden. Verbindung prüfen und erneut versuchen.';
const DELETE_FAILURE_MSG = 'Löschen fehlgeschlagen. Verbindung prüfen und erneut versuchen.';
const DELETE_RACE_MSG = 'Eintrag wurde bereits entfernt.';
const SESSION_EXPIRED_MSG = 'Sitzung abgelaufen. Seite neu laden und erneut anmelden.';

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

window.toggleFavorite = toggleFavorite;
window.copyContent = copyContent;
window.deleteConversion = deleteConversion;
