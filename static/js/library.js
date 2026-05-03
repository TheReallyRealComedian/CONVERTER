/* Library list view: per-card actions (favorite, copy, delete). */

function toggleFavorite(id, btn) {
    const isFav = btn.classList.contains('active');
    fetch(`/api/conversions/${id}`, {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({is_favorite: !isFav})
    }).then(r => {
        if (r.ok) {
            btn.classList.toggle('active');
            btn.innerHTML = btn.classList.contains('active') ? '&#9733;' : '&#9734;';
        }
    });
}

function copyContent(id) {
    const card = document.querySelector(`[data-id="${id}"]`);
    const text = card.querySelector('.line-clamp-3').textContent;
    fallbackCopyText(text).then(() => {
        showToast('Content copied to clipboard');
    }).catch(() => {
        showToast('Copy failed');
    });
}

function deleteConversion(id, btn) {
    if (!confirm('Delete this conversion? This cannot be undone.')) return;
    fetch(`/api/conversions/${id}`, {method: 'DELETE'}).then(r => {
        if (r.ok) {
            const card = btn.closest('[data-id]');
            card.style.opacity = '0';
            card.style.transform = 'scale(0.95)';
            card.style.transition = 'opacity 0.2s, transform 0.2s';
            setTimeout(() => card.remove(), 200);
        }
    });
}

window.toggleFavorite = toggleFavorite;
window.copyContent = copyContent;
window.deleteConversion = deleteConversion;
