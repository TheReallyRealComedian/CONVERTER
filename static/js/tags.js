/* Tag-Manager-Page: list + delete (R1-B-C). */

function tagAlertContainer() { return document.getElementById('tag-alert-container'); }

async function loadTags() {
    const list = document.getElementById('tag-list');
    const empty = document.getElementById('tag-list-empty');
    if (!list) return;
    let resp;
    try {
        resp = await fetch('/api/tags');
    } catch (_) {
        showAlert(tagAlertContainer(), 'danger', 'Tags konnten nicht geladen werden. Verbindung prüfen.');
        return;
    }
    if (!resp.ok) {
        showAlert(tagAlertContainer(), 'danger', 'Tags konnten nicht geladen werden.');
        return;
    }
    const tags = await resp.json();
    list.innerHTML = '';
    if (tags.length === 0) {
        if (empty) empty.classList.remove('hidden');
        return;
    }
    if (empty) empty.classList.add('hidden');
    tags.forEach(tag => {
        const card = document.createElement('div');
        card.className = 'tag-manager-card';
        card.dataset.tagId = String(tag.id);

        const name = document.createElement('span');
        name.className = 'tag-manager-card__name';
        name.textContent = tag.name;
        card.appendChild(name);

        const hCount = tag.highlight_count || 0;
        const cCount = tag.conversion_count || 0;
        const hLabel = hCount === 1 ? '1 Markierung' : `${hCount} Markierungen`;
        const cLabel = cCount === 1 ? '1 Dokument' : `${cCount} Dokumente`;
        const count = document.createElement('span');
        count.className = 'tag-manager-card__count';
        count.textContent = `${hLabel} · ${cLabel}`;
        card.appendChild(count);

        const del = document.createElement('button');
        del.type = 'button';
        del.className = 'tag-manager-card__delete';
        del.textContent = 'Löschen';
        del.addEventListener('click', () => deleteTag(tag.id, tag.name));
        card.appendChild(del);

        list.appendChild(card);
    });
}

async function deleteTag(tagId, tagName) {
    if (!confirm(`Tag „${tagName}" wirklich löschen? Alle Markierungen verlieren diesen Tag.`)) return;
    let resp;
    try {
        resp = await fetch(`/api/tags/${tagId}`, { method: 'DELETE' });
    } catch (_) {
        showToast('Tag löschen fehlgeschlagen. Verbindung prüfen.', { level: 'danger' });
        return;
    }
    if (!resp.ok && resp.status !== 404) {
        showToast('Tag löschen fehlgeschlagen.', { level: 'danger' });
        return;
    }
    showToast('Tag gelöscht.');
    loadTags();
}

document.addEventListener('DOMContentLoaded', loadTags);
