/* Tag-Manager-Page (R1-B-C + LERN-GROUP P4).
   Tags: indented forest (parent_id), per-tag "Eltern"-dropdown → PATCH
   /api/tags/<id>, server cycle error surfaced as toast, card_count shown.
   Collections: lean CRUD (create/rename/delete + card_count). All writes ride
   the global base.html fetch wrapper (X-CSRFToken); user content via
   textContent / DOM nodes (XSS-safe). */

function tagAlertContainer() { return document.getElementById('tag-alert-container'); }

// --- Tags ---------------------------------------------------------------------

// Pre-order DFS over the parent_id forest → flat [{tag, depth}] list, siblings
// alphabetical (the API already sorts by name; we only group by parent).
function flattenForest(tags) {
    const childrenOf = new Map();
    tags.forEach((t) => {
        const key = t.parent_id == null ? 'root' : t.parent_id;
        if (!childrenOf.has(key)) childrenOf.set(key, []);
        childrenOf.get(key).push(t);
    });
    const out = [];
    const walk = (key, depth) => {
        (childrenOf.get(key) || []).forEach((t) => {
            out.push({ tag: t, depth });
            walk(t.id, depth + 1);
        });
    };
    walk('root', 0);
    return out;
}

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

    flattenForest(tags).forEach(({ tag, depth }) => {
        const card = document.createElement('div');
        card.className = 'tag-manager-card';
        card.dataset.tagId = String(tag.id);
        if (depth > 0) card.style.marginLeft = `${depth * 1.5}rem`;

        const name = document.createElement('span');
        name.className = 'tag-manager-card__name';
        name.textContent = tag.name;
        card.appendChild(name);

        const cardCount = tag.card_count || 0;
        const hCount = tag.highlight_count || 0;
        const cCount = tag.conversion_count || 0;
        const parts = [
            cardCount === 1 ? '1 Karte' : `${cardCount} Karten`,
            hCount === 1 ? '1 Markierung' : `${hCount} Markierungen`,
            cCount === 1 ? '1 Dokument' : `${cCount} Dokumente`,
        ];
        const count = document.createElement('span');
        count.className = 'tag-manager-card__count';
        count.textContent = parts.join(' · ');
        card.appendChild(count);

        // "Eltern"-dropdown: "— (Wurzel)" + every OTHER tag. Self is excluded;
        // subtree-cycles are left for the server to reject (→ toast), so the
        // cycle-guard path stays exercisable.
        const parentSel = document.createElement('select');
        parentSel.className = 'c-input tag-manager-card__parent';
        parentSel.setAttribute('aria-label', `Eltern-Tag für ${tag.name}`);
        const rootOpt = document.createElement('option');
        rootOpt.value = '';
        rootOpt.textContent = '— (Wurzel)';
        parentSel.appendChild(rootOpt);
        tags.forEach((other) => {
            if (other.id === tag.id) return;
            const opt = document.createElement('option');
            opt.value = String(other.id);
            opt.textContent = other.name;
            if (tag.parent_id === other.id) opt.selected = true;
            parentSel.appendChild(opt);
        });
        parentSel.addEventListener('change', () => setTagParent(tag.id, parentSel.value));
        card.appendChild(parentSel);

        const del = document.createElement('button');
        del.type = 'button';
        del.className = 'tag-manager-card__delete';
        del.textContent = 'Löschen';
        del.addEventListener('click', () => deleteTag(tag.id, tag.name));
        card.appendChild(del);

        list.appendChild(card);
    });
}

async function setTagParent(tagId, parentValue) {
    const parent_id = parentValue === '' ? null : Number(parentValue);
    let resp;
    try {
        resp = await fetch(`/api/tags/${tagId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ parent_id }),
        });
    } catch (_) {
        showToast('Zuordnung fehlgeschlagen. Verbindung prüfen.', { level: 'danger' });
        return;
    }
    if (resp.status === 400) {
        // Cycle / foreign parent — surface the server message, reset the view.
        showToast('Das ergibt eine Schleife — Tag kann nicht unter seinen eigenen Ast.', { level: 'danger' });
        loadTags();
        return;
    }
    if (!resp.ok) {
        showToast('Zuordnung fehlgeschlagen.', { level: 'danger' });
        loadTags();
        return;
    }
    showToast('Tag zugeordnet.');
    loadTags();
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

// --- Collections --------------------------------------------------------------

async function loadCollections() {
    const list = document.getElementById('collection-list');
    const empty = document.getElementById('collection-list-empty');
    if (!list) return;
    let resp;
    try {
        resp = await fetch('/api/collections');
    } catch (_) {
        showToast('Sammlungen konnten nicht geladen werden.', { level: 'danger' });
        return;
    }
    if (!resp.ok) {
        showToast('Sammlungen konnten nicht geladen werden.', { level: 'danger' });
        return;
    }
    const collections = await resp.json();
    list.innerHTML = '';
    if (collections.length === 0) {
        if (empty) empty.classList.remove('hidden');
        return;
    }
    if (empty) empty.classList.add('hidden');

    collections.forEach((col) => {
        const card = document.createElement('div');
        card.className = 'tag-manager-card';
        card.dataset.collectionId = String(col.id);

        const name = document.createElement('span');
        name.className = 'tag-manager-card__name';
        name.textContent = col.name;
        card.appendChild(name);

        const n = col.card_count || 0;
        const count = document.createElement('span');
        count.className = 'tag-manager-card__count';
        count.textContent = n === 1 ? '1 Karte' : `${n} Karten`;
        card.appendChild(count);

        const rename = document.createElement('button');
        rename.type = 'button';
        rename.className = 'tag-manager-card__action';
        rename.textContent = 'Umbenennen';
        rename.addEventListener('click', () => renameCollection(col.id, col.name));
        card.appendChild(rename);

        const del = document.createElement('button');
        del.type = 'button';
        del.className = 'tag-manager-card__delete';
        del.textContent = 'Löschen';
        del.addEventListener('click', () => deleteCollection(col.id, col.name));
        card.appendChild(del);

        list.appendChild(card);
    });
}

async function createCollection() {
    const input = document.getElementById('collection-new-name');
    const name = (input.value || '').trim();
    if (!name) { showToast('Bitte einen Namen eingeben.', { level: 'danger' }); return; }
    let resp;
    try {
        resp = await fetch('/api/collections', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name }),
        });
    } catch (_) {
        showToast('Anlegen fehlgeschlagen. Verbindung prüfen.', { level: 'danger' });
        return;
    }
    if (resp.status === 409) { showToast('Sammlung existiert bereits.', { level: 'danger' }); return; }
    if (!resp.ok) { showToast('Anlegen fehlgeschlagen.', { level: 'danger' }); return; }
    input.value = '';
    showToast('Sammlung angelegt.');
    loadCollections();
}

async function renameCollection(collectionId, currentName) {
    const next = prompt('Sammlung umbenennen:', currentName);
    if (next === null) return;
    const name = next.trim();
    if (!name) { showToast('Name darf nicht leer sein.', { level: 'danger' }); return; }
    if (name === currentName) return;
    let resp;
    try {
        resp = await fetch(`/api/collections/${collectionId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name }),
        });
    } catch (_) {
        showToast('Umbenennen fehlgeschlagen. Verbindung prüfen.', { level: 'danger' });
        return;
    }
    if (resp.status === 409) { showToast('Sammlung existiert bereits.', { level: 'danger' }); return; }
    if (!resp.ok) { showToast('Umbenennen fehlgeschlagen.', { level: 'danger' }); return; }
    showToast('Umbenannt.');
    loadCollections();
}

async function deleteCollection(collectionId, name) {
    if (!confirm(`Sammlung „${name}" wirklich löschen? Die Karten bleiben erhalten.`)) return;
    let resp;
    try {
        resp = await fetch(`/api/collections/${collectionId}`, { method: 'DELETE' });
    } catch (_) {
        showToast('Löschen fehlgeschlagen. Verbindung prüfen.', { level: 'danger' });
        return;
    }
    if (!resp.ok && resp.status !== 404) { showToast('Löschen fehlgeschlagen.', { level: 'danger' }); return; }
    showToast('Sammlung gelöscht.');
    loadCollections();
}

document.addEventListener('DOMContentLoaded', () => {
    loadTags();
    loadCollections();
    const createBtn = document.getElementById('collection-create-btn');
    const newName = document.getElementById('collection-new-name');
    if (createBtn) createBtn.addEventListener('click', createCollection);
    if (newName) newName.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); createCollection(); }
    });
});
