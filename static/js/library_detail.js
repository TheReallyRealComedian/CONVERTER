/* Library detail view: inline editing, copy/download/delete, Notion send. */

const CONVERSION_ID = window.PageData.conversionId;
const LIBRARY_URL = window.PageData.libraryUrl;
const DOWNLOAD_FILENAME = window.PageData.downloadFilename;
const DEFAULT_TARGET = window.PageData.defaultNotionTarget;

const SAVE_MESSAGES = {
    title: {
        success: 'Titel gespeichert',
        failure: 'Titel konnte nicht gespeichert werden. Verbindung prüfen und erneut versuchen.',
    },
    tags: {
        success: 'Tags gespeichert',
        failure: 'Tags konnten nicht gespeichert werden. Verbindung prüfen und erneut versuchen.',
    },
};

// Inputs that participate in the dirty-indicator + flush-on-hide flow.
const AUTOSAVE_INPUTS = { title: 'detail-title', tags: 'tags-input' };
const DIRTY_TOOLTIP = 'Ungespeicherte Änderung — Tab oder Klick außerhalb speichert.';

const NOTION_TARGET_LABELS = { meetings: 'Meeting', notes: 'Notiz', inbox: 'Inbox' };

function detailAlertContainer() { return document.getElementById('detail-alert-container'); }
function notionAlertContainer() { return document.getElementById('notion-alert-container'); }

function clearDetailAlert() {
    const c = detailAlertContainer();
    if (c) c.innerHTML = '';
}

function clearNotionAlert() {
    const c = notionAlertContainer();
    if (c) c.innerHTML = '';
}

function withServerSuffix(msg, status) {
    if (status >= 500) return msg + ' Server-Fehler — bitte später erneut versuchen.';
    return msg;
}

function updateField(field, value) {
    const body = {};
    body[field] = value;
    const messages = SAVE_MESSAGES[field] || {
        success: `${field} gespeichert`,
        failure: `${field} konnte nicht gespeichert werden. Verbindung prüfen und erneut versuchen.`,
    };
    fetch(`/api/conversions/${CONVERSION_ID}`, {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
    }).then(r => {
        if (r.ok) {
            clearDetailAlert();
            showToast(messages.success);
            clearDirtyState(field);
            if (field === 'title') updatePageTitle(value);
        } else {
            showAlert(detailAlertContainer(), 'danger', withServerSuffix(messages.failure, r.status));
        }
    }).catch(() => {
        showAlert(detailAlertContainer(), 'danger', messages.failure);
    });
}

function clearDirtyState(field) {
    const id = AUTOSAVE_INPUTS[field];
    if (!id) return;
    const input = document.getElementById(id);
    if (input) {
        input.classList.remove('c-input--dirty');
        if (input.dataset.titleBeforeDirty !== undefined) {
            input.title = input.dataset.titleBeforeDirty;
            delete input.dataset.titleBeforeDirty;
        }
    }
}

function updatePageTitle(value) {
    const cleanTitle = (value == null ? '' : String(value)).trim();
    document.title = `${cleanTitle || 'Ohne Titel'} – Library`;
}

function toggleFavorite(btn) {
    const isFav = btn.classList.contains('active');
    fetch(`/api/conversions/${CONVERSION_ID}`, {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({is_favorite: !isFav})
    }).then(r => {
        if (r.ok) {
            clearDetailAlert();
            btn.classList.toggle('active');
            btn.innerHTML = btn.classList.contains('active') ? '&#9733;' : '&#9734;';
        } else {
            showAlert(detailAlertContainer(), 'danger',
                withServerSuffix('Favorit konnte nicht aktualisiert werden. Verbindung prüfen und erneut versuchen.', r.status));
        }
    }).catch(() => {
        showAlert(detailAlertContainer(), 'danger',
            'Favorit konnte nicht aktualisiert werden. Verbindung prüfen und erneut versuchen.');
    });
}

function copyFullContent() {
    const text = document.querySelector('.detail-content-text').textContent;
    fallbackCopyText(text).then(() => showToast('Kopiert'))
        .catch(() => showToast('Kopieren fehlgeschlagen', { level: 'danger' }));
}

function downloadContent() {
    const text = document.querySelector('.detail-content-text').textContent;
    const blob = new Blob([text], {type: 'text/plain'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = DOWNLOAD_FILENAME;
    a.click();
    URL.revokeObjectURL(a.href);
}

function storeForReuse() {
    const text = document.querySelector('.detail-content-text').textContent;
    localStorage.setItem('libraryReuse', text);
}

function deleteConversion() {
    if (!confirm('Diesen Eintrag wirklich löschen? Das kann nicht rückgängig gemacht werden.')) return;
    const btn = document.getElementById('delete-btn');
    const originalText = btn ? btn.textContent : '';
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Lösche …';
    }
    const restoreBtn = () => {
        if (btn) {
            btn.disabled = false;
            btn.textContent = originalText;
        }
    };
    fetch(`/api/conversions/${CONVERSION_ID}`, {method: 'DELETE'}).then(r => {
        if (r.ok) {
            window.location.href = LIBRARY_URL;
        } else if (r.status === 404) {
            // Race: row already gone — inform briefly, then navigate.
            showAlert(detailAlertContainer(), 'info',
                'Eintrag wurde bereits entfernt. Du wirst zur Library zurückgeleitet.',
                { autoDismissMs: 3000 });
            setTimeout(() => { window.location.href = LIBRARY_URL; }, 1500);
        } else {
            restoreBtn();
            showAlert(detailAlertContainer(), 'danger',
                r.status >= 500
                    ? 'Löschen fehlgeschlagen — Server-Fehler. Bitte später erneut versuchen.'
                    : 'Löschen fehlgeschlagen. Verbindung prüfen und erneut versuchen.');
        }
    }).catch(() => {
        restoreBtn();
        showAlert(detailAlertContainer(), 'danger',
            'Löschen fehlgeschlagen. Verbindung prüfen und erneut versuchen.');
    });
}

// --- Notion Integration ---
let currentTarget = DEFAULT_TARGET;
let notionSuggestions = null;

// Body-pool keys: cross-target text fields that share a slot. When the user
// switches Meeting -> Inbox, what was in `summary` lands in `description`,
// and vice versa, instead of being silently wiped.
const NOTION_BODY_POOL = ['summary', 'description', 'note', 'text'];

function collectNotionFieldValues(container) {
    const snapshot = {};
    if (!container) return snapshot;
    container.querySelectorAll('input, textarea').forEach(el => {
        const key = el.id.replace('nf-', '');
        snapshot[key] = el.value;
    });
    return snapshot;
}

function restoreNotionFieldValues(container, snapshot) {
    if (!container || !snapshot) return;
    container.querySelectorAll('input, textarea').forEach(el => {
        const key = el.id.replace('nf-', '');
        if (Object.prototype.hasOwnProperty.call(snapshot, key)) {
            el.value = snapshot[key];
            return;
        }
        if (NOTION_BODY_POOL.includes(key)) {
            for (const k of NOTION_BODY_POOL) {
                if (snapshot[k]) {
                    el.value = snapshot[k];
                    return;
                }
            }
        }
    });
}

function toggleNotionPanel() {
    const panel = document.getElementById('notion-panel');
    const icon = document.getElementById('notion-toggle-icon');
    const toggleBtn = document.getElementById('notion-toggle-btn');
    const isHidden = panel.classList.toggle('hidden');
    icon.innerHTML = isHidden ? '&#9662;' : '&#9652;';
    if (toggleBtn) toggleBtn.setAttribute('aria-expanded', String(!isHidden));
    if (!isHidden) {
        const container = document.getElementById('notion-fields');
        // Only render on initial open; re-toggle preserves user inputs.
        if (!container || !container.children.length) {
            selectTarget(DEFAULT_TARGET);
        }
        loadSuggestions();
    }
}

function loadSuggestions() {
    if (notionSuggestions) return;
    const fallback = {people: [], projects: [], meeting_types: [], note_types: []};
    fetch('/api/notion/suggestions').then(async r => {
        if (!r.ok) {
            notionSuggestions = fallback;
        } else {
            try {
                notionSuggestions = await safeJSON(r);
            } catch (_) {
                notionSuggestions = fallback;
            }
        }
        // Re-render so datalists populate, but preserve any values the
        // user typed before the suggestions arrived.
        const container = document.getElementById('notion-fields');
        if (container && container.children.length) {
            const snapshot = collectNotionFieldValues(container);
            renderNotionFields(currentTarget);
            restoreNotionFieldValues(container, snapshot);
        }
    }).catch(() => {
        notionSuggestions = fallback;
    });
}

function selectTarget(target) {
    const container = document.getElementById('notion-fields');
    const isInitial = !container || !container.children.length;
    const isSwitch = !isInitial && currentTarget !== target;
    const snapshot = isSwitch ? collectNotionFieldValues(container) : null;
    currentTarget = target;
    document.querySelectorAll('#notion-target-group button').forEach(btn => {
        btn.classList.toggle('c-btn--primary', btn.dataset.target === target);
    });
    renderNotionFields(target);
    if (snapshot) {
        restoreNotionFieldValues(document.getElementById('notion-fields'), snapshot);
    }
    if (isSwitch) {
        const status = document.getElementById('notion-target-status');
        if (status) {
            const label = NOTION_TARGET_LABELS[target] || target;
            status.textContent = `Ziel gewechselt zu ${label} — passende Felder übernommen.`;
        }
    }
}

function renderNotionFields(target) {
    const title = document.getElementById('detail-title').value;
    const tags = document.getElementById('tags-input').value;
    const now = formatDatetimeLocalNow();
    const s = notionSuggestions || {people: [], projects: [], meeting_types: [], note_types: []};

    const fieldDefs = {
        meetings: [
            {key: 'title', label: 'Titel', value: title, required: true},
            {key: 'datum', label: 'Datum', value: now, type: 'datetime-local'},
            {key: 'project', label: 'Projekt', value: '', list: s.projects},
            {key: 'people', label: 'Personen', value: '', placeholder: 'kommagetrennt', list: s.people},
            {key: 'type', label: 'Typ', value: '', list: s.meeting_types},
            {key: 'summary', label: 'Zusammenfassung', value: '', type: 'textarea'},
        ],
        notes: [
            {key: 'title', label: 'Titel', value: title, required: true},
            {key: 'project', label: 'Projekt', value: '', list: s.projects},
            {key: 'type', label: 'Typ', value: '', list: s.note_types},
            {key: 'tags', label: 'Tags', value: tags, placeholder: 'kommagetrennt'},
            {key: 'people', label: 'Personen', value: '', placeholder: 'kommagetrennt', list: s.people},
            {key: 'summary', label: 'Zusammenfassung', value: '', type: 'textarea'},
        ],
        inbox: [
            {key: 'name', label: 'Name', value: title, required: true},
            {key: 'description', label: 'Beschreibung', value: '', type: 'textarea'},
            {key: 'source', label: 'Quelle', value: 'CONVERTER'},
            {key: 'project', label: 'Projekt', value: '', list: s.projects},
            {key: 'people', label: 'Personen', value: '', placeholder: 'kommagetrennt', list: s.people},
        ]
    };
    const container = document.getElementById('notion-fields');
    let datalistsHtml = '';
    container.innerHTML = fieldDefs[target].map(f => {
        const esc = v => v.replace(/"/g, '&quot;');
        let listAttr = '';
        if (f.list && f.list.length) {
            const dlId = `dl-${f.key}`;
            listAttr = ` list="${dlId}"`;
            datalistsHtml += `<datalist id="${dlId}">${f.list.map(o => `<option value="${esc(o)}">`).join('')}</datalist>`;
        }
        const input = f.type === 'textarea'
            ? `<textarea class="c-input w-full text-xs" id="nf-${f.key}" rows="2" placeholder="${f.placeholder || ''}">${f.value}</textarea>`
            : `<input type="${f.type || 'text'}" class="c-input w-full text-xs" id="nf-${f.key}" value="${esc(f.value)}" placeholder="${f.placeholder || ''}"${listAttr}>`;
        return `<div><label class="text-[11px] text-neo-faint mb-0.5 block">${f.label}${f.required ? ' *' : ''}</label>${input}</div>`;
    }).join('') + datalistsHtml;
}

function sendToNotion() {
    clearNotionAlert();
    const btn = document.getElementById('notion-submit-btn');
    btn.disabled = true;
    btn.textContent = 'Sende …';

    const fields = {};
    document.querySelectorAll('#notion-fields input, #notion-fields textarea').forEach(el => {
        const key = el.id.replace('nf-', '');
        const val = el.value.trim();
        if (val) fields[key] = val;
    });

    const content = document.querySelector('.detail-content-text').textContent;
    if (currentTarget === 'meetings') {
        fields.transcript = content;
    } else {
        fields.content = content;
    }

    if (fields.people) fields.people = fields.people.split(',').map(s => s.trim()).filter(Boolean);
    if (fields.tags) fields.tags = fields.tags.split(',').map(s => s.trim()).filter(Boolean);

    fetch(`/api/conversions/${CONVERSION_ID}/send-to-notion`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({target: currentTarget, fields: fields})
    })
    .then(r => r.json().then(data => ({status: r.status, data})))
    .then(({status, data}) => {
        if (status < 400) {
            showToast('An Notion gesendet');
            if (data.url) window.open(data.url, '_blank');
        } else {
            const detail = data.error || data.detail;
            const msg = detail
                ? `Senden fehlgeschlagen: ${detail}.`
                : 'Senden an Notion fehlgeschlagen. Erneut versuchen oder Server-Konfiguration prüfen.';
            showAlert(notionAlertContainer(), 'danger', msg);
        }
    })
    .catch(() => {
        showAlert(notionAlertContainer(), 'danger',
            'Verbindung zu Notion fehlgeschlagen. Netzwerk und Notion-MCP-Server-Status prüfen.');
    })
    .finally(() => { btn.disabled = false; btn.textContent = 'An Notion senden'; });
}

// --- Tag chips (P9) ---

function parseTagsCsv(csv) {
    const seen = new Set();
    const out = [];
    (csv || '').split(',').forEach(part => {
        const tag = part.trim();
        if (tag && !seen.has(tag)) {
            seen.add(tag);
            out.push(tag);
        }
    });
    return out;
}

function renderTagChips(csv, container) {
    if (!container) return;
    container.innerHTML = '';
    const tags = parseTagsCsv(csv);
    if (!tags.length) {
        const empty = document.createElement('p');
        empty.className = 'tag-chip-empty';
        empty.textContent = 'Noch keine Tags. Mit Komma trennen, um mehrere zu speichern.';
        container.appendChild(empty);
        return;
    }
    const list = document.createElement('div');
    list.className = 'tag-chip-list';
    tags.forEach(tag => {
        const chip = document.createElement('span');
        chip.className = 'c-tag tag-chip';
        const label = document.createElement('span');
        label.textContent = tag;
        chip.appendChild(label);
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'tag-chip__remove';
        remove.setAttribute('aria-label', 'Tag entfernen');
        remove.textContent = '×';
        remove.addEventListener('click', () => {
            const next = tags.filter(t => t !== tag).join(', ');
            const input = document.getElementById('tags-input');
            if (input) input.value = next;
            renderTagChips(next, container);
            updateField('tags', next);
        });
        chip.appendChild(remove);
        list.appendChild(chip);
    });
    container.appendChild(list);
}

// --- Auto-Save dirty indicator + flush-on-hide (P2) ---

function markDirty(field) {
    const input = document.getElementById(AUTOSAVE_INPUTS[field]);
    if (!input || input.classList.contains('c-input--dirty')) return;
    input.classList.add('c-input--dirty');
    if (input.dataset.titleBeforeDirty === undefined) {
        input.dataset.titleBeforeDirty = input.title || '';
    }
    input.title = DIRTY_TOOLTIP;
}

function flushDirtyInputs() {
    Object.keys(AUTOSAVE_INPUTS).forEach(field => {
        const input = document.getElementById(AUTOSAVE_INPUTS[field]);
        if (input && input.classList.contains('c-input--dirty')) {
            updateField(field, input.value);
        }
    });
}

function setupAutoSaveTracking() {
    Object.keys(AUTOSAVE_INPUTS).forEach(field => {
        const input = document.getElementById(AUTOSAVE_INPUTS[field]);
        if (!input) return;
        input.addEventListener('input', () => markDirty(field));
    });
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'hidden') flushDirtyInputs();
    });
    window.addEventListener('beforeunload', flushDirtyInputs);
}

function setupTagChipSync() {
    const container = document.getElementById('tag-chip-container');
    const input = document.getElementById('tags-input');
    if (!container || !input) return;
    renderTagChips(input.value, container);
    input.addEventListener('input', () => renderTagChips(input.value, container));
}

document.addEventListener('DOMContentLoaded', () => {
    setupAutoSaveTracking();
    setupTagChipSync();
});

window.updateField = updateField;
window.toggleFavorite = toggleFavorite;
window.copyFullContent = copyFullContent;
window.downloadContent = downloadContent;
window.storeForReuse = storeForReuse;
window.deleteConversion = deleteConversion;
window.toggleNotionPanel = toggleNotionPanel;
window.selectTarget = selectTarget;
window.sendToNotion = sendToNotion;
