/* Library detail view: inline editing, copy/download/delete, Notion send. */

const CONVERSION_ID = window.PageData.conversionId;
const LIBRARY_URL = window.PageData.libraryUrl;
const DOWNLOAD_FILENAME = window.PageData.downloadFilename;
const DEFAULT_TARGET = window.PageData.defaultNotionTarget;

function updateField(field, value) {
    const body = {};
    body[field] = value;
    fetch(`/api/conversions/${CONVERSION_ID}`, {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
    }).then(r => {
        if (r.ok) showToast(`${field} updated`);
    });
}

function toggleFavorite(btn) {
    const isFav = btn.classList.contains('active');
    fetch(`/api/conversions/${CONVERSION_ID}`, {
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

function copyFullContent() {
    const text = document.querySelector('.detail-content-text').textContent;
    fallbackCopyText(text).then(() => showToast('Copied to clipboard'))
        .catch(() => showToast('Copy failed'));
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
    if (!confirm('Delete this conversion? This cannot be undone.')) return;
    fetch(`/api/conversions/${CONVERSION_ID}`, {method: 'DELETE'}).then(r => {
        if (r.ok) window.location.href = LIBRARY_URL;
    });
}

// --- Notion Integration ---
let currentTarget = DEFAULT_TARGET;
let notionSuggestions = null;

function toggleNotionPanel() {
    const panel = document.getElementById('notion-panel');
    const icon = document.getElementById('notion-toggle-icon');
    const isHidden = panel.classList.toggle('hidden');
    icon.innerHTML = isHidden ? '&#9662;' : '&#9652;';
    if (!isHidden) {
        selectTarget(DEFAULT_TARGET);
        loadSuggestions();
    }
}

function loadSuggestions() {
    if (notionSuggestions) return;
    fetch('/api/notion/suggestions').then(r => r.json()).then(data => {
        notionSuggestions = data;
        renderNotionFields(currentTarget);
    }).catch(() => { notionSuggestions = {people: [], projects: [], meeting_types: [], note_types: []}; });
}

function selectTarget(target) {
    currentTarget = target;
    document.querySelectorAll('#notion-target-group button').forEach(btn => {
        btn.classList.toggle('c-btn--primary', btn.dataset.target === target);
    });
    renderNotionFields(target);
}

function renderNotionFields(target) {
    const title = document.getElementById('detail-title').value;
    const tags = document.getElementById('tags-input').value;
    const now = new Date().toISOString().slice(0, 16);
    const s = notionSuggestions || {people: [], projects: [], meeting_types: [], note_types: []};

    const fieldDefs = {
        meetings: [
            {key: 'title', label: 'Title', value: title, required: true},
            {key: 'datum', label: 'Date', value: now, type: 'datetime-local'},
            {key: 'project', label: 'Project', value: '', list: s.projects},
            {key: 'people', label: 'People', value: '', placeholder: 'comma-separated', list: s.people},
            {key: 'type', label: 'Type', value: '', list: s.meeting_types},
            {key: 'summary', label: 'Summary', value: '', type: 'textarea'},
        ],
        notes: [
            {key: 'title', label: 'Title', value: title, required: true},
            {key: 'project', label: 'Project', value: '', list: s.projects},
            {key: 'type', label: 'Type', value: '', list: s.note_types},
            {key: 'tags', label: 'Tags', value: tags, placeholder: 'comma-separated'},
            {key: 'people', label: 'People', value: '', placeholder: 'comma-separated', list: s.people},
            {key: 'summary', label: 'Summary', value: '', type: 'textarea'},
        ],
        inbox: [
            {key: 'name', label: 'Name', value: title, required: true},
            {key: 'description', label: 'Description', value: '', type: 'textarea'},
            {key: 'source', label: 'Source', value: 'CONVERTER'},
            {key: 'project', label: 'Project', value: '', list: s.projects},
            {key: 'people', label: 'People', value: '', placeholder: 'comma-separated', list: s.people},
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
    const btn = document.getElementById('notion-submit-btn');
    btn.disabled = true;
    btn.textContent = 'Sending...';

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
            showToast('Saved to Notion!');
            if (data.url) window.open(data.url, '_blank');
        } else {
            showToast('Error: ' + (data.error || 'Unknown error'));
        }
    })
    .catch(() => showToast('Failed to connect to Notion'))
    .finally(() => { btn.disabled = false; btn.textContent = 'Send to Notion'; });
}

window.updateField = updateField;
window.toggleFavorite = toggleFavorite;
window.copyFullContent = copyFullContent;
window.downloadContent = downloadContent;
window.storeForReuse = storeForReuse;
window.deleteConversion = deleteConversion;
window.toggleNotionPanel = toggleNotionPanel;
window.selectTarget = selectTarget;
window.sendToNotion = sendToNotion;
