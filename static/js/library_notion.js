/* Library detail view: the "An Notion senden" panel.

   Moved verbatim out of library_detail.js (NOTION-MEETING-LINK, Phase 0) —
   loaded AFTER it. From library_detail.js this file uses, at call time only:
   CONVERSION_ID and conversionTagsState (top-level const/let of a classic
   script live in the shared global scope); from _utils.js: showAlert,
   showToast, safeJSON, formatDatetimeLocalNow. Nothing here runs at load
   time except reading window.PageData. */

const DEFAULT_TARGET = window.PageData.defaultNotionTarget;

const NOTION_TARGET_LABELS = { meetings: 'Meeting', notes: 'Notiz', inbox: 'Inbox' };

function notionAlertContainer() { return document.getElementById('notion-alert-container'); }

function clearNotionAlert() {
    const c = notionAlertContainer();
    if (c) c.innerHTML = '';
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
    // Brief opacity fade on target switch as a visual hint that fields swapped.
    // CSS owns the 150ms transition timing.
    if (isSwitch && container) {
        container.style.opacity = '0';
    }
    renderNotionFields(target);
    if (snapshot) {
        restoreNotionFieldValues(document.getElementById('notion-fields'), snapshot);
    }
    if (isSwitch && container) {
        // Force reflow so the transition runs from the initial state.
        // eslint-disable-next-line no-unused-expressions
        container.offsetHeight;
        container.style.opacity = '1';
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
    const tags = conversionTagsState.map(t => t.name).join(', ');
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
    // Field values come from user-editable inputs (title, tags) and Notion
    // suggestions — both untrusted. Escape before interpolating into HTML.
    const escHtml = v => String(v == null ? '' : v)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    let datalistsHtml = '';
    container.innerHTML = fieldDefs[target].map(f => {
        let listAttr = '';
        if (f.list && f.list.length) {
            const dlId = `dl-${f.key}`;
            listAttr = ` list="${dlId}"`;
            datalistsHtml += `<datalist id="${dlId}">${f.list.map(o => `<option value="${escHtml(o)}">`).join('')}</datalist>`;
        }
        const placeholder = escHtml(f.placeholder || '');
        const input = f.type === 'textarea'
            ? `<textarea class="c-input w-full text-xs" id="nf-${f.key}" rows="2" placeholder="${placeholder}">${escHtml(f.value)}</textarea>`
            : `<input type="${f.type || 'text'}" class="c-input w-full text-xs" id="nf-${f.key}" value="${escHtml(f.value)}" placeholder="${placeholder}"${listAttr}>`;
        return `<div><label class="text-[11px] text-neo-faint mb-0.5 block">${escHtml(f.label)}${f.required ? ' *' : ''}</label>${input}</div>`;
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

    const content = document.getElementById('content-source').textContent;
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
            if (data.url) window.open(data.url, '_blank', 'noopener,noreferrer');
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

window.toggleNotionPanel = toggleNotionPanel;
window.selectTarget = selectTarget;
window.sendToNotion = sendToNotion;
