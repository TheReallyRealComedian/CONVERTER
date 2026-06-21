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
};

// Inputs that participate in the dirty-indicator + flush-on-hide flow.
// Tags moved off this list in R2-A — they have their own POST/DELETE
// attach/detach endpoints now, no auto-save needed.
const AUTOSAVE_INPUTS = { title: 'detail-title' };
const DIRTY_TOOLTIP = 'Ungespeicherte Änderung — Tab oder Klick außerhalb speichert.';
const KINDLE_FAILURE_MSG = 'Versand an Kindle fehlgeschlagen. Verbindung prüfen und erneut versuchen.';
const SESSION_EXPIRED_MSG = 'Sitzung abgelaufen. Seite neu laden und erneut anmelden.';

// Conversion-Tag-State: list of {id, name, ...} dicts seeded from PageData
// at load time, then mutated by addTagToConversion / removeTagFromConversion.
let conversionTagsState = Array.isArray(window.PageData.conversionTags)
    ? window.PageData.conversionTags.slice()
    : [];

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

// R2-H: the one flat move-action (detail view). POSTs the target place; the
// conversion id is fixed to this page. Success is silent (the pressed segment
// is the feedback) — no reload needed, the control just reflects the new place;
// errors use showToast. Subsumes the old lifecycle-toggle + queue-toggle pair
// and holds the four places mutually exclusive server-side (incl. dequeue).
function setPlace(place) {
    fetch(`/api/conversions/${CONVERSION_ID}/place`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ place })
    }).then(r => {
        if (r.ok) {
            clearDetailAlert();
            applyPlaceControl(place);
        } else {
            showToast(withServerSuffix('Ablage konnte nicht geändert werden. Verbindung prüfen und erneut versuchen.', r.status), { level: 'danger' });
        }
    }).catch(() => {
        showToast('Ablage konnte nicht geändert werden. Verbindung prüfen und erneut versuchen.', { level: 'danger' });
    });
}

function applyPlaceControl(place) {
    document.querySelectorAll('[data-place-control] .place-control__btn').forEach(btn => {
        const active = btn.dataset.place === place;
        btn.classList.toggle('is-active', active);
        btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
}

// R2-F/R2-H: Abschluss-Leiste-Archivieren. Nutzt jetzt denselben POST /place
// wie das Move-Control (archiv nimmt das Item auch von der Lese-Liste —
// Exklusivität), mit Abschluss-UX: Erfolgs-Toast, dann zurück zur Library.
// Kurzer Delay vor der Navigation, damit der Toast sichtbar ist, bevor die
// Seite entladen wird. Fehler bleiben auf der Seite (Toast, keine Navigation).
function finishArchive() {
    fetch(`/api/conversions/${CONVERSION_ID}/place`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({place: 'archiv'})
    }).then(r => {
        if (r.ok) {
            clearDetailAlert();
            applyPlaceControl('archiv');
            showToast('Archiviert');
            setTimeout(() => { window.location.href = LIBRARY_URL; }, 700);
        } else {
            showToast(withServerSuffix('Archivieren fehlgeschlagen. Verbindung prüfen und erneut versuchen.', r.status), { level: 'danger' });
        }
    }).catch(() => {
        showToast('Archivieren fehlgeschlagen. Verbindung prüfen und erneut versuchen.', { level: 'danger' });
    });
}

// R2-F: "Zurück zur Library" der Abschluss-Leiste. Wenn der direkte Referrer
// die Library-Liste war, via history.back() zurück — das erhält Scroll-Position
// + Filter-State der Liste. Sonst navigiert der Klick normal über den href
// (LIBRARY_URL), der auch ohne JS / History als Fallback funktioniert.
function initFinishBackLink() {
    const back = document.getElementById('reader-finish-back');
    if (!back) return;
    back.addEventListener('click', (e) => {
        try {
            const ref = document.referrer;
            if (!ref) return;
            const refUrl = new URL(ref);
            const libPath = new URL(LIBRARY_URL, location.origin).pathname;
            if (refUrl.origin === location.origin && refUrl.pathname === libPath) {
                e.preventDefault();
                history.back();
            }
        } catch (_) { /* malformed referrer → href-Fallback greift */ }
    });
}

function copyFullContent() {
    const text = document.getElementById('content-source').textContent;
    fallbackCopyText(text).then(() => showToast('Kopiert'))
        .catch(() => showToast('Kopieren fehlgeschlagen', { level: 'danger' }));
}

function downloadContent() {
    const text = document.getElementById('content-source').textContent;
    const blob = new Blob([text], {type: 'text/plain'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = DOWNLOAD_FILENAME;
    a.click();
    URL.revokeObjectURL(a.href);
}

function storeForReuse() {
    const text = document.getElementById('content-source').textContent;
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

// Send this conversion to the Kindle (EPUB via Send-to-Kindle email). Non-
// destructive, so no confirm. The button disables for the in-flight request
// (double-click guard) and restores in .finally. On error we surface the
// server's message verbatim (503 → „Kindle nicht konfiguriert.", 502 →
// „Versand an Kindle fehlgeschlagen.").
function sendToKindle() {
    const btn = document.getElementById('kindle-send-btn');
    if (btn) { btn.disabled = true; btn.textContent = 'Sende …'; }
    fetch(`/api/conversions/${CONVERSION_ID}/send-to-kindle`, { method: 'POST' }).then(r => {
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
        if (btn) { btn.disabled = false; btn.textContent = 'An Kindle'; }
    });
}

// --- Conversion tag picker (R2-A) ---

function renderConversionTagChips() {
    const container = document.getElementById('conversion-tag-chips');
    if (!container) return;
    container.innerHTML = '';
    if (conversionTagsState.length === 0) {
        const empty = document.createElement('span');
        empty.className = 'conversion-tag-chips__empty';
        empty.textContent = 'Noch keine Tags.';
        container.appendChild(empty);
        return;
    }
    conversionTagsState.forEach(tag => {
        const chip = document.createElement('span');
        chip.className = 'conversion-tag-chip';
        const label = document.createElement('span');
        label.textContent = tag.name;
        chip.appendChild(label);
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'conversion-tag-chip__remove';
        remove.setAttribute('aria-label', `Tag ${tag.name} entfernen`);
        remove.textContent = '×';
        remove.addEventListener('click', () => removeTagFromConversion(tag.id));
        chip.appendChild(remove);
        container.appendChild(chip);
    });
}

async function loadConversionTagSuggestions() {
    let resp;
    try {
        resp = await fetch('/api/tags');
    } catch (_) {
        return;
    }
    if (!resp.ok) return;
    const tags = await resp.json();
    const datalist = document.getElementById('conversion-tag-suggestions');
    if (!datalist) return;
    datalist.innerHTML = '';
    tags.forEach(tag => {
        const opt = document.createElement('option');
        opt.value = tag.name;
        datalist.appendChild(opt);
    });
}

async function addTagToConversion() {
    const input = document.getElementById('conversion-tag-input');
    if (!input) return;
    const raw = input.value;
    if (!raw || !raw.trim()) return;
    let resp;
    try {
        resp = await fetch(`/api/conversions/${CONVERSION_ID}/tags`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: raw }),
        });
    } catch (_) {
        showToast('Tag speichern fehlgeschlagen. Verbindung prüfen.', { level: 'danger' });
        return;
    }
    if (!resp.ok) {
        if (resp.status === 400) {
            showToast('Tag-Name ungültig oder zu lang.', { level: 'danger' });
        } else {
            showToast('Tag speichern fehlgeschlagen.', { level: 'danger' });
        }
        return;
    }
    const tag = await resp.json();
    if (!conversionTagsState.some(t => t.id === tag.id)) {
        conversionTagsState.push(tag);
    }
    input.value = '';
    renderConversionTagChips();
    loadConversionTagSuggestions();
    if (resp.status === 201) {
        showToast('Tag hinzugefügt.');
    }
}

async function removeTagFromConversion(tagId) {
    let resp;
    try {
        resp = await fetch(`/api/conversions/${CONVERSION_ID}/tags/${tagId}`, { method: 'DELETE' });
    } catch (_) {
        showToast('Tag entfernen fehlgeschlagen. Verbindung prüfen.', { level: 'danger' });
        return;
    }
    if (!resp.ok && resp.status !== 404) {
        showToast('Tag entfernen fehlgeschlagen.', { level: 'danger' });
        return;
    }
    conversionTagsState = conversionTagsState.filter(t => t.id !== tagId);
    renderConversionTagChips();
    showToast('Tag entfernt.');
}

function initConversionTagPicker() {
    renderConversionTagChips();
    loadConversionTagSuggestions();
    const addBtn = document.getElementById('conversion-tag-add-btn');
    if (addBtn) {
        addBtn.addEventListener('click', evt => {
            evt.preventDefault();
            addTagToConversion();
        });
    }
    const input = document.getElementById('conversion-tag-input');
    if (input) {
        input.addEventListener('keydown', evt => {
            if (evt.key === 'Enter') {
                evt.preventDefault();
                addTagToConversion();
            }
        });
    }
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

// --- Highlights (R1-B-A) ---

const HIGHLIGHT_CONTEXT_LEN = 32;
const HIGHLIGHT_EXACT_LIMIT = 5000;
// Mouseup-Selections kürzer als 3 chars werden ignoriert — schützt vor
// Accidental-Highlights bei Klick-mit-Mini-Drift oder Doppelklick auf 1-char-Tokens.
const MIN_HIGHLIGHT_LENGTH = 3;

function highlightReaderEl() { return document.querySelector('.reader-view'); }
function highlightActionPopover() { return document.getElementById('highlight-action-popover'); }
let activeHighlightId = null;
// Modul-Cache für die Sidebar — beide Render-Pfade (Reader-Apply + Sidebar) lesen daraus.
let highlightsState = [];
// IDs, deren Anchor sich nicht in einen DOM-Span wrappen ließ (Cross-Format).
let crossFormatHighlightIds = new Set();

function hideHighlightActionPopover() {
    const pop = highlightActionPopover();
    if (pop) pop.style.display = 'none';
    activeHighlightId = null;
}

// Concatenation of every text-node's nodeValue inside the reader, in
// document order. This is the SAME coordinate system that rangeForOffsets
// walks, so save-time anchors line up with load-time re-apply.
// Using innerText would inject block-level \n separators that don't appear
// in node.nodeValue, producing offset drift that splits highlight spans
// at the wrong character.
function readerRawText(reader) {
    const walker = document.createTreeWalker(reader, NodeFilter.SHOW_TEXT);
    let out = '';
    let node;
    while ((node = walker.nextNode())) out += node.nodeValue;
    return out;
}

// Map a DOM point (container, offset) to its character index within
// readerRawText (the nodeValue concatenation). The fast path handles
// text-node containers — the normal mouse-selection case — by summing node
// lengths until the container. Element containers (e.g. a triple-click that
// selects a whole block) fall back to a document-order comparison so the
// offset still lands in the same coordinate system.
function rawOffsetForPoint(reader, container, offset) {
    const walker = document.createTreeWalker(reader, NodeFilter.SHOW_TEXT);
    let consumed = 0;
    let node;
    if (container.nodeType === Node.TEXT_NODE) {
        while ((node = walker.nextNode())) {
            if (node === container) return consumed + offset;
            consumed += node.nodeValue.length;
        }
        return consumed;
    }
    // comparePoint(node, k): -1 if (node,k) is before our point, 0 if equal,
    // 1 if after. So cmpStart===1 means the point precedes this text node.
    const point = document.createRange();
    point.setStart(container, offset);
    while ((node = walker.nextNode())) {
        const len = node.nodeValue.length;
        if (point.comparePoint(node, 0) === 1) return consumed;
        if (point.comparePoint(node, len) >= 0) {
            let k = 0;
            while (k < len && point.comparePoint(node, k) === -1) k++;
            return consumed + k;
        }
        consumed += len;
    }
    return consumed;
}

// Build the highlight anchor entirely from readerRawText so save-time and
// load-time share ONE coordinate system. selection.toString() must NEVER be
// the stored search key: at block boundaries it inserts separator newlines
// (e.g. "\n\n") that the nodeValue concatenation does not contain, so
// locateHighlightOffset's indexOf would never re-find the anchor and the
// highlight silently became "cross-format" (READER-FIX-B root cause).
function extractSelectionContext(selection) {
    const reader = highlightReaderEl();
    if (!reader || selection.rangeCount === 0) return {exact: '', prefix: '', suffix: ''};
    const range = selection.getRangeAt(0);
    const fullText = readerRawText(reader);
    const rawStart = rawOffsetForPoint(reader, range.startContainer, range.startOffset);
    const rawEnd = rawOffsetForPoint(reader, range.endContainer, range.endOffset);
    const exact = fullText.slice(rawStart, rawEnd);
    const prefix = fullText.slice(Math.max(0, rawStart - HIGHLIGHT_CONTEXT_LEN), rawStart);
    const suffix = fullText.slice(rawEnd, rawEnd + HIGHLIGHT_CONTEXT_LEN);
    return {exact, prefix, suffix};
}

async function saveCurrentSelection() {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) return;
    const {exact, prefix, suffix} = extractSelectionContext(sel);
    if (!exact.trim()) return;
    if (exact.length > HIGHLIGHT_EXACT_LIMIT) {
        showToast('Markierung zu lang. Bitte einen kürzeren Abschnitt wählen.', { level: 'danger' });
        return;
    }
    sel.removeAllRanges();

    let resp;
    try {
        resp = await fetch(`/api/conversions/${CONVERSION_ID}/highlights`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({exact, prefix, suffix}),
        });
    } catch (_) {
        showToast('Markierung speichern fehlgeschlagen. Verbindung prüfen.', { level: 'danger' });
        return;
    }
    if (!resp.ok) {
        showToast('Markierung speichern fehlgeschlagen.', { level: 'danger' });
        return;
    }
    const highlight = await resp.json();
    const applied = applyHighlight(highlight);
    if (!applied) crossFormatHighlightIds.add(highlight.id);
    // Backend liefert chronologisch (created_at asc); neue Highlights gehören ans Ende.
    highlightsState.push(highlight);
    renderHighlightList();
    // Im Normal-Fall ist der gelbe Span sofort sichtbar — das ist das Feedback.
    // Reader-Mode bleibt still. Cross-Format hat keinen Span, daher Toast Pflicht.
    if (!applied) {
        showToast('Markierung gespeichert, Anzeige nicht möglich (Formatierungsgrenze).', { level: 'warning' });
    }
}

async function loadHighlights() {
    let resp;
    try {
        resp = await fetch(`/api/conversions/${CONVERSION_ID}/highlights`);
    } catch (_) {
        return;
    }
    if (!resp.ok) return;
    const highlights = await resp.json();
    highlightsState = highlights;
    crossFormatHighlightIds = new Set();
    highlights.forEach(h => {
        const applied = applyHighlight(h);
        if (!applied) crossFormatHighlightIds.add(h.id);
    });
    renderHighlightList();
}

function renderHighlightList() {
    const list = document.getElementById('highlight-list');
    const counter = document.getElementById('highlight-count');
    if (!list) return;
    if (counter) counter.textContent = `(${highlightsState.length})`;
    if (highlightsState.length === 0) {
        list.innerHTML = '<p class="text-sm text-neo-faint italic" id="highlight-list-empty">Noch keine Markierungen.</p>';
        return;
    }
    // Welche Cards waren vor dem Re-Render aufgeklappt? Merken, damit ein
    // Tag-Add/Remove (das renderHighlightList aufruft) die offene Card nicht
    // zuklappt und der frische Tag optisch "verschwindet". Re-Render baut die
    // Cards neu, also geht der DOM-State sonst verloren.
    const expandedIds = new Set(
        [...list.querySelectorAll('.highlight-card[data-expanded="true"]')]
            .map(c => c.dataset.highlightId)
    );
    list.innerHTML = '';
    highlightsState.forEach(h => {
        const card = document.createElement('div');
        card.className = 'highlight-card';
        card.dataset.highlightId = String(h.id);
        // Expand-State per-doc-view ephemeral; nicht persistiert, aber über
        // einen Re-Render hinweg bewahrt (siehe expandedIds oben). Nur die
        // Klasse wiederherstellen — kein scrollToHighlight, das gehört zum Klick.
        card.dataset.expanded = expandedIds.has(String(h.id)) ? 'true' : 'false';
        const isCrossFormat = crossFormatHighlightIds.has(h.id);
        if (isCrossFormat) {
            card.classList.add('highlight-card--cross-format');
            card.title = 'Markierung über Formatierungsgrenze. Klick: bearbeiten.';
        }
        const exactEl = document.createElement('div');
        exactEl.className = 'highlight-card__exact';
        // Anzeige: Whitespace kollabieren, damit block-übergreifende Roh-`exact`
        // (mit eingebettetem \n) auf der Card nicht ohne sichtbaren Umbruch
        // zusammenkleben. Gespeicherter Such-Key (h.exact) bleibt unangetastet.
        exactEl.textContent = (h.exact || '').replace(/\s+/g, ' ');
        card.appendChild(exactEl);
        if (h.note) {
            const noteEl = document.createElement('div');
            noteEl.className = 'highlight-card__note';
            noteEl.textContent = h.note;
            card.appendChild(noteEl);
        }
        if (Array.isArray(h.tags) && h.tags.length) {
            renderSidebarCardTagChips(card, h.tags);
        }
        // Edit-Button nur in expanded-State sichtbar (CSS). Öffnet das
        // Action-Popover wie bei Cross-Format — Card ist der Anker.
        const editBtn = document.createElement('button');
        editBtn.type = 'button';
        editBtn.className = 'c-btn highlight-card__edit-btn text-xs py-1 px-2';
        editBtn.textContent = 'Bearbeiten';
        editBtn.addEventListener('click', evt => {
            evt.stopPropagation();
            showHighlightActionPopover(card);
        });
        card.appendChild(editBtn);
        card.addEventListener('click', () => {
            const wasExpanded = card.dataset.expanded === 'true';
            const willExpand = !wasExpanded;
            card.dataset.expanded = willExpand ? 'true' : 'false';
            // Nur beim Expand scrollen; Collapse soll User-Lese-Position
            // nicht weg-bewegen. Cross-Format-Highlights haben keinen Span,
            // also kein Scroll-Ziel — Edit-Button in expanded ist der Pfad.
            if (willExpand && !isCrossFormat) {
                scrollToHighlight(h.id);
            }
        });
        list.appendChild(card);
    });
}

function scrollToHighlight(id) {
    const span = document.querySelector(
        `.reader-view span.highlight[data-highlight-id="${CSS.escape(String(id))}"]`
    );
    if (!span) {
        showToast(
            'Markierung über Formatierungsgrenze, im Text nicht direkt anspringbar.',
            { level: 'warning' }
        );
        return;
    }
    span.scrollIntoView({ behavior: 'smooth', block: 'center' });
    span.classList.add('highlight-flash');
    setTimeout(() => span.classList.remove('highlight-flash'), 1000);
}

// Locate the stored anchor in readerRawText. Returns {start, end} raw-offset
// bounds (same coordinate system extractSelectionContext writes), or null.
// New anchors are sliced from readerRawText at save time, so the fast path is
// an exact indexOf and prefix/suffix disambiguate a multi-match. Pre-fix
// anchors were stored from selection.toString() (block-separator newlines,
// collapsed whitespace) and never match exactly — locateWhitespaceTolerant
// rescues them best-effort.
function locateHighlightOffset(reader, highlight) {
    const fullText = readerRawText(reader);
    const {exact, prefix, suffix} = highlight;
    if (!exact) return null;
    const positions = [];
    let cursor = 0;
    while (cursor <= fullText.length) {
        const idx = fullText.indexOf(exact, cursor);
        if (idx === -1) break;
        positions.push(idx);
        cursor = idx + Math.max(1, exact.length);
    }
    if (positions.length === 0) return locateWhitespaceTolerant(fullText, exact);

    let start = positions[0];
    if (positions.length > 1) {
        let bestScore = -1;
        for (const pos of positions) {
            const actualPrefix = fullText.slice(Math.max(0, pos - (prefix?.length || 0)), pos);
            const actualSuffix = fullText.slice(pos + exact.length, pos + exact.length + (suffix?.length || 0));
            let score = 0;
            if (prefix && actualPrefix === prefix) score += 2;
            else if (prefix && actualPrefix.endsWith(prefix.slice(-8))) score += 1;
            if (suffix && actualSuffix === suffix) score += 2;
            else if (suffix && actualSuffix.startsWith(suffix.slice(0, 8))) score += 1;
            if (score > bestScore) {
                bestScore = score;
                start = pos;
            }
        }
    }
    return {start, end: start + exact.length};
}

// Best-effort rescue for pre-READER-FIX-B anchors stored from
// selection.toString(): collapse whitespace runs in both the reader text and
// the stored `exact`, match in that normalized space, then map the hit back
// to raw offsets via an index map (normalized char i → raw index map[i]).
// Returns {start, end} or null. First-match only — old anchors are few and
// this is a rescue, not the precise locate the fast path provides.
function locateWhitespaceTolerant(fullText, exact) {
    const map = [];
    let norm = '';
    let prevSpace = false;
    for (let i = 0; i < fullText.length; i++) {
        if (/\s/.test(fullText[i])) {
            if (!prevSpace && norm.length) { norm += ' '; map.push(i); }
            prevSpace = true;
        } else {
            norm += fullText[i]; map.push(i); prevSpace = false;
        }
    }
    const needle = exact.replace(/\s+/g, ' ').trim();
    if (!needle) return null;
    const nIdx = norm.indexOf(needle);
    if (nIdx === -1) return null;
    return {start: map[nIdx], end: map[nIdx + needle.length - 1] + 1};
}

// Walk text nodes inside `reader`, find the [start, end) offset range in
// reader.innerText terms, and return a Range object spanning that slice.
// The Range may span multiple text nodes — wrapSelectionAsHighlight handles
// per-node wrapping; surroundContents on the raw multi-node range would throw.
function rangeForOffsets(reader, startOffset, endOffset) {
    const walker = document.createTreeWalker(reader, NodeFilter.SHOW_TEXT);
    let consumed = 0;
    let startNode = null;
    let startNodeOffset = 0;
    let endNode = null;
    let endNodeOffset = 0;
    let node = walker.nextNode();
    while (node) {
        const len = node.nodeValue.length;
        if (startNode === null && consumed + len > startOffset) {
            startNode = node;
            startNodeOffset = startOffset - consumed;
        }
        if (startNode !== null && consumed + len >= endOffset) {
            endNode = node;
            endNodeOffset = endOffset - consumed;
            break;
        }
        consumed += len;
        node = walker.nextNode();
    }
    if (!startNode || !endNode) return null;
    const range = document.createRange();
    range.setStart(startNode, startNodeOffset);
    range.setEnd(endNode, endNodeOffset);
    return range;
}

// Sammle pro Text-Node innerhalb des Range das (textNode, startOffset, endOffset)-Tripel.
// Single-Node-Range → 1 Eintrag (R1-B-A-Pfad, backward-compatible).
// Multi-Node-Range (z.B. über `<strong>`) → N Einträge, alle wrappbar.
// Whitespace-only Nodes werden übersprungen — sonst entstehen leere Spans
// im DOM die niemand sieht aber das Normalize-Verhalten beim Delete stören.
function collectTextRangesInRange(range) {
    const result = [];
    const root = range.commonAncestorContainer;
    const walkerRoot = root.nodeType === Node.TEXT_NODE ? root.parentNode : root;
    const walker = document.createTreeWalker(
        walkerRoot,
        NodeFilter.SHOW_TEXT,
        {
            acceptNode(node) {
                if (!range.intersectsNode(node)) return NodeFilter.FILTER_REJECT;
                return NodeFilter.FILTER_ACCEPT;
            }
        }
    );
    let node = walker.nextNode();
    while (node) {
        if (!node.nodeValue || !node.nodeValue.trim()) {
            node = walker.nextNode();
            continue;
        }
        const startOffset = (node === range.startContainer) ? range.startOffset : 0;
        const endOffset = (node === range.endContainer) ? range.endOffset : node.nodeValue.length;
        if (startOffset < endOffset) {
            result.push({ textNode: node, startOffset, endOffset });
        }
        node = walker.nextNode();
    }
    return result;
}

// Wrap jedes Text-Node-Sub-Range in einen eigenen `<span.highlight>` mit der
// gleichen data-highlight-id. Reverse-Order-Wrap (Ende → Anfang) damit frühere
// DOM-Mutationen die Offsets späterer Sub-Ranges nicht verschieben.
// Returns true wenn mind. ein Span gesetzt wurde.
function wrapSelectionAsHighlight(range, highlightId) {
    const textRanges = collectTextRangesInRange(range);
    if (textRanges.length === 0) return false;
    for (let i = textRanges.length - 1; i >= 0; i--) {
        const { textNode, startOffset, endOffset } = textRanges[i];
        const subRange = document.createRange();
        subRange.setStart(textNode, startOffset);
        subRange.setEnd(textNode, endOffset);
        const span = document.createElement('span');
        span.className = 'highlight';
        span.dataset.highlightId = String(highlightId);
        try {
            subRange.surroundContents(span);
        } catch (err) {
            // Single-Text-Node-Sub-Range sollte immer wrappbar sein; nur
            // bei Custom-Elements/Shadow-DOM-Edge-Cases bricht das.
            console.warn('subRange surroundContents failed', err);
        }
    }
    return true;
}

// Entfernt alle Spans einer Highlight-ID und normalisiert den Reader einmal
// am Ende (statt N-mal pro Span). N kann nach Range-Walking > 1 sein.
function removeHighlightSpans(id) {
    const reader = highlightReaderEl();
    if (!reader) return;
    const spans = reader.querySelectorAll(
        `span.highlight[data-highlight-id="${CSS.escape(String(id))}"]`
    );
    spans.forEach(span => {
        const parent = span.parentNode;
        while (span.firstChild) parent.insertBefore(span.firstChild, span);
        parent.removeChild(span);
    });
    reader.normalize();
}

function applyHighlight(highlight) {
    const reader = highlightReaderEl();
    if (!reader) return false;
    const loc = locateHighlightOffset(reader, highlight);
    if (!loc || loc.start < 0 || loc.end <= loc.start) return false;
    const range = rangeForOffsets(reader, loc.start, loc.end);
    if (!range) return false;
    return wrapSelectionAsHighlight(range, highlight.id);
}

function showHighlightActionPopover(anchorEl) {
    // anchorEl ist normalerweise der DOM-Span im Reader-View. Für
    // Cross-Format-Highlights ohne Span wird stattdessen die Sidebar-Card
    // als Anker übergeben — beide tragen dataset.highlightId und
    // getBoundingClientRect, das Popover positioniert sich unter dem
    // jeweiligen Element.
    const pop = highlightActionPopover();
    if (!pop || !anchorEl) return;
    const rect = anchorEl.getBoundingClientRect();
    activeHighlightId = anchorEl.dataset.highlightId;
    const idNum = parseInt(activeHighlightId, 10);
    const highlight = highlightsState.find(h => h.id === idNum);
    const input = document.getElementById('highlight-note-input');
    if (input) {
        input.value = highlight && highlight.note ? highlight.note : '';
    }
    renderHighlightTagChips(highlight);
    const tagInput = document.getElementById('highlight-tag-input');
    if (tagInput) tagInput.value = '';
    loadTagSuggestions();
    pop.style.display = 'flex';
    pop.style.top = `${window.scrollY + rect.bottom + 6}px`;
    pop.style.left = `${window.scrollX + rect.left}px`;
    // Defer focus so the click that opened the popover doesn't immediately
    // bubble to the document-level "close-on-outside-click" handler.
    if (input) setTimeout(() => input.focus(), 0);
}

const SIDEBAR_TAG_CHIPS_VISIBLE = 3;

function renderHighlightTagChips(highlight) {
    const container = document.getElementById('highlight-tag-chips');
    if (!container) return;
    container.innerHTML = '';
    const tags = (highlight && highlight.tags) || [];
    if (tags.length === 0) {
        const empty = document.createElement('span');
        empty.className = 'highlight-tag-chips__empty';
        empty.textContent = 'Noch keine Tags.';
        container.appendChild(empty);
        return;
    }
    tags.forEach(tag => {
        const chip = document.createElement('span');
        chip.className = 'highlight-tag-chip';
        const label = document.createElement('span');
        label.textContent = tag.name;
        chip.appendChild(label);
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'highlight-tag-chip__remove';
        remove.setAttribute('aria-label', `Tag ${tag.name} entfernen`);
        remove.textContent = '×';
        remove.addEventListener('mousedown', evt => {
            evt.preventDefault();
            removeTagFromHighlight(tag.id);
        });
        chip.appendChild(remove);
        container.appendChild(chip);
    });
}

function renderSidebarCardTagChips(cardEl, tags) {
    if (!cardEl || !tags || tags.length === 0) return;
    const wrap = document.createElement('div');
    wrap.className = 'highlight-card__tags';
    // Alle Tags ins DOM rendern; CSS hidet die ab Position 4 im collapsed-State
    // und zeigt stattdessen den --more-Chip. Im expanded-State sind alle Tag-Chips
    // sichtbar, der --more-Chip versteckt. stopPropagation auf jeden Chip damit
    // ein Tag-Click die Card nicht ungewollt collapsed (READER-FIX-A Smoke 13).
    tags.forEach(tag => {
        const chip = document.createElement('span');
        chip.className = 'highlight-tag-chip highlight-tag-chip--compact';
        chip.textContent = tag.name;
        chip.addEventListener('click', evt => evt.stopPropagation());
        wrap.appendChild(chip);
    });
    if (tags.length > SIDEBAR_TAG_CHIPS_VISIBLE) {
        const more = document.createElement('span');
        more.className = 'highlight-tag-chip highlight-tag-chip--compact highlight-tag-chip--more';
        more.textContent = `+${tags.length - SIDEBAR_TAG_CHIPS_VISIBLE}`;
        wrap.appendChild(more);
    }
    cardEl.appendChild(wrap);
}

async function loadTagSuggestions() {
    let resp;
    try {
        resp = await fetch('/api/tags');
    } catch (_) {
        return;
    }
    if (!resp.ok) return;
    const tags = await resp.json();
    const datalist = document.getElementById('tag-suggestions');
    if (!datalist) return;
    datalist.innerHTML = '';
    tags.forEach(tag => {
        const opt = document.createElement('option');
        opt.value = tag.name;
        datalist.appendChild(opt);
    });
}

async function addTagToHighlight() {
    if (!activeHighlightId) return;
    const input = document.getElementById('highlight-tag-input');
    if (!input) return;
    const raw = input.value;
    if (!raw || !raw.trim()) return;
    const id = activeHighlightId;
    let resp;
    try {
        resp = await fetch(`/api/highlights/${id}/tags`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: raw }),
        });
    } catch (_) {
        showToast('Tag speichern fehlgeschlagen. Verbindung prüfen.', { level: 'danger' });
        return;
    }
    if (!resp.ok) {
        if (resp.status === 400) {
            showToast('Tag-Name ungültig oder zu lang.', { level: 'danger' });
        } else {
            showToast('Tag speichern fehlgeschlagen.', { level: 'danger' });
        }
        return;
    }
    const tag = await resp.json();
    const numericId = parseInt(id, 10);
    const highlight = highlightsState.find(h => h.id === numericId);
    if (highlight) {
        highlight.tags = highlight.tags || [];
        if (!highlight.tags.some(t => t.id === tag.id)) {
            highlight.tags.push(tag);
        }
        renderHighlightTagChips(highlight);
    }
    input.value = '';
    renderHighlightList();
    loadTagSuggestions();
    // Hier nur erreichbar wenn resp.ok — also 201 (neu angehängt) oder 200
    // (idempotenter Re-Add, Tag war schon dran). Beide bestätigen, sonst bleibt
    // der Re-Add still und der User weiß nicht, ob er gespeichert hat.
    if (resp.status === 201) {
        showToast('Tag hinzugefügt.');
    } else {
        showToast('Tag bereits vorhanden.');
    }
}

async function removeTagFromHighlight(tagId) {
    if (!activeHighlightId) return;
    const id = activeHighlightId;
    let resp;
    try {
        resp = await fetch(`/api/highlights/${id}/tags/${tagId}`, { method: 'DELETE' });
    } catch (_) {
        showToast('Tag entfernen fehlgeschlagen. Verbindung prüfen.', { level: 'danger' });
        return;
    }
    if (!resp.ok && resp.status !== 404) {
        showToast('Tag entfernen fehlgeschlagen.', { level: 'danger' });
        return;
    }
    const numericId = parseInt(id, 10);
    const highlight = highlightsState.find(h => h.id === numericId);
    if (highlight && Array.isArray(highlight.tags)) {
        highlight.tags = highlight.tags.filter(t => t.id !== tagId);
        renderHighlightTagChips(highlight);
    }
    renderHighlightList();
    showToast('Tag entfernt.');
}

async function saveHighlightNote() {
    if (!activeHighlightId) return;
    const id = activeHighlightId;
    const input = document.getElementById('highlight-note-input');
    if (!input) return;
    const noteValue = input.value;
    let resp;
    try {
        resp = await fetch(`/api/highlights/${id}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ note: noteValue }),
        });
    } catch (_) {
        showToast('Notiz speichern fehlgeschlagen. Verbindung prüfen.', { level: 'danger' });
        return;
    }
    if (!resp.ok) {
        showToast('Notiz speichern fehlgeschlagen.', { level: 'danger' });
        return;
    }
    const updated = await resp.json();
    const idx = highlightsState.findIndex(h => h.id === updated.id);
    if (idx >= 0) highlightsState[idx] = updated;
    renderHighlightList();
    hideHighlightActionPopover();
    showToast('Notiz gespeichert.');
}

async function deleteActiveHighlight() {
    if (!activeHighlightId) return;
    const id = activeHighlightId;
    let resp;
    try {
        resp = await fetch(`/api/highlights/${id}`, {method: 'DELETE'});
    } catch (_) {
        showToast('Löschen fehlgeschlagen. Verbindung prüfen.', { level: 'danger' });
        return;
    }
    if (!resp.ok && resp.status !== 404) {
        showToast('Löschen fehlgeschlagen.', { level: 'danger' });
        return;
    }
    removeHighlightSpans(id);
    const numericId = parseInt(id, 10);
    highlightsState = highlightsState.filter(h => h.id !== numericId);
    crossFormatHighlightIds.delete(numericId);
    renderHighlightList();
    hideHighlightActionPopover();
    showToast('Markierung entfernt.');
}

function initHighlights() {
    const reader = highlightReaderEl();
    if (!reader) return;
    const popover = highlightActionPopover();
    const deleteBtn = document.getElementById('highlight-delete-btn');

    reader.addEventListener('click', evt => {
        const span = evt.target.closest('span.highlight[data-highlight-id]');
        if (span) {
            evt.stopPropagation();
            showHighlightActionPopover(span);
        } else {
            hideHighlightActionPopover();
        }
    });
    if (deleteBtn) {
        deleteBtn.addEventListener('mousedown', evt => {
            evt.preventDefault();
            deleteActiveHighlight();
        });
    }
    const saveNoteBtn = document.getElementById('highlight-save-note-btn');
    if (saveNoteBtn) {
        saveNoteBtn.addEventListener('mousedown', evt => {
            evt.preventDefault();
            saveHighlightNote();
        });
    }
    const tagAddBtn = document.getElementById('highlight-tag-add-btn');
    if (tagAddBtn) {
        tagAddBtn.addEventListener('mousedown', evt => {
            evt.preventDefault();
            addTagToHighlight();
        });
    }
    const tagInput = document.getElementById('highlight-tag-input');
    if (tagInput) {
        tagInput.addEventListener('keydown', evt => {
            if (evt.key === 'Enter') {
                evt.preventDefault();
                addTagToHighlight();
            }
        });
    }
    document.addEventListener('click', evt => {
        if (
            popover
            && !popover.contains(evt.target)
            && !evt.target.closest('span.highlight[data-highlight-id]')
            && !evt.target.closest('.highlight-card--cross-format')
        ) {
            hideHighlightActionPopover();
        }
    });

    loadHighlights();
}

// --- Mark-on-Mouseup (READER-MODE) ---
// Mouseup im Reader-View mit nicht-leerer, ausreichend langer Selection
// triggert direkt das Highlight-Save. Ersetzt den schwebenden Markieren-Button.
// Cmd/Ctrl-Halten = nur Kopieren-Geste, kein Highlight.
function initMarkOnMouseup() {
    const reader = highlightReaderEl();
    if (!reader) return;
    reader.addEventListener('mouseup', evt => {
        if (evt.metaKey || evt.ctrlKey) return;
        // Defer um einen Tick — manche Browser propagieren Selection erst
        // nach dem mouseup-Listener; ohne setTimeout liest .toString() leeren String.
        setTimeout(() => {
            const sel = window.getSelection();
            if (!sel || sel.isCollapsed || sel.rangeCount === 0) return;
            if (!reader.contains(sel.anchorNode) || !reader.contains(sel.focusNode)) return;
            const text = sel.toString();
            if (text.trim().length < MIN_HIGHLIGHT_LENGTH) return;
            saveCurrentSelection();
        }, 10);
    });
}

// --- Reading-Progress-Bar (READER-MODE + R2-B Persist/Resume) ---
// 3px-sticky-Bar am Top des Reader-Containers, Fill-Width entspricht Scroll-Fortschritt.
// Reader-Container hat keine eigene overflow mehr (natural height) — Scroll passiert auf
// dem nearest scrollable ancestor (typisch `<div class="flex-1 ... overflow-auto">` aus
// block content) oder fallback auf window. Sticky `top: 0` klebt am Scrollport-Top.
// Bei Docs die nicht scrollen Bar versteckt.
// R2-B: der höchste erreichte Prozent-Wert (furthest-read) wird throttled
// persistiert (PATCH /api/conversions/<id>/progress) und beim Öffnen resumed —
// Zurückscrollen resettet den Fortschritt nicht.
function initReadingProgress() {
    const container = document.getElementById('content-body');
    const fill = document.getElementById('reading-progress-fill');
    const wrapper = container ? container.querySelector('.reading-progress') : null;
    if (!container || !fill || !wrapper) return;

    // Finde den nächsten scrollenden Vorfahren (overflow auto/scroll).
    // Fallback auf document.scrollingElement / window.
    function findScrollAncestor(el) {
        let p = el.parentElement;
        while (p && p !== document.body) {
            const oy = getComputedStyle(p).overflowY;
            if (oy === 'auto' || oy === 'scroll') return p;
            p = p.parentElement;
        }
        return document.scrollingElement || document.documentElement;
    }
    const scroller = findScrollAncestor(container);
    const scrollSource = scroller === document.scrollingElement ? window : scroller;

    // Mini-Scroll (z.B. weil Sidebar geringfügig höher ist als Reader-Content)
    // wird als "nicht scrollend" behandelt — sonst zeigt die Bar bei kurzen Docs
    // einen leeren Track ohne tatsächlichen Lese-Fortschritt.
    const MIN_SCROLLABLE_PX = 30;

    // Furthest-read: maxReached wird aus dem gespeicherten Wert geseedet, damit
    // Zurückscrollen den Fortschritt nicht senkt (Workshop #4). persistArmed
    // bleibt false bis der Resume-Scroll gesettled ist — so kann der
    // programmatische Resume-Scroll sich nicht selbst persistieren.
    const seeded = Number(window.PageData.lastReadPercent);
    let maxReached = Number.isFinite(seeded) ? seeded : 0;
    let persistArmed = false;
    let persistTimer = null;
    let lastPersistAt = 0;
    const PERSIST_THROTTLE_MS = 2000;

    // ">= 95" = "gelesen" (gleiche Schwelle wie Karte + Resume). Das "Gelesen"-
    // Flag hängt am furthest-read (maxReached), nicht an der Scroll-Position.
    // R2-G: Bar UND Flag zeigen jetzt beide den Max (Readwise-Verhalten,
    // Revision der R2-F-Positions-Bar). maxReached wächst beim Lesen monoton,
    // kann aber per "Als ungelesen markieren" (resetProgress) auf 0 fallen —
    // syncReadFlag folgt deshalb in BEIDE Richtungen, nicht mehr nur-monoton-an.
    const READ_COMPLETE_PERCENT = 95;
    const readFlag = document.getElementById('reader-read-flag');
    function syncReadFlag() {
        if (readFlag) readFlag.hidden = !(maxReached >= READ_COMPLETE_PERCENT);
    }
    syncReadFlag();

    function persistProgress(p, useKeepalive) {
        // Fire-and-forget über den globalen fetch-Wrapper (CSRF-Header
        // automatisch). Fehler still schlucken — der Reader bleibt ruhig,
        // kein Toast bei Progress-Save-Fail.
        fetch(`/api/conversions/${CONVERSION_ID}/progress`, {
            method: 'PATCH',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({percent: p}),
            keepalive: !!useKeepalive,
        }).catch(() => {});
    }

    // Inline timestamp+timer-Throttle (~2s) — Single-Call-Site, daher kein
    // _utils.js-Helper (feedback_helper_reuse_design_choice). Leading-edge
    // feuert sofort, trailing-edge-Timer garantiert dass der finale Wert landet.
    function schedulePersist(p) {
        const now = Date.now();
        const elapsed = now - lastPersistAt;
        if (elapsed >= PERSIST_THROTTLE_MS) {
            lastPersistAt = now;
            persistProgress(p, false);
        } else if (persistTimer === null) {
            persistTimer = setTimeout(() => {
                persistTimer = null;
                lastPersistAt = Date.now();
                persistProgress(maxReached, false);
            }, PERSIST_THROTTLE_MS - elapsed);
        }
    }

    function update() {
        const scrollTop = scroller.scrollTop;
        const scrollable = scroller.scrollHeight - scroller.clientHeight;
        if (scrollable < MIN_SCROLLABLE_PX) {
            wrapper.classList.add('reading-progress--hidden');
            fill.style.width = '0%';
            return;
        }
        wrapper.classList.remove('reading-progress--hidden');
        const percent = (scrollTop / scrollable) * 100;
        // R2-G: Die Bar zeigt furthest-read (max aus aktueller Position und
        // maxReached), nicht die reine Position — beim Hochscrollen bleibt sie
        // beim Max stehen, über den Max hinaus wächst sie mit (Readwise-Verhalten,
        // Revision der R2-F-Positions-Bar). maxReached ist aus PageData geseedet,
        // also steht die Bar schon vor dem rAF-Arming auf dem gespeicherten Max.
        fill.style.width = `${Math.min(100, Math.max(0, percent, maxReached))}%`;
        // Nur vorwärts und nur nach Settle: kurze Docs (oben gefiltert) können
        // den gespeicherten Wert nicht mit 0 clobbern.
        if (persistArmed && percent > maxReached) {
            maxReached = percent;
            schedulePersist(maxReached);
            syncReadFlag();
        }
    }

    scrollSource.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
    update();

    // Resume-on-Open nach Layout-Settle (rAF — Code-Blocks/Markdown sind
    // server-gerendert, Höhe steht nach dem ersten Frame). Nur mitten im Doc
    // (1 < gespeichert < 95): ≤1 ist sowieso oben, ≥95 als "gelesen" oben
    // öffnen statt ans Ende zu zwingen.
    requestAnimationFrame(() => {
        const scrollable = scroller.scrollHeight - scroller.clientHeight;
        if (maxReached > 1 && maxReached < READ_COMPLETE_PERCENT && scrollable >= MIN_SCROLLABLE_PX) {
            scroller.scrollTop = (maxReached / 100) * scrollable;
        }
        update();
        // Erst im Folge-Frame scharf schalten, damit das Scroll-Event des
        // Resume-Scrolls (oben bereits verarbeitet) keinen Persist auslöst.
        requestAnimationFrame(() => { persistArmed = true; });
    });

    // Flush bei Tab-Hide/Navigation: keepalive überlebt Tab-Close/Unload.
    document.addEventListener('visibilitychange', () => {
        if (document.hidden && maxReached > 0) {
            persistProgress(maxReached, true);
        }
    });

    // R2-G "Als ungelesen markieren": setzt den Fortschritt serverseitig auf
    // NULL (reset-Flag umgeht den Forward-Clamp) und zieht die lokale Anzeige
    // nach. Lebt in initReadingProgress, weil maxReached/update/syncReadFlag
    // Closure-State sind; window-Expose darum hier statt im Modul-Footer.
    function resetProgress() {
        fetch(`/api/conversions/${CONVERSION_ID}/progress`, {
            method: 'PATCH',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({reset: true}),
        }).then(r => {
            if (r.ok) {
                // Pending Trailing-Persist canceln, sonst feuert er gleich mit
                // dem (jetzt 0er) maxReached und schreibt 0.0 statt NULL zurück.
                if (persistTimer !== null) { clearTimeout(persistTimer); persistTimer = null; }
                maxReached = 0;
                // persistArmed kurz aus: das sofortige update() zeichnet die Bar
                // auf die aktuelle Position neu, soll sie aber NICHT als neuen
                // Fortschritt zurückpersistieren (sonst stünde der Server gleich
                // wieder auf currentPercent) — gleicher Self-Persist-Schutz wie
                // beim Resume-Scroll. Nächster echter Vorwärts-Scroll trackt wieder.
                persistArmed = false;
                update();        // Bar fällt auf max(currentPercent, 0)
                syncReadFlag();  // Label aus (maxReached < 95)
                requestAnimationFrame(() => { persistArmed = true; });
                showToast('Als ungelesen markiert.');
            } else {
                showToast(withServerSuffix('Zurücksetzen fehlgeschlagen. Verbindung prüfen und erneut versuchen.', r.status), { level: 'danger' });
            }
        }).catch(() => {
            showToast('Zurücksetzen fehlgeschlagen. Verbindung prüfen und erneut versuchen.', { level: 'danger' });
        });
    }
    window.resetProgress = resetProgress;
}

// --- Detail-Sidebar-Toggle (READER-MODE) ---
// Klappt die rechte Sidebar (Markierungen/Notion/Details) ein/aus.
// State in localStorage via loadViewState/saveViewState aus _utils.js.
const DETAIL_SIDEBAR_STATE_KEY = 'reader.detailSidebar';

function applyDetailSidebarState(collapsed) {
    document.body.classList.toggle('detail-sidebar--collapsed', collapsed);
    const btn = document.getElementById('detail-sidebar-toggle');
    if (btn) {
        btn.setAttribute('aria-expanded', String(!collapsed));
        const poly = btn.querySelector('polyline');
        // Chevron-Right (Default): "Sidebar nach rechts wegschieben" = einklappen.
        // Chevron-Left: "Sidebar nach links zurückholen" = ausklappen.
        if (poly) poly.setAttribute('points', collapsed ? '15 18 9 12 15 6' : '9 18 15 12 9 6');
    }
}

function initDetailSidebarToggle() {
    const btn = document.getElementById('detail-sidebar-toggle');
    if (!btn) return;
    const state = loadViewState(DETAIL_SIDEBAR_STATE_KEY, { collapsed: false });
    applyDetailSidebarState(!!state.collapsed);
    btn.addEventListener('click', () => {
        const nowCollapsed = !document.body.classList.contains('detail-sidebar--collapsed');
        applyDetailSidebarState(nowCollapsed);
        saveViewState(DETAIL_SIDEBAR_STATE_KEY, { collapsed: nowCollapsed });
    });
}

document.addEventListener('DOMContentLoaded', () => {
    setupAutoSaveTracking();
    initConversionTagPicker();
    initHighlights();
    initMarkOnMouseup();
    initReadingProgress();
    initDetailSidebarToggle();
    initFinishBackLink();
});

window.updateField = updateField;
window.setPlace = setPlace;
window.finishArchive = finishArchive;
window.copyFullContent = copyFullContent;
window.downloadContent = downloadContent;
window.storeForReuse = storeForReuse;
window.deleteConversion = deleteConversion;
window.toggleNotionPanel = toggleNotionPanel;
window.selectTarget = selectTarget;
window.sendToNotion = sendToNotion;
window.sendToKindle = sendToKindle;
