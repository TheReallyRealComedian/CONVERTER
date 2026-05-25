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

// --- Highlights (R1-B-A) ---

const HIGHLIGHT_CONTEXT_LEN = 32;
const HIGHLIGHT_EXACT_LIMIT = 5000;

function highlightReaderEl() { return document.querySelector('.reader-view'); }
function highlightCreateBtn() { return document.getElementById('highlight-create-btn'); }
function highlightActionPopover() { return document.getElementById('highlight-action-popover'); }
let activeHighlightId = null;
// Modul-Cache für die Sidebar — beide Render-Pfade (Reader-Apply + Sidebar) lesen daraus.
let highlightsState = [];
// IDs, deren Anchor sich nicht in einen DOM-Span wrappen ließ (Cross-Format).
let crossFormatHighlightIds = new Set();

function hideHighlightCreateBtn() {
    const btn = highlightCreateBtn();
    if (btn) btn.style.display = 'none';
}

function hideHighlightActionPopover() {
    const pop = highlightActionPopover();
    if (pop) pop.style.display = 'none';
    activeHighlightId = null;
}

function positionHighlightCreateBtn() {
    const reader = highlightReaderEl();
    const btn = highlightCreateBtn();
    if (!reader || !btn) return;
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) {
        hideHighlightCreateBtn();
        return;
    }
    const range = sel.getRangeAt(0);
    // Selection must start AND end inside the reader-view.
    if (!reader.contains(range.startContainer) || !reader.contains(range.endContainer)) {
        hideHighlightCreateBtn();
        return;
    }
    const rect = range.getBoundingClientRect();
    if (!rect || (rect.width === 0 && rect.height === 0)) {
        hideHighlightCreateBtn();
        return;
    }
    btn.style.display = 'inline-flex';
    btn.style.top = `${window.scrollY + rect.top - 40}px`;
    btn.style.left = `${window.scrollX + rect.left}px`;
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

function extractSelectionContext(selection) {
    const reader = highlightReaderEl();
    if (!reader || selection.rangeCount === 0) return {prefix: '', suffix: ''};
    const range = selection.getRangeAt(0);
    const exact = selection.toString();
    const walker = document.createTreeWalker(reader, NodeFilter.SHOW_TEXT);
    let preLen = 0;
    let node;
    while ((node = walker.nextNode())) {
        if (node === range.startContainer) {
            preLen += range.startOffset;
            break;
        }
        preLen += node.nodeValue.length;
    }
    const fullText = readerRawText(reader);
    const prefix = fullText.slice(Math.max(0, preLen - HIGHLIGHT_CONTEXT_LEN), preLen);
    const suffix = fullText.slice(preLen + exact.length, preLen + exact.length + HIGHLIGHT_CONTEXT_LEN);
    return {prefix, suffix};
}

async function saveCurrentSelection() {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) return;
    const exact = sel.toString();
    if (!exact.trim()) return;
    if (exact.length > HIGHLIGHT_EXACT_LIMIT) {
        showToast('Markierung zu lang. Bitte einen kürzeren Abschnitt wählen.', { level: 'danger' });
        return;
    }
    const {prefix, suffix} = extractSelectionContext(sel);
    sel.removeAllRanges();
    hideHighlightCreateBtn();

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
    if (applied) {
        showToast('Markiert.');
    } else {
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

function truncateForCard(str, max) {
    if (!str) return '';
    return str.length > max ? str.slice(0, max - 1) + '…' : str;
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
    list.innerHTML = '';
    highlightsState.forEach(h => {
        const card = document.createElement('div');
        card.className = 'highlight-card';
        card.dataset.highlightId = String(h.id);
        if (crossFormatHighlightIds.has(h.id)) {
            card.classList.add('highlight-card--cross-format');
            card.title = 'Markierung über Formatierungsgrenze, im Text nicht direkt anspringbar.';
        }
        const snippet = document.createElement('div');
        snippet.className = 'highlight-card__snippet';
        snippet.textContent = truncateForCard(h.exact, 80);
        card.appendChild(snippet);
        if (h.note) {
            const noteEl = document.createElement('div');
            noteEl.className = 'highlight-card__note';
            noteEl.textContent = truncateForCard(h.note, 60);
            card.appendChild(noteEl);
        }
        card.addEventListener('click', () => scrollToHighlight(h.id));
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

// Locate `exact` in the reader-view text. Returns the start-offset against
// reader.innerText (the same metric extractSelectionContext uses), so prefix
// and suffix from the stored highlight act as a tiebreaker for multi-match.
function locateHighlightOffset(reader, highlight) {
    const fullText = readerRawText(reader);
    const {exact, prefix, suffix} = highlight;
    if (!exact) return -1;
    const positions = [];
    let cursor = 0;
    while (cursor <= fullText.length) {
        const idx = fullText.indexOf(exact, cursor);
        if (idx === -1) break;
        positions.push(idx);
        cursor = idx + Math.max(1, exact.length);
    }
    if (positions.length === 0) return -1;
    if (positions.length === 1) return positions[0];

    let bestPos = -1;
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
            bestPos = pos;
        }
    }
    return bestPos >= 0 ? bestPos : positions[0];
}

// Walk text nodes inside `reader`, find the [start, end) offset range in
// reader.innerText terms, and return a Range object spanning that slice.
// Returns null if the slice crosses element boundaries we can't safely wrap
// (the cross-format case the sprint allows to fail gracefully).
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
    // For R1-B-A, only wrap when the range lives in a single text node —
    // multi-node ranges require splitting + multi-span wrapping that the
    // sprint explicitly allows to defer.
    if (startNode !== endNode) return null;
    const range = document.createRange();
    range.setStart(startNode, startNodeOffset);
    range.setEnd(endNode, endNodeOffset);
    return range;
}

function applyHighlight(highlight) {
    const reader = highlightReaderEl();
    if (!reader) return false;
    const startOffset = locateHighlightOffset(reader, highlight);
    if (startOffset < 0) return false;
    const endOffset = startOffset + highlight.exact.length;
    const range = rangeForOffsets(reader, startOffset, endOffset);
    if (!range) return false;
    const span = document.createElement('span');
    span.className = 'highlight';
    span.dataset.highlightId = String(highlight.id);
    try {
        range.surroundContents(span);
    } catch (_) {
        return false;
    }
    return true;
}

function showHighlightActionPopover(spanEl) {
    const pop = highlightActionPopover();
    if (!pop || !spanEl) return;
    const rect = spanEl.getBoundingClientRect();
    activeHighlightId = spanEl.dataset.highlightId;
    const idNum = parseInt(activeHighlightId, 10);
    const highlight = highlightsState.find(h => h.id === idNum);
    const input = document.getElementById('highlight-note-input');
    if (input) {
        input.value = highlight && highlight.note ? highlight.note : '';
    }
    pop.style.display = 'flex';
    pop.style.top = `${window.scrollY + rect.bottom + 6}px`;
    pop.style.left = `${window.scrollX + rect.left}px`;
    // Defer focus so the click that opened the popover doesn't immediately
    // bubble to the document-level "close-on-outside-click" handler.
    if (input) setTimeout(() => input.focus(), 0);
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
    const spans = document.querySelectorAll(`.highlight[data-highlight-id="${CSS.escape(id)}"]`);
    spans.forEach(span => {
        const parent = span.parentNode;
        while (span.firstChild) parent.insertBefore(span.firstChild, span);
        parent.removeChild(span);
        parent.normalize();
    });
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
    const createBtn = highlightCreateBtn();
    const popover = highlightActionPopover();
    const deleteBtn = document.getElementById('highlight-delete-btn');

    document.addEventListener('selectionchange', positionHighlightCreateBtn);
    if (createBtn) {
        // mousedown (not click) so the selection isn't cleared by the button getting focus first.
        createBtn.addEventListener('mousedown', evt => {
            evt.preventDefault();
            saveCurrentSelection();
        });
    }
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
    document.addEventListener('click', evt => {
        if (popover && !popover.contains(evt.target) && !evt.target.closest('span.highlight[data-highlight-id]')) {
            hideHighlightActionPopover();
        }
    });

    loadHighlights();
}

document.addEventListener('DOMContentLoaded', () => {
    setupAutoSaveTracking();
    setupTagChipSync();
    initHighlights();
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
