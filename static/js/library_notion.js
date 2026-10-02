/* Library detail view: the "An Notion senden" panel.

   Two halves. The FORM (new meeting, note, inbox) was moved verbatim out of
   library_detail.js (NOTION-MEETING-LINK, Phase 0) and is unchanged but for
   one line in sendToNotion (afterNotionSend). Below it, "Bestehendes Meeting":
   the candidate list, the send to an existing page and the remembered link.

   Loaded AFTER library_detail.js. From it this file uses, at call time only:
   CONVERSION_ID and conversionTagsState (top-level const/let of a classic
   script live in the shared global scope); from _utils.js: showAlert,
   showToast, safeJSON, formatDatetimeLocalNow. */

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
            afterNotionSend(data);
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

// --- NOTION-MEETING-LINK: "Bestehendes Meeting" ---
// New surface, new rules: everything below builds DOM nodes (createElement /
// textContent). Meeting titles come from Notion and the remembered link from
// the client-writable metadata bag — neither passes through innerHTML. The
// SERVER computes (order, preselection, day boundaries, every weekday/date/
// time text); nothing here does date or time-zone arithmetic.

const NOTION_SUBMIT_LABEL = 'An Notion senden';
const NOTION_NETWORK_MSG = 'Verbindung fehlgeschlagen. Netzwerk prüfen und erneut versuchen.';
const NOTION_LOAD_FAILED_MSG = 'Meetings konnten nicht geladen werden.';

// Audio rows usually belong to a meeting the calendar already created; every
// other type keeps the form as its default.
let notionWay = window.PageData.conversionType === 'audio_transcription' ? 'existing' : 'new';
let notionMeetings = null;      // last candidates answer; null = (re)load on show
let notionMeetingsRun = 0;      // stale guard: only the latest load may render
let notionMeetingsDay = null;   // the day last asked for (retry target)
let notionSelectedPage = null;
let notionSending = false;

function notionEl(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text != null) el.textContent = text;
    return el;
}

function existingWayActive() {
    return currentTarget === 'meetings' && notionWay === 'existing';
}

// "Verknüpft mit <Titel>, <Datum>" above the panel. The server only delivers
// a url that is https on a Notion host; the https check here is the belt.
function renderNotionLink(link) {
    const box = document.getElementById('notion-link-status');
    if (!box) return;
    box.replaceChildren();
    if (!link) {
        box.hidden = true;
        return;
    }
    let href = null;
    try {
        const parsed = new URL(link.url);
        if (parsed.protocol === 'https:') href = parsed.href;
    } catch (_) { /* no or malformed url → plain text */ }
    const title = link.title || 'Meeting';
    box.appendChild(document.createTextNode('Verknüpft mit '));
    if (href) {
        const a = notionEl('a', null, title);
        a.href = href;
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
        box.appendChild(a);
    } else {
        box.appendChild(notionEl('span', null, title));
    }
    if (link.date_text) box.appendChild(document.createTextNode(`, ${link.date_text}`));
    box.hidden = false;
}

// Called by the form's success branch (sendToNotion): a freshly created
// meeting is remembered server-side — show it, and let the list reload.
function afterNotionSend(data) {
    if (data && data.link) {
        renderNotionLink(data.link);
        notionMeetings = null;
    }
}

function updateNotionSubmit() {
    if (notionSending) return;
    const btn = document.getElementById('notion-submit-btn');
    btn.disabled = existingWayActive() && !notionSelectedPage;
}

// Shows the way switch (target "Meeting" only) and exactly one of the two
// bodies. Runs after every panel toggle, target click and way click.
function applyNotionWay() {
    const onMeetings = currentTarget === 'meetings';
    const existing = existingWayActive();
    const wayGroup = document.getElementById('notion-way-group');
    wayGroup.hidden = !onMeetings;
    wayGroup.querySelectorAll('button').forEach(btn => {
        const active = btn.dataset.way === notionWay;
        btn.classList.toggle('is-active', active);
        btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
    document.getElementById('notion-existing').hidden = !existing;
    document.getElementById('notion-new').hidden = existing;
    const panelOpen = !document.getElementById('notion-panel').classList.contains('hidden');
    if (existing && panelOpen && notionMeetings === null) loadNotionMeetings(notionMeetingsDay);
    updateNotionSubmit();
}

function toggleNotionDialog() {
    toggleNotionPanel();
    applyNotionWay();
}

function chooseNotionTarget(target) {
    selectTarget(target);
    applyNotionWay();
}

function chooseNotionWay(way) {
    if (way === notionWay) return;
    notionWay = way;
    clearNotionAlert();
    applyNotionWay();
}

function submitNotion() {
    if (existingWayActive()) sendToExistingMeeting(false);
    else sendToNotion();
}

function notionListNote(text) {
    return notionEl('p', 'notion-meeting-list__note', text);
}

function showNotionMeetingsError(message) {
    const list = document.getElementById('notion-meeting-list');
    const retry = notionEl('button', 'c-btn text-xs py-1 px-2', 'Erneut laden');
    retry.type = 'button';
    retry.addEventListener('click', () => loadNotionMeetings(notionMeetingsDay));
    list.replaceChildren(notionListNote(message || NOTION_LOAD_FAILED_MSG), retry);
}

// day: 'YYYY-MM-DD' or null (the server then opens on the remembered link's
// day, else on the recording's).
function loadNotionMeetings(day) {
    const run = ++notionMeetingsRun;
    notionMeetingsDay = day || null;
    notionSelectedPage = null;
    updateNotionSubmit();
    document.getElementById('notion-meeting-list').replaceChildren(notionListNote('Lade Meetings …'));
    const query = day ? `?day=${encodeURIComponent(day)}` : '';
    return fetch(`/api/conversions/${CONVERSION_ID}/notion-meetings${query}`).then(async r => {
        const data = await safeJSON(r);
        if (run !== notionMeetingsRun) return;
        if (!r.ok) {
            showNotionMeetingsError(data && data.error);
            return;
        }
        notionMeetings = data;
        notionMeetingsDay = data.day;
        renderNotionMeetings(data);
    }).catch(() => {
        if (run === notionMeetingsRun) showNotionMeetingsError(null);
    });
}

function notionMeetingEntry(meeting, data) {
    const label = notionEl('label', 'notion-meeting');
    if (meeting.day !== data.day) label.classList.add('notion-meeting--other-day');
    const radio = document.createElement('input');
    radio.type = 'radio';
    radio.name = 'notion-meeting';
    radio.value = meeting.page_id;
    radio.checked = meeting.page_id === data.preselected_page_id;
    radio.addEventListener('change', () => {
        notionSelectedPage = meeting.page_id;
        updateNotionSubmit();
    });
    label.appendChild(radio);

    const body = notionEl('span', 'notion-meeting__body');
    const when = [`${meeting.weekday}, ${meeting.date_text}`, meeting.time_text];
    if (meeting.length_text) when.push(meeting.length_text);
    body.appendChild(notionEl('span', 'notion-meeting__when', when.join(' · ')));
    body.appendChild(notionEl('span', 'notion-meeting__title', meeting.title));
    const meta = notionEl('span', 'notion-meeting__meta');
    if (meeting.type) meta.appendChild(notionEl('span', null, meeting.type));
    if (meeting.has_transcript) {
        meta.appendChild(notionEl('span', 'notion-badge notion-badge--transcript', 'hat schon Transkript'));
    }
    if (meeting.linked_here) {
        meta.appendChild(notionEl('span', 'notion-badge notion-badge--here', 'mit diesem Dokument verknüpft'));
    } else if (meeting.linked) {
        meta.appendChild(notionEl('span', 'notion-badge notion-badge--linked', 'schon mit CONVERTER verknüpft'));
    }
    if (meta.children.length) body.appendChild(meta);
    label.appendChild(body);
    return label;
}

function renderNotionMeetings(data) {
    const ref = data.reference;
    const parts = [`${ref.source === 'recorded_at' ? 'Aufnahme' : 'Upload'}: ${ref.text}`];
    if (ref.duration_text) parts.push(`Dauer ${ref.duration_text}`);
    document.getElementById('notion-reference').textContent = parts.join(' · ');
    const hint = document.getElementById('notion-reference-hint');
    hint.textContent = ref.hint || '';
    hint.hidden = !ref.hint;

    document.getElementById('notion-day-input').value = data.day;

    const list = document.getElementById('notion-meeting-list');
    const nodes = data.meetings.map(m => notionMeetingEntry(m, data));
    if (!nodes.length) nodes.push(notionListNote('Keine Meetings an diesem Tag und den Nachbartagen.'));
    if (data.truncated) nodes.push(notionListNote('Die Liste ist gekürzt.'));
    list.replaceChildren(...nodes);

    notionSelectedPage = data.preselected_page_id || null;
    updateNotionSubmit();
}

// Transcript → the chosen page. The request names the page and its day; the
// server builds the payload (the transcript comes from the stored row).
function sendToExistingMeeting(replace) {
    const meeting = notionMeetings
        && notionMeetings.meetings.find(m => m.page_id === notionSelectedPage);
    if (!meeting || notionSending) return;
    clearNotionAlert();
    const btn = document.getElementById('notion-submit-btn');
    notionSending = true;
    btn.disabled = true;
    btn.textContent = 'Sende …';
    const done = () => {
        notionSending = false;
        btn.textContent = NOTION_SUBMIT_LABEL;
        updateNotionSubmit();
    };

    fetch(`/api/conversions/${CONVERSION_ID}/send-to-notion`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            target: 'meetings', page_id: meeting.page_id, day: meeting.day,
            replace_transcript: replace === true,
        })
    })
    .then(async r => ({status: r.status, data: await safeJSON(r)}))
    .then(({status, data}) => {
        done();
        if (status === 409 && data.code === 'transcript_exists') {
            // Overwrite only after asking (native confirm, as for delete).
            const question = data.confirm || 'Das Meeting hat schon ein Transkript. Überschreiben?';
            if (confirm(question)) sendToExistingMeeting(true);
            return;
        }
        if (status >= 400) {
            showAlert(notionAlertContainer(), 'danger',
                data.error || 'Senden an Notion fehlgeschlagen. Später erneut versuchen.');
            return;
        }
        showToast('An Notion gesendet');
        if (data.link) {
            renderNotionLink(data.link);
        } else {
            showAlert(notionAlertContainer(), 'warning',
                'Gesendet. Die Verknüpfung konnte hier nicht gespeichert werden.');
        }
        // The list shows the new state (transcript, link) — same day.
        loadNotionMeetings(notionMeetings.day);
    })
    .catch(() => {
        done();
        showAlert(notionAlertContainer(), 'danger', NOTION_NETWORK_MSG);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    renderNotionLink(window.PageData.notionLink);
    document.querySelectorAll('#notion-way-group button').forEach(btn => {
        btn.addEventListener('click', () => chooseNotionWay(btn.dataset.way));
    });
    document.getElementById('notion-day-prev').addEventListener('click', () => {
        if (notionMeetings) loadNotionMeetings(notionMeetings.prev_day);
    });
    document.getElementById('notion-day-next').addEventListener('click', () => {
        if (notionMeetings) loadNotionMeetings(notionMeetings.next_day);
    });
    document.getElementById('notion-day-input').addEventListener('change', (e) => {
        // An emptied date field is not a day — keep the list as it is.
        if (e.target.value) loadNotionMeetings(e.target.value);
        else if (notionMeetings) e.target.value = notionMeetings.day;
    });
});

window.toggleNotionPanel = toggleNotionPanel;
window.selectTarget = selectTarget;
window.sendToNotion = sendToNotion;
window.toggleNotionDialog = toggleNotionDialog;
window.chooseNotionTarget = chooseNotionTarget;
window.submitNotion = submitNotion;
