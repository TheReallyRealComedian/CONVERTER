/* Document → Markdown converter page: drop zone, upload, save-to-library. */

let lastResult = null;

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('document_file');
const fileInfo = document.getElementById('file-info');
const fileName = document.getElementById('file-name');
const alertContainer = document.getElementById('alert-container');

const acceptedExtensions = (window.PageData && window.PageData.acceptedExtensions) || [];
const acceptedExtensionsLabel = 'PDF, DOCX, PPTX, EML, HTML, TXT, MD';
let warningTimer = null;

// DOC-WEB-ASYNC: a PDF is a JOB on the worker — the only container with the
// engine wiring (Docker socket for the mineru sibling container). It is
// submitted to the service and polled from here; every other format stays on
// the synchronous route (CPU-seconds, no container). Measured 2026-08-21:
// the single gunicorn worker serves NOTHING else while a synchronous
// conversion runs — only the job frees it.
const serviceUrl = (window.PageData && window.PageData.documentConversionsUrl)
    || '/api/document-conversions';
// Bumped per submit and per "clear file": a poll that outlives its run must
// never render a stale result over a newer state of the page.
let activeRun = 0;

function getExtension(filename) {
    const m = /\.([^.\\/]+)$/.exec(filename || '');
    return m ? m[1].toLowerCase() : '';
}

function isAcceptedFilename(filename) {
    if (!acceptedExtensions.length) return true;
    const ext = getExtension(filename);
    return acceptedExtensions.includes(ext);
}

function clearInvalidState() {
    dropZone.classList.remove('c-drop-zone--invalid');
    alertContainer.innerHTML = '';
}

function showWarningState() {
    dropZone.classList.add('c-drop-zone--warning');
    if (warningTimer) clearTimeout(warningTimer);
    warningTimer = setTimeout(() => {
        dropZone.classList.remove('c-drop-zone--warning');
        warningTimer = null;
    }, 2000);
}

function clearWarningState() {
    dropZone.classList.remove('c-drop-zone--warning');
    if (warningTimer) {
        clearTimeout(warningTimer);
        warningTimer = null;
    }
}

function rejectUnsupported() {
    showAlert(alertContainer, 'warning',
        'Dieser Dateityp wird nicht unterstützt. Erlaubt: ' + acceptedExtensionsLabel + '.');
    showWarningState();
}

dropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fileInput.click();
    }
});
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    // Browser security usually hides the file's MIME/name during dragover and
    // exposes only `kind === 'file'` plus a (sometimes generic) `.type`. So we
    // only flip into the warning tint when the type is present and clearly not
    // in our accept list — otherwise we wait for the actual drop to validate.
    let unsupported = false;
    const items = e.dataTransfer && e.dataTransfer.items;
    if (items && items.length === 1 && items[0].kind === 'file') {
        const t = (items[0].type || '').toLowerCase();
        if (t && t !== 'application/octet-stream') {
            const acceptedMimeFragments = ['pdf', 'word', 'officedocument', 'message/rfc822',
                'html', 'plain', 'markdown'];
            unsupported = !acceptedMimeFragments.some(frag => t.includes(frag));
        }
    }
    if (unsupported) {
        dropZone.classList.add('c-drop-zone--warning');
        dropZone.classList.remove('drop-zone-active');
    } else {
        clearWarningState();
        dropZone.classList.add('drop-zone-active');
    }
});
dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drop-zone-active');
    clearWarningState();
});
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drop-zone-active');
    if (!e.dataTransfer.files.length) return;
    const file = e.dataTransfer.files[0];
    if (!isAcceptedFilename(file.name)) {
        rejectUnsupported();
        return;
    }
    fileInput.files = e.dataTransfer.files;
    showFileInfo(file);
    clearInvalidState();
    clearWarningState();
});
fileInput.addEventListener('change', () => {
    if (!fileInput.files.length) return;
    const file = fileInput.files[0];
    if (!isAcceptedFilename(file.name)) {
        // User picked "All files" in the system picker and chose something the
        // accept-attribute would otherwise have hidden.
        fileInput.value = '';
        rejectUnsupported();
        return;
    }
    showFileInfo(file);
    clearInvalidState();
    clearWarningState();
});
document.getElementById('clear-file').addEventListener('click', () => {
    activeRun += 1;  // a job still polling for this file stops rendering
    fileInput.value = '';
    fileInfo.classList.add('hidden');
    document.getElementById('result-area').classList.add('hidden');
    document.getElementById('alert-container').innerHTML = '';
    clearWarningState();
    lastResult = null;
    resetSaveBtn();
});

function showFileInfo(file) {
    fileName.textContent = `${file.name} (${formatFileSize(file.size)})`;
    fileInfo.classList.remove('hidden');
}

function resetSaveBtn() {
    const btn = document.getElementById('save-btn');
    btn.disabled = false;
    btn.textContent = 'In Library speichern';
    btn.classList.remove('saved');
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function formatElapsed(ms) {
    const total = Math.floor(ms / 1000);
    return Math.floor(total / 60) + ':' + String(total % 60).padStart(2, '0');
}

// The service stores an RQ traceback TAIL as the error — its last line is the
// exception itself, the only line a user can do anything with.
function lastLine(text) {
    const lines = String(text || '').split('\n').map(l => l.trim()).filter(Boolean);
    return lines.length ? lines[lines.length - 1] : '';
}

async function readError(response) {
    const errData = await safeJSON(response);
    return new Error(errData.error || `Conversion failed (${response.status})`);
}

// Non-PDF: the synchronous route, unchanged since DOC-WEB.
async function convertSync(file) {
    const formData = new FormData();
    formData.append('document_file', file);
    const response = await fetch(window.PageData.transformDocumentUrl, {
        method: 'POST',
        body: formData
    });
    if (!response.ok) throw await readError(response);
    const data = await response.json();
    return {
        markdown: data.markdown || '',
        degradations: data.degradations || [],
        conversionId: null,
        deduped: false
    };
}

// PDF: submit to the service, then poll the same endpoint the service's own
// callers use until the job is ready or failed. The mode comes from Oli's
// service setting (no field sent — one switch for both entrances). Returns
// null when the run was cleared meanwhile.
async function convertViaService(file, runId) {
    const formData = new FormData();
    formData.append('file', file);
    const submit = await fetch(serviceUrl, {method: 'POST', body: formData});
    if (!submit.ok) throw await readError(submit);
    let data = await submit.json();
    const deduped = data.deduped === true;
    const pollUrl = serviceUrl + '/' + data.id;
    const startedAt = Date.now();
    while (data.status !== 'ready') {
        if (data.status === 'failed') {
            const detail = lastLine(data.error).slice(0, 200);
            throw new Error('Konvertierung fehlgeschlagen.' + (detail ? ' ' + detail : ''));
        }
        // 2 s while the run is young, 5 s once it is clearly a long one
        // (lokal ≈ 61 s model start + 2,5 s/Seite, cloud ≈ 15 s/Seite).
        await sleep(Date.now() - startedAt < 60000 ? 2000 : 5000);
        if (runId !== activeRun) return null;
        const poll = await fetch(pollUrl);
        if (!poll.ok) throw await readError(poll);
        data = await poll.json();
    }
    return {
        markdown: data.markdown || '',
        degradations: data.degradations || [],
        conversionId: data.id,
        deduped
    };
}

document.getElementById('convert-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    if (!fileInput.files.length) {
        showAlert(alertContainer, 'danger',
            'Bitte zuerst eine Datei auswählen oder per Drag & Drop hineinziehen.');
        dropZone.classList.add('c-drop-zone--invalid');
        dropZone.focus();
        return;
    }

    const file = fileInput.files[0];
    const isPdf = getExtension(file.name) === 'pdf';
    const runId = ++activeRun;

    const btn = document.getElementById('convert-btn');
    btn.disabled = true;
    btn.textContent = 'Wird umgewandelt …';
    dropZone.classList.add('c-drop-zone--loading');
    const resultArea = document.getElementById('result-area');
    resultArea.classList.add('hidden');
    document.getElementById('alert-container').innerHTML = '';
    resetSaveBtn();

    // A job takes minutes — an elapsed counter on the button says "still
    // working", not "stuck".
    const startedAt = Date.now();
    const ticker = isPdf ? setInterval(() => {
        btn.textContent = 'Wird umgewandelt … ' + formatElapsed(Date.now() - startedAt);
    }, 1000) : null;

    try {
        const result = isPdf
            ? await convertViaService(file, runId)
            : await convertSync(file);
        if (!result || runId !== activeRun) return;  // cleared meanwhile

        lastResult = {
            content: result.markdown,
            filename: file.name,
            mimetype: file.type,
            size: file.size,
            conversionId: result.conversionId
        };

        document.getElementById('result-content').textContent = result.markdown;
        renderDegradations(result.degradations);
        // Idempotency is the service's doing (same file, same mode, same
        // engine generation → the stored result, no fresh run). It must not
        // LOOK like a fresh run, so the result says so.
        const note = document.getElementById('result-note');
        note.textContent = result.deduped
            ? 'Diese Datei war schon umgewandelt – das gespeicherte Ergebnis wurde geladen.'
            : '';
        note.classList.toggle('hidden', !result.deduped);
        resultArea.classList.remove('hidden');
        resultArea.scrollIntoView({behavior: 'smooth', block: 'start'});
    } catch (err) {
        if (runId === activeRun) showAlert(alertContainer, 'danger', err.message);
    } finally {
        if (ticker) clearInterval(ticker);
        btn.disabled = false;
        btn.textContent = 'Dokument umwandeln';
        dropZone.classList.remove('c-drop-zone--loading');
    }
});

// Degradation notes from the server (DOC-WEB: what could not be converted
// cleanly reaches the user, not only the log). DOM nodes, no innerHTML —
// the messages quote raw tool output.
function renderDegradations(entries) {
    const box = document.getElementById('degradation-box');
    const list = document.getElementById('degradation-list');
    if (!box || !list) return;
    list.textContent = '';
    if (!entries.length) {
        box.classList.add('hidden');
        return;
    }
    entries.forEach((entry) => {
        const li = document.createElement('li');
        li.textContent = entry.message || entry.code || '';
        list.appendChild(li);
    });
    box.classList.remove('hidden');
}

function downloadResult() {
    if (!lastResult) return;
    const blob = new Blob([lastResult.content], {type: 'text/markdown'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    const stem = lastResult.filename.replace(/\.[^.]+$/, '');
    a.download = stem + '.md';
    a.click();
    URL.revokeObjectURL(a.href);
    showToast('✓ Markdown heruntergeladen');
}

async function saveToLibrary() {
    if (!lastResult) return;
    const btn = document.getElementById('save-btn');
    btn.disabled = true;
    btn.textContent = 'Speichert …';

    try {
        const stem = lastResult.filename.replace(/\.[^.]+$/, '');
        const ext = lastResult.filename.split('.').pop().toLowerCase();
        // A job-backed result (PDF) already IS a library row — the service
        // shelves it in the archive. "Speichern" moves that row into the
        // inbox (the place a saved conversion always landed) instead of
        // creating a second row with the same content.
        const response = lastResult.conversionId
            ? await fetch(`/api/conversions/${lastResult.conversionId}/place`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({place: 'inbox'})
            })
            : await fetch('/api/conversions', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    conversion_type: 'document_to_markdown',
                    title: stem,
                    content: lastResult.content,
                    source_filename: lastResult.filename,
                    source_mimetype: lastResult.mimetype,
                    source_size_bytes: lastResult.size,
                    metadata: {
                        file_extension: '.' + ext
                    }
                })
            });

        if (response.ok) {
            btn.textContent = '✓ Gespeichert';
            btn.classList.add('saved');
            return;
        }

        let serverError = null;
        try {
            const errData = await safeJSON(response);
            serverError = errData && errData.error;
        } catch (_) { /* fall back to generic message */ }

        const msg = serverError
            ? 'Speichern in die Library fehlgeschlagen: ' + serverError + '. Bitte erneut versuchen.'
            : 'Speichern in die Library fehlgeschlagen. Bitte erneut versuchen.';
        showAlert(alertContainer, 'danger', msg);
        resetSaveBtn();
    } catch (_err) {
        resetSaveBtn();
        showAlert(alertContainer, 'danger',
            'Speichern in die Library fehlgeschlagen. Bitte erneut versuchen.');
    }
}

window.downloadResult = downloadResult;
window.saveToLibrary = saveToLibrary;
