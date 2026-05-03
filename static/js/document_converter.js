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
    const formData = new FormData();
    formData.append('document_file', file);

    const btn = document.getElementById('convert-btn');
    btn.disabled = true;
    btn.textContent = 'Wird umgewandelt …';
    const resultArea = document.getElementById('result-area');
    resultArea.classList.add('hidden');
    document.getElementById('alert-container').innerHTML = '';
    resetSaveBtn();

    try {
        const response = await fetch(window.PageData.transformDocumentUrl, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errData = await safeJSON(response);
            throw new Error(errData.error || `Conversion failed (${response.status})`);
        }

        const text = await response.text();
        lastResult = {
            content: text,
            filename: file.name,
            mimetype: file.type,
            size: file.size
        };

        document.getElementById('result-content').textContent = text;
        resultArea.classList.remove('hidden');
        resultArea.scrollIntoView({behavior: 'smooth', block: 'start'});
    } catch (err) {
        showAlert(alertContainer, 'danger', err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Dokument umwandeln';
    }
});

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
        const response = await fetch('/api/conversions', {
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
