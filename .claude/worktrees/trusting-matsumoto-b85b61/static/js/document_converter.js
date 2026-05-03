/* Document → Markdown converter page: drop zone, upload, save-to-library. */

let lastResult = null;

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('document_file');
const fileInfo = document.getElementById('file-info');
const fileName = document.getElementById('file-name');

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drop-zone-active');
});
dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drop-zone-active');
});
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drop-zone-active');
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        showFileInfo(e.dataTransfer.files[0]);
    }
});
fileInput.addEventListener('change', () => {
    if (fileInput.files.length) showFileInfo(fileInput.files[0]);
});
document.getElementById('clear-file').addEventListener('click', () => {
    fileInput.value = '';
    fileInfo.classList.add('hidden');
});

function showFileInfo(file) {
    fileName.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`;
    fileInfo.classList.remove('hidden');
}

document.getElementById('convert-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    if (!fileInput.files.length) return;

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('document_file', file);

    const btn = document.getElementById('convert-btn');
    btn.disabled = true;
    btn.textContent = 'Converting...';
    document.getElementById('result-area').classList.add('hidden');
    document.getElementById('alert-container').innerHTML = '';

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
        document.getElementById('result-area').classList.remove('hidden');
        document.getElementById('save-btn').textContent = 'Save to Library';
        document.getElementById('save-btn').disabled = false;
    } catch (err) {
        document.getElementById('alert-container').innerHTML =
            `<div class="c-alert c-alert--danger">${err.message}</div>`;
    } finally {
        btn.disabled = false;
        btn.textContent = 'Transform to Text';
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
}

async function saveToLibrary() {
    if (!lastResult) return;
    const btn = document.getElementById('save-btn');
    btn.disabled = true;
    btn.textContent = 'Saving...';

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
            btn.textContent = 'Saved!';
            btn.classList.add('saved');
        } else {
            throw new Error('Save failed');
        }
    } catch (err) {
        btn.textContent = 'Save to Library';
        btn.disabled = false;
        alert('Failed to save: ' + err.message);
    }
}

window.downloadResult = downloadResult;
window.saveToLibrary = saveToLibrary;
