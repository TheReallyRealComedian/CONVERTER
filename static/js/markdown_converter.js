/* Markdown → PDF converter page: orientation toggle, reader mode, live preview,
   save-to-library, CSRF token refresh on submit. */

function setOrientation(value) {
    document.getElementById('orientation').value = value;
    document.getElementById('btn-portrait').classList.toggle('active', value === 'portrait');
    document.getElementById('btn-landscape').classList.toggle('active', value === 'landscape');

    // Update preview iframe aspect ratio
    const iframe = document.getElementById('preview-iframe');
    if (iframe) {
        iframe.style.aspectRatio = value === 'landscape' ? '297 / 210' : '210 / 297';
    }
}

/* Reader Mode Logic */
const READER_PREFS_KEY = 'readerPrefs';
const WIDTH_MAP = { narrow: '600px', medium: '800px', wide: '1000px', ultrawide: '80%' };

function getReaderPrefs() {
    return loadViewState(READER_PREFS_KEY, {});
}

function saveReaderPrefs(prefs) {
    saveViewState(READER_PREFS_KEY, prefs);
}

function toggleReaderMode() {
    const container = document.querySelector('.main-container');
    const isActive = container.classList.toggle('reader-mode');
    document.body.classList.toggle('reader-active', isActive);

    const prefs = getReaderPrefs();
    if (isActive) {
        if (prefs.dark) applyDarkMode(true);
        if (prefs.fontSize) container.style.setProperty('--reader-font-size', prefs.fontSize + 'px');
        if (prefs.width) applyWidth(prefs.width);
        updateWidthButtons(prefs.width || 'medium');
    } else {
        document.body.classList.remove('reader-active');
        document.documentElement.removeAttribute('data-theme');
    }
    prefs.modeOn = isActive;
    saveReaderPrefs(prefs);
    if (typeof window.renderIframe === 'function') window.renderIframe();
}

function toggleDarkMode() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    applyDarkMode(!isDark);
    const prefs = getReaderPrefs();
    prefs.dark = !isDark;
    saveReaderPrefs(prefs);
}

function applyDarkMode(on) {
    document.documentElement.setAttribute('data-theme', on ? 'dark' : 'light');
    const btn = document.getElementById('dark-mode-btn');
    if (btn) btn.textContent = on ? '\u2600\uFE0F' : '\uD83C\uDF19';
    if (typeof window.renderIframe === 'function') window.renderIframe();
}

function changeFontSize(delta) {
    const container = document.querySelector('.main-container');
    const current = parseInt(getComputedStyle(container).getPropertyValue('--reader-font-size')) || 18;
    const next = Math.min(32, Math.max(12, current + delta));
    container.style.setProperty('--reader-font-size', next + 'px');
    const prefs = getReaderPrefs();
    prefs.fontSize = next;
    saveReaderPrefs(prefs);
    if (typeof window.renderIframe === 'function') window.renderIframe();
}

function changeWidth(size) {
    applyWidth(size);
    const prefs = getReaderPrefs();
    prefs.width = size;
    saveReaderPrefs(prefs);
}

function applyWidth(size) {
    const px = WIDTH_MAP[size] || WIDTH_MAP.medium;
    document.querySelector('.main-container').style.setProperty('--reader-width', px);
    updateWidthButtons(size);
}

function updateWidthButtons(active) {
    ['narrow', 'medium', 'wide', 'ultrawide'].forEach(function(s) {
        const btn = document.getElementById('width-' + s);
        if (btn) btn.classList.toggle('active', s === active);
    });
}

const READER_ESC_FORM_TAGS = ['TEXTAREA', 'INPUT', 'SELECT', 'BUTTON'];

document.addEventListener('keydown', function(e) {
    if (e.key !== 'Escape') return;
    if (!document.querySelector('.main-container.reader-mode')) return;
    const active = document.activeElement;
    if (active && READER_ESC_FORM_TAGS.includes(active.tagName)) return;
    toggleReaderMode();
});

function getMarkdownAlertContainer() {
    return document.querySelector('.editor-pane .px-6.pt-4');
}

async function saveMarkdownToLibrary() {
    const content = document.getElementById('markdown_text').value.trim();
    const alertContainer = getMarkdownAlertContainer();
    if (!content) {
        showAlert(alertContainer, 'warning', 'Kein Inhalt zum Speichern vorhanden.');
        return;
    }

    const btn = document.getElementById('save-markdown-btn');
    btn.disabled = true;
    btn.textContent = 'Speichert …';

    try {
        const firstLine = content.split('\n')[0].replace(/^#+\s*/, '').trim();
        const title = firstLine.substring(0, 100) || 'Untitled Markdown';
        const theme = document.getElementById('style_theme').value;
        const filename = document.getElementById('output_filename').value || '';
        const fileInput = document.getElementById('markdown_file');
        const fromFile = fileInput.files && fileInput.files.length > 0;

        const response = await fetch('/api/conversions', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                conversion_type: 'markdown_input',
                title: title,
                content: content,
                source_filename: fromFile ? fileInput.files[0].name : null,
                metadata: {
                    style_theme: theme,
                    output_filename: filename,
                    from_file_upload: fromFile
                }
            })
        });

        if (response.ok) {
            btn.textContent = '✓ Gespeichert';
            btn.classList.add('saved');
            setTimeout(() => {
                btn.textContent = 'In Library speichern';
                btn.classList.remove('saved');
                btn.disabled = false;
            }, 2000);
        } else {
            let detail = '';
            try {
                const errData = await safeJSON(response);
                detail = errData && errData.error ? errData.error : '';
            } catch (e) {
                detail = e.message || '';
            }
            const msg = detail
                ? 'In Library speichern fehlgeschlagen: ' + detail
                : 'In Library speichern fehlgeschlagen. Verbindung prüfen und erneut versuchen.';
            throw new Error(msg);
        }
    } catch (err) {
        btn.textContent = 'In Library speichern';
        btn.disabled = false;
        showAlert(alertContainer, 'danger', err.message
            || 'In Library speichern fehlgeschlagen. Verbindung prüfen und erneut versuchen.');
    }
}

function attachAutoDismissToServerBanners() {
    const container = getMarkdownAlertContainer();
    if (!container) return;
    const banners = container.querySelectorAll('.c-alert:not(.c-alert--danger)');
    banners.forEach(banner => {
        setTimeout(() => banner.remove(), 6000);
    });
}

window.setOrientation = setOrientation;
window.toggleReaderMode = toggleReaderMode;
window.toggleDarkMode = toggleDarkMode;
window.changeFontSize = changeFontSize;
window.changeWidth = changeWidth;
window.saveMarkdownToLibrary = saveMarkdownToLibrary;

window.addEventListener('load', function() {
    if (typeof markdownit === 'undefined') {
        console.error('markdown-it library failed to load');
        const ifr = document.getElementById('preview-iframe');
        if (ifr) ifr.srcdoc = '<html><body><p style="color:#b45309;font-family:sans-serif;padding:1em;">Live-Vorschau nicht verfügbar — markdown-it konnte nicht geladen werden. PDF-Erstellung funktioniert trotzdem; Internet-Verbindung prüfen oder Seite neu laden.</p></body></html>';
        return;
    }

    const markdownInput = document.getElementById('markdown_text');
    const fileInput = document.getElementById('markdown_file');
    const styleSelector = document.getElementById('style_theme');
    const previewIframe = document.getElementById('preview-iframe');

    const md = markdownit({
        html: true,
        breaks: true,
        linkify: true,
    });

    let currentThemeCSS = '';

    function isDarkActive() {
        const root = document.documentElement;
        const readerActive = document.querySelector('.main-container.reader-mode') !== null;
        const readerTheme = root.getAttribute('data-theme');
        // Reader-Scope wins when reader-mode is active: an explicit Reader-Light
        // pick must override a global dark theme. Outside reader-mode, only the
        // global attribute is consulted.
        if (readerActive) {
            if (readerTheme === 'light') return false;
            if (readerTheme === 'dark') return true;
        }
        return root.getAttribute('data-global-theme') === 'dark';
    }

    function getReaderFontSize() {
        const container = document.querySelector('.main-container');
        if (!container) return null;
        const v = parseInt(getComputedStyle(container).getPropertyValue('--reader-font-size'));
        return Number.isFinite(v) ? v : null;
    }

    const DARK_OVERRIDES_CSS = `
        html, body { background: #1a1a2e !important; color: #d4d4d8 !important; }
        .pdf-page { background: transparent !important; color: #d4d4d8 !important; }
        .pdf-page h1, .pdf-page h2 { color: #f0f0f4 !important; border-bottom-color: rgba(255,255,255,0.15) !important; }
        .pdf-page h3, .pdf-page h4 { color: #e8e8ee !important; }
        .pdf-page h5, .pdf-page h6 { color: #b0b0be !important; }
        .pdf-page p, .pdf-page li, .pdf-page span, .pdf-page dd, .pdf-page dt { color: #d4d4d8 !important; }
        .pdf-page strong, .pdf-page b { color: #ececf0 !important; }
        .pdf-page em, .pdf-page i { color: #c8c8d4 !important; }
        .pdf-page a { color: #7aa2f7 !important; }
        .pdf-page code { background: rgba(255,255,255,0.10) !important; color: #e8b4b8 !important; border-color: rgba(255,255,255,0.12) !important; }
        .pdf-page pre { background: #16161e !important; border-color: rgba(255,255,255,0.10) !important; box-shadow: none !important; }
        .pdf-page pre code { background: transparent !important; color: #c9d1d9 !important; }
        .pdf-page blockquote { border-left-color: #7aa2f7 !important; background: rgba(255,255,255,0.04) !important; color: #b0b0c0 !important; }
        .pdf-page table { background: rgba(255,255,255,0.03) !important; border-color: rgba(255,255,255,0.10) !important; box-shadow: none !important; color: #d4d4d8 !important; }
        .pdf-page thead { background: rgba(255,255,255,0.06) !important; }
        .pdf-page th { color: #e0e0e8 !important; border-bottom-color: rgba(255,255,255,0.15) !important; }
        .pdf-page td { color: #d4d4d8 !important; border-bottom-color: rgba(255,255,255,0.08) !important; }
        .pdf-page tbody tr:hover { background-color: rgba(255,255,255,0.04) !important; }
        .pdf-page tbody tr:nth-child(even) { background-color: rgba(255,255,255,0.02) !important; }
        .pdf-page hr { background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent) !important; border-color: rgba(255,255,255,0.15) !important; }
    `;

    function buildIframeDoc(themeCSS, bodyHTML) {
        const dark = isDarkActive();
        const readerActive = document.querySelector('.main-container.reader-mode') !== null;
        const fontSize = getReaderFontSize();
        const baseBg = dark ? '#1a1a2e' : '#fff';
        const baseFg = dark ? '#d4d4d8' : 'inherit';
        const wrapperCSS = `html,body{margin:0;background:${baseBg};color:${baseFg};}` +
                           `.pdf-page{padding:2cm;min-height:calc(29.7cm - 4cm);box-sizing:border-box;}` +
                           `@media (max-width: 700px){.pdf-page{padding:1cm;}}`;
        const fontCSS = (readerActive && fontSize)
            ? `html{font-size:${fontSize}px;}body{font-size:${fontSize}px;line-height:1.7;}`
            : '';
        const darkCSS = dark ? DARK_OVERRIDES_CSS : '';
        return `<!DOCTYPE html><html${dark ? ' data-theme="dark"' : ''}><head><meta charset="UTF-8">` +
               `<style>${wrapperCSS}</style>` +
               `<style>${themeCSS}</style>` +
               `<style>${darkCSS}${fontCSS}</style>` +
               `</head><body><div class="pdf-page">${bodyHTML}</div></body></html>`;
    }

    function renderIframe() {
        const text = markdownInput.value;
        let body;
        if (!text.trim()) {
            body = '<p style="color:#888;font-style:italic;font-family:sans-serif;">Start typing to see the preview…</p>';
        } else {
            try { body = md.render(text); }
            catch (e) { body = '<p style="color:#b04040;font-family:sans-serif;">Preview error: ' + e.message + '</p>'; }
        }
        previewIframe.srcdoc = buildIframeDoc(currentThemeCSS, body);
    }
    window.renderIframe = renderIframe;

    const themeObserver = new MutationObserver(renderIframe);
    themeObserver.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ['data-theme', 'data-global-theme']
    });

    function updateStyle() {
        const theme = styleSelector.value;
        if (!theme || theme === 'none') {
            currentThemeCSS = '';
            renderIframe();
            return;
        }
        fetch(`/static/css/pdf_styles/${theme}.css`)
            .then(r => {
                if (!r.ok) throw new Error('HTTP ' + r.status);
                return r.text();
            })
            .then(css => { currentThemeCSS = css; renderIframe(); })
            .catch(err => {
                console.error('Error loading theme CSS:', err);
                currentThemeCSS = '';
                renderIframe();
                const themeLabel = theme.replace(/_/g, ' ');
                showAlert(getMarkdownAlertContainer(), 'warning',
                    `Vorlage „${themeLabel}“ konnte nicht geladen werden. Vorschau zeigt jetzt ohne diese Vorlage.`);
            });
    }

    const fileInfo = document.getElementById('markdown-file-info');
    const fileInfoLabel = document.getElementById('markdown-file-info-label');
    const fileInfoLive = document.getElementById('markdown-file-info-live');
    const fileInfoClear = document.getElementById('markdown-file-info-clear');

    function showFileInfo(file) {
        if (!fileInfo) return;
        const sizeText = formatFileSize(file.size);
        fileInfoLabel.textContent = `${file.name} (${sizeText})`;
        fileInfo.classList.remove('hidden');
        if (fileInfoLive) fileInfoLive.textContent = `${file.name}, ${sizeText}, ausgewählt.`;
    }

    function clearFileInfo() {
        if (!fileInfo) return;
        fileInfo.classList.add('hidden');
        fileInfoLabel.textContent = '';
        if (fileInfoLive) fileInfoLive.textContent = 'Datei abgewählt.';
    }

    function handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) {
            clearFileInfo();
            return;
        }
        showFileInfo(file);
        const reader = new FileReader();
        reader.onload = function(e) {
            markdownInput.value = e.target.result;
            renderIframe();
        };
        reader.readAsText(file);
    }

    if (fileInfoClear) {
        fileInfoClear.addEventListener('click', function() {
            fileInput.value = '';
            markdownInput.value = '';
            clearFileInfo();
            renderIframe();
        });
    }

    markdownInput.addEventListener('input', renderIframe);
    styleSelector.addEventListener('change', updateStyle);
    fileInput.addEventListener('change', handleFileSelect);

    const reuseContent = localStorage.getItem('libraryReuse');
    if (reuseContent) {
        markdownInput.value = reuseContent;
        localStorage.removeItem('libraryReuse');
    }

    // Rehydrate reader-mode state from localStorage. Width-buttons get their
    // active marker before the first reader-toggle so the toolbar shows the
    // correct selection from frame one.
    const initialPrefs = getReaderPrefs();
    updateWidthButtons(initialPrefs.width || 'medium');
    if (initialPrefs.modeOn === true) {
        toggleReaderMode();
    }

    updateStyle();
});

document.addEventListener('DOMContentLoaded', attachAutoDismissToServerBanners);

/* Refresh CSRF token right before submitting the PDF form
   so a long-idle page does not get rejected with HTTP 400. */
(function() {
    const form = document.getElementById('convert-form');
    if (!form) return;
    const submitBtn = form.querySelector('button[type="submit"]');
    const acceptedExtensions = (window.PageData && window.PageData.acceptedExtensions) || [];

    function fileExtensionAllowed(file) {
        if (!file || !acceptedExtensions.length) return true;
        const name = file.name || '';
        const dot = name.lastIndexOf('.');
        if (dot < 0) return false;
        return acceptedExtensions.includes(name.slice(dot + 1).toLowerCase());
    }

    form.addEventListener('submit', async function(event) {
        if (form.dataset.tokenRefreshed === '1') return;
        event.preventDefault();

        const fileInput = document.getElementById('markdown_file');
        const file = fileInput && fileInput.files && fileInput.files[0];
        const alertContainer = document.querySelector('.editor-pane .px-6.pt-4');
        const markdownInput = document.getElementById('markdown_text');
        const hasText = markdownInput && markdownInput.value.trim().length > 0;

        if (!file && !hasText) {
            showAlert(alertContainer, 'warning',
                'Bitte zuerst Markdown im Feld eintragen oder eine Datei hochladen.');
            if (markdownInput) markdownInput.focus();
            return;
        }

        if (file && !fileExtensionAllowed(file)) {
            showAlert(alertContainer, 'danger',
                'Dateiformat nicht unterstützt. Erlaubt: .md, .markdown.');
            return;
        }

        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = 'Wird vorbereitet …';
        }
        try {
            const resp = await fetch('/api/csrf-token', {
                credentials: 'same-origin',
                cache: 'no-store'
            });
            if (resp.ok) {
                const data = await resp.json();
                const field = form.querySelector('input[name="csrf_token"]');
                if (field && data.csrf_token) field.value = data.csrf_token;
            }
        } catch (err) {
            console.warn('CSRF token refresh failed, submitting with existing token:', err);
        } finally {
            form.dataset.tokenRefreshed = '1';
            // P13: do not restore the original label here. Submit-btn stays in
            // loading state ("Wird vorbereitet …") until page navigation, so
            // the user keeps a visible indicator while Playwright spins up.
            form.submit();
        }
    });
})();
