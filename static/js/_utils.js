/* Shared front-end helpers loaded by base.html for every page that extends it. */
(function () {
    'use strict';

    /* Safely parse a JSON response, surfacing redirect-to-login HTML as a
       readable error instead of a JSON parse failure. */
    async function safeJSON(response) {
        const contentType = response.headers.get('content-type') || '';
        if (!contentType.includes('application/json')) {
            if (response.redirected || contentType.includes('text/html')) {
                throw new Error('Session expired – please reload the page and log in again.');
            }
            throw new Error(`Server returned unexpected response (${response.status})`);
        }
        return response.json();
    }

    function fallbackCopyText(text) {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            return navigator.clipboard.writeText(text);
        }
        return new Promise((resolve, reject) => {
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.style.position = 'fixed';
            ta.style.opacity = '0';
            document.body.appendChild(ta);
            ta.select();
            try {
                document.execCommand('copy') ? resolve() : reject();
            } catch (e) { reject(e); }
            finally { document.body.removeChild(ta); }
        });
    }

    /* Render an alert banner inside containerEl. Replaces any prior content.
       Message is set via textContent (XSS-safe). Defaults: closable=true and
       autoDismissMs=6000 for non-danger levels (danger persists until closed). */
    function showAlert(containerEl, level, message, options) {
        if (!containerEl) return null;
        const opts = options || {};
        const closable = opts.closable !== false;
        const autoDismissMs = Object.prototype.hasOwnProperty.call(opts, 'autoDismissMs')
            ? opts.autoDismissMs
            : (level === 'danger' ? null : 6000);

        containerEl.innerHTML = '';
        const banner = document.createElement('div');
        banner.className = 'c-alert c-alert--' + level;

        const messageEl = document.createElement('span');
        messageEl.className = 'c-alert__message';
        messageEl.textContent = message;
        banner.appendChild(messageEl);

        let dismissTimer = null;

        if (closable) {
            const closeBtn = document.createElement('button');
            closeBtn.type = 'button';
            closeBtn.className = 'c-alert__close';
            closeBtn.setAttribute('aria-label', 'Meldung schließen');
            closeBtn.textContent = '×';
            closeBtn.addEventListener('click', () => {
                if (dismissTimer) clearTimeout(dismissTimer);
                banner.remove();
            });
            banner.appendChild(closeBtn);
        }

        containerEl.appendChild(banner);

        if (autoDismissMs) {
            dismissTimer = setTimeout(() => banner.remove(), autoDismissMs);
        }

        return banner;
    }

    /* Format a byte count with a sensible unit (B / KB / MB) and DE decimal
       comma. Examples: 222 → "222 B", 4731 → "4,6 KB", 1234567 → "1,2 MB". */
    function formatFileSize(bytes) {
        const n = Number(bytes) || 0;
        if (n < 1024) return n.toFixed(0) + ' B';
        if (n < 1024 * 1024) return (n / 1024).toFixed(1).replace('.', ',') + ' KB';
        return (n / (1024 * 1024)).toFixed(1).replace('.', ',') + ' MB';
    }

    /* Singleton toast: removes any visible toast before showing a new one.
       Message is set via textContent (XSS-safe). Defaults: level='success',
       durationMs=2500. */
    function showToast(message, options) {
        const opts = options || {};
        const level = opts.level || 'success';
        const durationMs = Object.prototype.hasOwnProperty.call(opts, 'durationMs')
            ? opts.durationMs
            : 2500;

        document.querySelectorAll('.toast-notification').forEach(el => el.remove());

        const toast = document.createElement('div');
        toast.className = 'toast-notification toast-notification--' + level;
        toast.textContent = message;
        document.body.appendChild(toast);

        // Force a reflow so the .show transition runs from the initial state.
        // eslint-disable-next-line no-unused-expressions
        toast.offsetHeight;
        toast.classList.add('show');

        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, durationMs);

        return toast;
    }

    window.safeJSON = safeJSON;
    window.fallbackCopyText = fallbackCopyText;
    window.showAlert = showAlert;
    window.showToast = showToast;
    window.formatFileSize = formatFileSize;
})();
