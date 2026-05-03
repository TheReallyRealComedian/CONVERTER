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

    function showToast(message) {
        const toast = document.createElement('div');
        toast.className = 'toast-notification';
        toast.textContent = message;
        document.body.appendChild(toast);
        setTimeout(() => toast.classList.add('show'), 10);
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 2000);
    }

    window.safeJSON = safeJSON;
    window.fallbackCopyText = fallbackCopyText;
    window.showAlert = showAlert;
    window.showToast = showToast;
})();
