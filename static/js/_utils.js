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

    /* Render a persistent alert banner inside containerEl. Replaces any prior
       content. Message is set via textContent (XSS-safe). No close button or
       auto-dismiss — caller clears the container or calls showAlert again. */
    function showAlert(containerEl, level, message) {
        if (!containerEl) return null;
        containerEl.innerHTML = '';
        const banner = document.createElement('div');
        banner.className = 'c-alert c-alert--' + level;
        banner.textContent = message;
        containerEl.appendChild(banner);
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
