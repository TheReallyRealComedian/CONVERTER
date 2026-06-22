/* Shared reader settings (READER-ADJ): text-size + column-width controls behind
   an on-demand "Aa" popover, used by both the markdown-converter preview and the
   library reading view. Extracted from markdown_converter.js — same readerPrefs
   schema (loadViewState/saveViewState), so the two readers stay in sync.

   The controls live in a small popover that opens on click of a discrete corner
   trigger and dismisses on outside-click + Esc — never a persistent bar over the
   text (the anti-pattern this replaces). The target element (which receives the
   --reader-width / --reader-font-size custom properties) is a parameter, so the
   same code drives .main-container (markdown) and #content-body (library). */
(function () {
    'use strict';

    const READER_PREFS_KEY = 'readerPrefs';
    const WIDTH_MAP = { narrow: '600px', medium: '800px', wide: '1000px', ultrawide: '80%' };
    const WIDTH_KEYS = ['narrow', 'medium', 'wide', 'ultrawide'];
    const FONT_MIN = 12, FONT_MAX = 32, FONT_DEFAULT = 18;

    function getReaderPrefs() {
        return loadViewState(READER_PREFS_KEY, {});
    }
    function saveReaderPrefs(prefs) {
        saveViewState(READER_PREFS_KEY, prefs);
    }

    /* Build a reader-settings controller bound to one consumer.
       opts.target   — element that receives --reader-width / --reader-font-size
       opts.trigger  — the "Aa" button (gated visible only in reader-mode)
       opts.popover  — the popover element (JS toggles the .is-open class)
       opts.onChange — called after a size change (markdown re-renders its iframe)
       opts.onDark   — optional; wires the [data-reader-dark] button if present
       opts.onExit   — optional; wires the [data-reader-exit] button if present */
    function createReaderSettings(opts) {
        opts = opts || {};
        const target = opts.target || null;
        const trigger = opts.trigger || null;
        const popover = opts.popover || null;
        const onChange = typeof opts.onChange === 'function' ? opts.onChange : function () {};
        const onDark = typeof opts.onDark === 'function' ? opts.onDark : null;
        const onExit = typeof opts.onExit === 'function' ? opts.onExit : null;

        function updateWidthButtons(active) {
            if (!popover) return;
            WIDTH_KEYS.forEach(function (s) {
                const btn = popover.querySelector('[data-reader-width="' + s + '"]');
                if (btn) btn.classList.toggle('active', s === active);
            });
        }

        function applyWidth(size) {
            const px = WIDTH_MAP[size] || WIDTH_MAP.medium;
            if (target) target.style.setProperty('--reader-width', px);
            updateWidthButtons(size);
        }

        function changeWidth(size) {
            applyWidth(size);
            const prefs = getReaderPrefs();
            prefs.width = size;
            saveReaderPrefs(prefs);
        }

        function changeFontSize(delta) {
            if (!target) return;
            const current = parseInt(getComputedStyle(target).getPropertyValue('--reader-font-size')) || FONT_DEFAULT;
            const next = Math.min(FONT_MAX, Math.max(FONT_MIN, current + delta));
            target.style.setProperty('--reader-font-size', next + 'px');
            const prefs = getReaderPrefs();
            prefs.fontSize = next;
            saveReaderPrefs(prefs);
            onChange();
        }

        /* Push the persisted width + font-size onto the target. Called when a
           reader-mode is entered so the saved look applies from frame one. */
        function applyPrefs(prefs) {
            prefs = prefs || getReaderPrefs();
            if (target && prefs.fontSize) {
                target.style.setProperty('--reader-font-size', prefs.fontSize + 'px');
            }
            applyWidth(prefs.width || 'medium');
        }

        function isOpen() {
            return !!popover && popover.classList.contains('is-open');
        }
        function openPopover() {
            if (!popover) return;
            popover.classList.add('is-open');
            if (trigger) trigger.setAttribute('aria-expanded', 'true');
        }
        function closePopover() {
            if (!popover) return;
            popover.classList.remove('is-open');
            if (trigger) trigger.setAttribute('aria-expanded', 'false');
        }
        function togglePopover() {
            isOpen() ? closePopover() : openPopover();
        }

        /* Esc consumed by the popover (close it) takes priority over the
           consumer's reader-mode Esc-to-exit. Returns true when it closed an
           open popover, so the caller's handler can bail. */
        function handleEscape() {
            if (isOpen()) { closePopover(); return true; }
            return false;
        }

        if (trigger) {
            trigger.addEventListener('click', function () { togglePopover(); });
        }
        if (popover) {
            popover.querySelectorAll('[data-reader-font]').forEach(function (btn) {
                btn.addEventListener('click', function () {
                    changeFontSize(parseInt(btn.getAttribute('data-reader-font'), 10));
                });
            });
            popover.querySelectorAll('[data-reader-width]').forEach(function (btn) {
                btn.addEventListener('click', function () {
                    changeWidth(btn.getAttribute('data-reader-width'));
                });
            });
            const darkBtn = popover.querySelector('[data-reader-dark]');
            if (darkBtn) {
                if (onDark) darkBtn.addEventListener('click', function () { onDark(darkBtn); });
                else darkBtn.remove();
            }
            const exitBtn = popover.querySelector('[data-reader-exit]');
            if (exitBtn) {
                if (onExit) exitBtn.addEventListener('click', function () { closePopover(); onExit(); });
                else exitBtn.remove();
            }
        }

        // Outside-click dismiss. The trigger.contains() guard lets the trigger's
        // own toggle run without this handler immediately re-closing the popover.
        document.addEventListener('click', function (evt) {
            if (!isOpen()) return;
            if (popover.contains(evt.target)) return;
            if (trigger && trigger.contains(evt.target)) return;
            closePopover();
        });

        return {
            changeFontSize: changeFontSize,
            changeWidth: changeWidth,
            applyWidth: applyWidth,
            updateWidthButtons: updateWidthButtons,
            applyPrefs: applyPrefs,
            openPopover: openPopover,
            closePopover: closePopover,
            togglePopover: togglePopover,
            isOpen: isOpen,
            handleEscape: handleEscape,
        };
    }

    window.ReaderSettings = {
        create: createReaderSettings,
        getPrefs: getReaderPrefs,
        savePrefs: saveReaderPrefs,
        WIDTH_MAP: WIDTH_MAP,
    };
})();
