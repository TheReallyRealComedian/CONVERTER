/* Layout-level wiring shared by every page that extends base.html:
   sidebar toggle (mobile), pane resizer (split-pane templates), theme toggle. */

/* Sidebar toggle for mobile */
document.getElementById('sidebar-toggle')?.addEventListener('click', function() {
    document.getElementById('sidebar').classList.toggle('-translate-x-full');
    document.getElementById('sidebar-overlay').classList.toggle('hidden');
});
document.getElementById('sidebar-overlay')?.addEventListener('click', function() {
    document.getElementById('sidebar').classList.add('-translate-x-full');
    document.getElementById('sidebar-overlay').classList.add('hidden');
});

/**
 * Pane Resizing Logic
 */
document.addEventListener('DOMContentLoaded', function() {
    const editorPane = document.querySelector('.editor-pane');
    const previewPane = document.querySelector('.preview-pane');

    if (editorPane && previewPane) {
        const resizer = document.createElement('div');
        resizer.className = 'resizer';
        editorPane.parentNode.insertBefore(resizer, previewPane);

        let isResizing = false;

        resizer.addEventListener('mousedown', (e) => {
            e.preventDefault();
            isResizing = true;
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', stopResize);
        });

        function handleMouseMove(e) {
            if (!isResizing) return;
            const container = editorPane.parentNode;
            const containerRect = container.getBoundingClientRect();
            const newEditorWidth = e.clientX - containerRect.left;
            const totalWidth = containerRect.width;
            editorPane.style.flexBasis = `${(newEditorWidth / totalWidth) * 100}%`;
            previewPane.style.flexBasis = `${((totalWidth - newEditorWidth) / totalWidth) * 100}%`;
        }

        function stopResize() {
            isResizing = false;
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', stopResize);
        }
    }
});

/* Theme toggle */
(function() {
    var btn = document.getElementById('theme-toggle');
    var sunIcon = document.getElementById('theme-icon-sun');
    var moonIcon = document.getElementById('theme-icon-moon');

    function isDark() {
        return document.documentElement.getAttribute('data-global-theme') === 'dark';
    }

    function updateIcons() {
        if (isDark()) {
            sunIcon.classList.remove('hidden');
            moonIcon.classList.add('hidden');
        } else {
            sunIcon.classList.add('hidden');
            moonIcon.classList.remove('hidden');
        }
    }

    function setTheme(dark) {
        document.documentElement.classList.add('theme-transitioning');
        if (dark) {
            document.documentElement.setAttribute('data-global-theme', 'dark');
        } else {
            document.documentElement.removeAttribute('data-global-theme');
        }
        localStorage.setItem('globalTheme', dark ? 'dark' : 'light');
        updateIcons();
        setTimeout(function() {
            document.documentElement.classList.remove('theme-transitioning');
        }, 350);
    }

    updateIcons();
    btn?.addEventListener('click', function() { setTheme(!isDark()); });

    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
        if (!localStorage.getItem('globalTheme')) setTheme(e.matches);
    });
})();
