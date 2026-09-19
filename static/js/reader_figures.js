/* Reader figures (RICH-MEDIA): Mermaid fences rendered client-side.
 *
 * Inline SVG and images arrive finished from the server (one renderer for
 * reader, PDF, EPUB). Mermaid is the one figure kind that renders HERE — and
 * the one constraint that shapes everything below is the highlight anchor:
 *
 *   A highlight is a text-quote selector over `readerRawText` = the nodeValue
 *   of EVERY text node under .reader-view — hidden ones included, the
 *   TreeWalker knows no visibility (library_detail.js). Whatever this file
 *   puts into the reader's light DOM as TEXT moves that coordinate system.
 *
 * So (gesperrte Entscheidung 4):
 *   - the fence's <pre><code class="language-mermaid"> STAYS in the DOM; it is
 *     hidden, never replaced — its text nodes keep their place;
 *   - the rendered diagram AND all of this file's UI text (toggle label,
 *     error hint) live in a SHADOW ROOT on an empty host element next to the
 *     <pre>. A TreeWalker does not descend into shadow trees and a selection
 *     does not reach into them: the diagram's own label text (Mermaid writes
 *     plenty) never enters readerRawText. Rendering a diagram therefore leaves
 *     readerRawText byte-equal — measured by scripts/smoke_reader_media.py.
 *
 * Existing highlights can sit INSIDE a fence's source (Oli marked whole
 * diagrams on #133 when they were still code). Hiding the source hides them,
 * so: the host gets a marker while its source holds a highlight, the shadow
 * UI offers "Quelltext zeigen", and scrollToHighlight reveals the source
 * first (ReaderFigures.revealSourceContaining).
 *
 * Mermaid: same CDN and major as the converter page, but PINNED to an exact
 * version with SRI (the converter's `@10` floats), loaded only when a document
 * has a fence and one comes near the viewport; securityLevel 'strict' — on the
 * converter page the user is the author, in the reader it is the agent.
 */
(function () {
    'use strict';

    const MERMAID_SRC = 'https://cdn.jsdelivr.net/npm/mermaid@10.9.8/dist/mermaid.min.js';
    const MERMAID_SRI = 'sha384-N3QqR/7q+xm3BGX+CBbNI8AUmRRqcsDzToy+0z1NLDI0QmTKW8zvwLvqulJgk3dP';
    const SOURCE_HIDDEN_CLASS = 'mermaid-source--hidden';
    const HOST_MARKED_CLASS = 'reader-mermaid--marked';

    // Shadow-tree styles. Document CSS does not cross the boundary; the --nm-*
    // custom properties do (they inherit), so the tokens work in both themes.
    const SHADOW_CSS = `
        :host { display: block; margin: 0 0 1em; }
        .figure { overflow-x: auto; text-align: center; }
        .figure svg { max-width: 100%; height: auto; }
        .hint { margin: 0 0 0.5em; font: 0.875rem/1.4 var(--nm-font, sans-serif);
                color: var(--nm-text-muted, #777); font-style: italic; }
        .toggle { margin-top: 0.25em; padding: 0.15em 0.6em; border: none; cursor: pointer;
                  font: 0.8125rem/1.4 var(--nm-font, sans-serif); border-radius: 999px;
                  color: var(--nm-text-secondary, #555); background: transparent;
                  text-decoration: underline; }
        .toggle:focus-visible { outline: 2px solid var(--nm-accent, #4a7); outline-offset: 2px; }
        [hidden] { display: none !important; }
    `;

    const entries = [];        // one per fence
    let mermaidLoading = null;  // Promise, set on first need
    let renderSeq = 0;

    function isDark() {
        return document.documentElement.getAttribute('data-global-theme') === 'dark';
    }

    function loadMermaid() {
        if (mermaidLoading) return mermaidLoading;
        mermaidLoading = new Promise((resolve, reject) => {
            if (window.mermaid) { resolve(window.mermaid); return; }
            const script = document.createElement('script');
            script.src = MERMAID_SRC;
            script.integrity = MERMAID_SRI;
            script.crossOrigin = 'anonymous';
            script.onload = () => (window.mermaid ? resolve(window.mermaid)
                                                  : reject(new Error('mermaid global missing')));
            script.onerror = () => reject(new Error('mermaid script failed to load'));
            document.head.appendChild(script);
        });
        return mermaidLoading;
    }

    function configure(mermaid) {
        mermaid.initialize({
            startOnLoad: false,
            theme: isDark() ? 'dark' : 'default',
            securityLevel: 'strict',
        });
    }

    function setSourceVisible(entry, visible) {
        entry.pre.classList.toggle(SOURCE_HIDDEN_CLASS, !visible);
        entry.toggle.textContent = visible ? 'Quelltext verbergen' : 'Quelltext zeigen';
        entry.toggle.setAttribute('aria-expanded', visible ? 'true' : 'false');
    }

    function refreshMark(entry) {
        const marked = !!entry.pre.querySelector('span.highlight[data-highlight-id]');
        entry.host.classList.toggle(HOST_MARKED_CLASS, marked);
    }

    function showFailure(entry, message) {
        // UI text, never agent text → textContent. The source stays visible,
        // the rest of the document is untouched.
        entry.figure.replaceChildren();
        entry.hint.textContent = message;
        entry.hint.hidden = false;
        entry.toggle.hidden = true;
        entry.pre.classList.remove(SOURCE_HIDDEN_CLASS);
        entry.host.dataset.mermaidState = 'failed';
    }

    async function renderEntry(entry, isRerender) {
        entry.host.dataset.mermaidState = 'rendering';
        let mermaid;
        try {
            mermaid = await loadMermaid();
        } catch (_e) {
            showFailure(entry, 'Diagramm-Bibliothek nicht erreichbar. Der Quelltext steht darunter.');
            return;
        }
        const source = entry.code.textContent;
        try {
            configure(mermaid);
            // parse() first: a syntax error throws here and render() never
            // gets to park its "Syntax error" bomb graphic in <body>.
            await mermaid.parse(source);
            const { svg } = await mermaid.render(`reader-mermaid-${++renderSeq}`, source);
            // The ONE innerHTML sink of the reader's figure code (CARD-MD
            // doctrine is DOM nodes, not innerHTML — this is the argued
            // exception): the string comes from mermaid.js in 'strict' mode,
            // not from the document, and it lands in a shadow root.
            entry.figure.innerHTML = svg;
        } catch (_e) {
            showFailure(entry, 'Dieses Diagramm lässt sich nicht darstellen (Syntaxfehler). '
                + 'Der Quelltext steht darunter.');
            return;
        }
        entry.hint.hidden = true;
        entry.toggle.hidden = false;
        // A re-render (theme switch) keeps a source the reader has opened.
        if (!isRerender) setSourceVisible(entry, false);
        entry.host.dataset.mermaidState = 'rendered';
        refreshMark(entry);
    }

    function buildEntry(code) {
        const pre = code.closest('pre');
        if (!pre || pre.dataset.mermaidBound) return null;
        pre.dataset.mermaidBound = '1';

        // An EMPTY light-DOM element: no text node enters the reader.
        const host = document.createElement('div');
        host.className = 'reader-mermaid';
        host.dataset.mermaidState = 'pending';
        const shadow = host.attachShadow({ mode: 'open' });

        const style = document.createElement('style');
        style.textContent = SHADOW_CSS;
        const hint = document.createElement('p');
        hint.className = 'hint';
        hint.hidden = true;
        const figure = document.createElement('div');
        figure.className = 'figure';
        const toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'toggle';
        toggle.hidden = true;
        shadow.append(style, hint, figure, toggle);

        const entry = { pre, code, host, hint, figure, toggle };
        toggle.addEventListener('click', () => {
            setSourceVisible(entry, pre.classList.contains(SOURCE_HIDDEN_CLASS));
        });
        // Highlights come and go inside the source (async load, create,
        // delete) — watch the <pre> instead of hooking every call site.
        new MutationObserver(() => refreshMark(entry))
            .observe(pre, { childList: true, subtree: true });

        pre.before(host);
        return entry;
    }

    function init() {
        const reader = document.querySelector('.reader-view');
        if (!reader) return;
        reader.querySelectorAll('pre > code.language-mermaid').forEach(code => {
            const entry = buildEntry(code);
            if (entry) entries.push(entry);
        });
        if (!entries.length) return;  // no fence → no script, no observer

        if (!('IntersectionObserver' in window)) {
            entries.forEach(entry => renderEntry(entry, false));
        } else {
            const observer = new IntersectionObserver((seen, obs) => {
                seen.forEach(item => {
                    if (!item.isIntersecting) return;
                    obs.unobserve(item.target);
                    const entry = entries.find(e => e.pre === item.target);
                    if (entry) renderEntry(entry, false);
                });
            }, { rootMargin: '300px 0px' });
            // The <pre> is the target: it has an area while the host is still empty.
            entries.forEach(entry => observer.observe(entry.pre));
        }

        // Theme follows data-global-theme (pattern: mermaid_converter.js /
        // markdown_converter.js): re-render what is already rendered.
        new MutationObserver(() => {
            entries.forEach(entry => {
                if (entry.host.dataset.mermaidState === 'rendered') renderEntry(entry, true);
            });
        }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-global-theme'] });
    }

    // For scrollToHighlight: a span inside a hidden source cannot be scrolled
    // to. Returns true if a source was opened for it.
    function revealSourceContaining(node) {
        const entry = entries.find(e => e.pre.contains(node));
        if (!entry || !entry.pre.classList.contains(SOURCE_HIDDEN_CLASS)) return false;
        setSourceVisible(entry, true);
        return true;
    }

    window.ReaderFigures = { init, revealSourceContaining };
    document.addEventListener('DOMContentLoaded', init);
})();
