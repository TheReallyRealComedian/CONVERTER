/* Mermaid diagram renderer: live preview with sanitization. */
document.addEventListener('DOMContentLoaded', function() {
    const isDark = document.documentElement.getAttribute('data-global-theme') === 'dark';
    mermaid.initialize({
        startOnLoad: false,
        theme: isDark ? 'dark' : 'default',
        securityLevel: 'loose'
    });

    const mermaidCodeInput = document.getElementById('mermaid_code');
    const previewContainer = document.getElementById('mermaid-preview');
    const errorContainer = document.getElementById('error-container');

    function sanitizeMermaidCode(code) {
        let sanitizedCode = code;

        sanitizedCode = sanitizedCode.replace(
            /(subgraph\s+\w+\[)([^"\]\n][^\]\n]*)(\])/g,
            (match, start, content, end) => {
                if (/[()\[\]{},]/.test(content)) {
                    return `${start}"${content}"${end}`;
                }
                return match;
            }
        );

        sanitizedCode = sanitizedCode.replace(
            /(--(?:>|o|x)?|==(?:>)?|-?-\.(?:->)?|-?-o(?:->)?)\s*\|([^"|\n][^|\n]*?)(\|)/g,
             (match, arrow, content, endPipe) => {
                if (/[()\[\]{},]/.test(content)) {
                    return `${arrow}|"${content}"${endPipe}`;
                }
                return match;
            }
        );

        sanitizedCode = sanitizedCode.replace(
            /(\w+\[)(.*?)(\])/g,
            (match, start, content, end) => {
                if (content.startsWith('"') && content.endsWith('"')) {
                    return match;
                }
                const sanitizedContent = content.replace(/"/g, '&quot;');
                return `${start}${sanitizedContent}${end}`;
            }
        );

        return sanitizedCode;
    }

    async function renderMermaid() {
        const rawCode = mermaidCodeInput.value;

        if (!rawCode.trim()) {
            previewContainer.innerHTML = '';
            errorContainer.classList.add('hidden');
            return;
        }

        const mermaidCode = sanitizeMermaidCode(rawCode);

        try {
            const { svg } = await mermaid.render('mermaid-graph', mermaidCode);
            previewContainer.innerHTML = svg;
            errorContainer.classList.add('hidden');
        } catch (error) {
            console.error("Mermaid rendering error:", error);
            errorContainer.textContent = "Error rendering diagram: \n" + error.message;
            errorContainer.classList.remove('hidden');
            previewContainer.innerHTML = '';
        }
    }

    mermaidCodeInput.addEventListener('input', renderMermaid);
    renderMermaid();
});
