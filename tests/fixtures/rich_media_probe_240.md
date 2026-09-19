# Render-Probe: SVG in Markdown

Drei Varianten, dieselbe Skizze (Carbonat, Grenzstruktur A). Welche wird als Bild angezeigt?

## Variante 1 — inline `<svg>`

<svg xmlns="http://www.w3.org/2000/svg" width="260" height="150" viewBox="0 0 260 150" font-family="sans-serif" font-size="18">
  <rect width="260" height="150" fill="#fff" stroke="#999"/>
  <text x="122" y="90" text-anchor="middle">C</text>
  <text x="122" y="32" text-anchor="middle">O</text>
  <text x="40" y="128" text-anchor="middle">O</text>
  <text x="204" y="128" text-anchor="middle">O</text>
  <line x1="118" y1="74" x2="118" y2="40" stroke="#222" stroke-width="2"/>
  <line x1="126" y1="74" x2="126" y2="40" stroke="#222" stroke-width="2"/>
  <line x1="110" y1="92" x2="54" y2="118" stroke="#222" stroke-width="2"/>
  <line x1="134" y1="92" x2="190" y2="118" stroke="#222" stroke-width="2"/>
  <text x="22" y="112" font-size="12">−</text>
  <text x="218" y="112" font-size="12">−</text>
  <text x="130" y="140" text-anchor="middle" font-size="11" fill="#c00">Variante 1: inline svg</text>
</svg>

## Variante 2 — `<img>` mit data-URI

<img alt="Variante 2" src="data:image/svg+xml;utf8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22260%22%20height%3D%2260%22%3E%3Crect%20width%3D%22260%22%20height%3D%2260%22%20fill%3D%22%23eef%22%20stroke%3D%22%2399f%22%2F%3E%3Ctext%20x%3D%22130%22%20y%3D%2236%22%20text-anchor%3D%22middle%22%20font-family%3D%22sans-serif%22%20font-size%3D%2216%22%3EVariante%202%3A%20img%20data-URI%3C%2Ftext%3E%3C%2Fsvg%3E">

## Variante 3 — Markdown-Bild mit data-URI

![Variante 3](data:image/svg+xml;utf8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22260%22%20height%3D%2260%22%3E%3Crect%20width%3D%22260%22%20height%3D%2260%22%20fill%3D%22%23efe%22%20stroke%3D%22%239c9%22%2F%3E%3Ctext%20x%3D%22130%22%20y%3D%2236%22%20text-anchor%3D%22middle%22%20font-family%3D%22sans-serif%22%20font-size%3D%2216%22%3EVariante%203%3A%20md%20data-URI%3C%2Ftext%3E%3C%2Fsvg%3E)

## Variante 4 — Mermaid (Codeblock)

```mermaid
flowchart LR
  A["Formel sagt: 1 Doppel-, 2 Einfachbindungen"] --> B["Messung: alle gleich lang"]
  B --> C["Formel zu grob → Grenzstrukturen"]
```

Ende der Probe.
