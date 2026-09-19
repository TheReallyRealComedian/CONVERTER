#!/usr/bin/env python3
"""Byte gate for the one renderer: DEPLOYED vs WORKING TREE over every prod
document (RICH-MEDIA, gesperrte Entscheidung 6).

``render_markdown_to_html`` feeds the reader, the PDF and the EPUB. A change
to it is only safe to ship when the existing library renders to the SAME
string — except for the documents the change is about. pytest cannot answer
that (it does not know the library); this does, against the real documents,
without touching anything:

* it runs INSIDE the web container, where the deployed renderer and the repo
  pins (nh3!) live — the Mac's nh3 is a different version and proves nothing;
* the working tree's three renderer modules travel as STRINGS inside the
  payload and are loaded in memory (``exec`` into module objects) — no file is
  written on the Mintbox, nothing to clean up, no unversioned file left;
* the DB is opened ``mode=ro``; the ``sys.modules`` stubs are process-local.

Output: the ids whose rendered HTML differs, what kind of media each carries,
whether the difference is ONLY the forced ``<img>`` attributes
(``loading="lazy" referrerpolicy="no-referrer"``), and a clipped diff
otherwise. It prints document MARKUP lines, clipped to 150 chars — run it
where that is fine to see. The judgement (which ids may differ) is the
operator's; the script measures and always exits 0.

How to run (from the Mac, in the repo root — nothing to deploy first):

    python3 scripts/gate_render_bytes.py \\
        | ssh mintbox 'docker exec -i markdown-converter-web python3 -'

RICH-MEDIA result 2026-09-19 (223 documents): differing = #240 (the media
probe) and #137/#138/#140/#181 — those four ONLY by the forced img pair.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Working-tree modules the new renderer consists of: payload key → path. The
# keys double as the in-container module names under the ``services`` stub.
_SOURCES = {
    'svg_sanitize': 'services/svg_sanitize.py',
    'doc_media': 'services/doc_media.py',
    'markdown_render': 'app_pkg/markdown_render.py',
}

_CONTAINER_BODY = r'''
import difflib, re, sqlite3, sys, types
import importlib.metadata as md


def load(name, source, filename):
    mod = types.ModuleType(name)
    mod.__file__ = filename
    sys.modules[name] = mod
    exec(compile(source, filename, 'exec'), mod.__dict__)
    return mod


# OLD: the deployed file, loaded by path under an alias — no app import.
DEPLOYED = '/app/app_pkg/markdown_render.py'
old = load('deployed_markdown_render', open(DEPLOYED, encoding='utf-8').read(), DEPLOYED)

# NEW: the working tree, in memory only. The stub keeps the deployed
# services/__init__ (SDK imports) and the deployed svg_sanitize out of it.
stub = types.ModuleType('services')
stub.__path__ = []
sys.modules['services'] = stub
load('services.svg_sanitize', NEW_SOURCES['svg_sanitize'], '<tree svg_sanitize>')
load('services.doc_media', NEW_SOURCES['doc_media'], '<tree doc_media>')
new = load('tree_markdown_render', NEW_SOURCES['markdown_render'], '<tree markdown_render>')

print('container pins:', ' | '.join(
    f'{p} {md.version(p)}' for p in ('nh3', 'markdown-it-py', 'mdit-py-plugins', 'Pygments')))

con = sqlite3.connect('file:/app/data/converter.db?mode=ro', uri=True)
rows = con.execute('SELECT id, content FROM conversion ORDER BY id').fetchall()
con.close()
print('documents:', len(rows))

MERMAID = re.compile(r'^ {0,3}(`{3,}|~{3,})[ \t]*mermaid', re.I | re.M)
IMAGE = re.compile(r'<img\b|!\[[^\]]*\]\(', re.I)
FORCED = ' loading="lazy" referrerpolicy="no-referrer"'


def clip(line, n=150):
    return line if len(line) <= n else line[:n] + f'…[+{len(line) - n}]'


def kinds(content):
    low = content.lower()
    found = [name for name, hit in (
        ('svg', '<svg' in low), ('data:image', 'data:image' in low),
        ('mermaid', bool(MERMAID.search(content))), ('img', bool(IMAGE.search(content))),
    ) if hit]
    return ' '.join(found) or 'no media'


differing, media = [], []
for cid, content in rows:
    content = content or ''
    low = content.lower()
    if '<svg' in low or 'data:image' in low or MERMAID.search(content):
        media.append(cid)
    a = old.render_markdown_to_html(content)
    b = new.render_markdown_to_html(content)
    if a == b:
        continue
    differing.append(cid)
    only_forced = b.replace(FORCED, '') == a
    print(f'\n=== #{cid} differs | {kinds(content)} | only forced img attrs: {only_forced}')
    if only_forced:
        print(f'    <img> tags: {b.count("<img")}, forced pairs: {b.count(FORCED)}')
        continue
    diff = list(difflib.unified_diff(a.splitlines(), b.splitlines(),
                                     'deployed', 'tree', lineterm='', n=0))
    for line in diff[:24]:
        print('   ', clip(line))
    if len(diff) > 24:
        print(f'    … {len(diff) - 24} more diff lines')

print('\ndiffering ids            :', differing)
print('media ids (svg|data|mmd) :', media)
print('differing but NOT media  :', [i for i in differing if i not in media])
'''


def main():
    sources = {}
    for key, rel in _SOURCES.items():
        with open(os.path.join(REPO, rel), encoding='utf-8') as f:
            sources[key] = f.read()
    sys.stdout.write('NEW_SOURCES = ' + repr(sources) + '\n' + _CONTAINER_BODY)


if __name__ == '__main__':
    main()
