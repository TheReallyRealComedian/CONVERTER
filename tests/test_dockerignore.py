"""IMG-CONTEXT — the build context carries no secrets and no tool state.

``COPY . .`` copies the directory, not the repository: everything
``.dockerignore`` does not exclude lands in the image, including what
``.gitignore`` hides from Git. Measured in the deployed image
``2a97cb4ed290``: ``/app/google-credentials.json`` (the GCP service-account
key), ``/app/.codebuddy/db`` (28 MB), ``/app/.claude/``,
``/app/.pytest_cache/``, ``.env.example``, master docs and two root scripts
— none with a runtime reader. These sentinels pin the boundary:

1. Every pattern of the must-list stands as its own line, and no ``!``
   exception can re-include anything behind the list's back.
2. No pattern excludes what the running app reads: a pattern that matches a
   keep-name, or reaches into one with its first path component, fails.
   ``keyterms.json`` stays on purpose — ``DeepgramService.load_keyterms``
   reads it from ``/app``.

Docker reads the patterns root-anchored (Go ``filepath.Match``, no
gitignore-style recursion): ``data/`` hits ``./data`` only. Only ``**``
reaches deeper — it matches zero or more directories, the root included, so
a leading ``**`` reaches into every keep directory and trips rule 2. Exempt
are exactly the ``ANYWHERE`` patterns (bytecode caches, Finder metadata,
AppleDouble — nothing reads them at any depth); at the root they still
count. Whether Docker reads the patterns as intended is measured on the
built image (``ls -A /app``, ``find``), not here.
"""
import posixpath
from fnmatch import fnmatchcase
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Junk at any depth; the only patterns allowed to reach into every directory.
ANYWHERE = ('**/__pycache__/', '**/*.pyc', '**/.DS_Store', '**/._*')

MUST_EXCLUDE = ANYWHERE + (
    # secrets — the key arrives per compose bind, the env per env_file
    'google-credentials.json', '.env*',
    # tool state
    '.claude/', '.codebuddy/', '.pytest_cache/',
    # DB copies (MINTBOX-BAK)
    'app_data_bak-*',
    # compose files, the Mac-dev override included
    'docker-compose*.yml',
    # master docs and root scripts without a runtime reader
    'BACKLOG.md', 'STATUS.md', 'CLAUDE.md', 'MASTER_BACKLOG_HANDOFF_*.md',
    'pytest.ini', 'test_redis_connection.py', 'test_worker_libraries.py',
)

MUST_KEEP = (
    'keyterms.json', 'static', 'templates', 'app_pkg', 'services', 'scripts',
    'app.py', 'tasks.py', 'worker.py', 'models.py', 'requirements.txt',
    'constraints.txt',
)


def _lines():
    """The patterns as Docker reads them: a ``#`` in column 0 is a comment,
    everything else is trimmed, blank lines drop out."""
    dockerignore = REPO / '.dockerignore'
    if not dockerignore.exists():
        pytest.skip('.dockerignore not shipped alongside the tests')
    lines = (line for line in dockerignore.read_text().splitlines()
             if not line.startswith('#'))
    return [line.strip() for line in lines if line.strip()]


def _normalized(pattern):
    # What Docker matches: filepath.Clean, then a leading '/' dropped.
    return posixpath.normpath(pattern).lstrip('/')


def _hits(pattern, name):
    """Does ``pattern`` exclude ``name`` or reach into it? Normalized like
    Docker, so ``static/`` and ``/static`` count as ``static``. An
    ``ANYWHERE`` pattern is judged by what it does at the root."""
    pattern = _normalized(pattern)
    if pattern in {_normalized(p) for p in ANYWHERE}:
        pattern = pattern.removeprefix('**/')
    return (fnmatchcase(name, pattern)
            or fnmatchcase(name, pattern.split('/')[0]))


def test_the_must_list_stands_line_by_line():
    lines = _lines()
    missing = [p for p in MUST_EXCLUDE if p not in lines]
    assert not missing, missing
    # The last matching line wins in Docker: `!google-credentials.json` after
    # the list would quietly put the key back. Needs one? Decide it here.
    exceptions = [line for line in lines if line.startswith('!')]
    assert not exceptions, exceptions


def test_no_pattern_hits_what_the_app_reads():
    hits = [(p, name) for p in _lines() for name in MUST_KEEP
            if _hits(p, name)]
    assert not hits, hits


def test_the_keep_check_can_fire():
    # Positive control: a matcher that never fires would pass the test above.
    assert _hits('*.json', 'keyterms.json')
    assert _hits('static/', 'static')
    assert _hits('/templates', 'templates')
    assert _hits('services/gemini/*.py', 'services')
    assert _hits('**/*.css', 'static')  # a new ** pattern must be decided
    assert not _hits('.claude/', 'static')
    assert not _hits('**/__pycache__/', 'app_pkg')
