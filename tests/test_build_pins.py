"""ARCH-BUILD — the build is pinned, checksummed and loud (text sentinels).

Until this sprint the Docker layer cache was the only lockfile: five lines
in ``requirements.txt`` floated, every rebuild of the pip layer re-resolved
all transitives, and four fuses had no guard. These sentinels pin what the
Dockerfile and the requirement files must SAY, so a later edit cannot
quietly bring a float, an unchecked download or a silent NLTK failure
back. What actually RUNS is protocolled in ``docs/build/pip-freeze.txt``
(``scripts/freeze_image.sh``) — measured on the image, not here.

1. The Playwright base-image tag and the ``playwright==`` pin name the same
   version: the image ships Playwright AND its browsers (OKTOBER).
2. Every requirement and every constraint is an exact ``==`` pin — no
   range, no compat operator, no wildcard, no bare name.
3. Every ``curl`` download in the Dockerfile is followed by ``sha256sum -c``
   on the downloaded file BEFORE the artifact is used, with one hash per
   architecture and a default arm that fails. Exactly the two known
   downloads exist (pandoc deb, docker CLI tgz) — a third is decided here.
4. The NLTK block fails the build when a resource does not download or
   cannot be found afterwards — ``nltk.download`` returns ``False`` instead
   of raising, and the old block ignored the value — looks every resource
   up in the target directory only, and does not switch TLS verification
   off. The block is EXECUTED here against a stub ``nltk``: behaviour, not
   wording.

Skips when the build files are not shipped alongside the tests (the
container run without the mounts).
"""
import re
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.test_nonroot import NLTK_DIR, _instructions

REPO = Path(__file__).resolve().parent.parent

BASE_IMAGE = re.compile(
    r'^mcr\.microsoft\.com/playwright/python:v(\d+\.\d+\.\d+)-noble$')
# name[extras]==version — PEP 440 local/epoch/post allowed, `*` is not
# (``==1.*`` is a range in disguise).
EXACT_PIN = re.compile(
    r'^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9._,-]+\])?==[A-Za-z0-9.+!-]+$')
SHA256 = re.compile(r'\b[0-9a-f]{64}\b')
# Trailing comments: pip ends a requirement at whitespace + '#'.
COMMENT = re.compile(r'(^|\s)#.*$')

# What the Dockerfile may download, by its curl -o target. A new download
# is a decision for this list, never a quiet third line.
KNOWN_DOWNLOADS = ('/tmp/docker.tgz', '/tmp/pandoc.deb')
ARCHES = {'/tmp/pandoc.deb': ('amd64', 'arm64'),        # dpkg --print-architecture
          '/tmp/docker.tgz': ('x86_64', 'aarch64')}      # uname -m
# What unstructured actually needs at runtime (SEC-NONROOT measured the
# punkt_tab failure in the prod image); the block downloads more, fine.
RUNTIME_RESOURCES = {'punkt_tab', 'averaged_perceptron_tagger_eng'}


def _text(name):
    path = REPO / name
    if not path.exists():
        pytest.skip(f'{name} not shipped alongside the tests')
    return path.read_text()


def _spec_lines(text):
    out = []
    for raw in text.splitlines():
        line = COMMENT.sub('', raw).strip()
        if line:
            out.append(line)
    return out


def _commands(run_args):
    return [c.strip() for c in run_args.split('&&')]


def _download_runs():
    instructions = _instructions(_text('Dockerfile'))
    return [args for kw, args in instructions
            if kw == 'RUN' and re.search(r'\bcurl\b', args)]


def _curl_target(command):
    m = re.search(r'\bcurl\b.*?\s-o\s+(\S+)', command)
    assert m, command
    return m.group(1)


# --- 1. base image tag == playwright pin -----------------------------------

def test_base_image_tag_matches_the_playwright_pin():
    froms = [args for kw, args in _instructions(_text('Dockerfile'))
             if kw == 'FROM']
    assert len(froms) == 1, froms
    m = BASE_IMAGE.match(froms[0])
    # -noble is the Python-3.12 decision (OKTOBER); another distro suffix is
    # a decision for this regex, not a drift.
    assert m, froms[0]
    pins = [line for line in _spec_lines(_text('requirements.txt'))
            if line.startswith('playwright==')]
    assert pins == [f'playwright=={m.group(1)}'], (froms[0], pins)


# --- 2. no floats -----------------------------------------------------------

@pytest.mark.parametrize('name', ['requirements.txt', 'constraints.txt'])
def test_every_line_is_an_exact_pin(name):
    lines = _spec_lines(_text(name))
    assert lines, name
    floats = [line for line in lines if not EXACT_PIN.fullmatch(line)]
    assert not floats, floats


def test_the_pin_check_can_fire():
    # Positive control for the regex above.
    assert EXACT_PIN.fullmatch('Flask[async]==3.1.3')
    assert EXACT_PIN.fullmatch('pdfminer.six==20251230')
    assert EXACT_PIN.fullmatch('torch==2.12.1+cpu')
    for bad in ('google-genai>=1.0.0', 'pytest', 'numpy~=2.2', 'rq==2.*',
                'redis>=7,<8', 'a==1==2', 'Flask[async]'):
        assert not EXACT_PIN.fullmatch(bad), bad


# --- 3. checksummed downloads ----------------------------------------------

def test_exactly_the_known_downloads_exist():
    targets = sorted(_curl_target(run) for run in _download_runs())
    assert targets == sorted(KNOWN_DOWNLOADS), targets


def test_every_download_is_checked_before_the_artifact_is_used():
    runs = _download_runs()
    assert runs
    for run in runs:
        cmds = _commands(run)
        curl = [i for i, c in enumerate(cmds) if re.search(r'\bcurl\b', c)]
        check = [i for i, c in enumerate(cmds)
                 if re.search(r'\bsha256sum\s+(-c|--check)\b', c)]
        use = [i for i, c in enumerate(cmds)
               if re.search(r'\bdpkg -i\b|\btar\s+-x', c)]
        assert len(curl) == 1 and len(check) == 1 and use, cmds
        assert curl[0] < check[0] < min(use), cmds
        # ...and the check is on the file curl wrote, not on something else.
        assert _curl_target(cmds[curl[0]]) in cmds[check[0]], cmds[check[0]]


def test_every_download_pins_one_hash_per_architecture_and_fails_otherwise():
    for run in _download_runs():
        target = _curl_target(run)
        hashes = set(SHA256.findall(run))
        assert len(hashes) == len(ARCHES[target]), (target, hashes)
        for arch in ARCHES[target]:
            # a `case` arm per architecture carrying its own hash
            assert re.search(rf'\b{arch}\)[^;]*[0-9a-f]{{64}}', run), (target, arch)
        # unknown architecture: fail, never download unchecked
        assert re.search(r'\*\)[^;]*;\s*exit 1', run), target


# --- 4. the NLTK block fails the build, verifies TLS ------------------------

def _nltk_block():
    runs = [args for kw, args in _instructions(_text('Dockerfile'))
            if kw == 'RUN' and 'nltk' in args]
    assert len(runs) == 1, [r[:60] for r in runs]
    header, _, body = runs[0].partition('\n')
    assert header.startswith('python -'), header
    return body


def _run_nltk_block(download_ok=lambda resource: True,
                    find_ok=lambda probe: True, download_raises=()):
    """Execute the heredoc against a stub ``nltk``; return (exit code or
    None, the recorded download/find calls)."""
    calls = []
    nltk = types.ModuleType('nltk')

    def download(resource, download_dir=None, quiet=False, **_kwargs):
        calls.append(('download', resource, download_dir))
        if resource in download_raises:
            raise OSError('simulated network failure')
        return download_ok(resource)

    def find(probe, paths=None):
        calls.append(('find', probe, tuple(paths) if paths else None))
        if not find_ok(probe):
            raise LookupError(probe)
        return f'{(paths or ["?"])[0]}/{probe}'

    nltk.download = download
    nltk.data = types.SimpleNamespace(find=find)
    # A re-added TLS shim would meet this empty module, never the real one.
    ssl_stub = types.ModuleType('ssl')
    code = compile(_nltk_block(), 'Dockerfile:nltk', 'exec')
    with patch.dict(sys.modules, {'nltk': nltk, 'ssl': ssl_stub}), \
            patch('os.walk', return_value=iter(())):
        try:
            exec(code, {'__name__': '__main__'})
        except SystemExit as exc:
            return exc.code, calls
    return None, calls


def test_nltk_block_passes_when_every_resource_downloads_and_is_found():
    code, calls = _run_nltk_block()
    assert code is None, calls
    downloads = [c for c in calls if c[0] == 'download']
    finds = [c for c in calls if c[0] == 'find']
    assert {c[1] for c in downloads} >= RUNTIME_RESOURCES, downloads
    assert all(c[2] == NLTK_DIR for c in downloads), downloads
    # one lookup per download, pinned to the target directory
    assert len(finds) == len(downloads), (len(finds), len(downloads))
    assert all(c[2] == (NLTK_DIR,) for c in finds), finds


def test_nltk_block_fails_the_build_when_a_download_reports_failure():
    code, calls = _run_nltk_block(download_ok=lambda r: r != 'punkt_tab')
    assert code == 1, calls


def test_nltk_block_fails_the_build_when_a_download_raises():
    code, calls = _run_nltk_block(download_raises={'stopwords'})
    assert code == 1, calls


def test_nltk_block_fails_the_build_when_a_resource_is_not_findable():
    code, calls = _run_nltk_block(find_ok=lambda p: 'punkt_tab' not in p)
    assert code == 1, calls


def test_nltk_block_keeps_tls_verification():
    assert '_create_unverified_context' not in _text('Dockerfile')
    assert 'import ssl' not in _nltk_block()
