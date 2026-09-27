"""SEC-NONROOT (F-8) — web and worker run as uid 1000; the code stays root's.

The Dockerfile decides who the processes are. These sentinels pin the
decisions, so a later edit cannot quietly bring root back or break the
fresh-volume inheritance (text-level, like the compose sentinels):

1. Exactly one ``USER``, ``1000:1000``, after the last ``COPY`` and before
   ``CMD`` — everything that needs root (apt, pip, the mount points) runs
   before it, and ``CMD`` inherits it.
2. The three data mount points are created and handed to 1000:1000 BEFORE
   ``COPY . .``: a fresh named volume takes content and owner only from a
   directory that exists in the image (Docker creates a missing mount point
   root:root). ``.dockerignore`` keeps same-named directories out of the
   build context — ``COPY`` resets such a directory's owner to root
   (measured on the Mintbox, BuildKit 29.8.1).
3. ``COPY . .`` carries no ``--chown``: the code stays root-owned and
   read-only for the process. ``HOME`` is set and writable (Chromium).
4. No runtime data under ``/root`` (700): the NLTK assets are downloaded
   into a system directory of NLTK's own search path — from
   ``/root/nltk_data`` every unstructured partition died as uid 1000 on
   ``Resource 'punkt_tab' not found`` (measured in the prod image).
5. Compose keeps the image's user for web and worker. The launcher's
   explicit root is pinned in tests/test_compose_socket.py.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DATA_DIRS = ('/app/data', '/app/output_podcasts', '/app/doclocal_exchange')
NLTK_DIR = '/usr/local/share/nltk_data'


def _dockerfile_text():
    dockerfile = REPO / 'Dockerfile'
    if not dockerfile.exists():
        pytest.skip('Dockerfile not shipped alongside the tests')
    return dockerfile.read_text()


def _instructions(text):
    """``(KEYWORD, arguments)`` per instruction: comments dropped,
    continuation lines joined, a heredoc body attached to its instruction."""
    out = []
    lines = iter(text.splitlines())
    for line in lines:
        joined = line.strip()
        if not joined or joined.startswith('#'):
            continue
        while joined.endswith('\\'):
            nxt = next(lines).strip()
            while not nxt or nxt.startswith('#'):
                nxt = next(lines).strip()
            joined = f'{joined[:-1].rstrip()} {nxt}'
        heredoc = re.search(r"<<-?(['\"]?)(\w+)\1", joined)
        if heredoc:
            body = []
            for body_line in lines:
                if body_line.strip() == heredoc.group(2):
                    break
                body.append(body_line)
            joined += '\n' + '\n'.join(body)
        keyword, _, args = joined.partition(' ')
        out.append((keyword.upper(), args.strip()))
    return out


def _index(instructions, keyword, predicate=lambda args: True):
    hits = [i for i, (kw, args) in enumerate(instructions)
            if kw == keyword and predicate(args)]
    assert len(hits) == 1, (keyword, hits)
    return hits[0]


def test_exactly_one_user_after_the_last_copy_and_before_cmd():
    instructions = _instructions(_dockerfile_text())
    user = _index(instructions, 'USER')
    assert instructions[user][1] == '1000:1000'
    last_copy = max(i for i, (kw, _) in enumerate(instructions) if kw == 'COPY')
    cmd = _index(instructions, 'CMD')
    assert last_copy < user < cmd
    assert cmd == len(instructions) - 1  # nothing after CMD can re-open root


def test_mount_points_belong_to_1000_before_the_code_is_copied():
    instructions = _instructions(_dockerfile_text())

    def commands(args):
        return [c.strip() for c in args.split('&&')]

    def prepares_mount_points(args):
        cmds = commands(args)
        mkdir = [c for c in cmds if c.startswith('mkdir -p ')]
        chown = [c for c in cmds if c.startswith('chown 1000:1000 ')]
        return (len(mkdir) == 1 and len(chown) == 1
                and set(mkdir[0].split()[2:]) == set(DATA_DIRS)
                and set(chown[0].split()[2:]) == set(DATA_DIRS))

    prepare = _index(instructions, 'RUN', prepares_mount_points)
    code_copy = _index(instructions, 'COPY', lambda args: args == '. .')
    assert prepare < code_copy < _index(instructions, 'USER')


def test_code_stays_root_owned_and_home_is_set():
    instructions = _instructions(_dockerfile_text())
    copies = [args for kw, args in instructions if kw in ('COPY', 'ADD')]
    assert copies and not any('--chown' in args for args in copies), copies
    assert ('ENV', 'HOME=/home/converter') in instructions
    # uid 1000 is the base image's 'ubuntu', renamed: one passwd entry, and
    # its home moves to the HOME above (--move-home keeps owner 1000:1000).
    _index(instructions, 'RUN', lambda args: all(
        part in args for part in (
            'usermod --login converter --home /home/converter --move-home ubuntu',
            'groupmod --new-name converter ubuntu')))


def test_dockerignore_keeps_the_mount_points_out_of_the_context():
    dockerignore = REPO / '.dockerignore'
    if not dockerignore.exists():
        pytest.skip('.dockerignore not shipped alongside the tests')
    entries = {line.strip() for line in dockerignore.read_text().splitlines()
               if line.strip() and not line.startswith('#')}
    for mount_point in DATA_DIRS:
        assert f'{mount_point.removeprefix("/app/")}/' in entries, mount_point


def test_nltk_assets_live_outside_root():
    text = _dockerfile_text()
    assert f"NLTK_DIR = '{NLTK_DIR}'" in text
    calls = re.findall(r'nltk\.download\(([^)]*)\)', text)
    assert calls and all('download_dir=NLTK_DIR' in c for c in calls), calls
    # The directory must be one NLTK searches by default for EVERY user —
    # that is the whole point of leaving ~/nltk_data.
    nltk = pytest.importorskip('nltk')
    assert NLTK_DIR in nltk.data.path


def test_compose_keeps_the_images_user_for_web_and_worker():
    compose_file = REPO / 'docker-compose.yml'
    if not compose_file.exists():
        pytest.skip('docker-compose.yml not shipped alongside the tests')
    yaml = pytest.importorskip('yaml')
    services = yaml.safe_load(compose_file.read_text())['services']
    for name in ('markdown-converter', 'worker'):
        assert 'user' not in services[name], name
