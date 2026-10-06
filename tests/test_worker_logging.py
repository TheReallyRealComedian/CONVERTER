"""ARCH-BUILD — the worker process logs (Auflage aus ARCH-NARR5).

Until this sprint ``worker.py`` configured no logging: the root logger had
no handler and sat at WARNING, so every ``logger.info`` from ``tasks`` and
``services.*`` vanished and ``docker logs markdown-converter-worker`` showed
only RQ's own lines (measured in the container). ``configure_logging()``
puts ONE stream handler in the launcher's format on the root logger at
INFO, before the worker starts.

RQ's side (rq 2.8.0, ``rq.logutils.setup_loghandlers``) adds its own
``rq.worker`` handlers only when no handler exists anywhere up the logger
hierarchy (``_has_effective_handler``) — with the root handler in place its
lines travel through it exactly once. That half is measured in ``docker
logs`` on deploy (the Mac has rq 1.16.0, not the pin); pinned here is our
half: the handler, the format, the level, idempotence, and the order in
``__main__``.
"""
import ast
import io
import logging
from pathlib import Path

import worker
from services import mineru_launcher


def _restore(root, handlers, level):
    for handler in root.handlers[:]:
        if handler not in handlers:
            root.removeHandler(handler)
    root.setLevel(level)


def test_configure_logging_makes_info_lines_from_tasks_and_services_visible():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    stream = io.StringIO()
    try:
        worker.configure_logging(stream)
        worker.configure_logging(stream)  # idempotent: no second handler
        added = [h for h in root.handlers if h not in handlers]
        assert len(added) == 1, added
        assert added[0].formatter._fmt == worker.LOG_FORMAT
        assert root.level == logging.INFO

        logging.getLogger('tasks').info('Narration %s gerendert', 'abc')
        logging.getLogger('services.narration_render').info('Chunk 1/2')
        logging.getLogger('services.deepgram_service').debug('unsichtbar')
        lines = stream.getvalue().splitlines()
        assert len(lines) == 2, lines
        assert lines[0].endswith(' INFO tasks: Narration abc gerendert'), lines[0]
        assert lines[1].endswith(' INFO services.narration_render: Chunk 1/2'), lines[1]
    finally:
        _restore(root, handlers, level)


def test_worker_and_launcher_share_one_line_format():
    assert worker.LOG_FORMAT == mineru_launcher.LOG_FORMAT
    assert worker.LOG_FORMAT == '%(asctime)s %(levelname)s %(name)s: %(message)s'


def test_main_configures_logging_before_the_worker_starts():
    tree = ast.parse(Path(worker.__file__).read_text())
    mains = [node for node in tree.body
             if isinstance(node, ast.If) and '__main__' in ast.unparse(node.test)]
    assert len(mains) == 1
    statements = [ast.unparse(stmt) for stmt in mains[0].body]
    assert statements[0] == 'configure_logging()', statements
    assert any('.work()' in stmt for stmt in statements[1:]), statements
    # build_worker itself is untouched (its serializer sentinel lives in
    # tests/test_rq_serializer.py)
    assert any(isinstance(n, ast.FunctionDef) and n.name == 'build_worker'
               for n in tree.body)
