"""
RQ Worker for background generation tasks (faithful-narration rendering).
Runs as a separate container/process and pulls jobs from Redis.
"""
import logging
import os
import sys

import redis
from rq import Worker, Queue

from app_pkg.config import RQ_SERIALIZER

# ARCH-BUILD: one line shape for worker and launcher (services/mineru_launcher.py
# carries the same literal; tests/test_worker_logging.py pins the equality).
LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'

listen = ['default']

redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
conn = redis.from_url(redis_url)
logger = logging.getLogger('worker')


def configure_logging(stream=None):
    """Root logger at INFO with ONE stream handler in ``LOG_FORMAT``.

    Until ARCH-BUILD this process configured no logging: the root logger had
    no handler and sat at WARNING, so every ``logger.info`` from ``tasks``
    and ``services.*`` vanished and ``docker logs`` showed only RQ's own
    lines (ARCH-NARR5 finding, measured in the container). RQ (2.8.0,
    ``rq.logutils.setup_loghandlers``) adds its own ``rq.worker`` handlers
    only when NO handler exists anywhere up the logger hierarchy — with this
    root handler in place its lines travel through it exactly once, in this
    format (measured in ``docker logs`` on deploy).

    Idempotent (a second call adds no second handler), and deliberately not
    ``logging.basicConfig``: that is a no-op once any root handler exists
    (pytest installs some), and ``force=True`` would tear foreign handlers
    down. ``stream`` defaults to the current ``sys.stderr``.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(getattr(handler, '_converter_worker', False) for handler in root.handlers):
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handler._converter_worker = True
        root.addHandler(handler)
    return root


def build_worker(conn):
    """Worker + its queues, all on ``RQ_SERIALIZER`` (SEC-REDIS-AUTH).

    Both need it: the Worker dequeues with its own serializer, each Queue
    object carries one too.
    """
    queues = [Queue(name, connection=conn, serializer=RQ_SERIALIZER) for name in listen]
    return Worker(queues, connection=conn, serializer=RQ_SERIALIZER)


if __name__ == '__main__':
    configure_logging()
    logger.info('Worker gestartet und wartet auf Jobs...')
    build_worker(conn).work()
