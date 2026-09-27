"""
RQ Worker for background generation tasks (faithful-narration rendering).
Runs as a separate container/process and pulls jobs from Redis.
"""
import os
import redis
from rq import Worker, Queue

from app_pkg.config import RQ_SERIALIZER

listen = ['default']

redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
conn = redis.from_url(redis_url)


def build_worker(conn):
    """Worker + its queues, all on ``RQ_SERIALIZER`` (SEC-REDIS-AUTH).

    Both need it: the Worker dequeues with its own serializer, each Queue
    object carries one too.
    """
    queues = [Queue(name, connection=conn, serializer=RQ_SERIALIZER) for name in listen]
    return Worker(queues, connection=conn, serializer=RQ_SERIALIZER)


if __name__ == '__main__':
    print("Worker gestartet und wartet auf Jobs...")
    build_worker(conn).work()
