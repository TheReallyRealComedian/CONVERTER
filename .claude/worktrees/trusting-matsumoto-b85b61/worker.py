"""
RQ Worker for background podcast generation tasks.
Runs as a separate container/process and pulls jobs from Redis.
"""
import os
import redis
from rq import Worker, Queue, Connection

listen = ['default']

redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
conn = redis.from_url(redis_url)

if __name__ == '__main__':
    print("Worker gestartet und wartet auf Jobs...")
    with Connection(conn):
        worker = Worker(list(map(Queue, listen)))
        worker.work()
