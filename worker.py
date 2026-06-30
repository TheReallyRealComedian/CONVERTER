"""
RQ Worker for background generation tasks (faithful-narration rendering).
Runs as a separate container/process and pulls jobs from Redis.
"""
import os
import redis
from rq import Worker, Queue

listen = ['default']

redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
conn = redis.from_url(redis_url)

if __name__ == '__main__':
    print("Worker gestartet und wartet auf Jobs...")
    queues = [Queue(name, connection=conn) for name in listen]
    worker = Worker(queues, connection=conn)
    worker.work()
