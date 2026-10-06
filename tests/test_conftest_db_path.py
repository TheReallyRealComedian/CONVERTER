"""ARCH-BUILD — the test database is one file PER PROCESS.

The fixed name ``<tempdir>/converter-test.db`` made two concurrent pytest
runs on one machine destroy each other's fixtures (measured 2026-10-05:
279 passed / 1359 errors in parallel, 1639 + 1 alone). Now the file carries
the pid; leftovers of a crashed run are swept only once their process is
gone, and the own file is removed at exit.
"""
import os

from tests import conftest


def test_the_test_database_carries_the_pid():
    assert conftest._TEST_DB_FILE.name == f'converter-test-{os.getpid()}.db'
    assert os.environ['DATABASE_URL'].endswith(conftest._TEST_DB_FILE.name)


def test_a_live_process_is_never_swept():
    assert conftest._pid_alive(os.getpid())
