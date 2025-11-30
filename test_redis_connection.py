# test_redis_connection.py
import os
from redis import Redis

def check_redis():
    print("🔍 Prüfe Verbindung zu Redis...")

    # URL aus Environment oder Default Docker DNS
    redis_host = 'redis'

    try:
        r = Redis(host=redis_host, port=6379, socket_timeout=3)
        r.ping()
        print("✅ Redis Verbindung: ERFOLGREICH")
        print(f"   Redis Info: {r.info()['redis_version']}")
    except Exception as e:
        print(f"❌ Redis Verbindung FEHLGESCHLAGEN: {e}")
        print("   Tipp: Läuft der 'redis' Container im docker-compose Verbund?")

if __name__ == "__main__":
    check_redis()
