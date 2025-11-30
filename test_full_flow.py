# test_full_flow.py
import requests
import time
import sys

# URL anpassen, falls dein Docker Port anders gemappt ist
BASE_URL = "http://localhost:5656"

def test_podcast_generation():
    print(f"🚀 Starte End-to-End Test gegen {BASE_URL}...")

    # 1. Job starten
    # Gültige Gemini TTS Stimmen: achernar, achird, algenib, algieba, alnilam,
    # aoede, autonoe, callirrhoe, charon, despina, enceladus, erinome, fenrir,
    # gacrux, iapetus, kore, laomedeia, leda, orus, puck, pulcherrima, rasalgethi,
    # sadachbia, sadaltager, schedar, sulafat, umbriel, vindemiatrix, zephyr, zubenelgenubi
    payload = {
        "dialogue": [
            {"speaker": "Kore", "text": "Hallo, dies ist ein Test des Worker Systems."},
            {"speaker": "Puck", "text": "Alles klar, ich empfange das Signal."}
        ],
        "language": "de",
        "tts_model": "gemini-2.5-flash-preview-tts"
    }

    try:
        print("1️⃣  Sende Generierungs-Anfrage...")
        response = requests.post(f"{BASE_URL}/generate-gemini-podcast", json=payload)

        if response.status_code != 200:
            print(f"❌ Fehler beim Starten: {response.text}")
            return

        data = response.json()
        job_id = data.get("job_id")
        print(f"✅ Job gestartet! ID: {job_id}")

    except Exception as e:
        print(f"❌ Server nicht erreichbar. Läuft Docker? Fehler: {e}")
        return

    # 2. Polling (Warten auf Worker)
    print("2️⃣  Warte auf Worker (Polling)...")
    start_time = time.time()

    while True:
        status_res = requests.get(f"{BASE_URL}/podcast-status/{job_id}")
        status_data = status_res.json()
        status = status_data.get("status")

        sys.stdout.write(f"\r   Status: {status} ({int(time.time() - start_time)}s)")
        sys.stdout.flush()

        if status == "completed":
            print("\n✅ Worker fertig!")
            break
        elif status == "failed":
            print(f"\n❌ Job fehlgeschlagen! Error: {status_data.get('error')}")
            return

        time.sleep(2)

    # 3. Download
    print("3️⃣  Lade Datei herunter...")
    down_res = requests.get(f"{BASE_URL}/podcast-download/{job_id}")

    if down_res.status_code == 200:
        filename = "test_result.wav"
        with open(filename, "wb") as f:
            f.write(down_res.content)
        print(f"🎉 ERFOLG! Datei gespeichert als '{filename}'.")
        print(f"   Dateigröße: {len(down_res.content)} bytes")
    else:
        print(f"❌ Download fehlgeschlagen: {down_res.status_code}")

if __name__ == "__main__":
    test_podcast_generation()
