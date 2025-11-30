# test_worker_libraries.py
import os
import shutil
import sys

def check_dependencies():
    print("🔍 Prüfe Worker-Abhängigkeiten...")

    # 1. FFmpeg Check
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        print(f"✅ FFmpeg gefunden: {ffmpeg_path}")
    else:
        print("❌ FFmpeg NICHT gefunden! (Hast du das Dockerfile neu gebaut?)")

    # 2. PyDub Check
    try:
        from pydub import AudioSegment
        print("✅ PyDub importierbar")
    except ImportError:
        print("❌ PyDub NICHT installiert (requirements.txt prüfen)")
    except Exception as e:
        print(f"⚠️ PyDub Fehler: {e}")

    # 3. Output Ordner Check
    output_dir = '/app/output_podcasts'
    if os.path.exists(output_dir):
        print(f"✅ Output Volume existiert: {output_dir}")
        # Schreibtest
        try:
            with open(f"{output_dir}/test_write.txt", "w") as f:
                f.write("test")
            os.remove(f"{output_dir}/test_write.txt")
            print("✅ Schreibrechte auf Volume vorhanden")
        except Exception as e:
            print(f"❌ Schreibfehler auf Volume: {e}")
    else:
        print(f"❌ Output Ordner fehlt: {output_dir}")

if __name__ == "__main__":
    check_dependencies()
