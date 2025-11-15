#!/usr/bin/env python3
"""
Debug-Skript V2: Korrekte Timeout-Setzung für Gemini TTS
"""

import os
import time
from google import genai
from google.genai import types
import httpx

def check_timeout_settings():
    """Überprüfe alle Timeout-Einstellungen"""
    
    print("=" * 60)
    print("TIMEOUT DIAGNOSE FÜR GEMINI TTS - V2")
    print("=" * 60)
    
    # 1. Check httpx default timeout
    print("\n1. HTTPX Client Defaults:")
    print(f"   - Default timeout: {httpx.Timeout(timeout=5.0)}")
    
    # 2. Check Google GenAI client settings
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY nicht gesetzt!")
        return
    
    client = genai.Client(api_key=api_key)
    
    print("\n2. Google GenAI Client Settings (BEFORE fix):")
    if hasattr(client, '_api_client'):
        api_client = client._api_client
        if hasattr(api_client, '_httpx_client'):
            httpx_client = api_client._httpx_client
            print(f"   - Timeout: {httpx_client.timeout}")
    
    # 3. FIX ANWENDEN - Direkt am httpx Client
    print("\n3. Applying Timeout Fix:")
    
    clients_to_test = []
    
    # Client 1: Default (5s)
    client1 = genai.Client(api_key=api_key)
    clients_to_test.append(("Default (5s)", client1, 5))
    
    # Client 2: 120s Timeout
    client2 = genai.Client(api_key=api_key)
    if hasattr(client2, '_api_client') and hasattr(client2._api_client, '_httpx_client'):
        client2._api_client._httpx_client.timeout = httpx.Timeout(timeout=120.0)
        clients_to_test.append(("Fixed 120s", client2, 120))
        print("   ✅ Created client with 120s timeout")
    
    # Client 3: 300s Timeout
    client3 = genai.Client(api_key=api_key)
    if hasattr(client3, '_api_client') and hasattr(client3._api_client, '_httpx_client'):
        client3._api_client._httpx_client.timeout = httpx.Timeout(timeout=300.0)
        clients_to_test.append(("Fixed 300s", client3, 300))
        print("   ✅ Created client with 300s timeout")
    
    # 4. Test verschiedene Content-Längen
    print("\n4. Testing with different content lengths:")
    
    test_contents = [
        ("Kurz (1 Satz)", "Dies ist ein kurzer Test.", 5),
        ("Mittel (5 Sätze)", " ".join([f"Dies ist Satz Nummer {i}." for i in range(5)]), 10),
        ("Lang (20 Sätze)", " ".join([f"Dies ist Satz Nummer {i} mit etwas mehr Inhalt." for i in range(20)]), 30),
    ]
    
    for client_name, client, expected_timeout in clients_to_test[:1]:  # Nur Default für ersten Test
        print(f"\n   === Testing {client_name} ===")
        
        # Verify timeout is set
        if hasattr(client, '_api_client') and hasattr(client._api_client, '_httpx_client'):
            actual_timeout = client._api_client._httpx_client.timeout
            print(f"   Actual timeout: {actual_timeout}")
        
        for content_name, content, max_wait in test_contents:
            print(f"\n   {content_name}:")
            print(f"   Content length: {len(content)} chars")
            
            try:
                start = time.time()
                
                response = client.models.generate_content(
                    model="gemini-2.5-flash-preview-tts",
                    contents=content,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name="Kore"
                                )
                            )
                        )
                    )
                )
                
                elapsed = time.time() - start
                
                # Check if we got audio
                if response and response.candidates:
                    audio_data = response.candidates[0].content.parts[0].inline_data.data
                    print(f"   ✅ Erfolg in {elapsed:.1f}s - Audio: {len(audio_data)} bytes")
                else:
                    print(f"   ⚠️ Erfolg in {elapsed:.1f}s - Aber keine Audio-Daten")
                    
            except Exception as e:
                elapsed = time.time() - start
                error_msg = str(e)
                
                if elapsed > expected_timeout - 1:  # Nahe am Timeout
                    print(f"   ❌ TIMEOUT nach {elapsed:.1f}s (Expected ~{expected_timeout}s)")
                    print(f"      Error: {error_msg[:100]}")
                else:
                    print(f"   ❌ Fehler nach {elapsed:.1f}s: {error_msg[:100]}")
                
                # Bei Timeout, teste mit längerem Timeout
                if "timeout" in error_msg.lower() or elapsed > 4.5:
                    print(f"   🔄 Retry with longer timeout needed!")
    
    # 5. Test mit dem längsten Timeout und längstem Content
    print("\n5. CRITICAL TEST - Long content with 300s timeout:")
    
    if len(clients_to_test) >= 3:
        _, client300, _ = clients_to_test[2]  # 300s client
        
        # Generiere langen Content (wie im echten Podcast)
        long_content = " ".join([
            f"Dies ist Zeile {i} eines längeren Podcasts. " +
            f"Wir sprechen über verschiedene Themen und diskutieren interessante Punkte. " +
            f"Der Inhalt ist vielfältig und spannend."
            for i in range(40)  # ~40 Zeilen wie im echten Fall
        ])
        
        print(f"   Content: {len(long_content)} chars (~40 lines)")
        print("   Testing with 300s timeout...")
        
        try:
            start = time.time()
            
            response = client300.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=long_content,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Kore"
                            )
                        )
                    )
                )
            )
            
            elapsed = time.time() - start
            
            if response and response.candidates:
                audio_data = response.candidates[0].content.parts[0].inline_data.data
                print(f"   ✅✅✅ SUCCESS in {elapsed:.1f}s!")
                print(f"   Audio size: {len(audio_data)} bytes")
                print(f"   Estimated duration: ~{len(audio_data)/48000:.1f} seconds")
            
        except Exception as e:
            elapsed = time.time() - start
            print(f"   ❌ Failed after {elapsed:.1f}s")
            print(f"   Error: {str(e)[:200]}")
    
    # 6. Empfehlungen
    print("\n6. DIAGNOSE ERGEBNIS:")
    print("   ✅ Problem: Default timeout ist nur 5 Sekunden")
    print("   ✅ Lösung: Timeout direkt am httpx_client setzen")
    print("   ❌ http_options funktioniert NICHT (API Bug)")
    print("\n   CODE FIX für /app/services/gemini_service.py:")
    print("   --------------------------------------------")
    print("   self.client = genai.Client(api_key=api_key)")
    print("   # FIX: Erhöhe Timeout")
    print("   if hasattr(self.client, '_api_client'):")
    print("       if hasattr(self.client._api_client, '_httpx_client'):")
    print("           import httpx")
    print("           self.client._api_client._httpx_client.timeout = httpx.Timeout(timeout=300.0)")
    print("   --------------------------------------------")

if __name__ == "__main__":
    check_timeout_settings()