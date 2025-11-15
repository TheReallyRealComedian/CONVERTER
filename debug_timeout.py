#!/usr/bin/env python3
"""
Debug-Skript V3: nginx Timeout Verifizierung
Nach nginx proxy_read_timeout Fix von 60s → 300s
"""

import os
import time
from google import genai
from google.genai import types
import httpx

def check_timeout_settings():
    """Überprüfe alle Timeout-Einstellungen nach nginx Fix"""
    
    print("=" * 70)
    print("TIMEOUT DIAGNOSE V3 - NGINX TIMEOUT FIX VERIFIZIERUNG")
    print("=" * 70)
    
    # 1. Environment Check
    print("\n1. ENVIRONMENT CHECK:")
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY nicht gesetzt!")
        return
    print("   ✅ GEMINI_API_KEY gefunden")
    
    # 2. Setup Client mit 300s Timeout
    print("\n2. CLIENT SETUP:")
    client = genai.Client(api_key=api_key)
    
    # Apply timeout fix
    if hasattr(client, '_api_client') and hasattr(client._api_client, '_httpx_client'):
        client._api_client._httpx_client.timeout = httpx.Timeout(timeout=300.0)
        actual_timeout = client._api_client._httpx_client.timeout
        print(f"   ✅ httpx Client timeout: {actual_timeout}")
    else:
        print("   ⚠️ Could not set timeout - using defaults")
    
    # 3. Test Content verschiedener Längen
    print("\n3. NGINX TIMEOUT TEST:")
    print("   Vorher: nginx proxy_read_timeout = 60s (default)")
    print("   Jetzt:  nginx proxy_read_timeout = 300s (konfiguriert)")
    print("\n   Wir testen Content der >60s Generierung braucht...")
    
    test_contents = [
        ("BASELINE: Kurz (30 chars)", 
         "Dies ist ein kurzer Test.", 
         "~2s - Sollte immer funktionieren"),
        
        ("TEST 1: Mittel (500 chars)", 
         " ".join([f"Dies ist Satz Nummer {i} mit etwas Inhalt." for i in range(10)]), 
         "~10s - War nie das Problem"),
        
        ("TEST 2: Lang (2000 chars)", 
         " ".join([f"Dies ist Zeile {i} eines längeren Podcasts mit interessanten Themen und Diskussionen." for i in range(20)]), 
         "~30s - Sollte funktionieren"),
        
        ("TEST 3: Sehr Lang (5000 chars) - KRITISCH!", 
         " ".join([
             f"Absatz {i}: Hier diskutieren wir verschiedene wichtige Themen. " +
             f"Wir gehen in die Tiefe und behandeln viele Aspekte des Themas. " +
             f"Es gibt zahlreiche interessante Punkte zu besprechen und zu analysieren."
             for i in range(40)
         ]), 
         "~70s - War vorher nginx timeout bei 60s!"),
        
        ("TEST 4: Ultra Lang (10000 chars) - STRESS TEST!", 
         " ".join([
             f"Kapitel {i}: In diesem Abschnitt behandeln wir ausführlich verschiedene Aspekte. " +
             f"Wir analysieren detailliert die Hintergründe und Zusammenhänge. " +
             f"Es gibt viele wichtige Punkte die wir diskutieren müssen. " +
             f"Die Thematik ist komplex und erfordert eine gründliche Betrachtung."
             for i in range(80)
         ]), 
         "~120s - Ultimate Test für 300s Timeout"),
    ]
    
    results = []
    
    for test_name, content, expected_behavior in test_contents:
        print(f"\n   === {test_name} ===")
        print(f"   Content length: {len(content)} chars")
        print(f"   Expected: {expected_behavior}")
        
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
                audio_kb = len(audio_data) / 1024
                
                # Determine if this was in the critical zone
                status = "✅ SUCCESS"
                if elapsed > 60 and elapsed < 70:
                    status = "✅✅✅ CRITICAL SUCCESS - Überschritt 60s nginx default!"
                elif elapsed > 120:
                    status = "✅✅✅ ULTRA SUCCESS - Funktioniert bei 120s+!"
                
                print(f"   {status}")
                print(f"   Time: {elapsed:.1f}s")
                print(f"   Audio: {audio_kb:.1f} KB ({len(audio_data)} bytes)")
                
                results.append({
                    'test': test_name,
                    'success': True,
                    'time': elapsed,
                    'chars': len(content),
                    'audio_kb': audio_kb
                })
                
            else:
                print(f"   ⚠️ Response received in {elapsed:.1f}s - Aber keine Audio-Daten")
                results.append({
                    'test': test_name,
                    'success': False,
                    'time': elapsed,
                    'chars': len(content),
                    'error': 'No audio data'
                })
                    
        except Exception as e:
            elapsed = time.time() - start
            error_msg = str(e)
            
            # Analyze error type
            if elapsed > 58 and elapsed < 62:
                print(f"   ❌❌❌ NGINX TIMEOUT bei {elapsed:.1f}s!")
                print(f"   Problem: nginx proxy_read_timeout noch auf 60s?")
                print(f"   Lösung: sudo nginx -s reload ausführen!")
            elif elapsed > 298:
                print(f"   ❌ Timeout bei {elapsed:.1f}s (erreicht 300s Limit)")
            elif "timeout" in error_msg.lower():
                print(f"   ❌ Timeout nach {elapsed:.1f}s")
                print(f"   Error: {error_msg[:150]}")
            else:
                print(f"   ❌ Fehler nach {elapsed:.1f}s: {error_msg[:150]}")
            
            results.append({
                'test': test_name,
                'success': False,
                'time': elapsed,
                'chars': len(content),
                'error': error_msg[:100]
            })
    
    # 4. Zusammenfassung
    print("\n" + "=" * 70)
    print("4. TEST ZUSAMMENFASSUNG:")
    print("=" * 70)
    
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    
    print(f"\n✅ Erfolgreiche Tests: {success_count}/{total_count}")
    
    # Detailed results
    print("\nDetaillierte Ergebnisse:")
    print(f"{'Test':<40} {'Zeit':<10} {'Chars':<8} {'Status':<10}")
    print("-" * 70)
    
    for r in results:
        status = "✅ OK" if r['success'] else "❌ FAIL"
        time_str = f"{r['time']:.1f}s"
        
        # Highlight critical results
        if r['success'] and r['time'] > 60:
            status = "✅ CRITICAL"
        
        print(f"{r['test']:<40} {time_str:<10} {r['chars']:<8} {status:<10}")
    
    # 5. Diagnose & Empfehlungen
    print("\n" + "=" * 70)
    print("5. DIAGNOSE:")
    print("=" * 70)
    
    # Check if any test crossed the 60s barrier
    crossed_60s = any(r['success'] and r['time'] > 60 for r in results)
    failed_near_60s = any(not r['success'] and 58 < r['time'] < 62 for r in results)
    
    if crossed_60s:
        print("\n✅✅✅ NGINX TIMEOUT FIX ERFOLGREICH!")
        print("   - Tests mit >60s Generierung funktionieren")
        print("   - nginx proxy_read_timeout = 300s ist aktiv")
        print("   - Lange Podcasts sollten jetzt funktionieren")
    elif failed_near_60s:
        print("\n❌❌❌ NGINX TIMEOUT NOCH AKTIV!")
        print("   - Tests failen bei ~60s")
        print("   - nginx config wurde nicht neu geladen?")
        print("\n   FIX:")
        print("   1. sudo nginx -t  # Test config")
        print("   2. sudo nginx -s reload  # Reload config")
        print("   3. Script erneut ausführen")
    else:
        print("\n⚠️ UNKLAR - Keine Tests über 60s")
        print("   - Entweder alles zu schnell oder alles gefailed")
        print("   - Mehr Tests mit längerem Content nötig")
    
    # Performance metrics
    successful_results = [r for r in results if r['success']]
    if successful_results:
        avg_time = sum(r['time'] for r in successful_results) / len(successful_results)
        avg_chars = sum(r['chars'] for r in successful_results) / len(successful_results)
        chars_per_sec = avg_chars / avg_time if avg_time > 0 else 0
        
        print(f"\n📊 Performance Metriken:")
        print(f"   - Durchschnitt: {avg_time:.1f}s für {avg_chars:.0f} Zeichen")
        print(f"   - Rate: ~{chars_per_sec:.0f} Zeichen/Sekunde")
        print(f"   - Für 10.000 Zeichen: ~{10000/chars_per_sec:.0f}s erwartet")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    check_timeout_settings()