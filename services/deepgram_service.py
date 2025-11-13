# services/deepgram_service.py
import logging
from pathlib import Path
import json
from deepgram import DeepgramClient

logger = logging.getLogger(__name__)


class DeepgramService:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY is required")
        self.api_key = api_key
        self.client = DeepgramClient(api_key=api_key)
    
    def load_keyterms(self, language='en'):
        """
        Load domain-specific keyterms for improved transcription accuracy.
        Returns a list of keyterms for the specified language.
        """
        try:
            keyterms_path = Path('/app/keyterms.json')
            if not keyterms_path.exists():
                logger.warning("keyterms.json not found, using empty keyterms list")
                return []
            
            with open(keyterms_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Combine language-specific and universal terms
            terms = data.get(language, []) + data.get('universal', [])
            
            # Limit to 100 terms as per Deepgram recommendation
            terms = terms[:100]
            
            logger.info(f"Loaded {len(terms)} keyterms for language '{language}'")
            return terms
        except Exception as e:
            logger.error(f"Error loading keyterms: {e}")
            return []
    
    def transcribe_file(self, audio_data, language='en'):
        """
        Transcribe audio file using Nova-3 model with keyterm prompting.
        
        Args:
            audio_data: Audio file data (bytes)
            language: Language code (e.g., 'en', 'de')
        
        Returns:
            str: Transcribed text
        """
        try:
            # Load keyterms for the selected language
            keyterms = self.load_keyterms(language)
            
            logger.info(f"Transcribing with Nova-3, language={language}, keyterms={len(keyterms)}")
            
            # SDK 5.1.0: Direct API call with kwargs
            response = self.client.listen.v1.media.transcribe_file(
                request=audio_data,
                model="nova-3",
                smart_format=True,
                utterances=True,
                punctuate=True,
                language=language,
                numerals=True,
                paragraphs=True,
                keyterm=keyterms,
            )
            
            # Extract transcript
            transcript = response.results.channels[0].alternatives[0].transcript
            
            return transcript
        
        except Exception as e:
            logger.error(f"Deepgram transcription failed: {e}", exc_info=True)
            raise
    
    def get_api_key(self):
        """Return API key for client-side WebSocket connection."""
        return self.api_key